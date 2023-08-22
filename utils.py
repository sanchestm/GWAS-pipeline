def dosage_compensation(a, male_indices):
    b = a.copy()
    b[male_indices] *= 2
    return b

def find_overlap(r1, r2):
    ol = [num for num, val in enumerate(r1.i_list) if val >= r2.i_min]
    return {int(k):int(v) for v, k in enumerate(ol)}
    
def generate_umap_chunks(chrom, win: int = int(2e6), overlap: float = .5,
                        nsampled_snps: int = 40000,nsampled_rats: int = 20000,
                        random_sample_snps: bool = False,nautosomes=20,
                        founderfile = '/projects/ps-palmer/gwas/databases/founder_genotypes/founders7.2',
                        latest_round = '/projects/ps-palmer/gwas/databases/rounds/r10.1.1',
                        pickle_output = False,
                        save_path = ''):
    print(f'starting umap chunks for {chrom}, nsnps = {nsampled_snps}, maxrats = {nsampled_rats}')
    if type(founderfile) == str: bimf, famf, genf = pandas_plink.read_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(latest_round) == str: bim, fam, gen = pandas_plink.read_plink(latest_round)
    else: bim, fam, gen = latest_round
    offset = win//2

    bim1 = bim[bim.chrom.isin([chrom, str(chrom)]) & bim.snp.isin(bimf.snp)]#.query('chrom == "12"')
    start, stop = bim1['i'].agg(['min', 'max']).values
    if random_sample_snps: 
        sampled_snps = sorted(np.random.choice(range(start, stop), size = min(nsampled_snps, gen.T.shape[1] ) , replace = False))
    else:
        sampled_snps = bim1[::bim1.shape[0]//nsampled_snps+1].i.to_list()
        
    bim1 = bim1[bim1.i.isin(sampled_snps)]
    bim1i = bim1.set_index('pos')
    bim1isnp = bim1.set_index('snp')

    sampled_snps_names = list(bim1.loc[sampled_snps].snp)
    bimf1 = bimf[bimf.snp.isin(sampled_snps_names)]
    bimf1isnp = bimf1.set_index('snp')
    bimf1i = bimf1.set_index('pos')

    if str(chrom).lower() in [str(nautosomes+2), 'y']: 
        allowed_rats = fam[fam.gender == '1'].i
    else: 
        allowed_rats = range(gen.T.shape[0])
    sampled_rats =  sorted(np.random.choice(allowed_rats, size = min(nsampled_rats, len(allowed_rats) ) , replace = False))
    nsampled_rats = len(sampled_rats)

    aggreg = lambda df: \
             pd.DataFrame([df.index.values.mean(),
                          df.i.values.min(), 
                          df.i.values.max(), 
                          df.i.values,
                          df.snp.values,
                           ]).set_axis(['pos_mean', 'i_min', 'i_max', 'i_list', 'snp_list' ]).T

    out = pd.concat([aggreg(k)\
        for i in range(offset,bim1.pos.max()+ offset, int(win*(1-overlap) )) \
        if (k := bim1i.loc[i - offset:i + offset+1]).shape[0]> 0]).reset_index(drop =True)
    out['nsnps'] = out.i_list.map(len)
    out.loc[[0], 'relationship'] = [{}]
    out.loc[1:, 'relationship'] = [find_overlap(out.iloc[i], out.iloc[i+1]) for i in range(out.shape[0]-1)]
    out['relationship_i'] = out.apply(lambda r: r.i_list[list(r.relationship.values())], axis = 1)
    out['relationship_snp'] = out.apply(lambda r: r.snp_list[list(r.relationship.values())], axis = 1)
    print('getting the genotypes for umap with...')
    #out['genotypes'] = out.i_list.progress_apply(lambda x: np.vstack([gen.T[sampled_rats][:, x ], 
    #                                                                  genf.T[:, x]]).compute())
    out['genotypes'] = out.snp_list.progress_apply(lambda x: np.vstack([gen.T[sampled_rats][:, bim1isnp.loc[x].i ], 
                                                                  genf.T[:, bimf1isnp.loc[x].i]]))#.compute()

    if str(chrom).lower() in [str(nautosomes+1), 'x']:
        male_indices = fam.loc[sampled_rats].reset_index(drop= True).query('gender in [2, "2"]').index.to_list()
        out['genotypes'] = out.genotypes.progress_apply(dosage_compensation,  male_indices = male_indices)

    #out['genotypes'] = out['genotypes'].progress_apply(lambda x: KNNImputer().fit_transform(x))
    out['genotypes'] = out['genotypes'].progress_apply(lambda x: make_pipeline(KNNImputer(),StandardScaler() ).fit_transform(x)) #

    label_dict = defaultdict(lambda: 'AAunk', {row.i: row.iid for name, row in famf.iterrows()})
    i2label = lambda y: label_dict[y]
    out['label'] = out.nsnps.apply(lambda x: np.hstack([-np.ones(len(sampled_rats)),(famf.i).values ]))
    out['label_str'] = out.label.apply(lambda x:list(map(i2label, x)))
    aligned_mapper = umap.AlignedUMAP(metric='euclidean').fit(out.genotypes.to_list(), \
                                                                relations=out.relationship[1:].to_list(), \
                                                                y = out.label.to_list())  
    out['embeddings'] = aligned_mapper.embeddings_
    out['embeddings'] = out['embeddings'].apply(lambda x: np.array(x))
    #v1 = SelfTrainingClassifier(SVC(kernel="rbf", gamma=0.5, probability=True))
    ls = LabelSpreading()

    out['predicted_label'] = out.progress_apply(lambda r: ls\
                                                .fit(r.embeddings, r.label.astype(int)).predict(r.embeddings) , axis = 1)
    out['predicted_label_str'] = out.predicted_label.apply(lambda x:list(map(i2label, x)))
    X = np.stack(out.embeddings)
    X = np.clip(X, -100, 100)
    #out['predicted_label_str']#.apply(Counter)
    
    founder_colors = defaultdict(lambda: 'white', {'BN': '#1f77b4', 'ACI':'#ff7f0e', 'MR': '#2ca02c', 'M520':'#d62728',
                      'F344': '#9467bd', 'BUF': '#8c564b', 'WKY': '#e377c2', 'WN': '#17becf'})
    
    palette = px.colors.diverging.Spectral
    traces = [
        go.Scatter3d(
            x=X[:, i, 0],
            y=X[:, i, 1],
            z=out.pos_mean.values,
            mode="lines+markers",
            line=dict(width = .2 if out.iloc[0].label[i] == -1 else 5,
                      color=out.predicted_label_str.apply(lambda x: founder_colors[x[i]]).to_list() if out.iloc[0].label[i] == -1 \
                            else out.label_str.apply(lambda x: founder_colors[x[i]]).to_list())    ,
            #hovertemplate='<b>%{text}</b>',
            #text = [fam.loc[sampled_rats, 'iid'].iloc[i]]*out.shape[0], 
            marker=dict(size=4 if out.iloc[0].label[i] == -1 else 5,
                        symbol = 'circle' if out.iloc[0].label[i] == -1 else 'diamond',
                        #color=out.predicted_label_str.apply(lambda x: founder_colors[x[i]]).to_list()
                        color=out.predicted_label_str.apply(lambda x: founder_colors[x[i]]).to_list() if out.iloc[0].label[i] == -1 \
                            else out.label_str.apply(lambda x: founder_colors[x[i]]).to_list()  
                        ,opacity=0.3 if out.iloc[0].label[i] == -1 else 1)) 
            for i in list(np.random.choice(nsampled_rats, 300, replace = False)) \
        + list(range(nsampled_rats, nsampled_rats + famf.shape[0] ))
    ]
    fig = go.Figure(data=traces)
    fig.update_layout(  width=1800, height=1000,autosize=False,showlegend=False )
    os.makedirs(f'{save_path}images/genotypes/3dchrom', exists_ok = True)
    fig.write_html(f'{save_path}images/genotypes/3dchrom/chr_{chrom}_compressed.html')
    
    stack = np.hstack([
           np.concatenate(out.embeddings),
           np.concatenate(out.predicted_label_str.apply(np.array)).reshape(-1,1),
           np.concatenate(out.pos_mean.apply(lambda r: np.array([r]*(nsampled_rats + famf.shape[0])))).reshape(-1,1),
           np.concatenate(out.label_str.apply(lambda x: np.array([2 if y == 'AAunk' else 20 for y in x]))).reshape(-1,1),
           np.concatenate(out.label_str.apply(lambda x: np.array([1 if y != 'AAunk' else .4 for y in x]))).reshape(-1,1),
           np.concatenate(out.label_str.apply(np.array)).reshape(-1,1)])
    stack = pd.DataFrame(stack, columns = ['umap1', 'umap2', 'label', 'chr_pos', 'label_size','alpha' , 'founder_label'])
    stack = stack.reset_index(names = 'i')
    colnames = ['umap1','umap2', 'chr_pos','alpha', 'label_size']
    stack[colnames] = stack[colnames].astype(float)
    
    fig2 = px.scatter(stack, x="umap1", y="umap2", animation_frame="chr_pos", 
                      color="label",size = 'label_size') #opacity = stack.alpha,,  symbol = 'founder_label'
    #, hover_data=['rfid'] 
    fig2.update_layout(
        autosize=False, width=1800, height=1000,
        updatemenus=[dict( type="buttons", buttons=[dict(label="Play", method="animate",
                              args=[None, {"frame": {"duration": 5, "redraw": False},}])])])
    fig2.update_traces(marker=dict(line=dict(width=.4, color='black') ))
    os.makedirs(f'{save_path}images/genotypes/animations', exists_ok = True)
    fig2.write_html(f'{save_path}images/genotypes/animations/genomeanimation_{chrom}.html')
    
    founder_count = pd.concat(out['predicted_label_str'].apply(lambda x: pd.DataFrame(Counter(x), index = ['count'])).values)\
          .fillna(0).astype(int).set_index(out.pos_mean.astype(int))

    fig = dash_bio.Clustergram(data=founder_count,
        column_labels=list(founder_count.columns.values),
        row_labels=list(founder_count.index),cluster = 'col', display_ratio=[0.9, 0.1], color_map='Spectral'
    )
    fig.update_layout(  width=800, height=800,autosize=False,showlegend=False )
    os.makedirs(f'{save_path}images/genotypes/founders_cnt', exists_ok = True)
    fig.write_html(f'{save_path}images/genotypes/founders_cnt/founders_cnt_{chrom}.html')
    if pickle_output:
        with open(f'{save_path}results/umap_per_chunk/{chrom}.pkl', 'wb') as fil:
            pickle.dump(out.drop('genotypes', axis = 1), fil )
    #fig2.show()
    return out