import matplotlib.pyplot as plt
import pandas as pd
import pandas_plink
from plink import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading
from umap import UMAP
from wrapbash import bash

@numba.njit()
def nan_sim(x, y):
    o = np.nansum(np.abs(x-y))
    if ~np.isnan(o): return 1/(o+1)
    return 0

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

@numba.njit()
def nan_dist(x, y):
    o = np.nansum(np.abs(x-y))
    if ~np.isnan(o): return o
    return 1e10

def combine_hex_values(d):
    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
    green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
    blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
    zpad = lambda x: x if len(x)==2 else '0' + x
    return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])


def _distance_to_founders(subset_geno,founderfile,fname,c, scaler: str = 'ss', verbose = False, nautosomes = 20):
    if type(founderfile) == str: bimf, famf, genf = pandas_plink.read_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(subset_geno) == str: bim, fam, gen = pandas_plink.read_plink(subset_geno)
    else: bim, fam, gen = subset_geno
    if str(c).lower() in [str(nautosomes+2), 'y']: fam = fam[fam.gender == '1']
    
    snps = bim[bim['chrom'] == c]
    snps = snps[::snps.shape[0]//2000+1]
    ys = pd.DataFrame(gen[snps.i][:, fam.i].compute().T,columns = snps.snp, index = fam.iid )
    
    if str(c).lower() in [str(nautosomes+2), 'x']:
        ys.loc[fam.query('gender == "2"').iid, :] *= 2
    founder_gens = plink2pddf( (bimf, famf, genf),snplist= list(snps.snp))
    if (aa := bimf.merge(snps, on = 'snp', how = "inner").query('a0_x != a0_y')).shape[0]:
        printwithlog('allele order mixed between founders and samples')
        display(aa)
    Scaler = {'tfidf': make_pipeline(KNNImputer(), TfidfTransformer()), 
              'ss': StandardScaler(), 'passthrough': make_pipeline('passthrough')}
    founder_colors = defaultdict(lambda: 'white', {'BN': '#1f77b4', 'ACI':'#ff7f0e', 'MR': '#2ca02c', 'M520':'#d62728',
                      'F344': '#9467bd', 'BUF': '#8c564b', 'WKY': '#e377c2', 'WN': '#17becf'})
    shared_snps = list(set(ys.columns) & set(founder_gens.columns))
    merged1 = pd.concat([ys[shared_snps], founder_gens[shared_snps]])
    merged1.loc[:, :] = Scaler[scaler].fit_transform(merged1) if scaler != 'tfidf' \
                        else Scaler[scaler].fit_transform(merged1).toarray()
    dist2f = pd.DataFrame(cdist(merged1.loc[ys.index], merged1.loc[founder_gens.index] , metric = nan_sim),
                          index = ys.index, columns=founder_gens.index)
    matchdf = dist2f.apply(lambda x: pd.Series(get_topk_values(x,2)), axis = 1)\
                    .set_axis(['TopMatch', "2ndMatch"], axis = 1)\
                    .fillna(method='ffill', axis = 1)
    matchdfcolors = matchdf.applymap(lambda x: '#'+combine_hex_values({founder_colors[k][1:]: 1 for k in x.split('|')}))
    genders = fam.set_index('iid').loc[dist2f.index.to_list(), :].gender.map(lambda x: ['white','steelblue', 'pink'][int(x)]).to_frame()
    rowcols = pd.concat([genders,matchdfcolors], axis = 1)
    #sns.clustermap(dist2f , cmap = 'turbo',  figsize= (8, 8), 
    #               square = True, norm=PowerNorm(4, vmax = 1, vmin = 0), 
    #                row_colors=rowcols) #vmin = dist2f.melt().quantile(.099).value,norm=LogNorm(),
    sns.clustermap(dist2f.div(dist2f.sum(axis = 1), axis = 0).fillna(0) , cmap = 'turbo',  figsize= (15, 15), 
               square = True, norm=PowerNorm(.5, vmax = 1, vmin = 0), 
                row_colors=rowcols)
    printwithlog(f'the following founders are present in the chr {c}:\n')
    display(matchdf.TopMatch.value_counts().to_frame())
    plt.savefig(fname)
    if verbose: plt.show()
    plt.close()
    return

def _make_umap_plot(subset_geno, founderfile, fname, c, verbose = False, nautosomes = 20):
    if type(founderfile) == str: bimf, famf, genf = pandas_plink.read_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(subset_geno) == str: bim, fam, gen = pandas_plink.read_plink(subset_geno)
    else: bim, fam, gen = subset_geno
    
    if str(c).lower() in [str(nautosomes+2), 'y']: fam = fam[fam.gender == '1']
    snps = bim[bim['chrom'] == c]
    snps = snps[::snps.shape[0]//2000+1]
    ys = pd.DataFrame(gen[snps.i][:, fam.i].compute().T,columns = snps.snp, index = fam.iid )
    if str(c).lower() in [str(nautosomes+2), 'x']:  ys.loc[fam.query('gender == "2"').iid, :] *= 2
    
    founder_gens = plink2pddf( (bimf, famf, genf),snplist= list(snps.snp))
    shared_snps = list(set(ys.columns) & set(founder_gens.columns))
    merged = pd.concat([ys[shared_snps], founder_gens[shared_snps]])
    merged.iloc[:, :] = make_pipeline(KNNImputer(), StandardScaler()).fit_transform(merged)
    merged['label'] = merged.index.to_series().apply(lambda x: x if x in founder_gens.index else 'AAunk')
    le = LabelEncoder().fit(merged.label)
    merged['labele'] = le.transform(merged.label) -1
    o = UMAP(metric =nan_dist).fit_transform((merged.loc[:, merged.columns.str.contains(':')]), y=merged.labele)#'manhattan'
    o = pd.DataFrame(o, index = merged.index, columns=['UMAP1', 'UMAP2'])
    o['label'] = merged['label']
    o['size'] = o.label.apply(lambda x: 200 if x!= 'AAunk' else 20)
    o['alpha'] = o.label.apply(lambda x: .95 if x!= 'AAunk' else .3)
    labeler =  LabelSpreading().fit(o[['UMAP1', 'UMAP2']], merged.labele)
    o['predictedLabel'] = le.inverse_transform(labeler.predict(o[['UMAP1', 'UMAP2']])+1)
    f, ax = plt.subplots(1, 1, sharex = True, sharey = True, figsize=(10,10))
    sns.scatterplot(o, x = 'UMAP1', y = 'UMAP2', alpha = o.alpha,s = o['size'] ,hue= 'predictedLabel', edgecolor = 'Black', ax = ax)
    sns.despine()
    plt.savefig(fname)
    if verbose: plt.show()
    plt.close()

def make_LD_plot(ys: pd.DataFrame, fname: str):
    r2mat = ys.corr()**2 
    dist_r2 = r2mat.reset_index(names = ['snp1']).melt(id_vars = 'snp1', value_name='r2')
    dist_r2['distance'] = (dist_r2.snp1.str.extract(':(\d+)').astype(int) - dist_r2.snp.str.extract(':(\d+)').astype(int)).abs()
    fr2, axr2 = plt.subplots(1, 2,  figsize=(20,10))
    sns.regplot(dist_r2.assign(logdils = 1/np.log10(dist_r2.distance+100)).sample(2000, weights='logdils'), 
                x='distance', y = 'r2' ,logistic = True, line_kws= {'color': 'red'},
               scatter_kws={'linewidths':1,'edgecolor':'black', 'alpha': .2},  ax = axr2[0])
    sns.heatmap(r2mat, ax = axr2[1], cmap = 'Greys' , cbar = False, vmin = 0, vmax = 1 )
    sns.despine()
    plt.savefig(fname)
    plt.close()