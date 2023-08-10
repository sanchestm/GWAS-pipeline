#gene2go = download_ncbi_associations()
#geneid2gos_rat= Gene2GoReader(gene2go, taxids=[10116])
#import sleep
from  os.path import dirname, basename
from IPython.display import display
from IPython.utils import io
from collections import Counter, defaultdict
from collections import defaultdict, namedtuple
from datetime import datetime
from glob import glob
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_ncbi_associations
from io import StringIO
from pathlib import Path
from pdf2image import convert_from_path
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.utilities import regressor_coefficients 
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list, linkage
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression#, RobustRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading
from statsReport import quantileTrasformEdited as quantiletrasform
from time import sleep
from tqdm import tqdm
from umap import UMAP
import dash_bio as dashbio
import dask.array as da
import dask.dataframe as dd
import gc
import goatools
import gzip
import inspect
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import  PowerNorm
import mygene
import numpy as np
import numba
import os
import pandas as pd
import pandas_plink
import plotly.express as px
import plotly.graph_objects as go
import prophet
import psycopg2
import re
import requests
from scipy.spatial.distance import cdist
import seaborn as sns
import statsReport
import subprocess
import sys
from umap import UMAP
import warnings
mg = mygene.MyGeneInfo()
tqdm.pandas()
sys.setrecursionlimit(10000)
warnings.filterwarnings('ignore')
#conda create --name gpipe -c conda-forge openjdk=17 ipykernel pandas seaborn scikit-learn umap-learn psycopg2 dask
#conda activate gpipe
#conda install -c bioconda gcta plink snpeff mygene
#pip install goatools
#wget https://snpeff.blob.core.windows.net/versions/snpEff_latest_core.zip

na_values_4_pandas = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null', 'UNK']

def decompose_grm(grm_path, n_comp = 50, verbose = True):
    (grmxr, ar) =  pandas_plink.read_grm(grm_path)
    eigval, eigvec = np.linalg.eig(grmxr)
    eig_pairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(len(eigval))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    explained_var = np.array([eig_pairs[i][0] for i in range(n_comp)])/sum(list(zip(*eig_pairs))[0])
    if verbose: print(f'explained_variances:{explained_var}\
          \n total explained var:{sum(explained_var)}' )
    return pd.DataFrame(np.vstack((eig_pairs[i][1] for i in range(n_comp))).T,
             columns = [f'GRM_PC{x}' for x in range(1, n_comp+1)],
             index= grmxr.sample_0.astype(str))

def decompose_grm_pca(grm_path, n_comp = 5, verbose = True):
    (grmxr, ar) =  pandas_plink.read_grm(grm_path)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp)
    decomp = pca.fit_transform(grmxr)
    if verbose: print(f'explained_variances:{pca.explained_variance_ratio_}\
          \ntotal explained var:{sum(pca.explained_variance_ratio_)}' )
    return pd.DataFrame(decomp, columns = [f'GRM_PC{x}' for x in range(1, decomp.shape[1]+1)],
             index= grmxr.sample_0.astype(str))

def plink2pddf(plinkpath,rfids = 0, c = 0, pos_start = 0, pos_end = 0, snplist = 0):
    if type(plinkpath) == str: 
        snps, iid, gen = pandas_plink.read_plink(plinkpath)
    else: snps, iid, gen = plinkpath
    snps.chrom = snps.chrom.astype(int)
    snps.pos = snps.pos.astype(int)
    isnps = snps.set_index(['chrom', 'pos'])
    iiid = iid.set_index('iid')
    if not snplist:
        if (pos_start == pos_start == 0 ):
            if not c: index = isnps
            else: index = isnps.loc[(slice(c, c)), :]
        index = isnps.loc[(slice(c, c),slice(pos_start, pos_end) ), :]
    else:
        snplist = list(set(snplist) & set(isnps.snp))
        index = isnps.set_index('snp').loc[snplist].reset_index()
    col = iiid  if not rfids else iiid.loc[rfids]
    return pd.DataFrame(gen.astype(np.float16)[index.i.values ][:, col.i].T, index = col.index.values.astype(str), columns = index.snp.values )

def plink(print_call = False, **kwargs):
    call = 'plink ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                for k,v in kwargs.items() if k not in ['print_call'] ])
    call = re.sub(r' +', ' ', call).strip(' ')
    bash(call, print_call=print_call)
    return

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

@numba.njit()
def nan_sim(x, y):
    o = np.nansum(np.abs(x-y))
    if ~np.isnan(o): return 1/(o+1)
    return 0

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

def get_topk_values(s, k): 
    return s.groupby(s.values).agg(lambda x: '|'.join(x.index) ).sort_index()[::-1].values[:k]

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
        print('allele order mixed between founders and samples')
        display(aa)
    Scaler = {'tfidf': make_pipeline(KNNImputer(), TfidfTransformer()), 
              'ss': StandardScaler(), 'passthrough': make_pipeline('passthrough')}
    founder_colors = defaultdict(lambda: 'white', {'BN': '#1f77b4', 'ACI':'#ff7f0e', 'MR': '#2ca02c', 'M520':'#d62728',
                      'F344': '#9467bd', 'BUF': '#8c564b', 'WKY': '#e377c2', 'WN': '#17becf'})
    shared_snps = list(set(ys.columns) & set(founder_gens.columns))
    merged1 = pd.concat([ys[shared_snps], founder_gens[shared_snps]])
    merged1.loc[:, :] = Scaler[scaler].fit_transform(merged1) if scaler != 'tfidf' \
                        else Scaler[scaler].fit_transform(merged1).toarray()
    dist2f = pd.DataFrame( cdist(merged1.loc[ys.index], merged1.loc[founder_gens.index] , metric = nan_sim),
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
    print(f'the following founders are present in the chr {c}:\n')
    display(matchdf.TopMatch.value_counts().to_frame())
    plt.savefig(fname)
    if verbose: plt.show()
    plt.close()
    return

def _make_umap_plot(subset_geno, founderfile, fname, c, verbose = False, nautosomes = 20):
    if type(founderfile) == str: bimf, famf, genf = pandas_plink.read_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(subset_geno) == str: bim, fam, gen = pandas_plink.read_plink(founderfile)
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

class vcf_manipulation:
    def corrfunc(x, y, ax=None, **kws):
        r, _ = pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

    def get_vcf_header(vcf_path):
        with gzip.open(vcf_path, "rt") as ifile:
            for num, line in enumerate(ifile):
                if line.startswith("#CHROM"): return line.strip().split('\t')
                if num > 10000: return '-1'
        return '-1'

    def read_vcf(filename, method = 'pandas'):
        if method == 'dask':
            return dd.read_csv(filename,  compression='gzip', comment='#',  delim_whitespace=True, header=None, 
                               names = vcf_manipulation.get_vcf_header(filename),blocksize=None,  dtype=str, ).repartition(npartitions = 100000)
        # usecols=['#CHROM', 'POS']
        return pd.read_csv(filename,  compression='gzip', comment='#',  delim_whitespace=True,
                           header=None, names = vcf_manipulation.get_vcf_header(filename),  dtype=str )

    def name_gen2(filename):
        return filename.split('/')[-1].split('.')[0]
    
    def get_vcf_metadata(vcf_path):
        out = ''
        with gzip.open(vcf_path, "rt") as ifile:
            for num, line in enumerate(ifile):
                if line.startswith("#CHROM"): return out
                out += line 
        return '-1'

    def pandas2vcf(df, filename, metadata = ''):
        if  metadata == '':
            header = '\n'.join(["##fileformat=VCFv4.1",
            '##fileDate=20090805',
            '##source=myImputationProgramV3.1',
            '##reference=000GenomesPilot-NCBI36',
            '##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">',
            '##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype Quality">']) + '\n'
        elif metadata[-4:] == '.vcf': header = get_vcf_metadata(metadata)

        with open(filename, 'w') as vcf: 
            vcf.write(header)
        df.to_csv(filename, sep="\t", mode='a', index=False)

def bash(call, verbose = 0, return_stdout = True, print_call = True):
    if print_call: print(call)
    out = subprocess.run(call.strip(' ').split(' '), capture_output = True) 
    if verbose and not return_stdout: print(out.stdout)
    if out.stderr: 
        try:print(out.stderr.decode('ascii'))
        except: print(out.stderr.decode('utf-8'))
    if return_stdout: 
        try: oo =  out.stdout.decode('ascii').strip().split('\n')
        except: oo =  out.stdout.decode('utf-8').strip().split('\n')
        return oo
    return out

def qsub(call: str, queue = 'condo', walltime = 8, ppn = 12, out = 'log/', err = 'logerr/', project_dir = ''):
    err_path = f'{project_dir}{err}$PBS_JOBNAME.err'
    out_path = f'{project_dir}{out}$PBS_JOBNAME.out'
    call_path = f'{project_dir}{call}'
    return bash(f'qsub -q {queue} -l nodes=1:ppn={ppn} -j oe -o {out_path} -e {err_path} -l walltime={walltime}:00:00 {call_path}')

def vcf2plink(vcf = 'round9_1.vcf.gz', n_autosome = 20, out_path = 'zzplink_genotypes/allgenotypes_r9.1'):
    bash(f'plink --thread-num 16 --vcf {vcf} --chr-set {n_autosome} no-xy --set-missing-var-ids @:# --make-bed --out {out_path}')
    
def _prophet_reg(dforiginal = "",y_column = 'y', 
                 categorical_regressors=[], regressors = [], 
                 ds_column= 'age', rolling_avg_days: float = 0, seasonality: list = [],
                 growth = 'logistic', removed_months = [], removed_weekday =[], removed_hours =[], index_col = 'rfid',
                 return_full_df = False, save_explained_vars = False, threshold = 0.00, path = '', extra_label = ''):
    
    df = dforiginal.copy().set_index(index_col)
    
    if df[ds_column].dtype in [int, float]:
        df[ds_column] =(df[ds_column]*1e9*24*3600).apply(pd.to_datetime)
    elif df[ds_column].dtype in [int, float]: pass
    else: raise TypeError('datime_column not int, float or pandas datetime')
    df = df.sort_values(ds_column)
    
    #### rolling average if requested
    if rolling_avg_days > 0.1:
        df = df.set_index(ds_column)
        df = df.rolling(window= str(int(rolling_avg_days *24))+'H').mean().reset_index()
    df = df[[ds_column, y_column]+ regressors + categorical_regressors].dropna() ### added dropna()
    df.columns = ['ds', 'y']+ regressors + categorical_regressors
    
    ### onehot_encoding
    ohe = OneHotEncoder()
    oheencoded = ohe.fit_transform(df[categorical_regressors].astype(str)).todense()
    df[[f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_regressors)]] = oheencoded
    
    ### prophet seasonality
    season_true_kwards = {x: 25 for x in seasonality} 
    season_false_kwards = {x: False for x in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality'] if x not in seasonality}
    
    ###setting prophet growth strategy
    cap = df.y.max() + df.y.std()*.1
    floor = 0
    if growth == 'logistic':
        df['cap'] = cap
        df['floor'] = floor
     
    #remove months, weekdays, hours
    df = df[~(df['ds'].dt.month.isin(removed_months))]
    df = df[~(df['ds'].dt.weekday.isin(removed_weekday))]
    df = df.query('ds.dt.hour not in @removed_hours')  
    
    ### explained_var_threshold
    covs = df.columns[df.columns.str.contains('OHE')].to_list() + [x for x in regressors if x!= ds_column]
    temp = (df[['y']+covs].corr()**2).loc[['y'], covs].rename({'y': y_column})
    approved_cols = temp[temp > threshold].dropna(axis = 1).columns.to_list()
    if save_explained_vars:
        try: prev = pd.read_csv(f'{path}melted_explained_variances.csv')
        except: prev = pd.DataFrame()
        temp2 = temp.melt().assign(group = temp.index[0])[['group','variable','value']].set_axis(['variable', 'group', 'value'], axis = 1)
        if extra_label: temp2.variable = temp2.variable +'_'+extra_label 
        pd.concat([prev,temp2]).to_csv(f'{path}melted_explained_variances.csv', index = False)
    
    #### fit model 
    fbmodel = prophet.Prophet(growth = growth,seasonality_mode ='additive', seasonality_prior_scale =10.,
              n_changepoints= 25, changepoint_prior_scale=10 , changepoint_range = .8,
              **season_true_kwards, **season_false_kwards) #mcmc_samples=100
    for i in approved_cols: fbmodel.add_regressor(i)
    fbmodel.fit(df)
    future = pd.DataFrame()
    forecast = fbmodel.predict(pd.concat([df, future], axis = 0).reset_index()).set_index(df.index)
    
    #### make preprocessing plots
    graphlist = []
    graphlist += [plot_plotly(fbmodel,forecast, figsize = (1240, 960),  xlabel=ds_column, ylabel=y_column)]
    graphlist += [plot_components_plotly(fbmodel,forecast, figsize = (1240, 340))]
    if len(approved_cols):
        regressor_coefs = regressor_coefficients(fbmodel)
        regressor_coefs['coef_abs'] = regressor_coefs.coef.apply(abs)
        regressor_coefs = regressor_coefs.sort_values('coef_abs', ascending=False)
        graphlist += [px.bar(regressor_coefs, x="regressor", y="coef", hover_data=regressor_coefs.columns, template='plotly',height=800, width=1240)]
        
    def stack_graphs(g):
        a = '<html>\n<head><meta charset="utf-8" /></head>\n<body>'
        b = '\n'.join([y[y.find('<body>')+len('<body>'): y.find('</body>')] for x in g if (y:= x.to_html())])
        c = '\n</body>\n</html>'
        return a+b+c
    
    os.makedirs(f'{path}results/preprocessing/', exist_ok=True)
    with open(f'{path}results/preprocessing/test_{y_column}{extra_label}.html', 'w') as f: 
        f.write(stack_graphs(graphlist))
    
    outdf = pd.concat([dforiginal.set_index('rfid'), forecast[['yhat']]], axis =1)    
    outdf[f'regressedlr_{y_column}'] = outdf[y_column] - outdf['yhat']
    if return_full_df: return outdf
    return outdf[[f'regressedlr_{y_column}']]

def regressout_timeseries(dataframe = '', data_dictionary = '', covariates_threshold = 0.02, groupby_columns = ['sex'],  
                          ds_column= 'age', save_explained_vars = True, path = ''):
    dd = data_dictionary.copy()
    ddtraits = dd[dd.trait_covariate == 'trait']
    if ds_column not in dataframe.columns:
        dataframe[ds_column] = 0
    if save_explained_vars: pd.DataFrame().to_csv(f'{path}melted_explained_variances.csv', index = False)
    ddtraits.covariates = ddtraits.covariates.str.replace('|'.join(groupby_columns), '').str.replace(',+', ',').str.strip(',')
    get_covs = lambda tipe, cov:  dd[(dd.trait_covariate == tipe) 
                                & (dd.measure.isin(cov.split(',')))
                                & (dd.measure.isin(dataframe.columns))
                                & (~dd.measure.isin(groupby_columns))].measure.to_list()
    
    datadic_regress_df = lambda X, label: pd.concat(ddtraits.apply(lambda row: _prophet_reg(X, row.measure,
                                        categorical_regressors = get_covs('covariate_categorical', row.covariates),
                                        regressors = get_covs('covariate_continuous', row.covariates),
                                        threshold=covariates_threshold, ds_column = ds_column,
                                        save_explained_vars = save_explained_vars, path = path,
                                        extra_label = label), axis = 1,
                                        ).to_list(), axis = 1)
    
    if not len(groupby_columns):
        outdf =  pd.concat([dataframe.loc[:, ~dataframe.columns.str.contains('regressedlr_')].set_index('rfid'),
                          datadic_regress_df(dataframe, '')], axis = 1)
    
    else: outdf = pd.concat([dataframe.loc[:, ~dataframe.columns.str.contains('regressedlr_')].set_index('rfid').drop(groupby_columns, axis = 1),
                      dataframe.groupby(groupby_columns).apply(lambda df: datadic_regress_df(df, df.name)).reset_index().set_index('rfid')], axis = 1)#drop = True
    
    if save_explained_vars:
        piv = pd.DataFrame(pd.read_csv(f'{path}melted_explained_variances.csv').groupby('variable')['group'].apply(list)).reset_index()
        piv.to_csv(f'{path}pivot_explained_variances.csv', index = False)
    
    outdf['regressedlr_'+ddtraits.measure] = quantiletrasform(outdf,'regressedlr_'+ ddtraits.measure)
    if save_explained_vars: outdf.to_csv(f'{path}processed_data_ready.csv', index = False)
    return outdf
    
    
class gwas_pipe:
    '''
    to run the gwas pipeline we need to inzzplink_genotypes/lize the project as a class
    the essential information to run the project is the path the folder that we will use
    because we create multiple subfolders for the results.
    We opted to input the data separately, but in general we will use the processed_data_ready.csv of each project
    We did not include the initial preprocessing and normalization because each project has it's own nuances
    Adding a project name is very important of logging the phewas results in the database 
    most things work out of the box but snpeff still can be very finicky, we advide testing it before running 
    (the conda install losses the path to snpeff, which makes it a very frustating experience)

    Parameters
    ----------
    path: str = f'{Path().absolute()}/'
        path to the project we will use as a base directory
    
    project_name: str = 'test'
        name of the project being ran, 
        do not include date, run identifiers 
        this will be used for the phewas db, so remember to follow the naming conventions in
        https://docs.google.com/spreadsheets/d/1S7_ZIGpMkNmIjhAKUjHBAmcieDihFS-47-_XsgGTls8/edit#gid=1440209927
    
    raw: pd.DataFrame() = pd.DataFrame()
        pandas dataframe that contains all the phenotypic data before regression, 
        genomic identity of a sample has to be provided in the 'rfid' column
    
    data: pd.DataFrame() = pd.DataFrame()
        pandas dataframe that contains all the phenotypic data necessary to run gcta, 
        genomic identity of a sample has to be provided in the 'rfid' column
    
    traits: list = []
        list of strings of the phenotype columns that will be used for the GWAS analysis
        this could be integrated with the data dictionaries of the project in the future
    
    trait_descriptions: list = []
        list of strings that describe the phenotype columns
        
    all_genotypes: str = '/projects/ps-palmer/apurva/riptide/genotypes/round9_1'
        path to file with all genotypes, we are currently at the round 9.1 
    
    chrList: list = []
        list of chromosomes to be included in genetic analysis in not provided will use all autosomes and the x,
        x should be lowercase and values should be strings
    
    snpeff_path: str = ''
        path to snpeff program for GWAS
    
    threads: int = os.cpu_count()
        number of threads when running multithread code
        default is the number of cpus in the machine
        
    use_tscc_modules: list = []
        list of strings to load modules from a HPC using 'module load'

    gtca_path: str = ''
        path to gcta64 program for GWAS, if not provided will use the gcta64 dowloaded with conda

    phewas_db: str = 'phewasdb.parquet.gz'
        path to the phewas database file 
        The phewas db will be maintained in the parquet format but we are considering using apache feather for saving memory
        this could be integrated with tscc by using scp of the dataframe
        currently we are using just one large file, but we should consider breaking it up in the future

    Attributes
    ---------
    path
    project_name
    raw
    df
    traits
    get_trait_descriptions
    chrList
    all_genotypes
    failed_full_grm
    founder_genotypes
    heritability_path
    phewas_db
    sample_path
    genotypes_subset
    genotypes_subset_vcf
    autoGRM
    xGRM
    log
    threadnum
    thrflag
    sample_sex_path
    sample_sex_path_gcta
    gcta
    snpeff_path
    print_call

    Examples
    --------
    df = pd.read_csv('~/Documents/GitHub/sanchest/hsrats_round9_1/Normalized_filtered_tom_jhou_U01_lowercase.csv')
    gwas = gwas_pipe(path = 'test/',
             all_genotypes = 'round9_1.vcf.gz',
             data = df,
             project_name = 'tj',
             traits =  df.loc[:, 'locomotor1':].columns.tolist(),
             threads=12)
    gwas.make_dir_structure()
    gwas.SubsampleMissMafHweFilter()
    gwas.generateGRM()
    gwas.snpHeritability()
    gwas.gwasPerChr()
    gwas.GWAS()
    gwas.addGWASresultsToDb(researcher='tsanches', project='tj', round_version='9.1', gwas_version='0.0.1-comitversion')
    qtls = gwas.callQTLs(NonStrictSearchDir = 'test/results/gwas/')
    gwas.annotate(qtls)
    gwas.eQTL(qtls, annotate= True)
    gwas.phewas(qlts, annotate=True)
    '''
    
    def __init__(self, 
                 path: str = f'{Path().absolute()}/', 
                 project_name: str = 'test',
                 data: pd.DataFrame() = pd.DataFrame(),
                 #regressed: pd.DataFrame() = pd.DataFrame(), 
                 traits: list = [],
                 trait_descriptions: list = [],
                 chrList: list = [], 
                 n_autosome: int = 20,
                 all_genotypes: str = '/projects/ps-palmer/gwas/databases/rounds/round10_1',
                 founder_genotypes: str = '/projects/ps-palfounder_genotypes/Ref_panel_mRatBN7_2_chr_GT', 
                 founderfile: str = '/projects/ps-palmer/gwas/databases/founder_genotypes/founders7.2',
                 gtca_path: str = '',
                 snpeff_path: str =  'snpEff/',
                 phewas_db: str = 'phewasdb.parquet.gz',
                 use_tscc_modules: list = [],
                 threads: int = os.cpu_count()): 

        if use_tscc_modules: bash(f'module load {" ".join(use_tscc_modules)}')
        self.gcta = 'gcta64' if not gtca_path else gtca_path
        self.path = path
        self.all_genotypes = all_genotypes
        self.founder_genotypes = founder_genotypes
        self.snpeff_path = snpeff_path
        self.n_autosome = n_autosome
        
        if os.path.exists(f'{self.path}temp'): bash(f'rm -r {self.path}temp')
        
        if type(data) == str: 
            df = pd.read_csv(data, dtype={'rfid': str}).replace([np.inf, -np.inf], np.nan)
        else: df = data.replace([np.inf, -np.inf], np.nan)
        df.columns = df.columns.str.lower()
        if 'vcf' in self.all_genotypes:
            sample_list_inside_genotypes = vcf_manipulation.get_vcf_header(self.all_genotypes)
        else:
            sample_list_inside_genotypes = pd.read_csv(self.all_genotypes+'.fam', header = None, sep='\s+', dtype = str)[1].to_list()
        df = df.sort_values('rfid').reset_index(drop = True).dropna(subset = 'rfid').drop_duplicates(subset = ['rfid'])
        self.df = df[df.astype(str).rfid.isin(sample_list_inside_genotypes)].copy()
        
        if self.df.shape[0] != df.shape[0]:
            missing = set(df.rfid.astype(str).unique()) - set(self.df.rfid.astype(str).unique())
            print(f"missing {len(missing)} rfids for project {project_name}")
            
        
        self.traits = [x.lower() for x in traits]
        print(self.df[self.traits].count())
        self.make_dir_structure()
        for trait in self.traits:
            trait_file = f'{self.path}data/pheno/{trait}.txt'            
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file, index = False, sep = ' ', header = None)
            
        self.get_trait_descriptions = defaultdict(lambda: 'UNK', {k:v for k,v in zip(traits, trait_descriptions)})
        
        self.phewas_db = phewas_db
        self.project_name = project_name
        
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.genotypes_subset = f'{self.path}genotypes/genotypes'
        self.genotypes_subset_vcf = f'{self.path}genotypes/genotypes_subset_vcf.vcf.gz'
        if len(founderfile): self.foundersbimfambed = pandas_plink.read_plink(founderfile)
        
        self.autoGRM = f'{self.path}grm/AllchrGRM'
        self.xGRM = f'{path}grm/xchrGRM'
        self.yGRM = f'{path}grm/ychrGRM'
        self.mtGRM = f'{path}grm/mtchrGRM'
        self.log = pd.DataFrame( columns = ['function', 'call', 'out'])
        self.thrflag = f'--thread-num {threads}'
        self.threadnum = threads
        self.print_call = True
        
        self.replacenumstoXYMT = lambda x: str(x).replace(str(self.n_autosome+1), 'x')\
                                                 .replace(str(self.n_autosome+2), 'y')\
                                                 .replace(str(self.n_autosome+4), 'mt')\
        
        if not chrList:
            self.chrList = [self.replacenumstoXYMT(i) for i in range(1,self.n_autosome+5) if i != self.n_autosome+3]
        self.failed_full_grm = False
        
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.sample_sex_path = f'{self.path}genotypes/sample_rfids_sex_info.txt'
        self.sample_sex_path_gcta = f'{self.path}genotypes/sample_rfids_sex_info_gcta.txt'
        self.heritability_path = f'{self.path}results/heritability/heritability.tsv'
        
    def regressout(self, data_dictionary: pd.DataFrame(), covariates_threshold: float = 0.02, verbose = False):
        if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
        df, datadic = self.df.copy(), data_dictionary
        datadic = datadic[datadic.measure.isin(df.columns)].drop_duplicates(subset = ['measure'])
        def getcols(df, string): return df.columns[df.columns.str.contains(string)].to_list()
        dfohe = df.copy()
        ohe = OneHotEncoder()
        categorical_all = list(datadic.query('trait_covariate == "covariate_categorical"').measure)
        dfohe = df.copy()
        ohe = OneHotEncoder()
        oheencoded = ohe.fit_transform(dfohe[categorical_all].astype(str)).todense()
        dfohe[[f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_all)]] = oheencoded
        alltraits = list(datadic.query('trait_covariate == "trait"').measure.unique())
        dfohe.loc[:, alltraits] = QuantileTransformer(n_quantiles = 100).fit_transform(dfohe.loc[:, alltraits])
        continuousall = list(datadic.query('trait_covariate == "covariate_continuous"').measure)
        #print(f'all continuous variables {continuousall}')
        if continuousall:
            dfohe.loc[:, continuousall] = QuantileTransformer(n_quantiles = 100).fit_transform(dfohe.loc[:, continuousall])
        variablesall = []
        all_explained_vars = []
        stR = statsReport.stat_check(df)
        for name, tempdf in datadic.query('trait_covariate =="trait"').groupby('covariates'):
            variables = list(tempdf.measure.unique())
            stR = statsReport.stat_check(dfohe)
            variablesall += variables
            categorical = set(name.split(',')) & set(datadic.query('trait_covariate == "covariate_categorical"').measure)
            continuous = set(name.split(',')) & set(datadic.query('trait_covariate == "covariate_continuous"').measure)
            all_covariates = list(set(itertools.chain.from_iterable([ getcols(dfohe, F'OHE_{x}') for x in categorical]))) + list(continuous)
            if verbose:
                print(f'variables:{variables}-categorical:{categorical}-continuous:{continuous}')
                display(stR.plot_var_distribution(targets=variables, covariates = list(categorical)))
            partial_explained_vars = statsReport.stat_check(dfohe).explained_variance(variables,all_covariates)
            melted_variances = partial_explained_vars.reset_index().melt(id_vars = ['index'], 
                                                                  value_vars=partial_explained_vars.columns[:])\
                                                                  .rename({'index':'group'}, axis =1 ).query(f'value > {covariates_threshold}')
            all_explained_vars += [melted_variances]
        all_explained_vars = pd.concat(all_explained_vars).drop_duplicates(subset = ['group', 'variable'])
        if verbose: display(all_explained_vars)
        all_explained_vars.to_csv(f'{self.path}melted_explained_variances.csv', index = False)
        melt_list = pd.DataFrame(all_explained_vars.groupby('variable')['group'].apply(list)).reset_index()
        melt_list.to_csv(f'{self.path}pivot_explained_variances.csv',index = False)
        tempdf = dfohe.loc[:, dfohe.columns.isin(melted_variances.group.unique())].copy()
        dfohe.loc[:, dfohe.columns.isin(melted_variances.group.unique())] = tempdf.fillna(tempdf.mean())
        aaaa = melt_list.apply(lambda x: statsReport.regress_out(dfohe,[x.variable],   x.group), axis =1)
        resid_dataset = pd.concat(list(aaaa), axis = 1)
        non_regressed_cols = [x for x in alltraits if x not in resid_dataset.columns.str.replace('regressedLR_', '')]
        non_regressed_df = df[non_regressed_cols].rename(lambda x: 'regressedLR_' + x, axis = 1)
        resid_dataset = pd.concat([resid_dataset, non_regressed_df], axis = 1)
        cols2norm = resid_dataset.columns[resid_dataset.columns.str.contains('regressedLR_')]
        resid_dataset[cols2norm] = statsReport.quantileTrasformEdited(resid_dataset, cols2norm)
        dfcomplete = pd.concat([df,resid_dataset],axis = 1)
        dfcomplete.columns = dfcomplete.columns.str.lower()
        dfcomplete = dfcomplete.loc[:,~dfcomplete.columns.duplicated()]
        strcomplete = statsReport.stat_check(dfcomplete.reset_index(drop = True))
        dfcomplete.to_csv(f'{self.path}processed_data_ready.csv')
        self.df = dfcomplete
        strcomplete.make_report(f'{self.path}data_distributions.html')
        self.traits = [x.lower() for x in cols2norm]
        simplified_traits = [x.replace('regressedlr_', '') for x in self.traits]
        display(self.df[self.traits].count())
        for trait in self.traits:
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(f'{self.path}data/pheno/{trait}.txt' ,  index = False, sep = ' ', header = None)
            
        trait_descriptions = [datadic.set_index('measure').loc[x, 'description'] if (x in datadic.measure.values) else 'UNK' for x in simplified_traits]
        self.get_trait_descriptions = defaultdict(lambda: 'UNK', {k:v for k,v in zip(self.traits, trait_descriptions)})
        
        return dfcomplete
    
    def regressout_timeseries(self, data_dictionary: pd.DataFrame(), covariates_threshold: float = 0.02,
                              verbose = False, groupby_columns = ['sex'], ds_column = 'age', save = True):
        if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
        df, datadic = self.df.copy(), data_dictionary
        display(df.count())
        #display(datadic.count())
        datadic = datadic[datadic.measure.isin(df.columns)].drop_duplicates(subset = ['measure'])
        self.df = regressout_timeseries(df, datadic, covariates_threshold = covariates_threshold,
                                     groupby_columns = groupby_columns, ds_column = ds_column,
                                    save_explained_vars = save, path = self.path)
        
        self.df.columns = self.df.columns.str.lower()
        self.df = self.df.loc[:,~self.df.columns.duplicated()]
        strcomplete = statsReport.stat_check(self.df.reset_index(drop = True))
        self.df.to_csv(f'{self.path}processed_data_ready.csv')
        strcomplete.make_report(f'{self.path}data_distributions.html')
        self.traits = [x.lower() for x in self.df.columns[self.df.columns.str.contains('regressedlr_')]]
        simplified_traits = [x.replace('regressedlr_', '') for x in self.traits]
        display(self.df[self.traits].count())
        self.df = self.df.reset_index()
        for trait in self.traits:
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(f'{self.path}data/pheno/{trait}.txt' ,  index = False, sep = ' ', header = None)
            
        trait_descriptions = [datadic.set_index('measure').loc[x, 'description'] if (x in datadic.measure.values) else 'UNK' for x in simplified_traits]
        self.get_trait_descriptions = defaultdict(lambda: 'UNK', {k:v for k,v in zip(self.traits, trait_descriptions)})
        
        return self.df.copy()
        
    def plink2Df(self, call, temp_out_filename = 'temp/temp', dtype = 'ld'):
        '''
        this function receives a plink call as a string, 
        runs the call, 
        then reads the output file with the ending dtype 
        and returns it as a pandas table
        '''
        
        full_call = re.sub(r' +', ' ', call + f'--out {self.path}{temp_out_filename}')
        
        ### add line to delete temp_out_filename before doing the 
        bash(full_call, print_call = False)

        try: out = pd.read_csv(f'{self.path}{temp_out_filename}.{dtype}', sep = '\s+')
        except:
            print(f"file not found")
            out = pd.DataFrame()
        return out 
    
    def plink(self, return_file = 'ld', outfile = '',  **kwargs):
        '''
        this function is a wrapper to run plink as a python function
        instead of having to have it as a string and bash call it
        if writing a flag that doesn't require a variable e.g.
        --make-grm use make_grm = ''
        if outfile is empty and return file is not
            we will return a pandas dataframe from the plink outputfile with ending return_file
        otherwise we will save the files under the filename outfile
        '''
        
        call = 'plink ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                    for k,v in kwargs.items()])
        if not outfile and return_file:
            return self.plink2Df(call ,dtype = f'{return_file}' ) 
        else:
            bash(call + f' --out {outfile}', print_call=False)
        return
    
    def gcta(self,  **kwargs):
        '''
        this function is a wrapper to run gcta as a python function
        instead of having to have it as a string and bash call it
        if writing a flag that doesn't require a variable e.g.
        --make-grm use make_grm = ''
        '''
        outfile = 'out.out'
        call = f'{self.gcta} ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                    for k,v in kwargs.items()])
        bash(call + f' --out {outfile}', print_call=False)
        return
    
    def plink_sex_encoding(self, s, male_code = 'M', female_code = 'F'):
        'function to encode the sex column in the plink format'
        if s == male_code: return 1
        if s == female_code: return 2
        return 0
        
    def bashLog(self, call, func, print_call = True):
        '''
        bash log is an easy function to run a bash call and then save the std.out to the log folder
        '''
        self.append2log(func, call , bash(re.sub(r' +', ' ', call), print_call = print_call))
        
        
    def append2log(self, func, call, out):
        self.log.loc[len(self.log)] = [func, call, out]
        logval = '\n'.join(out).lower()
        loc = 'err' if (('error' in logval) ) else ''
        with open(f'{self.path}log{loc}/{func}.log', 'w') as f:
                f.write('\n'.join(out))
        if loc == 'err':
            print(f'found possible error in log, check the file {self.path}log{loc}/{func}.log')
            #raise ValueError(f'found possible error in log, check the file {self.path}/log{loc}/{func}.log')
            
    def make_dir_structure(self, folders: list = ['data', 'genotypes', 'grm', 'log', 'logerr', 'images/genotypes',
                                            'results', 'temp', 'data/pheno', 'results/heritability', 'results/preprocessing',
                                             'results/gwas',  'results/loco', 'results/qtls','results/eqtl','results/sqtl',
                                                  'results/phewas', 'temp/r2', 'results/lz/', 'images/', 'images/scattermatrix/', 
                                                  'images/manhattan/', 'images/genotypes/heatmaps', 'images/genotypes/lds',
                                                 'images/genotypes/dist2founders', 'images/genotypes/umap']):
        
        '''
        creates the directory structure for the project
        '''
        for folder in folders:
            os.makedirs(f'{self.path}{folder}', exist_ok = True)

    def SubsetAndFilter(self, rfids=[] ,thresh_m: float = 0.1, thresh_hwe: float = 1e-10, thresh_maf: float = 0.005, verbose: bool = True,
                       filter_based_on_subset: bool = True, makefigures = True):

        sex_encoding = defaultdict(lambda : 0, {'M': '1', 'F':'2'})
        sex_encoder = lambda x: sex_encoding[x]    
        self.sample_path = f'{self.path}genotypes/keep_rfids.txt'
        self.sample_path_males = f'{self.path}genotypes/keep_rfids_males.txt'
        self.sample_path_females = f'{self.path}genotypes/keep_rfids_females.txt'
        accepted_snps_path = f'{self.path}genotypes/accepted_snps.txt'
        os.makedirs(f'{self.path}genotypes', exist_ok = True)
        self.genotypes_subset = f'{self.path}genotypes/genotypes'

        if not rfids:
            famf = pd.read_csv(self.all_genotypes+'.fam', header = None, sep = '\s+', dtype = str)[[1, 4]].set_axis(['iid', 'gender'], axis = 1)
            gen_iids = famf['iid'].to_list()
            if not filter_based_on_subset: 
                famf[['iid', 'iid']].to_csv(self.sample_path, index = False, header = None, sep = ' ')
                famf.query('gender in [1, "1"]')[['iid', 'iid']].to_csv(self.sample_path_males, index = False, header = None, sep = ' ')
                famf.query('gender in [2, "2"]')[['iid', 'iid']].to_csv(self.sample_path_females, index = False, header = None, sep = ' ')
            else:
                tempdf = self.df.query('rfid in @gen_iids')
                tempdf[['rfid', 'rfid']].to_csv(self.sample_path, index = False, header = None, sep = ' ')
                tempdf.query('sex in ["M", "m", "male", "1", 1]')[['rfid', 'rfid']].to_csv(self.sample_path_males, index = False, header = None, sep = ' ')
                tempdf.query('sex in ["F", "f", "female", "2", 2]')[['rfid', 'rfid']].to_csv(self.sample_path_females, index = False, header = None, sep = ' ')

        print('calculating missing hwe maf for autossomes and MT')
        plink(bfile = self.all_genotypes, chr = f'1-{self.n_autosome} MT', hardy = '', keep = self.sample_path, thread_num =  self.threadnum, 
              freq = '', missing = '', nonfounders = '', out = f'{self.path}genotypes/autosomes', 
              chr_set = f'{self.n_autosome} no-xy') #autosome_num = 20
        print('calculating missing hwe maf for X')
        plink(bfile = self.all_genotypes, chr = 'X', hardy = '', keep = self.sample_path, thread_num =  self.threadnum,
              freq = '' , missing = '', nonfounders = '', out = f'{self.path}genotypes/xfilter',
              filter_females = '', chr_set = f'{self.n_autosome} no-xy')
        print('calculating missing hwe maf for Y')
        plink(bfile = self.all_genotypes, chr = 'Y', hardy = '', keep = self.sample_path, thread_num =  self.threadnum,
              freq = '' , missing = '', nonfounders = '', out = f'{self.path}genotypes/yfilter', 
              filter_males = '', chr_set = f'{self.n_autosome} no-xy')

        full = pd.concat([pd.concat(
             [pd.read_csv(f'{self.path}genotypes/{x}.lmiss', sep = '\s+')[['CHR','SNP', 'F_MISS']].set_index('SNP'),
              pd.read_csv(f'{self.path}genotypes/{x}.hwe', sep = '\s+')[['SNP', 'GENO' ,'P']].set_index('SNP')\
                                                                                .set_axis(['GENOTYPES','HWE'], axis = 1),
              pd.read_csv(f'{self.path}genotypes/{x}.frq', sep = '\s+')[['SNP', 'MAF', 'A1', 'A2']].set_index('SNP')], axis = 1) 
                for x in tqdm(['autosomes', 'xfilter', 'yfilter'])])

        full['PASS_MISS'] = ((full.F_MISS < thresh_m) + \
                             (full.CHR == self.n_autosome + 2 )) > 0 
        full['PASS_MAF'] = ((full.MAF - .5).abs() <= .5 - thresh_maf)# +
                            #(full.CHR == self.n_autosome + 2)) > 0 
        full['PASS_HWE']= ((full.HWE > thresh_hwe) + \
                          (full.CHR == self.n_autosome + 4) + \
                          (full.CHR == self.n_autosome + 2 )) > 0 
        full['PASS'] = full['PASS_MISS'] * full['PASS_MAF'] * full['PASS_HWE']

        full[full.PASS].reset_index()[['SNP']].to_csv(accepted_snps_path,index = False, header = None)
        full.to_parquet(f'{self.path}genotypes/snpquality.parquet.gz', compression='gzip')
        
        with open(f'{self.path}genotypes/parameter_thresholds.txt', 'w') as f: 
            f.write(f'--geno {thresh_m}\n--maf {thresh_maf}\n--hwe {thresh_hwe}')

        if verbose:
            display(full.value_counts(subset=  full.columns[full.columns.str.contains('PASS')].to_list())\
                                                   .to_frame().set_axis(['count for all chrs'], axis = 1))
            for i in sorted(full.CHR.unique())[-4:]:
                display(full[full.CHR == i].value_counts(subset=  full.columns[full.columns.str.contains('PASS')].to_list())\
                                                   .to_frame().set_axis([f'count for chr {i}'], axis = 1))

        plink(bfile = self.all_genotypes, extract = accepted_snps_path, keep = self.sample_path, make_bed = '', thread_num =  self.threadnum,
              set_missing_var_ids = '@:#', keep_allele_order = '',  set_hh_missing = '' , make_founders = '', 
              out = self.genotypes_subset, chr_set = f'{self.n_autosome} no-xy') #

        if makefigures:
            bim, fam, gen = pandas_plink.read_plink(self.genotypes_subset)
            print('making plots for heterozygosity per CHR')
            for numc, c in tqdm(list(enumerate(bim.chrom.unique().astype(str)))):
                snps = bim[bim['chrom'] == c]
                if int(c)<= self.n_autosome: snps = snps[::snps.shape[0]//2000+1]
                else: snps = snps[::snps.shape[0]//6000+1]
                f, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(20,20))
                for num, g in enumerate(['M', 'F']):
                    fams= fam[fam.gender == sex_encoder(g)]#[::snps.shape[0]//300]
                    ys = pd.DataFrame(gen[snps.i][:, fams.i].compute().T,columns = snps.snp, index = fams.iid )
                    f2, ax2 = plt.subplots(1, 1)
                    order  = dendrogram(linkage(ys, metric = nan_dist), ax = ax2 )['ivl']
                    plt.close(f2)
                    ys = ys.iloc[[int(i) for i in order], :]
                    sns.heatmap((ys - 1).abs(),  cbar=False, cmap = 'RdBu', ax = ax[num, 0])
                    ax[num, 0].title.set_text(f'G for {g} samples - Red:Het Blue:Hom ')
                    sns.heatmap(ys, cmap = 'Spectral', ax = ax[num, 1], cbar=False)
                    ax[num, 1].title.set_text(f'genotypes for {g} samples Blue:REF Red:ALT')
                plt.tight_layout()
                plt.savefig(f'{self.path}images/genotypes/heatmaps/chr{c}_genotypes_heatmap.png')
                plt.close()
                ys = pd.DataFrame(gen[snps.i][:, fam.i].compute().T,columns = snps.snp, index = fam.iid )
                make_LD_plot(ys, f'{self.path}images/genotypes/lds/ld_clumps_chr{c}')
                _distance_to_founders((bim, fam, gen), self.foundersbimfambed,
                                      f'{self.path}images/genotypes/dist2founders/dist2founder_chr{c}',c , nautosomes = self.n_autosome)
                _make_umap_plot((bim, fam, gen), self.foundersbimfambed, f'{self.path}images/genotypes/umap/umap_chr{c}',c,
                                nautosomes = self.n_autosome)
            
    def generateGRM(self, autosome_list: list = [], print_call: bool = True, allatonce: bool = False,
                    extra_chrs: list = ['X', 'Y', 'MT'], just_autosomes: bool = True, just_full_grm: bool = True,
                   full_grm: bool = True, **kwards):
        
        '''
        generates the grms, one per chromosome and one with all the chromossomes
        
        Parameters
        ----------
        autosome_list: list = []
            list of chromosomes that will be used, if not provided will use all autosomes + X
        print_call: bool = True
            prints every gcta call, doesn't work well with tqdm if all grms are done for all chrs
        extra_chrs: list = ['xchr']
            extra chromosomes to calculate the GRM MT has still to be implemented
        just_autosomes: bool = True
            uses just the autossomes for building the GRM
        just_full_grm: bool = True
            Runs only the full GRM
        '''
        print('generating GRM...')
        funcName = inspect.getframeinfo(inspect.currentframe()).function
        
        if not autosome_list:
            autosome_list = list(range(1,self.n_autosome+1))
            
        
        if allatonce:
            auto_flags = f'--autosome-num {int(len(autosome_list))} --autosome' if just_autosomes else ''
            sex_flags = f'--update-sex {self.sample_sex_path_gcta} --dc 1' #f' --sex {self.sample_sex_path}'
            
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} {sex_flags}\
                           --make-grm-bin {auto_flags} --out {self.autoGRM}_allatonce',
                           funcName, print_call = print_call) 
            
        all_filenames_partial_grms = pd.DataFrame(columns = ['filename'])

        if 'X' in extra_chrs:
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --autosome-num {self.n_autosome} \
                           --make-grm-xchr --out {self.xGRM}',
                        f'{funcName}_chrX', print_call = False)
            all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.xGRM
            
        if 'Y' in extra_chrs:
            try:
                self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --keep {self.sample_path_males} --autosome-num {self.n_autosome+4} \
                               --make-grm-bin --chr {self.n_autosome+2} --out {self.yGRM}',
                            f'{funcName}_chrY', print_call = False)
                #all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.yGRM
            except: print('could not make grm for chr Y')
            
        if 'MT' in extra_chrs:
            try:
                self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --autosome-num {self.n_autosome+6} --chr {self.n_autosome+4}\
                               --make-grm-bin --out {self.mtGRM}',
                            f'{funcName}_chrMT', print_call = False)
                all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.mtGRM
            except:print('could not make grm for chr MT')
            
        for c in tqdm(autosome_list):
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --chr {c} --autosome-num {self.n_autosome}\
                         --make-grm-bin --out {self.path}grm/{c}chrGRM',
                        f'{funcName}_chr{c}',  print_call = False)

            all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = f'{self.path}grm/{c}chrGRM'

        all_filenames_partial_grms.to_csv(f'{self.path}grm/listofchrgrms.txt', index = False, sep = ' ', header = None)

        self.bashLog(f'{self.gcta} {self.thrflag} --mgrm {self.path}grm/listofchrgrms.txt \
                       --make-grm-bin --out {self.autoGRM}', f'{funcName}_mergedgrms',  print_call = False )

        #if os.path.exists(f'{self.autoGRM}.grm.bin'): 
        #    self.failed_full_grm = False 
        
        return 1
    
    def scattermatrix(self, traitlist: list = []):
        if not traitlist: traitlist = self.traits
        for i in np.unique([x.replace('regressedlr_', '').split('_')[0] for x in traitlist] ):
            p = sns.PairGrid(self.df, vars=[x.replace('regressedlr_', '') for x in traitlist if i in x], hue="sex")
            p.map_diag(sns.distplot, hist=True) #kde=True, hist_kws={'alpha':0.5})
            p.map_upper(sns.scatterplot)
            p.map_lower(sns.kdeplot, levels=4, color=".2")
            plt.savefig(f'{self.path}images/scattermatrix/prefix{i}.png')
            
    def snpHeritability(self,  print_call: bool = False, save: bool = True, **kwards):
        '''
        The snpHeritability function is used to calculate the heritability of a set of traits in a dataset.
        
        Parameters
        ----------
        print_call: bool = False
            whether the command line call for each trait is printed.
        
        Design
        ------
        1.Initializes an empty DataFrame h2table to store the heritability results.
        2.Iterates over the list of traits (stored in the self.traits attribute)
          using the tqdm library for progress tracking.
        3.For each trait, it creates a file containing the phenotype data for that trait, by reading in a file,
          dropping missing data, filling in missing values with "NA", converting to string format and writing to a file.
        4.It then runs the command line program gtca to calculate the heritability of the trait using the phenotype file, 
          a pre-calculated genomic relationship matrix (self.autoGRM) and the --reml and --thrflag options. 
        5.The results are saved to a specified file.
        6.It reads in the results file, parses it and concatenates it to the h2table DataFrame.
        7.At the end of the loop, it writes the resulting h2table DataFrame to a file and returns it.
        '''
        print(f'starting snp heritability {self.project_name}')       
        
        h2table = pd.DataFrame()
        for trait in tqdm(self.traits):
            trait_file = f'{self.path}data/pheno/{trait}.txt'
            out_file = f'{self.path}results/heritability/{trait}' 
            
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file,  index = False, sep = ' ', header = None)
            
            if self.failed_full_grm:
                self.bashLog(f'{self.gcta} --reml {self.thrflag} --reml-no-constrain --autosome-num {self.n_autosome}\
                                           --pheno {trait_file} --mgrm {self.path}grm/listofchrgrms.txt --out {out_file}',
                            f'snpHeritability_{trait}', print_call = print_call) 
            else:
                self.bashLog(f'{self.gcta} --reml {self.thrflag}  --autosome-num {self.n_autosome}\
                                           --pheno {trait_file} --grm {self.autoGRM} --out {out_file}',
                            f'snpHeritability_{trait}', print_call = print_call) #--autosome
            try:
                a = pd.read_csv(f'{out_file}.hsq', skipfooter=6, sep = '\t',engine='python')
                b = pd.read_csv(f'{out_file}.hsq', skiprows=6, sep = '\t', header = None, index_col = 0).T.rename({1: trait})
                newrow = pd.concat(
                    [a[['Source','Variance']].T[1:].rename({i:j for i,j in enumerate(a.Source)}, axis = 1).rename({'Variance': trait}),
                    b],axis =1 )
                newrow.loc[trait, 'heritability_SE'] = a.set_index('Source').loc['V(G)/Vp', 'SE']
            except: 
                newrow = pd.DataFrame(np.array(['Fail']*10)[:, None].T, columns =['V(G)','V(e)','Vp','V(G)/Vp','logL0','LRT','df','Pval','n','heritability_SE'], index = [trait])
            
            h2table= pd.concat([h2table,newrow])

        if save: h2table.to_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t')
        return h2table
    
    def genetic_correlation_matrix(self,traitlist: list = [], print_call = False) -> pd.DataFrame():
        '''
        Create a blup model to get SNP effects and breeding values.
        
        Parameters
        ----------
        print_call: bool = False
            whether the command line call for each trait is printed.
        
        Design
        ------

        '''
        print(f'starting genetic correlation matrix {self.project_name}...')
        if not traitlist: traitlist = self.traits
        d_ = {t: str(num) for num, t in enumerate(['rfid']+ traitlist)} 
        self.df[['rfid', 'rfid']+ traitlist].fillna('NA').to_csv(f'{self.path}data/allpheno.txt', sep = '\t', header = None, index = False)
        outg = pd.DataFrame()
        outp = pd.DataFrame()
        genetic_table = pd.DataFrame()
        for trait1, trait2 in tqdm(list(itertools.combinations(traitlist, 2))):
            self.bashLog(f'''{self.gcta} --reml-bivar {d_[trait1]} {d_[trait2]} {self.thrflag} \
                --grm {self.autoGRM} --pheno {self.path}data/allpheno.txt --reml-maxit 1000 \
                --reml-bivar-lrt-rg 0 --out {self.path}temp/gencorr.temp''', 'genetic_correlation', print_call=False)
            temp = pd.read_csv(f'{self.path}temp/gencorr.temp.hsq', sep = '\t',dtype= {'Variance': float}, index_col=0 ,skipfooter=6)
            outg.loc[trait1, trait2] = f"{temp.loc['rG', 'Variance']}+-{temp.loc['rG', 'SE']}"
            outg.loc[trait2, trait1] = f"{temp.loc['rG', 'Variance']}+-{temp.loc['rG', 'SE']}"
            genetic_table.loc[len(genetic_table), ['trait1', 'trait2', 'genetic_correlation', 'rG_SE']] = [trait1, trait2, temp.loc['rG', 'Variance'], temp.loc['rG', 'SE']]
            if os.path.exists(f'{self.path}logerr/genetic_correlation.log'):
                outg.loc[trait1, trait2] = f"0 +- *"
                outg.loc[trait2, trait1] = f"0 +- *"
                bash(f'rm {self.path}logerr/genetic_correlation.log')
            phecorr = str(self.df[[trait1, trait2]].corr().iloc[0,1])
            outp.loc[trait2, trait1] = phecorr.replace('nan', '0')+ ' +- ' + ( '*' if 'nan' in phecorr else '0')
            outp.loc[trait1, trait2] = phecorr.replace('nan', '0')+ ' +- ' + ( '*' if 'nan' in phecorr else '0')
            #out.loc[trait2, trait1] = out.loc[trait1, trait2] 
        outg = outg.fillna('1+-0').rename(lambda x: x.replace('regressedlr_', '')).rename(lambda x: x.replace('regressedlr_', ''), axis = 1)
        outp = outp.fillna('1+-0').rename(lambda x: x.replace('regressedlr_', '')).rename(lambda x: x.replace('regressedlr_', ''), axis = 1)

        hieg = linkage(distance.pdist(outg.applymap(lambda x: float(x.split('+-')[0])).T)) #method='average'
        lk = leaves_list(hieg)
        outg = outg.loc[[x.replace('regressedlr_', '') for x in traitlist], [x.replace('regressedlr_', '') for x in traitlist]].iloc[lk, lk]
        outp = outp.loc[[x.replace('regressedlr_', '') for x in traitlist], [x.replace('regressedlr_', '') for x in traitlist]].iloc[lk, lk] 
        hie = linkage(distance.pdist(outg.applymap(lambda x: float(x.split('+-')[0])).T)) #, method='euclidean'
        genetic_table.to_csv(f'{self.path}results/heritability/genetic_correlation_melted_table.csv')
        outg.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv')
        outmixed = outg.mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('')  +  outp.mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('')
        if not os.path.isfile(self.heritability_path): self.snpHeritability()
        H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col= 0)
        H2['her_str'] = H2['V(G)/Vp'].round(3).astype(str) + ' +- ' + H2.heritability_SE.round(3).astype(str)
        for i in outmixed.columns: outmixed.loc[i,i] =  H2.loc['regressedlr_'+i, 'her_str']
        outmixed.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix.csv')
        a = sns.clustermap(outmixed.applymap(lambda x: float(x.split('+-')[0])).copy().T,  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                annot=outmixed.applymap(lambda x: '' if '*' not in x else '*').copy().T, vmin =-1, vmax =1, center = 0 , fmt = '', square = True, linewidth = .3, figsize=(25, 25) )
        dendrogram(hie, ax = a.ax_col_dendrogram)
        plt.savefig(f'{self.path}images/genetic_correlation_matrix.png', dpi = 400)
        return outmixed
        
    def BLUP(self,  print_call: bool = False, save: bool = True, frac: float = 1.,**kwards):
        '''
        Create a blup model to get SNP effects and breeding values.
        
        Parameters
        ----------
        print_call: bool = False
            whether the command line call for each trait is printed.
        
        Design
        ------

        '''
        print(f'starting BLUP model {self.project_name}...')      

        for trait in tqdm(self.traits):
            os.makedirs( f'{self.path}data/BLUP', exist_ok = True)
            os.makedirs( f'{self.path}results/BLUP', exist_ok = True)
            trait_file, trait_rfids = f'{self.path}data/BLUP/{trait}_trait_.txt', f'{self.path}data/BLUP/{trait}_train_rfids.txt'
            out_file = f'{self.path}results/BLUP/{trait}' 
            tempdf = self.df.sample(frac = frac) if frac < .999 else self.df.copy()
            tempdf[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file,  index = False, sep = ' ', header = None)
            tempdf[['rfid', 'rfid']].to_csv(trait_rfids, header = None, index = False, sep = ' ')
            self.bashLog(f'{self.gcta} --grm {self.autoGRM} --autosome-num {self.n_autosome}  --keep {trait_rfids} \
                                        --make-grm  --out {out_file}_GRMsubset',
                        f'BLUP_{trait}_GRM', print_call = print_call)
            self.bashLog(f'{self.gcta} --reml {self.thrflag} \
                                       --pheno {trait_file} --grm {out_file}_GRMsubset --reml-pred-rand --out {out_file}_BV',
                        f'BLUP_{trait}_BV', print_call = print_call) #--autosome
            self.bashLog(f'{self.gcta} --bfile {self.genotypes_subset} {self.thrflag} --blup-snp {out_file}_BV.indi.blp \
                           --autosome --autosome-num {self.n_autosome} --out {out_file}_snpscores',
                        f'BLUP_{trait}_snpscores', print_call = print_call) #--autosome

        BVtable = pd.concat([pd.read_csv(f'{self.path}results/BLUP/{trait}_BV.indi.blp',sep = '\t',  header = None)\
                                    .dropna(how = 'all', axis = 1).iloc[:, [0, -1]]\
                                    .set_axis(['rfid',f'BV_{trait}_{self.project_name}'], axis = 1).set_index('rfid')
                            for trait in tqdm(self.traits)],axis=1)
            
            
        if save: BVtable.to_csv(f'{self.path}results/BLUP/BLUP.tsv', sep = '\t')
        return BVtable
    
    def BLUP_predict(self,genotypes2predict: str = '', rfid_subset: list = [], traits: list = [], print_call: bool = False, save: bool = True) -> pd.DataFrame():
        '''
        Create a blup model to get SNP effects and breeding values.

        Parameters
        ----------
        print_call: bool = False
            whether the command line call for each trait is printed.

        Design
        ------

        '''
        print(f'starting blup prediction for {self.project_name} -> {genotypes2predict}')
        if type(traits) == str: traits = [traits]
        pred_path = f'{self.path}results/BLUP/predictions'
        os.makedirs( pred_path, exist_ok = True) 
        #for file in glob(f'{pred_path}/*'): os.remove(file)
        if len(rfid_subset) > 0:
            tempdf = pd.DataFrame(rfid_subset, columns = ['rfid'])
            tempdf[['rfid', 'rfid']].to_csv(f'{pred_path}/test_rfids', header = None, index = False, sep = ' ')
            keep_flag = f'--keep {pred_path}/test_rfids'
        else: keep_flag = ''
        traitlist = self.traits if not traits else traits
        genotypes2predictname = genotypes2predict.split('/')[-1]
        #print(traitlist)

        for trait in tqdm(traitlist):
            self.bashLog(f'plink {self.thrflag} --bfile {genotypes2predict} {keep_flag} --score {self.path}results/BLUP/{trait}_snpscores.snp.blp 1 2 3 sum --out {pred_path}/{trait}_{genotypes2predictname}', 
                 f'blup_predictions_{trait}_{genotypes2predictname}',print_call = print_call)
        outdf = pd.concat([pd.read_csv(f'{pred_path}/{trait}_{genotypes2predictname}.profile', sep = '\s+')[['IID', 'SCORESUM']].set_axis(['rfid', trait], axis =1).set_index('rfid') 
                   for trait in traitlist], axis = 1)
        if save: outdf.to_csv(f'{self.path}results/BLUP/BLUP_predictions.tsv', sep = '\t')

        return outdf
        
    
    def GWAS(self, traitlist: list = [] ,subtract_grm: bool = True, loco: bool = True , run_per_chr: bool = False,
             print_call: bool = False, **kwards):
        """
        This function performs a genome-wide association study (GWAS) on the provided genotype data using the GCTA software.
        
        Parameters
        ----------
    
        subtract_grm: 
            a boolean indicating whether to subtract the genomic relatedness matrix (GRM) from the GWAS results (default is False)
        loco: 
            a boolean indicating whether to perform leave-one-chromosome-out (LOCO) analysis (default is True)
        print_call: 
            a boolean indicating whether to print the command that is being executed (default is False)

        Design
        ------
        
        for each trait in the trait list:
        it creates a command to run GWAS using GCTA, with the necessary arguments and flags, 
        including the genotype data file,
        phenotype data file, and options for subtracting the GRM and performing LOCO analysis
        it runs the command using the bashLog method, which logs the command and its output, 
        and displays the command if print_call is set to True
        it saves the GWAS results to a file in the 'results/gwas' directory,
        with the filename indicating the trait, GRM subtraction status, and LOCO status.
        """

        print(f'starting GWAS {self.project_name}')
        from joblib import Parallel, delayed
        
        if len(traitlist) == 0:
            traitlist = self.traits
        #results = Parallel(n_jobs=2)(delayed(countdown)(10**7) for _ in range(20))
        
        if not run_per_chr:
            print('running gwas per trait...')
            for trait in tqdm(traitlist):
                grm_flag = f'--grm {self.path}grm/AllchrGRM ' if subtract_grm else ''
                loco_flag = '-loco' if loco else ''
                self.bashLog(f"{self.gcta} {self.thrflag} {grm_flag} \
                --autosome-num {self.n_autosome}\
                --pheno {self.path}data/pheno/{trait}.txt \
                --bfile {self.genotypes_subset} \
                --mlma{loco_flag} \
                --out {self.path}results/gwas/{trait}",\
                            f'GWAS_{loco_flag[1:]}_{trait}',  print_call = print_call)
                if not os.path.exists(f'{self.path}results/gwas/{trait}.loco.mlma')+os.path.exists(f'{self.path}results/gwas/{trait}.mlma'):
                    print(f"couldn't run trait: {trait}")
                    self.GWAS(traitlist = traitlist, run_per_chr = True, print_call= print_call)
                    return 2
            ranges = [self.n_autosome+1, self.n_autosome+2, self.n_autosome+4]
        else:
            print('running gwas per chr per trait...')
            ranges = [i for i in range(1,self.n_autosome+2) if i != self.n_autosome+3]
        
        for trait, chrom in tqdm(list(itertools.product(traitlist,ranges))):
            chromp2 = self.replacenumstoXYMT(chrom)
            subgrmflag = f'--mlma-subtract-grm {self.path}grm/{chromp2}chrGRM' if chromp2 != 'y' else ''
            self.bashLog(f'{self.gcta} {self.thrflag} --pheno {self.path}data/pheno/{trait}.txt --bfile {self.genotypes_subset} \
                                       --grm {self.path}grm/AllchrGRM --autosome-num {self.n_autosome} \
                                       --chr {chrom} {subgrmflag} --mlma \
                                       --out {self.path}results/gwas/{trait}_chrgwas{chromp2}', 
                        f'GWAS_{chrom}_{trait}', print_call = print_call)
                
        return 1
    
    def addGWASresultsToDb(self, researcher: str , round_version: str, gwas_version: str,filenames: list = [],
                           pval_thresh: float = 1e-4, safe_rebuild: bool = True,**kwards):
        '''
        The addGWASresultsToDb function is used to add GWAS results to a pre-existing PheWAS database. 
        The function takes in several input parameters, including a list of GWAS result filenames,
        a threshold for the GWAS p-value, and various metadata (researcher, round_version, gwas_version, etc.).
        
        Parameters
        ----------
        researcher: str
        round_version: str
        gwas_version: str
        filenames: list = []
        pval_thresh: float = 1e-4
        safe_rebuild: bool = True
        
        Design
        ------
        The function first checks if the input filenames are a list or a single string,
        and if it's empty it will look for the files in the results/gwas/ directory, 
        it then reads in the GWAS results and assigns various metadata to the data.
        It also adds the number of snps and filters the data by the pval_thresh parameter.
        It then concatenates the new data with the pre-existing PheWAS database,
        and drops any duplicate rows based on specified columns.
        Finally, it saves the updated PheWAS database to a file.
        
        '''
        print(f'starting adding gwas to database ... {self.project_name}') 
        if type(filenames) == str:
            filenames = [filenames]
            
        elif len(filenames) == 0 :
            #filenames = [f'{self.path}results/gwas/{trait}.loco.mlma' for trait in self.traits]
            filenames = glob(f'{self.path}results/gwas/*.mlma') 
        all_new = []
        for file in tqdm(filenames):
            trait = re.findall('/([^/]+).mlma', file)[0].replace('.loco', '').split('_chrgwas')[0]
            new_info = pd.read_csv(file, sep = '\t').assign(**kwards).assign(uploadeddate = datetime.today().strftime('%Y-%m-%d'),
                                                                               researcher = researcher,
                                                                               project = self.project_name,
                                                                               trait = trait.replace('regressedlr_', ''),
                                                                               trait_description = self.get_trait_descriptions[trait],
                                                                               filename = file,
                                                                               pval_threshold = pval_thresh,
                                                                               genotypes_file = self.all_genotypes,
                                                                               round_version = round_version,
                                                                               gwas_version = gwas_version)

            new_info['n_snps'] =  new_info.shape[0]
            new_info = new_info.query(f'p < {pval_thresh}')
            all_new += [new_info]

        all_new = pd.concat(all_new)  
        if not os.path.isfile(self.phewas_db):
            alldata = all_new
        else:
            try:
                alldata = pd.concat([all_new, pd.read_parquet(self.phewas_db)])
            except:
                print(f"Could not open phewas database in file: {self.phewas_db}, rebuilding db with only this project")
                if safe_rebuild: 
                    raise ValueError('not doing anything further until data is manually verified')
                    return
                else: 
                    alldata = all_new
        
        alldata.drop_duplicates(subset = ['researcher', 'project', 'round_version', 'trait', 'SNP', 'uploadeddate'], 
                                keep='first').to_parquet(self.phewas_db, index = False, compression='gzip')
        
        return 1
        
        ###scp this to tscc
        # bash(#scp tsanches@tscc-login.sdsc.edu:/projects/ps-palmer/hs_rats/rattacafinalgenotypes/RattacaG01
        #/RattacaG01_QC_Sex_Het_pass_n971.vcf.gz.tbi rattaca_genotypes.vcf.gz.tbi
        
        
    def callQTLs(self, threshold: float = 5.3591, suggestive_threshold: float = 5.58, window: int = 2e6, subterm: int = 2,  add_founder_genotypes: bool = True,
                 ldwin = 7e6, ldkb = 7000, ldr2 = .4, qtl_dist = 7*1e6, NonStrictSearchDir = True, **kwards): # annotate_genome: str = 'rn7',
        
        '''
        The function callQTLs() is used to call quantitative trait loci (QTLs) from GWAS results. 
        It takes in several input parameters, including a threshold for the GWAS p-value, 
        a window size for identifying correlated SNPs, and parameters for linkage disequilibrium (LD) calculations.
        
        Parameters
        ----------
         threshold: float = 5.3
         window: int = 1e6
         ldwin: int = 1e6
         ldkb: int = 11000 
         ldr2: float = .4
         qtl_dist: int = 2*1e6
         NonStrictSearchDir = ''
         
        Design
        ------
        
        The function reads in GWAS results for each trait and chromosome. 
        If a non-strict search directory is provided, it reads in all GWAS results files in that directory.
        The top SNPs with p-values below the threshold are then selected and stored in the topSNPs DataFrame.
        
        The function then iterates over each trait and chromosome, and for each group, 
        it sets the index of the DataFrame to the SNP position. It then applies a negative log10 transformation to the p-values,
        and enters a while loop that continues until there are no more SNPs with p-values above the threshold.
        Within the loop, the SNP with the highest p-value is selected and correlated SNPs within the specified window are identified.
        If there are more than 2 correlated SNPs, the SNP is considered a QTL. The QTL is then added to the output DataFrame out,
        and linkage disequilibrium is calculated for the SNP using the plink command-line tool.
        '''
        print(f'starting call qtl ... {self.project_name}') 
        thresh = 10**(-threshold)
        
        if not NonStrictSearchDir:
            topSNPs = pd.DataFrame()
            for t, chrom in tqdm(list(itertools.product(self.traits, range(1,self.n_autosome+4)))):
                    chrom = self.replacenumstoXYMT(chrom)
                    filename = f'{self.path}results/gwas/{t}_chrgwas{chrom}.mlma'
                    if os.path.exists(filename):
                        topSNPs = pd.concat([topSNPs, pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)])
                    else: pass #print(f'could not locate {filename}')
            for t  in tqdm(self.traits):
                    filename = f'{self.path}results/gwas/{t}.loco.mlma'
                    if os.path.exists(filename):
                        topSNPs = pd.concat([topSNPs, pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)])
                    else: print(f'could not locate {filename}')
                    

        else:
            topSNPs = pd.concat([pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=re.findall('/([^/]*).mlma', 
                                                                                            filename)[0].replace('.loco', '').split('_chrgwas')[0]) for
                                 filename in tqdm(glob(f"{self.path}results/gwas/*.mlma"))])

        out = pd.DataFrame()

        for (t, c), df in tqdm(topSNPs.groupby(['trait','Chr'])):
            df = df.set_index('bp')
            df.p = -np.log10(df.p)

            while df.query('p > @threshold').shape[0]:
                idx = df.p.idxmax()
                maxp = df.loc[idx]
                correlated_snps = df.loc[int(idx- window//2): int(idx + window//2)].query('p > @maxp.p - @subterm')
                qtl = True if correlated_snps.shape[0] > 2 else False

                ldfilename = f'{self.path}temp/r2/temp_qtl_n_{t}'
                self.bashLog(f'plink --bfile {self.genotypes_subset} --chr {c} --ld-snp {maxp.SNP} \
                                     --ld-window {ldwin} {self.thrflag} \
                                     --nonfounders --r2 \
                                     --ld-window-r2 {ldr2} --out {ldfilename}',\
                             f'qlt_{t}', False )#--ld_window_kb {ldkb} --nonfounders might be able to remove 

                try: 
                    plinkrestemp =  pd.read_csv(f'{ldfilename}.ld', sep = r'\s+').query('R2 > @ldr2')
                    ldSNPS = plinkrestemp.SNP_B.to_list() + [maxp.SNP]
                    ldSNPS_LEN = plinkrestemp.BP_B.agg(lambda x: (x.max()-x.min())/1e6)
                    df = df.query('~(@idx - @qtl_dist//2 < index < @idx + @qtl_dist//2) and (SNP not in @ldSNPS)')
                except:
                    ldSNPS = [maxp.SNP]
                    ldSNPS_LEN = 0
                    df = df.query('(SNP not in @ldSNPS)') ##### should this be different than the one above?
                #if sum(cnt.values()) % 10 == 0: print(cnt)
                            
                out = pd.concat([out,
                                 maxp.to_frame().T.assign(QTL= qtl, interval_size = '{:.2f} Mb'.format(ldSNPS_LEN))],
                                 axis = 0)
                
        if not len(out):
            print('no SNPS were found, returning an empty dataframe')
            out.to_csv(f'{self.path}results/qtls/allQTLS.csv', index = False)
            return out
            

        out =  out.reset_index().rename({'index': 'bp'}, axis = 1).sort_values('trait')#.assign(project = self.project_name)
        out['trait_description'] = out.trait.apply(lambda x: self.get_trait_descriptions[x])
        out['trait'] = out.trait.apply(lambda x: x.replace('regressedlr_', ''))
        if add_founder_genotypes:
            dffounder = pd.read_csv(self.founder_genotypes, sep = '\t')
            dffounder['SNP'] = dffounder['chr'].astype('str')+ ':'+dffounder.pos.astype('str')
            out = out.merge(dffounder.drop(['chr', 'pos'], axis = 1), on = 'SNP', how = 'left')
            
        out['significance_level'] = out.p.apply(lambda x: '5%' if x > suggestive_threshold else '10%')
        
        self.allqtlspath = f'{self.path}results/qtls/allQTLS.csv'
        
        out.to_csv(self.allqtlspath.replace('allQTLS', 'QTLSb4CondAnalysis'), index = False)
        print('running conditional analysis...')
        self.pbim, self.pfam, self.pgen = pandas_plink.read_plink(self.genotypes_subset)
        out = self.conditional_analysis_filter(out, threshold)
        out.to_csv(self.allqtlspath, index = False)
        out.to_csv(self.allqtlspath.replace('.csv', '.tsv'), index = False, sep = '\t')
        return out.set_index('SNP') 
    
    def conditional_analysis(self, trait: str, snpdf: pd.DataFrame() = pd.DataFrame(), threshold: float = 5.3591):
        os.makedirs(f'{self.path}results/cojo', exist_ok=True)
        os.makedirs(f'{self.path}temp/cojo',exist_ok=True)
        
        if not snpdf.shape[0]: print(f'running conditional analysis for trait {trait} and all snps above threshold {threshold}')
        else: 
            #print(snpdf.shape)
            snpstring = ' '.join(snpdf.SNP)
            print(f'running conditional analysis for trait {trait} and all snps below threshold {snpstring}')

        pbimtemp = self.pbim.assign(n = self.df.count()[trait] ).rename({'snp': 'SNP', 'n':'N'}, axis = 1)[['SNP', 'N']] #- da.isnan(pgen).sum(axis = 1)
        tempdf = pd.concat([pd.read_csv(f'{self.path}results/gwas/{trait}.loco.mlma', sep = '\t'),
                           pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasx.mlma', sep = '\t'),
                           pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasy.mlma', sep = '\t'),
                           pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasmt.mlma', sep = '\t')]).rename({'Freq': 'freq'}, axis =1 )
        tempdf = tempdf.merge(pbimtemp, on = 'SNP')[['SNP','A1','A2','freq','b','se','p','N' ]]
        mafile, snpl = f'{self.path}temp/cojo/tempmlma.ma', f'{self.path}temp/cojo/temp.snplist'
        tempdf.to_csv(mafile, index = False, sep = '\t')
        tempdf[-np.log10(tempdf.p) >threshold][['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
        mafile, snpl = f'{self.path}temp/cojo/tempmlma.ma', f'{self.path}temp/cojo/temp.snplist'
        tempdf.to_csv(mafile, index = False, sep = '\t')
        if not snpdf.shape[0]:
            tempdf[-np.log10(tempdf.p) > threshold][['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
        else: snpdf[['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
        cojofile = f'{self.path}temp/cojo/tempcojo'
        self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --cojo-slct --cojo-collinear 0.99 --cojo-p {10**-(threshold-2)} --cojo-file {mafile} --cojo-cond {snpl} --out {cojofile}', f'cojo_test', print_call=False)
        if os.path.isfile(f'{cojofile}.jma.cojo'):
            return pd.read_csv(f'{cojofile}.jma.cojo', sep = '\t')
        print(f'Conditional Analysis Failed for  trait {trait} and all snps below threshold {snpstring}, returning the top snp only')
        return pd.DataFrame(snpdf.SNP.values, columns = ['SNP'])

    def conditional_analysis_filter(self, qtltable, threshold: float = 5.3591):
        return qtltable.groupby(['Chr', 'trait']).progress_apply(lambda df: df.loc[df.SNP.isin(self.conditional_analysis('regressedlr_' +df.name[1].replace('regressedlr_', ''), df, threshold).SNP.to_list())]
                                                            if df.shape[0] > 1 else df).reset_index(drop= True)
        
    def effectsize(self, qtltable: pd.DataFrame(), display_plots: bool = True):
        print(f'starting effect size plot... {self.project_name}') 
        out = qtltable.reset_index()
        usnps = out.SNP.unique()
        aa = ','.join(usnps)
        self.bashLog(f'plink --bfile {self.genotypes_subset} --snps {aa} --recode --alleleACGT --out {self.path}temp/stripplot', 'stripplot', print_call=False)
        snporder = list(pd.read_csv(f'{self.path}temp/stripplot.map', sep = '\s+', header = None)[1])
        temp = pd.read_csv(f'{self.path}temp/stripplot.ped', sep = '\s+', index_col = [0,1,2,3,4,5], header = None)
        temp = temp.iloc[:, 1::2].set_axis(snporder, axis = 1) + temp.iloc[:, 0::2].set_axis(snporder, 1)
        temp = temp.reset_index().drop([1,2,3,4,5], axis = 1).rename({0:'rfid'},axis = 1)
        temp.rfid = temp.rfid.astype(str)
        temp = temp.merge(self.df[self.traits + 
                          [t.replace('regressedlr_', '') for t in self.traits] + 
                          ['rfid', 'sex']].rename(lambda x: x.replace('regressedlr_', 'normalized '), axis =1), on = 'rfid')

        for name, row in tqdm(list(out.iterrows())):
            f, ax = plt.subplots(1, 2)
            for num, ex in enumerate(['normalized ', '']):
                sns.boxplot(temp[temp[row.SNP]!= '00'].sort_values(row.SNP), x = row.SNP, y = ex+ row.trait, color = 'steelblue', ax = ax[num] )#cut = 0,  bw= .2, hue="sex", split=True
                sns.stripplot(temp[temp[row.SNP]!= '00'].sort_values(row.SNP), x = row.SNP,  y =ex+ row.trait, color = 'black', jitter = .2, alpha = .4, ax = ax[num] )
                ax[num].hlines(y = 0 if num==0 else temp[temp[row.SNP]!= '00'][ex+ row.trait].mean() , xmin =-.5, xmax=2.5, color = 'black', linewidth = 2, linestyle = '--')
                sns.despine()
            os.makedirs(f'{self.path}images/boxplot/', exist_ok=True)
            plt.savefig(f'{self.path}images/boxplot/boxplot{row.SNP}__{row.trait}.png'.replace(':', '_'))
            plt.tight_layout()
            plt.show()
            plt.close()   
    
    def locuszoom(self, qtltable: pd.DataFrame(), threshold: float = 5.3591, suggestive_threshold: float = 5.58, qtl_r2_thresh: float = .6, padding: float = 2e5, annotate_genome: str = 'rn7'):
        '''
        Only works on TSCC
        '''
        print(f'generating locuszoom info for project {self.project_name}')
        out = qtltable.reset_index()
        causal_snps = []
        for name, row in tqdm(list(out.iterrows())):
            ldfilename = f'{self.path}results/lz/temp_qtl_n_@{row.trait}@{row.SNP}'
            r2 = self.plink(bfile = self.genotypes_subset, chr = row.Chr, ld_snp = row.SNP, ld_window_r2 = 0.001, r2 = 'dprime',\
                                    ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb =  6000, nonfounders = '').loc[:, ['SNP_B', 'R2', 'DP']] 
            gwas = pd.concat([pd.read_csv(x, sep = '\t') for x in glob(f'{self.path}results/gwas/regressedlr_{row.trait}.loco.mlma') \
                                + glob(f'{self.path}results/gwas/regressedlr_{row.trait}_chrgwasx.mlma')]).drop_duplicates(subset = 'SNP')
            tempdf = pd.concat([gwas.set_index('SNP'), r2.rename({'SNP_B': 'SNP'}, axis = 1).drop_duplicates(subset = 'SNP').set_index('SNP')], join = 'inner', axis = 1)
            tempdf = self.annotate(tempdf.reset_index(), annotate_genome, 'SNP', save = False).set_index('SNP').fillna('UNK')
            tempdf.to_csv( f'{self.path}results/lz/lzplottable@{row.trait}@{row.SNP}.tsv', sep = '\t')
            
            ## potential causal mutations
            subcausal = tempdf.query("putative_impact not in ['UNK', 'MODIFIER']").assign(trait = row.trait, SNP_qtl = row.SNP)
            subcausal.columns = subcausal.columns.astype(str)
            if subcausal.shape[0] > 0:
                subcausal = subcausal.loc[subcausal.R2 > qtl_r2_thresh,  ~subcausal.columns.str.contains('\d\d', regex = True) ]\
                                     .sort_values('putative_impact', ascending = False).drop('errors', errors = 'ignore' , axis =1 ).reset_index()\
                                     .drop(['Chr', 'bp', 'se', 'geneid'], errors = 'ignore' ,axis = 1).reset_index().set_index('SNP_qtl')
                causal_snps += [subcausal]
            else: pass
            
        causal_snps = pd.concat(causal_snps).to_csv(f'{self.path}results/qtls/possible_causal_snps.tsv', sep = '\t')
        
        genome_lz_path = {'rn6': 'rn6', 
                          'rn7':'rn7', 
                          'cfw': 'm38',
                          'm38': 'm38'}[annotate_genome]

        for num, (_, qtl_row) in tqdm(list(enumerate(qtltable.reset_index().iterrows()))):
            topsnpchr, topsnpbp = qtl_row.SNP.split(':')
            topsnpchr = topsnpchr.replace('X',str(self.n_autosome+1))
            try:test = pd.read_csv(f'{self.path}results/lz/lzplottable@{qtl_row.trait}@{qtl_row.SNP}.tsv', 
                               dtype = {'p':float, 'bp': int, 'R2': float, 'DP': float}, sep = '\t',
                                   na_values =  na_values_4_pandas)\
                         .replace([np.inf, -np.inf], np.nan)\
                         .dropna(how = 'any', subset = ['Freq','b','se','p','R2','DP'])
            except: raise Exception(f"couldn't open this file {self.path}results/lz/lzplottable@{qtl_row.trait}@{qtl_row.SNP}.tsv") 
            test['-log10(P)'] = -np.log10(test.p)
            range_interest = test.query('R2> .6')['bp'].agg(['min', 'max'])
            test = test.query(f'{range_interest["min"] - padding}<bp<{range_interest["max"] + padding}')
            test.SNP = 'chr'+test.SNP
            os.makedirs(f'{self.path}results/lz/p/', exist_ok = True)
            os.makedirs(f'{self.path}results/lz/r2/', exist_ok = True)
            os.makedirs(f'{self.path}images/lz/', exist_ok = True)
            lzpvalname, lzr2name = f'{self.path}results/lz/p/{qtl_row.trait}@{qtl_row.SNP}.tsv', f'{self.path}results/lz/r2/{qtl_row.trait}@{qtl_row.SNP}.tsv'
            test.rename({'SNP': 'MarkerName', 'p':"P-value"}, axis = 1)[['MarkerName', 'P-value']].to_csv(lzpvalname, index = False, sep = '\t')
            test.assign(snp1 = 'chr'+qtl_row.SNP).rename({"SNP": 'snp2', 'R2': 'rsquare', 'DP': 'dprime'}, axis = 1)\
                        [['snp1', 'snp2', 'rsquare', 'dprime']].to_csv(lzr2name, index = False, sep = '\t')
            os.system(f'chmod +x {lzpvalname}')
            os.system(f'chmod +x {lzr2name}')
            for filest in glob(f'{self.path}temp/{qtl_row.trait}*{qtl_row.SNP}'): os.system(f'rm -r {filest}')
            
            os.system(f'''module load R && module load python && \
                /projects/ps-palmer/software/local/src/locuszoom/bin/locuszoom \
                --metal {lzpvalname} --ld {lzr2name} \
                --refsnp {qtl_row.SNP} --chr {int(topsnpchr)} --start {int(range_interest["min"] - padding)} --end {int(range_interest["max"] + padding)} --build manual \
                --db /projects/ps-palmer/gwas/databases/databases_lz/{genome_lz_path}.db \
                --plotonly showRecomb=FALSE showAnnot=FALSE --prefix {self.path}temp/{qtl_row.trait} signifLine="{threshold},{suggestive_threshold}" signifLineColor="red,blue" \
                title = "{qtl_row.trait} SNP {qtl_row.SNP}" > /dev/null 2>&1 ''')
            
            #os.system(f'''conda run -n lzenv && \
            #    locuszoomfiles/bin/locuszoom \ 
            #    --metal {lzpvalname} --ld {lzr2name} \
            #    --refsnp {qtl_row.SNP} --chr {int(topsnpchr)} --start {int(range_interest["min"] - padding)} --end {int(range_interest["max"] + padding)} --build manual \
            #    --db /projects/ps-palmer/gwas/databases/databases_lz/{genome_lz_path}.db \
            #    --plotonly showRecomb=FALSE showAnnot=FALSE --prefix {self.path}temp/{qtl_row.trait} signifLine="{threshold},{suggestive_threshold}" signifLineColor="red,blue" \
            #    title = "{qtl_row.trait} SNP {qtl_row.SNP}" > /dev/null 2>&1 ''') #--build u01_peter_kalivas_v7 module load R && module load python
            path = glob(f'{self.path}temp/{qtl_row.trait}*{qtl_row.SNP}/*.pdf'.replace(':', '_'))[0]
            for num,image in enumerate(convert_from_path(path)):
                bn = basename(path).replace('.pdf', '.png')
                if not num: image.save(f'{self.path}images/lz/lz__{qtl_row.trait}__{qtl_row.SNP}.png'.replace(':', '_'), 'png')
    
    def phewas(self, qtltable: pd.DataFrame(), ld_window: int = int(3e6), pval_threshold: float = 1e-4, nreturn: int = 1 ,r2_threshold: float = .8,\
              annotate: bool = True, annotate_genome: str = 'rn7', **kwards) -> pd.DataFrame():
        '''
        This function performs a phenotype-wide association study (PheWAS) on a 
        given QTL table by comparing the QTLs to a pre-existing PheWAS database. 
        
        Parameters
        ----------

        qtltable: pd.DataFrame()
            the QTL table to perform PheWAS on
        ld_window: int = int(3e6)
            the size of the window around the QTL to search for PheWAS associations (default is 3e6)
        pval_threshold: float = 1e-5
            the threshold for the p-value of the PheWAS associations (default is 1e-5)
        nreturn: int = 1
            the number of PheWAS associations to return for each QTL (default is 1)
        r2_threshold: float = .8 
            the threshold for the r-squared value of the PheWAS associations (default is 0.8)
        annotate: bool : 
            a boolean indicating whether to annotate the resulting PheWAS table (default is True)
        annotate_genome: 
            the genome to use for annotation, if annotate is True (default is 'rn6')

        Design
        ------

        1.Reads in PheWAS data from a pre-existing PheWAS database file and 
          filters the data based on the provided p-value threshold and project name.
        2.Merges the filtered PheWAS data with the QTL table on the 'SNP' column and returns only the exact matches
        3.it saves the exact match table to a file
        4.it gets the nearby snp that are in ld with the QTL using plink
        5.it merges the PheWAS data with the nearby snp and returns only the matches that pass the R2 threshold
        6.it saves the window match table to a file
        7.it groups the resulting dataframe by 'SNP_QTL','project', 'trait_phewasdb' 
          and for each group it returns the nreturn smallest pval
        8.it saves the final table to a file
        9.if annotate is True, it annotates the PheWAS table using the annotate method and the provided annotate_genome
        10.return the final PheWAS table.
        '''
        print(f'starting phewas ... {self.project_name}')  
        if qtltable.shape == (0,0): qtltable = pd.read_csv(self.annotatedtablepath).set_index('SNP')
        db_vals = pd.read_parquet(self.phewas_db).query(f'p < {pval_threshold}')  #, compression='gzip'   and project != "{self.project_name}   
        db_vals.SNP = db_vals.SNP.str.replace('chr', '')
        db_vals.trait_description = db_vals.trait_description.astype(str).apply(lambda x: re.sub(r'[^\d\w ]+',' ', x))
        
        table_exact_match = db_vals.merge(qtltable.reset_index(), on = 'SNP', how = 'inner', suffixes = ('_phewasdb', '_QTL'))
        table_exact_match = table_exact_match.query(f'project != "{self.project_name}" or trait_phewasdb != trait_QTL ')
        self.phewas_exact_match_path = f'{self.path}results/phewas/table_exact_match.csv'
        table_exact_match.to_csv(self.phewas_exact_match_path )
        #pd.concat([qtltable, db_vals] ,join = 'inner', axis = 1)
        
        nearby_snps = pd.concat([
             self.plink(bfile = self.genotypes_subset, chr = row.Chr, r2 = 'dprime', ld_snp = row.name,
               ld_window = ld_window, thread_num = 12, nonfounders = '')\
              .query(f'R2 > {r2_threshold}')\
              .drop(['CHR_A', 'BP_A', 'CHR_B'], axis = 1)\
              .rename({'SNP_A': 'SNP', 'SNP_B': 'NearbySNP', 'BP_B': 'NearbyBP'}, axis = 1)\
              .assign(**row.to_dict())\
              .set_index('SNP')
              for  _, row in tqdm(list(qtltable.iterrows())) ])
        nearby_snps.NearbySNP = nearby_snps.NearbySNP.str.replace('chr', '')
        
        table_window_match = db_vals.merge(nearby_snps.reset_index(), left_on= 'SNP', 
                                                         right_on='NearbySNP', how = 'inner', suffixes = ('_phewasdb', '_QTL'))
        table_window_match = table_window_match.query(f'project != "{self.project_name}" or trait_phewasdb != trait_QTL ')
        self.phewas_window_r2 = f'{self.path}results/phewas/table_window_match.csv'
        
        if table_window_match.shape[0] == 0:
            print('No QTL window matches')
            pd.DataFrame().to_csv(self.phewas_window_r2, index = False)
            return -1
            
        out = table_window_match.groupby([ 'SNP_QTL','project', 'trait_phewasdb'])\
                                .apply(lambda df : df[df.uploadeddate == df.uploadeddate.max()]\
                                                   .nsmallest(n = nreturn, columns = 'p_phewasdb'))\
                                .reset_index(drop = True)\
                                .assign(phewas_r2_thresh = r2_threshold, phewas_p_threshold = pval_threshold )
        if annotate:
            out = self.annotate(out.rename({'A1_phewasdb':'A1', 'A2_phewasdb': 'A2',
                                            'Chr_phewasdb':'Chr', 'bp_phewasdb':'bp'}, axis = 1), \
                                annotate_genome, 'NearbySNP', save = False)
        out.to_csv(self.phewas_window_r2, index = False)
        
        
        ##### make prettier tables
        phewas_info =   pd.read_csv(self.phewas_exact_match_path).drop('QTL', axis = 1).reset_index()
        phewas_info = phewas_info.loc[:, ~phewas_info.columns.str.contains('Chr|A\d|bp_|b_|se_|Nearby|research|filename')]\
                                  .rename(lambda x: x.replace('_phewasdb', '_PheDb'), axis = 1)\
                                  .drop(['genotypes_file'], axis = 1)
        phewas_info['p_PheDb'] = -np.log10(phewas_info.p_PheDb)
        aa = phewas_info[ ['SNP', 'trait_QTL','project','trait_PheDb', 
                      'trait_description_PheDb' ,'Freq_PheDb', 
                       'p_PheDb','round_version' ,'uploadeddate']].drop_duplicates()
        aa.to_csv(f'{self.path}results/phewas/pretty_table_exact_match.tsv', index = False, sep = '\t')
        aa['SNP_QTL'], aa['SNP_PheDb'] = aa.SNP, aa.SNP
        aa = aa.drop('SNP', axis = 1)
        
        phewas_info =   pd.read_csv(self.phewas_window_r2).drop('QTL', axis = 1).reset_index()
        phewas_info = phewas_info.loc[:, ~phewas_info.columns.str.contains('Chr|A\d|bp_|b_|se_|Nearby|research|filename')]\
                                  .rename(lambda x: x.replace('_phewasdb', '_PheDb'), axis = 1)\
                                  .drop(['genotypes_file'], axis = 1)
        phewas_info['p_PheDb'] = -np.log10(phewas_info.p_PheDb)
        bb = phewas_info[ ["R2", 'DP','SNP_QTL', 'trait_QTL','project','SNP_PheDb', 
                      'trait_PheDb', 'trait_description_PheDb' ,'Freq_PheDb', 
                       'p_PheDb','round_version' ,'uploadeddate']].drop_duplicates()
        bb.to_csv(f'{self.path}results/phewas/pretty_table_window_match.tsv', index = False, sep = '\t')
        
        oo = pd.concat([aa, bb]).fillna('Exact match SNP').to_csv(f'{self.path}results/phewas/pretty_table_both_match.tsv', index = False, sep = '\t')
        
        return phewas_info
        
    def eQTL(self, qtltable: pd.DataFrame(), pval_thresh: float = 1e-4, r2_thresh: float = .6, nreturn: int =1, ld_window: int = 3e6,\
            tissue_list: list = ['BLA','Brain','Eye','IL','LHb','NAcc','NAcc2','OFC','PL','PL2'],\
            annotate = True, genome = 'rn7', **kwards) -> pd.DataFrame():
        
        '''
        #'Adipose','Liver'
        This function performs eQTL analysis on a given QTL table by iterating over a list of tissues
        and searching for cis-acting eQTLs (expression quantitative trait loci) that are in linkage disequilibrium
        (LD) with the QTLs.
        
        Parameters 
        ----------
        
        qtltable: pd.DataFrame()
            the QTL table to perform eQTL analysis on
        pval_thresh: float = 1e-4
            the threshold for the p-value of the eQTLs (default is 1e-4)
        r2_thresh: float = .8
            the threshold for the r-squared value of the eQTLs (default is 0.8)
        nreturn: 
            the number of eQTLs to return for each QTL (default is 1)
        ld_window:
            the size of the window around the QTL to search for eQTLs (default is 3e6)
        tissue_list: 
            a list of tissues to perform eQTL analysis on (default is a list of specific tissues)
        annotate: 
            a boolean indicating whether to annotate the resulting eQTL table (default is True)
        annotate_genome: 
            the genome to use for annotation, if annotate is True (default is 'rn6')
            
        Design
        ------
        The function first iterates over the list of tissues and performs the following steps for each tissue:

            1.Reads in a eQTL data from a remote file
            2.get the nearby snp that are in ld with the QTL using plink
            3.merge the eQTL data with the nearby snp
            4.filter the resulting dataframe using the provided pval_threshold, R2_threshold
            5.return the nreturn smallest pval
            6.concatenate all the above results to form one dataframe
            7.if annotate is True, it annotates the eQTL table using the annotate method and the provided annotate_genome
            8.save the final eQTL table to a file and return the final eQTL table.
        '''
        print(f'starting eqtl ... {self.project_name}') 
        #d =  {'rn6': '', 'rn7': '.rn7.2'}[genome]
        mygene_species = {'rn6': 'rat', 'rn7': 'rat', 'm38': 'mouse', 'cfw': 'mouse'}[genome]
        if qtltable.shape == (0,0): qtltable = pd.read_csv(self.annotatedtablepath).set_index('SNP')
        out = []
        for tissue in tqdm(tissue_list,  position=0, desc="tissue", leave=True):

            tempdf = pd.read_csv(f'https://ratgtex.org/data/eqtl/{tissue}.{genome}.cis_qtl_signif.txt.gz', sep = '\t').assign(tissue = tissue)\
                                                                                                             .rename({'variant_id': 'SNP'}, axis = 1)
            tempdf['SNP'] =tempdf.SNP.str.replace('chr', '')
            out += [pd.concat([ 
                   self.plink(bfile = self.genotypes_subset, chr = row.Chr,ld_snp = row.name,r2 = 'dprime',\
                   ld_window = ld_window, thread_num = 12, nonfounders = '')\
                  .drop(['CHR_A', 'BP_A', 'CHR_B'], axis = 1)\
                  .rename({'SNP_A': 'SNP', 'SNP_B': 'NearbySNP', 'BP_B': 'NearbyBP'}, axis = 1)\
                  .assign(**row.to_dict())\
                  .merge(tempdf, right_on= 'SNP',  left_on='NearbySNP', how = 'inner', suffixes = ('_QTL', '_eqtldb'))\
                  .query(f'R2 > {r2_thresh} and pval_nominal < {pval_thresh}')\
                  .nsmallest(nreturn, 'pval_nominal')
                  for  _, row in qtltable.iterrows() ])]

        out = pd.concat(out).reset_index(drop=True)
        if annotate:
            out = self.annotate(out, genome, 'NearbySNP', save = False)
        self.eqtl_path = f'{self.path}results/eqtl/eqtl.csv'
        out['presence_samples'] = out.ma_samples.astype(str) + '/'+ out.ma_count.astype(str)
        out.to_csv(self.eqtl_path, index= False)
        
        #### make pretty tables
        eqtl_info = pd.read_csv(self.eqtl_path).rename({'p':'-log10(P-value)'}, axis = 1)
        eqtl_info['-log10(pval_nominal)'] = -np.log10(eqtl_info['pval_nominal'])
        gene_conv_table = pd.DataFrame([(x['query'], defaultdict(lambda: '', x)['symbol']) for x \
                                in  mg.querymany(eqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species=mygene_species)],
                              columns = ['gene_id','Ensembl_gene'])
        gene_conv_table= gene_conv_table.groupby('gene_id').agg(lambda df: ' | '.join(df.drop_duplicates())).reset_index()
        eqtl_info = eqtl_info.merge(gene_conv_table, how = 'left', on = 'gene_id')
        eqtl_info.Ensembl_gene = eqtl_info.Ensembl_gene.fillna('')
        #eqtl_info['Ensembl_gene'] = [defaultdict(lambda: '', x)['symbol'] for x in  mg.querymany(eqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species='rat')];
        eqtl_info = eqtl_info.loc[:,  ['trait','SNP_QTL','-log10(P-value)','R2','SNP_eqtldb','tissue', '-log10(pval_nominal)','DP' ,
                                       'Ensembl_gene','gene_id', 'slope' ,'tss_distance', 'af', 'presence_samples']].rename(lambda x: x.replace('_QTL', ''), axis = 1)
        eqtl_info.SNP = 'chr' + eqtl_info.SNP
        eqtl_info.to_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv', index = False)
        return eqtl_info
    
    
    def sQTL(self, qtltable: pd.DataFrame(), pval_thresh: float = 1e-4, r2_thresh: float = .6, nreturn: int =1, ld_window: int = 3e6, just_cis = True,
             tissue_list: list = ['BLA','Brain','Eye','IL','LHb','NAcc','NAcc2','OFC','PL','PL2'], annotate = True, genome = 'rn7', **kwards) -> pd.DataFrame():
        
        '''
        #'Adipose','Liver'
        This function performs eQTL analysis on a given QTL table by iterating over a list of tissues
        and searching for cis-acting eQTLs (expression quantitative trait loci) that are in linkage disequilibrium
        (LD) with the QTLs.
        
        Parameters 
        ----------
        
        qtltable: pd.DataFrame()
            the QTL table to perform eQTL analysis on
        pval_thresh: float = 1e-4
            the threshold for the p-value of the eQTLs (default is 1e-4)
        r2_thresh: float = .8
            the threshold for the r-squared value of the eQTLs (default is 0.8)
        nreturn: 
            the number of eQTLs to return for each QTL (default is 1)
        ld_window:
            the size of the window around the QTL to search for eQTLs (default is 3e6)
        tissue_list: 
            a list of tissues to perform eQTL analysis on (default is a list of specific tissues)
        annotate: 
            a boolean indicating whether to annotate the resulting eQTL table (default is True)
        annotate_genome: 
            the genome to use for annotation, if annotate is True (default is 'rn6')
            
        Design
        ------
        The function first iterates over the list of tissues and performs the following steps for each tissue:

            1.Reads in a eQTL data from a remote file
            2.get the nearby snp that are in ld with the QTL using plink
            3.merge the eQTL data with the nearby snp
            4.filter the resulting dataframe using the provided pval_threshold, R2_threshold
            5.return the nreturn smallest pval
            6.concatenate all the above results to form one dataframe
            7.if annotate is True, it annotates the eQTL table using the annotate method and the provided annotate_genome
            8.save the final eQTL table to a file and return the final eQTL table.
        '''
        print(f'starting spliceqtl ... {self.project_name}') 
        mygene_species = {'rn6': 'rat', 'rn7': 'rat', 'm38': 'mouse', 'cfw': 'mouse'}[genome]
        #d =  {'rn6': '', 'rn7': '.rn7.2'}[genome]
        if qtltable.shape == (0,0): qtltable = pd.read_csv(self.annotatedtablepath).set_index('SNP')
        out = []
        loop_str = [('cis','cis_qtl_signif')] if just_cis else [('cis','cis_qtl_signif'), ('trans','trans_qtl_pairs')]
        
        for tissue, (typ, prefix) in tqdm(list(itertools.product(tissue_list, loop_str)),  position=0, desc="tissue+CisTrans", leave=True):

            tempdf = pd.read_csv(f'https://ratgtex.org/data/splice/{tissue}.{genome}.splice.{prefix}.txt.gz', sep = '\t').assign(tissue = tissue)\
                                                                                                             .rename({'variant_id': 'SNP', 'pval': 'pval_nominal'}, axis = 1)  
            tempdf['SNP'] =tempdf.SNP.str.replace('chr', '')
            out += [pd.concat([ 
                   self.plink(bfile = self.genotypes_subset, chr = row.Chr,ld_snp = row.name,r2 = 'dprime',\
                   ld_window = ld_window, thread_num = 12, nonfounders = '')\
                  .drop(['CHR_A', 'BP_A', 'CHR_B'], axis = 1)\
                  .rename({'SNP_A': 'SNP', 'SNP_B': 'NearbySNP', 'BP_B': 'NearbyBP'}, axis = 1)\
                  .assign(**row.to_dict())\
                  .merge(tempdf, right_on= 'SNP',  left_on='NearbySNP', how = 'inner', suffixes = ('_QTL', '_sqtldb'))\
                  .query(f'R2 > {r2_thresh} and pval_nominal < {pval_thresh}')\
                  .nsmallest(nreturn, 'pval_nominal').assign(sQTLtype = typ)
                  for  _, row in qtltable.iterrows() ])]
            
        #tempdf = pd.read_csv(f'https://ratgtex.org/data/splice/rn7.top_assoc_splice.txt', sep = '\t').rename({'variant_id': 'SNP', 'pval': 'pval_nominal'}, axis = 1)
        #out += [pd.concat([ 
        #           self.plink(bfile = self.genotypes_subset, chr = row.Chr,ld_snp = row.name,r2 = 'dprime',\
        #           ld_window = ld_window, thread_num = 12, nonfounders = '')\
        #          .drop(['CHR_A', 'BP_A', 'CHR_B'], axis = 1)\
        #          .rename({'SNP_A': 'SNP', 'SNP_B': 'NearbySNP', 'BP_B': 'NearbyBP'}, axis = 1)\
        #          .assign(**row.to_dict())\
        #          .merge(tempdf, right_on= 'SNP',  left_on='NearbySNP', how = 'inner', suffixes = ('_QTL', '_sqtldb'))\
        #          .query(f'R2 > {r2_thresh} and pval_nominal < {pval_thresh}')\
        #          .nsmallest(nreturn, 'pval_nominal').assign(sQTLtype = 'cis')
        #          for  _, row in qtltable.iterrows() ])]
        #https://ratgtex.org/data/splice/IL.rn7.splice.cis_qtl_signif.txt.gz


        out = pd.concat(out).reset_index(drop=True)
        out['gene_id'] = out.phenotype_id.str.extract('(ENSRNOG\d+)')
        if annotate: out = self.annotate(out, genome, 'NearbySNP', save = False)
        out['presence_samples'] = out.ma_samples.astype(str) + '/'+ out.ma_count.astype(str)
        self.sqtl_path = f'{self.path}results/sqtl/sqtl_table.csv'
        out.to_csv(self.sqtl_path, index= False)
        
        #### make pretty tables
        sqtl_info = pd.read_csv(self.sqtl_path).rename({'p':'-log10(P-value)'}, axis = 1)
        sqtl_info['-log10(pval_nominal)'] = -np.log10(sqtl_info['pval_nominal'])
        gene_conv_table = pd.DataFrame([(x['query'], defaultdict(lambda: '', x)['symbol']) for x \
                                in  mg.querymany(sqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species=mygene_species)],
                              columns = ['gene_id','Ensembl_gene'])
        gene_conv_table= gene_conv_table.groupby('gene_id').agg(lambda df: ' | '.join(df.drop_duplicates())).reset_index()
        sqtl_info = sqtl_info.merge(gene_conv_table, how = 'left', on = 'gene_id')
        #sqtl_info['Ensembl_gene'] = [defaultdict(lambda: '', x)['symbol'] for x in  mg.querymany(sqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species='rat')];
        sqtl_info = sqtl_info.loc[:,  ['trait','SNP_QTL','-log10(P-value)','R2','SNP_sqtldb','tissue', '-log10(pval_nominal)','DP' ,
                                       'Ensembl_gene','gene_id', 'slope' , 'af', 'sQTLtype', 'presence_samples']].rename(lambda x: x.replace('_QTL', ''), axis = 1)
        sqtl_info.SNP = 'chr' + sqtl_info.SNP
        sqtl_info.to_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv', index = False)
        return sqtl_info
    
    
    def manhattanplot(self, traitlist: list = [], threshold: float = 5.3591, suggestive_threshold: float = 5.58, save_fmt: list = ['html', 'png'], display: bool = True):
        
        print(f'starting manhattanplot ... {self.project_name}')
        if len(traitlist) == 0: traitlist = self.traits
        for num, t in tqdm(list(enumerate(traitlist))):
            df_gwas,df_date = [], []
            #chrlist = [str(i) if i!=(self.n_autosome+1) else 'x' for i in range(1,self.n_autosome+2)]
            for opt in [f'{t}.loco.mlma'] + [f'{t}.mlma'] + [f'{t}_chrgwas{chromp2}.mlma' for chromp2 in self.chrList]:
                try: 
                    df_gwas += [pd.read_csv(f'{self.path}results/gwas/{opt}', sep = '\t')]
                    logopt = opt.replace('.loco.mlma', '.log').replace('.mlma', '.log')
                    df_date += [pd.read_csv(f'{self.path}results/gwas/{logopt}',
                                            sep = '!', header = None ).iloc[6, 0][:-1].replace('Analysis started at', '')]
                except: pass
            df_gwas = pd.concat(df_gwas)
            df_gwas['inv_prob'] = 1/np.clip(df_gwas.p, 1e-6, 1)
            df_gwas = pd.concat([df_gwas.query('p < 1e-3'),
                                    df_gwas.query('p > 1e-3').sample(500000)] ).sort_values(['Chr', 'bp']).reset_index().dropna()
            
            append_position = df_gwas.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
            df_gwas['Chromosome'] = df_gwas.apply(lambda row: row.bp + append_position[row.Chr], axis = 1)
            def mapcolor(c, thresh , p):
                if -np.log10(p)> thresh : return 'black' 
                elif int(str(c).replace('X',str(self.n_autosome+1)).replace('Y', str(self.n_autosome+2)).replace('MT', str(self.n_autosome+4)))%2 == 0: return 'steelblue'
                return 'firebrick'
            df_gwas['color']= df_gwas.apply(lambda row: mapcolor(row.Chr, threshold, row.p) ,axis =1)
            fig2 =  go.Figure(data=go.Scattergl(
                x = df_gwas['Chromosome'].values,
                y = -np.log10(df_gwas['p']),
                mode='markers', marker=dict(color=df_gwas.color,line_width=0)))
            for x in append_position.values: fig2.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray")
            fig2.add_hline(y=threshold, line_width=2,  line_color="red")
            fig2.add_hline(y=suggestive_threshold, line_width=2, line_color="blue")

            fig2.update_layout(yaxis_range=[0,max(6, -np.log10(df_gwas.p.min())+.5)], xaxis_range = df_gwas.Chromosome.agg(['min', 'max']),
                               template='simple_white',width = 1920, height = 800, showlegend=False, xaxis_title="Chromosome", yaxis_title="-log10(p)")
            #fig2.layout['title'] = 'Manhattan Plot'
            dfgwasgrouped = df_gwas.groupby('Chr')
            fig2.update_xaxes(ticktext = [self.replacenumstoXYMT(names) for names,dfs in dfgwasgrouped],
                  tickvals =(append_position + dfgwasgrouped.bp.agg('max').sort_index().cumsum())//2 )
            if 'png' in save_fmt: fig2.write_image(f"{self.path}images/manhattan/{t}.png",width = 1920, height = 800)
            if display: fig2.show(renderer = 'png',width = 1920, height = 800)
            
    
    def porcupineplot(self, qtltable: pd.DataFrame(), traitlist: list = [], threshold: float = 5.3591, run_only_qtls = True,
                      suggestive_threshold: float = 5.58, save_fmt: list = ['html', 'png'], display: bool = True, low_mem = False):

        print(f'starting porcupineplot ... {self.project_name} reading files')
        samplen = int(1e5/30) if low_mem else int(1e5) 
        rangen = range(160,180) if low_mem else range(80,90)
        if len(traitlist) == 0: 
            if run_only_qtls: traitlist = list(qtltable.trait.unique())
            else: traitlist = self.traits
        qtl = qtltable.query('QTL==True')
        df_gwas,df_date = [], []
        #chrlist = [str(i) if i!=self.n_autosome+1 else 'x' for i in range(1,self.n_autosome+2)]
        for t in tqdm(traitlist):
            for opt in [f'regressedlr_{t.replace("regressedlr_", "")}.loco.mlma',  f'regressedlr_{t.replace("regressedlr_", "")}.mlma']+ \
            [f'regressedlr_{t.replace("regressedlr_", "")}_chrgwas{chromp2}.mlma' for chromp2 in self.chrList]:
                try: 
                    g = pd.read_csv(f'{self.path}results/gwas/{opt}', sep = '\t', dtype = {'Chr': int, 'bp': int}).assign(trait = t)
                    g['inv_prob'] = 1/np.clip(g.p, 1e-6, 1)
                    g = pd.concat([g.query('p < 0.001'), g.query('p > 0.001').sample(samplen, weights='inv_prob'),
                                   g[::np.random.choice(rangen)]] ).sort_values(['Chr', 'bp']).reset_index(drop = True).dropna()
                    df_gwas += [g]
                except: pass
        df_gwas = pd.concat(df_gwas).sort_values(['Chr', 'bp']).reset_index(drop = True)

        append_position = df_gwas.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
        df_gwas['Chromosome'] = df_gwas.apply(lambda row: row.bp + append_position[row.Chr], axis = 1)
        cmap = sns.color_palette("tab10", len(traitlist))
        d = {t: cmap[v] for v,t in enumerate(traitlist)}
        def mapcolor(c, thresh , p, trait):
            if -np.log10(p)> thresh : return d[trait] 
            elif int(str(c).replace('X',str(self.n_autosome+1)).replace('Y', str(self.n_autosome+2)).replace('MT', str(self.n_autosome+4)))%2 == 0: return 'black'
            return 'gray'
        print(f'starting porcupineplot ... {self.project_name} colorcoding')
        df_gwas['color']= df_gwas.progress_apply(lambda row: mapcolor(row.Chr, threshold, row.p, row.trait) ,axis =1)
        df_gwas['annotate'] = (df_gwas.SNP+ df_gwas.trait.str.replace('regressedlr_', '') ).isin(qtl.reset_index().SNP+qtl.reset_index().trait.str.replace('regressedlr_', ''))
        fig2 =  go.Figure(data=go.Scattergl(
            x = df_gwas['Chromosome'].values,
            y = -np.log10(df_gwas['p']),
            mode='markers', marker=dict(color=df_gwas.color,line_width=0)))
        for x in append_position.values: fig2.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray")
        fig2.add_hline(y=threshold, line_width=2,  line_color="red")
        fig2.add_hline(y=suggestive_threshold, line_width=2, line_color="blue")

        fig2.update_layout(yaxis_range=[0,max(6, -np.log10(df_gwas.p.min())+.5)], xaxis_range = df_gwas.Chromosome.agg(['min', 'max']),
                           template='simple_white',width = 1920, height = 800, showlegend=False, xaxis_title="Chromosome", yaxis_title="-log10(p)")
        df_gwas.trait = df_gwas.trait.str.replace('regressedlr_', '')
        df_gwas.query('annotate == True').apply(lambda x: fig2.add_annotation(x=x.Chromosome, y=-np.log10(x.p),text=f"{x.trait}",showarrow=True,arrowhead=2), axis = 1)
        #fig2.layout['title'] = 'Porcupine Plot'
        
        dfgwasgrouped = df_gwas.groupby('Chr')
        fig2.update_xaxes(ticktext = [self.replacenumstoXYMT(names) for names,dfs in dfgwasgrouped],
                  tickvals =(append_position + dfgwasgrouped.bp.agg('max').sort_index().cumsum())//2 )
        if 'png' in save_fmt: fig2.write_image(f"{self.path}images/porcupineplot.png",width = 1920, height = 800)
        if display: fig2.show(renderer = 'png',width = 1920, height = 800)
        #return fig2, df_gwas

    def annotate(self, qtltable: pd.DataFrame(), genome: str = 'rn7', 
                 snpcol: str = 'SNP', save: bool = True, **kwards) -> pd.DataFrame():
        
        '''
        This function annotates a QTL table with the snpEff tool, 
        which is used to query annotations for QTL and phewas results. 
                
        Parameters 
        ----------

        qtltable: pd.DataFrame
            the QTL table to be annotated
        genome: str = 'rn7'
            the genome to use for annotation (default is 'rn7')
        snpcol: str = 'SNP'
            the column name for the SNP column in the QTL table (default is 'SNP')
        save: bool = True
            a boolean indicating whether to save the annotated table to a file (default is True)
            
        Design
        ------

        The function first defines a dictionary that maps the genome input to the corresponding genome version to use in snpEff.
        It then creates a temporary VCF file from the QTL table by resetting the index, selecting specific columns, 
        and assigning certain values to other columns. The VCF file is then passed through the snpEff tool
        and the results are parsed into a new DataFrame.  If save is True, the final annotated table is saved to a file. 
        The function returns the final annotated table.
        '''
        if qtltable.shape == (0,0): qtltable = pd.read_csv(self.allqtlspath).set_index('SNP')
        d = {'rn6': 'Rnor_6.0.99', 'rn7':'mRatBN7.2.105', 'cfw': 'GRCm38.99','m38': 'GRCm38.99'}[genome]
        #bash('java -jar snpEff/snpEff.jar download -v Rnor_6.0.99')
        #bash('java -jar snpEff/snpEff.jar download -v mRatBN7.2.105')
        #bash('java -jar snpEff/snpEff.jar download -v GRCm39.105') 
        #bash('java -jar snpEff/snpEff.jar download -v GRCm38.99') 
        
        temp  = qtltable.reset_index()\
                        .loc[:,[ 'Chr', 'bp', snpcol, 'A1', 'A2']]\
                        .assign(QUAL = 40, FILTER = 'PASS' ,INFO = '', FORMAT = 'GT:GQ:DP:HQ')
        temp.columns = ["##CHROM","POS","ID","REF","ALT", 'QUAL', 'FILTER', 'INFO', 'FORMAT']
        temp['##CHROM'] = 'chr'+ temp['##CHROM'].astype(str)
        vcf_manipulation.pandas2vcf(temp, f'{self.path}temp/test.vcf', metadata='')
        a = bash(f'java -Xmx8g -jar {self.snpeff_path}snpEff.jar {d} -no-intergenic -no-intron -noStats {self.path}temp/test.vcf', print_call = False )# 'snpefftest',
        res =pd.read_csv(StringIO('\n'.join(a)),  comment='#',  delim_whitespace=True,  header=None, names = temp.columns,  dtype=str).query('INFO != "skipping"')  
        ann = res['INFO'].str.replace('ANN=', '').str.split('|',expand=True)
        column_dictionary = defaultdict(lambda: 'UNK', {k:v for k,v in enumerate(['alt_temp', 'annotation', 'putative_impact', 'gene', 'geneid', 'featuretype', 'featureid', 'transcriptbiotype',
                          'rank', 'HGVS.c', 'HGVS.p', 'cDNA_position|cDNA_len', 'CDS_position|CDS_len', 'Protein_position|Protein_len',
                          'distancetofeature', 'errors'])})
        ann = ann.rename(column_dictionary, axis = 1)
        ann.index = qtltable.index
        out = pd.concat([qtltable, ann], axis = 1).replace('', np.nan).dropna(how = 'all', axis = 1).drop('alt_temp', axis = 1, errors ='ignore')
        if save:
            self.annotatedtablepath = f'{self.path}results/qtls/finalqtl.csv'
            out.reset_index().to_csv(self.annotatedtablepath, index= False) 
            out.reset_index().to_csv(f'{self.path}results/qtls/finalqtl.tsv', index= False, sep = '\t')
        
        return out 
    
    def report(self, round_version: str = '10', threshold: float = 5.3591, suggestive_threshold: float = 5.58, reportpath: str = 'gwas_report_auto.rmd',
               covariate_explained_var_threshold: float = 0.02, gwas_version='current'):
        pqfile = pd.read_parquet(self.phewas_db)
        pqfile.value_counts(['project','trait']).reset_index().rename({0:'number of SNPS below 1e-4 Pvalue'}, axis = 1).to_csv(f'{self.path}temp/phewas_t_temp.csv', index = False)
        PROJECTLIST = '* '+' \n* '.join(pqfile.project.unique())
        REGRESSTHRS = str(round(covariate_explained_var_threshold*100, 2)) + '%'
        with open(reportpath, 'r') as f: 
            report_txt = f.read()
        with open(f'{self.path}genotypes/parameter_thresholds.txt', 'r') as f: 
            out = f.read()
            params = {x:re.findall(f"--{x} ([^\n]+)", out)[0] for x in ['geno', 'maf', 'hwe']}
        with open(f'{self.path}genotypes/genotypes.log') as f:
            out = f.read()
            params['snpsb4'] = re.findall(f"(\d+) variants loaded from .bim file.", out)[0]
            params['snpsafter'], params['nrats'] = re.findall("(\d+) variants and (\d+) samples pass filters and QC.", out)[0]
            params['removed_geno'], params['removedhwe'], params['removedmaf'] = \
                   (~pd.read_parquet(f'{self.path}genotypes/snpquality.parquet.gz')[['PASS_MISS','PASS_MAF','PASS_HWE']])\
                   .sum().astype(str)


        for i,j in [('PROJECTNAME', self.project_name),('PATHNAME', self.path),('ROUND', round_version), ('NSNPSB4', params['snpsb4'] ), ('NSNPS', params['snpsafter'])
                    , ('GENODROP', params['removed_geno']), ('MAFDROP', params['removedmaf']), ('HWEDROP', params['removedhwe']),
                    ('GENO', params['geno']), ('MAF', params['maf']), ('HWE', params['hwe']), ('PROJECTLIST', PROJECTLIST), ('REGRESSTHRS', REGRESSTHRS ),
                    ('THRESHOLD10',str(round(threshold, 2))),('THRESHOLD',str(round(suggestive_threshold, 2))), ('NSAMPLES',params['nrats']),]:
            report_txt = report_txt.replace(i,j)
        with open(f'{self.path}results/gwas_report.rmd', 'w') as f: f.write(report_txt)
        try:bash(f'rm -r {self.path}results/gwas_report_cache')
        except:pass
        os.system(f'''conda run -n r-environment Rscript -e "rmarkdown::render('{self.path}results/gwas_report.rmd')" | grep -oP 'Output created: gwas_report.html' '''); #> /dev/null 2>&1 r-environment
        bash(f'''cp {self.path}results/gwas_report.html {self.path}results/gwas_report_{self.project_name}_round{round_version}_threshold{threshold}_n{self.df.shape[0]}_date{datetime.today().strftime('%Y-%m-%d')}_gwasversion_{gwas_version}.html''')
        #print('Output created: gwas_report.html') if 'Output created: gwas_report.html' in ''.join(repout) else print(''.join(repout))
        try:bash(f'rm -r {self.path}results/gwas_report_cache')
        except:pass
    
    def store(self, researcher: str , round_version: str, gwas_version: str, remove_folders: bool = True):
        '''
        zip project remove folders if remove folders is True
        '''
        all_folders = ' '.join([self.path + x for x in ['data', 'genotypes', 'grm', 'log', 'results', 'environment.yml', 'package_versions.txt']])
        info = '_'.join([self.project_name, researcher,round_version,gwas_version])
        bash(f'zip -r {self.path}run_{info}.zip {all_folders}')
        if remove_folders: 
            bash(f'rm -r {all_folders}')
            extra_folders = ' '.join([self.path + x for x in ['logerr', 'temp']])
            bash(f'rm -r {extra_folders}')
            
    def copy_results(self, destination: str = '/projects/ps-palmer/s3/data/tsanches_dash_genotypes/gwas_results', make_public = True):
        '''
        zip project remove folders if remove folders is True
        '''
        
        #destination += '/'+ self.project_name
        #print(f'copying {self.project_name} to {destination}')
        os.makedirs(f'{destination}', exist_ok = True)
        bash(f'cp -r {self.path} {destination}')
        if make_public:
            bash('/projects/ps-palmer/tsanches/mc anonymous set public myminio/tsanches_dash_genotypes --recursive')


    def print_watermark(self):
        'function to create a conda env watermark for the project and saves it to the project folder'
        with open(f'{self.path}environment.yml', 'w') as f:
            f.write('\n'.join(bash('conda env export --from-history')[1:-1]))
        with open(f'{self.path}package_versions.txt', 'w') as f:
            f.write('\n'.join(bash('conda list')[1:]))
            
    def qsub_until_phewas(self, queue = 'condo', walltime = 8, ppn = 12, out = 'log/', err = 'logerr/', project_dir = ''):
        qsub( queue = queue, walltime = walltime, ppn = ppn, out = 'log/', err = 'logerr/', call = 'gwas_cli.py')
        
    def qsub_phewas(self, queue = 'condo', walltime = 8, ppn = 12, out = 'log/', err = 'logerr/', project_dir = ''):
        qsub( queue = queue, walltime = walltime, ppn = ppn, out = 'log/', err = 'logerr/', call = 'phewas_cli.py')
        

def get_trait_descriptions_f(data_dic, traits):
    out = []
    for trait in traits:
        try: out +=  [data_dic[data_dic.measure == trait.replace('regressedlr_', '')].description.iloc[0]]
        except: out +=  ['UNK']
    return out
        