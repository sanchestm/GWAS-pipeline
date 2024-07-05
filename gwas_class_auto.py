#logging.basicConfig(filename=f'gwasRun.log', filemode='w', level=logging.DEBUG)
from IPython.display import display
from IPython.utils import io
from bokeh.resources import INLINE
from collections import Counter, defaultdict, namedtuple
from dask.diagnostics import ProgressBar 
from dask.distributed import Client, client, progress, wait
from datetime import datetime
from fancyimpute import SoftImpute, BiScaler, IterativeSVD
from functools import reduce, wraps
from glob import glob
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_ncbi_associations
from hdbscan import HDBSCAN
from holoviews.operation.datashader import datashade, bundle_graph, rasterize, shade, dynspread, spread
from holoviews.operation.resample import ResampleOperation2D
from holoviews.operation import decimate
from io import StringIO
from interactiveqc import interactive_QC
from matplotlib.colors import PowerNorm
from os.path import dirname, basename
from pathlib import Path
from pdf2image import convert_from_path
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.utilities import regressor_coefficients 
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list, linkage
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import KNNImputer,SimpleImputer, IterativeImputer, MissingIndicator
from sklearn.linear_model import LinearRegression#, RobustRegression
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
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
import datashader as ds
import gc
import goatools
import gzip
import holoviews as hv
import hvplot.pandas
import inspect
import itertools
import json
import logging
import matplotlib.pyplot as plt
import mygene
import networkx as nx
import numba
import numpy as np
from holoviews import opts
import os
import pandas as pd
import pandas_plink
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as plotio
import prophet
import psycopg2
import re
import requests
import seaborn as sns
import statsReport
import subprocess
import sys
import umap
import utils
import warnings
from time import time
# gene2go = download_ncbi_associations()
# geneid2gos_rat= Gene2GoReader(gene2go, taxids=[10116])

mg = mygene.MyGeneInfo()
def query_gene(genelis, species):
    mg = mygene.MyGeneInfo()
    species = translate_dict(species, {'rn7': 'rat', 'rn8':'rat', 'm38':'mouse', 'rn6': 'rat'})
    a = mg.querymany(genelis , scopes='all', fields='all', species=species, verbose = False, silent = True)
    res = pd.concat(pd.DataFrame({k:[v]  for k,v in x.items()}) for x in a)
    res = res.assign(**{k:np.nan for k in (set(['AllianceGenome','symbol', 'ensembl', 'notfound']) - set(res.columns))} )
    return res[res.notfound.isna()].set_index('query')
tqdm.pandas()
sys.setrecursionlimit(10000)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pn.extension()

def time_eval(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap
    
#warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#pd.options.plotting.backend = 'holoviews'
na_values_4_pandas = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null', 'UNK']

def annotate_vep(snpdf, species, snpcol = 'SNP', refcol = 'A2', altcol = 'A1', refseq = 1, expand_columns = True, intergenic = False):
    import requests, sys
    server = "https://rest.ensembl.org"
    ext = f"/vep/{species}/hgvs"
    res =  '"' + snpdf[snpcol].str.replace(':', ':g.').str.replace('chr', '') + snpdf[refcol] + '>'+snpdf[altcol] + '"' 
    res = f"[{','.join(res)}]"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    call = '{ "hgvs_notations" : ' + res + f', "refseq":{refseq}, "OpenTargets":1, "AlphaMissense":1,"Phenotypes":1, "Enformer":1,"LoF":1'+' }'
    print(call) # , 
    r = requests.post(server+ext, headers=headers, data=call) #
    if not r.ok:
      print('VEP annotation failed')#r.raise_for_status()
      return snpdf
    decoded = pd.json_normalize( r.json())
    #print(repr( r.json()))
    decoded['SNP'] = decoded.input.map(lambda x: x.replace(':g.', ':')[:-3])
    if (not intergenic) and ('intergenic_consequences' in decoded.columns):
        decoded = decoded.drop('intergenic_consequences', axis = 1)
    if 'colocated_variants' in decoded.columns: 
        decoded['colocated_variants'] = decoded['colocated_variants'].map(lambda x: pd.json_normalize(x) if isinstance(x, list) else pd.DataFrame())
        decoded.colocated_variants =  decoded.colocated_variants.map(lambda x: '|'.join(x['id']) if 'id' in x.columns else x)
    jsoncols = list(set(['transcript_consequences','intergenic_consequences']) & set(decoded.columns))
    if len(jsoncols):
        decoded[jsoncols] = decoded[jsoncols].applymap(lambda x: pd.json_normalize(x) if isinstance(x, list) else pd.DataFrame())
        if expand_columns:
            for i in jsoncols:
                tempdf = pd.concat(decoded.apply(lambda x: x[i].rename(lambda y: f'{i}_{y}', axis = 1).assign(SNP = x.SNP), axis = 1).to_list())
                tempdf = tempdf.applymap(lambda x: x[0] if isinstance(x, list) else x)
                tempdf.loc[:, tempdf.columns.str.contains('phenotypes')] = \
                     tempdf.loc[:, tempdf.columns.str.contains('phenotypes')].applymap(lambda x: x['phenotype'].replace(' ', '_') if isinstance(x, dict) else x)
                tempdf = tempdf.drop_duplicates(subset = tempdf.loc[:, ~tempdf.columns.str.contains('phenotypes')].filter(regex =f'{i}_') \
                                              .drop(['transcript_consequences_transcript_id', 'transcript_consequences_cdna_start', 'transcript_consequences_cdna_end'
                                                     'transcript_consequences_consequence_terms', 'transcript_consequences_strand',
                                                    'transcript_consequences_distance', 'transcript_consequences_variant_allele'], errors='ignore', axis =1).columns.to_list())
                decoded = decoded.merge(tempdf, how = 'left', on = 'SNP')
            decoded = decoded.drop(jsoncols, axis = 1)
            decoded = decoded.rename({'transcript_consequences_gene_symbol':'gene', 'transcript_consequences_gene_id': 'geneid',
                                     'transcript_consequences_biotype': 'transcriptbiotype', 'transcript_consequences_impact':'putative_impact'}, axis = 1)
            if 'putative_impact' in decoded.columns: decoded.putative_impact = decoded.putative_impact.fillna('MODIFIER')
            decoded = decoded.rename(lambda x: x.replace('consequences_',''), axis = 1)

    return snpdf.merge(decoded.drop(['id', 'seq_region_name', 'end', 'strand', 'allele_string', 'start'], axis = 1).rename({'input': ''}), on = 'SNP', how = 'left')


def printwithlog(string):
    print(string)
    logging.info(string)
    
def merge_duplicates_fancy(df):
    if not len(df): return
    if len(df) == 1: df.iloc[0]
    return pd.Series({y:'|'.join(df[y].astype(str).unique()) for y in df.columns})

def groupby_no_loss_merge(df, groupcol):
    return allwfus.groupby(col).progress_apply(merge_duplicates_fancy).reset_index(drop = True)

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

def generate_datadic(rawdata, trait_prefix = ['pr', 'sha', 'lga', 'shock'], main_cov = 'sex,cohort,weight_surgery,box,coatcolor', project_name = '', save = False):
    dd_new = rawdata.describe(include = 'all').T.reset_index(names= ['measure'])
    dd_new['trait_covariate'] = dd_new['unique'].apply(lambda x: 'covariate_categorical' if not np.isnan(x) else 'covariate_continuous')
    dd_new.loc[dd_new.measure.str.contains('^'+'|^'.join(trait_prefix), regex = True), ['trait_covariate', 'covariates']] = ['trait', main_cov]
    dd_new.loc[dd_new.measure.str.endswith('_age') , ['trait_covariate', 'covariates']] = ['covariate_continuous', '']
    dd_new['description'] = dd_new['measure']
    for pref in trait_prefix:
        addcov = ','.join(dd_new.loc[dd_new.measure.str.endswith('_age') &  dd_new.measure.str.startswith(pref)].description)
        if len(addcov): 
            dd_new.loc[dd_new.measure.str.startswith(pref), 'covariates'] = dd_new.loc[dd_new.measure.str.startswith(pref), 'covariates'] + ',' + addcov
    dd_new.loc[dd_new.measure.isin(['rfid', 'labanimalid']), 'trait_covariate'] = 'metadata'
    dd_new = dd_new.loc[~dd_new.measure.str.startswith('Unnamed: ')]
    def remove_outofboundages(s, trait):
       try:
           min, max = pd.Series(list(map(int , re.findall('\d{2}', trait)))).agg(['min', 'max'])
           #print(min, max)
           aa =  "|".join( [str(x).zfill(2) for x in range(min, max+1)])
       except: aa = '-9999999'
       return ','.join([x for x in s.split(',') if re.findall(f'({aa}).*(age)', x) or ('age' not in x)])
    dd_new.loc[dd_new.trait_covariate == 'trait', 'covariates'] = dd_new.loc[dd_new.trait_covariate == 'trait']\
          .apply(lambda row: remove_outofboundages(row.covariates,row.measure), axis = 1)
    if save:
        dd_new.to_csv(f'data_dict_{project_name}.csv')
    return dd_new

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
            return dd.read_csv(filename,  compression='gzip', comment='#',  sep ='\s+', header=None, 
                               names = vcf_manipulation.get_vcf_header(filename),blocksize=None,  dtype=str, ).repartition(npartitions = 100000)
        # usecols=['#CHROM', 'POS']
        return pd.read_csv(filename,  compression='gzip', comment='#',  sep ='\s+',
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
        elif metadata[-4:] == '.vcf': header = self.get_vcf_metadata(metadata)

        with open(filename, 'w') as vcf: 
            vcf.write(header)
        df.to_csv(filename, sep="\t", mode='a', index=False)

def bash(call, verbose = 0, return_stdout = True, print_call = True,silent = False,shell = False):
    call = re.sub(r' +', ' ', call).strip(' ')
    if print_call: printwithlog(call)
    out = subprocess.run(call if shell else call.split(' '), capture_output = True,  shell =shell) 
    if verbose and (not return_stdout) and (not silent): printwithlog(out.stdout)
    if out.stderr and (not silent): 
        try:printwithlog(out.stderr.decode('ascii'))
        except: printwithlog(out.stderr.decode('utf-8'))
    if return_stdout: 
        try: oo =  out.stdout.decode('ascii').strip().split('\n')
        except: oo =  out.stdout.decode('utf-8').strip().split('\n')
        return oo
    return out

def vcf2plink(vcf = 'round9_1.vcf.gz', n_autosome = 20, out_path = 'zzplink_genotypes/allgenotypes_r9.1'):
    bash(f'plink --thread-num 16 --vcf {vcf} --chr-set {n_autosome} no-xy --keep_allele_order --set-hh-missing --set-missing-var-ids @:# --make-bed --out {out_path}')


def impute_single_trait(dataframe: pd.DataFrame(), imputing_col: str , covariate_cols: list, groupby: list, scaler = StandardScaler(), imputer = SoftImpute(verbose = False)):
    if type(imputing_col) == str: imputing_col = [imputing_col]
    if type(covariate_cols) == str: covariate_cols = [covariate_cols]
    if type(groupby) == str: groupby = [groupby]
    covariate_cols = list(set(covariate_cols) - set(imputing_col) - set(groupby))
    groupby = list(set(groupby) - set(imputing_col) )
    
    dfs = dataframe.copy().reset_index().set_index('rfid')[imputing_col + covariate_cols + groupby]
    dfs = dfs.loc[:, ~dfs.columns.duplicated()]
    
    numeric_cols = dfs[covariate_cols].select_dtypes(include='number').columns.unique().to_list()
    if len(numeric_cols):
        dfs.loc[:, numeric_cols] = scaler.fit_transform(dfs.loc[:, numeric_cols])
    all_covs = numeric_cols + groupby
    if (catcovdf :=  dfs[covariate_cols].select_dtypes(exclude='number')).shape[1]:
        ohe = UMAP(n_components=5).fit_transform(OneHotEncoder().fit_transform(catcovdf))
        dfs.loc[:, [f'ohe{x}' for x in range(ohe.shape[1])]] = ohe
        all_covs += [f'ohe{x}' for x in range(ohe.shape[1])]

    dfsm = dfs[imputing_col + all_covs]
    missing_pct = dataframe[imputing_col].isna().mean().mean()
    flow = pd.DataFrame(columns = ['missing_pct_added', 'imputation_error'])
    for missing_pct_i in [i for i in np.linspace(0,min(.8, missing_pct*1.3), 11)] * 5:
        mask = pd.DataFrame(np.zeros(shape = dfsm.shape).astype(bool),columns = dfsm.columns, index = dfsm.index) #
        masknp = np.random.choice([True, False], size=dfsm.loc[:, imputing_col].shape, p=[missing_pct_i,1-missing_pct_i])
        mask.loc[:, imputing_col] = masknp
        mask = mask.values
        dfsmasked = dfsm.mask(mask)
        if  dfsmasked[imputing_col].isna().values.sum() == 0:
            imp = dfsmasked[imputing_col]
        else:
            if len(groupby):
                imp = dfsmasked.reset_index().groupby(groupby).apply(lambda x: pd.DataFrame(imputer.fit_transform(x.drop(['rfid']+groupby, axis = 1)),
                                                                                           columns = x.drop(['rfid']+groupby, axis = 1).columns, 
                                                                                           index = x.rfid) if x[imputing_col].isna().values.sum() \
                                                                     else x.drop(['rfid']+groupby, axis = 1))\
                                                              .sort_index(level = -1).set_axis(dfsmasked.index)[imputing_col]
            else:
                imp = dfsmasked.copy()
                imp.loc[:,imp.select_dtypes(include='number').columns] = imputer.fit_transform(imp.select_dtypes(include='number'))#
                imp = imp[imputing_col]
        if missing_pct_i == 0:
            out = imp
        else:
            dfcorrect = dfs.loc[:, imputing_col]
            errvals = np.abs((imp.mask(~masknp) - dfcorrect)/dfcorrect).values
            imp_err = np.nanmedian(errvals) if (~np.isnan(errvals)).sum() > 0  else np.nan
            flow.loc[len(flow)] = [missing_pct_i, imp_err]
    
    flow = flow.assign(imputed_column = ','.join(imputing_col), covariate_columns = ','.join(covariate_cols), groupby_columns = ','.join(groupby), imputer = str(imputer).split('(')[0])
    return pd.Series({'imputed': out, 'qc': flow})

def regressoutgb(dataframe: pd.DataFrame(), data_dictionary: pd.DataFrame(), covariates_threshold: float = 0.02, groupby = ['sex'], normalize = 'quantile'):
    if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
    df, datadic = dataframe.copy(), data_dictionary
    def getcols(df, string): return df.columns[df.columns.str.contains(string)].to_list()
    if type(groupby) == 'str': groupby = [groupby]
    categorical_all = list(datadic.query('trait_covariate == "covariate_categorical"').measure)
    dfohe = df.copy()
    ohe = OneHotEncoder()
    oheencoded = ohe.fit_transform(dfohe[categorical_all].astype(str)).todense()
    dfohe[[f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_all)]] = oheencoded
    alltraits = list(datadic.query('trait_covariate == "trait"').measure.unique())
    def getdatadic_covars(trait):
        covars = set(datadic.set_index('measure').loc[trait, 'covariates'].split(','))
        covars =  covars - set(groupby)
        covars = covars | set(itertools.chain.from_iterable([ getcols(dfohe, F'OHE_{x}') for x in covars]))
        covars = covars - set(datadic.query('trait_covariate == "covariate_categorical"').measure)
        return list(covars)
    continuousall = list(set(datadic.query('trait_covariate == "covariate_continuous"').measure) & set(df.columns))
    explained_vars = []
    reglist = []
    if not len(groupby): dfohe, groupby = dfohe.assign(tempgrouper = 'A'), ['tempgrouper']
    for group, gdf in tqdm(dfohe.groupby(groupby)):
        groupaddon = '|'.join(group)
        if continuousall:
            gdf.loc[:, continuousall] = QuantileTransformer(n_quantiles = min(100, gdf.shape[0]), output_distribution='normal').fit_transform(gdf.loc[:, continuousall])
        for trait in alltraits:
            expvars = statsReport.stat_check(gdf)\
                                .explained_variance([trait],getdatadic_covars(trait))\
                                .dropna(how = 'any')
            expvars = expvars[expvars > covariates_threshold].dropna()
            explained_vars += [expvars.rename(lambda x: f"{x}_{groupaddon}", axis = 1).reset_index(names = 'group').melt(id_vars='group')]
            gdf[list(expvars.index)] = gdf[list(expvars.index)].fillna(gdf[list(expvars.index)].mean())
            if not list(expvars.index):
                reg = gdf.set_index('rfid')[list(expvars.columns)].rename(lambda x: 'regressedlr_'+x, axis =1)
            else:
                reg = statsReport.regress_out(gdf.set_index('rfid'), list(expvars.columns),  list(expvars.index)).rename(lambda x: x.lower(), axis = 1)
            if normalize == 'quantile': reg = statsReport.quantileTrasformEdited(reg, reg.columns)
            if normalize == 'boxcox':
                from sklearn.preprocessing import PowerTransformer
                reg.loc[:, reg.columns] = PowerTransformer().fit_transform(reg.loc[:, reg.columns])
            reglist += [reg.reset_index().melt(id_vars='rfid')]
    melted_explained_vars = pd.concat(explained_vars).reset_index(drop = True)[['variable', 'group', 'value']]
    regresseddf = pd.concat(reglist).pivot(columns= 'variable', index = 'rfid').droplevel(0, axis = 1)
    if normalize == 'quantile': 
        regresseddf = statsReport.quantileTrasformEdited(regresseddf, regresseddf.columns)
    if normalize == 'boxcox':
        from sklearn.preprocessing import PowerTransformer
        regresseddf.loc[:, regresseddf.columns] = PowerTransformer().fit_transform(regresseddf.loc[:, regresseddf.columns])
    #if normalize in ['false', 'False', False, 'passthrough']: pass
    outdf = pd.concat([df.set_index('rfid'), regresseddf], axis = 1)
    outdf = outdf.loc[:,~outdf.columns.duplicated()]
    passthrough_traits = list(datadic.query('trait_covariate == "trait" and covariates == "passthrough"').measure.unique())
    if len(passthrough_traits):
        outdf[['regressedlr_' + x for x in passthrough_traits]] = outdf[passthrough_traits]
    return {'regressed_dataframe': outdf.reset_index().sort_values('rfid'), 
            'covariatesr2': melted_explained_vars,
            'covariatesr2pivoted': pd.DataFrame(melted_explained_vars.groupby('variable')['group'].apply(list)).reset_index()}
    
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
    
    if df.shape[0] < 10:
        printwithlog(f' could not run the trait {y_column}, take a look at the input data, looks like too few samples')
        display(dforiginal[[y_column]])
        printwithlog('we will set this values to NAN')
        return dforiginal.copy().set_index('rfid').assign(**{f'regressedlr_{y_column}': np.nan})[[f'regressedlr_{y_column}']]
        
    
    ### onehot_encoding
    if len(categorical_regressors):
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
        if extra_label: 
            if type(extra_label) != str: extra_label = '|'.join(extra_label)
            temp2.variable = temp2.variable +'_'+extra_label 
        pd.concat([prev,temp2]).to_csv(f'{path}melted_explained_variances.csv', index = False)
    
    #### fit model 
    fbmodel = prophet.Prophet(growth = growth,seasonality_mode ='additive', seasonality_prior_scale =10.,
              n_changepoints= 25, changepoint_prior_scale=10 , changepoint_range = .8,
              **season_true_kwards, **season_false_kwards) #mcmc_samples=100
    for i in approved_cols: fbmodel.add_regressor(i)
    try: fbmodel.fit(df)
    except: 
        printwithlog(f' could not run the trait {y_column}, take a look at the input data')
        display(dforiginal[[y_column]])
        printwithlog('we will set this values to NAN')
        return dforiginal.copy().set_index('rfid').assign(**{f'regressedlr_{y_column}': np.nan})[[f'regressedlr_{y_column}']]
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
    passthrough_traits = list(ddtraits.query('covariates == "passthrough"').measure.unique())
    if len(passthrough_traits):
        outdf[['regressedlr_' + x for x in passthrough_traits]] = outdf[passthrough_traits]
    
    if save_explained_vars: 
        outdf.to_csv(f'{path}processed_data_ready.csv', index = False)
        outdf.to_csv(f'{path}results/processed_data_ready_n{outdf.shape[0]}_date{datetime.today().strftime("%Y-%m-%d")}.csv', index = False)
    return outdf

def plotly_read_from_html(file):
    with open(file, 'r') as f: html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html[-2**16:])[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}    
    return plotio.from_json(json.dumps(plotly_json))
    
def fancy_display(df, download_name = 'default.csv'):
    pn.extension('tabulator')
    df = df.drop(['index'] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
    numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
    df[numeric_cols] = df[numeric_cols].applymap(round, ndigits=3)
    d = {x : {'type': 'number', 'func': '>=', 'placeholder': 'Enter minimum'} for x in numeric_cols} | \
        {x : {'type': 'input', 'func': 'like', 'placeholder': 'Similarity'} for x in df.columns[~df.columns.isin(numeric_cols)]}
    download_table = pn.widgets.Tabulator(df,pagination='local' ,page_size= 15, header_filters=d, layout = 'fit_data_fill')
    filename, button = download_table.download_menu(text_kwargs={'name': 'Enter filename', 'value': download_name},button_kwargs={'name': 'Download table'})
    return pn.Column(pn.Row(filename, button), download_table)
    
def plotly_histograms_to_percent(fig):
    for trace in fig.data:
        if type(trace) == plotly.graph_objs._histogram.Histogram:
            trace.histfunc = 'count'
            trace.histnorm = 'probability'
            trace.nbinsx = trace.nbinsy = 30
            trace.hovertemplate = trace.hovertemplate.replace('<br>count=%', '<br>percent=%')
    return fig

def sql2pandas(file):
    import sqlite3
    conn = sqlite3.connect(file)
    out = pd.DataFrame([[table,pd.read_sql_query(f"SELECT * FROM {table};", conn)] 
                        for table in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name']],
                       columns = ['name', 'df']).set_index('name')
    conn.close()
    return out

def translate_dict(s, d):
    if not d: return s
    return re.sub(f"({')|('.join(d.keys())})",lambda y: d[str(y.group(0))] ,s )

def bayes_ppi(pvals, islog10 = False):
    from scipy.special import ndtri
    if islog10: z = ndtri(np.power(10, -pvals)/2)
    else: z = ndtri(pvals/2)
    bfi = np.exp(np.power(z,2)/2)
    return bfi/(bfi.sum())

def credible_set_idx(ppi, cs_threshold = .99, return_series= False):
    if isinstance(ppi, pd.Series): ppi = ppi.values
    sorted_index = np.argsort(ppi)[::-1]
    sarray = ppi[sorted_index].cumsum() < cs_threshold
    if return_series: return pd.Series(data = sarray, index = sorted_index )
    return sorted_index[sarray]
    
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
                 traits: list = [],
                 trait_descriptions: list = [],
                 chrList: list = [], 
                 #n_autosome: int = 20,
                 all_genotypes: str = '/tscc/projects/ps-palmer/gwas/databases/rounds/round10_1',
                 founderfile: str = '/tscc/projects/ps-palmer/gwas/databases/founder_genotypes/founders7.2',
                 #gtca_path: str = '',
                 #snpeff_path: str =  'snpEff/',
                 #locuszoom_path: str = 'locuszoom/',
                 phewas_db: str = 'phewasdb.parquet.gz',
                 threshold: float = 'auto',
                 threshold05: float = 5.643286,
                 genome: str = 'rn7',
                 genome_accession: str = 'GCF_015227675.2',
                 threads: int = os.cpu_count()): 

        self.gcta = 'gcta64' #if not gtca_path else gtca_path
        self.path = path
        self.all_genotypes = all_genotypes
        #self.founder_genotypes = founder_genotypes
        #self.snpeff_path = snpeff_path
        #self.locuszoom_path = locuszoom_path
        #self.n_autosome = n_autosome
        #self.genome = genome
        self.threshold = threshold
        self.threshold05 = threshold05

        logging.basicConfig(filename=f'{self.path}gwasRun.log', 
                            filemode='w', level=logging.INFO, format='%(asctime)s %(message)s') #INFO

           
        if os.path.exists(f'{self.path}temp'): bash(f'rm -r {self.path}temp')
        try: self.pull_NCBI_genome_info(genome_accession,redownload = False)
        except: self.ask_user_genome_accession()
        if not chrList:
            self.chrList = lambda: [self.replacenumstoXYMT(i) for i in range(1,self.n_autosome+5) if i != self.n_autosome+3]
        
        if type(data) == str: 
            df = pd.read_csv(data, dtype={'rfid': str}).replace([np.inf, -np.inf], np.nan)
        else: df = data.replace([np.inf, -np.inf], np.nan)
        df.columns = df.columns.str.lower()
        if 'vcf' in self.all_genotypes:
            sample_list_inside_genotypes = vcf_manipulation.get_vcf_header(self.all_genotypes)
        else:
            sample_list_inside_genotypes = pd.read_csv(self.all_genotypes+'.fam', header = None, sep='\s+', dtype = str)[1].to_list()
        df = df.sort_values('rfid').reset_index(drop = True).dropna(subset = 'rfid').drop_duplicates(subset = ['rfid'])
        self.df = df[df.rfid.astype(str).isin(sample_list_inside_genotypes)].copy()#.sample(frac = 1)
        self.df = self.df.loc[self.df.rfid.astype(str).map(hash).sort_values().index, :].reset_index(drop = True)
        #self.df = self.df.sample(frac=1, random_state=42).reset_index(drop = True)
        
        if self.df.shape[0] != df.shape[0]:
            missing = set(df.rfid.astype(str).unique()) - set(self.df.rfid.astype(str).unique())
            pd.DataFrame(list(missing)).to_csv(f'{self.path}missing_rfid_list.txt', header = None, index = False)
            printwithlog(f"missing {len(missing)} rfids for project {project_name}, see missing_rfid_list.txt")
            
        self.traits = [x.lower() for x in traits]
        if not len(self.traits):
            try: 
                if len(temptraits := df.columns[df.columns.str.contains('regressedlr_')]):
                    self.traits = temptraits.to_list()
                else:
                    printwithlog(f'importing traits from {self.path}data_dict_{project_name}.csv')
                    tempdd = pd.read_csv(f'{self.path}data_dict_{project_name}.csv')
                    tempdd_traits = tempdd.query('trait_covariate == "trait"').measure
                    self.traits = self.df.columns[self.df.columns.isin(tempdd_traits)].to_list()
                    if len(_ :=  self.df.columns[~self.df.columns.isin(tempdd.measure)]):
                        printwithlog(f'traits in raw_data but not in data_dict: {self.path}missing_inRawDataNotInDatadic.txt')
                        _.to_frame().to_csv(f'{self.path}missing_inRawDataNotInDatadic.txt',  index = False, header = None)
                    if len(_ := tempdd.measure[~tempdd.measure.isin(self.df.columns)]):
                        printwithlog(f'traits in data_dict but not in raw_data: {self.path}missing_inDatadicNotInRawData.txt')
                        _.to_frame().to_csv(f'{self.path}missing_inDatadicNotInRawData.txt',index = False, header = None)
                
            except:
                printwithlog(f'could not open {self.path}data_dict_{project_name}.csv, traits will be set to empty list')
                self.traits = []
            
        printwithlog(self.df[self.traits].count())
        self.make_dir_structure()
        for trait in self.traits:
            trait_file = f'{self.path}data/pheno/{trait}.txt'            
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file, index = False, sep = ' ', header = None)
        
        if not len(trait_descriptions):
            try: 
                printwithlog(f'importing trait descriptions from {self.path}data_dict_{project_name}.csv')
                trait_descriptions = get_trait_descriptions_f(pd.read_csv(f'{self.path}data_dict_{project_name}.csv'), traits)
            except:
                printwithlog(f'could not open {self.path}data_dict_{project_name}.csv, descriptions will be filled with "UNK"')
                trait_descriptions = ['UNK']*len(traits)
        
        self.get_trait_descriptions = defaultdict(lambda: 'UNK', {k:v for k,v in zip(traits, trait_descriptions)})
        
        self.phewas_db = phewas_db
        self.project_name = project_name
        
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.genotypes_subset = f'{self.path}genotypes/genotypes'
        self.genotypes_subset_vcf = f'{self.path}genotypes/genotypes_subset_vcf.vcf.gz'
        if founderfile not in ['none', None, '', 0]: 
            try: self.foundersbimfambed = pandas_plink.read_plink(founderfile)
            except: 
                printwithlog(f'could not open {founderfile}')
                self.foundersbimfambed = []
        else: self.foundersbimfambed = []
        
        self.autoGRM = f'{self.path}grm/AllchrGRM'
        self.xGRM = f'{path}grm/xchrGRM'
        self.yGRM = f'{path}grm/ychrGRM'
        self.mtGRM = f'{path}grm/mtchrGRM'
        self.log = pd.DataFrame( columns = ['function', 'call', 'out'])
        self.thrflag = f'--thread-num {threads}'
        self.threadnum = int(threads)
        self.print_call = True
        
        self.failed_full_grm = False
        
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.sample_sex_path = f'{self.path}genotypes/sample_rfids_sex_info.txt'
        self.sample_sex_path_gcta = f'{self.path}genotypes/sample_rfids_sex_info_gcta.txt'
        self.heritability_path = f'{self.path}results/heritability/heritability.tsv'
        self.sample_path_males = f'{self.path}genotypes/keep_rfids_males.txt'
        self.sample_path_females = f'{self.path}genotypes/keep_rfids_females.txt'

        if self.threshold == 'auto':
            threshpath = f'{self.path}pvalthresh/PVALTHRESHOLD.csv'
            if os.path.exists(threshpath):
                printwithlog(f'found significance threshold at {self.path}pvalthresh/PVALTHRESHOLD.csv, pulling info from this file')
                self.threshold, self.threshold05 = pd.read_csv(threshpath, index_col=0).loc[['10%', '5%'], 'thresholds'].tolist()
            else:
                if os.path.exists(f'{self.path}genotypes/genotypes.bed'): self.estimate_pval_threshold()
                else: printwithlog('significance threshold not calculated yet, will be performed after subseting the genotypes')
        
    def clear_directories(self, folders: list = ['data', 'genotypes', 'grm', 'log', 'logerr', 'images/',
                                            'temp', 'results/heritability', 'results/preprocessing',
                                             'results/gwas',  'results/loco', 'results/qtls','results/eqtl','results/sqtl',
                                                  'results/phewas','results/cojo','results/BLUP', 'results/lz/']):
        '''
        remove all files except report
        '''
        for folder in folders:
            os.system(f'rm -r {self.path}{folder}')
            printwithlog(f'removing file {self.path}{folder}')
        self.make_dir_structure()

    def impute_traits(self, data_dictionary: str = '', groupby_columns = ['sex'], crosstrait_imputation = False, trait_subset = []):
        printwithlog(f'running imputation {"groupedby:"+ ",".join(groupby_columns) if len(groupby_columns) else ""}...')
        if type(data_dictionary) == str:
            if not len(data_dictionary): data_dictionary = pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv')
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
    
        data_dic_trait = data_dictionary.query('trait_covariate == "trait"').set_index('measure')
        data_dic_trait['covariates'] = data_dic_trait['covariates'].astype(str).apply(lambda x: list(set(x.split(',')) & set(self.df.columns)))
        if crosstrait_imputation:
            data_dic_trait['covariates'] = data_dic_trait['covariates'].apply(lambda x: list(set(x) | (set(self.df.columns) & set(data_dic_trait.index))))
    
        if len(trait_subset): 
            printwithlog(f'running imputation for traits {",".join(trait_subset)}...')
            data_dic_trait = data_dic_trait.loc[data_dic_trait.index.isin(trait_subset)]
        imputed_list = data_dic_trait.progress_apply(lambda x: impute_single_trait(dataframe=self.df, imputing_col=x.name, \
                                                                             covariate_cols=x.covariates, groupby=groupby_columns), axis = 1)
        imputeddf = pd.concat(imputed_list.imputed.to_list(), axis = 1)
        imputedqc = pd.concat(imputed_list.qc.to_list())
    
        if 'rfid' in self.df.columns: self.df = self.df.set_index('rfid')
        self.df = self.df.combine_first(imputeddf)
        self.df.to_csv(f'{self.path}imputed_data.csv')
        imputedqc.to_csv(f'{self.path}results/preprocessing/imputedqc.csv')
        return self.df

    def regressout_groupby(self, data_dictionary: pd.DataFrame(), covariates_threshold: float = 0.02, groupby_columns = ['sex']):
        printwithlog(f'running regressout groupedby {",".join(groupby_columns)}...')
        reg = regressoutgb(dataframe=self.df, data_dictionary=data_dictionary, groupby = groupby_columns, covariates_threshold = covariates_threshold)
        self.df = reg['regressed_dataframe']
        
        self.traits = self.df.filter(regex='regressedlr_*').columns.to_list()
        display(self.df[self.traits].count())
        statsReport.stat_check(self.df).make_report(f'{self.path}data_distributions.html')
        
        self.df.to_csv(f'{self.path}processed_data_ready.csv', index = False)
        self.df.to_csv(f'{self.path}results/processed_data_ready_n{self.df.shape[0]}_date{datetime.today().strftime("%Y-%m-%d")}.csv', index = False)
        reg['covariatesr2'].to_csv(f'{self.path}melted_explained_variances.csv', index = False)
        reg['covariatesr2pivoted'].to_csv(f'{self.path}pivot_explained_variances.csv',index = False)
        
        for trait in self.traits:
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(f'{self.path}data/pheno/{trait}.txt' ,  index = False, sep = ' ', header = None)
        simplified_traits = [x.replace('regressedlr_', '') for x in self.traits]
    
        if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
        trait_descriptions = [data_dictionary.set_index('measure').loc[x, 'description'] if (x in data_dictionary.measure.values) else 'UNK' for x in simplified_traits]
        self.get_trait_descriptions = defaultdict(lambda: 'UNK', {k:v for k,v in zip(self.traits, trait_descriptions)})
        return self.df.copy()
        
    def regressout(self, data_dictionary: pd.DataFrame(), covariates_threshold: float = 0.02, verbose = False):
        printwithlog(f'running regressout...')
        if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
        df, datadic = self.df.copy(), data_dictionary
        datadic = datadic[datadic.measure.isin(df.columns)].drop_duplicates(subset = ['measure'])
        def getcols(df, string): return df.columns[df.columns.str.contains(string)].to_list()
        categorical_all = list(datadic.query('trait_covariate == "covariate_categorical"').measure)
        dfohe = df.copy()
        ohe = OneHotEncoder()
        oheencoded = ohe.fit_transform(dfohe[categorical_all].astype(str)).todense()
        dfohe[[f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_all)]] = oheencoded
        alltraits = list(datadic.query('trait_covariate == "trait"').measure.unique())
        dfohe.loc[:, alltraits] = QuantileTransformer(n_quantiles = 100).fit_transform(dfohe.loc[:, alltraits].apply(pd.to_numeric, errors='coerce'))
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
                printwithlog(f'variables:{variables}-categorical:{categorical}-continuous:{continuous}')
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
        passthrough_traits = list(datadic.query('trait_covariate == "trait" and covariates == "passthrough"').measure.unique())
        if len(passthrough_traits):
            dfcomplete[['regressedlr_' + x for x in passthrough_traits]] = dfcomplete[passthrough_traits]
        strcomplete = statsReport.stat_check(dfcomplete.reset_index(drop = True))
        dfcomplete.to_csv(f'{self.path}processed_data_ready.csv')
        dfcomplete.to_csv(f'{self.path}results/processed_data_ready_n{self.df.shape[0]}_date{datetime.today().strftime("%Y-%m-%d")}.csv')
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
        printwithlog(f'running timeseries regressout groupedby {",".join(groupby_columns)}...')
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
        self.df.to_csv(f'{self.path}results/processed_data_ready_n{self.df.shape[0]}_date{datetime.today().strftime("%Y-%m-%d")}.csv', index = False)
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

    def add_latent_spaces(self):
        latspace = pd.concat([self._make_eigen3d_figure(ret = 'data'), self._make_umap3d_figure(ret = 'data')], axis = 1).rename(lambda x: x.lower(), axis =1)
        latspace.loc[:, latspace.columns.str.contains("regressedlr_\w+_clusters")] = latspace.filter(regex = "regressedlr_\w+_clusters").astype(str).replace('-1', np.nan)
        latspace =  pd.get_dummies(latspace, columns=latspace.filter(regex = "regressedlr_\w+_clusters").columns, dummy_na=True, dtype = float)
        for i in latspace.columns[latspace.columns.str.contains('^regressedlr_\w+_clusters_nan')]:
            latspace.loc[latspace[i].astype(bool), latspace.columns.str.contains(i[:-3])] = np.nan
        latspace = latspace.loc[:, ~latspace.columns.str.contains('regressedlr_\w+_clusters_nan') & (latspace.count()> 10)]
        self.df = self.df.loc[:, ~self.df.columns.str.lower().str.contains('unnamed: ')]
        self.df = self.df.set_index('rfid').combine_first(latspace.rename(columns= lambda x: x.replace('regressedlr_', '')))
        self.df = self.df.combine_first(latspace).reset_index()#
        self.traits = sorted(list(set(self.traits)| set(latspace.columns)))
        for idx, row in latspace.columns.str.extract('(umap|pc|pca)(\d+|_clusters_\d+)').iterrows():
            if "_clusters" not in str(row[1]):
                self.get_trait_descriptions[latspace.columns[idx]] = f'{"Principal Component" if row[0] == "pc" else "UMAP" } {row[1]} of all traits'
            else:
                self.get_trait_descriptions[latspace.columns[idx]] = f'HDBSCAN cluster {row[1].split("_")[-1]} of {"PCA" if row[0] in ["pc", "pca"] else "UMAP"}'
        self.df = self.df.loc[:,~self.df.columns.duplicated()]
        strcomplete = statsReport.stat_check(self.df.reset_index(drop = True))
        self.df.to_csv(f'{self.path}processed_data_ready.csv', index = False)
        self.df.to_csv(f'{self.path}results/processed_data_ready_n{self.df.shape[0]}_date{datetime.today().strftime("%Y-%m-%d")}.csv', index = False)
        appended_traits = list(self.df.columns[ self.df.columns.str.startswith('regressedlr_umap') | self.df.columns.str.startswith('regressedlr_pc')])
        strcomplete.make_report(f'{self.path}data_distributions.html')
        display(self.df.loc[:,  appended_traits].count())
        for trait in appended_traits: 
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(f'{self.path}data/pheno/{trait}.txt' ,  index = False, sep = ' ', header = None)
        return self.df.copy()
        
    def plink2Df(self, call, temp_out_filename = 'temp/temp', dtype = 'ld'):
        '''
        this function receives a plink call as a string, 
        runs the call, 
        then reads the output file with the ending dtype 
        and returns it as a pandas table
        '''
        random_id = np.random.randint(1e8)
        
        full_call = re.sub(r' +', ' ', call + f' --out {self.path}{temp_out_filename}{random_id}')
        
        ### add line to delete temp_out_filename before doing the 
        bash(full_call, print_call = False)

        try: 
            out = pd.read_csv(f'{self.path}{temp_out_filename}{random_id}.{dtype}', sep = '\s+')
            os.system(f'rm {self.path}{temp_out_filename}{random_id}.{dtype}')
            os.system(f'rm {self.path}{temp_out_filename}{random_id}.log')
        except:
            printwithlog(f"file not found at self.plink")
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
    
    def _gcta(self,  **kwargs):
        '''
        this function is a wrapper to run gcta as a python function
        instead of having to have it as a string and bash call it
        if writing a flag that doesn't require a variable e.g.
        --make-grm use make_grm = ''
        '''
        call = f'{self.gcta} ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                    for k,v in kwargs.items()])
        if 'out' not in kwargs.items():
            call += f' --out {self.path}temp/temp'
        bash(call, print_call=False)
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
            printwithlog(f'found possible error in log, check the file {self.path}log{loc}/{func}.log')
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
                       filter_based_on_subset: bool = True, makefigures = False):

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
                tempdf.query('sex in ["M", "m", "male", "1", 1]')[['rfid', 'rfid']].to_csv(self.sample_path_males, 
                                                                                           index = False, header = None, sep = ' ')
                tempdf.query('sex in ["F", "f", "female", "2", 2]')[['rfid', 'rfid']].to_csv(self.sample_path_females, 
                                                                                             index = False, header = None, sep = ' ')

        printwithlog('calculating missing hwe maf for autossomes and MT')
        plink(bfile = self.all_genotypes, chr = f'1-{self.n_autosome} MT', hardy = '', keep = self.sample_path, thread_num =  self.threadnum, 
              freq = '', missing = '', nonfounders = '', out = f'{self.path}genotypes/autosomes', 
              chr_set = f'{self.n_autosome} no-xy') #autosome_num = 20
        printwithlog('calculating missing hwe maf for X')
        plink(bfile = self.all_genotypes, chr = 'X', hardy = '', keep = self.sample_path, thread_num =  self.threadnum,
              freq = '' , missing = '', nonfounders = '', out = f'{self.path}genotypes/xfilter',
              filter_females = '', chr_set = f'{self.n_autosome} no-xy')
        printwithlog('calculating missing hwe maf for Y')
        plink(bfile = self.all_genotypes, chr = 'Y', hardy = '', keep = self.sample_path, thread_num =  self.threadnum,
              freq = '' , missing = '', nonfounders = '', out = f'{self.path}genotypes/yfilter', 
              filter_males = '', chr_set = f'{self.n_autosome} no-xy')
        full = []
        for x in tqdm(['autosomes', 'xfilter', 'yfilter']):
            full_sm = []
            if os.path.isfile(f'{self.path}genotypes/{x}.lmiss'): 
                full_sm += [pd.read_csv(f'{self.path}genotypes/{x}.lmiss', sep = '\s+')[['CHR','SNP', 'F_MISS']].set_index('SNP')]
            if os.path.isfile(f'{self.path}genotypes/{x}.hwe'): 
                full_sm += [pd.read_csv(f'{self.path}genotypes/{x}.hwe', sep = '\s+')[['SNP', 'GENO' ,'P']].set_index('SNP').set_axis(['GENOTYPES','HWE'], axis = 1)]
            if os.path.isfile(f'{self.path}genotypes/{x}.frq'): 
                full_sm += [pd.read_csv(f'{self.path}genotypes/{x}.frq', sep = '\s+')[['SNP', 'MAF', 'A1', 'A2']].set_index('SNP')]
            if len(full_sm): full += [pd.concat(full_sm, axis = 1)]
        full = pd.concat(full)

        full['PASS_MISS'] = ((full.F_MISS <= thresh_m) + \
                             (full.CHR == self.n_autosome + 2 )) > 0 
        full['PASS_MAF'] = ((full.MAF - .5).abs() <= .5 - thresh_maf)# +
                            #(full.CHR == self.n_autosome + 2)) > 0 
        full['PASS_HWE']= ((full.HWE >= thresh_hwe) + \
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
            printwithlog('making plots for heterozygosity per CHR')
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
                if len(self.foundersbimfambed):
                    _distance_to_founders((bim, fam, gen), self.foundersbimfambed,
                                          f'{self.path}images/genotypes/dist2founders/dist2founder_chr{c}',c , nautosomes = self.n_autosome)
                    _make_umap_plot((bim, fam, gen), self.foundersbimfambed, f'{self.path}images/genotypes/umap/umap_chr{c}',c,
                                    nautosomes = self.n_autosome)
            
    def generateGRM(self, autosome_list: list = [], print_call: bool = True, extra_chrs: list = ['X', 'Y', 'MT'],addx2full = False , **kwards):
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
        '''
        printwithlog('generating GRM...')
        funcName = inspect.getframeinfo(inspect.currentframe()).function
        
        if not autosome_list:
            autosome_list = list(range(1,self.n_autosome+1))
            
        all_filenames_partial_grms = pd.DataFrame(columns = ['filename'])

        if 'X' in extra_chrs:
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --autosome-num {self.n_autosome} \
                           --make-grm-xchr --out {self.xGRM}',
                        f'{funcName}_chrX', print_call = False)
            if addx2full: all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.xGRM
            
        if 'Y' in extra_chrs:
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --keep {self.sample_path_males} --autosome-num {self.n_autosome+4} \
                               --make-grm-bin --chr {self.n_autosome+2} --out {self.yGRM}',
                            f'{funcName}_chrY', print_call = False)
            #all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.yGRM
            
        if 'MT' in extra_chrs:
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --autosome-num {self.n_autosome+6} --chr {self.n_autosome+4}\
                           --make-grm-bin --out {self.mtGRM}',
                        f'{funcName}_chrMT', print_call = False)
            all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.mtGRM
            
        for c in tqdm(autosome_list):
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --chr {c} --autosome-num {self.n_autosome}\
                         --make-grm-bin --out {self.path}grm/{c}chrGRM',
                        f'{funcName}_chr{c}',  print_call = False)

            all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = f'{self.path}grm/{c}chrGRM'

        all_filenames_partial_grms = all_filenames_partial_grms[all_filenames_partial_grms.filename.apply(lambda x: os.path.isfile(f'{x}.grm.bin'))]
        all_filenames_partial_grms.to_csv(f'{self.path}grm/listofchrgrms.txt', index = False, sep = ' ', header = None)

        self.bashLog(f'{self.gcta} {self.thrflag} --mgrm {self.path}grm/listofchrgrms.txt \
                       --make-grm-bin --out {self.autoGRM}', f'{funcName}_mergedgrms',  print_call = False )

        if not os.path.isfile(f'{self.autoGRM}.grm.bin'): raise ValueError('could not merge the grms')
            
        if self.threshold == 'auto': 
            printwithlog('threshold was set to auto, at this point we can recalculate the threshold...')
            self.estimate_pval_threshold()
        return 1
    
    def make_genetic_PCA_fig(self) -> tuple:
        self._gcta(grm_bin = f'{self.path}grm/AllchrGRM', pca=20, thread_num = self.threadnum)
        eigenvals = pd.read_csv(f'{self.path}temp/temp.eigenval', header = None ).rename(lambda x: f'gPC{x+1}').set_axis(['eigenvalues'],axis = 1)
        pcs = pd.read_csv(f'{self.path}temp/temp.eigenvec', header = None, sep = '\s+', index_col=[0,1] ).rename(lambda x: f'gPC{x-1}', axis = 1).droplevel(0)
        pcs['hdbscan'] = HDBSCAN(min_cluster_size=50).fit_predict(pcs)
        nclasses = len(pcs['hdbscan'].unique())
        pcs.index.names = ['rfid']
        fig_eigen = px.scatter_3d(pcs.reset_index(), x = 'gPC1', y='gPC2', z = 'gPC3', opacity=.6, hover_name = 'rfid',
                                hover_data = { i:True for i in  pcs.columns[:5].to_list() + [f'hdbscan']})
        fig_eigen.update_traces(marker=dict(line=dict(width=3, color='black'),  color = pcs[f'hdbscan'], 
                                            colorscale=[[0, 'rgb(0,0,0)']]+ [[(i+1)/nclasses, f"rgb{sns.color_palette('tab10')[i%10]}"] for i in range(nclasses)],
                                           colorbar=dict(thickness=10, outlinewidth=0, len = .5, title = 'hdbscan')))
        fig_eigen.update_layout( width=1200,height=1200,autosize=False, template = 'simple_white', 
                                    scene=go.layout.Scene(
                                        xaxis=go.layout.scene.XAxis(title='gPC1 ' + str((100*eigenvals.loc['gPC1', 'eigenvalues']/eigenvals.eigenvalues.sum()).round(3))),
                                        yaxis=go.layout.scene.YAxis(title='gPC2 ' + str((100*eigenvals.loc['gPC2', 'eigenvalues']/eigenvals.eigenvalues.sum()).round(3))),
                                        zaxis=go.layout.scene.ZAxis(title='gPC3 ' + str((100*eigenvals.loc['gPC3', 'eigenvalues']/eigenvals.eigenvalues.sum()).round(3)))),
                                    coloraxis_colorbar_x=1.05,coloraxis_colorbar_y = .4,coloraxis_colorbar_len = .8,)  
        return fig_eigen, (pcs, eigenvals)
    
    def make_panel_genetic_PCA(self):
        pn.extension('plotly')
        fig, (evec, eval) = self.make_genetic_PCA_fig()
        cols3d = [x for x in self.traits if '_just_' not in x]
        df3d = self.df.set_index('rfid')[cols3d].dropna(how = 'all')
        pca = PCA(n_components=3)
        df3d.loc[:, [f'PC{i}' for i in range(1,4)]] = make_pipeline(KNNImputer(), pca ).fit_transform(df3d)
        cgram = pd.concat([df3d.rename(lambda x: str(x)).rename(lambda x: x.replace('regressedlr_', ''), axis = 1), 
                           evec.rename(lambda x: str(x)).iloc[:, :5]], axis = 1).corr()
        _ = cgram.columns.str.startswith('gPC')
        cgram = cgram.loc[_, ~_].T.fillna(0)
        fig2 = dashbio.Clustergram(cgram, center_values = False, column_labels=cgram.columns.to_list(), row_labels=cgram.index.to_list() , color_map = 'RdBu', line_width = 1, )
        fig2.update_layout( width=1200,height=1000,autosize=False, template = 'simple_white',
                         coloraxis_colorbar_x=1.05,coloraxis_colorbar_y = .4,coloraxis_colorbar_len = .8)
        
        return pn.Card(pn.Card(fig, title = 'genetic PCA', collapsed = True), 
                pn.Card(fig2, title = 'correlation between genomic PCs and Traits', collapsed = True), 
                title = 'Genomic PCA', collapsed = True)
    
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
        printwithlog(f'starting snp heritability {self.project_name}')       
        
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
        Generates a genetic correlation matrix using GCTA.
    
        Parameters
        ----------
        traitlist : list, optional
            List of traits for which genetic correlations will be computed.
            If not provided, the function uses the traits from the object's attribute.
        
        print_call : bool, optional
            Whether the command line call for each trait is printed.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame representing the genetic correlation matrix.
    
        Design
        ------
        This function performs the following steps:
        1. Checks for or creates a Dask client for parallel computation.
        2. Prepares data and directories.
        3. Computes genetic correlations and phenotypic correlations for all trait pairs.
        4. Processes the results and creates a melted table.
        5. Saves the genetic correlation matrix with hierarchical clustering.
        6. Adds heritability information to the matrix.
        7. Generates and saves a clustered heatmap using Seaborn.
        8. Returns the final genetic correlation matrix.
        '''
        printwithlog(f'Starting genetic correlation matrix v2 for {self.project_name}...')
        if not client._get_global_client(): 
            # cluster = SLURMCluster(cores=1, processes=1,memory='2g', account="ACCOUNTNAME", queue="condo", scheduler_options={'dashboard_address': ':PORTN'}, \
            #                         worker_extra_args=["-p condo -q condo -N 1 -n 1 -c 1 -t 01:00:00"])
            # cluster.adapt(minimum_jobs = 1, maximum_jobs = 10)
            # cline = Client(cluster)
            cline = Client( processes = False)
        else: 
            cline = client._get_global_client()
        try:
            if traitlist in [0, '', None, 'nan', []]: traitlist = self.traits
        except: pass
        d_ = {t: str(num) for num, t in enumerate(['rfid']+ traitlist)} 
        self.df[['rfid', 'rfid']+ traitlist].fillna('NA').to_csv(f'{self.path}data/allpheno.txt', sep = '\t', header = None, index = False)
        loop = pd.DataFrame(itertools.combinations(traitlist, 2), columns = ['trait1', 'trait2'])
        lp =  dd.from_pandas(loop,npartitions=min(200, max(10, int(loop.shape[0]/5))))
        
        os.makedirs(f'{self.path}temp/rG', exist_ok=True)
        
        def get_gcorr_phecorr(trait1, trait2):
            randomid = trait1+trait2
            bash(f'''{self.gcta} --reml-bivar {d_[trait1]} {d_[trait2]} {self.thrflag} \
                --grm {self.autoGRM} --pheno {self.path}data/allpheno.txt --reml-maxit 200 \
                --reml-bivar-lrt-rg 0 --out {self.path}temp/rG/gencorr.temp{randomid}''', print_call=False)
            if not os.path.exists(f'{self.path}temp/rG/gencorr.temp{randomid}.hsq'):
                rG, rGse, strrG = 0, 100, f"0 +- *"
            else:
                temp = pd.read_csv(f'{self.path}temp/rG/gencorr.temp{randomid}.hsq', sep = '\t',engine='python',
                                   dtype= {'Variance': float}, index_col=0 ,skipfooter=6)
                rG, rGse = temp.loc['rG', 'Variance'], temp.loc['rG', 'SE']
                strrG = f"{rG} +- {rGse}" if (abs(rGse) < 1) else f"0 +- *"
            phecorr = str(self.df[[trait1, trait2]].corr().iloc[0,1])
            strphe = phecorr.replace('nan', '0')+ ' +- ' + ( '*' if 'nan' in phecorr else '0')
            return [rG, rGse, strrG, phecorr, strphe]
            
        def dfget_gcorr_phecorr(df):
            return df.apply(lambda x: get_gcorr_phecorr(x.trait1, x.trait2), axis = 1)
        
        _ = lp.map_partitions(dfget_gcorr_phecorr, meta = pd.Series())
        future = cline.compute(_)
        progress(future,notebook = False, interval="300s")
        loop[['rG', 'rGse', 'strrG', 'phecorr', 'strphe']] = [eval(x) for x in future.result()]
        del future
        os.system(f'rm -r {self.path}temp/rG')
        
        for i in traitlist: loop.loc[loop.shape[0], :] = [i,i, 1., 0, '1 +- 0', 1., '1 +- 0']
        loop = loop.sort_values(['trait1', 'trait2'])
        loop.rename({'rG': 'genetic_correlation','rGse': 'rG_SE'}, axis =1).to_csv(f'{self.path}results/heritability/genetic_correlation_melted_table.csv')
        
        outg = loop.pivot(columns = 'trait1', index = 'trait2', values = 'strrG')
        outg = outg.combine_first(outg.T) \
                   .rename(lambda x: x.replace('regressedlr_', '')) \
                   .rename(lambda x: x.replace('regressedlr_', ''), axis = 1)
        outp = loop.pivot(columns = 'trait1', index = 'trait2', values = 'strphe')
        outp = outp.combine_first(outp.T) \
                   .rename(lambda x: x.replace('regressedlr_', '')) \
                   .rename(lambda x: x.replace('regressedlr_', ''), axis = 1)
        
        hieg = linkage(distance.pdist(outg.applymap(lambda x: float(x.split(' +- ')[0])))) #method='average'
        lk = leaves_list(hieg)
        outg, outp = outg.iloc[lk, lk], outp.iloc[lk, lk]
        
        outg.index, outp.index = outg.index.values, outp.index.values
        outg.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv')
        outmixed = outg.mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('').T  +  outp.mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('').T
        
        ### add heritability
        if not os.path.isfile(self.heritability_path): self.snpHeritability()
        H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col= 0).applymap(lambda x: 0 if x =='Fail' else x).astype(float)
        H2['her_str'] = H2['V(G)/Vp'].round(3).astype(str) + ' +- ' + H2.heritability_SE.round(3).astype(str)
        for i in outmixed.columns: outmixed.loc[i,i] =  H2.loc['regressedlr_'+i, 'her_str']
        outmixed.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix.csv')
        
        ## make figure
        a = sns.clustermap(outmixed.applymap(lambda x: float(x.split(' +- ')[0])),  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                annot=outmixed.applymap(lambda x: '' if '*' not in x else '*'), vmin =-1, vmax =1, center = 0 , fmt = '', square = True, linewidth = .3, figsize=(25, 25) )
        dendrogram(hieg, ax = a.ax_col_dendrogram)
        a.ax_cbar.set_position([.1, .2, .05, 0.5])
        plt.savefig(f'{self.path}images/genetic_correlation_matrix.png', dpi = 400)
        plt.savefig(f'{self.path}images/genetic_correlation_matrix.eps')
        
        outmixed2 = outmixed.T.sort_index().T.sort_index()
        a = sns.clustermap(outmixed2.applymap(lambda x: float(x.split('+-')[0])).copy().T,  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                        annot=outmixed2.applymap(lambda x: '' if '*' not in x else '*').copy().T, vmin =-1, vmax =1, center = 0 , fmt = '', square = True, linewidth = .3, figsize=(15, 15) )
        a.ax_heatmap.plot([0,outmixed2.shape[0]], [0, outmixed2.shape[0]], color = 'black')
        a.ax_cbar.set_position([.1, .2, .05, 0.5])
        plt.savefig(f'{self.path}images/genetic_correlation_matrix_sorted.png', dpi = 400)
        return outmixed
    
    def genetic_correlation_matrix_old(self,traitlist: list = [], print_call = False, save_fmt = ['png']) -> pd.DataFrame():
        '''
        Create a blup model to get SNP effects and breeding values.
        
        Parameters
        ----------
        print_call: bool = False
            whether the command line call for each trait is printed.
        
        Design
        ------

        '''
        printwithlog(f'starting genetic correlation matrix {self.project_name}...')
        if not traitlist: traitlist = self.traits
        d_ = {t: str(num) for num, t in enumerate(['rfid']+ traitlist)} 
        self.df[['rfid', 'rfid']+ traitlist].fillna('NA').to_csv(f'{self.path}data/allpheno.txt', sep = '\t', header = None, index = False)
        outg = pd.DataFrame()
        outp = pd.DataFrame()
        genetic_table = pd.DataFrame()
        os.makedirs(f'{self.path}temp/rG', exist_ok=True)
        for trait1, trait2 in tqdm(list(itertools.combinations(traitlist, 2))):
            self.bashLog(f'''{self.gcta} --reml-bivar {d_[trait1]} {d_[trait2]} {self.thrflag} \
                --grm {self.autoGRM} --pheno {self.path}data/allpheno.txt --reml-maxit 200 \
                --reml-bivar-lrt-rg 0 --out {self.path}temp/rG/gencorr.temp{trait1}{trait2}''', f'gcorr{trait1}{trait2}', print_call=False)
            if os.path.exists(f'{self.path}temp/rG/gencorr.temp{trait1}{trait2}.hsq'):
                # temp = pd.read_csv(f'{self.path}temp/rG/gencorr.temp{trait1}{trait2}.hsq', sep = '\t',engine='python' ,
                #                    dtype= {'Variance': float}, index_col=0 ,skipfooter=6)
                temp = pd.read_csv(f'{self.path}temp/rG/gencorr.temp{trait1}{trait2}.hsq', sep = '\t',engine='python', 
                                   index_col=0 ,names = ['Source','Variance','SE'], skiprows=1, dtype={'SE':float})
                temp['Variance'] = temp['Variance'].map(lambda x: float(str(x).split(' ')[0]))
                outg.loc[trait1, trait2] = f"{temp.loc['rG', 'Variance']}+-{temp.loc['rG', 'SE']}"
                outg.loc[trait2, trait1] = f"{temp.loc['rG', 'Variance']}+-{temp.loc['rG', 'SE']}"
                phecorr = str(self.df[[trait1, trait2]].corr().iloc[0,1])
                genetic_table.loc[len(genetic_table), ['trait1', 'trait2','phenotypic_correlation','genetic_correlation', 'rG_SE', 'pval']] = \
                                                      [trait1, trait2, phecorr, temp.loc['rG', 'Variance'], temp.loc['rG', 'SE'], temp.loc['Pval', 'Variance'] ]
                if (abs(temp.loc['rG', 'SE']) > 1):
                    outg.loc[trait1, trait2] = f"0 +- *"
                    outg.loc[trait2, trait1] = f"0 +- *"
                else: pass
            else: 
                #printwithlog(f'could not find {self.path}temp/rG/gencorr.temp{trait1}{trait2}.hsq')
                phecorr = str(self.df[[trait1, trait2]].corr().iloc[0,1])
                genetic_table.loc[len(genetic_table), ['trait1', 'trait2','phenotypic_correlation','genetic_correlation', 'rG_SE', 'pval']] = \
                                                      [trait1, trait2, phecorr, 0, 1000, 1]
                outg.loc[trait1, trait2] = f"0 +- *"
                outg.loc[trait2, trait1] = f"0 +- *"
                #bash(f'rm {self.path}logerr/genetic_correlation.log',print_call = False)
            
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
        if not os.path.isfile(self.heritability_path): self.snpHeritability()
        H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col=0).applymap(lambda x: 0 if x =='Fail' else x).astype(float)
        H2['her_str'] = H2['V(G)/Vp'].round(3).astype(str) + ' +- ' + H2.heritability_SE.round(3).astype(str)

        for version in ['clustering', 'sorted', 'original']:
            if version == 'clustering':
                outmixed = outg.mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('')  +  outp.mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('')
                hie = linkage(distance.pdist(outg.applymap(lambda x: float(x.split('+-')[0])).T))
            elif version == 'sorted':
                sct = sorted(outmixed.columns)
                outmixed = outg.loc[sct,sct].mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('')  +  outp.loc[sct,sct].mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('')
                hie = linkage(distance.pdist(outg.loc[sct,sct].applymap(lambda x: float(x.split('+-')[0])).T))
            elif version == 'original':
                sct = [x.replace('regressedlr_', '') for x in traitlist]
                outmixed = outg.loc[sct,sct].mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('')  +  outp.loc[sct,sct].mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('')
                hie = linkage(distance.pdist(outg.loc[sct,sct].applymap(lambda x: float(x.split('+-')[0])).T))
            for i in outmixed.columns: outmixed.loc[i,i] =  H2.loc['regressedlr_'+i, 'her_str']
            if version == 'clustering': 
                outmixed.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix.csv')
            a = sns.clustermap(outmixed.applymap(lambda x: float(x.split('+-')[0])).copy().T,  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                    annot=outmixed.applymap(lambda x: '' if '*' not in x else '*').copy().T, vmin =-1, vmax =1, center = 0 , fmt = '', square = True, linewidth = .3, figsize=(15, 15) )
            if version == 'clustering': dendrogram(hie, ax = a.ax_col_dendrogram)
            a.ax_cbar.set_position([0, .2, .03, 0.5])
            a.ax_heatmap.plot([0,outmixed.shape[0]], [0, outmixed.shape[0]], color = 'black')
            if 'png' in save_fmt: plt.savefig(f'{self.path}images/genetic_correlation_matrix{"_"+version if version != "clustering" else ""}.png', dpi = 400)
            for sfm in (set(save_fmt) - {'png'}):
                plt.savefig(f'{self.path}images/genetic_correlation_matrix{"_"+version if version != "clustering" else ""}.{sfm}')
            plt.close()
        return outmixed

    def make_genetic_correlation_figure(self, order = 'sorted', traits= [], save = True, include=['gcorr', 'pcorr'], size = 'pval'):
            if not len(traits): traits = self.traits
            traits = [x.replace('regressedlr_', '') for x in traits]
            gcorr = pd.read_csv(f'{self.path}results/heritability/genetic_correlation_melted_table.csv', index_col = 0)
            gcorr[['trait1','trait2']] =gcorr[['trait1','trait2']].applymap(lambda x:x.replace('regressedlr_', ''))
            gcorr = gcorr[gcorr.trait1.isin(traits) & gcorr.trait2.isin(traits)]
            if size == 'rG_SE': gcorr['size'] = (1.1 - gcorr.rG_SE.map(lambda x: min(x, 1)).fillna(1.))*20*30/len(traits)
            elif size == 'pval':
                gcorr['size'] = MinMaxScaler(feature_range = (1, 570) )\
                                .fit_transform(np.log(np.clip(-np.log10(gcorr[['pval']].replace(0, 1e-6).fillna(1)), a_min=.1, a_max= 100)))\
                                /len(traits)
            alltraits = list(sorted(set(gcorr.trait1) | set(gcorr.trait2)))
            if order == 'sorted': torder = {t:n for n,t in enumerate(alltraits)}
            else: 
                alltraits, torder = traits,{t:n for n,t in enumerate(traits)}
            gcorr = pd.concat([gcorr, gcorr.rename({'trait1' : 'trait2', 'trait2': 'trait1'}, axis = 1)])
            gcorr = gcorr.assign(or1 = gcorr.trait1.map(torder),or2 = gcorr.trait2.map(torder)).sort_values(['trait1', 'trait2'])
            H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col=0).applymap(lambda x: 0 if x =='Fail' else x).astype(float).rename({'V(G)/Vp': 'g'}, axis = 1).reset_index(names = ['trait'])
            H2['trait'] = H2['trait'].map(lambda x: x.replace('regressedlr_', ''))
            H2['size'] = (1.1 - H2.heritability_SE.map(lambda x: min(x, 1)).fillna(1.))*20*30/len(traits)
            H2 = H2[H2.trait.isin(traits)]
            
            if order == 'cluster': 
                gp = gcorr.pivot(index ='trait1', columns='trait2', values='genetic_correlation') 
                gp.loc[:,:] = 1- np.abs(gp.fillna(0).values + np.eye(len(gp)))
                from scipy.spatial.distance import squareform
                tt =  HDBSCAN(metric = 'precomputed', allow_single_cluster= True)
                cc = tt.fit(gp)
                alltraits = np.array(alltraits)[leaves_list(tt.single_linkage_tree_._linkage)]
                torder = {t:n for n,t in enumerate(alltraits)}
                gcorr = gcorr.drop(['or1', 'or2'], axis = 1).assign(or1 = gcorr.trait1.map(torder),or2 = gcorr.trait2.map(torder)).sort_values(['or1', 'or2'])
                print('pass')
            
            kdims1=hv.Dimension('trait1', values=alltraits)
            kdims2=hv.Dimension('trait2', values=alltraits)
            fig = hv.Points(gcorr.query('or2< or1'), kdims = [kdims1, kdims2],vdims=['phenotypic_correlation','genetic_correlation','rG_SE', 'size']) \
                                                      .opts( color='genetic_correlation', cmap='RdBu', size=hv.dim('size')*1.5 if ('gcorr' in include) else 0,
                                                            colorbar=True, frame_width=900, frame_height=900, tools=['hover'],line_color='black', padding=0.05) #
            fig = fig*hv.Points(gcorr.query('or2> or1'), kdims = [kdims1, kdims2],vdims=['phenotypic_correlation','genetic_correlation','rG_SE', 'size']) \
                                                      .opts( color='phenotypic_correlation', cmap='RdBu', size=18*30/len(traits)*1.5 if ('pcorr' in include) else 0, marker = 'square',
                                                            colorbar=True, frame_width=900, frame_height=900, tools=['hover'],line_color='black', padding=0.005) #
            # fig = fig*hv.Points(H2, kdims = ['trait', 'trait'],vdims=['V(G)','V(e)','Vp', 'heritability_SE'	, 'g', 'size']) \
            #                                           .opts( color='g', cmap='Greys', size=hv.dim('size'), marker = '+', 
            #                                                 colorbar=True, width=400, height=400, tools=['hover'],line_color='black', padding=0.005, angle=45) #
            fig = fig*hv.Labels(H2.assign(gtex = H2.g.map(lambda x: f"{int(x*100)}%")), kdims = ['trait', 'trait'],vdims=['gtex']).opts(text_font_size=f'{min(int(7*1.5*30/len(traits)), 20)}pt', text_color='black')
            fig = fig.opts(frame_height=900, frame_width=900,title = f'Genetic correlation', xlabel = '', ylabel = '',
                           fontsize={ 'xticks': f'{min(int(7*1.5*30/len(traits)), 20)}pt', 'yticks': f'{min(int(7*1.5*30/len(traits)), 20)}pt'},
                           xrotation=45,invert_yaxis = True, yrotation=45)
            if save:hv.save(fig, f"{self.path}images/genetic_correlation_matrix2{('_'+order).replace('_cluster', '')}.png")
            return fig
        
    
    def make_heritability_figure(self, traitlist: list = [], save_fmt = ['png', 'html', 'pdf', 'eps'], display = True, add_classes = True):
        '''
        Create a blup model to get SNP effects and breeding values.
        
        Parameters
        ----------
        traitlist: list = []
            list of traits to make figure, will default to all traits in self.traits
        
        print_call: bool = False
            whether the command line call for each trait is printed.
            
        save_fmt: list = ['png', 'html']
            formats to save the figures, defaults to png and html, other options are eps
            
        display: bool = True
            display figure after call, defaults to showing the figure
        
        Design
        ------

        '''
        if os.path.isfile(f'{self.path}results/heritability/heritability.tsv'): pass
        else:  self.snpHeritability()
        her = pd.read_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t', 
                          index_col=0).rename(lambda x: x.replace('regressedlr_', ''))
        her = her[~(her == 'Fail').all(1)].astype(float)
        her = her.rename({'V(G)/Vp': 'heritability'}, axis = 1).sort_values('heritability').dropna(subset = 'heritability')
        traitlist = pd.Series(her.index if not len(traitlist) else traitlist).str.replace('regressedlr_', '')
        her = her.loc[her.index.isin(traitlist)]
        if add_classes:
            if os.path.isfile(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv'): pass
            else: self.genetic_correlation_matrix()
            gcor = pd.read_csv(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv', 
                               index_col=0).applymap(lambda x: float(x.split('+-')[0]))
            classes = pd.DataFrame(HDBSCAN(metric = 'precomputed', min_cluster_size = 3).fit_predict(gcor.loc[her.index, her.index]), 
                                   index = her.index, columns = ['cluster']).astype(str)
            her = pd.concat([classes, her], axis = 1).sort_index().sort_values('heritability')
            fig = px.scatter(her.reset_index(names = 'trait'), x="trait", y="heritability", color="cluster", error_y="heritability_SE")
        elif not add_classes:
            fig = px.scatter(her.reset_index(names = 'trait').assign(color = 'steelblue'), x="trait", y="heritability", color="color", error_y="heritability_SE")
        fig.update_xaxes(categoryorder='array', categoryarray= her.index)
        fig.add_hline(y=0., line = {'color':'black', 'width': 3}, opacity= .7)
        for i in [0.1,0.2, 0.3]:
            fig.add_hline(y=i, line = {'color':'black', 'width': .7}, opacity= .7, line_dash="dot")
        fig.update_layout( template='simple_white',width = 1000, height = 800, showlegend=False)
        for fmt in save_fmt:
            if fmt == 'html': fig.write_html(f"{self.path}images/heritability_sorted.html")
            else: fig.write_image(f"{self.path}images/heritability_sorted.{fmt}",width = 1920, height = 1200)
        if display: fig.show(renderer = 'png',width = 1920, height = 1200)
        
    def BLUP(self,  print_call: bool = False, save: bool = True, frac: float = 1.,traits = [],**kwards):
        '''
        Create a blup model to get SNP effects and breeding values.
        
        Parameters
        ----------
        print_call: bool = False
            whether the command line call for each trait is printed.
        
        Design
        ------

        '''
        printwithlog(f'starting BLUP model {self.project_name}...')      
        if not len(traits): traits = self.traits
        for trait in tqdm(traits):
            os.makedirs( f'{self.path}data/BLUP', exist_ok = True)
            os.makedirs( f'{self.path}results/BLUP', exist_ok = True)
            trait_file, trait_rfids = f'{self.path}data/BLUP/{trait}_trait_.txt', f'{self.path}data/BLUP/{trait}_train_rfids.txt'
            out_file = f'{self.path}results/BLUP/{trait}' 
            if isinstance(frac, (list, pd.Series)):
                tempdf = self.df[self.df.rfid.isin(frac)]
            else: tempdf = self.df.sample(frac = frac) if frac < .999 else self.df.copy()
            tempdf[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file,  index = False, sep = ' ', header = None)
            tempdf[['rfid', 'rfid']].to_csv(trait_rfids, header = None, index = False, sep = ' ')
            self.bashLog(f'{self.gcta} --grm {self.autoGRM} --autosome-num {self.n_autosome} --keep {trait_rfids} \
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
        printwithlog(f'starting blup prediction for {self.project_name} and genotypes {genotypes2predict}')
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
        
    def GWAS(self, traitlist: list = [] ,subtract_grm: bool = True, loco: bool = True , run_per_chr: bool = False, skip_already_present = False,
             print_call: bool = False, **kwards):
        """
        This function performs a genome-wide association study (GWAS) on the provided genotype data using the GCTA software.
        
        Parameters
        ----------
    
        subtract_grm: 
            a boolean indicating whether to subtract the genomic relatedness matrix (GRM) from the GWAS results (default is True)
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

        printwithlog(f'starting GWAS {self.project_name}')
        from joblib import Parallel, delayed
        
        if len(traitlist) == 0:
            traitlist = self.traits
        #results = Parallel(n_jobs=2)(delayed(countdown)(10**7) for _ in range(20))
        check_present = lambda trait: os.path.exists(f'{self.path}results/gwas/{trait}.loco.mlma')\
                                      +os.path.exists(f'{self.path}results/gwas/{trait}.mlma')
        check_present_extra = lambda trait, chromp2: os.path.exists(f'{self.path}results/gwas/{trait}_chrgwas{chromp2}.mlma')
        
        if not run_per_chr:
            printwithlog('running gwas per trait...')
            for trait in tqdm(traitlist):
                if check_present(trait) and skip_already_present:
                    printwithlog(f'''skipping gwas autosomes for trait: {trait}, 
                          output files already present, to change this behavior use skip_already_present = False''')
                else:
                    grm_flag = f'--grm {self.path}grm/AllchrGRM ' if subtract_grm else ''
                    loco_flag = '-loco' if loco else ''
                    self.bashLog(f"{self.gcta} {self.thrflag} {grm_flag} \
                    --autosome-num {self.n_autosome}\
                    --pheno {self.path}data/pheno/{trait}.txt \
                    --bfile {self.genotypes_subset} \
                    --mlma{loco_flag} \
                    --out {self.path}results/gwas/{trait}",\
                                f'GWAS_{loco_flag[1:]}_{trait}',  print_call = print_call)
                    if not check_present(trait):
                        printwithlog(f"couldn't run trait: {trait}")
                        self.GWAS(traitlist = [trait], run_per_chr = True, print_call= print_call)
                        return 2
            ranges = [self.n_autosome+1, self.n_autosome+2, self.n_autosome+4]
        else:
            printwithlog('running gwas per chr per trait...')
            ranges = [i for i in range(1,self.n_autosome+5) if i != self.n_autosome+3]
        
        for trait, chrom in tqdm(list(itertools.product(traitlist,ranges))):
            chromp2 = self.replacenumstoXYMT(chrom)
            if check_present_extra(trait, chromp2) and skip_already_present:
                    printwithlog(f'''skipping gwas for trait: {trait} and chr {chromp2}, 
                          output files already present, to change this behavior use skip_already_present = False''')
            else:
                subgrmflag = f'--mlma-subtract-grm {self.path}grm/{chromp2}chrGRM' if chromp2 not in ['x','y'] else ''
                self.bashLog(f'{self.gcta} {self.thrflag} --pheno {self.path}data/pheno/{trait}.txt --bfile {self.genotypes_subset} \
                                           --grm {self.path}grm/AllchrGRM --autosome-num {self.n_autosome} \
                                           --chr {chrom} {subgrmflag} --mlma \
                                           --out {self.path}results/gwas/{trait}_chrgwas{chromp2}', 
                            f'GWAS_{chrom}_{trait}', print_call = print_call)
                
        return 1

    def fastGWAS(self, traitlist: list = [], chrlist: list = [], skip_already_present = False, print_call: bool = False, **kwards):
        """
        This function performs a genome-wide association study (GWAS) on the provided genotype data using the GCTA software.
        
        Parameters
        ----------
        traitlist:
            a list to subset the traits used for gwas (default is all traits in self.traits)
        traitlist:
            a list to subset the chromosomes used for gwas (default is all chromosomes)
        skip_already_present: 
            a boolean indicating whether skip if gwas was already performed to this particular trait|chr combination (default is False)
        print_call: 
            a boolean indicating whether to print the command that is being executed (default is False)
    
        Design
        ------
        it saves the GWAS results to a file in the 'results/gwas' directory,
        with the filename indicating the trait, GRM subtraction status, and LOCO status.
        """  
        if not client._get_global_client(): 
            # cluster = SLURMCluster(cores=1, processes=1,memory='2g', account="ACCOUNTNAME", queue="condo", scheduler_options={'dashboard_address': ':PORTN'}, \
            #                         worker_extra_args=["-p condo -q condo -N 1 -n 1 -c 1 -t 01:00:00"])
            # cluster.adapt(minimum_jobs = 1, maximum_jobs = 10)
            # cline = Client(cluster)
            #display(cline)
            cline = Client( processes = False)
        else: 
            #display(client._get_global_client())
            cline = client._get_global_client()
        if len(traitlist) == 0: traitlist = self.traits
        printwithlog(f'starting fastGWAS for {len(traitlist)} traits...')
        chrsingrm = pd.read_csv(f'{self.path}grm/listofchrgrms.txt', header = None)[0]\
                      .map(lambda x: basename(x).replace('chrGRM', '')).to_list()
        ranges = chrlist if len(chrlist) else [i for i in range(1,self.n_autosome+5) if i != self.n_autosome+3]
        looppd = pd.DataFrame(itertools.product(traitlist,ranges), columns = ['trait', 'chrom'] )
        print(f'estimated time assuming 10min per chr {round(looppd.shape[0]*600/3600/float(self.threadnum), 5)}H')
        loop = dd.from_pandas(looppd, npartitions=max(len(traitlist)*5, 200))#.persist() #npartitions=self.threadnum 
        check_present_extra = lambda trait, chromp2: os.path.exists(f'{self.path}results/gwas/{trait}_chrgwas{chromp2}.mlma')
        def _smgwas(trait, chrom):
            chromp2 = self.replacenumstoXYMT(chrom)
            if check_present_extra(trait, chromp2) and skip_already_present:
                    printwithlog(f'''skipping gwas for trait: {trait} and chr {chromp2}, 
                          output files already present, to change this behavior use skip_already_present = False''')
            else:
                subgrmflag = f'--mlma-subtract-grm {self.path}grm/{chromp2}chrGRM' if chromp2 in chrsingrm else ''
                bash(f'{self.gcta} --thread-num 1 --pheno {self.path}data/pheno/{trait}.txt --bfile {self.genotypes_subset} \
                                           --grm {self.path}grm/AllchrGRM --autosome-num {self.n_autosome} \
                                           --chr {chrom} {subgrmflag} --mlma \
                                           --out {self.path}results/gwas/{trait}_chrgwas{chromp2}', 
                            print_call = print_call)#f'GWAS_{chrom}_{trait}', 
        def _gwas(df):
            df.apply(lambda x: _smgwas(x.trait, x.chrom), axis = 1)
            return 1
        _ = loop.map_partitions(_gwas, meta = pd.Series())
        future = cline.compute(_)
        progress(future,notebook = False,  interval="300s") #, group_by="spans"
        wait(future)
        del future
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
        printwithlog(f'starting adding gwas to database ... {self.project_name}') 
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

        phedb = self.phewas_db.split(',')[0]
        all_new = pd.concat(all_new)  
        if not os.path.isfile(phedb):
            alldata = all_new
        else:
            try:
                alldata = pd.concat([all_new, pd.read_parquet(phedb)])
            except:
                printwithlog(f"Could not open phewas database in file: {phedb}, rebuilding db with only this project")
                if safe_rebuild: 
                    raise ValueError('not doing anything further until data is manually verified')
                    return
                else: 
                    alldata = all_new
        
        alldata.drop_duplicates(subset = ['researcher', 'project', 'round_version', 'trait', 'SNP', 'uploadeddate'], 
                                keep='first').to_parquet(phedb, index = False, compression='gzip')
        
        return 1

    def prune_genotypes(self):
        printwithlog('starting genotype prunner...')
        snps,_,gens = pandas_plink.read_plink(self.genotypes_subset)
        gens = da.nan_to_num(gens, -1).astype(np.int8)
        def prune_dups(array):
            dict = defaultdict(list, {})
            for num, i in enumerate(array): dict[i.tobytes()] += [num]
            return dict
        printwithlog('starting genotype dups finder...')    
        pruned = prune_dups(gens.compute())
        first_snps = [snps.loc[v[0], 'snp'] for k,v in pruned.items()]
        printwithlog(f'saving resulst to:\n1){self.path}pvalthresh/genomaping.parquet.gz\n2){self.path}pvalthresh/prunned_dup_snps.in\n3){self.path}genotypes/prunedgenotypes')  
        prunedset = pd.DataFrame([[k,'|'.join(map(str, v))] for k,v in pruned.items()], columns = ['genotypes', 'snps'])
        prunedset.to_parquet(f'{self.path}pvalthresh/genomaping.parquet.gz', compression = 'gzip')
        pd.DataFrame(first_snps).to_csv(f'{self.path}pvalthresh/prunned_dup_snps.in', index = False, header = None)
        plink(bfile=self.genotypes_subset,  thread_num = self.threadnum, extract = f'{self.path}pvalthresh/prunned_dup_snps.in',
              make_bed = '', out = f'{self.path}genotypes/prunedgenotypes')
        printwithlog(f'prunned data has {format(prunedset.shape[0], ",")} out of the original {format(gens.shape[0], ",")}')
        return f'saved prunned genotypes to {self.path}genotypes/prunedgenotypes'
        
    def estimate_pval_threshold(self, replicates = 1000, sample_size = 'all', exact_prunner = True ,prunning_window = 5000000, prunning_step = 1000, remove_after = False):
        printwithlog('starting P-value threshold calculation...')
        if sample_size == 'all': sample_size = len(self.df)
        if sample_size < 1: round(len(self.df)*sample_size)
        cline = Client( processes = False) if not client._get_global_client() else client._get_global_client()
        os.makedirs(f'{self.path}pvalthresh', exist_ok = True)
        os.makedirs(f'{self.path}pvalthresh/gwas/', exist_ok = True)
        os.makedirs(f'{self.path}pvalthresh/randomtrait/', exist_ok = True)
        
        if exact_prunner: self.prune_genotypes()
        if not exact_prunner:
            prunning_params = f'{prunning_window} {prunning_step} 0.999'
            printwithlog(f'prunning gentoypes using {prunning_params}')
            if not os.path.exists(f'{self.path}pvalthresh/pruned_data.prune.in'):
                plink(bfile=self.genotypes_subset, indep_pairwise = prunning_params, out = f'{self.path}pvalthresh/pruned_data', thread_num = self.threadnum)
                plink(bfile=self.genotypes_subset,  thread_num = self.threadnum, extract = f'{self.path}pvalthresh/pruned_data.prune.in', 
                      make_bed = '', out = f'{self.path}genotypes/prunedgenotypes')
            npruned_snps = [pd.read_csv(f'{self.path}pvalthresh/pruned_data.prune.{i}', header = None).shape[0] for i in ['in', 'out']]
            display(f'prunned data has {npruned_snps[0]} out of the original {npruned_snps[1]}' )
        
        def get_maxp_1sample(ranid, skip_already_present = True, remove_after = True ):
            os.makedirs(f'{self.path}pvalthresh/gwas/{ranid}', exist_ok = True)
            r = np.random.RandomState(ranid)
            valuelis = r.normal(size = self.df.shape[0])
            valuelis *= r.choice([1, np.nan],size = self.df.shape[0] , 
                                 p = [sample_size/self.df.shape[0], 1-sample_size/self.df.shape[0]])
            self.df[['rfid', 'rfid']].assign(trait = valuelis).fillna('NA').astype(str).to_csv(f'{self.path}pvalthresh/randomtrait/{ranid}.txt',  index = False, sep = ' ', header = None)
            maxp = 0
            for c in self.chrList():
                chrom = self.replaceXYMTtonums(c)
                filename = f'{self.path}pvalthresh/gwas/{ranid}/chrgwas{c}' 
                if os.path.exists(f'{filename}.mlma') and skip_already_present: pass
                    #printwithlog(f'''skipping gwas for trait: {ranid} and chr {c}''')
                else:
                    subgrmflag = f'--mlma-subtract-grm {self.path}grm/{c}chrGRM' if c not in ['x','y'] else ''
                    bash(f'{self.gcta} --thread-num 1 --pheno {self.path}pvalthresh/randomtrait/{ranid}.txt --bfile {self.path}genotypes/prunedgenotypes \
                                               --grm {self.path}grm/AllchrGRM --autosome-num {self.n_autosome} \
                                               --chr {chrom} {subgrmflag} --mlma \
                                               --out {filename}', 
                                 print_call = False)#f'GWAS_{chrom}_{ranid}',
                if os.path.exists(f'{filename}.mlma'): chrmaxp = np.log10(pd.read_csv(f'{filename}.mlma', sep = '\t')['p'].min())
                else: chrmaxp = 0
                if chrmaxp < maxp: maxp = chrmaxp
            if remove_after:
                bash(f'rm -r {self.path}pvalthresh/gwas/{ranid}')
            return maxp
        
        looppd = pd.DataFrame(range(replicates), columns = ['reps'])
        loop   = dd.from_pandas(looppd, npartitions=min(replicates, 200))
        # %time get_maxp_1sample(34)
        def _gwas_pval(df, rmv_aft):
            ret = df['reps'].map(lambda x: get_maxp_1sample(x,  skip_already_present = True, remove_after = rmv_aft)) # skip_already_present = True,remove_after= rmv_aft
            if len(df) != len(ret): print(ret)
            return ret
        _ = loop.map_partitions(lambda x: _gwas_pval(x, rmv_aft = remove_after), meta = pd.Series())
        printwithlog(f'running gwas for {replicates} replicates')
        future = cline.compute(_)
        progress(future,notebook = False,  interval="300s") #, group_by="spans"
        wait(future)
        out = looppd.assign(maxp = future.result())
        for tf in [True, False]:
            maxrange = 2000 if tf else len(out)
            lis = pd.concat([out.sample(n = x, replace = tf)['maxp'].describe(percentiles=[.1, .05, 0.01, 1e-3, 1e-4]).abs().to_frame().rename({'min': x}, axis = 1) \
                   for x in np.linspace(1, maxrange, 200).round().astype(int)], axis = 1)
            lis = lis.T.reset_index(drop = True).rename({'count': 'samplesize'}, axis = 1)
            lis = lis.drop(['mean', 'min', 'max'], axis = 1).fillna(0)
            melted = lis.melt(id_vars=['samplesize'], value_vars=lis.columns[1:], value_name='pval')
            lis.to_csv(f'{self.path}pvalthresh/maxpvaltable{"with" if tf else "without"}replacement.csv', index = False)
            fig = sns.lmplot(x="samplesize", y="pval",
                 hue="variable",  data=melted.query('variable != "std"'),logx= True,height = 5, aspect=2 )
            fig.savefig(f'{self.path}pvalthresh/threshfig{"with" if tf else "without"}replacement.png')
        oo = out['maxp'].describe(percentiles=[.1, .05, 0.01, 1e-3, 1e-4]).to_frame().set_axis(['thresholds'], axis =1).abs()
        oo.to_csv(f'{self.path}pvalthresh/PVALTHRESHOLD.csv')
        display(oo)
        printwithlog(f"new_thresholds = 5% : {oo.loc['5%','thresholds']} , 10% : {oo.loc['10%','thresholds']}")
        self.threshold = oo.loc['10%','thresholds']
        self.threshold05 = oo.loc['5%','thresholds']
        return oo   
        
        
    def callQTLs(self, window: int = 0.5e6, subterm: int = 4,  add_founder_genotypes: bool = True, save = True, displayqtl = True, annotate = True,
                 ldwin = int(12e6), ldkb = 12000, ldr2 = .8, qtl_dist = 12*1e6, NonStrictSearchDir = True, conditional_analysis = True, **kwards): # annotate_genome: str = 'rn7',
        
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
        printwithlog(f'starting call qtl ... {self.project_name}') 
        thresh = 10**(-self.threshold)
        if type(NonStrictSearchDir) == type(pd.DataFrame()):
            topSNPs = NonStrictSearchDir
            save = False
        
        elif not NonStrictSearchDir:
            topSNPslist = []
            # topSNPs = pd.DataFrame()
            for t, chrom in tqdm(list(itertools.product(self.traits, range(1,self.n_autosome+4)))):
                    chrom = self.replacenumstoXYMT(chrom)
                    filename = f'{self.path}results/gwas/{t}_chrgwas{chrom}.mlma'
                    if os.path.exists(filename):
                        #topSNPs = pd.concat([topSNPs, pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)])
                        topSNPslist += [pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)]
                    else: pass #print(f'could not locate {filename}')
            for t in tqdm(self.traits):
                    filename = f'{self.path}results/gwas/{t}.loco.mlma'
                    if os.path.exists(filename):
                        #topSNPs = pd.concat([topSNPs, pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)])
                        topSNPslist += [pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)]
                    else: pass
                        #printwithlog(f'could not locate {filename}')
            topSNPs = pd.concat(topSNPslist)#.drop_duplicates(subset= ['SNP', 'trait'], keep = 'first')
                    

        else:
            topSNPs = pd.concat([pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=re.findall('/([^/]*).mlma', 
                                                                                            filename)[0].replace('.loco', '').split('_chrgwas')[0]) for
                                 filename in tqdm(glob(f"{self.path}results/gwas/*.mlma"))])#.drop_duplicates(subset= ['SNP', 'trait'], keep = 'first')

        out = pd.DataFrame()

        for (t, c), df in tqdm(topSNPs.groupby(['trait','Chr'])):
            df = df.set_index('bp').sort_index()
            df.p = -np.log10(df.p)

            while df.query(f'p > {self.threshold}').shape[0]:
                idx = df.p.idxmax()
                maxp = df.loc[idx]
                correlated_snps = df.loc[int(idx - window//2): int(idx + window//2)].query('p > @maxp.p - @subterm').query('p > 4')
                qtl = True if correlated_snps.shape[0] > 2 else False
                try: 
                    r2temp = self.plink(bfile = self.genotypes_subset, chr = c, ld_snp = maxp.SNP, 
                                   ld_window_r2 = ldr2, r2 = 'dprime', ld_window = 100000, thread_num = int(self.threadnum),
                                   ld_window_kb =  12000, nonfounders = '')#.query('R2 > @ldr2')
                    ldSNPS = r2temp.SNP_B.to_list() + [maxp.SNP]
                    ldSNPS_LEN = r2temp.BP_B.agg(lambda x: (x.max()-x.min())/1e6)
                    df = df.query('~(@idx - @qtl_dist//2 < index < @idx + @qtl_dist//2) and (SNP not in @ldSNPS)')
                except:
                    printwithlog('could not run plink...')
                    ldSNPS = [maxp.SNP]
                    ldSNPS_LEN = 0
                    df = df.query('(SNP not in @ldSNPS)')
                            
                out = pd.concat([out,
                                 maxp.to_frame().T.assign(QTL= qtl, interval_size = '{:.2f} Mb'.format(ldSNPS_LEN))],
                                 axis = 0)
                
        if not len(out):
            printwithlog('no SNPS were found, returning an empty dataframe')
            if save: out.to_csv(f'{self.path}results/qtls/allQTLS.csv', index = False)
            return out
            
        out =  out.sort_values('trait').reset_index(names = 'bp')#.assign(project = self.project_name)
        out['trait_description'] = out.trait.apply(lambda x: self.get_trait_descriptions[x])
        out['trait'] = out.trait.apply(lambda x: x.replace('regressedlr_', ''))
        self.allqtlspath = f'{self.path}results/qtls/allQTLS.csv'
        if save: out.to_csv(self.allqtlspath.replace('allQTLS', 'QTLSb4CondAnalysis'), index = False)
        self.pbim, self.pfam, self.pgen = pandas_plink.read_plink(self.genotypes_subset)
        if displayqtl: display(out)
        if save: qtlb4cond = pd.read_csv(f'{self.path}results/qtls/QTLSb4CondAnalysis.csv').query('QTL').reset_index(drop=True)
        else: qtlb4cond = out.query('QTL').reset_index(drop=True)
        if conditional_analysis:    
            printwithlog('running conditional analysis...')
            if conditional_analysis == 'parallel': 
                printwithlog('running parallel conditional analysis...')
                out = self.conditional_analysis_filter_chain_parallel(qtlb4cond)
            else: out = self.conditional_analysis_filter_chain(qtlb4cond)
        else: pass
        if add_founder_genotypes and len(self.foundersbimfambed):
            dff_idx = self.foundersbimfambed[0].query('snp in @out.SNP').reset_index(drop = True)
            dffounders = pd.DataFrame(self.foundersbimfambed[2][dff_idx.i.values], columns = self.foundersbimfambed[1].iid.values)
            dffounders = dffounders.assign(SNP=dff_idx.snp,v0 = dff_idx.a0 +' '+dff_idx.a0,  v1 = dff_idx.a0 +' ' + dff_idx.a1,v2 = dff_idx.a1 +' ' + dff_idx.a1)
            dffounders = dffounders.set_index('SNP').apply(lambda x: x[:-3].map(lambda y: x[-3:][int(y)]), axis = 1).reset_index()
            out = out.merge(dffounders, on = 'SNP', how = 'left')
        out['significance_level'] = out.p.apply(lambda x: '5%' if x >= self.threshold05 else '10%')
        if save: out.to_csv(self.allqtlspath, index = False)
        if annotate: 
            out = self.annotatevep(out, save = True)
            out = out.reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
            if save: out.to_csv(f'{self.path}results/qtls/finalqtl.csv', index= False)
        return out.set_index('SNP') 
    
    def conditional_analysis(self, trait: str, snpdf: pd.DataFrame() = pd.DataFrame()):
        os.makedirs(f'{self.path}results/cojo', exist_ok=True)
        os.makedirs(f'{self.path}temp/cojo',exist_ok=True)
        
        if not snpdf.shape[0]: printwithlog(f'running conditional analysis for trait {trait} and all snps above threshold {self.threshold}')
        else: 
            #printwithlog(snpdf.shape)
            snpstring = ' '.join(snpdf.SNP)
            printwithlog(f'running conditional analysis for trait {trait} and all snps below threshold {snpstring}')

        pbimtemp = self.pbim.assign(n = self.df.count()[trait]).rename({'snp': 'SNP', 'n':'N'}, axis = 1)[['SNP', 'N']] #- da.isnan(pgen).sum(axis = 1)
        tempdf = pd.concat([pd.read_csv(f'{self.path}results/gwas/{trait}.loco.mlma', sep = '\t'),
                           pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasx.mlma', sep = '\t'),
                           pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasy.mlma', sep = '\t'),
                           pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasmt.mlma', sep = '\t')]).rename({'Freq': 'freq'}, axis =1 )
        tempdf = tempdf.merge(pbimtemp, on = 'SNP')[['SNP','A1','A2','freq','b','se','p','N' ]]
        mafile, snpl = f'{self.path}temp/cojo/tempmlma.ma', f'{self.path}temp/cojo/temp.snplist'
        tempdf.to_csv(mafile, index = False, sep = '\t')
        tempdf[-np.log10(tempdf.p) >self.threshold][['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
        mafile, snpl = f'{self.path}temp/cojo/tempmlma.ma', f'{self.path}temp/cojo/temp.snplist'
        tempdf.to_csv(mafile, index = False, sep = '\t')
        if not snpdf.shape[0]:
            tempdf[-np.log10(tempdf.p) > self.threshold][['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
        else: snpdf[['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
        cojofile = f'{self.path}temp/cojo/tempcojo'
        self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --cojo-slct --cojo-collinear 0.99 --cojo-p {10**-(self.threshold-2)} --cojo-file {mafile} --cojo-cond {snpl} --out {cojofile}', f'cojo_test', print_call=False)
        if os.path.isfile(f'{cojofile}.jma.cojo'):
            return pd.read_csv(f'{cojofile}.jma.cojo', sep = '\t')
        printwithlog(f'Conditional Analysis Failed for  trait {trait} and all snps below threshold {snpstring}, returning the top snp only')
        return pd.DataFrame(snpdf.SNP.values, columns = ['SNP'])

    def conditional_analysis_filter(self, qtltable):
        return qtltable.groupby(['Chr', 'trait']).progress_apply(lambda df: df.loc[df.SNP.isin(self.conditional_analysis('regressedlr_' +df.name[1].replace('regressedlr_', ''), df).SNP.to_list())]
                                                            if df.shape[0] > 1 else df).reset_index(drop= True)
    
    def conditional_analysis_chain_singletrait(self, snpdf: pd.DataFrame(), print_call: bool = False, nthread: str = ''):
        #bim, fam, gen = pandas_plink.read_plink(self.genotypes_subset)
        trait = snpdf.iloc[0]['trait']#.mode()[0]
        c = snpdf.iloc[0]['Chr']#snpdf['Chr'].mode()[0]
        bim = self.pbim.set_index('snp')
        snpdf2 = snpdf.copy()
        printwithlog(f'running conditional analysis for trait {trait} and chromosome {c}...')

        os.makedirs(f'{self.path}results/cojo', exist_ok=True)
        os.makedirs(f'{self.path}temp/cojo',exist_ok=True)
        covarlist = snpdf2.nlargest(1, 'p')
        threadedflag = f'--thread-num {nthread}' if nthread else self.thrflag
        nthreadsr2 = int(nthread) if nthread else int(self.threadnum)

        #### getting only nearby snps to reduce time 
        subset_snp_path = f'{self.path}temp/cojo/accepted_snps_{c}_{trait}'
        genotypesCA = f'{self.path}temp/cojo/genotypesCA_{c}_{trait}'
        # snps_of_int = pd.concat([bim.query(f'chrom == "{c}"').query(f'{pos}-4e6<pos<{pos}+4e6') for pos in snpdf2.bp]).reset_index().drop_duplicates(subset = ['snp'])
        # snps_of_int[['snp']].to_csv(subset_snp_path,index = False, header = None)l

        mlmares = pd.read_csv(f'{self.path}results/gwas/regressedlr_{trait}_chrgwas{self.replacenumstoXYMT(c)}.mlma', sep = '\t')
        snps_of_int = pd.concat([mlmares.query('p<1e-4').query(f'{pos}-8e6<bp<{pos}+8e6') for pos in snpdf2.bp]).drop_duplicates(subset = ['SNP'])
        snps_of_int[['SNP']].to_csv(subset_snp_path,index = False, header = None)

        plink(bfile = self.genotypes_subset, extract = subset_snp_path, make_bed = '', thread_num =  self.threadnum,
              set_missing_var_ids = '@:#', keep_allele_order = '',  set_hh_missing = '' , make_founders = '', 
              out = genotypesCA, chr_set = f'{self.n_autosome} no-xy') #
        
        
        for num in tqdm(range(snpdf.shape[0])):
            ##### make covarriates dataframe
            covar_snps = covarlist.SNP.to_list()
            geni = bim.loc[covar_snps, 'i'].to_list()
            covardf = pd.DataFrame(self.pgen[geni, :].T, columns =covar_snps , index = self.pfam['iid'])
            traitdf = pd.read_csv(f'{self.path}data/pheno/regressedlr_{trait}.txt', sep = '\s+',  header = None, index_col=0, dtype = {0: str})
            covardf = covardf.loc[traitdf.index, :]
            covardf = covardf.reset_index(names= 'rfid').set_axis(covardf.index).reset_index().fillna('NA')
            covarname = f'{self.path}temp/cojo/covardf.txt'
            covardf.to_csv(covarname, index = False, header = None, sep = '\t')

            ##### run mlma
            chromp2 = self.replacenumstoXYMT(c)
            subgrmflag = f'--mlma-subtract-grm {self.path}grm/{chromp2}chrGRM' if chromp2 not in ['x','y', 'mt'] else ''
            self.bashLog(f'{self.gcta} {threadedflag} --pheno {self.path}data/pheno/regressedlr_{trait}.txt --bfile {genotypesCA} \
                                       --grm {self.path}grm/AllchrGRM --autosome-num {self.n_autosome} --reml-maxit 1000 \
                                       --chr {c} {subgrmflag} --mlma --covar {covarname}\
                                       --out {self.path}temp/cojo/cojo_temp_gwas{chromp2}{trait}', 
                        f'GWAS_cojo_{c}_{trait}', print_call = print_call) #self.genotypes_subset

            ##### append top snp from run
            if not os.path.isfile(f'{self.path}temp/cojo/cojo_temp_gwas{chromp2}{trait}.mlma'):
                printwithlog(F'Conditional analysis error, early stopping the conditional analysis for {self.path}temp/cojo/cojo_temp_gwas{chromp2}{trait}.mlma returning at this stop')
                return covarlist
            mlmares = pd.read_csv(f'{self.path}temp/cojo/cojo_temp_gwas{chromp2}{trait}.mlma', sep = '\t')
            os.system(f'rm {self.path}temp/cojo/cojo_temp_gwas{chromp2}{trait}.mlma')
            add2ingsnp =  mlmares[~mlmares.SNP.isin(covar_snps)].nsmallest(1, 'p')
            add2ingsnp['p'] = -np.log10(add2ingsnp['p'])
            if add2ingsnp.iloc[0]['p'] < self.threshold:
                return covarlist
            if (aa := snpdf[snpdf.SNP.isin(add2ingsnp.SNP.values)]).shape[0] > 0:
                covarlist = pd.concat([covarlist, aa])
            else:
                r2lis = self.plink(bfile = self.genotypes_subset, chr = add2ingsnp.iloc[0].Chr, ld_snp = add2ingsnp.iloc[0].SNP, 
                                   ld_window_r2 = 0.6, r2 = 'dprime', ld_window = 100000, thread_num = nthreadsr2,
                                   ld_window_kb =  12000, nonfounders = '')
                intervalsize = r2lis['BP_B'].astype(int).agg(lambda x: (x.max()-x.min())/1e6)
                add2ingsnp = add2ingsnp.assign(interval_size =  '{:.2f} Mb'.format(intervalsize), 
                                               emergent_qtl = True, trait = trait, QTL = True,
                                               trait_description = snpdf['trait_description'].fillna('UNK').mode()[0])
                covarlist = pd.concat([covarlist,add2ingsnp ])
        return covarlist

    def conditional_analysis_filter_chain(self, qtltable: pd.DataFrame()):
        self.pbim, self.pfam, self.pgen = pandas_plink.read_plink(self.genotypes_subset)
        return qtltable.groupby(['Chr', 'trait'], group_keys=False)\
                       .progress_apply(lambda df: self.conditional_analysis_chain_singletrait( snpdf = df)
                                                 if df.shape[0] > 1 else df).reset_index(drop= True)

    def conditional_analysis_filter_chain_parallel(self, qtltable: pd.DataFrame()):
        self.pbim, self.pfam, self.pgen = pandas_plink.read_plink(self.genotypes_subset)
        cline = y if (y:=client._get_global_client()) else Client(processes = False) 
        ddqtl = dd.from_pandas(qtltable, npartitions=200)
        ddgb = ddqtl.groupby(['Chr', 'trait'], group_keys=False)
        threadnum = max(self.threadnum//ddqtl.npartitions,1)
        ddgb = ddqtl.groupby(['Chr', 'trait'], group_keys=False)
        _ = ddgb.apply(lambda df: self.conditional_analysis_chain_singletrait(snpdf = df, print_call =  False, nthread = threadnum)  if df.shape[0] > 1 else df
                       , meta = qtltable).reset_index(drop= True)#.compute()
        future = cline.compute(_)
        progress(future,notebook = False,  interval="300s")
        out = future.result().sort_values(['trait', 'Chr', 'bp'])
        del future
        return out

    def effectsize(self, qtltable: pd.DataFrame() = None, display_plots: bool = True):
        printwithlog(f'starting effect size plot... {self.project_name}') 
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)):
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')
        out = qtltable.reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        usnps = out.SNP.unique()
        aa = ','.join(usnps)
        self.bashLog(f'plink --bfile {self.genotypes_subset} --snps {aa} --recode --alleleACGT --out {self.path}temp/stripplot', 'stripplot', print_call=False)
        snporder = list(pd.read_csv(f'{self.path}temp/stripplot.map', sep = '\s+', header = None)[1])
        temp = pd.read_csv(f'{self.path}temp/stripplot.ped', sep = '\s+', index_col = [0,1,2,3,4,5], header = None)
        temp = temp.iloc[:, 1::2].set_axis(snporder, axis = 1) + temp.iloc[:, 0::2].set_axis(snporder, axis = 1)
        temp = temp.reset_index().drop([1,2,3,4,5], axis = 1).rename({0:'rfid'},axis = 1)
        temp.rfid = temp.rfid.astype(str)
        temp = temp.merge(self.df[self.traits + 
                          [t.replace('regressedlr_', '') for t in self.traits] + 
                          ['rfid', 'sex']].rename(lambda x: x.replace('regressedlr_', 'normalized '), axis =1), on = 'rfid')
        for name, row in tqdm(list(out.iterrows())):
            f, ax = plt.subplots(1, 2, figsize = (12, 6), dpi=400)
            if len(temp[f'normalized {row.trait}'].dropna().unique()) <= 4:
                for num, ex in enumerate(['normalized ', '']):
                    sns.countplot(temp[temp[row.SNP]!= '00'].sort_values(row.SNP), x = row.SNP, hue = ex+ row.trait, ax = ax[num], orient='Vertical')
                    ax[num].legend().set_title('')
                    ax[num].set_ylabel(ex+ row.trait)
            else:    
                for num, ex in enumerate(['normalized ', '']):
                    sns.boxplot(temp[temp[row.SNP]!= '00'].sort_values(row.SNP), x = row.SNP, y = ex+ row.trait, color = 'steelblue', ax = ax[num] )#cut = 0,  bw= .2, hue="sex", split=True
                    sns.stripplot(temp[temp[row.SNP]!= '00'].sort_values(row.SNP), x = row.SNP,  y =ex+ row.trait, color = 'black', jitter = .2, alpha = .4, ax = ax[num] )
                    ax[num].hlines(y = 0 if num==0 else temp[temp[row.SNP]!= '00'][ex+ row.trait].mean() , xmin =-.5, xmax=2.5, color = 'black', linewidth = 2, linestyle = '--')
            sns.despine()
            os.makedirs(f'{self.path}images/boxplot/', exist_ok=True)
            plt.tight_layout()
            plt.savefig(f'{self.path}images/boxplot/boxplot{row.SNP}__{row.trait}.png'.replace(':', '_'))
            plt.show()
            plt.close()   

    def get_r2_around_topsnp(self, qtltable:pd.DataFrame() = '', save = False, qtl_r2_thresh: float = .65):
        printwithlog(f'locuszoom: calculating r2 for nearby snps {self.project_name}')
        if isinstance(qtltable, str) and (qtltable == ''): 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv').reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        causal_snps = []
        for name, row in tqdm(list(qtltable.iterrows())):
            ldfilename = f'{self.path}results/lz/temp_qtl_n_@{row.trait}@{row.SNP}'
            r2 = self.plink(bfile = self.genotypes_subset, chr = row.Chr, ld_snp = row.SNP, ld_window_r2 = 0.00001, r2 = 'dprime',\
                            ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb =  7000, nonfounders = '').loc[:, ['SNP_B', 'R2', 'DP']] 
            gwas = pd.concat([pd.read_csv(x, sep = '\t') for x in glob(f'{self.path}results/gwas/regressedlr_{row.trait}.loco.mlma') \
                                + glob(f'{self.path}results/gwas/regressedlr_{row.trait}_chrgwas*.mlma')]).drop_duplicates(subset = 'SNP')
            tempdf = pd.concat([gwas.set_index('SNP'), r2.rename({'SNP_B': 'SNP'}, axis = 1).drop_duplicates(subset = 'SNP').set_index('SNP')], join = 'inner', axis = 1)
            tempdf = self.annotatevep(tempdf.reset_index(), 'SNP', save = False).set_index('SNP').fillna('UNK')
            tempdf.to_csv( f'{self.path}results/lz/lzplottable@{row.trait}@{row.SNP}.tsv', sep = '\t')
            subcausal = tempdf.query("putative_impact not in ['UNK', 'MODIFIER']").assign(trait = row.trait, SNP_qtl = row.SNP)
            subcausal.columns = subcausal.columns.astype(str)
            if subcausal.shape[0] > 0:
                subcausal = subcausal.loc[subcausal.R2 > qtl_r2_thresh,  ~subcausal.columns.str.contains('\d\d', regex = True) ]\
                                     .sort_values('putative_impact', ascending = False).drop('errors', errors = 'ignore' , axis =1 ).reset_index()\
                                     .drop(['Chr', 'bp', 'se', 'geneid'], errors = 'ignore' ,axis = 1).reset_index().set_index('SNP_qtl')
                causal_snps += [subcausal]
        if len(causal_snps) and save: 
            causal_snps = pd.concat(causal_snps).to_csv(f'{self.path}results/qtls/possible_causal_snps.tsv', sep = '\t')
    
    def locuszoom2(self, qtltable:pd.DataFrame() = None, qtl_r2_thresh: float = .65,  padding: float = 2e5, save = True, 
                   skip_ld_calculation=False, save_causal_table=True, credible_set_threshold = .99, topaxaxis = True):
        printwithlog(f'starting locuszoom... {self.project_name}') 
        from bokeh.models import NumeralTickFormatter
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                         .reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        if not skip_ld_calculation:
            self.get_r2_around_topsnp(save = save_causal_table, qtl_r2_thresh= qtl_r2_thresh)
        ff_lis = []
        res = {'r2thresh': [], 'minmax': []}
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        #gtf = self.gtf.query("biotype == 'transcript' and gene != ''")
        printwithlog(f'locuszoom: starting plots {self.project_name}')
        for ((idx, topsnp),boundary) in tqdm(list(itertools.product(qtltable.iterrows(), ['r2thresh','minmax']))):
            if not os.path.isfile(f'{self.path}results/lz/lzplottable@{topsnp.trait}@{topsnp.SNP}.tsv'):
                self.get_r2_around_topsnp(save = save_causal_table, qtl_r2_thresh= qtl_r2_thresh)
            data = pd.read_csv(f'{self.path}results/lz/lzplottable@{topsnp.trait}@{topsnp.SNP}.tsv', sep = '\t').query('p != "UNK"')
            data['-log10(P)'] = -np.log10(pd.to_numeric(data.p, errors = 'coerce')) 
            if boundary =="r2thresh":
                minval, maxval = data.query(f'R2 > {qtl_r2_thresh}').agg({'bp': [min, max]}).values.flatten() + np.array([-padding, padding])
            else:  minval, maxval = (np.array([-3e6, 3e6]) + topsnp.bp).astype(int)
            genes_in_section = self.gtf.query(f'Chr == {topsnp.Chr} and end > {minval} and start < {maxval}')\
                                       .reset_index(drop = True)\
                                       .query("source not in ['cmsearch','tRNAscan-SE']")
            if boundary =="r2thresh":
                ff_lis += [genes_in_section.query('gbkey == "Gene"').sort_values('gene').assign(SNP_origin = topsnp.SNP).drop_duplicates(subset='gene')]
            ngenes = len(genes_in_section.gene.unique())
            causal = pd.read_csv(f'{self.path}results/qtls/possible_causal_snps.tsv' , sep = '\t')\
                       .query(f'SNP_qtl == "{topsnp.SNP}"')
            causal['bp'] = causal.SNP.str.extract(':(\d+)').astype(int)
            causal = causal.merge(data[['SNP', '-log10(P)']], on='SNP')
            
            phewas = pd.read_csv(f'{self.path}results/phewas/pretty_table_both_match.tsv' , sep = '\t')\
                       .query(f'SNP_QTL == "{topsnp.SNP}"')
            phewas['bp'] = phewas.SNP_PheDb.str.extract(':(\d+)').astype(float)
            phewas = phewas.merge(data[['SNP', '-log10(P)']], left_on='SNP_PheDb', right_on='SNP')
            phewas['R2'] = phewas['R2'].map(lambda x: 1 if x == 'Exact match SNP' else float(x))
            phewas['phewas_file'] = phewas['phewas_file'].map(basename)
            
            eqtl = pd.read_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv' )
            eqtl['SNP'] = eqtl['SNP'].str.replace('chr', '')#\
            eqtl = eqtl.query(f'SNP == "{topsnp.SNP}"')
            eqtl['bp'] = eqtl.SNP_eqtldb.str.extract(':(\d+)').astype(float)
            
            eqtl = eqtl.merge(data[['SNP', '-log10(P)']], left_on='SNP_eqtldb', right_on='SNP')
            sqtl = pd.read_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv' )
            sqtl['SNP'] = sqtl['SNP'].str.replace('chr', '')#\
            sqtl = sqtl.query(f'SNP == "{topsnp.SNP}"')
            sqtl['bp'] = sqtl.SNP_sqtldb.str.extract(':(\d+)').astype(float)
            sqtl = sqtl.merge(data[['SNP', '-log10(P)']], left_on='SNP_sqtldb', right_on='SNP')
            tsf = topsnp.to_frame().T.rename({'p': '-log10(P)'}, axis = 1)
            
            genecolors = {'transcript': 'black', 'exon': 'seagreen', 'CDS': 'steelblue','start_codon': 'firebrick' ,'stop_codon': 'firebrick'}
            genesizes = {'transcript': .05, 'exon': .25, 'CDS': .35,'start_codon': .45 ,'stop_codon': .45}
            tt = genes_in_section.dropna(how = 'all', axis = 1).reset_index(drop= True).assign(y=1)
            tt[['start', 'end']] = tt[['start', 'end']]
            tt['mean_pos'] = tt[['start', 'end']].mean(axis = 1)
            tt = tt.assign(color = tt.biotype.map(genecolors), size = tt.biotype.map(genesizes), genestrand = tt.gene + tt.strand)
            ttgenes = tt.query('gbkey == "Gene"').reset_index(drop = True)#.set_index('end')
            spacing_genes = (maxval - minval)/(20 if len(ttgenes)<= 15 else 60)
            for idx, row in ttgenes.iterrows():
                if idx == 0: ttgenes.loc[idx, 'stackingy'] = 0
                genes_inreg = set(ttgenes[min(idx-1000, 0): idx].query(f"end + {spacing_genes} > @row.start").stackingy)
                if idx>0: ttgenes.loc[idx, 'stackingy'] = min(set(range(1000)) - genes_inreg)
            yticks3 = (1.1*ttgenes[['stackingy','stackingy']]).round(2).agg(lambda x: tuple(x), axis = 1).to_list()
            yticks3l = [(x[0], '') for x in yticks3]
            size_scalers = ttgenes['stackingy'].max()*1.1
            stackgenepos = defaultdict(lambda: -1, (ttgenes.set_index('gene')['stackingy']*1.1).to_dict())
            tt = tt.assign(stackingy = tt.gene.map(stackgenepos), 
                           stackingy0 = tt.gene.map(stackgenepos) - tt['size']/5, #*size_scalers/10/2
                           stackingy1 = tt.gene.map(stackgenepos) + tt['size']/5)
            
            kw_table = {'data': dict(color='R2', cmap='Spectral_r', size = 20,alpha = .7, clim= (0,1),\
                                    colorbar=True, line_width = .4, padding=0.05, xlabel = ''),
                        'eqtl': dict(color='R2', cmap='Blues', size = 25,alpha = .7, clim= (0,1), marker = 'inverted_triangle',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = ''),
                        'sqtl': dict(color='R2', cmap='Blues', size = 25,alpha = .7, clim= (0,1), marker = 'triangle',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = ''),
                        'phewas': dict(color='R2', cmap='Greys', size = 20,alpha = .7, clim= (0,1), marker = 'square',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = ''),
                        'causal': dict(color='R2', cmap='Oranges', size = 25,alpha = .7, clim= (0,1), marker = 'diamond',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = ''),
                        'tsf': dict(color = 'red', size = 35,alpha = 1, clim= (0,1), marker = 'star',tools = ['hover'],
                                    line_width = 1, padding=0.05, xlabel = '')}
            kw_tabled = {'data': data,'eqtl': eqtl,'sqtl': sqtl,
                        'phewas': phewas,'causal': causal, 'tsf': tsf}
            fig = reduce(lambda x,y: x*y, 
                   [hv.Points(kw_tabled[x].query(f'{minval}<bp<{maxval}').sort_values('-log10(P)'), kdims = ['bp', '-log10(P)'])\
                    .opts(**kw_table[x],width=1200, height=400,line_color='Black') \
                    for x in kw_table.keys()])
            fig = fig*hv.HLine(self.threshold).opts(color='red')
            fig = fig*hv.HLine((self.threshold05)).opts(color='blue')
            nrows_genes = int((np.array(yticks3)/1.1).max())
            hgenelabels = 300 + nrows_genes*70
            rect = hv.Rectangles(tt.sort_values('size', ascending = False).rename({'start':'bp'}, axis = 1), kdims=['bp', 'stackingy0', 'end', 'stackingy1'], 
                                 vdims=['color', 'gene'])\
                     .opts(xformatter=NumeralTickFormatter(format='0,0.[0000]a') ,xticks=5,xrotation = 0, width = 1200, height = hgenelabels,
                           tools = ['hover'],  invert_yaxis=True,
                          ylim= (-.5, max(np.array(yticks3).max()+1, 1.1*1.5)), xlim = ( minval, maxval)) 
            rect = rect.opts(hv.opts.Rectangles(fill_color='color', line_color='color', line_width=2, fill_alpha=0.3)).opts(ylabel = 'genes', yticks =yticks3l)# yticks =yticks3l, 
            fig = fig.opts(show_grid= True, gridstyle={'grid_line_color': 'black', 'grid_line_width': .0, 'xgrid_line_dash': [4, 4], 'ygrid_line_dash': [4, 4]},xlim = ( minval, maxval))
            rect = rect.opts(show_grid= True, gridstyle={'grid_line_color': 'black', 'grid_line_width': .0, 'xgrid_line_dash': [4, 4],'ygrid_line_dash': [4, 4]}) #, '
            
            fullgene = tt.query('gbkey == "Gene"')
            fullgene = fullgene.assign(ymax = fullgene.stackingy + max(genesizes.values())/5 ) #size_scalers/10/2
            labels = hv.Labels(fullgene, kdims = ['mean_pos','ymax'], vdims=['genestrand'])\
                       .opts( opts.Labels(text_color='Black',  text_font_size='15px' if (nrows_genes > 6 or len(ttgenes)> 15) else '30px',yoffset=+.35,
                                          angle=45 if (nrows_genes > 6 or len(ttgenes)> 15) else 0 ,width = 1200, height = hgenelabels) )
            fontsizes =  {'title': 40,  'labels': 25,   'xticks': 15,  'yticks': 15 }
            rect = rect*labels
            fig = fig.opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'),ylim = (-.1, max(6, topsnp.p*1.1)), 
                           xticks=5, xrotation = 0, width = 1200, height = 400, fontsize = fontsizes)
            if topaxaxis:
                fig = fig.opts(opts.Points( xaxis='top', xticks='top'))
            rect = rect.opts(  fontsize = fontsizes, yticks =yticks3l ,xlabel = f'Chr {topsnp.SNP.split(":")[0]}')
            if credible_set_threshold:
                datacs = data.query(f'({minval}<bp<{maxval})').sort_values('-log10(P)').reset_index(drop=True)
                datacs['ppi'] = bayes_ppi(datacs.p)
                datacs['ppi_s'] = MinMaxScaler(feature_range=(2, 10)).fit_transform(datacs.ppi.values.reshape(-1,1))
                cs = credible_set_idx(datacs.ppi.values,cs_threshold=credible_set_threshold)
                datacs = datacs.assign(credible_set = 0)
                datacs.loc[cs, 'credible_set'] = 1
                ff1 = hv.Points(datacs, kdims = ['bp', 'credible_set'] , vdims =['R2', 'ppi_s'] )\
                       .opts(size = 'ppi_s',marker = 'circle', color = 'R2', cmap = 'Spectral_r', width=1200, 
                             height=80,alpha = .7, clim= (0,1),yticks =[(0,'not included'), (1, 'included')], ylim = (-.5, 1.5),  xaxis=None)
                fig = fig+ff1
            finalfig = (fig + rect ).cols(1).opts(title = f'locuszoom Chr{topsnp.Chr} {topsnp.trait} {topsnp.SNP}')
            if save: 
                os.makedirs(f'{self.path}images/lz/{boundary}', exist_ok = True)
                hv.save(finalfig, f"{self.path}images/lz/{boundary}/lzi__{topsnp.trait}__{topsnp.SNP}.png".replace(':', '_'))
                hv.save(finalfig, f"{self.path}images/lz/{boundary}/lzi__{topsnp.trait}__{topsnp.SNP}.html".replace(':', '_'))
            res[boundary] += [finalfig]
        ff_lis = pd.concat(ff_lis)
        ff_lis['webpage'] = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene=' + ff_lis['gene']
        ff_lis['markdown'] = ff_lis.apply(lambda x: f'[{x.gene}]({x.webpage})', axis = 1)
        if save: 
            ff_lis.dropna(axis = 1, how = 'all').to_csv(f'{self.path}results/qtls/genes_in_range.csv', index = False)
            genes_in_range2 = self.make_genes_in_range_mk_table()
            genes_in_range2.to_csv(f'{self.path}results/qtls/genes_in_rangemk.csv')
        return res
    
    def locuszoom(self, qtltable: pd.DataFrame() = None,  qtl_r2_thresh: float = .65, save = True,
                  padding: float = 1e5, skip_ld_calculation = False, make_interactive = True, make_classic = True, print_call = False):
        '''
        Only works on TSCC
        '''
        printwithlog(f'generating locuszoom info for project {self.project_name}')
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)):
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')
        out = qtltable.reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        causal_snps = []
        
        os.makedirs(f'{self.path}results/lz/p/', exist_ok = True)
        os.makedirs(f'{self.path}results/lz/r2/', exist_ok = True)
        os.makedirs(f'{self.path}images/lz/', exist_ok = True)
        
        # linkdict = {'rn7':f'http://hgdownload.soe.ucsc.edu/goldenPath/rn7/bigZips/genes/ncbiRefSeq.gtf.gz' , 
        #     'rn6':'https://hgdownload.soe.ucsc.edu/goldenPath/rn6/bigZips/genes/rn6.ncbiRefSeq.gtf.gz',
        #    'm38': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/genes/mm10.ncbiRefSeq.gtf.gz'}
        
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        gtf = gtf.query("biotype == 'transcript' and gene != ''")
        if not skip_ld_calculation:
            printwithlog(f'locuszoom: calculating r2 for nearby snps {self.project_name}')
            for name, row in tqdm(list(out.iterrows())):
                ldfilename = f'{self.path}results/lz/temp_qtl_n_@{row.trait}@{row.SNP}'
                r2 = self.plink(bfile = self.genotypes_subset, chr = row.Chr, ld_snp = row.SNP, ld_window_r2 = 0.00001, r2 = 'dprime',\
                                        ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb =  7000, nonfounders = '').loc[:, ['SNP_B', 'R2', 'DP']] 
                gwas = pd.concat([pd.read_csv(x, sep = '\t') for x in glob(f'{self.path}results/gwas/regressedlr_{row.trait}.loco.mlma') \
                                    + glob(f'{self.path}results/gwas/regressedlr_{row.trait}_chrgwas*.mlma')]).drop_duplicates(subset = 'SNP')
                #+ glob(f'{self.path}results/gwas/regressedlr_{row.trait}_chrgwas\d+.mlma')
                tempdf = pd.concat([gwas.set_index('SNP'), r2.rename({'SNP_B': 'SNP'}, axis = 1).drop_duplicates(subset = 'SNP').set_index('SNP')], join = 'inner', axis = 1)
                #display(tempdf)
                tempdf = self.annotatevep(tempdf.reset_index(), 'SNP', save = False).set_index('SNP').fillna('UNK')
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

            if len(causal_snps): causal_snps = pd.concat(causal_snps).to_csv(f'{self.path}results/qtls/possible_causal_snps.tsv', sep = '\t')

        genome_lz_path = {'rn6': 'rn6', 
                          'rn7':'rn7', 
                          'cfw': 'm38',
                          'm38': 'm38'}[self.genome]

        ff_lis = []
        if make_interactive:
            self.locuszoom_interactive(qtltable=qtltable, qtl_r2_thresh = qtl_r2_thresh, padding=padding)

        printwithlog(f'locuszoom: running original locuszoom {self.project_name}')
        for num, (_, qtl_row) in tqdm(list(enumerate(qtltable.reset_index().iterrows()))):
            topsnpchr, topsnpbp = qtl_row.SNP.split(':')
            topsnpchr = self.replaceXYMTtonums(topsnpchr)
            try:test = pd.read_csv(f'{self.path}results/lz/lzplottable@{qtl_row.trait}@{qtl_row.SNP}.tsv', 
                               dtype = {'p':float, 'bp': int, 'R2': float, 'DP': float}, sep = '\t',
                                   na_values =  na_values_4_pandas)\
                         .replace([np.inf, -np.inf], np.nan)\
                         .dropna(how = 'any', subset = ['Freq','b','se','p','R2','DP'])
            except: raise Exception(f"couldn't open this file {self.path}results/lz/lzplottable@{qtl_row.trait}@{qtl_row.SNP}.tsv") 
            test['-log10(P)'] = -np.log10(test.p)
            range_interest = test.query(f'R2> {qtl_r2_thresh}')['bp'].agg(['min', 'max'])
            ff_lis += [gtf.query(f'(end> {range_interest["min"] - padding}) and (start< {range_interest["max"] + padding}) and (Chr == {int(topsnpchr)})').sort_values('gene').assign(SNP_origin = qtl_row.SNP).drop_duplicates(subset='gene')]#
            if make_classic:
                #test = test.query(f'{range_interest["min"] - padding}<bp<{range_interest["max"] + padding}')
                test.SNP = 'chr'+test.SNP
                
                lzpvalname, lzr2name = f'{self.path}results/lz/p/{qtl_row.trait}@{qtl_row.SNP}.tsv', f'{self.path}results/lz/r2/{qtl_row.trait}@{qtl_row.SNP}.tsv'
                test.rename({'SNP': 'MarkerName', 'p':"P-value"}, axis = 1)[['MarkerName', 'P-value']].to_csv(lzpvalname, index = False, sep = '\t')
                test.assign(snp1 = 'chr'+qtl_row.SNP).rename({"SNP": 'snp2', 'R2': 'rsquare', 'DP': 'dprime'}, axis = 1)\
                            [['snp1', 'snp2', 'rsquare', 'dprime']].to_csv(lzr2name, index = False, sep = '\t')
                os.system(f'chmod +x {lzpvalname}')
                os.system(f'chmod +x {lzr2name}')
                
                for filest in glob(f'{self.path}temp/{qtl_row.trait}*{qtl_row.SNP}'): os.system(f'rm -r {filest}')
                
                os.system(f'''conda run -n lzenv \
                    /tscc/projects/ps-palmer/software/local/src/locuszoom/bin/locuszoom \
                    --metal {lzpvalname} --ld {lzr2name} \
                    --refsnp {qtl_row.SNP} --chr {int(topsnpchr)} --start {int(range_interest["min"] - padding)} --end {int(range_interest["max"] + padding)} --build manual \
                    --db /tscc/projects/ps-palmer/gwas/databases/databases_lz/{genome_lz_path}.db \
                    --plotonly showRecomb=FALSE showAnnot=FALSE --prefix {self.path}temp/{qtl_row.trait} signifLine="{self.threshold},{self.threshold05}" signifLineColor="red,blue" \
                    title = "{qtl_row.trait} SNP {qtl_row.SNP}"  > /dev/null 2>&1 ''') #module load R && module load python && > /dev/null 2>&1 
                os.makedirs(f'{self.path}images/lz/6Mb/', exist_ok = True)
                lz12mbCall = f'''conda run -n lzenv /tscc/projects/ps-palmer/software/local/src/locuszoom/bin/locuszoom\
                    --metal {lzpvalname} --ld {lzr2name} \
                    --refsnp {qtl_row.SNP} --chr {int(topsnpchr)} --start {int(range_interest["min"] - int(3e6))} --end {int(range_interest["max"] + int(3e6))} --build manual \
                    --db /tscc/projects/ps-palmer/gwas/databases/databases_lz/{genome_lz_path}.db \
                    --plotonly showRecomb=FALSE showAnnot=FALSE --prefix {self.path}images/lz/6Mb/lz__{qtl_row.trait}_6Mb signifLine="{self.threshold},{self.threshold05}" signifLineColor="red,blue" \
                    title = "{qtl_row.trait} SNP {qtl_row.SNP} 6Mb" > /dev/null 2>&1  '''
                os.system(lz12mbCall) #module load R && module load python && {self.locuszoom_path}bin/locuszoom 
                if print_call: print(lz12mbCall)
                today_str = datetime.today().strftime('%y%m%d')
                path = glob(f'{self.path}temp/{qtl_row.trait}*{qtl_row.SNP}/*.pdf'.replace(':', '_')) + \
                       glob(f'{self.path}temp/{qtl_row.trait}_{today_str}_{qtl_row.SNP}*.pdf'.replace(':', '_'))
                if not len(path): printwithlog(f'could not find any pdf with {self.path}temp/{qtl_row.trait}*{qtl_row.SNP}/*.pdf')
                else:
                    path = path[0]
                    os.system(f'cp {path} {self.path}images/lz/lz__{qtl_row.trait}__{qtl_row.SNP}.pdf'.replace(':', '_'))
                    for num,image in enumerate(convert_from_path(path)):
                        bn = basename(path).replace('.pdf', '.png')
                        if not num: image.save(f'{self.path}images/lz/lz__{qtl_row.trait}__{qtl_row.SNP}.png'.replace(':', '_'), 'png')

        for impath in glob(f'{self.path}images/lz/6Mb/*.pdf'):
            impathout = re.sub('6Mb_\d+_', '6Mb__', impath).replace('.pdf', '.png')
            convert_from_path(impath)[0].save(impathout, 'png')
        ff_lis = pd.concat(ff_lis)
        ff_lis['webpage'] = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene=' + ff_lis['gene']
        ff_lis['markdown'] = ff_lis.apply(lambda x: f'[{x.gene}]({x.webpage})', axis = 1)
        if save: 
            ff_lis.dropna(axis = 1, how = 'all').to_csv(f'{self.path}results/qtls/genes_in_range.csv', index = False)
            genes_in_range2 = self.make_genes_in_range_mk_table()
            genes_in_range2.to_csv(f'{self.path}results/qtls/genes_in_rangemk.csv')
    
    def get_ncbi_gtf(self, extractall = False):
        printwithlog('reading gene list from NCBI RefSeq from NCBI GTF...')
        linkdict = {'rn7':'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/015/227/675/GCF_015227675.2_mRatBN7.2/GCF_015227675.2_mRatBN7.2_genomic.gtf.gz' , 
                        'rn6':'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/895/GCF_000001895.5_Rnor_6.0/GCF_000001895.5_Rnor_6.0_genomic.gtf.gz',
                       'm38': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_genomic.gtf.gz',
                      'rn8': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/annotation_releases/10116/GCF_036323735.1-RS_2024_02/GCF_036323735.1_GRCr8_genomic.gtf.gz'}
        gtf = pd.read_csv(linkdict[self.genome], sep = '\t', header = None, comment='#')\
                   .set_axis(['Chr', 'source', 'biotype', 'start', 'end', 'score', 'strand', 'phase', 'ID'], axis = 1)
        gtf = gtf[gtf.biotype != 'transcript'].reset_index(drop = True)
        #gtf['gene'] = gtf.ID.str.extract('gene_id "([^"]+)"')
        gtf['biotype'] = gtf['biotype'].str.replace('gene','transcript')
        gtfjson = pd.json_normalize(gtf['ID'].map(lambda x: {y.split(' "')[0].strip(' '): y.split(' "')[-1][:-1] for y in x.strip(';').split(';')}).to_list())
        gtf =pd.concat([gtf.drop('ID', axis = 1),gtfjson], axis = 1)
        gtf[['gene', 'gene_id']] = gtf[['gene', 'gene_id']].fillna('').astype(str)
        gtf = gtf[~gtf.gene.str.contains('-ps')]
        if self.genome == 'm38': gtf['Chr'] = gtf['Chr'].map(lambda x: translatechrmice[x]) 
        elif self.genome in ['rn6', 'rn7']:  gtf['Chr'] = gtf['Chr'].map(lambda x: translatechr[x])
        elif self.genome in ['rn8']:   gtf['Chr'] = gtf['Chr'].map(lambda x: translatechr8[x])
        else: raise ValueError('no genome that was able to download')
        gtf = gtf[~gtf.Chr.str.lower().str.contains('un|na| nc')]
        gtf = gtf.dropna(subset = 'gene')
        gtf = gtf[~gtf.gene.str.startswith('LOC')&~gtf.gene.str.startswith('NEW')]
        gtf['Chr'] = gtf['Chr'].map(lambda x: self.replaceXYMTtonums(x.split('_')[0]))
        gtf = gtf.loc[gtf.gene.fillna('') != '', ~gtf.columns.str.contains(' ')]
        return gtf

    def get_ucsc_gtf(self, extractall = False):
        printwithlog('reading gene list from NCBI RefSeq from UCSC...')
        linkdict = {'rn7':f'http://hgdownload.soe.ucsc.edu/goldenPath/rn7/bigZips/genes/ncbiRefSeq.gtf.gz' , 
                        'rn6':'https://hgdownload.soe.ucsc.edu/goldenPath/rn6/bigZips/genes/rn6.ncbiRefSeq.gtf.gz',
                       'm38': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/genes/mm10.ncbiRefSeq.gtf.gz'}
        gtf = pd.read_csv(linkdict[self.genome], sep = '\t', header = None)\
                   .set_axis(['Chr', 'source', 'biotype', 'start', 'end', 'score', 'strand', 'phase', 'ID'], axis = 1)
        gtf['ID'] = gtf['ID'].apply(lambda x: {y.split(' "')[0].strip(' '): y.split(' "')[-1][:-1] for y in x.strip(';').split(';')})
        gtf =pd.concat([gtf.drop('ID', axis = 1), pd.json_normalize(gtf['ID'].to_list())], axis = 1)#.query('biotype == "transcript"')
        gtf = gtf[~gtf.Chr.str.lower().str.contains('un|na')].rename({'gene_id':"gene"}, axis = 1).drop('gene_name', axis = 1)
        gtf = gtf[~gtf.gene.str.startswith('LOC')&~gtf.gene.str.startswith('NEW')]
        gtf['Chr'] = gtf['Chr'].map(lambda x: self.replaceXYMTtonums(x.split('_')[0]))
        if extractall:
            o = gtf.ID.str.extractall('([\w\d_]+) "([^"]*)"').reset_index()\
                   .drop_duplicates(['level_0', 0])\
                   .pivot(index = 'level_0', columns= 0, values= 1).reset_index(drop = True)
            gtf = pd.concat([gtf, o], axis = 1)
        return gtf

    def get_species_accession(self, species = "rattus norvegicus", return_latest = True, display_options = True):
        srtgi = pd.json_normalize(json.loads(bash(f'''datasets summary genome taxon "{species.lower().replace('_', ' ')}"''', shell = True, silent = True, print_call=False)[0])['reports'])\
              .rename({'paired_accession': 'annotated_accession', 'annotation_info.provider': 'provider', 
                       'annotation_info.release_date': 'release_date', 'annotation_info.report_url': 'report_url',
                       'assembly_info.assembly_name': 'assembly_name', 'assembly_info.biosample.description.organism.organism_name':'organism_name'}, axis = 1)\
              .sort_values('release_date', ascending = False).reset_index(drop = True)
        firstcols = ['assembly_name', 'accession','annotated_accession', 'provider', 'release_date', 'report_url', 'organism_name']
        disp = srtgi[firstcols + list(set(srtgi.columns)-set(firstcols))]
        disp['annotated_accession'] = disp.annotated_accession.str.replace('GCA_', 'GCF_')
        disp['accession'] = disp.annotated_accession.str.replace('GCF_', 'GCA_')
        if display_options: display(fancy_display(disp.dropna(subset = 'annotated_accession')))
        if not return_latest:return disp
        return disp.loc[0, 'annotated_accession']
    
    def ask_user_genome_accession(self): 
        genome_accession = ''
        while genome_accession == '':
            queryspecies = input('please write the scientific name of the species used:')
            queryspecies = ' '.join(re.findall('[A-Za-z]+', queryspecies))
            try: 
                latest =  self.get_species_accession(species = queryspecies,return_latest= False )
                while genome_accession not in latest['annotated_accession'].str.replace('GCA_', 'GCF_').to_list():
                    genome_accession = input('please write the GCF annotated assession id or leave it black to use the latest:').replace('GCA_', 'GCF_')
                    if genome_accession == '': 
                        print(f'''{latest.loc[0, 'annotated_accession']}''' )
                        genome_accession = latest.loc[0, 'annotated_accession'].replace('GCA_', 'GCF_')
            except:
                print('not a valid species')
        self.pull_NCBI_genome_info(GCF_assession_id = genome_accession, redownload = False)
    
    def pull_NCBI_genome_info(self, GCF_assession_id = 'GCF_036323735.1', redownload = False):
        self.genome_accession = GCF_assession_id
        self.gtf_path = f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/genomic.gtf'
        self.chrsyn = f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/chrsynon'
        needs_dowload = redownload or not os.path.isfile(f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/sequence_report.jsonl')
        if needs_dowload:
            bash(f'datasets download genome accession {self.genome_accession} --include genome,gtf,seq-report,gff3,rna,protein,cds', silent=True, shell = True,  print_call=True)
            bash(f'rm -r {self.path}genome_info/', silent = True, print_call=False)
            bash(f'unzip -d {self.path}genome_info/ ncbi_dataset.zip', silent = True, print_call=False)
            bash(f'rm {self.path}ncbi_dataset.zip', silent = True, print_call=False)
            for i in ['.gtf', '.gff']:
                i2 = self.gtf_path.replace('.gtf', i)
                bash(f'''grep -v "#" {i2} | sort -k1,1 -k4,4n -k5,5n -t$'\t' | bgzip -c > {i2}.gz''', shell = True, silent = True, print_call=False)
                bash(f"tabix -p gff {i2}.gz", silent = True, print_call=False)
        self.chr_conv_table = pd.read_json(f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/sequence_report.jsonl',  lines=True).rename({'chrName': 'Chr'}, axis = 1)
        self.n_autosome = self.chr_conv_table.Chr.str.extract('(\d+)')[0].fillna(-1).astype(int).max()
        self.replacenumstoXYMT = lambda x: str(int(float(x))).replace(str(self.n_autosome+1), 'x')\
                                                 .replace(str(self.n_autosome+2), 'y')\
                                                 .replace(str(self.n_autosome+4), 'mt')
        self.replaceXYMTtonums = lambda x: int(float(str(x).lower().replace('chr', '').replace('x', str(self.n_autosome+1))\
                                                 .replace('y', str(self.n_autosome+2))\
                                                 .replace('mt', str(self.n_autosome+4))\
                                                 .replace('m', str(self.n_autosome+4))))
        syntable = self.chr_conv_table[['Chr', 'refseqAccession']].query('Chr != "Un"')
        syntable['Chrnum'] = syntable.Chr.map(self.replaceXYMTtonums)
        syntable = syntable.melt(id_vars = 'refseqAccession').drop('variable', axis = 1).astype(str).drop_duplicates().sort_values('value')
        syntable = pd.concat([syntable, syntable.assign(value = 'chr'+syntable.value)])
        pd.concat([syntable, syntable.set_axis(syntable.columns[::-1], axis = 1)]).to_csv(self.chrsyn,  sep = '\t', header = None, index = False)
        
        self.genome_assession_info = pd.read_json(f'{self.path}genome_info/ncbi_dataset/data/assembly_data_report.jsonl', lines = True)
        self.genome_version= self.genome_assession_info.loc[0,'assemblyInfo']['assemblyName']
        self.genomefasta_path = f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/{self.genome_accession}_{self.genome_version}_genomic.fna'
        self.species = self.genome_assession_info.loc[0, 'organism']['organismName'].replace(' ', '_').lower()
        self.taxid = str(self.genome_assession_info.loc[0, 'organism']['taxId']  )      
    
    def get_gtf(self):
        if not hasattr(self, 'gtf_path'): self.pull_NCBI_genome_info(self.genome_accession, redownload = False)
        if not hasattr(self, 'chr_conv_table'): self.pull_NCBI_genome_info(self.genome_accession, redownload = False)    
        gtf = pd.read_csv(self.gtf_path, sep = '\t', header = None, comment='#')\
                .set_axis(['Chr', 'source', 'biotype', 'start', 'end', 'score', 'strand', 'phase', 'ID'], axis = 1)
        gtf = gtf[gtf.biotype != 'transcript'].reset_index(drop = True)
        gtf['biotype'] = gtf['biotype'].str.replace('gene','transcript')
        def gtfid2json(z):
            try: return {y[0]: y[1] for x in z.split('"; ')[:-1] if ' ' not in (y:= x.split(' "'))[0]}
            except: 
                print(f'gtf json convert failed with {z}')
                return {}
        gtfjson = pd.json_normalize(gtf['ID'].map(gtfid2json).to_list())
        gtf =pd.concat([gtf.drop('ID', axis = 1),gtfjson], axis = 1)
        gtf[['gene', 'gene_id']] = gtf[['gene', 'gene_id']].fillna('').astype(str)
        gtf = gtf[~gtf.gene.str.contains('-ps')].rename({'Chr': 'refseqAccession'}, axis = 1)
        gtf = self.chr_conv_table[['Chr', 'refseqAccession']].merge(gtf, on = 'refseqAccession', how = 'right')
        gtf= gtf[~gtf.Chr.fillna('UNK').isin(['Un', '', 'UNK'])]
        gtf['Chr'] = gtf['Chr'].map(lambda x: self.replaceXYMTtonums(x.split('_')[0]))
        gtf = gtf.loc[(gtf.gene.fillna('') != '') & ~gtf.transcript_id.str.contains('unassigned_transcript') , ~gtf.columns.str.contains(' ') ]
        gtf = gtf.assign(genomic_pos = gtf.Chr.astype(str)+':'+gtf.start.astype(str)+'-'+gtf.end.astype(str))
        self.gtf = gtf
        return gtf
        
    def locuszoom_interactive(self, qtltable:pd.DataFrame() = '', qtl_r2_thresh: float = .8,  padding: float = 1e5):

        printwithlog('starting interactive locuszoom generator...') 
        if type(qtltable) != pd.core.frame.DataFrame:
            qtltable = pd.read_csv(f'results/qtls/finalqtl.csv').reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        gtf = gtf.query('biotype != "transcript"')
        def glk(gene):
            return f'<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene}">{gene}</a>'
        def rn(x,n = 3):
            try: o = round(float(x), int(n))
            except: o = x
            return o

        for _, topsnp in tqdm(list(qtltable.iterrows())):
            #printwithlog(f'starting interactive locuszoom for SNP {topsnp.SNP}...')
            data = pd.read_csv(f'{self.path}results/lz/lzplottable@{topsnp.trait}@{topsnp.SNP}.tsv', sep = '\t').query('p != "UNK"')
            data['-log10(P)'] = -np.log10(pd.to_numeric(data.p, errors = 'coerce')) 
            minval, maxval = data.query(f'R2 > {qtl_r2_thresh}').agg({'bp': [min, max]}).values.flatten() + np.array([-padding, padding])
            genes_in_section = gtf.query(f'Chr == {topsnp.Chr} and  end > {minval} and start < {maxval}').reset_index(drop = True)
            causal = pd.read_csv(f'{self.path}results/qtls/possible_causal_snps.tsv' , sep = '\t')\
                       .query(f'SNP_qtl == "{topsnp.SNP}"')
            causal['bp'] = causal.SNP.str.extract(':(\d+)')
            causal = causal.merge(data[['SNP', '-log10(P)']], on='SNP')

            phewas = pd.read_csv(f'{self.path}results/phewas/pretty_table_both_match.tsv' , sep = '\t')\
                       .query(f'SNP_QTL == "{topsnp.SNP}"')
            phewas['bp'] = phewas.SNP_PheDb.str.extract(':(\d+)')
            phewas = phewas.merge(data[['SNP', '-log10(P)']], left_on='SNP_PheDb', right_on='SNP')

            eqtl = pd.read_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv' )
            eqtl['SNP'] = eqtl['SNP'].str.replace('chr', '')#\
            eqtl = eqtl.query(f'SNP == "{topsnp.SNP}"')
            eqtl['bp'] = eqtl.SNP_eqtldb.str.extract(':(\d+)')
            eqtl = eqtl.merge(data[['SNP', '-log10(P)']], left_on='SNP_eqtldb', right_on='SNP')

            sqtl = pd.read_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv' )
            sqtl['SNP'] = sqtl['SNP'].str.replace('chr', '')#\
            sqtl = sqtl.query(f'SNP == "{topsnp.SNP}"')
            sqtl['bp'] = sqtl.SNP_sqtldb.str.extract(':(\d+)')
            sqtl = sqtl.merge(data[['SNP', '-log10(P)']], left_on='SNP_sqtldb', right_on='SNP')

            fig = px.scatter(data_frame=data.query('@minval<bp<@maxval'), x= 'bp', y= '-log10(P)', #hover_data=data.columns,ice_r
                             color = 'R2', color_continuous_scale= 'Jet')#hover_data =['R2']
            fig.update_traces(marker = dict(size=10, opacity = .6, line_color = 'lightgray', line_width = 1,  symbol = 'circle-open'),
                              hoverinfo='none', hovertemplate='' )
            
            if causal.shape[0] > 0:
                fig.add_scattergl(x = causal.bp,y = causal['-log10(P)'],mode='markers', name = 'non synonymous',
                                   marker=dict( line_width = 1, size = 14, color ='orange', line_color = 'black',),
                                   text = [f'{x.SNP}<br>{glk(x.gene)}:{x.annotation}<br>R2: {x.R2}' for name, x in causal.iterrows()],
                                   hovertemplate='%{text}',marker_symbol = 'circle-x')#,  visible='legendonly'
            if sqtl.shape[0] > 0:
                fig.add_scattergl(x = sqtl.bp,y = sqtl['-log10(P)'],mode='markers',name = 'sqtl',
                                   marker=dict( line_width = 1, size = 14, color ='green', line_color = 'black',),
                                  text = [f'{x.SNP_sqtldb}<br>tissue:{x.tissue}<br>gene:{glk(x.Ensembl_gene)}<br>-log10(p): {rn(x["-log10(pval_nominal)"])}<br>R2: {rn(x.R2)}'
                                                                   for name, x in sqtl.iterrows()],  hovertemplate='%{text}',
                                   marker_symbol = 'diamond-x')#,  visible='legendonly'
            if eqtl.shape[0] > 0:
                fig.add_scattergl(x = eqtl.bp,y = eqtl['-log10(P)'],mode='markers',name = 'eqtl',
                                   marker=dict( line_width = 1, size = 14, color ='green', line_color = 'black',),
                                  text = [f'{x.SNP_eqtldb}<br>tissue:{x.tissue}<br>gene:{glk(x.Ensembl_gene)}<br>-log10(p): {rn(x["-log10(pval_nominal)"])}<br>R2: {rn(x.R2)}'
                                                                   for name, x in eqtl.iterrows()],  hovertemplate='%{text}',
                                   marker_symbol = 'diamond-cross')#,  visible='legendonly'
            if phewas.shape[0] > 0:
                fig.add_scattergl(x = phewas.bp,y = phewas['-log10(P)'],mode='markers', name = 'phewas', 
                                  hovertemplate='%{text}', text = [f'{x.SNP_PheDb}<br>{x.project}:{x.trait_PheDb}<br>-log10(p): {rn(x.p_PheDb)}<br>R2: {rn(x.R2)}'
                                                                   for name, x in phewas.iterrows()],
                                   marker=dict( line_width = 1, size = 14, color ='steelblue', line_color = 'black',),
                                   marker_symbol = 'star-square-dot')
            topsnpcols = ['Freq', 'b', 'se', 'p', 'trait', 'interval_size', 'significance_level']
            fig.add_scattergl(x = [topsnp.bp],y = [topsnp.p],mode='markers',name = 'TopSNP<br>'+ topsnp.SNP,
                               marker=dict( line_width = 1, size = 20, color ='firebrick', line_color = 'black',),
                              hovertemplate = '%{text}', text = ['<br>'.join([f'{k}: {rn(v)}' for k,v in topsnp[topsnpcols].to_dict().items()])],
                               marker_symbol = 'star-diamond-dot')
            translation_table = pd.DataFrame(list(zip(['black', 'gray', 'steelblue', 'black', 'lightblue', 'green', 'red'], 
                                                          [500,500,500,500,500,500,500])),
                                                 index =['transcript', 'exon', '5UTR', 'CDS', '3UTR', 'start_codon','stop_codon'], columns = ['color', 'width'])
            already_shown = set()
            for idx, i in genes_in_section.iterrows():
                    ini, fin = (i.start, i.end) # if i.strand == "+" else (i.start, i.end)[::-1]
                    annotation = glk(i['gene']) if i.biotype =='transcript' else i.biotype
                    fig.add_trace(go.Scatter(y=-.5 -(idx%1/2) + np.array([-1, -1, 1, 1, -1])/(5 if i.biotype !='transcript' else 50),#[i.yaxis,i.yaxis], 
                                            x = [ini, fin, fin, ini, ini],
                                            fill='toself', hoverinfo='text',
                                            showlegend=False,hovertemplate = '%{text}',
                                            text=annotation, mode="lines",  opacity=0.5,
                                            marker = dict(color = translation_table.loc[i.biotype, 'color']))) #
                    if i.biotype not in ['exon', 'CDS', '3UTR', '5UTR','start_codon','stop_codon']:
                        if (i.biotype == 'transcript') and (i.gene in already_shown): pass
                        else:
                            fig.add_trace(go.Scatter(x = np.linspace(ini,fin +1e-10, 8), y=8*[-.5 -(idx%1/2)] , text=annotation ,#+ ' ' + i.strand, 
                                                           showlegend=False, hoverinfo='text',  mode='lines',
                                                     marker = dict(color = translation_table.loc[i.biotype, 'color']),
                                                          textposition="bottom center"))
                            already_shown.add(i.gene)
                    if i.biotype =='transcript':
                        _, __ = i[['start', 'end']][:: -1 if i.strand == '-' else 1]
                        fig.add_annotation(x=__, y=-.5,ax=_, ay=-.5,
                          xref='x', yref='y', axref='x', ayref='y',text='',  
                          showarrow=True,arrowhead=1,arrowsize=1, arrowwidth=3, arrowcolor='black')
            fig.add_hline(y=self.threshold, line_width=2,  line_color="red", annotation_text="10% threshold",  line_dash="dot",
                          annotation_position="top right")
            fig.add_hline(y=self.threshold05, line_width=2, line_color="blue", annotation_text="5% threshold",  line_dash="dot",
                          annotation_position="top right")
            fig.add_hline(y=0, line_width=1, line_color="black", opacity = 1)
            fig.update_xaxes(showline=False, title="Chromosome",rangeslider=dict( visible=True), range=[minval+padding, maxval-padding])
            fig.update_yaxes(title="-log<sub>10</sub>(p)", title_font_family="Arial",
                             title_font_size = 40, tick0 = 0, tickmode = 'linear', dtick = 0.5)
            fig.update_layout(template='simple_white',width = 1200, height = 800, coloraxis_colorbar_x=1.05,
                              coloraxis_colorbar_y = .3,coloraxis_colorbar_len = .8,hovermode='x unified')
            #fig.show()
            fig.write_html(f'{self.path}images/lz/lz__{topsnp.trait}__{topsnp.SNP}.html'.replace(':', '_'))
            #plotio.write_json(fig, f'{self.path}images/lz/lz__{topsnp.trait}__{topsnp.SNP}.json'.replace(':', '_'))

    def get_closest_snp(self, s, include_snps_in_gene = False, include_snps_in_ld  = False):
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        gtff = gtf.query("biotype == 'transcript' and gene != ''")
        if s in gtff.gene.unique():
            l = gtff.set_index('gene').loc[s]
            chr, pos = l.Chr, l.loc[['start', 'end']].mean()
        else:
            chr, pos = s.split(':')
            chr, pos = int(float(self.replaceXYMTtonums(chr))), int(float(pos))
        aa =  pd.read_csv(f'{self.path}genotypes/genotypes.bim', sep = '\t', header = None ,index_col = [0], dtype = {0: int, 1 : str, 3: int},  usecols=[0,1,3])\
                             .set_axis(['SNP', 'bp'],axis = 1).loc[self.replaceXYMTtonums(chr)].reset_index().rename({0: 'Chr'}, axis = 1)
        if include_snps_in_gene and s in gtff.gene.unique(): 
            snpingene = aa.query('@l.start <= bp <= @l.end')
            if not len(snpingene): snpingene = aa.loc[(abs(aa.bp - pos)).idxmin()].to_frame().T.reset_index(drop = True)
        else: snpingene = aa.loc[(abs(aa.bp - pos)).idxmin()].to_frame().T.reset_index(drop = True)
        if include_snps_in_ld: 
            snps_ingene = ','.join(snpingene.SNP)
            snpingene = self.plink(bfile = self.genotypes_subset, chr = chr, ld_snps = snps_ingene, ld_window_r2 = include_snps_in_ld, r2 = 'dprime',
                             ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb =  7000, nonfounders = '').loc[:, ['CHR_B','SNP_B', 'R2', 'DP']].rename({'CHR_B': 'Chr', 'SNP_B': 'SNP'}, axis = 1)
        return snpingene.reset_index(drop = True)
    
    def locuszoom_widget(self):
        from ipywidgets import interact, interact_manual, widgets
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        cloc = widgets.Combobox(placeholder='Choose Gene or position "chr:pos" ',options=tuple(gtf.gene.unique()),description='gene or pos:',ensure_option=False,layout = widgets.Layout(width='500px'))
        ctrait = widgets.Dropdown(description ='trait:',options=[x.replace('regressedlr_', '') for x in self.traits],placeholder='choose a trait',layout = widgets.Layout(width='500px'))
        cqtl_r2_thresh = widgets.BoundedFloatText(description = 'R2 boudary', value = .65, max = 1, min = 0, step = 0.01,layout = widgets.Layout(width='150px')) 
        cpadding = widgets.BoundedIntText(description = 'Padding', value = int(.1e6), max = int(10e6), min = 0,format='0,0')
        def locuszoom_manual_query(loc, trait,qtl_r2_thresh, padding):
            aa = self.get_closest_snp(loc).loc[0, :]
            tempqtls = pd.read_csv(f'{self.path}results/gwas/regressedlr_{trait}_chrgwas{aa.Chr}.mlma', sep = '\t', index_col = 1).loc[aa.SNP].to_frame().T.reset_index(names = 'SNP')
            tempqtls['p'] = -np.log10(tempqtls['p'].astype(float))
            display(tempqtls)
            self.locuszoom(qtltable = tempqtls.assign(QTL = True, trait = trait, interval_size = 'UNK', significance_level = 'UNK') , 
                           save = False, qtl_r2_thresh=qtl_r2_thresh, padding=padding)
            display(pn.pane.PDF(f'{self.path}images/lz/lz__{trait}__{aa.SNP.replace(":", "_")}.pdf', width = 800, height = 400))
            #pn.extension('plotly')
            # display(pn.pane.Plotly(plotly_read_from_html(f'{self.path}images/lz/lz__{trait}__{aa.SNP.replace(":", "_")}.html'), width = 600, height = 400))
            # display(pn.pane.PNG(f'{self.path}images/lz/6Mb/lz__{trait}_6Mb__{aa.SNP.replace(":", "_")}.png', width = 600, height = 400))
            print('Done!')
        interact_manual(locuszoom_manual_query,loc = cloc, trait = ctrait, qtl_r2_thresh= cqtl_r2_thresh, padding = cpadding)
        return 
    
    def phewas(self, qtltable: pd.DataFrame() = None, phewas_file = '',
               ld_window: int = int(3e6), save:bool = True, pval_threshold: float = 1e-4, nreturn: int = 1 ,r2_threshold: float = .8,\
              annotate: bool = True, **kwards) -> pd.DataFrame():
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
        printwithlog(f'starting phewas ... {self.project_name}')  

        if phewas_file: 
            bsname = '_' + basename(phewas_file).split('.')[0]
        else:
            phewas_file, bsname = self.phewas_db, ''
        
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            printwithlog(f'reading file from {self.path}results/qtls/finalqtl.csv...') 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                         .reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                         .set_index('SNP').loc[:, : 'significance_level']

        db_vals = pd.concat(pd.read_parquet(x).query(f'p < {pval_threshold}').assign(phewas_file = x) for x in phewas_file.split(',')).reset_index(drop= True)
        db_vals.SNP = db_vals.SNP.str.replace('chr', '')
        db_vals.trait_description = db_vals.trait_description.astype(str).apply(lambda x: re.sub(r'[^\d\w ]+',' ', x))
        
        table_exact_match = db_vals.merge(qtltable.reset_index(), on = 'SNP', how = 'inner', suffixes = ('_phewasdb', '_QTL'))
        table_exact_match = table_exact_match.query(f'project != "{self.project_name}" or trait_phewasdb != trait_QTL ')        
        self.phewas_exact_match_path = f'{self.path}results/phewas/table_exact_match{bsname}.csv'
        if save: table_exact_match.to_csv(self.phewas_exact_match_path )
        #pd.concat([qtltable, db_vals] ,join = 'inner', axis = 1)
        
        nearby_snps = pd.concat([
             self.plink(bfile = self.genotypes_subset, chr = row.Chr, r2 = 'dprime', ld_snp = row.name, ld_window_r2 = 0.00001,
                        ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb =  12000, nonfounders = '')\
              .query(f'R2 > {r2_threshold}')\
              .query(f'@row.bp-{ld_window}/2<BP_B<@row.bp+{ld_window}/2')\
              .drop(['CHR_A', 'BP_A', 'CHR_B'], axis = 1)\
              .rename({'SNP_A': 'SNP', 'SNP_B': 'NearbySNP', 'BP_B': 'NearbyBP'}, axis = 1)\
              .assign(**row.to_dict())\
              .set_index('SNP')
              for  _, row in tqdm(list(qtltable.iterrows())) ])
        nearby_snps.NearbySNP = nearby_snps.NearbySNP.str.replace('chr', '')
        
        table_window_match = db_vals.merge(nearby_snps.reset_index(), left_on= 'SNP', 
                                                         right_on='NearbySNP', how = 'inner', suffixes = ('_phewasdb', '_QTL'))
        table_window_match = table_window_match.query(f'project != "{self.project_name}" or trait_phewasdb != trait_QTL ')
        self.phewas_window_r2 = f'{self.path}results/phewas/table_window_match{bsname}.csv'
        
        if table_window_match.shape[0] == 0:
            printwithlog('No QTL window matches')
            if save: pd.DataFrame().to_csv(self.phewas_window_r2, index = False)
            return -1
            
        if annotate:
            table_window_match = self.annotatevep(table_window_match.rename({'A1_phewasdb':'A1', 'A2_phewasdb': 'A2',
                                            'Chr_phewasdb':'Chr', 'bp_phewasdb':'bp'}, axis = 1), \
                                 'NearbySNP', save = False).rename({'gene': 'gene_phewasdb', 'annotation':'annotation_phewasdb'}, axis = 1)
            table_exact_match = self.annotatevep(table_exact_match.rename({'A1_QTL':'A1', 'A2_QTL': 'A2','Chr_QTL':'Chr', 'bp_QTL':'bp'}, axis = 1), save = False)\
                                    .rename({'A1': 'A1_QTL', 'A2':'A2_QTL','Chr':'Chr_QTL', 'bp':'bp_QTL'}, axis = 1)
            # table_exact_match= table_exact_match.assign(**{i:'' for i in set(['gene', 'annotation'])-set(table_window_match.columns)})\
            #                                     .rename({'gene': 'gene_phewasdb', 'annotation':'annotation_phewasdb'}, axis = 1)
            table_exact_match= table_exact_match.assign(**{i:'' for i in set(['gene', 'annotation'])-set(table_exact_match.columns)})\
                                                 .rename({'gene': 'gene_phewasdb', 'annotation':'annotation_phewasdb'}, axis = 1)
            
        out = table_window_match.groupby([ 'SNP_QTL','project', 'trait_phewasdb'])\
                                .apply(lambda df : df[df.uploadeddate == df.uploadeddate.max()]\
                                                   .nsmallest(n = nreturn, columns = 'p_phewasdb'))\
                                .reset_index(drop = True)\
                                .assign(phewas_r2_thresh = r2_threshold, phewas_p_threshold = pval_threshold ) #, 'annotation_phewasdb'  'gene_phewasdb'
        
        if save: out.to_csv(self.phewas_window_r2, index = False)
        
        
        ##### make prettier tables
        #phewas_info =   pd.read_csv(self.phewas_exact_match_path).drop('QTL', axis = 1).reset_index()
        phewas_info = table_exact_match.drop('QTL', axis = 1).reset_index()
        phewas_info = phewas_info.loc[:, ~phewas_info.columns.str.contains('Chr|A\d|bp_|b_|se_|Nearby|research|filename')]\
                                  .rename(lambda x: x.replace('_phewasdb', '_PheDb'), axis = 1)\
                                  .drop(['genotypes_file'], axis = 1)
        phewas_info['p_PheDb'] = -np.log10(phewas_info.p_PheDb)
        cols2select = list(set(['SNP', 'trait_QTL','project','trait_PheDb', 'trait_description_PheDb' ,'Freq_PheDb', 
                       'p_PheDb','gene_PheDb', 'annotation_PheDb','round_version' ,'uploadeddate', 'phewas_file']) & set(phewas_info.columns) )
        aa = phewas_info[cols2select ].drop_duplicates()
        if save: aa.to_csv(f'{self.path}results/phewas/pretty_table_exact_match{bsname}.tsv', index = False, sep = '\t')
        aa['SNP_QTL'], aa['SNP_PheDb'] = aa.SNP, aa.SNP
        aa = aa.drop('SNP', axis = 1)
        
        #phewas_info =   pd.read_csv(self.phewas_window_r2).drop('QTL', axis = 1).reset_index()
        phewas_info = out.drop('QTL', axis = 1).reset_index()
        phewas_info = phewas_info.loc[:, ~phewas_info.columns.str.contains('Chr|A\d|bp_|b_|se_|Nearby|research|filename')]\
                                  .rename(lambda x: x.replace('_phewasdb', '_PheDb'), axis = 1)\
                                  .drop(['genotypes_file'], axis = 1)
        phewas_info['p_PheDb'] = -np.log10(phewas_info.p_PheDb)
        cols2select = list(set(["R2", 'DP','SNP_QTL', 'trait_QTL','project','SNP_PheDb', 
                      'trait_PheDb', 'trait_description_PheDb' ,'Freq_PheDb', 
                       'p_PheDb','gene_PheDb', 'annotation_PheDb','round_version' ,'uploadeddate', 'phewas_file']) & set(phewas_info.columns) )
        bb = phewas_info[cols2select].drop_duplicates()
        if save: bb.to_csv(f'{self.path}results/phewas/pretty_table_window_match{bsname}.tsv', index = False, sep = '\t')
        
        oo = pd.concat([aa, bb]).fillna('Exact match SNP')
        #oo.drop_duplicates(subset = [])
        if save: oo.to_csv(f'{self.path}results/phewas/pretty_table_both_match{bsname}.tsv', index = False, sep = '\t')
        return oo
        
    def phewas_widget(self):
        from ipywidgets import interact, interact_manual, widgets
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        self.phewas_widget_result = None
        cloc = widgets.Combobox(placeholder='Choose Gene or position "chr:pos" ',options=tuple(gtf.gene.unique()), \
                                description='gene or pos:',ensure_option=False,layout = widgets.Layout(width='500px'))
        phewas_file = widgets.Combobox(placeholder='Choose parquet phewas file" ',
                                       options=[], value = self.phewas_db,
                                       description='phewas_file:',ensure_option=False,layout = widgets.Layout(width='500px'))
        cqtl_r2_thresh = widgets.BoundedFloatText(description = 'R2 boudary', value = .9, max = 1, min = 0, step = 0.01,layout = widgets.Layout(width='150px')) 
        def phewas_manual_query(loc,qtl_r2_thresh, phewas_file):
            snps = self.get_closest_snp( s= loc, include_snps_in_ld=False, include_snps_in_gene=True) \
                       .assign(trait = 'UNK', p = -1, QTL = True, trait_description = 'UNK', Freq = 'UNK').set_index('SNP')
            out = self.phewas(qtltable =snps,  save= False,r2_threshold = qtl_r2_thresh, annotate = True,phewas_file = phewas_file ).drop_duplicates()
            out = out.loc[out.R2.replace('Exact match SNP', 1.1).sort_values(ascending = False).index]\
                     .drop_duplicates(['SNP_PheDb','trait_PheDb', 'project'], keep = 'first').reset_index(drop = True)
            self.phewas_widget_result = out
            pn.extension('tabulator')
            bsname = basename(phewas_file).split('.')[0]
            display(fancy_display(self.phewas_widget_result, download_name = f'phewas_result_{loc}_{bsname}.csv'))
            print('Done! result is in self.phewas_widget_result')
            return 
        interact_manual(phewas_manual_query,loc = cloc, qtl_r2_thresh= cqtl_r2_thresh, phewas_file=phewas_file)
        return 
        
    def eQTL(self, qtltable: pd.DataFrame()= None,
             pval_thresh: float = 1e-4, r2_thresh: float = .6, nreturn: int =1, ld_window: int = 3e6,\
            tissue_list: list = ['BLA','Brain','Eye','IL','LHb','NAcc','NAcc2','OFC','PL','PL2'],\
            annotate = True, **kwards) -> pd.DataFrame():
        
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
        printwithlog(f'starting eqtl ... {self.project_name}') 
        if self.genome_accession not in ['GCF_000001895.4', 'GCF_015227675.2']: 
            pd.DataFrame(columns = ['trait', 'SNP', '-log10(P-value)', 'R2', 'SNP_eqtldb', 'tissue', \
                                    '-log10(pval_nominal)', 'DP', 'Ensembl_gene', 'gene_id', 'slope', 'tss_distance', 'af', 'presence_samples']) \
                        .to_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv', index = False)
            return -1
        #d =  {'rn6': '', 'rn7': '.rn7.2'}[genome]
        #mygene_species = {'rn6': 'rat', 'rn7': 'rat', 'm38': 'mouse', 'cfw': 'mouse'}[self.genome]
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            printwithlog(f'reading file from {self.path}results/qtls/finalqtl.csv...') 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                        .reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                        .set_index('SNP').loc[:, : 'significance_level']
                         
        out = []
        genomeacc2rnv = {'GCF_000001895.4': 'rn6', 'GCF_015227675.2': 'rn7' }
        #genomeacc2rnv[self.genome_accession]
        for tissue in tqdm(tissue_list,  position=0, desc="tissue", leave=True):

            tempdf = pd.read_csv(f'https://ratgtex.org/data/eqtl/{tissue}.{genomeacc2rnv[self.genome_accession]}.cis_qtl_signif.txt.gz', sep = '\t').assign(tissue = tissue)\
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

        out = pd.concat(out).reset_index(drop=True).drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        if annotate:
            out = self.annotatevep(out, 'NearbySNP', save = False)
        self.eqtl_path = f'{self.path}results/eqtl/eqtl.csv'
        out['presence_samples'] = out.ma_samples.astype(str) + '/'+ out.ma_count.astype(str)
        out.to_csv(self.eqtl_path, index= False)
        
        #### make pretty tables
        eqtl_info = pd.read_csv(self.eqtl_path).rename({'p':'-log10(P-value)'}, axis = 1)
        eqtl_info['-log10(pval_nominal)'] = -np.log10(eqtl_info['pval_nominal'])
        gene_conv_table = pd.DataFrame([(x['query'], defaultdict(lambda: '', x)['symbol']) for x \
                                in  mg.querymany(eqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species=self.taxid, verbose = False, silent = True)],
                              columns = ['gene_id','Ensembl_gene'])
        gene_conv_table= gene_conv_table.groupby('gene_id').agg(lambda df: ' | '.join(df.drop_duplicates())).reset_index()
        eqtl_info = eqtl_info.merge(gene_conv_table, how = 'left', on = 'gene_id')
        eqtl_info.Ensembl_gene = eqtl_info.Ensembl_gene.fillna('')
        #eqtl_info['Ensembl_gene'] = [defaultdict(lambda: '', x)['symbol'] for x in  mg.querymany(eqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species=self.taxid)];
        eqtl_info = eqtl_info.loc[:,  ['trait','SNP_QTL','-log10(P-value)','R2','SNP_eqtldb','tissue', '-log10(pval_nominal)','DP' ,
                                       'Ensembl_gene','gene_id', 'slope' ,'tss_distance', 'af', 'presence_samples']].rename(lambda x: x.replace('_QTL', ''), axis = 1)
        eqtl_info.SNP = 'chr' + eqtl_info.SNP
        eqtl_info = eqtl_info.drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        eqtl_info.to_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv', index = False)
        return eqtl_info
    
    
    def sQTL(self, qtltable: pd.DataFrame() = None,
             pval_thresh: float = 1e-4, r2_thresh: float = .6, nreturn: int =1, ld_window: int = 3e6, just_cis = True,
             tissue_list: list = ['BLA','Brain','Eye','IL','LHb','NAcc','NAcc2','OFC','PL','PL2'], annotate = True, **kwards) -> pd.DataFrame():
        
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
        printwithlog(f'starting spliceqtl ... {self.project_name}') 
        #mygene_species = {'rn6': 'rat', 'rn7': 'rat', 'm38': 'mouse', 'cfw': 'mouse'}[self.genome]
        #d =  {'rn6': '', 'rn7': '.rn7.2'}[genome]
        if self.genome_accession not in ['GCF_015227675.2', 'GCF_000001895.4']: 
            pd.DataFrame(columns = ['trait', 'SNP', '-log10(P-value)', 'R2', 'SNP_sqtldb', 'tissue',
                                    '-log10(pval_nominal)', 'DP', 'Ensembl_gene', 'gene_id', 'slope', 'tss_distance', 'af', 'presence_samples']).to_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv', index = False)
            return -1
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            printwithlog(f'reading file from {self.path}results/qtls/finalqtl.csv...') 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                         .reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                         .set_index('SNP').loc[:, : 'significance_level']
                         
        out = []
        loop_str = [('cis','cis_qtl_signif')] if just_cis else [('cis','cis_qtl_signif'), ('trans','trans_qtl_pairs')]
        genomeacc2rnv = {'GCF_000001895.4': 'rn6', 'GCF_015227675.2': 'rn7' }
        
        for tissue, (typ, prefix) in tqdm(list(itertools.product(tissue_list, loop_str)),  position=0, desc="tissue+CisTrans", leave=True):

            tempdf = pd.read_csv(f'https://ratgtex.org/data/splice/{tissue}.{genomeacc2rnv[self.genome_accession]}.splice.{prefix}.txt.gz', sep = '\t').assign(tissue = tissue)\
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

        out = pd.concat(out).reset_index(drop=True)
        out['gene_id'] = out.phenotype_id.str.extract('(ENSRNOG\d+)')
        if annotate: out = self.annotatevep(out, 'NearbySNP', save = False)
        out['presence_samples'] = out.ma_samples.astype(str) + '/'+ out.ma_count.astype(str)
        self.sqtl_path = f'{self.path}results/sqtl/sqtl_table.csv'
        out.drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1).to_csv(self.sqtl_path, index= False)
        
        #### make pretty tables
        sqtl_info = pd.read_csv(self.sqtl_path).rename({'p':'-log10(P-value)'}, axis = 1)
        sqtl_info['-log10(pval_nominal)'] = -np.log10(sqtl_info['pval_nominal'])
        gene_conv_table = pd.DataFrame([(x['query'], defaultdict(lambda: '', x)['symbol']) for x \
                                in  mg.querymany(sqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species=self.taxid, verbose = False, silent = True)],
                              columns = ['gene_id','Ensembl_gene'])
        gene_conv_table= gene_conv_table.groupby('gene_id').agg(lambda df: ' | '.join(df.drop_duplicates())).reset_index()
        sqtl_info = sqtl_info.merge(gene_conv_table, how = 'left', on = 'gene_id')
        #sqtl_info['Ensembl_gene'] = [defaultdict(lambda: '', x)['symbol'] for x in  mg.querymany(sqtl_info.gene_id , scopes='ensembl.gene', fields='symbol', species='rat')];
        sqtl_info = sqtl_info.loc[:,  ['trait','SNP_QTL','-log10(P-value)','R2','SNP_sqtldb','tissue', '-log10(pval_nominal)','DP' ,
                                       'Ensembl_gene','gene_id', 'slope' , 'af', 'sQTLtype', 'presence_samples']].rename(lambda x: x.replace('_QTL', ''), axis = 1)
        sqtl_info.SNP = 'chr' + sqtl_info.SNP
        sqtl_info = sqtl_info.drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        sqtl_info.to_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv', index = False)
        return sqtl_info
    
    
    def manhattanplot(self, traitlist: list = [], save_fmt: list = ['html', 'png'], display: bool = True):
        
        printwithlog(f'starting manhattanplot ... {self.project_name}')
        if len(traitlist) == 0: traitlist = self.traits
        for num, t in tqdm(list(enumerate(traitlist))):
            df_gwas,df_date = [], []
            #chrlist = [str(i) if i!=(self.n_autosome+1) else 'x' for i in range(1,self.n_autosome+2)]
            # for opt in [f'{t}.loco.mlma'] + [f'{t}.mlma'] + [f'{t}_chrgwas{chromp2}.mlma' for chromp2 in self.chrList()]:
            for opt in [f'regressedlr_{t.replace("regressedlr_", "")}.loco.mlma', 
                        f'regressedlr_{t.replace("regressedlr_", "")}.mlma']+ \
                       [f'regressedlr_{t.replace("regressedlr_", "")}_chrgwas{chromp2}.mlma' for chromp2 in self.chrList()]:
                if glob(f'{self.path}results/gwas/{opt}'):
                    df_gwas += [pd.read_csv(f'{self.path}results/gwas/{opt}', sep = '\t')]
                else: 
                    pass
            if len(df_gwas) == 0 :
                printwithlog(f'could not open mlma files for {t}')
            df_gwas = pd.concat(df_gwas)
            df_gwas['inv_prob'] = 1/np.clip(df_gwas.p, 1e-6, 1)
            df_gwas = pd.concat([df_gwas.query('p < 1e-3'),
                                    df_gwas.query('p > 1e-3').sample(500000)] ).sort_values(['Chr', 'bp']).reset_index().dropna()
            
            append_position = df_gwas.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
            df_gwas['Chromosome'] = df_gwas.apply(lambda row: row.bp + append_position[row.Chr], axis = 1)
            def mapcolor(c, thresh , p):
                if -np.log10(p)> thresh : return 'red' 
                elif int(str(c).replace('X',str(self.n_autosome+1)).replace('Y', str(self.n_autosome+2)).replace('MT', str(self.n_autosome+4)))%2 == 0: return 'steelblue'
                return 'navy'
            df_gwas['color']= df_gwas.apply(lambda row: mapcolor(row.Chr, self.threshold, row.p) ,axis =1)
            fig2 =  go.Figure(data=go.Scattergl(
                x = df_gwas['Chromosome'].values,
                y = -np.log10(df_gwas['p']),
                mode='markers', marker=dict(color=df_gwas.color,line_width=0)))
            for x in append_position.values: fig2.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray")
            fig2.add_hline(y=self.threshold, line_width=2,  line_color="red")
            fig2.add_hline(y=self.threshold05, line_width=2, line_color="blue")

            fig2.update_layout(yaxis_range=[0,max(6, -np.log10(df_gwas.p.min())+.5)], xaxis_range = df_gwas.Chromosome.agg(['min', 'max']),
                               template='simple_white',width = 1920, height = 800, showlegend=False, xaxis_title="Chromosome", yaxis_title="-log10(p)")
            fig2.layout['title'] = f'{t.replace("regressedlr_", "")} n={self.df["regressedlr_"+ t.replace("regressedlr_", "")].count()}'
            dfgwasgrouped = df_gwas.groupby('Chr')
            fig2.update_xaxes(ticktext = [self.replacenumstoXYMT(names) for names,dfs in dfgwasgrouped],
                  tickvals =(append_position + dfgwasgrouped.bp.agg('max').sort_index().cumsum())//2 )
            if 'png' in save_fmt: fig2.write_image(f"{self.path}images/manhattan/{t}.png",width = 1920, height = 800)
            if display: fig2.show(renderer = 'png',width = 1920, height = 800)
        return fig2, df_gwas
            
    
    def porcupineplot(self, qtltable: pd.DataFrame(), traitlist: list = [], run_only_qtls = True,
                    save_fmt: list = ['html', 'png'], display: bool = True,
                      low_mem = False, childlock = True, qtl_annotation: bool = True, nosmallp: bool = True):
        printwithlog(f'starting porcupineplot ... {self.project_name} reading files')
        samplen = int(1e5) if low_mem else int(5e5) 
        rangen = range(160,180) if low_mem else range(80,90)
        maxtraits = 10
        if len(traitlist) == 0: 
            if run_only_qtls: traitlist = list(qtltable.trait.unique())
            else: traitlist = self.traits
        traitlist = [x.replace("regressedlr_", "") for x in traitlist]
        qtl = qtltable.query('QTL==True')
        df_gwas,df_date = [], []
        for trait_pos, t in tqdm(list(enumerate(traitlist))):
            if childlock == True: childlock_val = np.inf if trait_pos < maxtraits else 0
            for opt in [f'regressedlr_{t.replace("regressedlr_", "")}.loco.mlma', 
                        f'regressedlr_{t.replace("regressedlr_", "")}.mlma']+ \
            [f'regressedlr_{t.replace("regressedlr_", "")}_chrgwas{chromp2}.mlma' for chromp2 in self.chrList()]:
                if glob(f'{self.path}results/gwas/{opt}'): 
                    samplenuse = samplen//20 if '_chrgwas' in opt else samplen
                    g = pd.read_csv(f'{self.path}results/gwas/{opt}', sep = '\t', dtype = {'Chr': int, 'bp': int}).assign(trait = t)
                    g['p'] = g['p'].fillna(1)
                    #g=g.applymap(np.nan_to_num)
                    g['inv_prob'] = 1/(np.clip(g.p, 1e-6, 1)) 
                    if not nosmallp:
                        if  g.query('p > 0.05').shape[0] > 0:
                            gweighted = [g.query('p > 0.05').sample(min(samplenuse, g.query('p > 0.05').shape[0], childlock_val), weights='inv_prob')]
                        else: gweighted = []
                        gweighted += [g[::np.random.choice(rangen)].sample(frac = (trait_pos< maxtraits))]
                    else: gweighted =[]
                    g = pd.concat([g.query('p < 0.05')]+ gweighted )\
                    .sort_values(['Chr', 'bp']).reset_index(drop = True).dropna()
                    df_gwas += [g]
                    if sum(map(len, df_gwas)) > 1e6: maxtraits = 0
                else: pass
        df_gwas = pd.concat(df_gwas).sort_values(['Chr', 'bp']).reset_index(drop = True)
        
        append_position = df_gwas.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
        df_gwas['Chromosome'] = df_gwas.apply(lambda row: row.bp + append_position[row.Chr], axis = 1)
        cmap = sns.color_palette("tab10", len(traitlist))
        d = {t: cmap[v] for v,t in enumerate(sorted(traitlist))}
        tnum = {t:num for num,t in enumerate(sorted(traitlist))}
        def mapcolor(c, thresh, p, trait):
            if -np.log10(p)> thresh : return d[trait] 
            elif int(str(c).replace('X',str(self.n_autosome+1)).replace('Y', str(self.n_autosome+2)).replace('MT', str(self.n_autosome+4)))%2 == 0: return 'black'
            return 'gray'
        
        printwithlog(f'starting porcupineplot ... {self.project_name} colorcoding')
        df_gwas['color']= df_gwas.progress_apply(lambda row: mapcolor(row.Chr, self.threshold, row.p, row.trait) ,axis =1)
        df_gwas['annotate'] = (df_gwas.SNP + df_gwas.trait.str.replace('regressedlr_', '') ) .isin(qtl.reset_index().SNP+qtl.reset_index().trait.str.replace('regressedlr_', ''))
        df_gwas.trait = df_gwas.trait.str.replace('regressedlr_', '')
        df_gwas['log10p'] = -np.log10(df_gwas['p'])
        
        fig2 =  go.Figure(data=[])
        fig2.add_scattergl(x = df_gwas['Chromosome'].values,y = -np.log10(df_gwas['p']), name = '', 
                           mode='markers', marker=dict(color=df_gwas.color,line_width=0), showlegend = False)
        for name, ite in tqdm(df_gwas.query('annotate').sort_values('trait').groupby('trait')):
            fig2.add_scattergl(x = ite.Chromosome,y = -np.log10(ite.p),mode='markers',name = f"{name} ({tnum[name]+1})",
                           marker=dict( line_width=1, size = 15, color ='rgb({},{},{})'.format(*ite.color.iloc[0]) ),
                           marker_symbol = 'star-diamond-dot')
        for x in append_position.values: fig2.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray")
        fig2.add_hline(y=self.threshold, line_width=2,  line_color="red")
        fig2.add_hline(y=self.threshold05, line_width=2, line_color="blue")
        if qtl_annotation: 
            showlegend = True
            df_gwas.query('annotate')\
                   .apply(lambda x: fig2.add_annotation(x=x.Chromosome, y=-np.log10(x.p),
                                                        text=f"({tnum[x.trait]+1})",showarrow=True,arrowhead=2), axis = 1)
        else:showlegend = True
        if nosmallp:
            # printwithlog('adding lines to porcupineplot')
            temp = df_gwas.query('log10p < 2')
            def add_loli(row):
                return  dict(type = 'line', x0 = row.Chromosome, x1 = row.Chromosome, y0 = 0, y1 =1.32 , line = {'color': row.color, "width": 7, 'dash': 'solid'})
            temp['shapes'] = temp.progress_apply(add_loli, axis = 1)
            temp = temp.assign(rounded = temp.bp.round(-4)).groupby(['Chr', 'rounded']).progress_apply(lambda x: x.nsmallest(1, 'Chromosome'))
            fig2.update_layout(shapes = temp.shapes.to_list())
        printwithlog(f'starting porcupineplot ... {self.project_name} making figure')
        fig2.update_layout(yaxis_range=[0,max(6, -np.log10(df_gwas.p.min())+.5)],
                           xaxis_range = df_gwas.Chromosome.agg(['min', 'max']),
                           template='simple_white',width = 1920, height = 800,  
                           showlegend=showlegend , xaxis_title="Chromosome", yaxis_title="-log10(p)")
        dfgwasgrouped = df_gwas.groupby('Chr')
        fig2.update_xaxes(ticktext = [self.replacenumstoXYMT(names) for names,dfs in dfgwasgrouped],
                  tickvals =(append_position + dfgwasgrouped.bp.agg('max').sort_index().cumsum())//2 )
        printwithlog(f'starting porcupineplot ... {self.project_name} saving figure')
        if 'png' in save_fmt: fig2.write_image(f"{self.path}images/porcupineplot.png",width = 1920, height = 800)
        if display: fig2.show(renderer = 'png',width = 1920, height = 800)
        return fig2, df_gwas

    def porcupineplotv2(self, qtltable = '', traitlist: list = [], display_figure = False, skip_manhattan = False, maxtraits = 60):
        printwithlog('starting porcupine plot v2')
        hv.opts.defaults(hv.opts.Points(width=1200, height=600), hv.opts.RGB(width=1200, height=600) )
        if type(qtltable) == str:
            if not len(qtltable): qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                                               .reset_index()\
                                               .drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                                               .query('QTL == True')
        if not len(traitlist): traitlist = list(map(lambda x:x.replace('regressedlr_', ''),self.traits))        
        cmap = sns.color_palette("tab20", len(traitlist))
        d = {t: cmap[v] for v,t in enumerate(sorted(traitlist))}
        d_inv = {cmap[v]:t for v,t in enumerate(sorted(traitlist))}
        tnum = {t:num for num,t in enumerate(sorted(traitlist))}    
        qtltable['color'] =  qtltable.trait.apply(lambda x: d[x]) 
        qtltable['traitnum'] =  qtltable.trait.apply(lambda x: f'{tnum[x]}') 
        if len(traitlist) > maxtraits: 
            traitlist_new = list(qtltable.trait.unique())
            if maxtraits - len(traitlist_new) > 0:
                traitlist_new += list(np.random.choice(list(set(traitlist) - set(traitlist_new)), maxtraits - len(traitlist_new), replace = False))
        else: traitlist_new = traitlist
        fdf = []
        h2file = pd.read_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t', index_col = 0).rename(lambda x: x.replace('regressedlr_', ''))
        for num, t in tqdm(list(enumerate(traitlist))):
            if not skip_manhattan or t in traitlist_new:
                df_gwas = []
                for opt in [f'regressedlr_{t.replace("regressedlr_", "")}.loco.mlma', 
                            f'regressedlr_{t.replace("regressedlr_", "")}.mlma']+ \
                           [f'regressedlr_{t.replace("regressedlr_", "")}_chrgwas{chromp2}.mlma' for chromp2 in self.chrList()]:
                    if glob(f'{self.path}results/gwas/{opt}'):
                        df_gwas += [pd.read_csv(f'{self.path}results/gwas/{opt}', sep = '\t')]
                    else:  pass
                if len(df_gwas) == 0 :  printwithlog(f'could not open mlma files for {t}')
                df_gwas = pd.concat(df_gwas)
                append_position = df_gwas.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
                qtltable['x'] = qtltable.apply(lambda x: x.bp +  append_position[x.Chr], axis = 1)
                df_gwas['-log10p'] = -np.log10(df_gwas.p)
                df_gwas.drop(['A1', 'A2', 'Freq', 'b', 'se', 'p'], axis = 1, inplace = True)
                def mapcolor(c): 
                    if int(str(c).replace('X',str(self.n_autosome+1)).replace('Y', str(self.n_autosome+2)).replace('MT', str(self.n_autosome+4)))%2 == 0: return 'black'
                    return 'gray'
                df_gwas = df_gwas.groupby('Chr') \
                                 .apply(lambda df: df.assign(color = mapcolor(df.Chr[0]), x = df.bp + append_position[df.Chr[0]])) \
                                 .reset_index(drop = True)
                df_gwas.loc[df_gwas['-log10p']> self.threshold, 'color' ] = str(d[t])[1:-1]
                df_gwas.loc[df_gwas['-log10p']> self.threshold, 'color' ] = df_gwas.loc[df_gwas['-log10p']> self.threshold, 'color' ].str.split(',').map(lambda x: tuple(map(float, x)))
                if not skip_manhattan:
                    yrange = (-.05,max(6, df_gwas['-log10p'].max()+.5))
                    xrange = tuple(df_gwas.x.agg(['min', 'max'])+ np.array([-1e7,+1e7]))
                    fig = []
                    for idx, dfs in df_gwas[df_gwas.color.isin(['gray', 'black'])].groupby('color'):
                        temp = datashade(hv.Points(dfs, kdims = ['x','-log10p']), pixel_ratio= 2, aggregator=ds.count(), width = 1200,height = 600, y_range= yrange,
                                 min_alpha=.7, cmap = [idx], dynamic = False )
                        temp = dynspread(temp, max_px=4,threshold= 1 )
                        fig += [temp]
                    fig = fig[0]*fig[1]
                    fig = fig*hv.HLine((self.threshold05)).opts(color='blue')*hv.HLine(self.threshold).opts(color='red')
                    fig = fig*hv.Points(df_gwas[df_gwas['-log10p']> self.threshold].drop('color', axis = 1), 
                                        kdims = ['x','-log10p']).opts(color = 'red', size = 5)
                    figh2 = round(h2file.loc[t.replace("regressedlr_", ""),'V(G)/Vp'],3)
                    fig = fig.opts(xticks=[((dfs.x.agg(['min', 'max'])).sum()//2 , self.replacenumstoXYMT(names)) for names,dfs in  df_gwas.groupby('Chr')],
                                                   xlim =xrange, ylim=yrange, width = 1200,height = 600,  xlabel='Chromosome',
                                   title = f'{t.replace("regressedlr_", "")} n={self.df["regressedlr_"+ t.replace("regressedlr_", "")].count()} h2={figh2}') 
                    hv.save(fig, f'{self.path}images/manhattan/{t.replace("regressedlr_", "")}.png')
                if t in traitlist_new: fdf += [df_gwas]
        fdf = pd.concat(fdf).reset_index(drop = True).sort_values('x')
        fig = []
        yrange = (-.05,max(6, fdf['-log10p'].max()+.5))
        xrange = tuple(fdf.x.agg(['min', 'max'])+ np.array([-1e7,+1e7]))
        for idx, dfs in fdf[fdf.color.isin(['gray', 'black'])].groupby('color'):
            temp = datashade(hv.Points(dfs, kdims = ['x','-log10p']), pixel_ratio= 2, aggregator=ds.count(), width = 1200,height = 600, y_range= yrange,
                     min_alpha=.7, cmap = [idx], dynamic = False )
            temp = dynspread(temp, max_px=4,threshold= 1 )
            fig += [temp]
        fig = fig[0]*fig[1]
        
        fig = fig*hv.HLine((self.threshold05)).opts(color='blue')
        fig = fig*hv.HLine(self.threshold).opts(color='red')
        
        for idx, dfs in fdf[~fdf.color.isin(['gray', 'black'])].groupby('color'):
            fig = fig*hv.Points(dfs.drop('color', axis = 1), kdims = ['x','-log10p']).opts(color = idx, size = 5)
        
        for t, dfs in qtltable.groupby('trait'):
            fig = fig*hv.Points(dfs.assign(**{'-log10p': qtltable.p}), kdims = ['x','-log10p'],vdims=[ 'trait','SNP' ,'A1','A2','Freq' ,'b','traitnum'], label = f'({tnum[t]}) {t}' ) \
                                          .opts(size = 17, color = d[t], marker='inverted_triangle', line_color = 'black', tools=['hover']) #
        fig = fig*hv.Labels(qtltable.rename({'p':'-log10p'}, axis = 1)[['x', '-log10p', 'traitnum']], 
                            ['x','-log10p'],vdims=['traitnum']).opts(text_font_size='5pt', text_color='black')
        fig.opts(xticks=[((dfs.x.agg(['min', 'max'])).sum()//2 , self.replacenumstoXYMT(names)) for names, dfs in fdf.groupby('Chr')],
                                   xlim =xrange, ylim=yrange, xlabel='Chromosome', shared_axes=False,
                               width=1200, height=600, title = f'porcupineplot',legend_position='right',show_legend=True)
        hv.save(fig, f'{self.path}images/porcupineplot.png')
        if display_figure: 
            display(fig)
            return
        return fig

    def GWAS_latent_space(self, traitlist = [], method = 'nmf'):
        printwithlog('starting GWAS latent space...')
        pn.extension('tabulator')
        pref = {'pca': 'PC', 'nmf':'NMF', 'spca': 'sPC', 'fa': 'FA', 'ica': 'ic', 'da': 'da'}
        if not len(traitlist): 
            traitlist = [x for x in self.traits if not len(re.findall('regressedlr_pc[123]|regressedlr_umap[123]|regressedlr_umap_clusters_\d+|regressedlr_pca_clusters_\d+',x))]
            traitlist = list(map(lambda x:x.replace('regressedlr_', ''),traitlist))        
        cmap = sns.color_palette("tab20", len(traitlist))
        d = {t: cmap[v] for v,t in enumerate(sorted(traitlist))}
        d_inv = {cmap[v]:t for v,t in enumerate(sorted(traitlist))}
        tnum = {t:num for num,t in enumerate(sorted(traitlist))}    
        fdf = []
        deliverable = {}
        for num, t in tqdm(list(enumerate(traitlist))):
            df_gwas = []
            for opt in [f'regressedlr_{t.replace("regressedlr_", "")}.loco.mlma', 
                        f'regressedlr_{t.replace("regressedlr_", "")}.mlma']+ \
                       [f'regressedlr_{t.replace("regressedlr_", "")}_chrgwas{chromp2}.mlma' for chromp2 in self.chrList()]:
                if glob(f'{self.path}results/gwas/{opt}'):
                    df_gwas += [pd.read_csv(f'{self.path}results/gwas/{opt}', sep = '\t')]
                else:  pass
            if len(df_gwas) == 0 :  printwithlog(f'could not open mlma files for {t}')
            df_gwas = pd.concat(df_gwas)
            fdf += [df_gwas.assign(**{f'{t}_p': -np.log10(df_gwas.p)}).set_index('SNP')[[f'{t}_p']]]
        fdf = pd.concat(fdf, axis = 1)
        
        from sklearn.decomposition import SparsePCA, NMF, FactorAnalysis, FastICA, DictionaryLearning
        npc= min(len(traitlist)-1, 10)
        pca = {'nmf': NMF(n_components=npc), 'pca':  PCA(n_components=npc), 'ica': FastICA(n_components=npc),  'da': DictionaryLearning(n_components=npc),  
               'spca':  SparsePCA(n_components=npc), 'fa': FactorAnalysis(n_components=npc) }[method]
        if len(fdf.count().unique()): display(fdf.count().agg(['max', 'min']))
        out = pca.fit_transform(fdf.fillna(0))
        pcadata = pd.DataFrame(pca.components_, columns = fdf.columns[fdf.columns.str.contains('_p$')].map(lambda x: x[:-2]),
                               index =[f'{pref[method]}{i+1}' for i in range(npc)] )
        if method == 'nmf':
            mean_p = fdf.fillna(0).mean().rename(lambda x: x[:-2])
            normed_pcdata = (pcadata.T/pcadata.sum(axis = 1))
            exp_avg_p = (mean_p*normed_pcdata.T).sum(axis = 1)
            out = (out*(exp_avg_p/out.mean(axis=0)).values)
        
        if method == 'pca':
            pcadata.index = pcadata.index + pd.Series(pca.explained_variance_ratio_).map(lambda x: f' ({round(x*100, 2)}%)')
        pcam = pcadata.reset_index(names= 'Latent Space').melt(id_vars='Latent Space', var_name='trait', value_name='importance')
        heatmap = hv.HeatMap(pcam.round(1), kdims=['trait','Latent Space'], vdims=['importance']).opts( tools=['hover'], height = 400, width = 1000 )
        heatmap = heatmap.opts(xrotation=90, cmap = 'Blues' if method =='nmf' else 'RdBu') * hv.Labels(heatmap).opts(padding=0, text_font_size='5pt', text_color='white')
        deliverable['pcafig'] = pn.pane.HoloViews(heatmap)
        plt.close()
        fdf.loc[:, list(map(lambda x: f'{pref[method]}{x}', range(1,npc+1)))] = out
        fdf = fdf.merge(df_gwas[['Chr', 'bp', 'SNP', 'A1', 'A2', 'Freq']], on= 'SNP', how = 'left')
        fdf = fdf.reset_index()
        append_position = fdf.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
        def mapcolor(c): 
            if int(str(c).replace('X',str(self.n_autosome+1)).replace('Y', str(self.n_autosome+2)).replace('MT', str(self.n_autosome+4)))%2 == 0: return 'black'
            return 'gray'
        fdf = fdf.groupby('Chr').apply(lambda df: df.assign(color = mapcolor(df.Chr.iloc[0]), x = df.bp + append_position[df.Chr.iloc[0]])) \
                         .reset_index(drop = True)
        dfm = fdf.melt(id_vars= ['Chr','SNP','bp','A1', 'A2', 'Freq','color', 'x'], value_vars=fdf.columns[fdf.columns.str.contains(pref[method])], value_name='-log10p', var_name= 'trait')
        def add_color(df, t):
            df.loc[df['-log10p'].abs()> self.threshold, 'color' ] =  str(cmap[int(t.replace(pref[method], ''))%30])[1:-1]
            df.loc[df['-log10p'].abs()> self.threshold, 'color' ] = df.loc[df['-log10p'].abs()> self.threshold, 'color' ].str.split(',').map(lambda x: tuple(map(float, x)))
            return df
        dfm = dfm.groupby('trait').progress_apply(lambda x: add_color(x, x.trait.iloc[0])).reset_index(drop = True)
        
        abvt = dfm[dfm['-log10p'].abs() > self.threshold]
        abvt['p'] = 10**(-dfm['-log10p'].abs())
        abvt = abvt.rename({'-log10p':'realp'}, axis = 1).reset_index(drop= True)
        oo = self.callQTLs( NonStrictSearchDir=abvt, add_founder_genotypes = True,conditional_analysis=False, displayqtl= False, save = False, annotate= False)
        if len(oo):
            oo = self.annotatevep(oo, save= False)
            send2d = oo.drop(['p', 'color', 'x', 'trait_description', 'Chr', 'bp'], axis = 1).rename({'realp': '-log10p'}, axis = 1)
            if 'gene' in send2d: send2d= send2d.loc[:,:'gene']
            deliverable['qtls'] = fancy_display(send2d)
        
        fig = []
        yrange = (dfm['-log10p'].min()-.5 if dfm['-log10p'].min() < 0 else 0, max(6, dfm['-log10p'].max()+.5))
        xrange = tuple(dfm.x.agg(['min', 'max'])+ np.array([-1e7,+1e7]))
        for idx, dfs in dfm[dfm.color.isin(['gray', 'black'])].groupby('color'):
            temp = datashade(hv.Points(dfs, kdims = ['x','-log10p']), pixel_ratio= 2, aggregator=ds.count(), width = 1200,height = 800, y_range= yrange,
                     min_alpha=.7, cmap = [idx], dynamic = False )
            temp = dynspread(temp, max_px=4,threshold= 1 )
            fig += [temp]
        fig = fig[0]*fig[1]
        if dfm['-log10p'].min() < 0: fig = fig*hv.HLine((0)).opts(color='linen', line_width=2, alpha=1, line_dash='solid')
        fig = fig*hv.HLine(( -self.threshold05)).opts(color='blue')
        fig = fig*hv.HLine((self.threshold05)).opts(color='blue')
        fig = fig*hv.HLine((self.threshold)).opts(color='red')
        fig = fig*hv.HLine((-self.threshold)).opts(color='red')
        for idx, dfs in dfm[~dfm.color.isin(['gray', 'black'])].groupby('color'):
            fig = fig*hv.Points(dfs.drop('color', axis = 1), kdims = ['x','-log10p']).opts(color = idx, size = 5)
        #.query('QTL')
        if len(oo):
            for (t, negv), dfs in oo.assign(neg =  (oo.realp < 0).astype(bool)).reset_index().groupby(['trait', 'neg']):
                markerstyle = 'triangle' if negv else 'inverted_triangle'
                fig = fig*hv.Points(dfs.assign(**{'-log10p': dfs.realp}), kdims = ['x','-log10p'],vdims=[ 'trait','SNP' ,'A1','A2', 'Freq'], label = f'({t.replace(pref[method], "")}) {t}' ) \
                                              .opts(size = 17, color = dfs.color.iloc[0], marker=markerstyle, line_color = 'black', tools=['hover']) #
            fig = fig*hv.Labels(oo.assign(**{'traitnum': oo.trait.str.replace(pref[method], ''), '-log10p': oo.realp})[['x', '-log10p', 'traitnum']], 
                                ['x','-log10p'],vdims=['traitnum']).opts(text_font_size='5pt', text_color='black')
        fig.opts(xticks=[((dfs.x.agg(['min', 'max'])).sum()//2 , self.replacenumstoXYMT(names)) for names, dfs in dfm.groupby('Chr')],
                                   xlim =xrange, ylim=yrange, xlabel='Chromosome', shared_axes=False,
                               width=1200, height=800, title = f'Latent Spaces',legend_position='right',show_legend=True)
        deliverable['figure'] = pn.pane.HoloViews(fig)
        deliverablef = pn.Card(*list(deliverable.values())[::-1], title = 'GWAS Latent Space', collapsed = True)#, width=1000
        return deliverablef, deliverable

    def annotate(self, qtltable: pd.DataFrame(),
                 snpcol: str = 'SNP', save: bool = False, **kwards) -> pd.DataFrame():
        
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
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            printwithlog('dataframe is empty, returning same dataframe')
            return qtltable #qtltable = pd.read_csv(self.allqtlspath).set_index('SNP')
        d = {'rn6': 'Rnor_6.0.99', 'rn7':'mRatBN7.2.105', 'cfw': 'GRCm38.99','m38': 'GRCm38.99'}[self.genome]
        #bash('java -jar snpEff/snpEff.jar download -v Rnor_6.0.99')
        #bash('java -jar snpEff/snpEff.jar download -v mRatBN7.2.105')
        #bash('java -jar snpEff/snpEff.jar download -v GRCm39.105') 
        #bash('java -jar snpEff/snpEff.jar download -v GRCm38.99') 
        qtltable['Chr'] = qtltable['Chr'].map(self.replacenumstoXYMT).map(str.upper)
        temp  = qtltable.reset_index()\
                        .loc[:,[ 'Chr', 'bp', snpcol, 'A2', 'A1']]\
                        .assign(QUAL = 40, FILTER = 'PASS' ,INFO = '', FORMAT = 'GT:GQ:DP:HQ')
        temp.columns = ["##CHROM","POS","ID","REF","ALT", 'QUAL', 'FILTER', 'INFO', 'FORMAT']
        temp['##CHROM'] = 'chr'+ temp['##CHROM'].astype(str)
        vcf_manipulation.pandas2vcf(temp, f'{self.path}temp/test.vcf', metadata='')
        #a = bash(f'java -Xmx8g -jar {self.snpeff_path}snpEff.jar {d} -noStats {self.path}temp/test.vcf', print_call = False )# 'snpefftest',  -no-intergenic -no-intron
        a = bash(f'$CONDA_PREFIX/share/snpeff-5.2-0/snpEff -Xmx8g {d} -noStats {self.path}temp/test.vcf', shell = True, silent = True, print_call = False )
        #a = subprocess.run(f'$CONDA_PREFIX/share/snpeff-5.2-0/snpEff -Xmx8g {d} -noStats {self.path}temp/test.vcf', capture_output = True, shell = True).stdout.decode('ascii').strip().split('\n') 
        res = pd.read_csv(StringIO('\n'.join(a)),  comment='#',  sep ='\s+', 
                          header=None, names = temp.columns,  dtype=str).query('INFO != "skipping"')  
        ann = res['INFO'].str.replace('ANN=', '').str.split('|',expand=True)
        column_dictionary = defaultdict(lambda: 'UNK', {k:v for k,v in enumerate(['alt_temp', 'annotation', 'putative_impact', 'gene', 'geneid', 'featuretype', 'featureid', 'transcriptbiotype',
                          'rank', 'HGVS.c', 'HGVS.p', 'cDNA_position|cDNA_len', 'CDS_position|CDS_len', 'Protein_position|Protein_len',
                          'distancetofeature', 'errors'])})
        ann = ann.rename(column_dictionary, axis = 1)
        ann.index = qtltable.index
        out = pd.concat([qtltable.loc[:,~qtltable.columns.isin(ann.columns)], ann], axis = 1).replace('', np.nan).dropna(how = 'all', axis = 1).drop('alt_temp', axis = 1, errors ='ignore')
        
        if 'geneid' in out.columns:
            species = translate_dict(self.genome, {'rn7': 'rat', 'rn8':'rat', 'm38':'mouse', 'rn6': 'rat'})
            gene_translation = {x['query']: x['symbol'] for x in mg.querymany(('-'.join(out.geneid)).split('-') ,\
                           scopes='ensembl.gene,symbol,RGD', fields='symbol', species=self.taxid, verbose = False, silent = True)  if 'symbol' in x.keys()}
            if gene_translation: out['gene'] = out.geneid.map(lambda x: translate_dict(x, gene_translation))
        
        if 'errors' in out.columns:  out = out.loc[:, :'errors']
        try: 
            out['Chr'] = out['Chr'].map(self.replaceXYMTtonums)
        except:
            print('Chr not in columns, returning with possible errors')
            return out
        if save:
            self.annotatedtablepath = f'{self.path}results/qtls/finalqtlannotated.csv'
            out.reset_index().to_csv(self.annotatedtablepath, index= False) 
            #out.reset_index().to_csv(f'{self.path}results/qtls/finalqtl.tsv', index= False, sep = '\t')
        
        return out 

    def annotatevep(self, qtltable: pd.DataFrame(), snpcol:str = 'SNP',  refcol:str = 'A2', 
                altcol:str = 'A1', save: bool = False, adjustchr =False,  **kwards) -> pd.DataFrame():
        if len(qtltable) == 0: 
            printwithlog('dataframe is empty, returning same dataframe')
            return qtltable 
        qtltable = qtltable.reset_index()
        if not {'Chr', 'bp'}.issubset(qtltable.columns): 
            qtltable[['Chr', 'bp']] = qtltable['SNP'].str.split(':').to_list()
        qtltable['Chr'] = qtltable['Chr'].map(self.replaceXYMTtonums)
        qtltable = qtltable.sort_values(['Chr', 'bp'])
        temp  = qtltable.loc[:,[ 'Chr', 'bp', snpcol, refcol, altcol]]\
                        .assign(QUAL = 40, FILTER = 'PASS' ,INFO = '', FORMAT = 'GT:GQ:DP:HQ')\
                        .set_axis(["##CHROM","POS","ID","REF","ALT", 'QUAL', 'FILTER', 'INFO', 'FORMAT'], axis = 1)
        vcf_manipulation.pandas2vcf(temp, f'{self.path}temp/test.vcf', metadata='')
        vdir = bash('echo $CONDA_PREFIX/share/ensembl-vep-112.0-0', shell = True, silent=True, print_call=False)[0]
        if not hasattr(self, 'gtf_path') or not hasattr(self, 'genomefasta_path'): 
            self.pull_NCBI_genome_info(self.genome_accession, redownload = False)
        gfffile = self.gtf_path.replace('.gtf', f"{'_adjusted' if adjustchr else ''}.gff.gz")
        fafile = self.genomefasta_path.replace('.fna',f"{'_adjusted' if adjustchr else ''}.fna")
        oo = bash(f'''vep -i {self.path}temp/test.vcf -o STDOUT --gff {gfffile} --species {self.species} \
                      --synonyms {self.chrsyn} -a {self.genome_accession} --no_check_variants_order \
                      --dir {vdir} --dir_cache {vdir} --dir_plugins {vdir} --fasta {fafile} --tab \
                      --regulatory --force_overwrite --nearest gene --domains --per_gene \
                      --appris --biotype --buffer_size 10000 --hgvs \
                      --distance 10000 --mane --show_ref_allele --sift b \
                      --symbol --transcript_version --tsl --uploaded_allele --refseq \
                      --sf {self.path}temp/test.html''', shell = True, print_call=False)
        translate_names_dict = {'SYMBOL': 'gene', 'Feature': 'featureid', 'biotype': 'transcriptbiotype', 'Feature_type': 'featuretype',
                                'IMPACT': 'putative_impact','DISTANCE':'distancetofeature', 'HGVSc': 'HGVS.c', 'HGVSp': 'HGVS.p', 'SNP': 'SNP' }
        oo = pd.read_csv(StringIO('\n'.join(oo)), sep= '\t', comment='##', engine = 'python').iloc[:, :-1]\
               .rename(lambda x: x.replace('#Uploaded_variation', 'SNP'), axis = 1)\
               .drop_duplicates(subset = ['SNP', 'UPLOADED_ALLELE'])\
               .drop(['Location','Allele','Gene', 'UPLOADED_ALLELE'], axis = 1)\
               .replace('-', np.nan).replace('genomic.gff.gz', f'{self.genome_accession}:refseq:gff')\
               .rename(lambda x: x.replace('_', '').lower() if x not in translate_names_dict else translate_dict(x,translate_names_dict ), axis = 1)
        if oo.columns.str.contains('position$').any() and len(oo):
            poscols = list(oo.columns[oo.columns.str.contains('position$')])
            oo['position_'+ '|'.join(map(lambda x:x.replace('position', ''), poscols))] = oo[poscols].astype(str).apply(lambda c: c.str.cat(sep='|').replace('nan', 'NA'), axis = 1)
            oo = oo.drop(poscols, axis = 1)
        oo = oo.T.dropna(how = 'all').T
        oo = qtltable.merge(oo.rename({'SNP': snpcol}, axis = 1), on = snpcol, how = 'left')\
                     .drop(['index'] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        if save:
            self.annotatedtablepath = f'{self.path}results/qtls/finalqtlannotated.csv'
            oo.reset_index().to_csv(self.annotatedtablepath, index= False) 
        return oo

    def qqplot(self, col2use = 'trait', add_pval_thresh = True, save = True, contour = True):
        from matplotlib.colors import rgb2hex as mplrgb2hex
        from io import StringIO
        import statsmodels.api as sm
        from sklearn.preprocessing import MinMaxScaler
        
        cline = client._get_global_client() if client._get_global_client() else Client( processes = False)
        # gwas_res = dd.read_csv(f'{self.path}results/gwas/*.mlma' , sep = '\t', include_path_column= True)
        # future = cline.compute(gwas_res)
        # progress(future,notebook = False, interval="5s")
        # gwas_res = future.result()
        # gwas_res = gwas_res.assign(trait = gwas_res.path.map(lambda x:x.split('regressedlr_')[-1].split('_chrgwas')[0] )) \
        #                    .drop('path',axis = 1)
        # del future
        gwas_res = pd.concat([ pd.read_csv(x, sep = '\t')\
                         .assign(trait = x.split('regressedlr_')[-1].split('_chrgwas')[0])\
                   for x in tqdm(glob(f'{self.path}results/gwas/*.mlma'))]).reset_index(drop=True)
        gwas_res['p'] = -np.log10(gwas_res.p)
        gb =  gwas_res.groupby('trait')
        if col2use == 'prefix':
            gwas_res['prefix'] = gwas_res['trait'].map(lambda x: x.split('_')[0])
        else:  gwas_res['prefix'] = 'passthrough'
        gwas_res['rank'] = -np.log10(gb['p'].rank(ascending = False, pct = True))
        qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtlannotated.csv').query('QTL == True').reset_index(drop=True)\
                     .drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                     .drop(['rank', 'p'], errors = 'ignore', axis = 1)\
                     .merge(gwas_res[['SNP', 'trait', 'rank', 'p', 'prefix']], on = ['SNP', 'trait'], how = 'left', suffixes=['', '2drop'])
        def readsample(tempt, minp = 0):
            temp = pd.concat(pd.read_csv(x, sep = '\t',usecols=['SNP', 'p', 'b'], dtype={'SNP':str, 'p':float, 'b':float}).assign(trait = tempt.split('/')[-1]) for x in glob(tempt+'/*.mlma'))
            temp['p'] = -np.log10(temp.p)
            temp['rnk'] = temp.p.rank(ascending = False)
            temp['rnk'] = -np.log10(MinMaxScaler(feature_range=(1, scaller)).fit_transform(temp[['rnk']])/scaller)
            temp['rnk_bin'] = temp.rnk.round(2).astype(str)
            if minp: temp = temp.query('p>@minp')
            return temp.reset_index(drop = True)
        
        if os.path.isfile(f'{self.path}pvalthresh/PVALTHRESHOLD.csv') and add_pval_thresh:
            dask_delay = 0
            cline = client._get_global_client() if client._get_global_client() else Client( processes = False)
            scaller = gb['p'].agg('count').mean()
            from sklearn.preprocessing import MinMaxScaler
            if not dask_delay:
                out = [readsample(fname) for fname in tqdm(glob(f'{self.path}pvalthresh/gwas/*'))]
                randmlma = pd.concat(out)
            if dask_delay:
                from dask import delayed
                dfs = [delayed(lambda x: readsample(x, minp = .5))(fname) for fname in glob(f'{self.path}pvalthresh/gwas/*')] #.query('p>1.5')
                ddf = dd.from_delayed(dfs)
                future = cline.compute(ddf)
                progress(future,notebook = False, interval="30s")
                randmlma = future.result()#pd.concat([pd.read_json(StringIO(x)) for x in future.result()])
            subsample =  randmlma.groupby('rnk_bin').sample(2000, replace= True)
            maxb = max(gwas_res.b.abs().max(),  subsample.b.abs().max())*1.1
        else: maxb = gwas_res.b.abs().max()*1.1
        scaller = gb['p'].agg('count').mean()
        traitlist = list(map(lambda x:x.replace('regressedlr_', ''),self.traits))
        cmap = sns.color_palette("tab20", len(traitlist))
        d = {t: mplrgb2hex(cmap[v]) for v,t in enumerate(sorted(traitlist))}
        d_inv = {mplrgb2hex(cmap[v]):t for v,t in enumerate(sorted(traitlist))}
        tnum = {t:num for num,t in enumerate(sorted(traitlist))}    
        qtltable['color'] =  qtltable.trait.map(d) 
        qtltable['traitnum'] =  qtltable.trait.map(tnum).astype(str) 
        yrange = (-.05,max(gwas_res.p.max(),gwas_res['rank'].max()) + .5)
        xrange = yrange
        
        xrange_v = (-maxb,maxb)
        fig = hv.HLine((self.threshold05)).opts(color='blue')
        fig_v = hv.HLine((self.threshold05)).opts(color='blue')
        if os.path.isfile(f'{self.path}pvalthresh/PVALTHRESHOLD.csv') and add_pval_thresh:
            if contour:
                fig = fig*hv.Bivariate(subsample.sample(30000), kdims = ['rnk','p'])\
                            .opts(opts.Bivariate(cmap='Greys', colorbar=False, filled=True))\
                            .opts(levels=250)*hv.HLine(self.threshold).opts(color='red')
                fig_v = fig_v*hv.Bivariate(subsample.sample(30000), kdims = ['b','p'])\
                                .opts(opts.Bivariate(cmap='Greys', colorbar=False, filled=True))\
                                .opts(levels=15)*hv.HLine(self.threshold).opts(color='red')
            else:
                fig = fig*datashade(hv.Points(subsample, kdims = ['rnk','p']), pixel_ratio= .5, aggregator=ds.count(), min_alpha=1, \
                                                width = 500,height = 500, y_range= yrange,x_range= xrange,  
                                    cmap = sns.dark_palette(mplrgb2hex([.9,.9,.9]), reverse=True, as_cmap=True), 
                                    dynamic = False )*hv.HLine(self.threshold).opts(color='red')
                fig_v = fig_v*datashade(hv.Points(randmlma.sample(int(1e7)), kdims = ['b','p']), pixel_ratio= .5, aggregator=ds.count(), min_alpha=1, \
                                                width = 500,height = 500, y_range= yrange,x_range= xrange_v,  
                                        cmap = sns.dark_palette(mplrgb2hex([.9,.9,.9]), reverse=True, as_cmap=True), 
                                    dynamic = False )*hv.HLine(self.threshold).opts(color='red')
        else: fig = fig*hv.HLine(self.threshold).opts(color='red')
        fig = fig*hv.HLine((self.threshold05)).opts(color='blue')
        
        crit = gwas_res.trait.isin(qtltable.trait)
        for idx, (trait, df) in tqdm(enumerate(gwas_res.loc[crit].groupby(col2use))):
            fig = fig*dynspread(datashade( hv.Points(df, kdims = ['rank','p']).opts(color=d[trait], size=0), pixel_ratio= 1, aggregator=ds.count(),
                                          width=500, height=500, y_range= yrange,x_range= xrange, \
                                          min_alpha=.7, cmap = [d[trait]], dynamic = False),  max_px=3,threshold= 1 )
            fig_v = fig_v*dynspread(datashade( hv.Points(df, kdims = ['b','p']).opts(color=d[trait], size=0), pixel_ratio= 1, aggregator=ds.count(),
                                          width=500, height=500, y_range= yrange,x_range= xrange_v, \
                                          min_alpha=.5, cmap = d[trait], dynamic = False),  max_px=1,threshold= 1 )
        fig = fig*hv.Slope(1,0).opts(color='black')
        fig_v = fig_v*(hv.VLine(0).opts(color='black', line_width = 1))
        
        for t, dfs in qtltable.groupby(col2use):
            fig = fig*hv.Points(dfs, kdims = ['rank','p'],vdims=list(dfs.columns[~dfs.columns.isin(['rank','p', 'errors'])]), label = f'({tnum[t]}) {t}' ) \
                                          .opts(size = 25, color = d[t], marker='star', line_color = 'black', tools=['hover']) #
            fig_v = fig_v*hv.Points(dfs, kdims = ['b','p'],vdims=list(dfs.columns[~dfs.columns.isin(['b','p', 'errors'])]), label = f'({tnum[t]}) {t}' ) \
                                          .opts(size = 25, color = d[t], marker='star', line_color = 'black', tools=['hover']) #
        fig = fig*hv.Labels(qtltable[['rank', 'p', 'traitnum']],  ['rank', 'p'],vdims=['traitnum']).opts(text_font_size='5pt', text_color='black')
        fig_v = fig_v*hv.Labels(qtltable[['b', 'p', 'traitnum']],  ['b', 'p'],vdims=['traitnum']).opts(text_font_size='5pt', text_color='black')
        if os.path.isfile(f'{self.path}pvalthresh/PVALTHRESHOLD.csv') and add_pval_thresh:
            qq = subsample.query('p>2.5')
            lr = sm.OLS(qq.p, qq.rnk.map(np.log)).fit()
            conf = lr.conf_int(0.05).rename({0: 'lower',1: 'upper' }, axis = 1)
            xr = np.linspace(2.5, 8, 1000)
            fig = fig*hv.Area((xr, conf.loc['rnk', 'lower']*np.log(xr) ,conf.loc['rnk', 'upper']*np.log(xr)), vdims=['y', 'y2'])
            
        fig.opts( shared_axes=False, frame_width=500, frame_height=500, title = f'Q-Q plot',show_legend=False, legend_position='right', xlabel = 'expected', ylabel = 'observed')
        fig_v.opts( shared_axes=False, frame_width=500, frame_height=500, title = f'volcano plot',show_legend=True, 
                   legend_position='right', xlabel = 'beta', ylabel = '-log10pvalue')
        
        fig = (fig.opts(ylim = yrange, xlim = xrange) + fig_v.opts(ylim = yrange))
        if save:
            hv.save(fig, f'{self.path}images/qqplot.png')
            hv.save(fig, f'{self.path}images/qqplot.html')
        return fig
    
    def _make_umap3d_figure(self, ret = 'figure'):
        rmcols = lambda x: x not in ['regressedlr_pc1', 'regressedlr_pc2', 'regressedlr_pc3', 'regressedlr_umap1', 'regressedlr_umap2', 'regressedlr_umap3','regressedlr_pca_clusters' ,'regressedlr_umap_clusters'] and  '_just_' not in x
        cols3d = [x for x in self.traits if rmcols(x)]
        df3d = self.df.set_index('rfid')[cols3d].dropna(how = 'all')
        umap = UMAP(n_components=3)
        df3d.loc[:, [f'UMAP{i}' for i in range(1,4)]] = make_pipeline(KNNImputer(), umap ).fit_transform(df3d)
        
        df3d['hdbscan'] = HDBSCAN(min_cluster_size=50).fit_predict(df3d[['UMAP1', 'UMAP2', 'UMAP3']])
        if ret == 'data': return df3d[['UMAP1', 'UMAP2', 'UMAP3', 'hdbscan']].rename(columns = {'hdbscan': 'umap_clusters'}).rename(lambda x: 'regressedlr_'+x, axis = 1)
        nclasses = df3d[f'hdbscan'].max()+1
        
        fig_eigen = px.scatter_3d(df3d.reset_index(), x = 'UMAP1', y='UMAP2', z = 'UMAP3', opacity=.6, hover_name = 'rfid',
                            hover_data = { i:True for i in  cols3d[:5] + [f'hdbscan']})
        fig_eigen.update_traces(marker=dict(line=dict(width=3, color='black'),  color = df3d[f'hdbscan'], 
                                            colorscale=[[0, 'rgb(0,0,0)']]+ [[(i+1)/nclasses, f"rgb{sns.color_palette('tab10')[i%10]}"] for i in range(nclasses)],
                                           colorbar=dict(thickness=10, outlinewidth=0, len = .5, title = 'hdbscan')))
        fig_eigen.update_layout( width=1200,height=1200,autosize=False, template = 'simple_white', 
                                coloraxis_colorbar_x=1.05,coloraxis_colorbar_y = .4,coloraxis_colorbar_len = .8,)
        
        #fig_eigen.write_html(f'ratacca_pred_{center}.html')
        return fig_eigen
    
    def _make_eigen3d_figure(self,  ret = 'figure'):
        rmcols = lambda x: x not in ['regressedlr_pc1', 'regressedlr_pc2', 'regressedlr_pc3', 'regressedlr_umap1', 'regressedlr_umap2', 'regressedlr_umap3','regressedlr_pca_clusters', 'regressedlr_umap_clusters']  and  '_just_' not in x
        cols3d = [x for x in self.traits if rmcols(x)]
        df3d = self.df.set_index('rfid')[cols3d].dropna(how = 'all')
        pca = PCA(n_components=3)
        df3d.loc[:, [f'PC{i}' for i in range(1,4)]] = make_pipeline(KNNImputer(), pca ).fit_transform(df3d)
        eigenvec = pd.DataFrame(pca.components_, index = [f'PC{i}' for i in range(1,4)], columns = cols3d).T
        pc1, pc2, pc3 = [f'PC{i+1}_{round(j, 2)}' for i,j in enumerate(pca.explained_variance_ratio_)]
        
        df3d['hdbscan'] = HDBSCAN(min_cluster_size=50).fit_predict(df3d[['PC1', 'PC2', 'PC3']])
        if ret == 'data': return df3d[['PC1', 'PC2', 'PC3', 'hdbscan']].rename(columns = {'hdbscan': 'pca_clusters'}).rename(lambda x: 'regressedlr_'+x, axis = 1)
        nclasses = df3d[f'hdbscan'].max()+1
        fig_eigen = px.scatter_3d(df3d.reset_index(), x = 'PC1', y='PC2', z = 'PC3', opacity=.6, hover_name = 'rfid',
                            hover_data = { i:True for i in  cols3d[:5] + [f'hdbscan']})
        fig_eigen.update_traces(marker=dict(line=dict(width=3, color='black'),  color = df3d[f'hdbscan'], 
                                            colorscale=[[0, 'rgb(0,0,0)']]+ [[(i+1)/nclasses, f"rgb{sns.color_palette('tab10')[i%10]}"] for i in range(nclasses)],
                                           colorbar=dict(thickness=10, outlinewidth=0, len = .5, title = 'hdbscan')))
        for name, i in eigenvec.iterrows():
            vector = fig_eigen.add_trace(go.Scatter3d( x = [0,i.PC1],y = [0,i.PC2],z = [0,i.PC3], name = name.replace('regressedlr_', ''),
                                   marker = dict( size = 7,color = "black", symbol= 'diamond-open'), showlegend=False,
                                   line = dict( color = "black",width = 6)))
        fig_eigen.update_layout( width=1200,height=1200,autosize=False, template = 'simple_white', 
                                scene=go.layout.Scene(
                                    xaxis=go.layout.scene.XAxis(title=pc1),
                                    yaxis=go.layout.scene.YAxis(title=pc2),
                                    zaxis=go.layout.scene.ZAxis(title=pc3)),
                                coloraxis_colorbar_x=1.05,coloraxis_colorbar_y = .4,coloraxis_colorbar_len = .8,)    
        return fig_eigen

    def andrews_curves_plot(self,traits= ''):
        from hvplot import andrews_curves
        if not len(traits): traits = self.traits
        else: 
            traits = list(map(lambda x: 'regressedlr_' + x  if ('regressedlr' not in x) else x , traits))
        aa = self.df[self.traits]   
        aa['hdbscan'] = 'class_'+ pd.Series(HDBSCAN().fit_predict(aa)).map(lambda x: x+1 if x>= 0 else 'noclass').astype(str).astype(str)
        return andrews_curves(aa, class_column='hdbscan').opts(width = 1000, height = 800)

    def make_genes_in_range_mk_table(self, path = ''):
        path = ''
        if not len(path): 
            if not os.path.isfile(f"{self.path}results/qtls/genes_in_range.csv"): self.locuszoom2()
            genes_in_range = pd.read_csv(f"{self.path}results/qtls/genes_in_range.csv")
        else: genes_in_range = pd.read_csv(path)
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        _genecardmk = lambda gene:f'[genecard](https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene})' if not pd.isna(gene) else ''
        _gwashubmk = lambda gene : f'[gwashub](https://www.ebi.ac.uk/gwas/genes/{gene})' if not pd.isna(gene) else ''
        _twashubmk = lambda gene : f'[twashub](http://twas-hub.org/genes/{gene.upper()})' if not pd.isna(gene) else ''
        _genecupmk = lambda gene : f'[genecup](https://genecup.org/progress?type=brain&type=addiction&type=drug&type=function&type=psychiatric&type=cell&type=stress&type=GWAS&query={gene})'\
                                   if not pd.isna(gene) else ''
        _genebassmk = lambda ensid: f'[genebass](https://app.genebass.org/gene/{ensid}?burdenSet=pLoF&phewasOpts=1&resultLayout=full)' if not pd.isna(ensid) else ''
        _rgdhtmk = lambda gene: f'[rgd](https://rgd.mcw.edu/rgdweb/report/gene/main.html?id={gene.split(":")[-1]})' if not pd.isna(gene) else ''
        def tryloc(x):
            try: return f"{x['chr']}:{x['start']}-{x['end']}"
            except: return 'remove'
        def distance2snp(snp, gene):
            s = str(snp).split(':')
            g = str(gene).split(':')
            if s[0] != g[0]: return 'remove'
            dist = min([abs(int(x) - int(s[1])) for x in g[1].split('-')])
            return 'remove' if dist > 6e6 else f"{round(dist/1e6, 2)}Mb"
        genes_in_range =genes_in_range.drop({'score', 'phase','source','biotype', 'Chr','start','end','strand',
                                             'transcript_id','gene_id', 'exon_id', 'exon_number', 'webpage', 'markdown'} \
                                            & set(genes_in_range.columns), axis = 1).drop_duplicates(subset=['gene', 'SNP_origin'])
        
        gene_annt = query_gene(genes_in_range.gene, self.taxid)[[ 'name', 'symbol', 'AllianceGenome', 'ensembl', 'entrezgene']]
        gene_annt['ensembl'] = gene_annt['ensembl'].map(lambda x: x[0] if isinstance(x, list) else x).map(lambda x: x['gene'] if isinstance(x, dict) else x)
        gene_annt = gene_annt
        
        genes_in_range = genes_in_range.rename({'gene': 'symbol'}, axis =1).merge(gene_annt.drop('symbol', axis = 1), 
                                                                                  left_on = ['symbol','description'], right_on =['query', 'name'], how = 'left')
        hgenes = query_gene(genes_in_range.symbol, species = 'human')['ensembl'].map(lambda x: x[0] if isinstance(x, list) else x).map(lambda x: x['gene'] if isinstance(x, dict) else x).dropna()
        hgenes = hgenes[~hgenes.index.duplicated(keep='first')]
        hgenes = hgenes.map(_genebassmk)
        genes_in_range = genes_in_range.merge(hgenes.rename('ensemblh'), left_on = 'symbol', right_on = 'query', how = 'left')
        genes_in_range['links'] = genes_in_range.apply(lambda r: ','.join([_genecardmk(r.symbol),_gwashubmk(r.symbol), _genecupmk(r.symbol),  _twashubmk(r.symbol)]) , axis = 1 )
        genes_in_range['links'] = (genes_in_range.links + ',' + genes_in_range.ensemblh.fillna('')).str.replace(',,', '').str.strip(',').str.replace('nan', '')
        genes_in_range = genes_in_range.drop('ensemblh', axis = 1)
        if self.species == 'rattus_norvegicus':
            genes_in_range['links'] = genes_in_range.apply(lambda r:f'{r.links},{_rgdhtmk(r.AllianceGenome)}'.replace('nan', '').replace(',,', '').strip(',') , axis = 1)
        genes_in_range = genes_in_range.loc[~genes_in_range.name.fillna('').str.contains('uncharacterized LOC|^Trna')]
        genes_in_range = genes_in_range.loc[~(genes_in_range.symbol.fillna('').str.contains('uncharacterized LOC|^Trna') & genes_in_range.name.isna())]
        genes_in_range = genes_in_range.loc[:, ~genes_in_range.columns.str.contains('Unnamed:')]#.drop_duplicates()
        genes_in_range.head()
        genes_in_range['distance'] = genes_in_range.apply(lambda x: distance2snp(x.SNP_origin, x.genomic_pos), axis = 1)
        return genes_in_range.set_index('SNP_origin').dropna(subset = ['name'])[['genomic_pos','distance','symbol', 'name','AllianceGenome','ensembl','entrezgene','links']]

    def report(self, round_version: str = '10.3.2', covariate_explained_var_threshold: float = 0.02, gwas_version='current', 
               sorted_gcorr = True, add_gwas_latent_space= 'nmf', add_experimental = True):
        printwithlog('generating report...')
        with open(f'{self.path}genotypes/parameter_thresholds.txt', 'r') as f: 
            out = f.read()
            params = {x:re.findall(f"--{x} ([^\n]+)", out)[0] for x in ['geno', 'maf', 'hwe']}
        with open(f'{self.path}genotypes/genotypes.log') as f:
            out = f.read()
            params['snpsb4'] = re.findall(f"(\d+) variants loaded from .bim file.", out)[0]
            params['snpsafter'], params['nrats'] = re.findall("(\d+) variants and (\d+) samples pass filters and QC.", out)[0]
            params['removed_geno'], params['removedmaf'], params['removedhwe'] = \
                   (~pd.read_parquet(f'{self.path}genotypes/snpquality.parquet.gz')[['PASS_MISS','PASS_MAF','PASS_HWE']])\
                   .sum().astype(str)

        if round(self.threshold, 2) == round(self.threshold05, 2):
            threshtext = f'''* {self.genome_version}:{round_version} 5%: {round(self.threshold, 2)}'''
        else:
            threshtext = f'''* {self.genome_version}:{round_version} 10%: {round(self.threshold, 2)}
* {self.genome_version}:{round_version} 5% : {round(self.threshold05, 2)}'''
                    
        text_sidepanel = f"""# General Information
<hr>

Phenotype Info

* n = *{params['nrats']}*

* phenotype data: [here](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/processed_data_ready.csv)

* covariate dropboxes: [here](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/data_dict_{self.project_name}.csv) 

* phenotype statistical descriptions file: [here](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/data_distributions.html) 
<hr>

Genotype Info

* genotypes version: \n*{self.genome_version}:{round_version}*

* gwas pipeline version: \n*{gwas_version}*

* number of snps: \nbefore filter *{format(int(params['snpsb4']), ',')}*, \nafter filter *{format(int(params['snpsafter']), ',')}*

* genotype missing rate filter: < *{params['geno']}* \n(*{format(int( params['removed_geno']),',')}* snps removed)

* minor allele frequency filter: > *{params['maf']}* \n(*{format(int(params['removedmaf']), ',')}* snps removed)

* hardy-weinberg equilibrium filter: < *{params['hwe']}* \n(*{format(int(params['removedhwe']), ',')}* snps removed)

Threshold Info

{threshtext}"""
        
        template = pn.template.BootstrapTemplate(title=f'GWAS REPORT')
        # Add components to the sidebar, main, and header
        template.sidebar.extend([
        pn.pane.Alert(text_sidepanel, alert_type="primary")
        ])
        ##### adding data dictionary
        dd = pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv').fillna('')\
           [['measure', 'trait_covariate','covariates', 'description']]\
           .query("measure != ''")

        trait_d_card = ['Collaborative data dictionary google document: ', fancy_display(dd, 'data_dictionary.csv')]
        if os.path.isfile(f'{self.path}missing_rfid_list.txt'): 
            trait_d_card += [pn.Card(fancy_display(pd.read_csv(f'{self.path}missing_rfid_list.txt', dtype = str, header = None, names=['not genotyped RFIDs']), 'missing_rats.csv'), title = 'not genotyped Rats', collapsed=True)]
        template.main.append(pn.Card(*trait_d_card , title = 'Trait Descriptions', collapsed=True))
        
        explained_vars =  pd.read_csv(f'{self.path}melted_explained_variances.csv').pivot(columns = 'group', values='value', index = 'variable')
        for x in set(map(lambda x:  x.replace('regressedlr_', ''), self.traits)) - set(explained_vars.index.values):
            explained_vars.loc[x, :] = [np.nan]
        fig_exp_vars = px.imshow((explained_vars*100).round(), text_auto=True, aspect="auto", color_continuous_scale='Reds')
        fig_exp_vars.update_layout(template = 'simple_white', width=800,height=800,autosize=False)
        
        g0 = px.imshow(self.df.set_index('rfid')[self.traits].rename(lambda x: x.replace('regressedlr_', ''), axis = 1), aspect = 3, color_continuous_scale='RdBu')
        g0.update_layout(  width=1000, height=1400,autosize=False,showlegend=False, template='simple_white',plot_bgcolor='black')
        g1df = self.df.set_index('rfid')[list(map(lambda x: x.replace('regressedlr_', ''), self.traits))]
        g1df.loc[:, :] = StandardScaler().fit_transform(g1df)
        g1 = px.imshow(g1df, aspect = 3, color_continuous_scale='RdBu')
        g1.update_layout(  width=1000, height=1400,autosize=False,showlegend=False, template='simple_white',plot_bgcolor='black')
        
        cov_text = f'''Covariates may confound the results of the analysis. Common covariates include “age”, “weight”, “coat color”, “cohort”, and “phenotyping center”. We work with individual PIs to determine which covariates should be considered. In order to “regress out” the part of the phenotypic variance that is related to known covariates, we follow the procedure of fitting a linear model that predicts the desired trait based only on the measured covariates. Then the trait is subtracted by the trait predictions generated by the linear model described above. The resulting subtraction is expected to be independent of the covariates as all the effects caused by those covariates were removed. Since this method utilizes a linear regression to remove those effects, non-linear effects of those covariates onto the traits will not be addressed and assumed to be null. In certain cases, it’s possible that accounting for too many covariates might ‘overcorrect’ the trait. To address this issue, we ‘regress out’ only the covariates that explain more than {covariate_explained_var_threshold} of the variance of the trait. This calculation is often called r^2 or pve (percent explained variance) and is estimated as cov (covariant, trait)/variance(trait). Lastly, the corrected trait is quantile normalized again, as it’s expected to follow a normal distribution. For time series regression we use the prophet package (https://facebook.github.io/prophet/) that uses a generalized additive model to decompose the timewise trend effects and covariates onto the mesurement of animal given its age. Because age might affect differently males and females, we first groupby the animals between genders before using the timeseries regression to remove covariate effects. After removing the covariate effects in with the timeseries regression, we then quantile normalize the residuals to be used for subsequent analysis.''' 
        # cov_card = pn.Card( pn.Card(cov_text, pn.pane.Plotly(fig_exp_vars), title = 'Covariate r<sup>2</sup> with traits in percent', collapsed=True),
        #                     pn.Card('Move the divider to see how the preprocessing changes the values of the data *(original - left | regressed out - right)*',\
        #                             pn.Swipe(pn.pane.Plotly(g1),pn.pane.Plotly(g0)), title = 'Changes after regressing out covariates', collapsed=True),
        #                     pn.Card(  \
        #                         pn.Card(pn.pane.Plotly(self._make_eigen3d_figure()), title = 'PCA representation of the data' , collapsed=True),\
        #                         pn.Card(pn.pane.Plotly(self._make_umap3d_figure()), title = 'UMAP representation of the data' , collapsed=True),\
        #                            title = 'EXPERIMENTAL' , collapsed=True),
        #                    title = 'Preprocessing', collapsed=True)
        self._make_eigen3d_figure().write_html(f'{self.path}images/traitPCA.html')
        self._make_umap3d_figure().write_html(f'{self.path}images/traitUMAP.html')
        cov_card = pn.Card( pn.Card(cov_text, pn.pane.Plotly(fig_exp_vars), title = 'Covariate r<sup>2</sup> with traits in percent', collapsed=True),\
                           title = 'Preprocessing', collapsed=True)
        template.main.append(cov_card)
        try:panel_genetic_pca = self.make_panel_genetic_PCA()
        except:
            printwithlog('genetic pca failed')
            panel_genetic_pca = pn.Card('failure calculating genomic PCA', title = 'Genomic PCA', collapsed = True)
        #template.main.append(panel_genetic_pca)
        
        gcorrtext = '''# *Genetic Correlation Matrix*

Genetic correlation is a statistical concept that quantifies the extent to which two traits share a common genetic basis. The estimation of genetic correlation can be accomplished using Genome-wide Complex Trait Analysis (GCTA), a software tool that utilizes summary statistics from genome-wide association studies (GWAS) to estimate the genetic covariance between pairs of traits. GCTA implements a method that decomposes the total phenotypic covariance between two traits into genetic and environmental components, providing an estimate of the genetic correlation between them. This approach allows researchers to examine the degree of shared genetic architecture between traits of interest and gain insights into the biological mechanisms underlying complex traits and diseases. 

For the figure, the upper triangle represents the genetic correlation (ranges from [-1:1]), while the lower triangle represents the phenotypic correlation. Meanwhile the diagonal displays the heritability (ranges from [0:1]) of the traits. Hierarchical clustering is performed using [scipy's linkage function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) with the genetic correlation. Dendrogram is drawn using [scipy dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html) where color coding for clusters depends on a distance threshold set to 70% of the maximum linkage distance. Asterisks means that test failed, for genetic relationship the main failure point is if the 2 traits being tested are colinear, while for the phenotypic correlation it's due to no overlapping rats between the 2 traits.'''
        bokehgcorrfig = self.make_genetic_correlation_figure(order = 'sorted' if sorted_gcorr else 'cluster' ,save=True)
        gcorrfig = pn.pane.HoloViews(bokehgcorrfig, max_width=1200, max_height=1200, width = 1200, height = 1200)
        #gcorrfig = pn.pane.PNG(f'{self.path}images/genetic_correlation_matrix2{"_sorted" if sorted_gcorr else "cluster"}.png', max_width=1200, max_height=1200, width = 1200, height = 1200)
        gcorr = pd.read_csv(f"{self.path}results/heritability/genetic_correlation_melted_table.csv", index_col=0).applymap(lambda x: round(x, 3) if type(x) == float else x.replace('regressedlr_', ''))
        gcorr = fancy_display(gcorr, 'genetic_correlation.csv')
        template.main.append( pn.Card(gcorrtext, gcorrfig, pn.Card(gcorr, title = 'tableView', collapsed=True),title = 'Genetic Correlation', collapsed=True))
        heritext = '''# **SNP Heritability Estimates h<sup>2</sup>** 

SNP heritability (often reported as h<sup>2</sup> ) is the fraction of phenotypic variance that can be explained by the genetic variance measured from the Biallelic SNPS called by the genotyping pipeline. It is conceptually similar to heritability estimates that are obtained from panels of inbred strains (or using a twin design in humans), but SNP heritability is expected to be lower.  Specifically, this section shows the SNP heritability (“narrow-sense heritability”) estimated for each trait by GCTA-GREML, which uses the phenotypes and genetic relatedness matrix (GRM) as inputs. Traits with higher SNP heritability are more likely to produce significant GWAS results. It is important to consider both the heritability estimate but also the standard error; smaller sample sizes typically have very large errors, making the results harder to interpret. 
Note that Ns for each trait may differ from trait to trait due to missing data for each trait. 

Column definitions: 

* trait: trait of interest
* N: number of samples (rats) containing a non-NA value for this trait
* heritability: quantifies the proportion of phenotypic variance of a trait that can be attributed to genetic variance
* heritability_se: standard error, variance that is affected by N and the distribution of trait values
* pval: probability of observing the estimated heritability under the NULL hypothesis (that the SNP heritability is 0)'''
        
        herfig = pn.pane.Plotly(plotly_read_from_html(f'{self.path}images/heritability_sorted.html'))
        her = pd.read_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t')\
             .set_axis(['trait', 'gen_var', 'env_var', 'phe_var', 'heritability', 'likelihood', 'lrt', 'df', 'pval', 'n', 'heritability_se'], axis = 1).drop(['env_var', 'lrt', 'df'],axis = 1)
        her.trait = her.trait.str.replace('regressedlr_', '')
        her = fancy_display(her, 'heritablitity.csv')
        template.main.append( pn.Card(heritext, herfig, her, title = 'Heritability', collapsed=True))
        qtls = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv').query('QTL').rename({'p':'-Log10(p)', 'b':'beta', 'se': 'betase', 'af': 'Freq', 'SNP': 'TopSNP'}, axis = 1).round(3)
        founder_ids = set(self.foundersbimfambed[1].fid.sort_values().to_list()) if len(self.foundersbimfambed) else set()
        qtlstext = f'''
# **Summary of QTLs** 

The genome-wide significance threshold (-log10p): 

{threshtext}

The values shown in the table below pass the {self.genome_version}:{round_version} subjective threshold. 

  Quantitative trait loci (QTLs) are regions in the genome that contain single nucleotide polymorphisms (SNPs) that correlate with a complex trait.
If there are multiple QTLs in a given chromosome, then the top SNP from the most significant QTL is used as a covariate for another GWAS analysis within the chromosome.  If the analysis results in another SNP with a p-value that exceeds the permutation-derived threshold then it is considered an independent locus. This continues until no more QTLs are devoted within a given chromosome. This method is described in details in (Chitre et al., 2020)


Column definitions: 


* TopSNP: SNPs with lowest p-value whithin an independent QTL. SNP name is defined by the location of the top SNP on the chromosome. Read it as follows chromosome: position, so 10:10486551 would be chromosome 10, location on the chromosome at 10486551
* af: frequency of the TopSNP in the rats used for this study
* beta: effect size of topSNP
* betase: standard error of effect size of topSNP
* -Log10(p): statistical significance of the association between the trait variability and the top SNP, displayed as -log10(p-value). The log-transformed p-value used in all figures and tables in this report
* trait: trait in which the snp was indentified
{f"* {', '.join(list(founder_ids)[:100])}: genotypes of founders at the topSNP" if len(founder_ids) else ''}'''
        
        
        qtls = qtls[['TopSNP','Freq','beta','betase','-Log10(p)','significance_level','trait'] + \
                    list(founder_ids & set(qtls.columns)) ]
        template.main.append( pn.Card(qtlstext, fancy_display(qtls, 'qtls.csv'), title = 'QTL', collapsed=True))
        
        
        porcupinetext = f'''# **Porcupine Plot**
        
Porcupine plot is a graphical tool that combines multiple Manhattan plots, each representing a single trait, into a single plot. The resulting plot provides a visual representation of the regions of the genome that influence multiple traits, enabling researchers to identify genetic loci that have pleiotropic effects. These plots allow for a quick and efficient analysis of multiple traits simultaneously. For the porcupine plots shown below, only traits with at least one significant QTL are shown.'''
        skipmanhattan = len(set(map(lambda x: x.replace('regressedlr_',''),self.traits)) \
                - set(map(lambda x: basename(x).replace('.png', ''), glob(f'{self.path}images/manhattan/*.png'))) )
        skipmanhattan = True if not skipmanhattan else False
        porcfig = pn.pane.HoloViews(self.porcupineplotv2(display_figure = False, skip_manhattan=skipmanhattan ),max_width=1200, max_height=600, width = 1200, height = 600)
        qqplotfig = pn.pane.HoloViews(self.qqplot(add_pval_thresh= True),max_width=1200, max_height=1200, width = 900, height = 900)
        #porcfig = pn.pane.PNG(f'{self.path}images/porcupineplot.png', max_width=1000, max_height=600, width = 1000, height = 600)
        template.main.append( pn.Card(porcupinetext, porcfig,qqplotfig,  title = 'Porcupine Plot', collapsed=True))
        
        manhattantext = f'''# **Manhattan plots (for significant QTLS)**
    
These Manhattan plots show QTLs that genome-wide significance threshold of: 

{threshtext}

The Manhattan plot displays the p-values of each SNP sampled, with the aim of finding specific SNPs that pass the significance threshold. The x-axis shows chromosomal position and the y-axis shows -log10 of the p-value. The GWAS analysis uses a linear mixed model implemented by the software package GCTA (function MLMA-LOCO) using dosage and genetic relatedness matrices (GRM) to account for relatedness between individuals in the population. The analysis also employs Leave One Chromosome Out (LOCO) to avoid proximal contamination. 

The genomic significance threshold is the genome-wide significance threshold calculated using permutation test, and the genotypes at the SNPs with p-values exceeding that threshold are considered statistically significantly associated with the trait variance. Since traits are quantile-normalized, the cutoff value is the same across all traits. QTLs are determined by scanning each chromosome for at least a SNP that exceeds the calculated permutation-derived threshold.

To control type I error, we estimated the significance threshold by a permutation test, as described in (Cheng and Palmer, 2013).'''
        
        manhatanfigs = [pn.Card(pn.pane.PNG(f'{self.path}images/manhattan/{trait}.png', max_width=1000, 
                     max_height=600, width = 1000, height = 600), fancy_display(qtls.query('trait == @trait')), title = trait, collapsed = True) for trait in qtls.trait.unique()]
        
        manhatanfigs2 = [pn.Card(pn.pane.PNG(f'{self.path}images/manhattan/{trait}.png', max_width=1000, 
                     max_height=600, width = 1000, height = 600), title = trait, collapsed = True) \
                        for trait in set(map(lambda x: x.replace('regressedlr_', ''), self.traits)) - set(qtls.trait)]
        
        
        
        template.main.append( pn.Card(manhattantext, 
                                      pn.Card(*manhatanfigs, title='Plots with QTLs', collapsed=True),
                                      pn.Card(*manhatanfigs2, title='Plots without QTLs', collapsed=True),
                                      title = 'Manhattan Plots', collapsed=True))
        db_vals_t = pd.concat(pd.read_parquet(x).assign(phewas_file = x) for x in self.phewas_db.split(',')).reset_index(drop= True)
        PROJECTLIST = '\n'.join(list(map(lambda x: '*  ' + x, db_vals_t['project'].unique())))
        eqtlstext = '' if self.species not in ['rattus_norvegicus'] else f'''## Gene Expression changes:

### expression QTL (eQTLs) 
We examine if the identified SNP does significant alter the gene expression of genes in cis, possibly describing a pathway in which the SNP impact the phenotype. We also examine SNPs within 3 Mb window and a correlation above 0.6 of the topSNP from the current study, in case the selected SNP is in high LD with another SNP that can alter the gene expression in cis.

Defining columns:


* SNP_eqtldb: SNP from eQTL database in LD with topSNP detected from GWAS 
* -Log10(p)_eqtldb: -log10(p-value) for the association between the eqtlSNP and the gene in cis described in the column Ensembl_gene
* tissue: tissue in which the gene expression patterns were measured
* R2: correlation between SNP from eQTL database (P-value threshold of 1e-4) and the topSNP for the current study
* DP: Dprime measure of linkage disequilibrium (correlation between the SNPs adjusted by the maximum possible correlation between those SNPs)
* gene: Gene where the SNP_eqtldb influences the gene expression
* slope: Effect size of the SNP_eqtldb onto the Ensembl_gene
* af: allele frequency of the SNP_eqtldb


### splice QTL (sQTLs) 
We examine if the identified SNP does significant alter the splicing patterns of genes in cis, possibly describing a pathway in which the SNP impact the phenotype. We also examine SNPs within 3 Mb window and a correlation above 0.6 of the topSNP from the current study, in case the selected SNP is in high LD with another SNP that can alter the splicing in cis.

Defining columns:


* SNP_sqtldb: SNP from sQTL database in LD with topSNP detected from GWAS 
* -Log10(p)_sqtldb: -log10(p-value) for the association between the sqtlSNP and the gene in cis described in the column Ensembl_gene
* tissue: tissue in which the splice patterns were measured
* R2: correlation between SNP from sQTL database (P-value threshold of 1e-4) and the topSNP for the current study
* DP: Dprime measure of linkage disequilibrium (correlation between the SNPs adjusted by the maximum possible correlation between those SNPs)
* gene: Gene where the SNP_sqtldb influences the gene expression
* slope: Effect size of the SNP_sqtldb onto the Ensembl_gene
* af: allele frequency of the SNP_sqtldb'''
        
        regional_assoc_text = f'''
# **Regional Association plots**

Where Manhattan Plots show SNPs associated with all the chromosomes, a Regional Association Plot zooms in on particular regions of a chromosome that contains a QTL for a given trait. The x-axis represents the position on a chromosome (in Mb) and the y-axis shows the significance of the association (-log10 p-value). The individual points represent SNPs, where the SNP with the lowest p-value (“top SNP”) is highlighted in purple. The colors represent the correlation, or level of linkage disequilibrium (LD), between the topSNP and the other SNPs. The LD was measured with [plink](https://www.cog-genomics.org/plink/1.9/ld) (raw inter-variant allele count squared correlations).

Linkage disequilibrium intervals for the remaining QTLs are determined by finding markers with at least r2=0.6 correlation with the peak marker.

## Phenotype Wide Association Study (PheWAS): 

These tables report the correlation between the topSNP and traits from other studies in HS rats conducted by the center. Use information from these tables to better understand what additional phenotypes this interval may be associated with. 

The PheWAS table examines the association between the topSNP for this phenotype and all other topSNPs that were mapped within a 3 Mb window of the topSNP from the current study and a correlation above 0.6. Instead of showing association of the topSNP with other traits like in the first table, the second table shows significant association identified for other traits within the nearby chromosomal interval.

Projects included in the PheWAS analysis (see project_name column in PheWAS tables). 

{PROJECTLIST}

Defining columns: 

* SNP_PheDb: SNP from Phewas database in LD with topSNP detected from GWAS  
* -Log10(p)PheDb: -log10(p-value) for trait from the associated study
* trait_PheDb: trait from the associated study with the same topSNP
* project: project from which the trait was studied
* trait_description_PheDb: trait definition
* R2: correlation between SNP from phewas database (P-value threshold of 1e-4) and the topSNP for the current study
* DP: Dprime measure of linkage disequilibrium (correlation between the SNPs adjusted by the maximum possible correlation between those SNPs)

{eqtlstext}'''
        
        ann = pd.read_csv(f'{self.path}results/qtls/possible_causal_snps.tsv', sep = '\t').drop(['A1','A2', 'featureid', 'rank', 
                                                                               'cDNA_position|cDNA_len','CDS_position|CDS_len',
                                                                               'Protein_position|Protein_len','distancetofeature'], errors = 'ignore' ,axis = 1)\
                 .query("putative_impact in ['MODERATE', 'HIGH']").sort_values('putative_impact')
        ann['p'] = -np.log10(ann.p)
        ann.rename({'p':'-Log10(p)'},axis=1,  inplace=True)
        
        phewas = pd.read_csv(f'{self.path}results/phewas/pretty_table_both_match.tsv', sep = '\t')
        phewas = phewas.loc[phewas.R2.replace('Exact match SNP', 1.1).astype(float).sort_values(ascending = False).index]\
                       .rename({'p_PheDb': '-Log10(p)PheDb'}, axis =1).drop(['round_version', 'uploadeddate'], axis =1)\
                                 .drop_duplicates(['SNP_QTL', 'SNP_PheDb','trait_QTL','trait_PheDb','project'])
        
        eqtl = pd.read_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv')\
                 .rename({'-log10(P-value)':'-Log10(p)', '-log10(pval_nominal)': '-Log10(p)_eqtldb' }, axis =1)
        
        
        sqtl = pd.read_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv')\
                 .rename({'-log10(P-value)':'-Log10(p)', '-log10(pval_nominal)': '-Log10(p)_sqtldb' }, axis = 1)
        
        genes_in_range = pd.read_csv(f"{self.path}results/qtls/genes_in_range.csv")
        genes_in_range2 = self.make_genes_in_range_mk_table().drop_duplicates()
        
        out = [regional_assoc_text]
        for index, row in tqdm(list(qtls.iterrows())):
            texttitle = f"Trait: {row.trait} SNP: {row.TopSNP}\n"
            #row_desc = fancy_display(row.to_frame().T)
            row_desc = pn.pane.Markdown(row.to_frame().T.fillna('').to_markdown())
            snp_doc = row.TopSNP.replace(":", '_')
            if row.TopSNP in genes_in_range2.index:
                giran = pn.Card(pn.pane.Markdown(genes_in_range2.loc[[row.TopSNP]].fillna('').to_markdown()), title = 'Gene Links', collapsed = False, min_width=500)
            else: 
                giran = pn.Card(f'no Genes in section for SNP {row.TopSNP}', title = 'Gene Links', collapsed = False, min_width=500)
            #lzplot = pn.pane.Plotly(plotio.read_json(f'{self.path}images/lz/lz__{row.trait}__{snp_doc}.json'))
            # lzplot = pn.pane.PNG(f'{self.path}images/lz/lz__{row.trait}__{snp_doc}.png',  max_width=1000, max_height=600, width = 1000, height = 600)
            # lzplot2 = pn.pane.PNG(f'{self.path}images/lz/6Mb/lz__{row.trait}_6Mb__{snp_doc}.png',  max_width=1000, max_height=600, width = 1000, height = 600)
            lzplot = pn.pane.PNG(f'{self.path}images/lz/r2thresh/lzi__{row.trait}__{snp_doc}.png',  max_width=1200, max_height=1200, width = 1200, height = 800)
            lzplot2 = pn.pane.PNG(f'{self.path}images/lz/minmax/lzi__{row.trait}__{snp_doc}.png',  max_width=1200, max_height=1200, width = 1200, height = 800)
            lztext = pn.pane.Markdown(f'[interactive version](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/images/lz/minmax/lzi__{row.trait}__{snp_doc}.html)')
            boxplot = pn.pane.PNG(f'{self.path}images/boxplot/boxplot{snp_doc}__{row.trait}.png', max_width=800, max_height=400, width = 800, height = 400)
        
            cau_title = pn.pane.Markdown(f"### Coding variants: {row.trait} {row.TopSNP}\n")
            try:cau = ann.query('SNP_qtl == @row.TopSNP and trait == @row.trait')[['SNP','Freq','b','-Log10(p)','R2','DP', 'annotation',
                                                                                   'putative_impact','gene','HGVS.c', 'HGVS.p']].drop_duplicates().sort_values('putative_impact')
            except:cau = ann.query('SNP_qtl == @row.TopSNP and trait == @row.trait').drop_duplicates().sort_values('putative_impact')
            if cau.shape[0]: cau = fancy_display(cau, f'CodingVariants_{row.trait}{row.TopSNP}.csv'.replace(':', '_'))
            else: cau = pn.pane.Markdown(' \n HIGH or MODERATE impact variants absent \n   \n')

            phewas_section = []
            for idx, tdf in phewas.groupby('phewas_file'):
                pboth_title = pn.pane.Markdown(f"### PheWAS: Lowest P-values for other phenotypes in a 3Mb window of {row.trait} {row.TopSNP} for {basename(tdf.phewas_file.iloc[0])}\n")
                pbothtemp = tdf.query('SNP_QTL == @row.TopSNP and trait_QTL == @row.trait')[['SNP_PheDb','-Log10(p)PheDb','R2', 'DP' ,'trait_PheDb', 'project', 'trait_description_PheDb']].drop_duplicates()
                if pbothtemp.shape[0]: pbothtemp = fancy_display(pbothtemp.fillna(''), f'phewas_{row.trait}{row.TopSNP}.csv'.replace(':', '_'))
                else: pbothtemp = pn.pane.Markdown(f' \n SNPS were not detected for other phenotypes in 3Mb window of topSNP  \n   \n')
                phewas_section += [pboth_title,pbothtemp]
        
            eqtl_title = pn.pane.Markdown(f"### eQTL: Lowest P-values for eqtls in a 3Mb window of {row.trait} {row.TopSNP}\n")
            eqtltemp = eqtl.query(f'SNP == "{"chr"+row.TopSNP}" and trait == "{row.trait}"').rename({'Ensembl_gene': 'gene'}, axis = 1)\
                           [['SNP_eqtldb', '-Log10(p)_eqtldb', 'tissue', 'R2', 'DP', 'gene', 'gene_id', 'slope', 'af']].drop_duplicates()
            if eqtltemp.shape[0]: eqtltemp = fancy_display(eqtltemp.fillna(''),f'eqtl_{row.trait}{row.TopSNP}.csv'.replace(':', '_'))
            else: eqtltemp = pn.pane.Markdown(' \n SNPS were not detected for eQTLs in 3Mb window of trait topSNP  \n   \n')
        
            sqtl_title = pn.pane.Markdown(f"### sQTL: Lowest P-values for splice qtls in a 3Mb window of {row.trait} {row.TopSNP}\n")
            sqtltemp = sqtl.query(f'SNP=="{"chr"+row.TopSNP}" and trait == "{row.trait}"').rename({'Ensembl_gene': 'gene'}, axis = 1)\
                           [['SNP_sqtldb', '-Log10(p)_sqtldb', 'tissue','R2', 'DP', 'gene','gene_id' , 'slope', 'af']].drop_duplicates()
            if sqtltemp.shape[0]: sqtltemp = fancy_display(sqtltemp.fillna(''), f'sqtl_{row.trait}{row.TopSNP}.csv'.replace(':', '_'))
            else: sqtltemp = pn.pane.Markdown(' \n  SNPS were not detected for sQTLs in 3Mb window of trait topSNP  \n   \n')

            dt2append = [giran, cau_title, cau] + phewas_section #+[phe_title,  phetemp, phew_title, phewtemp]
            if self.genome_accession in ['GCF_015227675.2', 'GCF_000001895.4']:
                dt2append += [eqtl_title, eqtltemp,sqtl_title, sqtltemp]
        
            out += [pn.Card(*[row_desc,lzplot,lzplot2,lztext,boxplot,pn.Card(*dt2append, title = 'tables', collapsed = False)]   ,title = texttitle, collapsed = True)]
            #
            
        template.main.append(pn.Card(*out, title = 'Regional Association Plots', collapsed = True))
        if add_experimental:
            geneenrichmenttext = f'''# **GeneEnrichment Plot**

This network plot shows the connectivity between traits through: 
* 1) the significant snps of each trait 
* 2) the closest gene associated with each SNP on (1)
* 3) for each trait we perform gene enrichment of the genes for associated with all the top SNPS linked to the trait
* 4) for each trait we perform gene enrichment of all the genes within all the qtls ideintified for this trait '''
            goeafig = pn.pane.HoloViews(self.GeneEnrichment_figure(),width=1000,height=1000, min_height=1000, min_width=1000)
            goeacard =  pn.Card(geneenrichmenttext, goeafig, title = 'Gene Enrichment', collapsed=True)
        
            graphviewtext = f'''# **Graph View Plot**
        
This network plot shows the connectivity between traits, topSNPS, eQTLS, sQTLS, annotated variants '''
            pjgraph = pn.pane.HoloViews(self.project_graph_view(),width=1000,height=1000, min_height=1000, min_width=1000)
            projviewcard = pn.Card(graphviewtext, pjgraph, title = 'Project Graph View', collapsed=True)
            add2card = [panel_genetic_pca,projviewcard,goeacard] #goeacard
            if add_gwas_latent_space != 'none':
                if add_gwas_latent_space in [True, 1, '1', 'True']: add_gwas_latent_space = 'nmf'
                add2card += [self.GWAS_latent_space(method = add_gwas_latent_space)[0]]
                add2card += [self.steiner_tree()]
            template.main.append( pn.Card(*add2card, title = 'Experimental', collapsed=True))
        #db_vals_t = pd.concat(pd.read_parquet(x).query(f'p < {pval_threshold}').assign(phewas_file = x) for x in self.phewas_db.split(',')).reset_index(drop= True)
        faqtable = db_vals_t[['project' ,'trait']].value_counts().to_frame().rename({0: 'number of SNPs'}, axis =1).reset_index()
        faqtext = f'''Do the traits look approximately normally distributed? 
    
* Our pipeline performs a quantile normalization, which is a transformation that preserves the rank of each subject but reassigns values such that the final distribution is perfectly normally distributed. When two or more subjects have identical values, the ‘tie’ is broken randomly (with a spefic random seed of 42), if there is a large proportion of ties, then that part of the distribution is random noise, which is bad (however, in our defense, there are no good solutions when multiple values are ties). 
    
Are there extreme outliers that might be errors? 
    
* By the time we send this report we would typically have tried to identify outliers/errors, but it is possible we’ve missed some so please let us know if you have concerns. 
    
Are there sex differences? 
    
* We regress out the effect of sex, so the GWAS essentially assumes that alleles have similar effects of the trait in males and females. This means we would not detect an interaction (e.g. an allele that affects a trait only in males). While it is possible to do GWAS for males and females separately, we don’t do this by default because the loss of power makes it hard to interpret the results. If sex differences are of major interest we can discuss ways of examining these.
    
    Which traits are included in the PheWAS database:'''
        faqtext = pn.pane.Markdown(faqtext)
        
        template.main.append(pn.Card(faqtext, fancy_display(faqtable, 'list_of_traits.csv'), title = 'FAQ', collapsed = True))
        
        reftext = '''* Chitre AS, Polesskaya O, Holl K, Gao J, Cheng R, Bimschleger H, Garcia Martinez A, George T, Gileta AF, Han W, Horvath A, Hughson A, Ishiwari K, King CP, Lamparelli A, Versaggi CL, Martin C, St Pierre CL, Tripi JA, Wang T, Chen H, Flagel SB, Meyer P, Richards J, Robinson TE, Palmer AA, Solberg Woods LC. Genome-Wide Association Study in 3,173 Outbred Rats Identifies Multiple Loci for Body Weight, Adiposity, and Fasting Glucose. Obesity (Silver Spring). 2020 Oct;28(10):1964-1973. doi: 10.1002/oby.22927. Epub 2020 Aug 29. PMID: 32860487; PMCID: PMC7511439.'''
        template.main.append(pn.Card(reftext, title = 'References', collapsed = True))
        template.header.append(f'## {self.project_name}')
        template.save(f'{self.path}results/gwas_report.html', resources=INLINE)
        bash(f'''cp {self.path}results/gwas_report.html {self.path}results/gwas_report_{self.project_name}_round{round_version}_threshold{round(self.threshold,2)}_n{self.df.shape[0]}_date{datetime.today().strftime('%Y-%m-%d')}_gwasversion_{gwas_version}.html''')
        #printwithlog(f'{destination.replace("/tscc/projects/ps-palmer/s3/data", "https://palmerlab.s3.sdsc.edu")}/{self.project_name}/results/gwas_report.html')
    
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
            
    def copy_results(self, destination: str = '/projects/ps-palmer/s3/data/tsanches_dash_genotypes/gwas_results', make_public = True, tscc = 2):
        '''
        zip project remove folders if remove folders is True
        '''
        
        print(f'copying {self.project_name} to {destination}')
        if tscc == 2: 
            destination = '/tscc' + destination
        os.makedirs(f'{destination}', exist_ok = True)
        out_path, pjname = (self.path, '') if self.path else ('.', f'/{self.project_name}')
        
        bash(f'cp -r {out_path} {destination}{pjname}')
        print('waiting 1 min for copying files...')
        sleep(60*1)
        if make_public:
            bash('/tscc/projects/ps-palmer/tsanches/mc anonymous set public /tscc/projects/ps-palmer/s3/data/tsanches_dash_genotypes --recursive')
            printwithlog(f'{destination.replace("/tscc/projects/ps-palmer/s3/data", "https://palmerlab.s3.sdsc.edu")}/{self.project_name}/results/gwas_report.html')


    def GeneEnrichment(self, qtls: pd.DataFrame = None, genome: str = 'rn7', padding: int = 2e6, r2thresh: float = .8,
                   append_nearby_genes:bool = False, select_nearby_genes_by:str = 'r2'):
        from goatools.base import download_go_basic_obo
        from goatools.obo_parser import GODag
        from goatools.anno.genetogo_reader import Gene2GoReader
        from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
        
        gene2go = download_ncbi_associations()
        obo_fname = download_go_basic_obo()
        geneid2gos_rat= Gene2GoReader(gene2go, taxids=[int(self.taxid)])
        
        obodag = GODag('go-basic.obo')
        ratassc = geneid2gos_rat.get_ns2assc()

        if isinstance(qtls, str): qtls = pd.read_csv(qtls)
        if not (isinstance(qtls, pd.DataFrame) and len(qtls)) : qtls = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')
        #species = translate_dict(self.genome, {'rn7': 'rat', 'rn8':'rat', 'm38':'mouse', 'rn6': 'rat'})
        def get_entrez_ids(genes):
            if type(genes) == str: genes = genes.split('-') + [genes]
            o = []
            for i in genes: 
                if not pd.isna(i): o += list(np.unique(i.split('-') + [i]))
                
            return list(np.unique([int(y.replace('ENSRNOG', '')) for x  in mg.querymany(o, scopes='ensemblgene,symbol,RGD', fields='all',
                                                                                        species=self.taxid, verbose = False, silent = True, entrezonly=True)\
                                                                          if (len(y := defaultdict(lambda:'', x)['entrezgene']) > 0) * ('ENSRN' not in y)])) #
        print('getting entrezid per snp...')
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        allgenes = gtf.query('gbkey == "Gene"')[['gene_id', 'Chr', 'start', 'end']].drop_duplicates()
        if select_nearby_genes_by == 'distance':
            if not os.path.isfile(f"{self.path}results/qtls/genes_in_range.csv"): self.locuszoom2()
            genes = pd.read_csv(f'{self.path}results/qtls/genes_in_range.csv')
            snpgroups = genes.groupby('SNP_origin')[['gene_id']].agg(lambda x: list(np.unique(x)))
            snpgroups = qtls[['SNP']].rename(lambda x: x+ '_origin', axis = 1)
            snpgroups['gene_id'] = snpgroups.SNP_origin.apply(lambda x: list(allgenes.query(f'''Chr == {int(x.split(':')[0])} and ({x.split(':')[1]}-{padding}<value<{x.split(':')[1]}+{padding})''').gene_id.unique()))
            snpgroups = snpgroups.set_index('SNP_origin')
            snpgroups['entrezid'] = snpgroups['gene_id'].progress_apply(get_entrez_ids)
            qtls = qtls.merge(snpgroups.rename(lambda x: x+ '_nearby', axis = 1).reset_index(), left_on = 'SNP', right_on = 'SNP_origin')
        
        qtls['gene'] = qtls['gene'].apply(lambda x: [x] if type(x)==str else x)
        if select_nearby_genes_by == 'r2':
            def get_nearby_genes_r2(rowdata, r2thresh):
                temp = self.plink(bfile = self.genotypes_subset, chr = rowdata.Chr, ld_snp = rowdata.SNP, ld_window_r2 = 0.01, r2 = '',\
                                                    ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb =  6000, nonfounders = '').loc[:, ['SNP_B', 'R2']] 
                minbp,maxbp = pd.Series(temp.query('R2> @r2thresh')['SNP_B'].to_list() + [rowdata.SNP]).str.extract(':(\d+)').astype(int).agg([min, max]).T.values.flatten()
                out = list(allgenes.query(f'Chr == {rowdata.Chr} and  end > {minbp} and start < {maxbp}').gene_id.unique())
                return [i for i in out ] #if ('LOC' not in i)
            qtls['gene_id_nearby'] = qtls.progress_apply(get_nearby_genes_r2, r2thresh = r2thresh, axis = 1)
            qtls['entrezid_nearby'] = qtls['gene_id_nearby'].progress_apply(get_entrez_ids)
        
        merged_qtls = qtls.groupby('trait')[['gene', 'gene_id_nearby','entrezid_nearby']].agg(sum).applymap(lambda x: [] if x == 0 else x)
        print('getting entrezid per qtl...')
        merged_qtls['entrezid'] = merged_qtls['gene'].progress_apply(get_entrez_ids)
        print(f'getting entrezid per for all {self.species} {self.taxid} genes...')
        allentrez = get_entrez_ids(gtf.gene.unique())
        
        print(f'initializing GO study...')
        goeaobj = GOEnrichmentStudyNS(pop=allentrez, ns2assoc=ratassc, godag=obodag,
                                     alpha = 0.05,
                                     method = ['fdr_bh'])
        def goea(genelist):
            return pd.DataFrame( [[x.GO, x.goterm.name, x.goterm.namespace, x.p_uncorrected, x.p_bonferroni, x.ratio_in_study[0], x.ratio_in_study[1]] \
                 for x in goeaobj.run_study(genelist, silent = True, verbose = False)],\
                columns = ['GO', 'term', 'class', 'p', 'p_corr', 'n_genes', 'n_study']).sort_values(['p_corr', 'p']).query('p<0.05')
        
        print(f'running GO study for closest genes per trait...')
        merged_qtls['goea'] = merged_qtls.entrezid.apply(goea)
        print(f'running GO study for nearby genes per trait...')
        merged_qtls['goea_nearby'] = merged_qtls.entrezid_nearby.apply(goea)
        print(f'running GO study for nearby genes for all traits...')
        nearby_genes_all_project = goea(merged_qtls.entrezid_nearby.sum())
        print(f'running GO study for closest genes for all traits...')
        genes_all_project = goea(merged_qtls.entrezid.sum())
        merged_qtls['enriched_pathways'] = merged_qtls.apply(lambda r: ('closest_genes: ' + r.goea.query('p_corr < 0.05')['term']).to_list()+\
                                                              ('nearby_genes: '+r.goea_nearby.query('p_corr < 0.05')['term']).to_list(), axis = 1)
        all_enriched_paths = list(set.union(*(merged_qtls.enriched_pathways.map(set).to_list())))
        os.makedirs(f'{self.path}results/geneEnrichment', exist_ok = True)
        merged_qtls.to_pickle(f'{self.path}results/geneEnrichment/gene_enrichment_mergedqtls.pkl') 
        qtls.to_pickle(f'{self.path}results/geneEnrichment/gene_enrichmentqtls.pkl') 
        hvg = self.GeneEnrichment_figure(qtls = qtls,merged_qtls=merged_qtls, append_nearby_genes = append_nearby_genes)
        hv.save(hvg, f'{self.path}images/gene_enrichment.html')
        return hvg

    def GeneEnrichment_figure(self, qtls='', merged_qtls='', append_nearby_genes = False):
        if type(qtls) == str:               
             if not len(qtls): qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichmentqtls.pkl') 
             else:  qtls = pd.read_pickle(qtls) 
        if type(merged_qtls) == str:               
             if not len(merged_qtls): merged_qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichment_mergedqtls.pkl') 
             else:  merged_qtls = pd.read_pickle(merged_qtls)                  
        MG = nx.MultiGraph()
        for _, r in qtls.iterrows():
            if not pd.isna(r.gene):
                if not MG.has_node(r['gene'][0]):
                    MG.add_node(r['gene'][0], what = 'gene',  color = 'seagreen',
                                size = 10 + qtls['gene'].dropna().map(lambda x: x[0]).value_counts()[r.gene[0]] )
            if not MG.has_node(r['SNP']):
                MG.add_node(r['SNP'], what = 'topSNP', color = 'black', 
                            size = 5 +  qtls['SNP'].value_counts()[r.SNP] )
            if not MG.has_node(r['trait']):
                MG.add_node(r['trait'], what = 'trait', color = 'steelblue', 
                            size =15+ qtls['trait'].value_counts()[r.trait])
            MG.add_edges_from([(r.SNP, r.trait)], weight=r.p, type = 'snp2gene')
            if not pd.isna(r.gene): MG.add_edges_from([(r.gene[0], r.trait)], weight=2, type = 'gene2trait')
            if append_nearby_genes:
                if not r.gene_id_nearby is np.nan: 
                    for gidnerby in r.gene_id_nearby:
                        if not MG.has_node(gidnerby): 
                            MG.add_node(gidnerby, what = 'nearby gene', size = 2, color = 'white' )
                            MG.add_edges_from([(gidnerby, r.SNP)], weight=0.5, type = 'nearbyGene2SNP')
        
        for trait, row in merged_qtls.iterrows():
            aa = row.goea.query(f'p_corr < {0.05}')
            aa[['p', 'p_corr']] = np.nan_to_num(-np.log10(aa[['p', 'p_corr']]), posinf=30, neginf=0)
            for _, rg in aa.iterrows():
                if not MG.has_node(f'{rg.GO}\n{rg.term}'):
                    MG.add_node(f'{rg.GO}\n{rg.term}' , what = f'Gene Ontology {trait}', 
                                size = 20*rg.p_corr, term = rg.term, cls = rg['class'], p = rg.p, color = 'firebrick')
                MG.add_edges_from([(trait, f'{rg.GO}\n{rg.term}')], weight=1.5*rg.p_corr, type = 'snp2gene')
            bb = row.goea_nearby.query(f'p_corr < {1e-15}')
            bb[['p', 'p_corr']] = np.nan_to_num(-np.log10(bb[['p', 'p_corr']]), posinf=30, neginf=0)
        
            for _, rg in bb.iterrows():
                if not MG.has_node(f'{rg.GO}\n{rg.term}'):
                    MG.add_node(f'{rg.GO}\n{rg.term}', what = f'Gene Ontology {trait} with nearby genes',
                                size = max(5,rg.p_corr/5), term = rg.term, cls = rg['class'], p = rg.p, color = 'orange')
                MG.add_edges_from([(trait, f'{rg.GO}\n{rg.term}')], weight=rg.p_corr, type = 'snp2gene')
        
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        hv.opts.defaults(hv.opts.Nodes(**kwargs), hv.opts.Graph(**kwargs))
        hvg = hv.Graph.from_networkx(MG, nx.layout.spring_layout, k=1) #spring_layout, k=1
        hvg.opts(node_size='size', edge_line_width=1,tools=['hover'], shared_axes=False,
                      node_line_color='black', node_color='color', directed=True,  arrowhead_length=0.01)
        labels = hv.Labels(hvg.nodes, ['x', 'y'], 'index')
        hvg = bundle_graph(hvg)
        #hvg = hvg * labels.opts(text_font_size='2pt', text_color='white')
        return hvg

    def steiner_tree(self):
        import scipy.sparse as sps
        from itertools import chain
        from nltk import ngrams
        mg = self.project_graph_view(obj2return='networkx')
        qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichmentqtls.pkl') 
        merged_qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichment_mergedqtls.pkl') 
        allgenesproject = list(chain.from_iterable([x.split('-') for x in  mg.nodes  if mg.nodes[x]['color'] == 'seagreen']))
        allgeneswithnearby = list(set(allgenesproject + list(chain.from_iterable(qtls.gene_id_nearby))))
        if not os.path.isfile(f'{self.path}kg.csv'):
            bash(f'wget -O {self.path}kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620', return_stdout=False)
        kg = pd.read_csv(f'{self.path}kg.csv',  low_memory=False)
        kg_ppi = kg.query('y_type == "gene/protein" and x_type == "gene/protein"')
        x_w = kg_ppi.x_name.value_counts().to_frame().set_axis(['x_w'], axis =1).reset_index()
        y_w = kg_ppi.y_name.value_counts().to_frame().set_axis(['y_w'],axis = 1).reset_index()
        kg_ppi = kg_ppi.merge(x_w, on='x_name', how = 'left').merge(y_w, on='y_name', how = 'left')
        kg_ppi['weight'] = kg_ppi.x_w *kg_ppi.y_w
        kg_ppi[['x_id', 'y_id']] = kg_ppi[['x_id', 'y_id']].astype(int)
        kg_ppi_piv = kg_ppi.pivot(index = 'x_id', columns = 'y_id', values = 'weight')
        kg_ppi_piv_order = kg_ppi.drop_duplicates('x_id').set_index('x_id').loc[kg_ppi_piv.index, 'x_name'].reset_index().reset_index(names = 'sps_index')
        sparse_ppi = sps.csr_matrix(kg_ppi_piv.fillna(0).values)
        
        gene_hits_list = kg_ppi_piv_order[kg_ppi_piv_order.x_name.isin(list(map(str.upper, allgenesproject)))]
        display('Could not find these genes in ppi: ' + '|'.join( set(map(str.upper, allgenesproject)) - set(kg_ppi_piv_order.x_name.values)))
        gene_hits_list_w_nearby = kg_ppi_piv_order[kg_ppi_piv_order.x_name.isin(list(map(str.upper, allgeneswithnearby)))]
        
        indices2search = gene_hits_list.sps_index.values if len(gene_hits_list_w_nearby)> 100 else gene_hits_list_w_nearby.sps_index.values
        display(f'searching for {len(indices2search)} genes in ppi, calculating distance matrix')
        distmat, preds = sps.csgraph.shortest_path(csgraph=sparse_ppi, directed=False, indices=indices2search, return_predecessors=True)
        aa = sps.coo_matrix(preds)
        invmap = {i:j for i,j in enumerate(indices2search)}
        aa.row = np.array(list(map(lambda x: invmap[x], aa.row)))
        new_preds = sps.csr_matrix((aa.data, (aa.row, aa.col)), shape = (preds.shape[1], preds.shape[1]))
        #cstree = sps.csgraph.reconstruct_path(csgraph=sparse_ppi, predecessors=new_preds, directed=False)
        def reconstruct_path(predecessors, i, j):
            cnt = 0
            path = [j]
            while path[-1] != i:
                path.append(predecessors[i, path[-1]])
                if path[-1] in [0, -9999]: return []
            return path[::-1]
        
        sps_index2gene = {row.sps_index: row.x_name for idx, row in kg_ppi_piv_order.iterrows()}
        vec_transf = np.vectorize(lambda x :sps_index2gene[x])
        edgelist = np.array(list(chain.from_iterable([(map(vec_transf, ngrams(reconstruct_path(new_preds, i,j),2))) for i, j in itertools.combinations(indices2search, 2)])))
        
        sps_index2gene = {row.sps_index: row.x_name for idx, row in kg_ppi_piv_order.iterrows()}
        vec_transf = np.vectorize(lambda x :sps_index2gene[x])
        
        edgelist = pd.DataFrame(list(chain.from_iterable([(map(vec_transf, ngrams(reconstruct_path(new_preds, i,j),2))) for i, j in itertools.combinations(indices2search, 2)])), columns = ['in', 'out'])
        edgelistid = pd.DataFrame(list(chain.from_iterable([( ngrams(reconstruct_path(new_preds, i,j),2)) for i, j in itertools.combinations(indices2search, 2)])), columns = ['in', 'out'])
        edgelist = pd.concat([edgelist, edgelist.set_axis(['out', 'in'], axis = 1)])
        edgelistid = pd.concat([edgelistid, edgelistid.set_axis(['out', 'in'], axis = 1)])
        edgesmat = sps.csr_matrix((np.ones(shape=len(edgelistid)),(edgelistid['in'], edgelistid['out'])), shape = sparse_ppi.shape)
        edgesmat.data = np.ones(shape = edgesmat.data.shape)
        ml = sparse_ppi.multiply(edgesmat)
        ml.eliminate_zeros()
        steinertree = sps.csgraph.minimum_spanning_tree(ml).tocoo()
        pdst = pd.DataFrame([1/steinertree.data,  kg_ppi_piv_order.set_index('sps_index').loc[steinertree.row, 'x_name'].values, 
                      kg_ppi_piv_order.set_index('sps_index').loc[steinertree.col, 'x_name'].values]).set_axis(['weight','source', 'target']).T
        st = nx.from_pandas_edgelist(pdst)
        for node in st.nodes: 
            st.nodes[node]['size'] = 20 if node in gene_hits_list.x_name.values else 10
            st.nodes[node]['color'] = 'firebrick' if node in gene_hits_list.x_name.values else 'steelblue'
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        hv.opts.defaults(hv.opts.Nodes(**kwargs), hv.opts.Graph(**kwargs))
        hvg = hv.Graph.from_networkx(st, nx.layout.spectral_layout) #spring_layout, k=1
        hvg.opts(node_size='size', edge_line_width=1,tools=['hover'], shared_axes=False,
                      node_line_color='black', node_color='color', directed=False,  arrowhead_length=0.01)
        labels = hv.Labels(hvg.nodes, ['x', 'y'], 'index')
        hv.save(hvg, f'{self.path}images/steiner_tree.html')
        return pn.Card(hvg, title='Steiner tree', collapsed=True)

    def project_graph_view(self, append_nearby_genes = False, add_gene_enriched = False, add_phewas= True,
                           add_eqtl = True,add_sqtl = True,add_variants = True, obj2return = 'image'):
        
        ann = pd.read_csv(f'{self.path}results/qtls/possible_causal_snps.tsv', sep = '\t').drop(['A1','A2', 'featureid', 'rank', 
                                                                               'cDNA_position|cDNA_len','CDS_position|CDS_len',
                                                                               'Protein_position|Protein_len','distancetofeature'], errors = 'ignore', axis = 1)\
                 .query("putative_impact in ['MODERATE', 'HIGH']").sort_values('putative_impact')
        ann['p'] = -np.log10(ann.p)
        ann.rename({'p':'-Log10(p)'},axis=1,  inplace=True)
        
        phewas_exact = pd.read_csv(f'{self.path}results/phewas/pretty_table_exact_match.tsv', sep = '\t').rename({ 'p_PheDb':'-Log10(p)PheDb'}, axis =1)\
                         .sort_values('uploadeddate', ascending = False).drop(['round_version', 'uploadeddate'], axis =1).drop_duplicates(['SNP','trait_QTL','trait_PheDb','project'])
        
        phewas = pd.read_csv(f'{self.path}results/phewas/pretty_table_window_match.tsv', sep = '\t').rename({'p_PheDb': '-Log10(p)PheDb'}, axis =1)\
                         .sort_values('uploadeddate', ascending = False).drop(['round_version', 'uploadeddate'], axis =1)\
                         .drop_duplicates(['SNP_QTL', 'SNP_PheDb','trait_QTL','trait_PheDb','project'])
        
        eqtl = pd.read_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv')\
                 .rename({'-log10(P-value)':'-Log10(p)', '-log10(pval_nominal)': '-Log10(p)_eqtldb' }, axis =1)
        
        sqtl = pd.read_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv')\
                 .rename({'-log10(P-value)':'-Log10(p)', '-log10(pval_nominal)': '-Log10(p)_sqtldb' }, axis = 1)
        
        qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichmentqtls.pkl') 
        merged_qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichment_mergedqtls.pkl') 
        for tdf in [phewas_exact, ann, phewas, eqtl, sqtl, qtls, merged_qtls]:
            tdf.loc[:, tdf.columns.str.contains('SNP')] = tdf.loc[:, tdf.columns.str.contains('SNP')].applymap(lambda x: x.replace('chr', '') if type(x)== str else x)
        
        MG = nx.MultiGraph()
        for _, r in qtls.iterrows():
            if not pd.isna(r.gene):
                if not MG.has_node(r['gene'][0]):
                    MG.add_node(r['gene'][0], what = defaultdict(int, {'gene': 1}), size = 20 + qtls['gene'].dropna().map(lambda x: x[0]).value_counts()[r.gene[0]], color = 'seagreen' )
                else:  
                    MG.nodes[r['gene'][0]]['what']['gene'] += 1
                    MG.nodes[r['gene'][0]]['size'] *= 2
            if not MG.has_node(r['SNP']):
                MG.add_node(r['SNP'], what = defaultdict(int, {'topsnp': 1}), size = 14 +  qtls['SNP'].value_counts()[r.SNP], color = 'black' )
            else:  
                MG.nodes[r.SNP]['what']['topsnp'] += 1
                MG.nodes[r.SNP]['size'] *= 2    
            if not MG.has_node(r['trait']):
                MG.add_node(r['trait'], what = defaultdict(int, {'trait': 1}), size =40 + qtls['trait'].value_counts()[r.trait], color = 'steelblue' )
            MG.add_edges_from([(r.SNP, r.trait)], weight=r.p, type = 'snp2gene')
            if not pd.isna(r.gene): MG.add_edges_from([(r.gene[0], r.trait)], weight=2, type = 'gene2trait')
            if append_nearby_genes:
                for gidnerby in r.gene_id_nearby:
                    if not MG.has_node(gidnerby): 
                        MG.add_node(gidnerby, what = defaultdict(int, {'nearby_genes': 1}), size = 2, color = 'white' )
                        MG.add_edges_from([(gidnerby, r.SNP)], weight=0.5, type = 'nearbyGene2SNP')
        
        if add_gene_enriched:
            for trait, row in merged_qtls.iterrows():
                aa = row.goea.query(f'p_corr < {0.05}')
                aa[['p', 'p_corr']] = np.nan_to_num(-np.log10(aa[['p', 'p_corr']]))
                for _, rg in aa.iterrows():
                    if not MG.has_node(f'{rg.GO}\n{rg.term}'):
                        MG.add_node(f'{rg.GO}\n{rg.term}' , what =  defaultdict('', {'GO': trait}), size = 20*rg.p_corr, 
                                    term = rg.term, cls = rg['class'], p = rg.p, color = 'firebrick')
                    MG.add_edges_from([(trait, f'{rg.GO}\n{rg.term}')], weight=1.5*rg.p_corr, type = 'snp2gene')
                bb = row.goea_nearby.query(f'p_corr < {1e-15}')
                bb[['p', 'p_corr']] = np.nan_to_num(-np.log10(bb[['p', 'p_corr']]))
            
                for _, rg in bb.iterrows():
                    if not MG.has_node(f'{rg.GO}\n{rg.term}'):
                        MG.add_node(f'{rg.GO}\n{rg.term}', what = defaultdict('', {'GO_w/_nearby_genes': trait}), 
                                    size = max(5,rg.p_corr/5), term = rg.term, 
                                    cls = rg['class'], p = rg.p, color = 'orange')
                    MG.add_edges_from([(trait, f'{rg.GO}\n{rg.term}')], weight=rg.p_corr, type = 'snp2gene')
        
        #['R2', 'DP', 'SNP_QTL', 'trait_QTL', 'project', 'SNP_PheDb', 'trait_PheDb', 'trait_description_PheDb', 'Freq_PheDb', '-Log10(p)PheDb']
        if add_phewas:
            for _, row in phewas.iterrows():
                if not MG.has_node(row.SNP_PheDb): 
                    MG.add_node(row.SNP_PheDb, what = defaultdict(int, {'phewas': 1}), size = max(8,row['-Log10(p)PheDb']), R2 = row.R2, 
                                    p = row['-Log10(p)PheDb'], color = 'black', trait = row.trait_PheDb) #row.SNP_QTL.replace("chr", "")
                else:  
                    MG.nodes[row.SNP_PheDb]['what']['phewas'] +=  1 #f'\n Phewas {row.SNP_QTL.replace("chr", "")}'
                    MG.nodes[row.SNP_PheDb]['size'] += 10
                MG.add_edges_from([(row.SNP_QTL.replace("chr", ''), row.SNP_PheDb)], weight=float(row['-Log10(p)PheDb']*row.R2), type = 'snp2phewas')
                if not MG.has_node(row.trait_PheDb): 
                    MG.add_node(row.trait_PheDb, what = defaultdict(int, {'trait phewas': 1}), size=30, R2 = row.R2, 
                               color='lightblue', trait=row.trait_PheDb)
                else:  
                    MG.nodes[row.trait_PheDb]['what']['trait_phewas'] +=  1
                    MG.nodes[row.trait_PheDb]['size'] += 20
                MG.add_edges_from([(row.SNP_PheDb, row.trait_PheDb)], weight=float(row['-Log10(p)PheDb']*row.R2), type = 'phewassnp2phewasgene')#
        # ['SNP_qtl', 'index', 'SNP', 'Freq', 'b', '-Log10(p)', 'R2', 'DP', 'annotation', 'putative_impact', 'gene', 
        #  'featuretype', 'transcriptbiotype', 'HGVS.c', 'HGVS.p', 'trait']
        if add_variants:
            for _, row in ann.iterrows():
                if not MG.has_node(row.SNP): 
                    MG.add_node(row.SNP, what = defaultdict(int, {'coding_Var': 1}), size = 8, R2 = row.R2, 
                                   color = 'yellow', trait = row.trait, gene = row.gene)
                else:  
                    MG.nodes[row.SNP]['what']['coding_Var'] +=  1
                    MG.nodes[row.SNP]['size'] += 7
                MG.add_edges_from([(row.SNP_qtl, row.SNP)], weight=float(row['-Log10(p)']*row.R2), type = 'snp2ann')
            
                if ~pd.isna(row.gene) and str(row.gene) != 'nan':
                    if not MG.has_node(row.gene): 
                        MG.add_node(row.gene, what = defaultdict(int, {'gene_w/_coding_Var': 1}), size=max(8,row['-Log10(p)']), R2 = row.R2, 
                                    p=row['-Log10(p)'], color='seagreen', trait=row.trait, gene=row.gene)
                    else:  
                        MG.nodes[row.gene]['what']['gene_w/_coding_Var'] +=  1
                        MG.nodes[row.gene]['size'] += 10
                    MG.add_edges_from([(row.SNP, row.gene)], weight=float(row['-Log10(p)']*row.R2), type = 'varsnp2vargene')#
        
        if add_eqtl:
            for _, row in eqtl.iterrows():
                if not MG.has_node(row.SNP_eqtldb): 
                    MG.add_node(row.SNP_eqtldb, what = defaultdict(int, {'eQTL': 1}), size = max(8,row['-Log10(p)']), R2 = row.R2, 
                                    tissue =  row.tissue , p = row['-Log10(p)'], color = 'black', trait = row.trait,
                                gene = row.Ensembl_gene, sqtlp = row['-Log10(p)_eqtldb'])
                else:  
                    MG.nodes[row.SNP_eqtldb]['what']['eQTL'] +=  1
                    MG.nodes[row.SNP_eqtldb]['size'] += 7
                MG.add_edges_from([(row.SNP.replace("chr", ''), row.SNP_eqtldb)], weight=float(row['-Log10(p)_eqtldb']*row.R2), type = 'snp2eqtl')
            
                if ~pd.isna(row.Ensembl_gene) and str(row.Ensembl_gene) != 'nan':
                    if not MG.has_node(row.Ensembl_gene): 
                        MG.add_node(row.Ensembl_gene, what = defaultdict(int, {'gene_eQTL': 1}), size=max(8,row['-Log10(p)']), R2 = row.R2, 
                                    tissue=row.tissue, p=row['-Log10(p)'], color='seagreen', trait=row.trait, gene=row.Ensembl_gene, eqtlp=row['-Log10(p)_eqtldb'])
                    else:  
                        MG.nodes[row.Ensembl_gene]['what']['gene_eQTL'] +=  1
                        MG.nodes[row.Ensembl_gene]['size'] += 10
                    MG.add_edges_from([(row.SNP_eqtldb, row.Ensembl_gene)], weight=float(row['-Log10(p)_eqtldb']*row.R2), type = 'eqtlsnp2eqtlgene')#
        
        if add_sqtl:
            for _, row in sqtl.iterrows():
                if not MG.has_node(row.SNP_sqtldb): 
                    MG.add_node(row.SNP_sqtldb, what =  defaultdict(int, {'sQTL': 1}), size = max(8,row['-Log10(p)']), R2 = row.R2, 
                                    tissue =  row.tissue , p = row['-Log10(p)'], color = 'black', trait = row.trait, 
                                gene = row.Ensembl_gene, sqtlp = row['-Log10(p)_sqtldb'])
                else:  
                    MG.nodes[row.SNP_sqtldb]['what']['sQTL'] +=  1
                    MG.nodes[row.SNP_sqtldb]['size'] += 7
                MG.add_edges_from([(row.SNP.replace("chr", ''), row.SNP_sqtldb)], weight=row['-Log10(p)_sqtldb']*row.R2, type = 'snp2sqtl')
                if ~pd.isna(row.Ensembl_gene) and str(row.Ensembl_gene) != 'nan':
                    if not MG.has_node(row.Ensembl_gene): 
                        MG.add_node(row.Ensembl_gene, what = defaultdict(int, {'gene_sQTL': 1}), size = max(8,row['-Log10(p)']), R2 = row.R2, 
                                        tissue =  row.tissue , p = row['-Log10(p)'], color = 'seagreen', trait = row.trait, 
                                    gene = row.Ensembl_gene, sqtlp = row['-Log10(p)_sqtldb'])
                    else:  
                        MG.nodes[row.Ensembl_gene]['what']['gene_sQTL'] +=  1
                        MG.nodes[row.Ensembl_gene]['size'] += 10
                    MG.add_edges_from([(row.SNP_sqtldb, row.Ensembl_gene)], weight=row['-Log10(p)_sqtldb']*row.R2, type = 'sqtlsnp2sqtlgene')
        
        weights = 50/max(nx.get_node_attributes(MG, 'size').values())
        bad_nodes = []
        for node in MG.nodes: 
            try: 
                MG.nodes[node]['size'] *= weights
                MG.nodes[node]['what'] =  ' | '.join(map(lambda x : f'{x[0]}:{x[1]}', MG.nodes[node]['what'].items()))
            except:
                print(f'Node {node} does not have a size')
                bad_nodes += [node]
        for node in bad_nodes: MG.remove_node(node)
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        hv.opts.defaults(hv.opts.Nodes(**kwargs), hv.opts.Graph(**kwargs))
        hvg = hv.Graph.from_networkx(MG, nx.layout.spring_layout, k=1) #
        hvg.opts(node_size='size', edge_line_width=1,tools=['hover'], shared_axes=False,
                      node_line_color='black', node_color='color', directed=True,  arrowhead_length=0.01)
        labels = hv.Labels(hvg.nodes, ['x', 'y'], 'index')
        hvg = bundle_graph(hvg)
        #hvg = hvg * labels.opts(text_font_size='5pt', text_color='white')
        hv.save(hvg, f'{self.path}images/project_graph_view.html')
        if obj2return == 'image': return hvg
        if obj2return in ['graph', 'networkx']: return MG
        return hvg

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
        
def display_graph(G):
    plt.figure(1, figsize=(8, 8))
    # layout graphs with positions using graphviz neato
    pos = graphviz_layout(G, prog="neato")
    # color nodes the same in each connected subgraph
    C = (G.subgraph(c) for c in nx.connected_components(G.to_undirected()))
    for g in C:
        c = [random.random()] * nx.number_of_nodes(g)  # random color...
        nx.draw(g, pos, node_size=20, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)
    plt.show()

def assembly(graph):
    R = nx.dag_longest_path(nx.DiGraph(graph), weight='weight', default_weight=1)
    return R[0] + ''.join([x[-1] for x in R[1:]])

def assembly_complete(graph):
    S = [nx.DiGraph(graph.subgraph(c).copy()) for c in nx.connected_components(graph.to_undirected())]
    return np.array([assembly(subg) for subg in S])

def tuple_to_graph_edge(graph, kmer1, kmer2,  ide):
    if graph.has_edge(kmer1, kmer2):  
        graph[kmer1][kmer2]['weight'] += 1 
        graph[kmer1][kmer2]['label'][ide] += 1
    else: graph.add_edge(kmer1, kmer2, weight=1, label = defaultdict(int, {ide:1}))
    return

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
                        founderfile = '/tscc/projects/ps-palmer/gwas/databases/founder_genotypes/founders7.2',
                        latest_round = '/tscc/projects/ps-palmer/gwas/databases/rounds/r10.1.1',
                        pickle_output = False, impute = True,
                        save_path = ''):
    print(f'starting umap chunks for {chrom}, nsnps = {nsampled_snps}, maxrats = {nsampled_rats}')
    if type(founderfile) == str: bimf, famf, genf = pandas_plink.read_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(latest_round) == str: bim, fam, gen = pandas_plink.read_plink(latest_round)
    else: bim, fam, gen = latest_round
    

    bim1 = bim[bim.chrom.isin([chrom, str(chrom)]) & bim.snp.isin(bimf.snp)]#.query('chrom == "12"')
    start, stop = bim1['i'].agg(['min', 'max']).values
    if random_sample_snps: 
        sampled_snps = sorted(np.random.choice(range(start, stop), size = min(nsampled_snps, gen.T.shape[1] ) , replace = False))
    else:
        sampled_snps = bim1[::bim1.shape[0]//nsampled_snps+1].i.to_list()
        
    bim1 = bim1[bim1.i.isin(sampled_snps)]
    bim1i = bim1.set_index('pos')
    bim1isnp = bim1.set_index('snp')
    
    rnd_next_mb = lambda x: int(np.ceil(1555555/1e6)*1e6)
    if (tempvar := bim1.pos.diff().max().round()) > int(win*overlap):
        win = rnd_next_mb(tempvar)
        print(f'increasing window to {round(win/1e6)}Mb')
    offset = int(win/2)
    
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
        for i in range(offset,bim1.pos.max()+ offset, int(win*overlap )) \
        if (k := bim1i.loc[i - offset:i + offset+1]).shape[0]> 0]).reset_index(drop =True)
    out['nsnps'] = out.i_list.map(len)
    out.loc[[0], 'relationship'] = [{}]
    out.loc[1:, 'relationship'] = [find_overlap(out.iloc[i], out.iloc[i+1]) for i in range(out.shape[0]-1)]
    out['relationship_i'] = out.apply(lambda r: r.i_list[list(r.relationship.values())], axis = 1)
    out['relationship_snp'] = out.apply(lambda r: r.snp_list[list(r.relationship.values())], axis = 1)
    label_dict = defaultdict(lambda: 'AAunk', {row.i: row.iid for name, row in famf.iterrows()})
    i2label = lambda y: label_dict[y]
    out['label'] = out.nsnps.apply(lambda x: np.hstack([-np.ones(len(sampled_rats)),(famf.i).values ]))
    out['label_str'] = out.label.apply(lambda x:list(map(i2label, x)))
    print('getting the genotypes for umap with...')
    out['genotypes'] = out.snp_list.progress_apply(lambda x: np.vstack([gen.T[sampled_rats][:, bim1isnp.loc[x].i ], 
                                                                  genf.T[:, bimf1isnp.loc[x].i]]).astype(np.float16).compute())#

    if str(chrom).lower() in [str(nautosomes+1), 'x']:
        male_indices = fam.loc[sampled_rats].reset_index(drop= True).query('gender in [2, "2"]').index.to_list()
        out['genotypes'] = out.genotypes.progress_apply(dosage_compensation,  male_indices = male_indices)

    if impute:
        print('doing KNN imputation and scaling  with...')
        metr = 'euclidean'
        out['genotypes'] = out['genotypes'].progress_apply(lambda x: np.nan_to_num(make_pipeline(\
                                                                                   KNNImputer(weights = 'distance'),
                                                                                   StandardScaler() )\
                                                           .fit_transform(x)))
    else:  metr = nan_euclidean_distances
    
    aligned_mapper = umap.AlignedUMAP(metric=metr, target_metric=metr).fit(out.genotypes.to_list(), \
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
    os.makedirs(f'{save_path}images/genotypes/3dchrom', exist_ok = True)
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
                      color="label",size = 'label_size', opacity = .9) #opacity = stack.alpha,,  symbol = 'founder_label'
    #, hover_data=['rfid'] 
    fig2.update_layout(
        autosize=False, width=1800, height=1000,
        #updatemenus=[dict( type="buttons", buttons=[dict(label="Play", method="animate",
        #                      args=[None, {"frame": {"duration": 5, "redraw": False},}])])]
    )
    fig2.update_traces(marker=dict(line=dict(width=.4, color='black') ))
    os.makedirs(f'{save_path}images/genotypes/animations', exist_ok = True)
    fig2.write_html(f'{save_path}images/genotypes/animations/genomeanimation_{chrom}.html')
    
    founder_count = pd.concat(out['predicted_label_str'].apply(lambda x: pd.DataFrame(Counter(x), index = ['count'])).values)\
          .fillna(0).astype(int).set_index(out.pos_mean.astype(int))

    fig = dashbio.Clustergram(data=founder_count,
        column_labels=list(founder_count.columns.values),
        row_labels=list(founder_count.index),cluster = 'col', display_ratio=[0.9, 0.1], color_map='Spectral' ,center_values=False
    )
    founder_count.to_csv(f'{save_path}images/genotypes/founders_cnt/csv_founders_cnt_{chrom}.csv')
    fig.update_layout(  width=800, height=800,autosize=False,showlegend=False )
    os.makedirs(f'{save_path}images/genotypes/founders_cnt', exist_ok = True)
    fig.write_html(f'{save_path}images/genotypes/founders_cnt/founders_cnt_{chrom}.html')
    if pickle_output:
        with open(f'{save_path}results/umap_per_chunk/{chrom}.pkl', 'wb') as fil:
            pickle.dump(out.drop('genotypes', axis = 1), fil )
    #fig2.show()
    return out


translatechr = defaultdict(lambda: 'UNK',{k.split('\t')[1]: k.split('\t')[0] for k in'''1	NC_051336.1
2	NC_051337.1
3	NC_051338.1
4	NC_051339.1
5	NC_051340.1
6	NC_051341.1
7	NC_051342.1
8	NC_051343.1
9	NC_051344.1
10	NC_051345.1
11	NC_051346.1
12	NC_051347.1
13	NC_051348.1
14	NC_051349.1
15	NC_051350.1
16	NC_051351.1
17	NC_051352.1
18	NC_051353.1
19	NC_051354.1
20	NC_051355.1
X	NC_051356.1
Y	NC_051357.1
MT	NC_001665.2'''.split('\n')})

translatechrmice = defaultdict(lambda: 'UNK',{k.split(' ')[1]: k.split(' ')[0] for k in '''1 NC_000067.6
2 NC_000068.7
3 NC_000069.6
4 NC_000070.6
5 NC_000071.6
6 NC_000072.6
7 NC_000073.6
8 NC_000074.6
9 NC_000075.6
10 NC_000076.6
11 NC_000077.6
12 NC_000078.6
13 NC_000079.6
14 NC_000080.6
15 NC_000081.6
16 NC_000082.6
17 NC_000083.6
18 NC_000084.6
19 NC_000085.6
X NC_000086.7
Y NC_000087.7
MT NC_005089.1'''.split('\n')})

translatechr8 = defaultdict(lambda: 'UNK',{k.split(' ')[1]: k.split(' ')[0] for k in'''1 NC_086019.1
2 NC_086020.1
3 NC_086021.1
4 NC_086022.1
5 NC_086023.1
6 NC_086024.1
7 NC_086025.1
8 NC_086026.1
9 NC_086027.1
10 NC_086028.1
11 NC_086029.1
12 NC_086030.1
13 NC_086031.1
14 NC_086032.1
15 NC_086033.1
16 NC_086034.1
17 NC_086035.1
18 NC_086036.1
19 NC_086037.1
20 NC_086038.1
X NC_086039.1
Y NC_086040.1
MT NC_001665.2'''.split('\n')})






