#logging.basicConfig(filename=f'gwasRun.log', filemode='w', level=logging.DEBUG)

from importlib import metadata as _meta, resources as impl_resources, import_module
from pathlib import Path
import subprocess, inspect

try:
    __version__ = _meta.version(__name__)
    #icon_path = impl_resources.files('gwas').joinpath("rat.ico")
    icon_path = Path(__file__).resolve().parent / "rat.ico"
except _meta.PackageNotFoundError:
    root = Path(__file__).resolve().parent
    icon_path = root / "rat.ico"
    try:
        print('running git describe to get version')
        __version__ = subprocess.check_output(  ["git", "-C", root, "describe", "--tags"], text=True).strip()
    except Exception:  __version__ = "0.0.0+unknown"

print(__version__)
print('print importing packages...')
from IPython.display import display
from IPython.utils import io
from bokeh.resources import INLINE, CDN
from bokeh.io import export_png
from bokeh.models import NumeralTickFormatter
from collections import Counter, defaultdict, namedtuple
from dask.diagnostics import ProgressBar 
from dask import delayed
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
from gwas.interactiveqc import interactive_QC
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from matplotlib.colors import PowerNorm
from matplotlib.colors import rgb2hex as mplrgb2hex
from os.path import dirname, basename
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.utilities import regressor_coefficients 
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list, linkage
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, zscore
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import KNNImputer,SimpleImputer, IterativeImputer, MissingIndicator
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, RidgeCV, RANSACRegressor
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.metrics import get_scorer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.utils.extmath import randomized_svd
from gwas.statsReport import quantileTrasformEdited as quantiletrasform
from gwas.statsReport import StepWiseRegression
from time import sleep, time
from tqdm import tqdm
from umap import UMAP
import base64
import dash_bio as dashbio
import dask
import dask.bag as db
import dask.array as da
import dask.dataframe as dd
import datashader as ds
import gc
import goatools
import gzip
import hashlib
import holoviews as hv
import hvplot.pandas
import hvplot.dask  
import itertools
import json
import logging
import lightgbm
import matplotlib.pyplot as plt
import mygene
import networkx as nx
import nltk
import numba
import numpy as np
from holoviews import opts
import os
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as plotio
import prophet
import psycopg2
import pysam
import re
import requests
import seaborn as sns
from . import statsReport, npplink, locuszoom_py3
import sys
import umap
#import utils
import warnings
import scipy.spatial as spa
import scipy.sparse as sps
import xarray as xr
print('done importing packages...')

# gene2go = download_ncbi_associations()
# geneid2gos_rat= Gene2GoReader(gene2go, taxids=[10116])
#bioconda::ensembl-vep=112.0-0
__VERSION__ = 'v0.4.1'

mg = mygene.MyGeneInfo()
ProgressBar(minimum =120).register()
tqdm.pandas()
sys.setrecursionlimit(10000)
#warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pn.extension()
pn.extension('mathjax') # "bokeh", "matplotlib", "plotly"
try:pd.set_option('future.no_silent_downcasting', True)
except: pass
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
#pd.options.plotting.backend = 'holoviews'
na_values_4_pandas = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null', 'UNK']
bp2str = lambda num : re.findall(r'^\d*\,?\d{2}',s := f'{abs(num):,}')[0].replace(',','.') + ['bp', 'Kb', 'Mb', 'Gb'][s.count(',')] if num>9 else f'{num}bp'
FOUNDER_COLORS = defaultdict(lambda: 'white', {'BN': '#000000', 'ACI':'#D32F2F', 'MR': '#8E24AA', 'M520':'#FBC02D',
                      'F344': '#388E3C', 'BUF': '#1976D2', 'WKY': '#F57C00', 'WN': '#02D3D3'})
#hv.extension("bokeh", "matplotlib", "plotly",  inline=True) #logo=False,
def combine_duplicated_indexes(df):
    dups = df.index.duplicated()
    return df[~dups].combine_first(df[dups])

def replace_w_mean(s, lim_l, lim_h):
    return s.where(((lim_l<s) & (s<lim_h)) | s.isna(), 
                   s[(s>lim_l) & (s<lim_h)].mean() )
    ## example final.update(final.groupby('cohort', group_keys = False).apply(lambda df: df.filter(regex= 'age').apply(replace_w_mean, lim_l = 25, lim_h=60)))

def clipboard_button(text,title='copy LLM query',**kwargs):
    zz = {
        'name': title,
        "button_type": "primary",
        "description": f"{title} clipboard",
        "icon": "clipboard",
        "sizing_mode": "fixed",
        "width": 30,
        "margin": 0}
    for key, value in zz.items():
        if not key in kwargs: kwargs[key] = value
    button = pn.widgets.Button(**kwargs)
    copy_code = f'''navigator.clipboard.writeText({json.dumps(text)});'''
    #copy_code = f'''navigator.clipboard.writeText("{text}");'''
    button.js_on_click(code=copy_code)
    return button

def read_csv_zip(path, zippath, file_func = basename, zipfile_func = basename, query_string=None, print_files=False,**kws):
    import zipfile
    pattern = re.compile(path)
    res = []
    if isinstance(zippath, str):zippath  = [zippath]
    for zppathi in zippath:
        resi = []
        if not os.path.isfile(zppathi):
            raise FileNotFoundError( f'[Errno 2] No such file or directory: {zippath}')
        with zipfile.ZipFile(zppathi) as zip_ref:
            gwas_files = [x for x in zip_ref.namelist() if pattern.match(x)]
            if print_files: print(gwas_files)
            for gf in tqdm(gwas_files) if len(gwas_files)>10 else gwas_files:
                with zip_ref.open(gf) as file:
                    tdf = pd.read_csv(file, **kws).assign(file = file_func(gf),  zipfile = zipfile_func(zppathi) )
                    if query_string is not None: tdf = tdf.query(query_string)
                    resi += [tdf]
        if resi: res += [pd.concat(resi)]
    if not res: 
        raise FileNotFoundError( f'[Errno 2] No such files or in pattern: {path}')
    return pd.concat(res)

from bokeh.models import CustomJS
def js_hook_factory( scaler=3e6,  base_px=15,  min_px=10,  max_px=50, font="Arial Narrow", **kws):
    def _js_hook(plot, element):
        p   = plot.state
        # locate the Text glyph renderer
        txt = next(
            r for r in p.renderers
            if getattr(r, "glyph", None)
               and r.glyph.__class__.__name__ == "Text")
        start, end = p.x_range.start, p.x_range.end
        if start is not None and end is not None:
            vis   = end - start
            raw   = scaler / vis
            px0   = base_px * raw
            px0   = max(min_px, min(max_px, px0))
            txt.glyph.text_font_size = f"{int(px0)}px"
            # txt.glyph.text_font      = font
            #txt.glyph.text_font_size = {"value": f"{int(px)}px"}
            txt.glyph.text_font      = {"value": font}
        fonttext = '{value: "'+ font+'"}'
        code = f"""
            const vis   = cb_obj.end - cb_obj.start;
            const raw   = {scaler} / vis;
            const scale = Math.log(raw + 1);
            let px = {base_px} * raw;
            px = Math.max({min_px}, Math.min({max_px}, px));
            txt.glyph.text_font_size = px.toFixed(0) + "px";
            txt.glyph.text_font      = {fonttext};
        """
        cb = CustomJS(args=dict(txt=txt), code=code)
        p.x_range.js_on_change("start", cb)
        p.x_range.js_on_change("end",   cb)
    return _js_hook
    
def GRM(X, scale = True, return_weights= False, nan_policy = 'ignore', correlation_matrix= False):
    ##### z calculation
    x = np.array(X)
    z = x - np.nanmean(x, axis = 0)
    if scale: z /=  np.nanstd(x, axis = 0)
    np.nan_to_num(z, copy = False,  nan=0.0, posinf=None, neginf=None )
    zzt = np.dot(z,z.T)
    if nan_policy == 'mean': zzt_w = x.shape[1]-1
    #### NA adjustment
    elif nan_policy in ['ignore', 'per_iid']: 
        zna = (~np.isnan(x)).astype(np.float64)
        zzt_w = np.dot(zna,zna.T)
        zzt_w = np.clip(zzt_w-1, a_min = 1, a_max = np.inf)
        if nan_policy == 'per_iid':
            zzt_w = zzt_w.max(axis =1)[:, None]
    grm = zzt/zzt_w
    if correlation_matrix:
        sig = np.sqrt(np.diag(grm))
        grm /= np.outer(sig, sig)
        np.fill_diagonal(grm, 1)
    if return_weights: 
        return {'zzt': zzt, 'weights': zzt_w, 'grm': grm }
    else:
        return grm

from sklearn.metrics import f1_score
class nan_ignore:
    def __init__(self, scorefunc, **kws):
        self.scorefunc = scorefunc
        self.kws = kws
    def __call__(self,x,y):
        mask = ~(np.isnan(x)+ np.isnan(y))
        return self.scorefunc(x[mask], y[mask], **self.kws)

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
class NamedColumnTransformer(ColumnTransformer):
    def __init__(self, transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True):
        super().__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out
        )
        self.conv_table = defaultdict(lambda: 'float64', {})
        self.col_names = None  # Initialize col_names attribute
        

    def _is_fitted(self,estimator, attributes=None, all_or_any=all):
        try:
            check_is_fitted(estimator, attributes=attributes, all_or_any=all_or_any)
            return True
        except NotFittedError:return False

    def _get_transformer_output_shape(self, transformer, X_subset):
        """Determine the shape of the transformer's output."""
        try:
            output = transformer.fit_transform(X_subset)
            transformer_name = type(transformer).__name__.lower()
            if hasattr(transformer, 'get_feature_names_out'):
                return [f'{i}' for i in transformer.get_feature_names_out()]
            return [f'{i}' for i in range(output.shape[1])] 
        except Exception as e:
            raise RuntimeError(f"Error determining output shape for transformer {transformer}: {e}")

    def _generate_column_names(self, X, result):
        """Generate column names dynamically based on the transformers."""
        col_names = []
        for name, transformer, columns in self.transformers:
            if isinstance(transformer, Pipeline):
                transformer = transformer[-1]
            if transformer == 'drop':
                continue
            elif transformer == 'passthrough':
                col_names.extend(list(columns))
                n_components = transformer.n_components
                transformer_name = type(transformer).__name__.lower()
                new_cols = [f'{name}_{transformer_name}_{i}' for i in range(1, n_components+1)] 
                col_names.extend(new_cols)
            elif hasattr(transformer, 'get_feature_names_out') and self._is_fitted(transformer):
                new_cols = [f'{name}_{i}' for i in transformer.get_feature_names_out()] 
                col_names.extend(new_cols)
            else:
                transformer_name = type(transformer).__name__
                if isinstance(columns, list):
                    X_subset = X[columns].iloc[:, :]
                elif isinstance(columns, slice):
                    X_subset = X.iloc[:, columns].iloc[:, :]
                else:
                    X_subset = X[[columns]].iloc[:, :]
                cc = self._get_transformer_output_shape(transformer, X_subset)
                c2add = [f"{name}_{i}" for i in cc]
                col_names.extend(c2add)
        return col_names

    def fit(self, X):
        result = super().fit(X)
        if self.col_names is None:
            self.col_names = self._generate_column_names(X, result)
        self.conv_table = defaultdict(lambda: 'float64', X.dtypes.to_dict())
        return self

    def transform(self, X):
        result = super().transform(X)
        if hasattr(result, "toarray"):result = result.toarray()
        if self.col_names is None:
            self.col_names = self._generate_column_names(X, result)
        res = pd.DataFrame(result, columns=self.col_names, index=X.index if hasattr(X, 'index') else None)
        res = res.astype({x:self.conv_table[x] for x in res.columns})
        return res

    def fit_transform(self, X, y=None):
        result = super().fit_transform(X, y)
        if hasattr(result, "toarray"):result = result.toarray()
        if self.col_names is None:
            self.col_names = self._generate_column_names(X, result)
        self.conv_table = defaultdict(lambda: 'float64', X.dtypes.to_dict())
        res = pd.DataFrame(result, 
                           columns=self.col_names, 
                           index=X.index if hasattr(X, 'index') else None)
        res = res.astype({x:self.conv_table[x] for x in res.columns})
        return res


# import torch
# from transformers import AutoTokenizer, AutoModel
# from transformers.models.bert.configuration_bert import BertConfig
# config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M",  cache_dir="/tscc/projects/ps-palmer/gwas/databases/ai_cache")
# model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config,  cache_dir="/tscc/projects/ps-palmer/gwas/databases/ai_cache")

def nan_l1(a, b):
    valid_mask = ~np.isnan(a) & ~np.isnan(b)
    if not np.any(valid_mask): return np.nan  # or 0
    return np.sum(np.abs(a[valid_mask] - b[valid_mask])) / np.sum(valid_mask)

def f1_fast(a,b):
    mask = ~(np.isnan(a)+ np.isnan(b))
    a = np.array(a)[mask].astype(bool)
    b = np.array(b)[mask].astype(bool)
    tp = (a&b).sum()
    fpfn = (b^a).sum()
    if not tp and not fpfn: return 1
    return 2*tp/(2*tp + fpfn)

@numba.njit
def pairwise_nan_l1(gs):
    n_samples, n_features = gs.shape
    dist = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            num_valid = 0
            acc = 0.0
            for k in range(n_features):
                a = gs[i, k]
                b = gs[j, k]
                if not (np.isnan(a) or np.isnan(b)):
                    acc += abs(a - b)
                    num_valid += 1
            dist_ij = acc / num_valid if num_valid > 0 else np.nan
            dist[i, j] = dist_ij
            dist[j, i] = dist_ij
    return dist

def f1(X, y=None):
    X = np.ma.masked_invalid(X)
    if y is None:y = X.copy()
    else: y = np.ma.masked_invalid(y)
    tp = np.ma.dot(X.T, y)
    fp  = np.ma.dot(X.T, 1-y)
    fn  = np.ma.dot(1-X.T, y)
    return (2*tp/(2*tp + fp + fn)).data

def r2toD(r2, snp1maf,snp2maf):
    return np.sqrt(r2*snp1maf*(1-snp1maf)*snp2maf*(1-snp2maf))

def R2(X, Y= None, return_named = True, return_square = True, statistic = 'r2', dtype = np.float64):
    if statistic not in ['r2', 'r', 'cov', 'D', 'D2', 'chord']: raise ValueError("statistic has to be in ['r2', 'r', 'cov', 'D', 'D2', 'chord']")
    x = np.array(X).astype(dtype)
    xna = (~np.isnan(x)).astype(dtype) ##get all nas
    xnaax0 = xna.sum(axis = 0)
    if statistic in ['D', 'D2']: 
        p_x = (np.nansum(x, axis=0) / xnaax0)
        x -= p_x
        p_x = np.clip(p_x*0.5, 1e-12, 1 - 1e-12)
    else: x -= (np.nansum(x, axis = 0)/xnaax0) #subtract mean
    np.nan_to_num(x, copy = False,  nan=0.0, posinf=None, neginf=None ) ### will not affect sums 
    xstd = np.sqrt(np.sum(x**2, axis = 0)/xnaax0) #estimate std
    xstd[xstd == 0] = np.nan
    if Y is None:  
        y, yna, ystd = x, xna, xstd 
        if statistic in ['D', 'D2']: p_y = p_x
    else:
        y = np.array(Y).astype(dtype)
        yna = (~np.isnan(y)).astype(dtype) ##get all nas
        ynaax0 = yna.sum(axis = 0)
        if statistic in ['D', 'D2']: 
            p_y = (np.nansum(x, axis=0) / xnaax0)
            y -= p_y
            p_y = np.clip(p_y*0.5, 1e-12, 1 - 1e-12)
        else:  y -= (np.nansum(y, axis = 0)/ynaax0) #subtract mean
        np.nan_to_num(y, copy = False,  nan=0.0, posinf=None, neginf=None ) ### will not affect sums 
        ystd = np.sqrt(np.sum(y**2, axis = 0)/ynaax0) #estimate std
        ystd[ystd == 0] = np.nan
    xty_w = np.dot(xna.T,yna)
    xty_w[xty_w == 0] = np.nan
    if statistic != 'cov': cov = np.dot(x.T,y) / xty_w
    else: res = np.dot(x.T,y) / xty_w
    if statistic in ['r2', 'D2']: 
        res = np.clip(np.power(cov/np.outer(xstd, ystd), 2), a_min = 0, a_max = 1)
        if statistic == 'D2': res *= np.outer(p_x*(1-p_x), p_y*(1-p_y))
    elif statistic in ['r', 'D', 'chord']: 
        res = np.clip(cov/np.outer(xstd, ystd), a_min = -1, a_max = 1)
        if statistic == 'chord': res = np.sqrt(2*(1-res))
        if statistic == 'D': res *= np.sqrt(np.outer(p_x*(1-p_x), p_y*(1-p_y)))

    rindex = X.columns if isinstance(X, pd.DataFrame) else list(range(x.shape[1]))
    if (Y is None) and isinstance(X, pd.DataFrame): rcolumns = X.columns
    elif isinstance(Y, pd.DataFrame): rcolumns = Y.columns
    else: rcolumns = list(range(y.shape[1]))
    if return_named: 
        res = pd.DataFrame(res, index = rindex, columns = rcolumns)  
        if not return_square:
            res = res.reset_index(names = 'bp1').melt(id_vars = 'bp1', var_name='bp2')
            chrom = res['bp1'].iloc[0].split(':')[0]
            pos = len(chrom)+ 1
            res['c'] = chrom
            res['distance'] = (res.bp1.str.slice(start=pos).astype(int)  - res.bp2.str.slice(start=pos).astype(int)).abs()
    return res


def read_grm_w(path):
    import pandas_plink
    a, b = pandas_plink.read_grm(path)
    zz = np.zeros_like(a) 
    zz[np.triu_indices(a.shape[0])] = b
    zz += np.triu(zz, 1).T
    return {'grm':a.to_pandas(),'w': zz, 'path': path}

def getX_from_G(G, thresh = 1):
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1] 
    cumsum = np.cumsum(eigvals/eigvals.sum())
    try:maxidx = np.argwhere(cumsum>thresh).min()
    except: maxidx = len(cumsum)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals, eigvecs = eigvals[:maxidx], eigvecs[:, :maxidx]
    return  eigvecs @ np.diag(np.sqrt(eigvals))

def getX_from_G_fast(G, thresh=1, k=50):
    if type(G) == pd.core.frame.DataFrame: index = G.index
    from sklearn.utils.extmath import randomized_svd
    k = min(k+1, G.shape[0])
    U, S, Vt = randomized_svd(np.array(G), n_components=k)
    eigvals_sum = S.sum()
    cumsum = np.cumsum(S / eigvals_sum)
    maxidx = np.searchsorted(cumsum, thresh, side='right')
    S, U = S[:maxidx], U[:, :maxidx]
    X = U @ np.diag(np.sqrt(S))
    if type(G) == pd.core.frame.DataFrame: X = pd.DataFrame(X, index = index, columns = [f'ped_{i:003}' for i in range(X.shape[1])])
    return X

def remove_relatedness_transformation(G, h2, yvar: float = 1 ):
    fG = yvar*(h2*G + (1-h2)*np.eye(G.shape[0]))
    eva, eve = np.linalg.eigh(fG)
    lam = np.diag(1/np.sqrt(eva))
    return eve@lam@eve.T

def daskR2(X, Y=None, return_named= True):
    x = da.array(np.array(X), dtype = np.float64) if not isinstance(X, da.Array) else X
    xna = (~da.isnan(x)).astype(np.float16)
    x -= da.nanmean(x, axis = 0)
    xstd = da.nanstd(x, axis = 0)#estimate std
    x[da.isnan(x)] = 0
    if Y is None:  y, yna, ystd = x, xna, xstd 
    else:
        y = da.array(np.array(Y), dtype = np.float64) if not isinstance(Y, da.Array) else Y
        yna = (~da.isnan(y)).astype(np.float16)
        ystd = da.nanstd(y, axis = 0)
        y -= da.nanmean(y, axis = 0)
        y[da.isnan(y)] = 0
        #estimate std
    cov = da.dot(x.T,y)/da.dot(xna.T,yna)
    res = da.power(cov/da.outer(xstd, ystd), 2)
    res[res<0] = 0
    res[res>1] = 1
    res = res.compute()
    rindex = X.columns if isinstance(X, pd.DataFrame) else list(range(x.shape[1]))
    if (Y is None) and isinstance(X, pd.DataFrame): rcolumns = X.columns
    elif isinstance(Y, pd.DataFrame): rcolumns = Y.columns
    else: rcolumns = list(range(y.shape[1]))
    if return_named: res = pd.DataFrame(res, index = rindex, columns = rcolumns)   
    return res

def off_diagonalR2(snps, snp_dist = 1000, min_snp_dist = 0, return_square = True):
    step = 200
    step_list = [list(snps.columns[step*x:step*x+step]) for x in range(int(snps.shape[1]/step)+1)]
    off_diagonal_range = (int(min_snp_dist/step)+1,int(snp_dist/step)+1)
    all_ngrams = db.from_sequence(nltk.everygrams(step_list, min_len=off_diagonal_range[0], max_len=off_diagonal_range[1]))
    def _dr2(tup):
        if not len(tup): res = R2(snps.loc[:,tup])
        else: res =  R2(snps.loc[:,tup[0]],snps.loc[:,tup[-1]] )
        res = res.reset_index(names = 'bp1').melt(id_vars = 'bp1', var_name='bp2')
        return pd.concat([res, res.rename({'bp1': 'bp2','bp2': 'bp1'}, axis = 1)])
    rr2 = pd.concat(db.map(_dr2, all_ngrams).compute(scheduler='threads'))\
           .astype({'bp1': str, 'bp2': str, 'value' : float})\
           .drop_duplicates(['bp1', 'bp2'])
    if return_square:
        rr2 = rr2.pivot(index = 'bp1', columns = 'bp2', values = 'value')\
                 .loc[list(snps.columns),list(snps.columns)]
    else:
        pos = len(rr2['bp1'].iloc[0].split(':')[0])+ 1
        rr2['distance'] = (rr2.bp1.str.slice(start=pos).astype(int)  - rr2.bp2.str.slice(start=pos).astype(int)).abs()
    return rr2

def SemiDefPosGRM(grm, eps = 1e-6):
    grm = np.array(grm)
    try: 
        np.linalg.cholesky(grm)
        return grm
    except:
        eigvals, eigvecs = np.linalg.eigh(grm)
        # Set any negative eigenvalues to zero
        eigvals[eigvals < 0] = 0
        # Reconstruct the matrix
        return eigvecs @ np.diag(eigvals) @ eigvecs.T + eps*np.eye(grm.shape[0])

def H2(y, grm, l = 'REML'):
    from scipy.optimize import minimize
    from scipy.linalg import cholesky, solve_triangular
    from scipy.stats import multivariate_normal as mvn
    """
    Estimate heritability (h2) using maximum likelihood estimation with Cholesky decomposition.

    :param y: Phenotypic values (numpy.ndarray), shape (N,) or (N, T)
    :param grm: Genetic Relationship Matrix (numpy.ndarray), shape (N, N)
    :return: Estimated heritability (h2)
    """
    y = np.array(y).flatten()
    grm = np.array(grm)
    obs = np.where(~np.isnan(y))[0]
    y = y[obs]
    grm = np.array(grm)
    grm = grm[obs.reshape(-1,1), obs]
    N = y.shape[0]
    def _L(h2):
        sp = np.nanvar(y, axis=0, ddof=1)  # Phenotypic variance
        sg = h2 * sp  # Genetic variance
        se = sp - sg  # Residual variance
        m = np.nanmean(y, axis=0)  # Mean of y
        cov = sg * grm + se * np.eye(grm.shape[0])
        L = cholesky(cov, lower=True)  # Cholesky decomposition
        # Log likelihood calculation using Cholesky decomposition
        alpha = solve_triangular(L, y - m, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_likelihood = -0.5 * (np.dot(alpha.T, alpha) + log_det + y.shape[0] * np.log(2 * np.pi))
        if l == 'REML':
            X = np.ones((N, 1))
            XtX_inv = np.linalg.inv(X.T @ X)
            P = np.eye(N) - X @ XtX_inv @ X.T
            P_log_det = np.linalg.slogdet(P)[1]  # log determinant of P
            log_likelihood = log_likelihood - 0.5 * P_log_det
        return -log_likelihood
    result = minimize(fun=_L, x0=0.5, bounds=[(0., 1.)])
    return result.x[0]

def H2SVD(y, grm, l='REML', n_components = None):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import solve_triangular
    from sklearn.utils.extmath import randomized_svd
    
    # Convert to arrays and select observed values.
    y = np.array(y).flatten()
    grm = np.array(grm)
    obs = np.where(~np.isnan(y))[0]
    y = y[obs]
    grm = grm[np.ix_(obs, obs)]
    N = y.shape[0]
    if n_components is None: n_components =grm.shape[0]
    # Precompute phenotypic variance and mean.
    sp = np.nanvar(y, ddof=1)  # total phenotypic variance
    m = np.nanmean(y)
    U, s, _ = randomized_svd(grm, n_components=n_components, random_state=0)
    def _L(h2_arr):
        h2 = h2_arr[0]
        sg = h2 * sp      # genetic variance
        se = sp - sg      # residual variance
        log_det = np.sum(np.log(sg * s + se))
        r = y - m
        Ur = np.dot(U.T, r)
        quad = np.sum((Ur**2) / (sg * s + se))
        log_likelihood = -0.5 * (quad + log_det + N * np.log(2 * np.pi))
        if l == 'REML':
            Xt_cov_inv_X = np.sum(1.0 / (sg * s + se))
            log_likelihood -= 0.5 * np.log(Xt_cov_inv_X)
        return -log_likelihood
    result = minimize(fun=_L, x0=[0.5], bounds=[(0., 1.)])
    return result.x[0]

def SemiDefPosGRM(grm, eps = 1e-6):
    grm = np.array(grm)
    try: 
        np.linalg.cholesky(grm)
        return grm
    except:
        eigvals, eigvecs = np.linalg.eigh(grm)
        # Set any negative eigenvalues to zero
        eigvals[eigvals < 0] = 0
        # Reconstruct the matrix
        return eigvecs @ np.diag(eigvals) @ eigvecs.T + eps*np.eye(grm.shape[0])

def query_gene(genelis: list, species: str) -> pd.DataFrame:
    """
    Query gene information from mygene API. 
    !!!! Use the taxid of the species to guarante it works!!!!
    
    This function takes a list of gene symbols and a species identifier, queries the mygene API, and returns a dataframe
    with detailed information about each gene. The species identifier is translated to a format recognized by the API.

    Steps:
    1. Initialize the mygene API client.
    2. Translate the species identifier.
    3. Query the mygene API with the list of gene symbols.
    4. Process the results into a pandas dataframe.
    5. Ensure all expected columns are present in the dataframe.
    6. Filter out entries where the gene was not found.
    7. Return the processed dataframe.

    :param genelis: List of gene symbols.
    :type genelis: list
    :param species: Species identifier.
    :type species: str
    :return: Dataframe with gene information.
    :rtype: pandas.DataFrame

    Example:
    >>> genes = ["BRCA1", "TP53"]
    >>> species = "rn7"
    >>> df = query_gene(genes, species)
    >>> print(df.head())
    """
    mg = mygene.MyGeneInfo()
    a = mg.querymany(genelis , scopes='all', fields='all', species=species, verbose = False, silent = True)
    res = pd.concat(pd.DataFrame({k:[v]  for k,v in x.items()}) for x in a)
    res = res.assign(**{k:np.nan for k in (set(['AllianceGenome','symbol', 'ensembl', 'notfound']) - set(res.columns))} )
    return res[res.notfound.isna()].set_index('query')

def time_eval(f):
    """
    Decorator to time the execution of a function.

    This decorator wraps a function to measure and print its execution time. The execution time is printed in seconds
    with four decimal places.

    Steps:
    1. Record the start time before calling the function.
    2. Call the wrapped function.
    3. Record the end time after the function returns.
    4. Calculate the execution time.
    5. Print the function name, arguments, and execution time.

    :param f: Function to be timed.
    :type f: function
    :return: Wrapper function.
    :rtype: function

    Example:
    >>> @time_eval
    ... def example_function(x):
    ...     return x * x
    >>> example_function(3)
    func:'example_function' args:[(3,), {}] took: 0.0001 sec
    9
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap
    


def annotate_vep_online(snpdf: pd.DataFrame, species: str, snpcol: str = 'SNP', 
                        refcol: str = 'A2', altcol: str = 'A1', refseq: int = 1, 
                        expand_columns: bool = True, intergenic: bool = False):
    import requests, sys
    server = "https://rest.ensembl.org"
    ext = f"/vep/{species}/hgvs"
    res =  '"' + snpdf[snpcol].str.replace(':', ':g.').str.replace('chr', '') + snpdf[refcol] + '>'+snpdf[altcol] + '"' 
    res = f"[{','.join(res)}]"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    call = '{ "hgvs_notations" : ' + res +\
            f', "refseq":{refseq}, "OpenTargets":1, "AlphaMissense":1,"Phenotypes":1, "Enformer":1,"LoF":1'\
    +' }'
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
        decoded[jsoncols] = decoded[jsoncols].map(lambda x: pd.json_normalize(x) if isinstance(x, list) else pd.DataFrame())
        if expand_columns:
            for i in jsoncols:
                tempdf = pd.concat(decoded.apply(lambda x: x[i].rename(lambda y: f'{i}_{y}', axis = 1).assign(SNP = x.SNP), axis = 1).to_list())
                tempdf = tempdf.map(lambda x: x[0] if isinstance(x, list) else x)
                tempdf.loc[:, tempdf.columns.str.contains('phenotypes')] = \
                     tempdf.loc[:, tempdf.columns.str.contains('phenotypes')].map(lambda x: x['phenotype'].replace(' ', '_') if isinstance(x, dict) else x)
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


def printwithlog(string: str)-> None:
    """
    Print a message and log it.

    This function prints a given message to the console and logs it using the logging module.

    :param string: Message to be printed and logged.
    :type string: str

    Example:
    >>> printwithlog("This is a log message.")
    This is a log message.
    """
    print(string)
    logging.info(string)
    
def merge_duplicates_fancy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicates in a dataframe by concatenating unique values.

    This function takes a dataframe and merges rows with duplicate values by concatenating unique values in each column.

    Steps:
    1. Check if the dataframe is empty. If empty, return.
    2. If the dataframe has only one row, return that row.
    3. For each column, concatenate unique values into a single string.
    4. Return the resulting series.

    :param df: Dataframe to merge duplicates.
    :type df: pandas.DataFrame
    :return: Series with merged duplicates.
    :rtype: pandas.Series

    Example:
    >>> data = {'A': [1, 1, 1], 'B': ['x', 'y', 'x'], 'C': [1.0, 1.0, 1.0]}
    >>> df = pd.DataFrame(data)
    >>> print(merge_duplicates_fancy(df))
    A    1
    B    x|y
    C    1.0
    dtype: object
    """
    if not len(df): return
    if len(df) == 1: df.iloc[0]
    return pd.Series({y:'|'.join(sorted(df[y].astype(str).unique())) for y in df.columns})

def combine_duplicated_indexes(df):
    dups = df.index.duplicated()
    return df[~dups].combine_first(df[dups])

def groupby_no_loss_merge(df: pd.DataFrame, groupcol: str)-> pd.DataFrame:
    """
    Group by a column and merge duplicates without data loss.

    This function groups a dataframe by a specified column and merges duplicates in each group by concatenating unique values.

    Steps:
    1. Group the dataframe by the specified column.
    2. Apply the merge_duplicates_fancy function to each group.
    3. Reset the index of the resulting dataframe.

    :param df: Dataframe to group by.
    :type df: pandas.DataFrame
    :param groupcol: Column to group by.
    :type groupcol: str
    :return: Grouped dataframe with merged duplicates.
    :rtype: pandas.DataFrame

    Example:
    >>> data = {'A': [1, 1, 2, 2], 'B': ['x', 'y', 'x', 'y'], 'C': [1.0, 1.0, 2.0, 2.0]}
    >>> df = pd.DataFrame(data)
    >>> print(groupby_no_loss_merge(df, 'A'))
       A    B    C
    0  1  x|y  1.0
    1  2  x|y  2.0
    """
    return df.groupby(groupcol).progress_apply(merge_duplicates_fancy).reset_index(drop = True)

def decompose_grm(grm_path: str, n_comp: int = 50, verbose: bool = True):
    """
    Decompose a GRM (Genetic Relationship Matrix) using eigen decomposition.

    This function reads a GRM file, performs eigen decomposition, and returns the top principal components.

    Steps:
    1. Read the GRM file using npplink.
    2. Perform eigen decomposition on the GRM matrix.
    3. Sort the eigenvalues and eigenvectors in descending order of eigenvalues.
    4. Calculate the explained variance for the top components.
    5. Print the explained variances if verbose is True.
    6. Create a dataframe with the top components and return it.

    :param grm_path: Path to the GRM file.
    :type grm_path: str
    :param n_comp: Number of components to decompose into.
    :type n_comp: int
    :param verbose: Whether to print detailed output.
    :type verbose: bool
    :return: Dataframe with decomposed components.
    :rtype: pandas.DataFrame

    Example:
    >>> grm_path = "path/to/grm_file"
    >>> components = decompose_grm(grm_path, n_comp=10)
    >>> print(components.head())
    """
    grmxr =  npplink.read_grm(grm_path)['grm']
    eigval, eigvec = np.linalg.eig(grmxr)
    eig_pairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(len(eigval))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    explained_var = np.array([eig_pairs[i][0] for i in range(n_comp)])/sum(list(zip(*eig_pairs))[0])
    if verbose: print(f'explained_variances:{explained_var}\
          \n total explained var:{sum(explained_var)}' )
    return pd.DataFrame(np.vstack((eig_pairs[i][1] for i in range(n_comp))).T,
             columns = [f'GRM_PC{x}' for x in range(1, n_comp+1)],
             index= grmxr.sample_0.astype(str))

def decompose_grm_pca(grm_path: str, n_comp: int = 5, verbose: bool = True):
    """
    Decompose a GRM (Genetic Relationship Matrix) using PCA (Principal Component Analysis).

    This function reads a GRM file, performs PCA, and returns the top principal components.

    Steps:
    1. Read the GRM file using npplink.
    2. Perform PCA on the GRM matrix.
    3. Print the explained variances if verbose is True.
    4. Create a dataframe with the top components and return it.

    :param grm_path: Path to the GRM file.
    :type grm_path: str
    :param n_comp: Number of components to decompose into.
    :type n_comp: int
    :param verbose: Whether to print detailed output.
    :type verbose: bool
    :return: Dataframe with decomposed components.
    :rtype: pandas.DataFrame

    Example:
    >>> grm_path = "path/to/grm_file"
    >>> components = decompose_grm_pca(grm_path, n_comp=5)
    >>> print(components.head())
    """
    grmxr =  npplink.read_grm(grm_path)['grm']
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp)
    decomp = pca.fit_transform(grmxr)
    if verbose: print(f'explained_variances:{pca.explained_variance_ratio_}\
          \ntotal explained var:{sum(pca.explained_variance_ratio_)}' )
    return pd.DataFrame(decomp, columns = [f'GRM_PC{x}' for x in range(1, decomp.shape[1]+1)],
             index= grmxr.sample_0.astype(str))

def plink(print_call: bool = False, **kwargs) -> None:
    """
    Call PLINK with the given arguments.

    This function constructs a PLINK command based on the provided keyword arguments and executes it.

    Steps:
    1. Construct the PLINK command string from the keyword arguments.
    2. Print the command if print_call is True.
    3. Execute the command using the bash function.

    :param print_call: Whether to print the PLINK command.
    :type print_call: bool
    :param kwargs: Keyword arguments for the PLINK command.
    :type kwargs: dict
    :return: None

    Example:
    >>> plink(bfile="data", out="output", make_bed="")
    """
    call = 'plink ' + ' '.join([f'--{str(k).replace("_", "-")} {str(v).replace("True", "")}'
                                for k,v in kwargs.items() if k not in ['print_call'] ])
    call = re.sub(r' +', ' ', call).strip(' ')
    bash(call, print_call=print_call)
    return

def plink2(print_call: bool = False, **kwargs) -> None:
    """
    Call PLINK with the given arguments.

    This function constructs a PLINK command based on the provided keyword arguments and executes it.

    Steps:
    1. Construct the PLINK command string from the keyword arguments.
    2. Print the command if print_call is True.
    3. Execute the command using the bash function.

    :param print_call: Whether to print the PLINK command.
    :type print_call: bool
    :param kwargs: Keyword arguments for the PLINK command.
    :type kwargs: dict
    :return: None

    Example:
    >>> plink(bfile="data", out="output", make_bed="")
    """
    call = 'plink2 ' + ' '.join([f'--{str(k).replace("_", "-")} {str(v).replace("True", "")}'
                                for k,v in kwargs.items() if k not in ['print_call'] ])
    call = re.sub(r' +', ' ', call).strip(' ')
    bash(call, print_call=print_call)
    return

def generate_datadic(rawdata: pd.DataFrame, trait_prefix: str = 'pr,sha,lga,shock', 
                     main_cov: str = 'sex,cohort,weight_surgery,box,coatcolor', 
                     project_name: str = basename(os.getcwd()), save: bool = False, description_dict: dict = {}) -> pd.DataFrame:
    """
    Generate a data dictionary from raw data.

    This function generates a data dictionary from a raw dataframe, categorizing columns as traits or covariates.

    Steps:
    1. Describe the raw dataframe and reset the index.
    2. Categorize columns based on the number of unique values.
    3. Assign covariates and descriptions to traits.
    4. Save the data dictionary to a CSV file if save is True.
    5. Return the data dictionary dataframe.

    :param rawdata: Raw data dataframe.
    :type rawdata: pandas.DataFrame
    :param trait_prefix: comma separated list of trait prefixes
    :type trait_prefix: str
    :param main_cov: Main covariates.
    :type main_cov: str
    :param project_name: Project name.
    :type project_name: str
    :param save: Whether to save the data dictionary to a file.
    :type save: bool
    :return: Data dictionary dataframe.
    :rtype: pandas.DataFrame

    Example:
    >>> rawdata = pd.read_csv("raw_data.csv")
    >>> datadic = generate_datadic(rawdata, project_name="my_project", save=True)
    >>> print(datadic.head())
    """
    if isinstance(trait_prefix, str): trait_prefix = trait_prefix.split(',')
    dd_new = rawdata.describe(include = 'all').T.reset_index(names= ['measure'])
    if 'unique' not in dd_new.columns: dd_new['unique'] = np.nan
    dd_new['trait_covariate'] = dd_new['unique'].apply(lambda x: 'covariate_categorical' if not np.isnan(x) else 'covariate_continuous')
    dd_new.loc[dd_new.measure.str.contains('^'+'|^'.join(trait_prefix), regex = True), ['trait_covariate', 'covariates']] = ['trait', main_cov]
    dd_new.loc[dd_new.measure.str.endswith('_age') , ['trait_covariate', 'covariates']] = ['covariate_continuous', '']
    dd_new['description'] = dd_new['measure'].map(lambda x: x if x not in description_dict.keys() else description_dict[x])
    for pref in trait_prefix:
        addcov =','+','.join(dd_new.loc[dd_new.measure.str.endswith('_age') &  dd_new.measure.str.startswith(pref)].description)
        if len(addcov): 
            dd_new.loc[dd_new.measure.str.startswith(pref) & dd_new.trait_covariate.isin(['trait']), 'covariates'] += addcov
    dd_new.loc[dd_new.measure.isin(['rfid', 'labanimalid']), 'trait_covariate'] = 'metadata'
    dd_new = dd_new.loc[~dd_new.measure.str.startswith('Unnamed: ')]
    def remove_outofboundages(s, trait):
       try:
           min, max = pd.Series(list(map(int , re.findall(r'\d{2}', trait)))).agg(['min', 'max'])
           #print(min, max)
           aa =  "|".join( [str(x).zfill(2) for x in range(min, max+1)])
       except: aa = '-9999999'
       return ','.join([x for x in s.split(',') if re.findall(f'({aa}).*(age)', x) or ('age' not in x)])
    dd_new.loc[dd_new.trait_covariate == 'trait', 'covariates'] = dd_new.loc[dd_new.trait_covariate == 'trait']\
          .apply(lambda row: remove_outofboundages(row.covariates,row.measure), axis = 1)
    dd_new.loc[dd_new.trait_covariate == 'trait', 'covariates'] = dd_new.loc[dd_new.trait_covariate == 'trait', 'covariates'].str.strip(',')
    if save:
        dd_new.to_csv(f'data_dict_{project_name}.csv')
    return dd_new

def make_LD_plot(ys: pd.DataFrame, fname: str)-> None:
    """
    Create a Linkage Disequilibrium (LD) plot.

    This function takes a dataframe of genotypes, computes the LD matrix, and generates a plot with both a scatter plot
    of LD versus distance and a heatmap of the LD matrix.

    Steps:
    1. Compute the squared correlation matrix for the genotypes.
    2. Melt the correlation matrix into a long-format dataframe.
    3. Compute the distance between SNPs.
    4. Create a scatter plot of LD versus distance.
    5. Create a heatmap of the LD matrix.
    6. Save the plots to a file.

    :param ys: Dataframe of genotypes.
    :type ys: pandas.DataFrame
    :param fname: Filename to save the plot.
    :type fname: str
    :return: None

    Example:
    >>> genotypes = pd.read_csv("genotypes.csv")
    >>> make_LD_plot(genotypes, "ld_plot.png")
    """
    r2mat = ys.corr()**2 
    dist_r2 = r2mat.reset_index(names = ['snp1']).melt(id_vars = 'snp1', value_name='r2')
    dist_r2['distance'] = (dist_r2.snp1.str.extract(r':(\d+)').astype(int) - dist_r2.snp.str.extract(r':(\d+)').astype(int)).abs()
    fr2, axr2 = plt.subplots(1, 2,  figsize=(20,10))
    sns.regplot(dist_r2.assign(logdils = 1/np.log10(dist_r2.distance+100)).sample(2000, weights='logdils'), 
                x='distance', y = 'r2' ,logistic = True, line_kws= {'color': 'red'},
               scatter_kws={'linewidths':1,'edgecolor':'black', 'alpha': .2},  ax = axr2[0])
    sns.heatmap(r2mat, ax = axr2[1], cmap = 'Greys' , cbar = False, vmin = 0, vmax = 1 )
    sns.despine()
    plt.savefig(fname)
    plt.close()

@numba.njit()
def nan_sim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the similarity between two vectors, handling NaN values.

    This function computes the similarity between two vectors, where similarity is defined as the inverse of the
    sum of absolute differences, ignoring NaN values.

    Steps:
    1. Compute the sum of absolute differences between x and y.
    2. Return the inverse of the sum plus one.

    :param x: First vector.
    :type x: numpy.ndarray
    :param y: Second vector.
    :type y: numpy.ndarray
    :return: Similarity value.
    :rtype: float

    Example:
    >>> x = np.array([1, 2, np.nan, 4])
    >>> y = np.array([1, 2, 3, 4])
    >>> print(nan_sim(x, y))
    0.3333333333333333
    """
    o = np.nansum(np.abs(x-y))
    if ~np.isnan(o): return 1/(o+1)
    return 0

def rgb2hex(r: int, g: int, b:int) -> str:
    """
    Convert RGB values to HEX.

    This function converts red, green, and blue values (each ranging from 0 to 255) to a hexadecimal string.

    Steps:
    1. Convert each RGB value to a hexadecimal string.
    2. Concatenate the hex values to form a hex color code.

    :param r: Red value.
    :type r: int
    :param g: Green value.
    :type g: int
    :param b: Blue value.
    :type b: int
    :return: HEX color code.
    :rtype: str

    Example:
    >>> print(rgb2hex(255, 0, 0))
    #ff0000
    """
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

@numba.njit()
def nan_dist(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the distance between two vectors, handling NaN values.

    This function computes the distance between two vectors, where distance is defined as the sum of absolute
    differences, ignoring NaN values.

    Steps:
    1. Compute the sum of absolute differences between x and y.
    2. Return the sum or a large value if the sum is NaN.

    :param x: First vector.
    :type x: numpy.ndarray
    :param y: Second vector.
    :type y: numpy.ndarray
    :return: Distance value.
    :rtype: float

    Example:
    >>> x = np.array([1, 2, np.nan, 4])
    >>> y = np.array([1, 2, 3, 4])
    >>> print(nan_dist(x, y))
    1.0
    """
    o = np.nansum(np.abs(x-y))
    if ~np.isnan(o): return o
    return 1e10

def combine_hex_values(d):
    """
    Combine multiple HEX color values into a single HEX value.

    This function takes a dictionary of HEX color values and their associated weights, and combines them into a single
    HEX color value.

    Steps:
    1. Sort the dictionary items by weight.
    2. Compute the weighted average for red, green, and blue components.
    3. Convert the weighted average components to a HEX color code.

    :param d: Dictionary of HEX values and weights.
    :type d: dict
    :return: Combined HEX color code.
    :rtype: str

    Example:
    >>> colors = {'ff0000': 1, '00ff00': 1, '0000ff': 1}
    >>> print(combine_hex_values(colors))
    555555
    """
    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
    green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
    blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
    zpad = lambda x: x if len(x)==2 else '0' + x
    return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])

def get_topk_values(s: pd.Series, k: int) -> np.ndarray: 
    """
    Get the top-k values in a series based on their frequency.

    This function takes a pandas series and returns the top-k values based on their frequency in the series.

    Steps:
    1. Group the series by values and count their occurrences.
    2. Sort the values by frequency in descending order.
    3. Return the top-k values.

    :param s: Input series.
    :type s: pandas.Series
    :param k: Number of top values to return.
    :type k: int
    :return: Top-k values.
    :rtype: numpy.ndarray

    Example:
    >>> s = pd.Series(['a', 'b', 'a', 'c', 'b', 'a'])
    >>> print(get_topk_values(s, 2))
    ['a', 'b']
    """
    return s.groupby(s.values).agg(lambda x: '|'.join(x.index) ).sort_index()[::-1].values[:k]

def _distance_to_founders(subset_geno: str,founderfile: str,
                          fname: str,c: int, scaler: str = 'ss', 
                          verbose = False, nautosomes = 20) -> None:
    """
    Compute the distance to founder strains and plot the results.

    This function computes the genetic distance from each sample to founder strains and plots the results as a clustermap.

    Steps:
    1. Read the genotype data for both the subset and the founders.
    2. Subset the SNPs based on chromosome and select a reduced set of SNPs.
    3. Scale the genotype data.
    4. Compute the distance from each sample to the founders.
    5. Plot the results as a clustermap and save the plot.

    :param subset_geno: Path to the subset genotype file or data.
    :type subset_geno: str or tuple
    :param founderfile: Path to the founder genotype file or data.
    :type founderfile: str or tuple
    :param fname: Filename to save the plot.
    :type fname: str
    :param c: Chromosome number.
    :type c: int
    :param scaler: Scaling method.
    :type scaler: str
    :param verbose: Whether to print detailed output.
    :type verbose: bool
    :param nautosomes: Number of autosomes.
    :type nautosomes: int
    :return: None

    Example:
    >>> _distance_to_founders("subset_geno.bed", "founderfile.bed", "distance_plot.png", 1)
    """
    if type(founderfile) == str: bimf, famf, genf =npplink.load_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(subset_geno) == str: bim, fam, gen = npplink.load_plink(subset_geno)
    else: bim, fam, gen = subset_geno
    if str(c).lower() in [str(nautosomes+2), 'y']: fam = fam[fam.gender == '1']
    
    snps = bim[bim['chrom'] == c]
    snps = snps[::snps.shape[0]//2000+1]
    ys = pd.DataFrame(gen[snps.i][:, fam.i].compute().T,columns = snps.snp, index = fam.iid )
    
    if str(c).lower() in [str(nautosomes+2), 'x']:
        ys.loc[fam.query('gender == "2"').iid, :] *= 2
    founder_gens = npplink.plink2df( (bimf, famf, genf),snplist= list(snps.snp))
    if (aa := bimf.merge(snps, on = 'snp', how = "inner").query('a0_x != a0_y')).shape[0]:
        printwithlog('allele order mixed between founders and samples')
        display(aa)
    Scaler = {'tfidf': make_pipeline(KNNImputer(), TfidfTransformer()), 
              'ss': StandardScaler(), 'passthrough': make_pipeline('passthrough')}
    founder_colors = defaultdict(lambda: 'white', {'BN': '#000000', 'ACI':'#D32F2F', 'MR': '#8E24AA', 'M520':'#FBC02D',
                      'F344': '#388E3C', 'BUF': '#1976D2', 'WKY': '#F57C00', 'WN': '#02D3D3'})
    shared_snps = list(set(ys.columns) & set(founder_gens.columns))
    merged1 = pd.concat([ys[shared_snps], founder_gens[shared_snps]])
    merged1.loc[:, :] = Scaler[scaler].fit_transform(merged1) if scaler != 'tfidf' \
                        else Scaler[scaler].fit_transform(merged1).toarray()
    dist2f = pd.DataFrame(cdist(merged1.loc[ys.index], merged1.loc[founder_gens.index] , metric = nan_sim),
                          index = ys.index, columns=founder_gens.index)
    matchdf = dist2f.apply(lambda x: pd.Series(get_topk_values(x,2)), axis = 1)\
                    .set_axis(['TopMatch', "2ndMatch"], axis = 1)\
                    .fillna(method='ffill', axis = 1)
    matchdfcolors = matchdf.map(lambda x: '#'+combine_hex_values({founder_colors[k][1:]: 1 for k in x.split('|')}))
    genders = fam.set_index('iid').loc[dist2f.index.to_list(), :].gender.map(lambda x: ['white','steelblue', 'pink'][int(x)]).to_frame()
    rowcols = pd.concat([genders,matchdfcolors], axis = 1)
    #sns.clustermap(dist2f , cmap = 'turbo',  figsize= (8, 8), 
    #               square = True, norm=PowerNorm(4, vmax = 1, vmin = 0), 
    #                row_colors=rowcols) #vmin = dist2f.melt().quantile(.099).value,norm=LogNorm(),
    sns.clustermap(dist2f.div(dist2f.sum(axis = 1), axis = 0).fillna(0) , cmap = 'turbo',  figsize= (15, 15), 
                norm=PowerNorm(.5, vmax = 1, vmin = 0), 
                row_colors=rowcols)
    printwithlog(f'the following founders are present in the chr {c}:\n')
    display(matchdf.TopMatch.value_counts().to_frame())
    plt.savefig(fname)
    if verbose: plt.show()
    plt.close()
    return

def _make_umap_plot(subset_geno: str, founderfile: str, fname: str, c: int, verbose: bool = False, nautosomes: int = 20) -> None:
    """
    Create a UMAP plot comparing samples to founder strains.

    This function creates a UMAP plot to visualize the genetic similarity between samples and founder strains.

    Steps:
    1. Read the genotype data for both the subset and the founders.
    2. Subset the SNPs based on chromosome and select a reduced set of SNPs.
    3. Scale the genotype data.
    4. Compute the UMAP embedding.
    5. Plot the UMAP embedding and save the plot.

    :param subset_geno: Path to the subset genotype file or data.
    :type subset_geno: str or tuple
    :param founderfile: Path to the founder genotype file or data.
    :type founderfile: str or tuple
    :param fname: Filename to save the plot.
    :type fname: str
    :param c: Chromosome number.
    :type c: int
    :param verbose: Whether to print detailed output.
    :type verbose: bool
    :param nautosomes: Number of autosomes.
    :type nautosomes: int
    :return: None

    Example:
    >>> _make_umap_plot("subset_geno.bed", "founderfile.bed", "umap_plot.png", 1)
    """
    if type(founderfile) == str: bimf, famf, genf = npplink.load_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(subset_geno) == str: bim, fam, gen = npplink.load_plink(subset_geno)
    else: bim, fam, gen = subset_geno
    
    if str(c).lower() in [str(nautosomes+2), 'y']: fam = fam[fam.gender == '1']
    snps = bim[bim['chrom'] == c]
    snps = snps[::snps.shape[0]//2000+1]
    ys = pd.DataFrame(gen[snps.i][:, fam.i].compute().T,columns = snps.snp, index = fam.iid )
    if str(c).lower() in [str(nautosomes+2), 'x']:  ys.loc[fam.query('gender == "2"').iid, :] *= 2
    
    founder_gens = npplink.plink2df( (bimf, famf, genf),snplist= list(snps.snp))
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


def get_founder_snp_balance_old(plinkpath:str, snplist: list):
    fsnps = npplink.plink2df(plinkpath=plinkpath, snplist=snplist)
    fsnps = (fsnps > 0).astype(object)
    for r in fsnps.index: fsnps.loc[r, fsnps.loc[r, :]] = r
    fsnps.replace(False, np.nan, inplace = True)
    def _get_smallest_class_(x):
        class1 = set(x.dropna())
        class2 = set(fsnps.index) - class1
        if not len(class1): return '|'.join(sorted(class2))
        if not len(class2): return '|'.join(sorted(class1))
        if len(class1) == len(class2):
            return sorted(['|'.join(sorted(class1)), '|'.join(sorted(class2))])[0]
        sort = sorted([class1, class2], key = len)[0]
        return '|'.join(sorted(sort))
    fsplit = fsnps.agg(_get_smallest_class_).value_counts()
    return fsplit.index[0] + f'({round(fsplit.values[0]/fsnps.shape[1]*100)}%)'

def get_founder_snp_balance(plinkpath:str, snplist: list, return_agg = True, round_scale = -1, add_percentage = True):
    fsnps = npplink.plink2df(plinkpath=plinkpath, snplist= snplist)
    fsnps = (fsnps > 0).astype(object)
    for r in fsnps.index: fsnps.loc[r, fsnps.loc[r, :]] = r
    fsnps.replace(False, np.nan, inplace = True)
    translate_dict = {}
    for mask in itertools.product([True, False], repeat = len(fsnps) ):
        g1 = tuple(x for x,m in zip(fsnps.index, mask) if m) 
        g2 = tuple(x for x,m in zip(fsnps.index, mask) if not m)
        if len(g1) == len(g2): gf = sorted(['|'.join(sorted(g1)), '|'.join(sorted(g2))])[0]
        else: gf = '|'.join(sorted([g1, g2], key = len)[0])
        if not len(gf): gf = 'fixed snp'
        translate_dict[g1],translate_dict[g2] = gf, gf
    res = fsnps.apply(lambda x: translate_dict[tuple(x.dropna())])
    if not return_agg: return res
    vcnt = res.value_counts()
    pct = int(round(vcnt.values[0]/len(res)*100, round_scale))
    return vcnt.index[0] + (f'({pct}%)' if add_percentage else '')
    
def UMAP_HDBSCAN_(plinkpath: str, c: str= None, pos_start: int = None, pos_end:int= None, r2_window = 3000, nsnps = None,
                  founderbimfambed= None, keep_xaxis = True, decompose_samples = False, subsample = None, hdbscan_confidence=.7):
    if isinstance(plinkpath, pd.DataFrame): snps = plinkpath
    else: snps = npplink.plink2df(plinkpath=plinkpath, c=c, pos_start=pos_start, pos_end=pos_end)
    if nsnps is not None: 
        snps = snps.loc[:, ::max(1,snps.shape[1]//nsnps)]
    if decompose_samples: 
        pipe = make_pipeline(SimpleImputer(), PCA(n_components=.9999))
        snps = pd.DataFrame(pipe.fit_transform(snps.dropna(how = 'all').T).T, columns = snps.columns)
    if subsample is not None: snps = snps.sample(n = subsample)
    r2test = off_diagonalR2(snps, r2_window)
    r2testm = r2test.rename(lambda x: int(x.split(':')[-1]), axis = 1)\
                          .rename(lambda x: int(x.split(':')[-1]), axis = 0)\
                          .stack().reset_index().rename({0: 'value'}, axis = 1)
    r2testm = r2testm.query('bp1<bp2')
    r2testm['bp'] = (r2testm.bp1+r2testm.bp2) / 2
    r2testm['y'] = -abs(r2testm.bp - r2testm.bp1)
    r2testm = r2testm.astype(float).dropna(subset='value')
    #sns.heatmap(r2test)
    hdb = HDBSCAN(min_cluster_size=60, metric= 'precomputed', gen_min_span_tree=True, allow_single_cluster=True, min_samples=10)
    hdbpd = pd.DataFrame(hdb.fit_predict(1-r2test.fillna(0).astype(np.float64)),columns = ['hdbscan_QTL'], index = snps.columns )
    hdbpd['confidence'] = hdb.probabilities_
    hdbpd['hdbscan_QTL'] = hdbpd.apply(lambda x: x.hdbscan_QTL if x.confidence>.1 else -1, axis = 1).astype(int)
    hdbpd['QTLCluster'] = hdbpd.hdbscan_QTL.map(lambda x: f'C{x+1}'.replace('C0', 'C~'))
    #display(hdbpd.groupby('QTLCluster')['confidence'].describe())
    hdbpd['confidence_s'] = (5*hdbpd['confidence']+1)**2+2
    hdbpd['bp'] = hdbpd.index.to_series().map(lambda x: int(x.split(':')[-1]))
    um_ = umap.UMAP(metric = 'precomputed', n_components=3, n_neighbors=30)
    umap_vals =  um_.fit_transform(1-r2test.fillna(0).astype(np.float64))
    hdbpd[['UMAP1','UMAP2','UMAP3']] = umap_vals
    scaledum = (umap_vals - umap_vals.min()) / (umap_vals.max() - umap_vals.min())
    hdbpd['color'] = [mplrgb2hex(list(i)) for i in scaledum]
    hdbscan_col = (umap_vals.max(axis = 0)-umap_vals.min(axis = 0)).argmax() + 1
    cluster_order = {int(v):int(k+1) for k,v in enumerate(hdbpd.sort_values(f'UMAP{hdbscan_col}').hdbscan_QTL.unique())}
    cluster_order_inv = {v:k for k,v in cluster_order.items()}
    hdbpd['clusternum'] = hdbpd.hdbscan_QTL.map(cluster_order)#.astype(int)
    
    if founderbimfambed is not None:
        print('mapping founder qtls')
        if isinstance(founderbimfambed, str): founderbimfambed = npplink.load_plink(founderbimfambed)
        newclusterlabels = hdbpd.query('hdbscan_QTL>=0').query(f'confidence>{hdbscan_confidence}').reset_index(names = 'SNPS').groupby('QTLCluster')['SNPS']\
                                .progress_apply(lambda s:get_founder_snp_balance(founderbimfambed, list(s))).to_dict()
        hdbpd['QTLCluster'] = hdbpd.QTLCluster.map(defaultdict(lambda:'~C', newclusterlabels))
    ### hdbscan clustering
    if len(hdbpd.query(f'confidence>{hdbscan_confidence}')):
        hdbpdaggs = hdbpd.query(f'confidence>{hdbscan_confidence}').reset_index(names= 'SNP').groupby('hdbscan_QTL')\
                     .apply(lambda df: pd.Series([df.bp.min(), df.nsmallest(15, 'bp').SNP.values,
                                                  df.bp.max(), df.nlargest(15, 'bp').SNP.values,
                                                 ]))\
                     .set_axis(['start','startSNPS','end', 'endSNPS'], axis = 1).sort_values('start')
        hdbpdaggs = hdbpdaggs[hdbpdaggs.index != -1]
        hdbpdaggs['endSNP'] = hdbpdaggs['endSNPS'].map(lambda x: x[-1])
        hdbpdaggs['startSNP'] = hdbpdaggs['startSNPS'].map(lambda x: x[0])
        def connect_clusters(row):
            back = hdbpd.query(f'confidence>{hdbscan_confidence}')\
                 .groupby('hdbscan_QTL')\
                 .apply(lambda df: np.array(df.query(f'bp<{row.start}').nlargest(15, 'bp').index))\
                 .reset_index()\
                 .set_axis(['endhdbscan_QTL','endSNPS'], axis = 1).query('endhdbscan_QTL> 0 ')#.reset_index() #and endhdbscan_QTL != @row.name
            back = back[back.endSNPS.map(len)>0]
            back['endSNP'] = back['endSNPS'].map(lambda x: x[-1])
            back['endbp'] = back.endSNP.map(lambda x: int(x.split(':')[-1]))
            back['starthdbscan_QTL'] = row.name
            back = back.merge(row.copy().to_frame().T.filter(regex = 'start').reset_index(names = 'starthdbscan_QTL'), on = 'starthdbscan_QTL')
            back = back.rename({'start': 'startbp'}, axis = 1).assign(fb ='backwards')
            if len(back):
                back['dist'] = back.apply(lambda row: np.nansum(r2test.loc[row.startSNPS, row.endSNPS]) - (row.startbp - row.endbp)*1e-10, axis = 1).to_list()
                back['shortlist'] = np.isclose(back.dist, back.dist.max())
            
            forward = hdbpd.query(f'confidence>{hdbscan_confidence}')\
                 .groupby('hdbscan_QTL')\
                 .apply(lambda df: np.array(df.query(f'bp>{row.end}').nsmallest(15, 'bp').index))\
                 .reset_index()\
                 .set_axis(['starthdbscan_QTL','startSNPS'], axis = 1).query('starthdbscan_QTL> 0 ')#.reset_index() #and endhdbscan_QTL != @row.name
            forward = forward[forward.startSNPS.map(len)>0]
            forward['startSNP'] = forward['startSNPS'].map(lambda x: x[0])
            forward['startbp'] = forward.startSNP.map(lambda x: int(x.split(':')[-1]))
            forward['endhdbscan_QTL'] = row.name
            forward = forward.merge(row.copy().to_frame().T.filter(regex = 'end').reset_index(names = 'endhdbscan_QTL'), on = 'endhdbscan_QTL')
            forward = forward.rename({'end': 'endbp'}, axis = 1).assign(fb ='forward')
            #forward['dist'] =  forward.apply(lambda row: np.nansum(r2test.loc[row.startSNPS, row.endSNPS]), axis = 1).to_list()
            if len(forward):
                forward['dist'] =  forward.apply(lambda row: np.nansum(r2test.loc[row.startSNPS, row.endSNPS]) - (row.startbp - row.endbp)*1e-10, axis = 1).to_list()
                forward['shortlist'] = np.isclose(forward.dist, forward.dist.max())
            return pd.concat([back, forward])
        pathfig = []
        if hdbpdaggs.shape[0] < 2: edgeconnections = pd.DataFrame()
        else:
            edgeconnections = pd.concat([connect_clusters(hdbpdaggs.iloc[i]) for i in range(len(hdbpdaggs))]).reset_index()\
                                .drop_duplicates(subset = ['endhdbscan_QTL', 'endSNP', 'starthdbscan_QTL', 'startbp', 'shortlist'])
            edgeconnections = edgeconnections.sort_values('dist', ascending = False).reset_index(drop = True)
            set_start, set_end = set(), set()
            for idx, row in edgeconnections.iterrows():
                if (f'{row.endhdbscan_QTL}:{row.endbp}' not in set_end) and (f'{row.starthdbscan_QTL}:{row.startbp}' not in set_start):
                    set_start.add(f'{row.starthdbscan_QTL}:{row.startbp}')
                    set_end.add(f'{row.endhdbscan_QTL}:{row.endbp}')
                    edgeconnections.loc[idx, 'shortlist'] = True
                else:  edgeconnections.loc[idx, 'shortlist'] = False
            for idx, row in edgeconnections.query('shortlist').iterrows():
                pathfig += [hv.Path(np.array([[row.endbp, cluster_order[row.endhdbscan_QTL]], [row.startbp, cluster_order[row.starthdbscan_QTL]]]))\
                              .opts(color = 'Black', line_width = 2)]
        for idx, row in hdbpdaggs.iterrows():
            pathfig += [hv.Path(np.array([[row.start, cluster_order[idx]], [row.end, cluster_order[idx]]])).opts(color = 'Black', line_width = 1)]
        #### hdbscan clustering figure
        edgecons = [edgeconnections]
    else: 
        pathfig, edgecons = [] , []
    hyticks = hdbpd.sort_values('clusternum').apply(lambda x: (x.clusternum, x.QTLCluster), 1).drop_duplicates().to_list()
    hdbfig = hdbpd.hvplot.scatter(x='bp', y='clusternum',c = 'color',s= 'confidence_s', frame_width = 1000,width = 1900, height = 300, cmap = 'Jet',  hover_cols = list(hdbpd.columns))\
                   .opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'), xaxis='top' ,colorbar = False,yticks= hyticks, yaxis='right', yrotation=0)
    hdbpd['QTLCluster_founders'] =hdbpd['QTLCluster'].str.split('(').str[0]
    hdbpd['founder_balance'] = get_founder_snp_balance(founderbimfambed, hdbpd.index, return_agg=False)
    hdbfig2 = hdbpd.hvplot.scatter(x='bp', y='QTLCluster_founders',c = 'color',s= 'confidence_s', frame_width = 1000,width = 1900, height = 300, 
                                  cmap = 'Jet', hover_cols = list(hdbpd.columns) )\
                   .opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'), xaxis='top' ,colorbar = False, yaxis='right', yrotation=0)
    fb = hdbpd.query('confidence==1').founder_balance.value_counts()
    good_classes = fb[fb>=fb.quantile(.8)].index
    hdbfig3 = hdbpd[hdbpd.confidence.gt(.999) & hdbpd.founder_balance.isin(good_classes)]\
        .hvplot.scatter(x='bp', y='founder_balance',c = 'color',s= 'confidence_s', frame_width = 1000,width = 1900, height = 300, 
         cmap = 'Jet', hover_cols = list(hdbpd.columns) )\
        .opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'), xaxis='top' ,colorbar = False, yaxis='right', yrotation=0)
    if not keep_xaxis: 
        hdbfig = hdbfig.opts(xaxis = None)
        hdbfig2 = hdbfig2.opts(xaxis = None)
    # for idx, row in edgeconnections.query('shortlist').iterrows():
    hdbfig = reduce(lambda x,y: x*y , [hdbfig] + pathfig)
    r2fig = r2testm.hvplot.scatter(x='bp', y='y',c = 'value', datashade=True,  cmap = 'gnuplot2_r', frame_width = 1000, height = 200, width = 1900, colorbar = True)\
                   .opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'), yaxis = None, xaxis = None)
    return {'figures': [hdbfig, r2fig, hdbfig2, hdbfig3], 'dataframes': [hdbpd, edgecons], 'r2': r2test, 'snps': snps}

def add_sex_specific_traits(raw_data: pd.DataFrame = None, data_dictionary:pd.DataFrame=None, project_name = basename(os.getcwd()), path = '', save = False):
    if raw_data is None: 
        if os.path.exists(f'{path}raw_data_without_sexsplit.csv'):
            raw_data = pd.read_csv(f'{path}raw_data_without_sexsplit.csv')
        else: raw_data = pd.read_csv(f'{path}raw_data.csv')
    elif isinstance(raw_data, str): raw_data = pd.read_csv(raw_data)
    if data_dictionary is None: 
        if os.path.exists(f'{path}data_dict_{project_name}_without_sexsplit.csv'):
            data_dictionary = pd.read_csv(f'{path}data_dict_{project_name}_without_sexsplit.csv', index_col='measure')
        else: data_dictionary = pd.read_csv(f'{path}data_dict_{project_name}.csv', index_col='measure')
    elif isinstance(data_dictionary, str): data_dictionary = pd.read_csv(data_dictionary, index_col='measure')
    if 'measure' in data_dictionary.columns:
        data_dictionary.set_index('measure', inplace = True)
    traits = data_dictionary[~data_dictionary.index.astype(str).str.contains('_males$|_females$')].query('trait_covariate == "trait"').index
    raw_data = pd.concat([raw_data,
                raw_data.loc[raw_data.sex.astype(str).str.lower().isin(['m', 'male','sire', '1']), traits].rename(lambda x: f'{x}_males', axis = 1),
                raw_data.loc[raw_data.sex.astype(str).str.lower().isin(['f', 'female','dame','2']), traits].rename(lambda x: f'{x}_females', axis = 1),
               ],axis = 1)
    for i, sex in itertools.product(traits, ['males', 'females']):
        data_dictionary.loc[f'{i}_{sex}'] = data_dictionary.loc[i].copy()
        data_dictionary.loc[f'{i}_{sex}', 'covariates'] = data_dictionary.loc[f'{i}_{sex}', 'covariates'].replace('sex', '').strip(',').replace(',,', ',')
        desc = raw_data[f'{i}_{sex}'].describe()
        data_dictionary.loc[f'{i}_{sex}', desc.index] = desc.values
        data_dictionary.loc[f'{i}_{sex}', 'description'] = data_dictionary.loc[f'{i}', 'description'] + f' just {sex} included'
    raw_data = raw_data.loc[:, ~raw_data.columns.str.contains(r'^index$|^level_\d+$')]
    raw_data = raw_data.T.dropna(how = 'all').T
    #data_dictionary = data_dictionary.query('count > 10')
    if save:
        if os.path.exists(f'{path}raw_data.csv'):
            pd.read_csv(f'{path}raw_data.csv').to_csv(f'{path}raw_data_without_sexsplit.csv')
        raw_data.to_csv(f'{path}raw_data.csv')
        if os.path.exists(f'{path}data_dict_{project_name}.csv'):
            pd.read_csv(f'{path}data_dict_{project_name}.csv').to_csv(f'{path}data_dict_{project_name}_without_sexsplit.csv')
        data_dictionary.to_csv(f'{path}data_dict_{project_name}.csv')
    return raw_data, data_dictionary

class vcf_manipulation:
    """
    Class for VCF file manipulation.

    This class provides methods for manipulating VCF (Variant Call Format) files, including reading and writing VCF files,
    and extracting metadata.

    Methods:
    - corrfunc: Calculate and annotate the correlation coefficient.
    - get_vcf_header: Get the header of a VCF file.
    - read_vcf: Read a VCF file into a pandas or dask dataframe.
    - get_vcf_metadata: Extract metadata from a VCF file.
    - pandas2vcf: Convert a pandas dataframe to a VCF file.
    """
    
    @staticmethod
    def corrfunc(x, y, ax=None, **kws) -> None:
        """
        Calculate and annotate the correlation coefficient on a plot.

        This function calculates the Pearson correlation coefficient between x and y and annotates it on the given plot axis.

        Steps:
        1. Calculate the Pearson correlation coefficient.
        2. Annotate the correlation coefficient on the plot.

        :param x: First variable.
        :type x: numpy.ndarray
        :param y: Second variable.
        :type y: numpy.ndarray
        :param ax: Matplotlib axis to annotate on.
        :type ax: matplotlib.axes._axes.Axes, optional
        :param kws: Additional keyword arguments.
        :return: None

        Example:
        >>> x = np.array([1, 2, 3, 4])
        >>> y = np.array([2, 3, 4, 5])
        >>> fig, ax = plt.subplots()
        >>> sns.scatterplot(x, y, ax=ax)
        >>> vcf_manipulation.corrfunc(x, y, ax=ax)
        """
        r, _ = pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

    @staticmethod
    def get_vcf_header(vcf_path: str) -> str:
        """
        Get the header of a VCF file.

        This function reads the header lines of a VCF file to extract the column names.

        Steps:
        1. Open the VCF file in read mode with gzip compression.
        2. Read lines until the header line starting with "#CHROM" is found.
        3. Return the header line split into column names.

        :param vcf_path: Path to the VCF file.
        :type vcf_path: str
        :return: List of column names.
        :rtype: list

        Example:
        >>> header = vcf_manipulation.get_vcf_header("variants.vcf.gz")
        >>> print(header)
        """
        with gzip.open(vcf_path, "rt") as ifile:
            for num, line in enumerate(ifile):
                if line.startswith("#CHROM"): return line.strip().split('\t')
                if num > 10000: return '-1'
        return '-1'

    @staticmethod
    def read_vcf(filename: str, method: str = 'pandas') -> pd.DataFrame:
        """
        Read a VCF file into a pandas or dask dataframe.

        This function reads a VCF file using pandas or dask, depending on the specified method.

        Steps:
        1. Get the header of the VCF file.
        2. Read the VCF file using the specified method (pandas or dask).

        :param filename: Path to the VCF file.
        :type filename: str
        :param method: Method to use for reading the file ('pandas' or 'dask').
        :type method: str
        :return: Dataframe with VCF data.
        :rtype: pandas.DataFrame or dask.dataframe.DataFrame

        Example:
        >>> vcf_df = vcf_manipulation.read_vcf("variants.vcf.gz", method='pandas')
        >>> print(vcf_df.head())
        """
        if method == 'dask':
            return dd.read_csv(filename,  compression='gzip', comment='#',  sep =r'\s+', header=None, 
                               names = vcf_manipulation.get_vcf_header(filename),blocksize=None,  dtype=str, ).repartition(npartitions = 100000)
        # usecols=['#CHROM', 'POS']
        return pd.read_csv(filename,  compression='gzip', comment='#',  sep =r'\s+',
                           header=None, names = vcf_manipulation.get_vcf_header(filename),  dtype=str )

    
    @staticmethod
    def get_vcf_metadata(vcf_path: str) -> str:
        """
        Extract metadata from a VCF file.

        This function extracts the metadata lines from a VCF file (lines starting with '##').

        Steps:
        1. Open the VCF file in read mode with gzip compression.
        2. Read lines until the header line starting with "#CHROM" is found.
        3. Concatenate and return the metadata lines.

        :param vcf_path: Path to the VCF file.
        :type vcf_path: str
        :return: Metadata lines concatenated as a string.
        :rtype: str

        Example:
        >>> metadata = vcf_manipulation.get_vcf_metadata("variants.vcf.gz")
        >>> print(metadata)
        """
        out = ''
        with gzip.open(vcf_path, "rt") as ifile:
            for num, line in enumerate(ifile):
                if line.startswith("#CHROM"): return out
                out += line 
        return '-1'

    @staticmethod
    def pandas2vcf(df: pd.DataFrame, filename: str, metadata: str = '') -> None:
        """
        Convert a pandas dataframe to a VCF file.

        This function converts a pandas dataframe to a VCF file format, including optional metadata.

        Steps:
        1. Open the output file in write mode.
        2. Write the metadata to the file.
        3. Write the dataframe to the file, tab-separated.

        :param df: Dataframe with VCF data.
        :type df: pandas.DataFrame
        :param filename: Path to the output VCF file.
        :type filename: str
        :param metadata: Metadata to include in the VCF file.
        :type metadata: str
        :return: None

        Example:
        >>> data = {'#CHROM': [1, 1], 'POS': [1000, 1001], 'ID': ['rs1', 'rs2'], 'REF': ['A', 'G'], 'ALT': ['C', 'T']}
        >>> df = pd.DataFrame(data)
        >>> vcf_manipulation.pandas2vcf(df, "output.vcf")
        """
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
        df.astype({"POS":int,'QUAL':int }).to_csv(filename, sep="\t", mode='a', index=False)
    
    @staticmethod
    def reformat_sample(x, fmt):
        y = x.str.split(':', expand = True).set_axis(fmt.index, axis = 1)
        mi, vals = [], []
        for idx, i in fmt.iterrows():
            mi += [np.array(tuple(itertools.product([x.name],[idx] , list(range(1, i.Number+1))))).T]
            if i.Number > 1:  
                vals += [y[idx].str.split(',', expand = True).astype(i.Type).rename(lambda x:f'{x}', axis = 1)]
            else: 
                y[idx] =  y[idx].astype(i.Type)
                vals += [y[idx].astype(i.Type)]
        mi = pd.MultiIndex.from_arrays(np.concatenate(mi, axis = 1), names=('sample', 'format', 'idx'))
        return pd.concat(vals, axis = 1).set_axis(mi, axis = 1)
    
    @staticmethod
    def vcf2df_lossless(url, multithread = False):
        def tryexcepteval(x):
            try: return eval(x)
            except: return np.nan
        def tryexceptnum(x):
            try: return int(x)
            except: return np.nan
        df = vcf_manipulation.read_vcf(url).drop('FORMAT', axis = 1)
        metadata = pd.Series(vcf_manipulation.get_vcf_metadata(url).split('\n')[1:-1])
        metadata= metadata.str[2:].str.split('=<', expand = True).set_axis(['type','specs'], axis = 1)
        metadata=pd.concat([metadata.iloc[:, :-1], 
                           pd.DataFrame.from_records(('dict('+metadata.specs.str[:-1]
                                                      .str.replace('0/0, 0/1, and 1/1', '0/0_0/1_1/1')
                                                      .str.replace('"', '')\
                                                      .str.replace('=', "='")\
                                                      .str.replace(',',"',")+"')").dropna().map(eval))], axis = 1)
        metadata.Type = metadata.Type.str.lower().replace('string', 'str').map(eval)
        metadata = metadata.dropna(subset = 'Type')
        metadata.Number = metadata.Number.replace('.', '0').astype(int)
        metadata= metadata.set_index('ID')
        metadata = metadata.groupby('type', group_keys=False).apply(lambda df: df.assign(pos=range(df.shape[0])), include_groups=True)
        fmt = metadata[metadata['type']=="FORMAT"]
        info = pd.DataFrame.from_records(('dict('+df['INFO'].str.replace(';', ',').replace('.', "_= np.nan") + ')').map(eval)).set_index(df.index)
        df.drop('INFO', axis = 1, inplace = True)
        df[info.columns]= info
        df.set_index(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER'] +info.columns.to_list(), inplace = True )
        #df2.xs(('HD'), level = ('format'), axis = 1 )
        if multithread and os.cpu_count() >= 10:
            import dask.bag as db
            return pd.concat(db.from_sequence(df[x] for x in df.columns).map(vcf_manipulation.reformat_sample, fmt = fmt).compute(scheduler = 'threads'))
        return pd.concat((vcf_manipulation.reformat_sample(df[x], fmt) for x in tqdm(df.columns)), axis = 1)
    

def bash(call: str, verbose: bool = False, return_stdout: bool = True, 
         print_call: bool = True,silent: bool = False,shell: bool = False):
    """
    Execute a shell command using subprocess.

    This function runs a shell command using the subprocess module with various options for output handling and command execution.

    Steps:
    1. Clean up the command string by removing extra spaces.
    2. Print the command if print_call is True.
    3. Execute the command using subprocess.run.
    4. Optionally print the standard output if verbose is True and return_stdout is False.
    5. Print any error messages unless silent is True.
    6. Return the standard output if return_stdout is True, otherwise return the subprocess.CompletedProcess object.

    :param call: Shell command to execute.
    :type call: str
    :param verbose: Whether to print the standard output.
    :type verbose: bool, optional
    :param return_stdout: Whether to return the standard output.
    :type return_stdout: bool, optional
    :param print_call: Whether to print the command before executing.
    :type print_call: bool, optional
    :param silent: Whether to suppress printing error messages.
    :type silent: bool, optional
    :param shell: Whether to execute the command through the shell.
    :type shell: bool, optional
    :return: Standard output if return_stdout is True, otherwise subprocess.CompletedProcess object.
    :rtype: list or subprocess.CompletedProcess

    Example:
    >>> output = bash("ls -l", verbose=True, return_stdout=True)
    >>> print(output)
    """
    call = re.sub(r' +', ' ', call).strip(' ')
    if print_call: printwithlog(call)
    out = subprocess.run(call if shell else call.split(' '), capture_output = True,  shell =shell, ) 
    if verbose and (not return_stdout) and (not silent): printwithlog(out.stdout)
    if out.stderr and (not silent): 
        try:printwithlog(out.stderr.decode('ascii'))
        except: printwithlog(out.stderr.decode('utf-8'))
    if return_stdout: 
        try: oo =  out.stdout.decode('ascii').strip().split('\n')
        except: oo =  out.stdout.decode('utf-8').strip().split('\n')
        return oo
    return out

def vcf2plink(vcf: str = 'round9_1.vcf.gz', n_autosome: int = 20, out_path: str = 'zzplink_genotypes/allgenotypes_r9.1'):
    """
    Convert a VCF file to PLINK format using PLINK.

    This function uses PLINK to convert a VCF file to PLINK binary format (BED, BIM, FAM files).

    Steps:
    1. Construct the PLINK command with specified parameters.
    2. Execute the PLINK command using the bashbash function.

    :param vcf: Path to the VCF file to be converted.
    :type vcf: str, optional
    :param n_autosome: Number of autosomes to set for PLINK.
    :type n_autosome: int, optional
    :param out_path: Output path for the generated PLINK files.
    :type out_path: str, optional
    :return: None

    Example:
    >>> vcf2plink(vcf='sample.vcf.gz', n_autosome=22, out_path='plink_output/sample')
    """
    bash(f'plink --thread-num 16 --vcf {vcf} --chr-set {n_autosome} no-xy --keep_allele_order --set-hh-missing --set-missing-var-ids @:# --make-bed --out {out_path}')


def impute_single_trait(dataframe: pd.DataFrame, imputing_col: str , covariate_cols: list, groupby: list, scaler = StandardScaler(), imputer = SoftImpute(verbose = False)):
    """
    Impute missing values in a specified column using specified covariates and imputation method.

    This function imputes missing values in the specified column(s) of a dataframe, using specified covariate columns and groupings. 
    The imputation is performed using a specified scaler and imputer.

    Steps:
    1. Ensure input columns and groupings are in list format.
    2. Prepare the dataframe by removing duplicate columns and setting appropriate indexes.
    3. Scale numeric covariate columns.
    4. Apply one-hot encoding to categorical covariate columns.
    5. Create a dataframe with the columns to impute and all covariates.
    6. Calculate the percentage of missing data.
    7. Perform multiple rounds of imputation with varying percentages of missing data to estimate imputation error.
    8. Apply the imputer to impute missing values, considering the specified groupings.
    9. Compile the results and return the imputed values and quality control metrics.

    :param dataframe: Dataframe containing the data.
    :type dataframe: pandas.DataFrame
    :param imputing_col: Column(s) to impute.
    :type imputing_col: str or list
    :param covariate_cols: Covariate columns to use for imputation.
    :type covariate_cols: list
    :param groupby: Columns to group by for imputation.
    :type groupby: list
    :param scaler: Scaler to use for numeric covariates.
    :type scaler: sklearn.preprocessing.StandardScaler
    :param imputer: Imputer to use for missing value imputation.
    :type imputer: SoftImpute
    :return: Series containing the imputed values and quality control metrics.
    :rtype: pandas.Series

    Example:
    >>> df = pd.DataFrame({
    ...     'rfid': [1, 2, 3, 4],
    ...     'trait': [1.0, 2.0, np.nan, 4.0],
    ...     'cov1': [0.5, 1.5, 0.5, 1.5],
    ...     'cov2': [1.0, 1.0, 2.0, 2.0],
    ...     'group': ['A', 'A', 'B', 'B']
    ... })
    >>> result = impute_single_trait(df, 'trait', ['cov1', 'cov2'], 'group')
    >>> print(result['imputed'])
    >>> print(result['qc'])
    """
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

def regressoutgb(dataframe: pd.DataFrame, data_dictionary: pd.DataFrame, covariates_threshold: float = 0.02, groupby = ['sex'], normalize = 'quantile', model = LinearRegression()):
    """
    Regress out covariates from traits in a dataframe 

    This function performs regression to remove the effect of covariates from trait columns in a dataframe. The covariates
    and traits are defined in a data dictionary. The regression is done using generalized boosting, and the results can
    be normalized.

    Steps:
    1. Load the data dictionary if it is provided as a file path.
    2. Prepare the dataframe and the data dictionary.
    3. One-hot encode categorical covariates.
    4. Identify all traits and covariates.
    5. Iterate over groups in the dataframe to perform regression and normalization.
    6. Compile the explained variances and regression results.
    7. Return the final dataframe with regressed traits and quality control metrics.

    :param dataframe: Dataframe containing the data.
    :type dataframe: pandas.DataFrame
    :param data_dictionary: Data dictionary defining traits and covariates.
    :type data_dictionary: pandas.DataFrame or str
    :param covariates_threshold: Threshold for including covariates based on explained variance.
    :type covariates_threshold: float, optional
    :param groupby: Columns to group by for regression.
    :type groupby: list or str, optional
    :param normalize: Method to normalize the regressed traits ('quantile' or 'boxcox').
    :type normalize: str, optional
    :return: Dictionary containing the regressed dataframe and quality control metrics.
    :rtype: dict

    Example:
    >>> df = pd.DataFrame({
    ...     'rfid': [1, 2, 3, 4],
    ...     'trait': [1.0, 2.0, 3.0, 4.0],
    ...     'cov1': [0.5, 1.5, 0.5, 1.5],
    ...     'cov2': [1.0, 1.0, 2.0, 2.0],
    ...     'sex': ['M', 'F', 'M', 'F']
    ... })
    >>> data_dict = pd.DataFrame({
    ...     'measure': ['trait', 'cov1', 'cov2'],
    ...     'trait_covariate': ['trait', 'covariate_continuous', 'covariate_continuous'],
    ...     'covariates': ['', 'trait,cov1,cov2', 'trait,cov1,cov2']
    ... })
    >>> result = regressoutgb(df, data_dict, groupby='sex')
    >>> print(result['regressed_dataframe'])
    >>> print(result['covariatesr2'])
    """
    if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
    df, datadic = dataframe.copy(), data_dictionary
    def getcols(df, string): return df.columns[df.columns.str.contains(string)].to_list()
    dd_covs_covcol = (set(datadic.covariates.replace('', np.nan).str.strip(',').dropna().str.split(',').sum()) )
    if type(groupby) == 'str': groupby = [groupby]
    categorical_all = list(datadic[datadic.trait_covariate.eq("covariate_categorical")\
                                   &datadic.measure.isin(dd_covs_covcol)\
                                   &datadic.measure.isin(df.columns)].measure)
    if len(categorical_all): df[categorical_all] = df[categorical_all].astype("category")
    dfohe = df.copy()
    ohe = OneHotEncoder()
    oheencoded = ohe.fit_transform(dfohe[categorical_all].astype(str)).todense()
    #dfohe[[f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_all)]] = oheencoded
    dfohe.loc[:, [f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_all)]] = oheencoded
    #alltraits = list(datadic.query('trait_covariate == "trait"').measure.unique())
    alltraits = list(datadic[datadic.trait_covariate.eq("trait")\
                            &datadic.measure.isin(df.columns)].measure.unique())
    def getdatadic_covars(trait):
        covars = set(datadic.set_index('measure').loc[trait, 'covariates'].split(','))
        covars =  covars - set(groupby)
        covars = covars | set(itertools.chain.from_iterable([ getcols(dfohe, F'OHE_{x}') for x in covars]))
        covars = covars - set(datadic[datadic.trait_covariate.eq("covariate_categorical")&\
                                      datadic.measure.isin(dd_covs_covcol)].measure)
        return list(covars)
    #continuousall = list(set(datadic.query('trait_covariate == "covariate_continuous" & measure.isin(@dd_covs_covcol)').measure) & set(df.columns))
    continuousall = list(datadic[datadic.trait_covariate.eq("covariate_continuous")\
                                   &datadic.measure.isin(dd_covs_covcol)\
                                   &datadic.measure.isin(df.columns)].measure)
    explained_vars = []
    reglist = []
    if not len(groupby): dfohe, groupby = dfohe.assign(tempgrouper = 'A'), ['tempgrouper']
    for group, gdf in tqdm(dfohe.groupby(groupby, observed=False)):
        groupaddon = '|'.join(group)
        if continuousall:
            gdf = gdf.astype({x: float for x in continuousall})
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
                reg = statsReport.regress_out(gdf.set_index('rfid'), list(expvars.columns),  list(expvars.index), model = model).rename(lambda x: x.lower(), axis = 1)
            reg = statsReport.ScaleTransformer(reg, reg.columns, method = normalize)
            reglist += [reg.reset_index().melt(id_vars='rfid')]
    melted_explained_vars = pd.concat(explained_vars).reset_index(drop = True)[['variable', 'group', 'value']]
    regresseddf = pd.concat(reglist).pivot(columns= 'variable', index = 'rfid').droplevel(0, axis = 1)
    regresseddf = statsReport.ScaleTransformer(regresseddf, regresseddf.columns, method = normalize)
    outdf = pd.concat([df.set_index('rfid'), regresseddf], axis = 1)
    outdf = outdf.loc[:,~outdf.columns.duplicated()]
    passthrough_traits = list(datadic.query('trait_covariate == "trait" and covariates == "passthrough"').measure.unique())
    if len(passthrough_traits):
        outdf[['regressedlr_' + x for x in passthrough_traits]] = outdf[passthrough_traits]
    return {'regressed_dataframe': outdf.reset_index().sort_values('rfid'), 
            'covariatesr2': melted_explained_vars,
            'covariatesr2pivoted': pd.DataFrame(melted_explained_vars.groupby('variable')['group'].apply(list)).reset_index()}

    
def _prophet_reg(dforiginal: pd.DataFrame,y_column: str = 'y',
                 categorical_regressors: list =[], regressors: list = [],
                 ds_column: str = 'age', rolling_avg_days: float = 0, seasonality: list = [],
                 growth: str = 'logistic', removed_months: list = [], removed_weekday: list = [],
                 removed_hours: list = [], index_col: str = 'rfid',return_full_df: bool = False,
                 save_explained_vars: bool = False, threshold: float = 0.00,
                 path: str = '', extra_label: str = ''):
    """
    Apply Prophet regression to a time series data.

    This function uses Facebook's Prophet to fit a regression model to time series data, allowing for advanced 
    modeling options including logistic growth, seasonality, and the use of additional regressors.

    Steps:
    1. Prepare the dataframe by setting the index and converting the date column to datetime format.
    2. Apply a rolling average if specified.
    3. One-hot encode categorical regressors.
    4. Configure Prophet model settings including seasonality and growth strategy.
    5. Remove specified months, weekdays, and hours from the data.
    6. Filter out regressors based on the explained variance threshold.
    7. Fit the Prophet model to the data.
    8. Generate future predictions and create plots for model diagnostics.
    9. Save the preprocessing plots and return the regressed data.

    :param dforiginal: Original dataframe containing the time series data.
    :type dforiginal: pandas.DataFrame
    :param y_column: Name of the column containing the target variable.
    :type y_column: str
    :param categorical_regressors: List of categorical regressor column names.
    :type categorical_regressors: list
    :param regressors: List of additional regressor column names.
    :type regressors: list
    :param ds_column: Name of the column containing the datetime values.
    :type ds_column: str
    :param rolling_avg_days: Number of days for rolling average.
    :type rolling_avg_days: float
    :param seasonality: List of seasonality components to include.
    :type seasonality: list
    :param growth: Growth strategy for the Prophet model ('logistic' or 'linear').
    :type growth: str
    :param removed_months: List of months to remove from the data.
    :type removed_months: list
    :param removed_weekday: List of weekdays to remove from the data.
    :type removed_weekday: list
    :param removed_hours: List of hours to remove from the data.
    :type removed_hours: list
    :param index_col: Column name to use as the index.
    :type index_col: str
    :param return_full_df: Whether to return the full dataframe with forecasts.
    :type return_full_df: bool
    :param save_explained_vars: Whether to save the explained variance of the regressors.
    :type save_explained_vars: bool
    :param threshold: Threshold for explained variance to include regressors.
    :type threshold: float
    :param path: Path to save the preprocessing plots and explained variances.
    :type path: str
    :param extra_label: Extra label to append to saved files.
    :type extra_label: str
    :return: Dataframe with the regressed target variable.
    :rtype: pandas.DataFrame

    Example:
    >>> df = pd.DataFrame({
    ...     'rfid': [1, 2, 3, 4],
    ...     'age': pd.date_range(start='2020-01-01', periods=4, freq='D'),
    ...     'y': [1.0, 2.5, 3.0, 4.5],
    ...     'cat': ['A', 'A', 'B', 'B'],
    ...     'reg1': [0.5, 1.5, 0.5, 1.5],
    ...     'reg2': [1.0, 1.0, 2.0, 2.0]
    ... })
    >>> result = _prophet_reg(df, y_column='y', categorical_regressors=['cat'], regressors=['reg1', 'reg2'])
    >>> print(result)
    """    
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

def regressout_timeseries(dataframe: pd.DataFrame, data_dictionary: pd.DataFrame, 
                          covariates_threshold: float = 0.02, groupby_columns: list = ['sex'],  normalize: str = 'quantile',
                          ds_column: str = 'age', save_explained_vars: bool = True, path: str = '') -> pd.DataFrame:
    """
    Regress out covariates from time series traits in a dataframe using Prophet regression.

    This function applies Prophet regression to remove the effect of covariates from trait columns in a dataframe, based on the definitions provided in a data dictionary. The regression is performed separately for each group defined by the groupby columns.

    Steps:
    1. Prepare the data dictionary and dataframe.
    2. Adjust covariate columns by removing groupby columns.
    3. Define helper functions to get covariates and perform regression.
    4. Apply Prophet regression to each trait in the data dictionary.
    5. Handle grouping if specified.
    6. Save explained variances if required.
    7. Apply quantile transformation to the regressed traits.
    8. Handle passthrough traits if any.
    9. Save the processed data if required.

    :param dataframe: Dataframe containing the data.
    :type dataframe: pandas.DataFrame
    :param data_dictionary: Data dictionary defining traits and covariates.
    :type data_dictionary: pandas.DataFrame
    :param covariates_threshold: Threshold for including covariates based on explained variance.
    :type covariates_threshold: float, optional
    :param groupby_columns: Columns to group by for regression.
    :type groupby_columns: list or str, optional
    :param ds_column: Name of the column containing the datetime values.
    :type ds_column: str
    :param save_explained_vars: Whether to save the explained variance of the regressors.
    :type save_explained_vars: bool, optional
    :param path: Path to save the explained variances and processed data.
    :type path: str, optional
    :return: Dataframe with the regressed traits.
    :rtype: pandas.DataFrame

    Example:
    >>> df = pd.DataFrame({
    ...     'rfid': [1, 2, 3, 4],
    ...     'age': pd.date_range(start='2020-01-01', periods=4, freq='D'),
    ...     'trait': [1.0, 2.5, 3.0, 4.5],
    ...     'sex': ['M', 'F', 'M', 'F'],
    ...     'cov1': [0.5, 1.5, 0.5, 1.5],
    ...     'cov2': [1.0, 1.0, 2.0, 2.0]
    ... })
    >>> data_dict = pd.DataFrame({
    ...     'measure': ['trait', 'cov1', 'cov2'],
    ...     'trait_covariate': ['trait', 'covariate_continuous', 'covariate_continuous'],
    ...     'covariates': ['', 'trait,cov1,cov2', 'trait,cov1,cov2']
    ... })
    >>> result = regressout_timeseries(df, data_dict, groupby_columns='sex', ds_column='age', save_explained_vars=True, path='/path/to/save/')
    >>> print(result)
    """
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
    
    outdf['regressedlr_'+ddtraits.measure] = statsReport.ScaleTransformer(outdf,'regressedlr_'+ ddtraits.measure, method = normalize)
    passthrough_traits = list(ddtraits.query('covariates == "passthrough"').measure.unique())
    if len(passthrough_traits):
        outdf[['regressedlr_' + x for x in passthrough_traits]] = outdf[passthrough_traits]
    
    if save_explained_vars: 
        outdf.to_csv(f'{path}processed_data_ready.csv', index = False)
        outdf.to_csv(f'{path}results/processed_data_ready_n{outdf.shape[0]}_date{datetime.today().strftime("%Y-%m-%d")}.csv', index = False)
    return outdf

def plotly_read_from_html(file: str):
    """
    Read a Plotly figure from an HTML file.

    This function extracts the Plotly figure data and layout from an HTML file and recreates the Plotly figure.

    Steps:
    1. Open and read the HTML file.
    2. Extract the Plotly call arguments from the HTML content.
    3. Reconstruct the Plotly JSON object from the extracted arguments.
    4. Create and return the Plotly figure from the JSON object.

    :param file: Path to the HTML file containing the Plotly figure.
    :type file: str
    :return: Reconstructed Plotly figure.
    :rtype: plotly.graph_objs._figure.Figure

    Example:
    >>> fig = plotly_read_from_html('plot.html')
    >>> fig.show()
    """
    with open(file, 'r') as f: html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot(.∗)(.*)', html[-2**16:])[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}    
    return plotio.from_json(json.dumps(plotly_json))
    
def fancy_display(df: pd.DataFrame, download_name: str = 'default.csv', max_width = 1400, max_height = 600, flexible = False, page_size = 20, wrap_text = 'wrap',
                  cell_font_size=12, header_font_size=14, layout =  'fit_data_fill', max_cell_width = 150, add_search = True, add_sort = True,
                  html_cols = None ,**kws) -> pn.widgets.Tabulator:
    """
    Display a dataframe with interactive filtering and downloading options using Panel Tabulator.

    This function displays a dataframe with interactive filters for each column and provides an option to download the filtered data as a CSV file.

    Steps:
    1. Initialize the Panel extension with Tabulator.
    2. Drop unnecessary columns from the dataframe.
    3. Round numeric columns to 3 decimal places.
    4. Create filters for numeric and non-numeric columns.
    5. Create a Tabulator widget with pagination and header filters.
    6. Add a download button for saving the table as a CSV file.
    7. Return a Panel Column containing the download button and the Tabulator widget.

    :param df: Dataframe to display.
    :type df: pandas.DataFrame
    :param download_name: Default name for the downloaded CSV file.
    :type download_name: str
    :return: Panel Column containing the interactive table and download button.
    :rtype: panel.layout.Column

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> fancy_display(df, 'my_table.csv')
    """
    pn.extension('tabulator')
    df = df.drop(['index'] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
    numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
    try:df[numeric_cols] = df[numeric_cols].map(round, ndigits=3)
    except: df[numeric_cols] = df[numeric_cols].applymap(round, ndigits=3)
    if not add_search: d = {}
    else: d = {x : {'type': 'number', 'func': '>=', 'placeholder': 'Enter minimum'} for x in numeric_cols} | \
              {x : {'type': 'input', 'func': 'like', 'placeholder': 'Similarity'} for x in df.columns[~df.columns.isin(numeric_cols)]}
    formatters = {}
    if html_cols:
        html_cols = [c for c in html_cols if c in df.columns]  # guard against missing
        if html_cols: formatters.update({c: {"type": "html"} for c in html_cols})
    tab_kwargs = dict(pagination='local', page_size=page_size, header_filters=d, layout=layout, show_index=False,
                       formatters=formatters, max_width=max_width, max_height=max_height,  **kws)
    if not add_sort: tab_kwargs["configuration"] = {"columnDefaults": {"headerSort": False}}
    download_table = pn.widgets.Tabulator(df, **tab_kwargs)
    if flexible:
        download_table.stylesheets = [""".tabulator { font-size: 10px; }
        .tabulator .tabulator-row .tabulator-cell 
         {padding: 0 8px; line-height: 1.3; max-width:""" + f'{max_cell_width}px'+"""; white-space: """ + wrap_text+""";font-size: """ + f'{cell_font_size}px'+""" }
        .tabulator .tabulator-header .tabulator-col { padding: 0 1px;line-height: 1.5;font-size: """ + f'{header_font_size}px'+"}"]

    filename, button = download_table.download_menu(text_kwargs={'name': 'Enter filename', 'value': download_name},button_kwargs={'name': 'Download table'})
    return pn.Column(pn.Row(filename, button), download_table)

import html
def pn_Iframe(file,width = 600, height = 600 ,return_html_iframe=False,**kws):
    if type(file)!=str:
        html_buffer = StringIO()
        hv.save(file,html_buffer)
        t = html.escape(html_buffer.getvalue())
    else:
        with open(file) as f: t = html.escape(f.read())
    iframe_code = f'<iframe srcdoc="{t}" style="height:100%; width:100%" frameborder="0"></iframe>'
    if return_html_iframe: return iframe_code
    return pn.pane.HTML(iframe_code, max_width = 1000, max_height = 1000, height = height, width=width, **kws)

def regex_plt(df, rg, max_cols = 10, full = True):
    if full:
        seq = '|'.join(rg) if not isinstance(rg, str) else rg
        table = fancy_display(df.loc[:, ~df.columns.str.contains(seq)], max_height=400, max_width=1000, download_name= 'regex_plot.csv')
        return pn.Column(table,  regex_plt(df,rg, max_cols = max_cols, full = False))
    if not isinstance(rg, str):
        fig = reduce(lambda x,y: x+y, [regex_plt(df, x, full = False).opts(shared_axes = False) for x in rg]).opts(shared_axes = False).cols(max_cols)
        return fig
    sset = df.filter(regex = rg).T
    return (sset.hvplot(kind='line', rot= 45, grid =True) \
           * sset.hvplot(kind='scatter', marker='o', size=50, rot= 45,line_width = 1, line_color='black', alpha = .7 ,  legend = False, title = rg, grid =True))\
           .opts(frame_width = 300, frame_height = 300,show_legend = False, title = rg)
    
def plotly_histograms_to_percent(fig):
    """
    Convert Plotly histograms to display percentages.

    This function modifies Plotly histograms in a figure to display data as percentages instead of counts.

    Steps:
    1. Iterate through each trace in the figure.
    2. Check if the trace is a histogram.
    3. Set the histogram function to 'count' and normalization to 'probability'.
    4. Set the number of bins for both x and y axes.
    5. Update the hover template to display percentages.

    :param fig: Plotly figure containing histograms.
    :type fig: plotly.graph_objs._figure.Figure
    :return: Modified Plotly figure with histograms displaying percentages.
    :rtype: plotly.graph_objs._figure.Figure

    Example:
    >>> fig = px.histogram(df, x="column")
    >>> fig = plotly_histograms_to_percent(fig)
    >>> fig.show()
    """
    for trace in fig.data:
        if type(trace) == plotly.graph_objs._histogram.Histogram:
            trace.histfunc = 'count'
            trace.histnorm = 'probability'
            trace.nbinsx = trace.nbinsy = 30
            trace.hovertemplate = trace.hovertemplate.replace('<br>count=%', '<br>percent=%')
    return fig

def sql2pandas(file: str):
    """
    Convert an SQLite database file to a dictionary of pandas dataframes.

    This function reads all tables from an SQLite database file and converts them into pandas dataframes.

    Steps:
    1. Connect to the SQLite database file.
    2. Retrieve the list of tables in the database.
    3. Read each table into a pandas dataframe.
    4. Store the dataframes in a dictionary with table names as keys.
    5. Close the database connection.

    :param file: Path to the SQLite database file.
    :type file: str
    :return: Dictionary of dataframes with table names as keys.
    :rtype: pandas.DataFrame

    Example:
    >>> data_dict = sql2pandas('database.sqlite')
    >>> print(data_dict['table_name'])
    """
    import sqlite3
    conn = sqlite3.connect(file)
    out = pd.DataFrame([[table,pd.read_sql_query(f"SELECT * FROM {table};", conn)] 
                        for table in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name']],
                       columns = ['name', 'df']).set_index('name')
    conn.close()
    return out

def translate_dict(s: str, d: dict):
    """
    Translate substrings in a string based on a dictionary of replacements.

    This function replaces substrings in the input string according to the given dictionary of replacements.

    Steps:
    1. Check if the dictionary is empty. If empty, return the original string.
    2. Use regular expressions to replace substrings based on the dictionary keys and values.

    :param s: Input string to be translated.
    :type s: str
    :param d: Dictionary of replacements where keys are substrings to be replaced and values are the replacements.
    :type d: dict
    :return: Translated string with replacements applied.
    :rtype: str

    Example:
    >>> translation_dict = {'hello': 'hi', 'world': 'earth'}
    >>> translated_string = translate_dict('hello world', translation_dict)
    >>> print(translated_string)
    'hi earth'
    """
    if not d: return s
    return re.sub(f"({')|('.join(d.keys())})",lambda y: d[str(y.group(0))] ,s )

def bayes_ppi(pvals: np.ndarray, islog10: bool = False):
    """
    Compute Bayesian posterior probabilities of inclusion (PPI) from p-values.

    This function converts p-values to Bayesian posterior probabilities of inclusion (PPI) using the inverse cumulative
    distribution function of the standard normal distribution and Bayes factors.

    Steps:
    1. Compute the z-scores from p-values using the inverse cumulative distribution function.
    2. Calculate the Bayes factors from the z-scores.
    3. Normalize the Bayes factors to obtain PPIs.

    :param pvals: Array of p-values.
    :type pvals: numpy.ndarray
    :param islog10: Whether the p-values are in log10 scale.
    :type islog10: bool, optional
    :return: Array of Bayesian posterior probabilities of inclusion.
    :rtype: numpy.ndarray

    Example:
    >>> pvals = np.array([0.01, 0.05, 0.1])
    >>> ppi = bayes_ppi(pvals)
    >>> print(ppi)
    """
    from scipy.special import ndtri
    if islog10: z = ndtri(np.power(10, -pvals)/2)
    else: z = ndtri(pvals/2)
    bfi = np.exp(np.power(z,2)/2)
    return bfi/(bfi.sum())

def credible_set_idx(ppi: np.ndarray, cs_threshold: float = .99, return_series: bool = False):
    """
    Compute the indices of the credible set based on PPIs.

    This function identifies the indices of elements that form the credible set, which contains the specified cumulative
    probability threshold of the total PPI.

    Steps:
    1. Sort the PPIs in descending order.
    2. Compute the cumulative sum of the sorted PPIs.
    3. Identify the indices where the cumulative sum is below the threshold.
    4. Return the indices of the credible set.

    :param ppi: Array or Series of posterior probabilities of inclusion.
    :type ppi: numpy.ndarray or pandas.Series
    :param cs_threshold: Cumulative probability threshold for the credible set.
    :type cs_threshold: float, optional
    :param return_series: Whether to return the result as a pandas Series.
    :type return_series: bool, optional
    :return: Indices of the credible set.
    :rtype: numpy.ndarray or pandas.Series

    Example:
    >>> ppi = np.array([0.4, 0.3, 0.2, 0.1])
    >>> cs_indices = credible_set_idx(ppi, cs_threshold=0.8)
    >>> print(cs_indices)
    """
    if isinstance(ppi, pd.Series): ppi = ppi.values
    sorted_index = np.argsort(ppi)[::-1]
    sarray = ppi[sorted_index].cumsum() < cs_threshold
    if return_series: return pd.Series(data = sarray, index = sorted_index )
    return sorted_index[sarray]
    
class gwas_pipe:
    """
    A class for running GWAS (Genome-Wide Association Study) pipelines.

    This class provides methods to set up and run GWAS pipelines, including data preparation, trait handling, and
    various configurations related to genotype data.

    Steps:
    1. Initialize attributes for paths, thresholds, and logging.
    2. Pull NCBI genome information or ask the user for genome accession.
    3. Prepare the phenotype dataframe by reading and filtering data.
    4. Handle missing RFIDs and log the information.
    5. Import traits and trait descriptions from the data dictionary or set to empty list.
    6. Create directory structure and save trait files.
    7. Initialize paths for various genotype and result files.
    8. Estimate p-value threshold if required.

    :param path: Path to the project directory.
    :type path: str, optional
    :param project_name: Name of the project.
    :type project_name: str, optional
    :param data: Dataframe containing the phenotypic data.
    :type data: pandas.DataFrame, optional
    :param traits: List of traits to analyze.
    :type traits: list, optional
    :param trait_descriptions: List of descriptions for the traits.
    :type trait_descriptions: list, optional
    :param chrList: List of chromosomes to analyze.
    :type chrList: list, optional
    :param all_genotypes: Path to the genotype files.
    :type all_genotypes: str, optional
    :param founderfile: Path to the founder genotype files.
    :type founderfile: str, optional
    :param phewas_db: Path to the PheWAS database.
    :type phewas_db: str, optional
    :param threshold: Threshold for significance.
    :type threshold: float or str, optional
    :param threshold05: Threshold for 5% significance.
    :type threshold05: float, optional
    :param genome: Genome version.
    :type genome: str, optional
    :param genome_accession: Genome accession number.
    :type genome_accession: str, optional
    :param threads: Number of threads to use.
    :type threads: int, optional

    Example:
    >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project', data=pd.DataFrame(), traits=['trait1', 'trait2'])
    >>> gwas.run()
    """
    def __init__(self,
                 all_genotypes: str ,
                 founderfile: str = '' ,
                 path: str = f'{Path().absolute()}/', 
                 project_name: str = basename(os.getcwd()),
                 data: pd.DataFrame = pd.DataFrame,
                 traits: list = [],
                 trait_descriptions: list = [],
                 chrList: list = [], 
                 phewas_db: str = 'phewasdb.parquet.gz', 
                 threshold: float = 'auto',
                 threshold05: float = 5.643286,
                 genome_accession: str = 'GCF_015227675.2',
                 threads: int = os.cpu_count()): 

        self.gcta = 'gcta64' #if not gtca_path else gtca_path
        self.path = path.rstrip('/') + '/' if len(path) else ''
        self.all_genotypes = all_genotypes
        self.threshold = threshold
        self.threshold05 = threshold05

        logging.basicConfig(filename=f'{self.path}gwasRun.log', 
                            filemode='w', level=logging.INFO, format='%(asctime)s %(message)s') #INFO

           
        if os.path.exists(f'{self.path}temp'): bash(f'rm -r {self.path}temp')
        try: self.pull_NCBI_genome_info(genome_accession,redownload = False)
        except Exception as e:
            print(e)
            self.ask_user_genome_accession()
        if not chrList:
            self.chrList = lambda:   [self.replacenumstoXYMT(i) for i in range(1,self.n_autosome+5) if i != self.n_autosome+3]
        else: self.chrList = lambda: [i for i in chrList]
        
        if type(data) == str: 
            df = pd.read_csv(data, dtype={'rfid': str}).replace([np.inf, -np.inf], np.nan)
        else: 
            df = data.replace([np.inf, -np.inf], np.nan)
            df['rfid'] = df.rfid.astype(str)
        df.columns = df.columns.str.lower()
        if 'vcf' in self.all_genotypes:
            sample_list_inside_genotypes = vcf_manipulation.get_vcf_header(self.all_genotypes)
        else:
            sample_list_inside_genotypes = pd.read_csv(self.all_genotypes+'.fam', header = None, sep=r'\s+', dtype = str)[1].to_list()
        df = df.sort_values('rfid').reset_index(drop = True).dropna(subset = 'rfid').drop_duplicates(subset = ['rfid'])
        self.df = df[df.rfid.astype(str).isin(sample_list_inside_genotypes)].copy()#.sample(frac = 1)
        self.df = self.df.loc[self.df.rfid.astype(str).map(lambda x:hashlib.sha3_512(x.encode()).digest()).sort_values().index, :].reset_index(drop = True)
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
        self.df[self.traits] = self.df[self.traits].apply(pd.to_numeric, errors = 'coerce')
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
            try: self.foundersbimfambed = npplink.load_plink(founderfile)
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
        """
        Clear specified directories within the project path.
    
        This function removes all files and subdirectories within the specified directories, except for the report.
        It then recreates the directory structure.
    
        Steps:
        1. Iterate over each folder in the provided list of folders.
        2. Remove the folder and its contents.
        3. Log the removal of the folder.
        4. Recreate the directory structure.
    
        :param folders: List of folders to clear.
        :type folders: list, optional
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.clear_directories()
        """
        for folder in folders:
            os.system(f'rm -r {self.path}{folder}')
            printwithlog(f'removing file {self.path}{folder}')
        self.make_dir_structure()

    def impute_traits(self, data_dictionary: str = None, groupby_columns: list = ['sex'], crosstrait_imputation: bool = False, trait_subset: list = []):
        """
        Impute missing values for traits in the dataframe.
    
        This function performs imputation of missing values for traits in the dataframe using specified covariates and 
        optional grouping. It supports cross-trait imputation and subset trait imputation.
    
        Steps:
        1. Load the data dictionary if provided as a file path.
        2. Prepare the dataframe by removing duplicated columns.
        3. Extract and process the covariates for each trait.
        4. Perform imputation for each trait using the specified covariates and grouping.
        5. Combine the imputed values with the original dataframe.
        6. Save the imputed data and quality control metrics to CSV files.
    
        :param data_dictionary: Path to the data dictionary CSV file.
        :type data_dictionary: str, optional
        :param groupby_columns: Columns to group by for imputation.
        :type groupby_columns: list, optional
        :param crosstrait_imputation: Whether to include other traits as covariates for imputation.
        :type crosstrait_imputation: bool, optional
        :param trait_subset: Subset of traits to impute.
        :type trait_subset: list, optional
        :return: Dataframe with imputed values.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> imputed_df = gwas.impute_traits(data_dictionary='data_dict.csv', groupby_columns=['sex'], crosstrait_imputation=True, trait_subset=['trait1', 'trait2'])
        >>> print(imputed_df)
        """
        printwithlog(f'running imputation {"groupedby:"+ ",".join(groupby_columns) if len(groupby_columns) else ""}...')
        if data_dictionary is None:
            data_dictionary = pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv')
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

    def add_sex_specific_traits(self, data_dictionary: str = None, save = False):
        self.df, datadic = add_sex_specific_traits(self.df, data_dictionary = data_dictionary, project_name = self.project_name, path = self.path, save = save)
        
        self.traits = datadic.query('trait_covariate == "trait"').index.to_list()
        for i in self.traits:
            if self.get_trait_descriptions[i] == 'UNK': self.get_trait_descriptions[i] = datadic.loc[i, 'description']

    def regressout_groupby(self, data_dictionary: pd.DataFrame, covariates_threshold: float = 0.02, groupby_columns = ['sex'], model = LinearRegression(), normalize = 'quantile'):
        """
        Regress out covariates from traits in the dataframe, grouped by specified columns.
    
        This function performs regression to remove the effect of covariates from trait columns in the dataframe, based on the
        definitions provided in a data dictionary. The regression is done separately for each group defined by the groupby columns.
    
        Steps:
        1. Run the regression to remove covariates grouped by the specified columns.
        2. Update the dataframe with the regressed data.
        3. Update the list of traits with the regressed trait columns.
        4. Generate and save a report of data distributions.
        5. Save the processed data and explained variances to CSV files.
        6. Save the regressed trait data to individual phenotype files.
        7. Update trait descriptions from the data dictionary.
    
        :param data_dictionary: Data dictionary defining traits and covariates.
        :type data_dictionary: pandas.DataFrame
        :param covariates_threshold: Threshold for including covariates based on explained variance.
        :type covariates_threshold: float, optional
        :param groupby_columns: Columns to group by for regression.
        :type groupby_columns: list, optional
        :return: Dataframe with regressed traits.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> regressed_df = gwas.regressout_groupby(data_dictionary=pd.read_csv('data_dict.csv'), covariates_threshold=0.02, groupby_columns=['sex'])
        >>> print(regressed_df)
        """
        printwithlog(f'running regressout groupedby {",".join(groupby_columns)}...')
        reg = regressoutgb(dataframe=self.df, data_dictionary=data_dictionary, groupby = groupby_columns, covariates_threshold = covariates_threshold, model = model, normalize = normalize)
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
        
    def regressout(self, data_dictionary: pd.DataFrame, covariates_threshold: float = 0.02, verbose:bool = False, model = LinearRegression(), normalize = 'quantile') -> pd.DataFrame:
        """
        Regress out covariates from traits in the dataframe.
    
        This function performs regression to remove the effect of covariates from trait columns in the dataframe, based on the
        definitions provided in a data dictionary. The regression is done without grouping.
    
        Steps:
        1. Load the data dictionary if provided as a file path.
        2. Prepare the dataframe by removing duplicated columns.
        3. One-hot encode categorical covariates.
        4. Normalize traits and continuous covariates.
        5. Identify covariates for each trait and calculate explained variances.
        6. Regress out covariates from traits and handle missing values.
        7. Normalize the regressed traits.
        8. Save the processed data and explained variances to CSV files.
        9. Update trait descriptions from the data dictionary.
    
        :param data_dictionary: Data dictionary defining traits and covariates.
        :type data_dictionary: pandas.DataFrame
        :param covariates_threshold: Threshold for including covariates based on explained variance.
        :type covariates_threshold: float, optional
        :param verbose: Whether to print detailed information during processing.
        :type verbose: bool, optional
        :return: Dataframe with regressed traits.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> regressed_df = gwas.regressout(data_dictionary=pd.read_csv('data_dict.csv'), covariates_threshold=0.02, verbose=True)
        >>> print(regressed_df)
        """
        printwithlog(f'running regressout...')
        if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
        df, datadic = self.df.copy(), data_dictionary
        datadic = datadic[datadic.measure.isin(df.columns)].drop_duplicates(subset = ['measure'])
        def getcols(df, string): return df.columns[df.columns.str.contains(string)].to_list()
        categorical_all = list(datadic.query('trait_covariate == "covariate_categorical"').measure)
        dfohe = df.copy()
        ohe = OneHotEncoder()
        oheencoded = ohe.fit_transform(dfohe[categorical_all].astype(str)).todense()
        #dfohe[[f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_all)]] = oheencoded
        dfohe.loc[:, [f'OHE_{x}'for x in ohe.get_feature_names_out(categorical_all)]] = oheencoded
        alltraits = list(datadic.query('trait_covariate == "trait"').measure.unique())
        dfohe.loc[:, alltraits] = QuantileTransformer(n_quantiles = 100).fit_transform(dfohe.loc[:, alltraits].apply(pd.to_numeric, errors='coerce'))
        continuousall = list(datadic.query('trait_covariate == "covariate_continuous"').measure)
        #print(f'all continuous variables {continuousall}')
        
        if continuousall:
            dfohe.loc[:, continuousall] = QuantileTransformer(n_quantiles = 100).fit_transform(dfohe.loc[:, continuousall])
        variablesall = []
        all_explained_vars = []
        stR = statsReport.stat_check(df)

        for name, tempdf in datadic.query('trait_covariate =="trait" and covariates != "passthrough"').groupby('covariates'):
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
        if len(all_explained_vars):
            all_explained_vars = pd.concat(all_explained_vars).drop_duplicates(subset = ['group', 'variable'])
            if verbose: display(all_explained_vars)
            all_explained_vars.to_csv(f'{self.path}melted_explained_variances.csv', index = False)
            melt_list = pd.DataFrame(all_explained_vars.groupby('variable')['group'].apply(list)).reset_index()
            melt_list.to_csv(f'{self.path}pivot_explained_variances.csv',index = False)
            tempdf = dfohe.loc[:, dfohe.columns.isin(melted_variances.group.unique())].copy()
            dfohe.loc[:, dfohe.columns.isin(melted_variances.group.unique())] = tempdf.fillna(tempdf.mean())
            aaaa = melt_list.apply(lambda x: statsReport.regress_out(dfohe,[x.variable],   x.group, model = model), axis =1)
            resid_dataset = pd.concat(list(aaaa), axis = 1)
            non_regressed_cols = [x for x in alltraits if x not in resid_dataset.columns.str.replace('regressedLR_', '')]
            non_regressed_df = df[non_regressed_cols].rename(lambda x: 'regressedLR_' + x, axis = 1)
            resid_dataset = pd.concat([resid_dataset, non_regressed_df], axis = 1)
            cols2norm = resid_dataset.columns[resid_dataset.columns.str.contains('regressedLR_')]
            resid_dataset = statsReport.ScaleTransformer(resid_dataset, cols2norm, method = normalize)
        else: 
            pd.DataFrame(columns = ['group', 'variable', 'value']).to_csv(f'{self.path}melted_explained_variances.csv', index = False)
            pd.DataFrame(columns = ['variable', 'group']).to_csv(f'{self.path}pivot_explained_variances.csv',index = False)
            resid_dataset = pd.DataFrame()
        dfcomplete = pd.concat([df,resid_dataset],axis = 1)
        dfcomplete.columns = dfcomplete.columns.str.lower()
        dfcomplete = dfcomplete.loc[:,~dfcomplete.columns.duplicated()]
        passthrough_traits = list(datadic.query('trait_covariate == "trait" and covariates == "passthrough"').measure.unique())
        if len(passthrough_traits):
            dfcomplete[['regressedlr_' + x for x in passthrough_traits]] = dfcomplete[passthrough_traits]
        allcols_regex = '|'.join(set(['rfid', 'sex'] + alltraits + continuousall + categorical_all))
        strcomplete = statsReport.stat_check(dfcomplete.reset_index(drop = True).filter(regex = allcols_regex))
        dfcomplete.to_csv(f'{self.path}processed_data_ready.csv')
        dfcomplete.to_csv(f'{self.path}results/processed_data_ready_n{self.df.shape[0]}_date{datetime.today().strftime("%Y-%m-%d")}.csv')
        self.df = dfcomplete
        strcomplete.make_report(f'{self.path}data_distributions.html')
        self.traits = list(self.df.columns[self.df.columns.str.contains('regressedlr_')])
        #self.traits = [x.lower() for x in cols2norm]
        simplified_traits = [x.replace('regressedlr_', '') for x in self.traits]
        display(self.df[self.traits].count())
        for trait in self.traits:
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(f'{self.path}data/pheno/{trait}.txt' ,  index = False, sep = ' ', header = None)
        trait_descriptions = [datadic.set_index('measure').loc[x, 'description'] if (x in datadic.measure.values) else 'UNK' for x in simplified_traits]
        self.get_trait_descriptions = defaultdict(lambda: 'UNK', {k:v for k,v in zip(self.traits, trait_descriptions)})
        return dfcomplete
    
    def regressout_timeseries(self, data_dictionary: pd.DataFrame, covariates_threshold: float = 0.02,
                              verbose: bool = False, groupby_columns: list = ['sex'], ds_column:str = 'age', save: bool = True):
        """
        Regress out covariates from time series traits in the dataframe, grouped by specified columns.
    
        This function performs regression to remove the effect of covariates from time series trait columns in the dataframe,
        based on the definitions provided in a data dictionary. The regression is done separately for each group defined by 
        the groupby columns.
    
        Steps:
        1. Load the data dictionary if provided as a file path.
        2. Prepare the dataframe by removing duplicated columns.
        3. Perform time series regression using the specified covariates and grouping.
        4. Update the dataframe with the regressed data.
        5. Generate and save a report of data distributions.
        6. Save the processed data and explained variances to CSV files.
        7. Save the regressed trait data to individual phenotype files.
        8. Update trait descriptions from the data dictionary.
    
        :param data_dictionary: Data dictionary defining traits and covariates.
        :type data_dictionary: pandas.DataFrame
        :param covariates_threshold: Threshold for including covariates based on explained variance.
        :type covariates_threshold: float, optional
        :param verbose: Whether to print detailed information during processing.
        :type verbose: bool, optional
        :param groupby_columns: Columns to group by for regression.
        :type groupby_columns: list, optional
        :param ds_column: Column name containing the datetime values.
        :type ds_column: str, optional
        :param save: Whether to save the explained variances.
        :type save: bool, optional
        :return: Dataframe with regressed traits.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> regressed_ts_df = gwas.regressout_timeseries(data_dictionary=pd.read_csv('data_dict.csv'), covariates_threshold=0.02, groupby_columns=['sex'], ds_column='age')
        >>> print(regressed_ts_df)
        """
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
        allcols_regex = '|'.join(set(['rfid', 'sex'] + list(datadic.measure)))
        strcomplete = statsReport.stat_check(self.df.reset_index(drop = True).filter(regex = allcols_regex))
        #strcomplete = statsReport.stat_check(self.df.reset_index(drop = True))
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
        """
        Add latent spaces to the dataframe using PCA and UMAP.
    
        This function adds latent space representations of the traits using PCA and UMAP, including clustering results.
        It then integrates these latent spaces into the main dataframe.
    
        Steps:
        1. Generate PCA and UMAP latent spaces and combine them.
        2. Handle missing values and dummy encoding for clusters.
        3. Integrate latent spaces with the main dataframe.
        4. Update trait descriptions with the new latent space traits.
        5. Save the processed data and generate a report of data distributions.
        6. Save the new latent space traits to individual phenotype files.
    
        :return: Updated dataframe with latent spaces.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> updated_df = gwas.add_latent_spaces()
        >>> print(updated_df)
        """
        latspace = pd.concat([self._make_eigen3d_figure(ret = 'data'), self._make_umap3d_figure(ret = 'data')], axis = 1).rename(lambda x: x.lower(), axis =1)
        latspace.loc[:, latspace.columns.str.contains(r"regressedlr_\w+_clusters")] = latspace.filter(regex = r"regressedlr_\w+_clusters").astype(str).replace('-1', np.nan)
        latspace =  pd.get_dummies(latspace, columns=latspace.filter(regex = r"regressedlr_\w+_clusters").columns, dummy_na=True, dtype = float)
        for i in latspace.columns[latspace.columns.str.contains(r'^regressedlr_\w+_clusters_nan')]:
            latspace.loc[latspace[i].astype(bool), latspace.columns.str.contains(i[:-3])] = np.nan
        latspace = latspace.loc[:, ~latspace.columns.str.contains(r'regressedlr_\w+_clusters_nan') & (latspace.count()> 10)]
        self.df = self.df.loc[:, ~self.df.columns.str.lower().str.contains('unnamed: ')]
        self.df = self.df.set_index('rfid').combine_first(latspace.rename(columns= lambda x: x.replace('regressedlr_', '')))
        self.df = self.df.combine_first(latspace).reset_index()#
        self.traits = sorted(list(set(self.traits)| set(latspace.columns)))
        for idx, row in latspace.columns.str.extract(r'(umap|pc|pca)(\d+|_clusters_\d+)').iterrows():
            if "_clusters" not in str(row[1]):
                self.get_trait_descriptions[latspace.columns[idx]] = f'{"Principal Component" if row[0] == "pc" else "UMAP" } {row[1]} of all traits for project {self.project_name}'
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
        
    def plink2Df(self, call: str, temp_out_filename: str = 'temp/temp', dtype: str = 'ld'):
        """
        Run a PLINK command and return the output as a dataframe.
    
        This function executes a specified PLINK command and reads the output file into a pandas dataframe.
    
        Steps:
        1. Generate a random ID for the temporary output file.
        2. Construct the full PLINK command.
        3. Execute the PLINK command.
        4. Read the output file into a pandas dataframe.
        5. Remove the temporary output and log files.
        6. Return the dataframe.
    
        :param call: The PLINK command to execute.
        :type call: str
        :param temp_out_filename: Base name for the temporary output file.
        :type temp_out_filename: str, optional
        :param dtype: The type of PLINK output file to read (e.g., 'ld').
        :type dtype: str, optional
        :return: Dataframe containing the PLINK output.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> ld_df = gwas.plink2Df(call='plink --bfile data', dtype='ld')
        >>> print(ld_df)
        """

        random_id = np.random.randint(1e8)
        
        full_call = re.sub(r' +', ' ', call + f' --out {self.path}{temp_out_filename}{random_id}')
        
        ### add line to delete temp_out_filename before doing the 
        bash(full_call, print_call = False)

        try: 
            out = pd.read_csv(f'{self.path}{temp_out_filename}{random_id}.{dtype}', sep = r'\s+')
            os.system(f'rm {self.path}{temp_out_filename}{random_id}.{dtype}')
            os.system(f'rm {self.path}{temp_out_filename}{random_id}.log')
        except:
            printwithlog(f"file not found at self.plink")
            out = pd.DataFrame()
        return out 
    
    def plink(self, return_file: str = 'ld', outfile: str = '',  **kwargs):
        """
        Run a PLINK command with specified arguments.
    
        This function constructs and executes a PLINK command based on provided keyword arguments. It can return the output
        as a dataframe or write it to a specified output file.
    
        Steps:
        1. Construct the PLINK command with specified arguments.
        2. If no output file is specified and return_file is set, execute the command and return the output as a dataframe.
        3. If an output file is specified, execute the command and write the output to the file.
    
        :param return_file: The type of PLINK output file to return (e.g., 'ld').
        :type return_file: str, optional
        :param outfile: Path to the output file.
        :type outfile: str, optional
        :param kwargs: Keyword arguments for the PLINK command.
        :return: None or Dataframe containing the PLINK output.
        :rtype: None or pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.plink(bfile='data', assoc='', return_file='assoc')
        """
        call = 'plink ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                    for k,v in kwargs.items()])
        if not outfile and return_file:
            return self.plink2Df(call, dtype = f'{return_file}' ) 
        else:
            bash(call + f' --out {outfile}', print_call=False)
        return
    
    def _gcta(self,  **kwargs):
        """
        Run a GCTA command with specified arguments. (if writing a flag that doesn't require a variable e.g.
        --make-grm use make_grm = '')
    
        This function constructs and executes a GCTA command based on provided keyword arguments. It is a wrapper to run
        GCTA as a Python function instead of using a string and bash call.
    
        Steps:
        1. Construct the GCTA command with specified arguments.
        2. If 'out' is not specified in the arguments, set a default output path.
        3. Execute the GCTA command.
    
        :param kwargs: Keyword arguments for the GCTA command.
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas._gcta(bfile='data', make_grm='')
        """
        call = f'{self.gcta} ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                    for k,v in kwargs.items()])
        if 'out' not in kwargs.items():
            call += f' --out {self.path}temp/temp'
        bash(call, print_call=False)
        return

    @staticmethod
    def plink_sex_encoding(s: str, male_code: str = 'M', female_code: str = 'F'):
        """
        Encode the sex column in PLINK format.
    
        This function encodes the sex column in the PLINK format, where males are encoded as 1, females as 2, and unknown as 0.
    
        :param s: Sex value to encode.
        :type s: str
        :param male_code: Code representing male sex.
        :type male_code: str, optional
        :param female_code: Code representing female sex.
        :type female_code: str, optional
        :return: Encoded sex value.
        :rtype: int
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> encoded_sex = gwas.plink_sex_encoding('M')
        >>> print(encoded_sex)
        1
        """
        if s == male_code: return 1
        if s == female_code: return 2
        return 0
        
    def bashLog(self, call: str, func, print_call: bool = True):
        """
        Run a shell command and log the output.
    
        This function executes a specified shell command, logs the output, and appends the result to the log.
    
        Steps:
        1. Execute the shell command.
        2. Append the function name, command, and output to the log.
    
        :param call: The shell command to execute.
        :type call: str
        :param func: Name of the function calling this method.
        :type func: str
        :param print_call: Whether to print the command before executing.
        :type print_call: bool, optional
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.bashLog('ls -l', 'list_files')
        """
        self.append2log(func, call , bash(re.sub(r' +', ' ', call), print_call = print_call))
        
        
    def append2log(self, func, call: str, out: str):
        """
        Append a function call and its output to the log.
    
        This function logs the function name, command, and output. If an error is detected in the output, it logs to a separate error log.
    
        Steps:
        1. Append the function name, command, and output to the log dataframe.
        2. Write the output to a log file.
        3. If an error is detected in the output, log to a separate error log and notify the user.
    
        :param func: Name of the function calling this method.
        :type func: str
        :param call: The shell command that was executed.
        :type call: str
        :param out: Output from the shell command.
        :type out: list
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.append2log('list_files', 'ls -l', ['file1', 'file2'])
        """
        self.log.loc[len(self.log)] = [func, call, out]
        logval = '\n'.join(out).lower()
        loc = 'err' if (('error' in logval) ) else ''
        with open(f'{self.path}log{loc}/{func}.log', 'w') as f:
                f.write('\n'.join(out))
        if loc == 'err':
            printwithlog(f'found possible error in log, check the file {self.path}log{loc}/{func}.log')
            
    def make_dir_structure(self, folders: list = ['data', 'genotypes', 'grm', 'log', 'logerr', 'images/genotypes',
                                            'results', 'temp', 'data/pheno', 'results/heritability', 'results/preprocessing',
                                             'results/gwas',  'results/loco', 'results/qtls','results/eqtl','results/sqtl',
                                                  'results/phewas', 'temp/r2', 'results/lz/', 'images/', 'images/scattermatrix/', 
                                                  'images/manhattan/', 'images/genotypes/heatmaps', 'images/genotypes/lds',
                                                 'images/genotypes/dist2founders', 'images/genotypes/umap']):
        """
        Create the directory structure for the project.
    
        This function creates the necessary directory structure for the GWAS project.
    
        Steps:
        1. Iterate over the list of folders.
        2. Create each folder if it does not already exist.
    
        :param folders: List of folders to create.
        :type folders: list, optional
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.make_dir_structure()
        """
        for folder in folders:
            os.makedirs(f'{self.path}{folder}', exist_ok = True)

    def SubsetAndFilter(self, rfids: list = [] ,thresh_m: float = 0.1, thresh_hwe: float = 1e-10, thresh_maf: float = 0.005, verbose: bool = True,
                       filter_based_on_subset: bool = True, makefigures: bool = False, save=True):
        """
        Subset and filter genotype data based on specified thresholds.
    
        This function subsets the genotype data based on the provided RFIDs and filters the SNPs based on missingness,
        Hardy-Weinberg equilibrium (HWE), and minor allele frequency (MAF) thresholds. Optionally, it generates figures
        to visualize the genotype data.
    
        Steps:
        1. Set file paths for sample and SNP data.
        2. Subset the sample data based on RFIDs and sex.
        3. Calculate missingness, HWE, and MAF for autosomes, X, and Y chromosomes.
        4. Filter SNPs based on the specified thresholds.
        5. Save the filtered SNPs and quality metrics.
        6. Optionally, generate figures to visualize the genotype data.
    
        :param rfids: List of RFIDs to subset the genotype data.
        :type rfids: list, optional
        :param thresh_m: Missingness threshold for SNPs.
        :type thresh_m: float, optional
        :param thresh_hwe: Hardy-Weinberg equilibrium threshold for SNPs.
        :type thresh_hwe: float, optional
        :param thresh_maf: Minor allele frequency threshold for SNPs.
        :type thresh_maf: float, optional
        :param verbose: Whether to print detailed information during processing.
        :type verbose: bool, optional
        :param filter_based_on_subset: Whether to filter based on the subset of samples.
        :type filter_based_on_subset: bool, optional
        :param makefigures: Whether to generate figures to visualize the genotype data.
        :type makefigures: bool, optional
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.SubsetAndFilter(rfids=['RFID1', 'RFID2'], thresh_m=0.05, thresh_hwe=1e-6, thresh_maf=0.01, verbose=True, filter_based_on_subset=True, makefigures=True)
        """    
        self.sample_path = f'{self.path}genotypes/keep_rfids.txt'
        self.sample_path_males = f'{self.path}genotypes/keep_rfids_males.txt'
        self.sample_path_females = f'{self.path}genotypes/keep_rfids_females.txt'
        accepted_snps_path = f'{self.path}genotypes/accepted_snps.txt'
        os.makedirs(f'{self.path}genotypes', exist_ok = True)
        self.genotypes_subset = f'{self.path}genotypes/genotypes'
        fullgeno = self.all_genotypes

        famf = pd.read_csv(fullgeno+'.fam', header = None, sep = r'\s+', dtype = str)[[1, 4]].set_axis(['iid', 'gender'], axis = 1)
        gen_iids = famf['iid'].to_list() if not rfids else rfids
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
        plink(bfile = fullgeno, chr = f'1-{self.n_autosome} MT', hardy = '', keep = self.sample_path, thread_num =  self.threadnum, 
              freq = '', missing = '', nonfounders = '', keep_allele_order ='' ,out = f'{self.path}genotypes/autosomes', 
              chr_set = f'{self.n_autosome} no-xy') #autosome_num = 20
        printwithlog('calculating missing hwe maf for X')
        plink(bfile = fullgeno, chr = 'X', hardy = '', keep = self.sample_path, thread_num =  self.threadnum,
              freq = '' , missing = '', nonfounders = '', keep_allele_order ='', out = f'{self.path}genotypes/xfilter',
              filter_females = '', chr_set = f'{self.n_autosome} no-xy')
        printwithlog('calculating missing hwe maf for Y')
        plink(bfile = fullgeno, chr = 'Y', hardy = '', keep = self.sample_path, thread_num =  self.threadnum,
              freq = '' , missing = '', nonfounders = '', keep_allele_order ='', out = f'{self.path}genotypes/yfilter', 
              filter_males = '', chr_set = f'{self.n_autosome} no-xy')
        full = []
        for x in tqdm(['autosomes', 'xfilter', 'yfilter']):
            full_sm = []
            if os.path.isfile(f'{self.path}genotypes/{x}.lmiss'): 
                full_sm += [pd.read_csv(f'{self.path}genotypes/{x}.lmiss', sep = r'\s+')[['CHR','SNP', 'F_MISS']].set_index('SNP')]
            if os.path.isfile(f'{self.path}genotypes/{x}.hwe'): 
                full_sm += [pd.read_csv(f'{self.path}genotypes/{x}.hwe', sep = r'\s+')[['SNP', 'GENO' ,'P']].set_index('SNP').set_axis(['GENOTYPES','HWE'], axis = 1)]
            if os.path.isfile(f'{self.path}genotypes/{x}.frq'): 
                full_sm += [pd.read_csv(f'{self.path}genotypes/{x}.frq', sep = r'\s+')[['SNP', 'MAF', 'A1', 'A2']].set_index('SNP')]
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
        full.index = full.index.str.replace('chr', '')
        if not save: return full
        full.to_parquet(f'{self.path}genotypes/snpquality.parquet.gz', compression='gzip')
        
        with open(f'{self.path}genotypes/parameter_thresholds.txt', 'w') as f: 
            f.write(f'--geno {thresh_m}\n--maf {thresh_maf}\n--hwe {thresh_hwe}')

        if verbose:
            display(full.value_counts(subset= full.columns[full.columns.str.contains('PASS')].to_list())\
                                                   .to_frame().set_axis(['count for all chrs'], axis = 1))
            for i in sorted(full.CHR.unique())[-4:]:
                display(full[full.CHR == i].value_counts(subset=  full.columns[full.columns.str.contains('PASS')].to_list())\
                                                   .to_frame().set_axis([f'count for chr {i}'], axis = 1))

        plink(bfile =  fullgeno, extract = accepted_snps_path, keep = self.sample_path, make_bed = '', thread_num =  self.threadnum,
              set_missing_var_ids = '@:#', keep_allele_order = '',  set_hh_missing = '' , make_founders = '', 
              out = self.genotypes_subset, chr_set = f'{self.n_autosome} no-xy') #
        newbim = pd.read_csv(f'{self.genotypes_subset}.bim', sep='\t', header = None, 
                        names = ['chr', 'snp', 'cm', 'bp', 'a1', 'a2'], 
                        dtype = {'chr': int, 'bp' : int})
        newbim['snp']= newbim['snp'].str.replace('chr', '')
        newbim.to_csv(f'{self.genotypes_subset}.bim', index = False, header = False, sep = '\t')

        if makefigures: self.make_figure_genotypes()

    def make_figure_genotypes(self, genotypes: str = None, founders: str = None):
        """
        Generate figures to visualize genotype data.
    
        This function creates various plots to visualize the genotype data, including heterozygosity per chromosome,
        genotype heatmaps, linkage disequilibrium (LD) clumps, and distance to founders.
    
        Steps:
        1. Read the genotype data from the specified files or use the default subset.
        2. Read the founder genotype data from the specified files or use the default.
        3. Create heterozygosity plots for each chromosome.
        4. Create genotype heatmaps for male and female samples.
        5. Save the heatmaps to the specified directory.
        6. Create LD plots and save them.
        7. If founder data is available, create distance to founders and UMAP plots.
    
        :param genotypes: Path to the genotype files or a tuple containing (bim, fam, gen) data.
        :type genotypes: str or tuple, optional
        :param founders: Path to the founder genotype files or a tuple containing (bim, fam, gen) data.
        :type founders: str or tuple, optional
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.make_figure_genotypes()
        """
        if genotypes is None:
            bim, fam, gen = npplink.load_plink(self.genotypes_subset)
        elif isinstance(genotypes, str):
            bim, fam, gen = npplink.load_plink(genotypes)
        else: bim, fam, gen = genotypes

        if founders is None:
            founders = self.foundersbimfambed
        elif isinstance(founders, str):
            founders = npplink.load_plink(founders)
            
        printwithlog('making plots for heterozygosity per CHR')
        for numc, c in tqdm(list(enumerate(bim.chrom.unique().astype(str)))):
            snps = bim[bim['chrom'] == c]
            if int(c)<= self.n_autosome: snps = snps[::snps.shape[0]//2000+1]
            else: snps = snps[::snps.shape[0]//2000+1]
            f, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(20,20))
            for num, g in enumerate(['M', 'F']):
                fams= fam[fam.gender.astype(str) == str(self.plink_sex_encoding(g))]#[::snps.shape[0]//300]
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
                _distance_to_founders((bim, fam, gen), founders,
                                      f'{self.path}images/genotypes/dist2founders/dist2founder_chr{c}',c , nautosomes = self.n_autosome)
                _make_umap_plot((bim, fam, gen), founders, f'{self.path}images/genotypes/umap/umap_chr{c}',c,
                                nautosomes = self.n_autosome)
            
    def generateGRM(self, autosome_list: list = [], print_call: bool = True, extra_chrs: list = ['X', 'Y', 'MT'], 
                    ldpruned:bool = False, addx2full: bool = False , **kwards) -> int:
        """
        Generate Genetic Relationship Matrices (GRMs) for specified chromosomes.
    
        This function generates GRMs for autosomes and specified extra chromosomes (X, Y, MT) using GCTA. The GRMs for 
        autosomes can be merged into a single GRM.
    
        Steps:
        1. Initialize the autosome list if not provided.
        2. Generate GRM for the X chromosome if specified.
        3. Generate GRM for the Y chromosome if specified.
        4. Generate GRM for the MT chromosome if specified.
        5. Generate GRMs for each autosome in the list.
        6. Save the list of partial GRM filenames.
        7. Merge the partial GRMs into a single GRM.
        8. Optionally, estimate the p-value threshold if it is set to 'auto'.
    
        :param autosome_list: List of autosomes to generate GRMs for.
        :type autosome_list: list, optional
        :param print_call: Whether to print the GCTA commands before executing.
        :type print_call: bool, optional
        :param extra_chrs: List of extra chromosomes to generate GRMs for ('X', 'Y', 'MT').
        :type extra_chrs: list, optional
        :param addx2full: Whether to add the X chromosome GRM to the merged GRM.
        :type addx2full: bool, optional
        :param kwards: Additional keyword arguments for GCTA commands.
        :return: 1 if GRM generation is successful.
        :rtype: int
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.generateGRM(autosome_list=[1, 2, 3], extra_chrs=['X', 'Y'], addx2full=True)
        """
        
        printwithlog('generating GRM...')
        funcName = inspect.getframeinfo(inspect.currentframe()).function

        if not ldpruned: genotypes2use = self.genotypes_subset
        else:
            if not os.path.exists(f'{self.path}pvalthresh/pruned_data.prune.in'):
                prunning_window,prunning_step =  5000000, 1000
                prunning_params = f'{prunning_window} {prunning_step} 0.95'
                plink(bfile=self.genotypes_subset, indep_pairwise = prunning_params, out = f'{self.path}pvalthresh/pruned_data', thread_num = self.threadnum)
                plink(bfile=self.genotypes_subset,  thread_num = self.threadnum, extract = f'{self.path}pvalthresh/pruned_data.prune.in', 
                      make_bed = '', out = f'{self.path}genotypes/prunedgenotypes', set_missing_var_ids = '@:#', keep_allele_order = '',  set_hh_missing = '' , make_founders = '')
            genotypes2use = f'{self.path}genotypes/prunedgenotypes'
        
        if not autosome_list:
            autosome_list = list(range(1,self.n_autosome+1))
            
        all_filenames_partial_grms = pd.DataFrame(columns = ['filename'])

        if 'X' in extra_chrs:
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {genotypes2use} --autosome-num {self.n_autosome} \
                           --make-grm-xchr --out {self.xGRM}',
                        f'{funcName}_chrX', print_call = False)
            if addx2full: all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.xGRM
            
        if 'Y' in extra_chrs:
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {genotypes2use} --keep {self.sample_path_males} --autosome-num {self.n_autosome+4} \
                               --make-grm-bin --chr {self.n_autosome+2} --out {self.yGRM}',
                            f'{funcName}_chrY', print_call = False)
            #all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.yGRM
            
        if 'MT' in extra_chrs:
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {genotypes2use} --autosome-num {self.n_autosome+6} --chr {self.n_autosome+4}\
                           --make-grm-bin --out {self.mtGRM}',
                        f'{funcName}_chrMT', print_call = False)
            all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.mtGRM
            
        for c in tqdm(autosome_list):
            self.bashLog(f'{self.gcta} {self.thrflag} --bfile {genotypes2use} --chr {c} --autosome-num {self.n_autosome}\
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
        """
        Generate a 3D scatter plot of genetic PCA and return the PCA components and eigenvalues.
    
        This function runs PCA on the genetic relationship matrix (GRM) and generates a 3D scatter plot using the first 
        three principal components. It also performs HDBSCAN clustering on the PCA components.
    
        Steps:
        1. Run PCA on the GRM.
        2. Load the eigenvalues and eigenvectors.
        3. Perform HDBSCAN clustering on the PCA components.
        4. Create a 3D scatter plot of the PCA components.
        5. Return the plot and the PCA components with eigenvalues.
    
        :return: A tuple containing the 3D scatter plot and a tuple with PCA components and eigenvalues.
        :rtype: tuple (plotly.graph_objects.Figure, (pandas.DataFrame, pandas.DataFrame))
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> fig, (pcs, eigenvals) = gwas.make_genetic_PCA_fig()
        >>> fig.show()
        """
        self._gcta(grm_bin = f'{self.path}grm/AllchrGRM', pca=20, thread_num = self.threadnum)
        eigenvals = pd.read_csv(f'{self.path}temp/temp.eigenval', header = None ).rename(lambda x: f'gPC{x+1}').set_axis(['eigenvalues'],axis = 1)
        pcs = pd.read_csv(f'{self.path}temp/temp.eigenvec', header = None, sep = r'\s+', index_col=[0,1] ).rename(lambda x: f'gPC{x-1}', axis = 1).droplevel(0)
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
        """
        Generate a 3D scatter plot of genetic PCA and return the PCA components and eigenvalues.
        
        This function runs PCA on the genetic relationship matrix (GRM) and generates a 3D scatter plot using the first 
        three principal components. It also performs HDBSCAN clustering on the PCA components.
        
        Steps:
        1. Run PCA on the GRM.
        2. Load the eigenvalues and eigenvectors.
        3. Perform HDBSCAN clustering on the PCA components.
        4. Create a 3D scatter plot of the PCA components.
        5. Return the plot and the PCA components with eigenvalues.
        
        :return: A tuple containing the 3D scatter plot and a tuple with PCA components and eigenvalues.
        :rtype: tuple (plotly.graph_objects.Figure, (pandas.DataFrame, pandas.DataFrame))
        
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> fig, (pcs, eigenvals) = gwas.make_genetic_PCA_fig()
        >>> fig.show()
        """
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
        """
        Generate scatter matrix plots for specified traits.
    
        This function creates scatter matrix plots for the specified traits. If no traits are specified, it uses all available traits.
        The plots include pairwise scatter plots, histograms, and KDE plots.
    
        Steps:
        1. Determine the list of traits to plot.
        2. Create scatter matrix plots for each unique prefix in the trait names.
        3. Save the plots to the specified directory.
    
        :param traitlist: List of traits to include in the scatter matrix.
        :type traitlist: list, optional
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.scattermatrix(traitlist=['trait1', 'trait2'])
        """
        if not traitlist: traitlist = self.traits
        for i in np.unique([x.replace('regressedlr_', '').split('_')[0] for x in traitlist] ):
            p = sns.PairGrid(self.df, vars=[x.replace('regressedlr_', '') for x in traitlist if i in x], hue="sex")
            p.map_diag(sns.distplot, hist=True) #kde=True, hist_kws={'alpha':0.5})
            p.map_upper(sns.scatterplot)
            p.map_lower(sns.kdeplot, levels=4, color=".2")
            plt.savefig(f'{self.path}images/scattermatrix/prefix{i}.png')
            
    def snpHeritability(self, grm:str = None, print_call: bool = False, save: bool = True, **kwards):
        """
        Calculate SNP heritability for each trait in the project.
    
        This function estimates the heritability of each trait using the Genetic Relationship Matrix (GRM) with GCTA.
        It generates and saves the heritability estimates for each trait.
    
        Steps:
        1. Print the starting message for SNP heritability calculation.
        2. Initialize an empty dataframe for heritability results.
        3. Loop through each trait and calculate heritability using GCTA.
        4. Save the heritability results to a file if specified.
        5. Return the heritability results dataframe.
    
        :param print_call: Whether to print the GCTA commands before executing.
        :type print_call: bool, optional
        :param save: Whether to save the heritability results to a file.
        :type save: bool, optional
        :param kwards: Additional keyword arguments for GCTA commands.
        :return: Dataframe containing heritability results for each trait.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> heritability_df = gwas.snpHeritability(print_call=True, save=True)
        >>> print(heritability_df)
        """
        printwithlog(f'starting snp heritability {self.project_name}')  
        if grm is None: grm = self.autoGRM
        
        h2table = pd.DataFrame()
        for trait in tqdm(self.traits):
            trait_file = f'{self.path}data/pheno/{trait}.txt'
            out_file   = f'{self.path}results/heritability/{trait}' 
            
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file,  index = False, sep = ' ', header = None)
            
            self.bashLog(f'{self.gcta} --reml {self.thrflag}  --autosome-num {self.n_autosome}\
                                       --pheno {trait_file} --grm {grm} --out {out_file}',
                        f'snpHeritability_{trait}', print_call = print_call) #--autosome --mgrm {self.path}grm/listofchrgrms.txt 
            if os.path.isfile(f'{out_file}.hsq'):
                a = pd.read_csv(f'{out_file}.hsq', skipfooter=6, sep = '\t',engine='python')
                b = pd.read_csv(f'{out_file}.hsq', skiprows=6, sep = '\t', header = None, index_col = 0).T.rename({1: trait})
                newrow = pd.concat(
                    [a[['Source','Variance']].T[1:].rename({i:j for i,j in enumerate(a.Source)}, axis = 1).rename({'Variance': trait}),
                    b],axis =1 )
                newrow.loc[trait, 'heritability_SE'] = a.set_index('Source').loc['V(G)/Vp', 'SE']
            else: 
                printwithlog(f'could not find file {out_file}.hsq')
                newrow = pd.DataFrame(np.array(['Fail']*10)[:, None].T, columns =['V(G)','V(e)','Vp','V(G)/Vp','logL0','LRT','df','Pval','n','heritability_SE'], index = [trait])
            
            h2table= pd.concat([h2table,newrow])

        if save: h2table.to_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t')
        return h2table

    def genetic_correlation_matrixv2(self,traitlist: list = [], print_call = False) -> pd.DataFrame:
        """
        Calculate the genetic correlation matrix for a list of traits.
    
        This function calculates the genetic correlation matrix for the specified traits using GCTA. It generates and saves
        the genetic correlation results and creates visualizations.
    
        Steps:
        1. Start the genetic correlation matrix calculation and initialize the Dask client.
        2. Determine the list of traits to process.
        3. Prepare the phenotype data file.
        4. Calculate pairwise genetic correlations using GCTA.
        5. Compile and save the genetic correlation results.
        6. Generate and save visualizations of the genetic correlation matrix.
    
        :param traitlist: List of traits to include in the genetic correlation matrix.
        :type traitlist: list, optional
        :param print_call: Whether to print the GCTA commands before executing.
        :type print_call: bool, optional
        :return: Dataframe containing the genetic correlation matrix.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> correlation_matrix = gwas.genetic_correlation_matrixv2(traitlist=['trait1', 'trait2'], print_call=True)
        >>> print(correlation_matrix)
        """
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
        
        os.makedirs(f'{self.path}results/rG', exist_ok=True)
        
        def get_gcorr_phecorr(trait1, trait2):
            randomid = trait1+trait2
            bash(f'''{self.gcta} --reml-bivar {d_[trait1]} {d_[trait2]} {self.thrflag} \
                --grm {self.autoGRM} --pheno {self.path}data/allpheno.txt --reml-maxit 50 \
                --reml-bivar-lrt-rg 0 --out {self.path}results/rG/gencorr:{trait1}{trait2}''', print_call=False)
            if not os.path.exists(f'{self.path}results/rG/gencorr:{trait1}{trait2}.hsq'):
                rG, rGse, strrG = 0, 100, f"0 +- *"
            else:
                temp = pd.read_csv(f'{self.path}results/rG/gencorr:{trait1}{trait2}.hsq', sep = '\t',engine='python',
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
        #os.system(f'rm -r {self.path}results/rG')
        
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
        
        hieg = linkage(distance.pdist(outg.map(lambda x: float(x.split(' +- ')[0])))) #method='average'
        lk = leaves_list(hieg)
        outg, outp = outg.iloc[lk, lk], outp.iloc[lk, lk]
        
        outg.index, outp.index = outg.index.values, outp.index.values
        outg.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv')
        outmixed = outg.mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('').T  +  outp.mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('').T
        
        ### add heritability
        if not os.path.isfile(self.heritability_path): self.snpHeritability()
        H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col= 0).map(lambda x: 0 if x =='Fail' else x).astype(float)
        H2['her_str'] = H2['V(G)/Vp'].round(3).astype(str) + ' +- ' + H2.heritability_SE.round(3).astype(str)
        for i in outmixed.columns: outmixed.loc[i,i] =  H2.loc['regressedlr_'+i, 'her_str']
        outmixed.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix.csv')
        
        ## make figure
        try: a = sns.clustermap(outmixed.map(lambda x: float(x.split(' +- ')[0])),  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                annot=outmixed.map(lambda x: '' if '*' not in x else '*'), vmin =-1, vmax =1, center = 0 , fmt = '', linewidth = .3, figsize=(25, 25) )
        except: a = sns.clustermap(outmixed.applymap(lambda x: float(x.split(' +- ')[0])),  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                annot=outmixed.applymap(lambda x: '' if '*' not in x else '*'), vmin =-1, vmax =1, center = 0 , fmt = '', linewidth = .3, figsize=(25, 25) )
        dendrogram(hieg, ax = a.ax_col_dendrogram)
        a.ax_cbar.set_position([.1, .2, .05, 0.5])
        plt.savefig(f'{self.path}images/genetic_correlation_matrix.png', dpi = 400)
        #plt.savefig(f'{self.path}images/genetic_correlation_matrix.eps')
        
        outmixed2 = outmixed.T.sort_index().T.sort_index()
        try:a = sns.clustermap(outmixed2.map(lambda x: float(x.split('+-')[0])).copy().T,  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                        annot=outmixed2.map(lambda x: '' if '*' not in x else '*').copy().T, vmin =-1, vmax =1, center = 0 , fmt = '', linewidth = .3, figsize=(15, 15) )
        except: a = sns.clustermap(outmixed2.applymap(lambda x: float(x.split('+-')[0])).copy().T,  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                        annot=outmixed2.applymap(lambda x: '' if '*' not in x else '*').copy().T, vmin =-1, vmax =1, center = 0 , fmt = '', linewidth = .3, figsize=(15, 15) )
        a.ax_heatmap.plot([0,outmixed2.shape[0]], [0, outmixed2.shape[0]], color = 'black')
        a.ax_cbar.set_position([.1, .2, .05, 0.5])
        plt.savefig(f'{self.path}images/genetic_correlation_matrix_sorted.png', dpi = 400)
        return outmixed
    
    def genetic_correlation_matrix(self,traitlist: list = [], print_call = False, save_fmt = ['png'], skip_present = False) -> pd.DataFrame:
        """
        Calculate the genetic correlation matrix for a list of traits using an older method.
    
        This function calculates the genetic correlation matrix for the specified traits using GCTA. It generates and saves
        the genetic correlation results and creates visualizations.
    
        Steps:
        1. Print the starting message for SNP heritability calculation.
        2. Determine the list of traits to process.
        3. Prepare the phenotype data file.
        4. Calculate pairwise genetic correlations using GCTA.
        5. Compile and save the genetic correlation results.
        6. Generate and save visualizations of the genetic correlation matrix.
    
        :param traitlist: List of traits to include in the genetic correlation matrix.
        :type traitlist: list, optional
        :param print_call: Whether to print the GCTA commands before executing.
        :type print_call: bool, optional
        :param save_fmt: List of formats to save the figures in (e.g., ['png', 'pdf']).
        :type save_fmt: list, optional
        :return: Dataframe containing the genetic correlation matrix.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> correlation_matrix = gwas.genetic_correlation_matrix(traitlist=['trait1', 'trait2'], print_call=True, save_fmt=['png', 'pdf'])
        >>> print(correlation_matrix)
        """
        printwithlog(f'starting genetic correlation matrix {self.project_name}...')
        if not traitlist: traitlist = self.traits
        d_ = {t: str(num) for num, t in enumerate(['rfid']+ traitlist)} 
        self.df[['rfid', 'rfid']+ traitlist].fillna('NA').to_csv(f'{self.path}data/allpheno.txt', sep = '\t', header = None, index = False)
        outg = pd.DataFrame()
        outp = pd.DataFrame()
        genetic_table = pd.DataFrame()
        os.makedirs(f'{self.path}results/rG', exist_ok=True)
        if len(traitlist)<2:
            printwithlog('less than 2 traits, there is no genetic correlation to be calculated')
            return
        for trait1, trait2 in tqdm(list(itertools.combinations(traitlist, 2))):
            if not ((os.path.exists(f'{self.path}results/rG/gencorr:{trait1}{trait2}.hsq') or os.path.exists(f'{self.path}logerr/gcorr{trait1}{trait2}.log') ) and skip_present):
                self.bashLog(f'''{self.gcta} --reml-bivar {d_[trait1]} {d_[trait2]} {self.thrflag} \
                    --grm {self.autoGRM} --pheno {self.path}data/allpheno.txt --reml-maxit 50 \
                    --reml-bivar-lrt-rg 0 --out {self.path}results/rG/gencorr:{trait1}{trait2}''', f'gcorr{trait1}{trait2}', print_call=False)
            if os.path.exists(f'{self.path}results/rG/gencorr:{trait1}{trait2}.hsq'):
                temp = pd.read_csv(f'{self.path}results/rG/gencorr:{trait1}{trait2}.hsq', sep = '\t',engine='python', 
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
                phecorr = str(self.df[[trait1, trait2]].corr().iloc[0,1])
                genetic_table.loc[len(genetic_table), ['trait1', 'trait2','phenotypic_correlation','genetic_correlation', 'rG_SE', 'pval']] = \
                                                      [trait1, trait2, phecorr, 0, 1000, 1]
                outg.loc[trait1, trait2] = f"0 +- *"
                outg.loc[trait2, trait1] = f"0 +- *"
                #bash(f'rm {self.path}logerr/genetic_correlation.log',print_call = False)
            
            outp.loc[trait2, trait1] = phecorr.replace('nan', '0')+ ' +- ' + ( '*' if 'nan' in phecorr else '0')
            outp.loc[trait1, trait2] = phecorr.replace('nan', '0')+ ' +- ' + ( '*' if 'nan' in phecorr else '0')
        outg = outg.fillna('1+-0').rename(lambda x: x.replace('regressedlr_', '')).rename(lambda x: x.replace('regressedlr_', ''), axis = 1)
        outp = outp.fillna('1+-0').rename(lambda x: x.replace('regressedlr_', '')).rename(lambda x: x.replace('regressedlr_', ''), axis = 1)
        try: hieg = linkage(distance.pdist(outg.map(lambda x: float(x.split('+-')[0])).T)) #method='average'
        except: hieg = linkage(distance.pdist(outg.applymap(lambda x: float(x.split('+-')[0])).T))
        lk = leaves_list(hieg)
        outg = outg.loc[[x.replace('regressedlr_', '') for x in traitlist], [x.replace('regressedlr_', '') for x in traitlist]].iloc[lk, lk]
        outp = outp.loc[[x.replace('regressedlr_', '') for x in traitlist], [x.replace('regressedlr_', '') for x in traitlist]].iloc[lk, lk] 
        try:hie = linkage(distance.pdist(outg.map(lambda x: float(x.split('+-')[0])).T)) #, method='euclidean'
        except:hie = linkage(distance.pdist(outg.applymap(lambda x: float(x.split('+-')[0])).T)) 
        genetic_table.to_csv(f'{self.path}results/heritability/genetic_correlation_melted_table.csv')
        outg.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv')
        if not os.path.isfile(self.heritability_path): self.snpHeritability()
        try:H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col=0).map(lambda x: 0 if x =='Fail' else x).astype(float)
        except: H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col=0).applymap(lambda x: 0 if x =='Fail' else x).astype(float)
        H2['her_str'] = H2['V(G)/Vp'].round(3).astype(str) + ' +- ' + H2.heritability_SE.round(3).astype(str)

        for version in ['clustering', 'sorted', 'original']:
            if version == 'clustering':
                outmixed = outg.mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('')  +  outp.mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('')
                try:hie = linkage(distance.pdist(outg.map(lambda x: float(x.split('+-')[0])).T))
                except: hie = linkage(distance.pdist(outg.applymap(lambda x: float(x.split('+-')[0])).T))
            elif version in ['original', 'sorted']:
                sct = [x.replace('regressedlr_', '') for x in traitlist] if version == 'original' else sorted(outmixed.columns)
                outmixed = outg.loc[sct,sct].mask(np.triu(np.ones_like(outg, dtype=bool))).fillna('')  +  outp.loc[sct,sct].mask(np.tril(np.ones_like(outg, dtype=bool), -1)).fillna('')
                try: hie = linkage(distance.pdist(outg.loc[sct,sct].map(lambda x: float(x.split('+-')[0])).T))
                except: hie = linkage(distance.pdist(outg.applymap(lambda x: float(x.split('+-')[0])).T))
            for i in outmixed.columns: outmixed.loc[i,i] =  H2.loc['regressedlr_'+i, 'her_str']
            if version == 'clustering': 
                outmixed.to_csv(f'{self.path}results/heritability/genetic_correlation_matrix.csv')
            try:a = sns.clustermap(outmixed.map(lambda x: float(x.split('+-')[0])).copy().T,  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                    annot=outmixed.map(lambda x: '' if '*' not in x else '*').copy().T, vmin =-1, vmax =1, center = 0 , fmt = '', linewidth = .3, figsize=(15, 15) )
            except: a = sns.clustermap(outmixed.applymap(lambda x: float(x.split('+-')[0])).copy().T,  cmap="RdBu_r", col_cluster= False, row_cluster=False,
                    annot=outmixed.applymap(lambda x: '' if '*' not in x else '*').copy().T, vmin =-1, vmax =1, center = 0 , fmt = '', linewidth = .3, figsize=(15, 15) )
            if version == 'clustering': dendrogram(hie, ax = a.ax_col_dendrogram)
            a.ax_cbar.set_position([0, .2, .03, 0.5])
            a.ax_heatmap.plot([0,outmixed.shape[0]], [0, outmixed.shape[0]], color = 'black')
            if 'png' in save_fmt: plt.savefig(f'{self.path}images/genetic_correlation_matrix{"_"+version if version != "clustering" else ""}.png', dpi = 400)
            for sfm in (set(save_fmt) - {'png'}):
                plt.savefig(f'{self.path}images/genetic_correlation_matrix{"_"+version if version != "clustering" else ""}.{sfm}')
            plt.close()
        return outmixed

    def make_genetic_correlation_figure(self, order = 'sorted', traits= [], save = True, include=['gcorr', 'pcorr'], size = 'pval'):
        """
        Generate a genetic correlation figure for the specified traits.
    
        This function generates a visual representation of the genetic correlations among the specified traits. It saves
        the figure to a file and returns the figure object.
    
        Steps:
        1. Read the genetic correlation data.
        2. Filter the data for the specified traits.
        3. Calculate the size of the points based on the specified size parameter.
        4. Sort or cluster the traits as specified.
        5. Create a scatter plot of the genetic and phenotypic correlations.
        6. Save the figure to a file.
        7. Return the figure object.
    
        :param order: The order of the traits in the figure ('sorted' or 'cluster').
        :type order: str, optional
        :param traits: List of traits to include in the figure.
        :type traits: list, optional
        :param save: Whether to save the figure to a file.
        :type save: bool, optional
        :param include: List of correlations to include in the figure ('gcorr' for genetic correlation, 'pcorr' for phenotypic correlation).
        :type include: list, optional
        :param size: The size parameter for the points ('rG_SE' for standard error, 'pval' for p-value).
        :type size: str, optional
        :return: The generated figure object.
        :rtype: holoviews.Overlay
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> fig = gwas.make_genetic_correlation_figure(order='sorted', traits=['trait1', 'trait2'], save=True, include=['gcorr', 'pcorr'], size='pval')
        >>> hv.save(fig, 'genetic_correlation_figure.html')
        """
        if not len(traits): traits = self.traits
        traits = [x.replace('regressedlr_', '') for x in traits]
        if len(traits)<2:
            printwithlog('less than 2 traits, there is no genetic correlation figure to be made')
            return hv.Curve([]).opts(title="no genetic Correlation",  width=400, height=200)
        
        gcorr = pd.read_csv(f'{self.path}results/heritability/genetic_correlation_melted_table.csv', index_col = 0)
        try:gcorr[['trait1','trait2']] =gcorr[['trait1','trait2']].map(lambda x:x.replace('regressedlr_', ''))
        except: gcorr[['trait1','trait2']] = gcorr[['trait1','trait2']].applymap(lambda x:x.replace('regressedlr_', ''))
        gcorr = gcorr[gcorr.trait1.isin(traits) & gcorr.trait2.isin(traits)]
        if size == 'rG_SE': 
            gcorr['size'] = (1.1 - gcorr.rG_SE.map(lambda x: min(x, 1)).fillna(1.))*20*30/len(traits)
        elif size == 'pval':
            values = 1-gcorr[['pval']].fillna(1).clip(1e-5, .8)
            gcorr['size'] = values * (570/len(traits) - 1) + 1
            # values = -np.log10(gcorr[['pval']].replace(0, 1e-6).fillna(1))
            # values = np.log(np.clip(values, a_min=1.1, a_max= 6.5))
            # gcorr['size'] = values/1.8 * (570/len(traits) - 1) + 1
            # gcorr['size'] = MinMaxScaler(feature_range = (1, 570/len(traits)))\
            #                 .fit_transform(np.log(np.clip(-np.log10(gcorr[['pval']].replace(0, 1e-6).fillna(1)), a_min=.1, a_max= 100)))
                            
        alltraits = list(sorted(set(gcorr.trait1) | set(gcorr.trait2)))
        if order == 'sorted': torder = {t:n for n,t in enumerate(alltraits)}
        else: 
            alltraits, torder = traits,{t:n for n,t in enumerate(traits)}
        gcorr = pd.concat([gcorr, gcorr.rename({'trait1' : 'trait2', 'trait2': 'trait1'}, axis = 1)])
        gcorr = gcorr.assign(or1 = gcorr.trait1.map(torder),or2 = gcorr.trait2.map(torder)).sort_values(['trait1', 'trait2'])
        try:H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col=0).map(lambda x: 0 if x =='Fail' else x).astype(float).rename({'V(G)/Vp': 'g'}, axis = 1).reset_index(names = ['trait'])
        except:H2 = pd.read_csv(self.heritability_path, sep = '\t', index_col=0).applymap(lambda x: 0 if x =='Fail' else x).astype(float).rename({'V(G)/Vp': 'g'}, axis = 1).reset_index(names = ['trait'])
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
            #print('pass')
        
        kdims1=hv.Dimension('trait1', values=alltraits)
        kdims2=hv.Dimension('trait2', values=alltraits)
        fig_opts = {'frame_width': 900, 'frame_height': 900, 'tools':['hover'], 'padding': 0.05, 'cmap':'RdBu', 'colorbar':True, 'clim': (-1, 1), 'line_color':'black', 'line_width':1}
        fig = hv.Points(gcorr.query('(or2<or1) and (rG_SE<1)'), kdims = [kdims1, kdims2],vdims=['phenotypic_correlation','genetic_correlation','rG_SE', 'size']) \
                                                  .opts( color='genetic_correlation', size=hv.dim('size')*1.5 if ('gcorr' in include) else 0, **fig_opts) #
        fig = fig*hv.Points(gcorr.query('or2> or1'), kdims = [kdims1, kdims2],vdims=['phenotypic_correlation','genetic_correlation','rG_SE', 'size']) \
                                                  .opts( color='phenotypic_correlation', size=18*30/len(traits)*1.5 if ('pcorr' in include) else 0, marker = 'square',
                                                        **fig_opts) #
        fig = fig*hv.Labels(H2.assign(gtex = H2.g.map(lambda x: f"{int(x*100)}%")), kdims = ['trait', 'trait'],vdims=['gtex']).opts(text_font_size=f'{min(int(7*1.5*30/len(traits)), 20)}pt', text_color='black')
        fig = fig.opts(frame_height=900, frame_width=900,title = f'Genetic correlation', xlabel = '', ylabel = '',
                       fontsize={ 'xticks': f'{min(int(7*1.5*30/len(traits)), 20)}pt', 'yticks': f'{min(int(7*1.5*30/len(traits)), 20)}pt'},
                       xrotation=45,invert_yaxis = True, yrotation=45)
        if save:hv.save(fig, f"{self.path}images/genetic_correlation_matrix2{('_'+order).replace('_cluster', '')}.png")
        return fig
        
    
    def make_heritability_figure(self, traitlist: list = [], save:bool = True, add_classes: bool = False):
        """
        Generate a heritability figure for the specified traits.
    
        This function generates a visual representation of the heritability of the specified traits. It saves
        the figure to files in multiple formats and displays it.
    
        Steps:
        1. Check if heritability data exists; if not, calculate SNP heritability.
        2. Read the heritability data and filter for the specified traits.
        3. Optionally, add clustering classes to the traits based on genetic correlation.
        4. Create a scatter plot of the heritability data.
        5. Add reference lines to the plot.
        6. Save the figure to files in the specified formats.
        7. Optionally, display the figure.
    
        :param traitlist: List of traits to include in the figure.
        :type traitlist: list, optional
        :param save_fmt: List of formats to save the figures in (e.g., ['png', 'html', 'pdf', 'eps']).
        :type save_fmt: list, optional
        :param display: Whether to display the figure.
        :type display: bool, optional
        :param add_classes: Whether to add clustering classes to the traits based on genetic correlation.
        :type add_classes: bool, optional
        :return: None
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> gwas.make_heritability_figure(traitlist=['trait1', 'trait2'], save_fmt=['png', 'pdf'],  add_classes=True)
        """
        if not os.path.isfile(f'{self.path}results/heritability/heritability.tsv'): self.snpHeritability()
        her = pd.read_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t', 
                          index_col=0).rename(lambda x: x.replace('regressedlr_', ''))
        her = her[~(her == 'Fail').all(1)].astype(float)
        her = her.rename({'V(G)/Vp': 'heritability'}, axis = 1).sort_values('heritability').dropna(subset = 'heritability')
        traitlist = pd.Series(her.index if not len(traitlist) else traitlist).str.replace('regressedlr_', '')
        her = her.loc[her.index.isin(traitlist)]
        if add_classes and len(traitlist)>1:
            if os.path.isfile(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv'): pass
            else: self.genetic_correlation_matrix()
            try:gcor = pd.read_csv(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv', 
                               index_col=0).map(lambda x: float(x.split('+-')[0]))
            except:gcor = pd.read_csv(f'{self.path}results/heritability/genetic_correlation_matrix_justgenetic.csv', 
                               index_col=0).applymap(lambda x: float(x.split('+-')[0]))
            classes = pd.DataFrame(HDBSCAN(metric = 'precomputed', min_cluster_size = 3).fit_predict(gcor.loc[her.index, her.index]), 
                                   index = her.index, columns = ['cluster']).astype(str)
            her = pd.concat([classes, her], axis = 1).sort_index().sort_values('heritability')
            her.cluster = ('cluster_'+ her.cluster).replace('cluster_-1', 'unassigned')
        
        fontsizes = {'title': 40,  'labels': 25,   'xticks': 10,  'yticks': 20, 'legend': 15 }
        yrange = -.01 , min(1, (her.heritability + her.heritability_SE).max()+.02) #max(-.1, (her.heritability -her.heritability_SE).min()-.02)
        fig = (hv.HLine(0).opts(color = 'black')*\
         reduce(lambda x,y: x*y,  [hv.HLine(x).opts(color = 'black', line_dash = 'dashed', line_width = 2) for x in np.linspace(.1, 1, 10)])*\
         her.hvplot.scatter(y= 'heritability', frame_height = 600, frame_width = 900, color ='cluster' if 'cluster' in her.columns else 'steelblue',
                            line_width = 2, line_color = 'black', size = 400, 
                            alpha = .7, hover_cols =her.columns.to_list())*\
         her.reset_index(names=['trait']).hvplot.errorbars(x='trait', y='heritability', yerr1='heritability_SE', line_width = 2
                                                         )).opts(xrotation = 45, ylabel = r'Heritability', ylim =yrange,fontsize=fontsizes )
        if save:
            hv.save(fig, f"{self.path}images/heritability_sorted.html")
            hv.save(fig, f"{self.path}images/heritability_sorted.png")
        return fig
        
    def BLUP(self,  print_call: bool = False, save: bool = True, frac: float = 1.,traits = [],**kwards):
        """
        Perform Best Linear Unbiased Prediction (BLUP) for specified traits.
    
        This function performs BLUP for the specified traits using the GCTA software. It generates the necessary GRM subset,
        computes BLUP values, and SNP effects, and saves the results.
    
        Steps:
        1. Create necessary directories.
        2. Prepare phenotype and RFID files for each trait.
        3. Generate the GRM subset for the trait.
        4. Compute BLUP values and SNP effects.
        5. Compile and save the BLUP values table.
    
        :param print_call: Whether to print the GCTA commands before executing.
        :type print_call: bool, optional
        :param save: Whether to save the BLUP values table to a file.
        :type save: bool, optional
        :param frac: Fraction of the data to use for training.
        :type frac: float, optional
        :param traits: List of traits to perform BLUP on.
        :type traits: list, optional
        :param **kwards: Additional keyword arguments for the GCTA commands.
        :return: Dataframe containing the BLUP values for each trait.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> blup_table = gwas.BLUP(traits=['trait1', 'trait2'], save=True)
        >>> print(blup_table)
        """
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
        """
        Predict traits using BLUP (Best Linear Unbiased Prediction) model.
    
        This function predicts traits for the specified genotypes using precomputed BLUP SNP effects. It saves the predicted values.
    
        Steps:
        1. Prepare the directories and necessary files.
        2. For each trait, perform BLUP prediction using PLINK.
        3. Compile and save the predicted values.
    
        :param genotypes2predict: Path to the genotype file to predict.
        :type genotypes2predict: str
        :param rfid_subset: List of RFIDs to include in the prediction.
        :type rfid_subset: list, optional
        :param traits: List of traits to predict.
        :type traits: list, optional
        :param print_call: Whether to print the PLINK commands before executing.
        :type print_call: bool, optional
        :param save: Whether to save the predicted values to a file.
        :type save: bool, optional
        :return: Dataframe containing the predicted values for each trait.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> predictions = gwas.BLUP_predict(genotypes2predict='/path/to/genotypes', traits=['trait1', 'trait2'], save=True)
        >>> print(predictions)
        """
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
        outdf = pd.concat([pd.read_csv(f'{pred_path}/{trait}_{genotypes2predictname}.profile', sep = r'\s+')[['IID', 'SCORESUM']].set_axis(['rfid', trait], axis =1).set_index('rfid') 
                   for trait in traitlist], axis = 1)
        if save: outdf.to_csv(f'{self.path}results/BLUP/BLUP_predictions.tsv', sep = '\t')

        return outdf
        

    def fastGWASold(self, traitlist: list = [], chrlist: list = [], skip_already_present = False, print_call: bool = False, **kwards):
        """
        Perform fast GWAS (Genome-Wide Association Study) for specified traits and chromosomes.
    
        This function performs a fast GWAS for the specified traits and chromosomes using the GCTA software. It supports
        parallel processing using Dask.
    
        Steps:
        1. Initialize the Dask client for parallel processing.
        2. Prepare the list of traits and chromosomes to analyze.
        3. Perform GWAS for each trait and chromosome combination.
        4. Optionally, skip already processed trait-chromosome combinations.
        5. Monitor and manage the Dask tasks.
    
        :param traitlist: List of traits to analyze. If empty, all traits in the class will be used.
        :type traitlist: list, optional
        :param chrlist: List of chromosomes to analyze. If empty, all autosomes and extra chromosomes will be used.
        :type chrlist: list, optional
        :param skip_already_present: Whether to skip analysis for trait-chromosome combinations that have already been processed.
        :type skip_already_present: bool, optional
        :param print_call: Whether to print the GCTA commands before executing.
        :type print_call: bool, optional
        :param **kwards: Additional keyword arguments for the GCTA commands.
        :return: 1 when the GWAS analysis is completed.
        :rtype: int
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> result = gwas.fastGWAS(traitlist=['trait1', 'trait2'], chrlist=[1, 2, 'X'], skip_already_present=True)
        >>> print(result)
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
        #print(f'estimated time assuming 10min per chr {round(looppd.shape[0]*600/3600/float(self.threadnum), 5)}H')
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

    def fastGWAS(self, traitlist: list = [], chrlist: list = [], skip_already_present = False, print_call: bool = False, **kwards):
        """
        Perform fast GWAS (Genome-Wide Association Study) for specified traits and chromosomes.
    
        This function performs a fast GWAS for the specified traits and chromosomes using the GCTA software. It supports
        parallel processing using Dask.
    
        Steps:
        1. Initialize the Dask client for parallel processing.
        2. Prepare the list of traits and chromosomes to analyze.
        3. Perform GWAS for each trait and chromosome combination.
        4. Optionally, skip already processed trait-chromosome combinations.
        5. Monitor and manage the Dask tasks.
    
        :param traitlist: List of traits to analyze. If empty, all traits in the class will be used.
        :type traitlist: list, optional
        :param chrlist: List of chromosomes to analyze. If empty, all autosomes and extra chromosomes will be used.
        :type chrlist: list, optional
        :param skip_already_present: Whether to skip analysis for trait-chromosome combinations that have already been processed.
        :type skip_already_present: bool, optional
        :param print_call: Whether to print the GCTA commands before executing.
        :type print_call: bool, optional
        :param **kwards: Additional keyword arguments for the GCTA commands.
        :return: 1 when the GWAS analysis is completed.
        :rtype: int
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> result = gwas.fastGWAS(traitlist=['trait1', 'trait2'], chrlist=[1, 2, 'X'], skip_already_present=True)
        >>> print(result)
        """
        if not client._get_global_client():  cline = Client( processes = False)
        else: cline = client._get_global_client()
        if len(traitlist) == 0: traitlist = self.traits
        printwithlog(f'starting fastGWAS for {len(traitlist)} traits...')
        chrsingrm = pd.read_csv(f'{self.path}grm/listofchrgrms.txt', header = None)[0]\
                      .map(lambda x: basename(x).replace('chrGRM', '')).to_list()
        ranges = chrlist if len(chrlist) else [i for i in range(1,self.n_autosome+5) if i != self.n_autosome+3]
        path, gcta, nauto, genotypes   = self.path, self.gcta , self.n_autosome, self.genotypes_subset
        num2xymt =  lambda x: str(int(float(x))).replace(str(nauto+1), 'x')\
                                                .replace(str(nauto+2), 'y')\
                                                .replace(str(nauto+4), 'mt')
        check_present_extra = lambda trait, chromp2: os.path.exists(f'{path}results/gwas/{trait}_chrgwas{chromp2}.mlma')
        def _smgwas(trait, chrom):
            chromp2 = num2xymt(chrom)
            if check_present_extra(trait, chromp2) and skip_already_present:
                    printwithlog(f'''skipping gwas for trait: {trait} and chr {chromp2}, 
                          output files already present, to change this behavior use skip_already_present = False''')
            else:
                subgrmflag = f'--mlma-subtract-grm {path}grm/{chromp2}chrGRM' if chromp2 in chrsingrm else ''
                bash(f'{gcta} --thread-num 1 --pheno {path}data/pheno/{trait}.txt --bfile {genotypes} \
                                           --grm {path}grm/AllchrGRM --autosome-num {nauto} \
                                           --chr {chrom} {subgrmflag} --mlma \
                                           --out {path}results/gwas/{trait}_chrgwas{chromp2}', 
                            print_call = print_call, shell = False)
            return True
        traits_, chroms_ = zip(*itertools.product(traitlist, ranges))
        futures = cline.map(_smgwas, traits_, chroms_)
        progress(futures, notebook=False, interval="300s")
        results = cline.gather(futures)
        del futures, results
        return 1
    
    def addGWASresultsToDb(self, researcher: str , round_version: str, gwas_version: str = None,filenames: list = [],
                           pval_thresh: float = 1e-4, safe_rebuild: bool = True,**kwards) -> int:
        """
        Add GWAS results to the database.
    
        This function adds GWAS results to the database by reading the GWAS result files, filtering them based on the specified p-value threshold, and then merging them into the existing database.
    
        Steps:
        1. Read GWAS result files.
        2. Filter results based on the p-value threshold.
        3. Assign metadata to the results.
        4. Concatenate new results with the existing database.
        5. Save the updated database.
    
        :param researcher: Name of the researcher conducting the GWAS.
        :type researcher: str
        :param round_version: Version of the round of GWAS analysis.
        :type round_version: str
        :param gwas_version: Version of the GWAS analysis.
        :type gwas_version: str
        :param filenames: List of filenames containing GWAS results. If empty, all .mlma files in the results directory will be used.
        :type filenames: list, optional
        :param pval_thresh: P-value threshold for filtering significant results.
        :type pval_thresh: float, optional
        :param safe_rebuild: Whether to safely rebuild the database if the existing database cannot be opened.
        :type safe_rebuild: bool, optional
        :param **kwards: Additional metadata to be added to the results.
        :return: 1 if the process is successful.
        :rtype: int
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> result = gwas.addGWASresultsToDb(researcher='John Doe', round_version='v1', gwas_version='v2', pval_thresh=1e-5)
        >>> print(result)
        """
        if gwas_version is None: gwas_version = __version__
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
        if 'https://' in phedb: 
            phedb = f'{self.path}phewasdb.parquet.gz'
            printwithlog(f'the phewas database is online, we will save the phewas of this project in the file: {phedb}')
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
        
        try:alldata.drop_duplicates(subset = ['researcher', 'project', 'round_version', 'trait', 'SNP', 'uploadeddate'], 
                                keep='first').to_parquet(phedb, index = False, compression='gzip')
        except:
            printwithlog(f'could not add the gwas mlmas to the phewas database... starting a new one at {self.path}phewasdb.parquet.gz')
            alldata.drop_duplicates(subset = ['researcher', 'project', 'round_version', 'trait', 'SNP', 'uploadeddate'], 
                                keep='first').to_parquet(f'{self.path}phewasdb.parquet.gz', index = False, compression='gzip')
        
        return 1

    def prune_genotypes(self):
        """
        Prune genotypes to remove duplicate SNPs.
    
        This function reads genotype data and identifies duplicate SNPs, then prunes the duplicates and saves the pruned data.
    
        Steps:
        1. Read genotype data.
        2. Identify duplicate SNPs.
        3. Save the results to specified paths.
        4. Prune the duplicate SNPs using PLINK.
        5. Return the number of pruned SNPs and the total number of SNPs.
    
        :return: List containing the number of pruned SNPs and the total number of SNPs.
        :rtype: list
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> pruned_snps = gwas.prune_genotypes()
        >>> print(pruned_snps)
        """
        printwithlog('starting genotype prunner...')
        os.makedirs(f'{self.path}pvalthresh', exist_ok = True)
        os.makedirs(f'{self.path}pvalthresh/gwas/', exist_ok = True)
        os.makedirs(f'{self.path}pvalthresh/randomtrait/', exist_ok = True)
        snps,_,gens = npplink.load_plink(self.genotypes_subset)
        gens = da.nan_to_num(gens, -1).astype(np.int8).T
        def prune_dups(array):
            dic = defaultdict(list, {})
            for num, i in enumerate(array): dic[i.tobytes()] += [num]
            return dic
        printwithlog('starting genotype dups finder...')    
        pruned = prune_dups(gens.compute())
        first_snps = [snps.loc[v[0], 'snp'] for k,v in pruned.items()]
        printwithlog(f'saving results to:\n1){self.path}pvalthresh/genomaping.parquet.gz\n2){self.path}pvalthresh/prunned_dup_snps.in\n3){self.path}genotypes/prunedgenotypes')  
        prunedset = pd.DataFrame([[k,'|'.join(map(str, v))] for k,v in pruned.items()], columns = ['genotypes', 'snps'])
        prunedset.to_parquet(f'{self.path}pvalthresh/genomaping.parquet.gz', compression = 'gzip')
        pd.DataFrame(first_snps).to_csv(f'{self.path}pvalthresh/prunned_dup_snps.in', index = False, header = None)
        plink(bfile=self.genotypes_subset,  thread_num = self.threadnum, extract = f'{self.path}pvalthresh/prunned_dup_snps.in',
              make_bed = '', out = f'{self.path}genotypes/prunedgenotypes', set_missing_var_ids = '@:#', keep_allele_order = '',  set_hh_missing = '' , make_founders = '')
        printwithlog(f'prunned data has {format(prunedset.shape[0], ",")} out of the original {format(gens.shape[0], ",")}')
        return [prunedset.shape[0], gens.shape[0]]
        
    def estimate_pval_threshold(self, replicates: int = 1000, sample_size = 'all', exact_prunner: bool = True ,prunning_window: int = 50000, 
                                prunning_step: int = 50, remove_after: bool = False, max_nsnps = 300000):
        """
        Estimate the p-value threshold for genome-wide association studies (GWAS).
    
        This function estimates the p-value threshold by performing GWAS on randomly generated traits and calculating the distribution of maximum p-values.
    
        Steps:
        1. Prune genotypes to reduce the number of SNPs.
        2. Generate random traits and perform GWAS on them.
        3. Calculate the maximum p-value for each replicate.
        4. Estimate the p-value threshold based on the distribution of maximum p-values.
    
        :param replicates: Number of replicates to perform for estimating the p-value threshold.
        :type replicates: int, optional
        :param sample_size: Sample size to use for each replicate. Can be an integer or a float between 0 and 1.
        :type sample_size: int or float, optional
        :param exact_prunner: Whether to use exact pruning of genotypes.
        :type exact_prunner: bool, optional
        :param prunning_window: Window size for LD pruning.
        :type prunning_window: int, optional
        :param prunning_step: Step size for LD pruning.
        :type prunning_step: int, optional
        :param remove_after: Whether to remove the GWAS results after calculating the p-value threshold.
        :type remove_after: bool, optional
        :return: DataFrame containing the estimated p-value thresholds.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> thresholds = gwas.estimate_pval_threshold()
        >>> print(thresholds)
        """
        printwithlog('starting P-value threshold calculation...')
        if sample_size == 'all': sample_size = len(self.df)
        if sample_size < 1: sample_size = round(len(self.df)*sample_size)
        cline = Client( processes = False) if not client._get_global_client() else client._get_global_client()
        os.makedirs(f'{self.path}pvalthresh', exist_ok = True)
        os.makedirs(f'{self.path}pvalthresh/gwas/', exist_ok = True)
        os.makedirs(f'{self.path}pvalthresh/randomtrait/', exist_ok = True)
        
        if exact_prunner: 
            npruned_snps = self.prune_genotypes()
            if npruned_snps[0] > max_nsnps:
                printwithlog(f'resulting prune has more than {max_nsnps} many snps, using ld prune for further prunning...')
                exact_prunner = False
        if not exact_prunner:
            prunning_params = f'{prunning_window} {prunning_step} 0.999'
            printwithlog(f'prunning gentoypes using {prunning_params}')
            if not os.path.exists(f'{self.path}pvalthresh/pruned_data.prune.in'):
                plink(bfile=self.genotypes_subset, indep_pairwise = prunning_params, out = f'{self.path}pvalthresh/pruned_data', thread_num = self.threadnum)
                plink(bfile=self.genotypes_subset,  thread_num = self.threadnum, extract = f'{self.path}pvalthresh/pruned_data.prune.in', 
                      make_bed = '', out = f'{self.path}genotypes/prunedgenotypes', set_missing_var_ids = '@:#', keep_allele_order = '',  set_hh_missing = '' , make_founders = '' )
            npruned_snps = [pd.read_csv(f'{self.path}pvalthresh/pruned_data.prune.{i}', header = None).shape[0] for i in ['in', 'out']]
            display(f'prunned data has {npruned_snps[0]} out of the original {npruned_snps[1]}' )
        path, gcta, nauto, genotypes, testdf, clist   = self.path, self.gcta , self.n_autosome, self.genotypes_subset, self.df.copy(), self.chrList()
        xymt2num =  lambda x: int(float(str(x).lower().replace('chr', '').replace('x', str(nauto+1))\
                                                 .replace('y', str(nauto+2))\
                                                 .replace('mt', str(nauto+4))\
                                                 .replace('m', str(nauto+4))))
        def get_maxp_1sample(ranid, skip_already_present = True, remove_after = True ):
            os.makedirs(f'{path}pvalthresh/gwas/{ranid}', exist_ok = True)
            r = np.random.RandomState(ranid)
            valuelis = r.normal(size = testdf.shape[0])
            valuelis *= r.choice([1, np.nan],size = testdf.shape[0] , 
                                 p = [sample_size/testdf.shape[0], 1-sample_size/testdf.shape[0]])
            testdf[['rfid', 'rfid']].assign(trait = valuelis).fillna('NA').astype(str).to_csv(f'{path}pvalthresh/randomtrait/{ranid}.txt',  index = False, sep = ' ', header = None)
            maxp = 0
            for c in clist:
                chrom = xymt2num(c)
                filename = f'{path}pvalthresh/gwas/{ranid}/chrgwas{c}' 
                if os.path.exists(f'{filename}.mlma') and skip_already_present: pass
                else:
                    subgrmflag = f'--mlma-subtract-grm {path}grm/{c}chrGRM' if c not in ['x','y'] else ''
                    bash(f'{gcta} --thread-num 1 --pheno {path}pvalthresh/randomtrait/{ranid}.txt --bfile {path}genotypes/prunedgenotypes \
                                               --grm {path}grm/AllchrGRM --autosome-num {nauto} \
                                               --chr {chrom} {subgrmflag} --mlma \
                                               --out {filename}', 
                                 print_call = False)#f'GWAS_{chrom}_{ranid}',
                if os.path.exists(f'{filename}.mlma'): chrmaxp = np.log10(pd.read_csv(f'{filename}.mlma', sep = '\t', usecols=['p'], dtype = {'p':float} )['p'].min())
                else: chrmaxp = 0
                if chrmaxp < maxp: maxp = chrmaxp
            if remove_after:  bash(f'rm -r {path}pvalthresh/gwas/{ranid}')
            return maxp
        
        looppd = pd.DataFrame(range(replicates), columns = ['reps'])
        loop   = dd.from_pandas(looppd, npartitions=min(replicates, 200))
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
        
        
    def callQTLs(self, window: int = 3e6, subterm: int = 4,  add_founder_genotypes: bool = True, save = True, displayqtl = True, annotate = True,
                 ldkb = 12000, ldr2:float = .8, qtl_dist = 12e6, NonStrictSearchDir = True, conditional_analysis = True, **kwards): # annotate_genome: str = 'rn7',
        """
        Call quantitative trait loci (QTLs) based on GWAS results.
    
        This function identifies QTLs by analyzing significant SNPs from GWAS results and performs conditional analysis if specified.
    
        :param window: Size of the window around the top SNPs to search for correlated SNPs, in base pairs.
        :type window: int, optional
        :param subterm: Subterm for filtering SNPs based on p-value difference.
        :type subterm: int, optional
        :param add_founder_genotypes: Whether to add founder genotypes to the output.
        :type add_founder_genotypes: bool, optional
        :param save: Whether to save the QTL results to a file.
        :type save: bool, optional
        :param displayqtl: Whether to display the QTL results.
        :type displayqtl: bool, optional
        :param annotate: Whether to annotate the QTL results.
        :type annotate: bool, optional
        :param ldkb: Window size for LD calculation, in kilobases.
        :type ldkb: int, optional
        :param ldr2: R-squared threshold for LD calculation.
        :type ldr2: float, optional
        :param qtl_dist: Distance for filtering QTLs, in base pairs.
        :type qtl_dist: float, optional
        :param NonStrictSearchDir: Whether to use non-strict search directory.
        :type NonStrictSearchDir: bool or pandas.DataFrame, optional
        :param conditional_analysis: Whether to perform conditional analysis.
        :type conditional_analysis: bool or str, optional
        :return: DataFrame containing the identified QTLs.
        :rtype: pandas.DataFrame
    
        Example:
        >>> gwas = gwas_pipe(path='/path/to/project/', project_name='my_gwas_project')
        >>> qtls = gwas.callQTLs(window=1e6, subterm=3)
        >>> print(qtls)
        """
        printwithlog(f'starting call qtl ... {self.project_name}') 
        thresh = 10**(-self.threshold)
        if type(NonStrictSearchDir) == type(pd.DataFrame()):
            topSNPs, save = NonStrictSearchDir, False
        
        elif not NonStrictSearchDir:
            topSNPslist = []
            for t, chrom in tqdm(list(itertools.product(self.traits, range(1,self.n_autosome+4)))):
                    chrom = self.replacenumstoXYMT(chrom)
                    filename = f'{self.path}results/gwas/{t}_chrgwas{chrom}.mlma'
                    if os.path.exists(filename):
                        topSNPslist += [pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)]
                    else: pass #print(f'could not locate {filename}')
            for t in tqdm(self.traits):
                    filename = f'{self.path}results/gwas/{t}.loco.mlma'
                    if os.path.exists(filename):
                        topSNPslist += [pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)]
                    else: pass #printwithlog(f'could not locate {filename}')
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
                    start_qtl, end_qtl = r2temp.BP_B.agg(['min', 'max'])
                    ldSNPS_LEN = (end_qtl - start_qtl)# / 1e6   #r2temp.BP_B.agg(lambda x: (x.max()-x.min())/1e6)
                    df = df.query('~(@idx - @qtl_dist//2 < index < @idx + @qtl_dist//2) and (SNP not in @ldSNPS)')
                except:
                    printwithlog('could not run plink...')
                    ldSNPS = [maxp.SNP]
                    start_qtl, end_qtl, ldSNPS_LEN = 0, 0, 0
                    df = df.query('(SNP not in @ldSNPS)')
                            
                out = pd.concat([out,
                                 maxp.to_frame().T.assign(QTL= qtl, interval_size = bp2str(ldSNPS_LEN), 
                                                          start_qtl = f'{start_qtl:,}', end_qtl = f'{end_qtl:,}')],
                                 axis = 0) #'{:.2f} Mb'.format(ldSNPS_LEN)
                
        if not len(out):
            printwithlog('no SNPS were found, returning an empty dataframe')
            if save: out.to_csv(f'{self.path}results/qtls/allQTLS.csv', index = False)
            return out
            
        out =  out.sort_values('trait').reset_index(names = 'bp')#.assign(project = self.project_name)
        out['trait_description'] = out.trait.apply(lambda x: self.get_trait_descriptions[x])
        out['trait'] = out.trait.apply(lambda x: x.replace('regressedlr_', ''))
        self.allqtlspath = f'{self.path}results/qtls/allQTLS.csv'
        if save: out.to_csv(self.allqtlspath.replace('allQTLS', 'QTLSb4CondAnalysis'), index = False)
        self.pbim, self.pfam, self.pgen = npplink.load_plink(self.genotypes_subset)
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
            dffounders = npplink.plink2df(self.foundersbimfambed, snplist = out.SNP, recodeACGT=True).T.reset_index(names = 'SNP')
            out = out.merge(dffounders, on = 'SNP', how = 'left')
        
        
        out['significance_level'] = out.p.apply(lambda x: '5%' if x >= self.threshold05 else '10%')
        if save: out.to_csv(self.allqtlspath, index = False)
        if annotate: 
            out = self.annotate(out, save = True)
            out = out.reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
            if save: out.to_csv(f'{self.path}results/qtls/finalqtl.csv', index= False)
        return out.set_index('SNP') 

    def add_r2_qtl_boundaries(self, qtls: str = None, snp_col = 'SNP',r2_thresh = .8):
        if qtls is None: qtls = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv', index_col = 0)
        if isinstance(qtls, str): qtls = pd.read_csv(qtls)
        dists = pd.concat(qtls[snp_col].map(lambda x: self.plink(bfile = self.genotypes_subset, chr = self.replaceXYMTtonums(x.split(':')[0]), 
                                                                   ld_snp = x, ld_window_r2 = 0.4, r2 = 'dprime',ld_window = 100000, 
                                                                   thread_num = int(self.threadnum), ld_window_kb = 7000, nonfounders = '')\
                   .loc[:, ['SNP_B', 'R2', 'DP']]\
                   .query(f'R2 > {r2_thresh}')\
                   .SNP_B.map(lambda x: int(x.split(':')[1]))\
                   .agg(['min', 'max'])\
                   .map(lambda x: f'{x:,}').to_frame().T.set_axis([x])).to_list()).set_axis(['start_qtl', 'end_qtl'], axis = 1)
        return qtls.merge(dists, left_on = snp_col, right_index = True)
        
    # def conditional_analysis(self, trait: str, snpdf: pd.DataFrame() = pd.DataFrame()):
    #     """
    #     Perform conditional analysis for a given trait.
    
    #     This function performs conditional analysis using GCTA-COJO to identify independent genetic loci associated with a trait.
    
    #     :param trait: The name of the trait to analyze.
    #     :type trait: str
    #     :param snpdf: DataFrame containing SNPs to condition on. If empty, all SNPs above the threshold are used.
    #     :type snpdf: pandas.DataFrame, optional
    #     :return: DataFrame with conditional analysis results.
    #     :rtype: pandas.DataFrame
    #     """
    #     os.makedirs(f'{self.path}results/cojo', exist_ok=True)
    #     os.makedirs(f'{self.path}temp/cojo',exist_ok=True)
        
    #     if not snpdf.shape[0]: printwithlog(f'running conditional analysis for trait {trait} and all snps above threshold {self.threshold}')
    #     else: 
    #         #printwithlog(snpdf.shape)
    #         snpstring = ' '.join(snpdf.SNP)
    #         printwithlog(f'running conditional analysis for trait {trait} and all snps below threshold {snpstring}')

    #     pbimtemp = self.pbim.assign(n = self.df.count()[trait]).rename({'snp': 'SNP', 'n':'N'}, axis = 1)[['SNP', 'N']] #- da.isnan(pgen).sum(axis = 1)
    #     tempdf = pd.concat([pd.read_csv(f'{self.path}results/gwas/{trait}.loco.mlma', sep = '\t'),
    #                        pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasx.mlma', sep = '\t'),
    #                        pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasy.mlma', sep = '\t'),
    #                        pd.read_csv(f'{self.path}results/gwas/{trait}_chrgwasmt.mlma', sep = '\t')]).rename({'Freq': 'freq'}, axis =1 )
    #     tempdf = tempdf.merge(pbimtemp, on = 'SNP')[['SNP','A1','A2','freq','b','se','p','N' ]]
    #     mafile, snpl = f'{self.path}temp/cojo/tempmlma.ma', f'{self.path}temp/cojo/temp.snplist'
    #     tempdf.to_csv(mafile, index = False, sep = '\t')
    #     tempdf[-np.log10(tempdf.p) >self.threshold][['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
    #     mafile, snpl = f'{self.path}temp/cojo/tempmlma.ma', f'{self.path}temp/cojo/temp.snplist'
    #     tempdf.to_csv(mafile, index = False, sep = '\t')
    #     if not snpdf.shape[0]:
    #         tempdf[-np.log10(tempdf.p) > self.threshold][['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
    #     else: snpdf[['SNP']].to_csv(snpl, index = False, header = None,sep = '\t')
    #     cojofile = f'{self.path}temp/cojo/tempcojo'
    #     self.bashLog(f'{self.gcta} {self.thrflag} --bfile {self.genotypes_subset} --cojo-slct --cojo-collinear 0.99 --cojo-p {10**-(self.threshold-2)} --cojo-file {mafile} --cojo-cond {snpl} --out {cojofile}', f'cojo_test', print_call=False)
    #     if os.path.isfile(f'{cojofile}.jma.cojo'):
    #         return pd.read_csv(f'{cojofile}.jma.cojo', sep = '\t')
    #     printwithlog(f'Conditional Analysis Failed for  trait {trait} and all snps below threshold {snpstring}, returning the top snp only')
    #     return pd.DataFrame(snpdf.SNP.values, columns = ['SNP'])

    # def conditional_analysis_filter(self, qtltable):
        # """
        # Filter QTL table using conditional analysis.
    
        # This function applies conditional analysis to each group of QTLs by chromosome and trait, retaining only the significant ones.
    
        # :param qtltable: DataFrame containing QTL information.
        # :type qtltable: pandas.DataFrame
        # :return: DataFrame with filtered QTLs after conditional analysis.
        # :rtype: pandas.DataFrame
        # """
    #     return qtltable.groupby(['Chr', 'trait']).progress_apply(lambda df: df.loc[df.SNP.isin(self.conditional_analysis('regressedlr_' +df.name[1].replace('regressedlr_', ''), df).SNP.to_list())]
    #                                                         if df.shape[0] > 1 else df).reset_index(drop= True)
    
    def conditional_analysis_chain_singletrait(self, snpdf: pd.DataFrame, print_call: bool = False, nthread: str = ''):
        """
        Perform conditional analysis for a single trait and chromosome.
    
        This function runs a chain of conditional analyses for a single trait and chromosome, identifying independent genetic loci.
    
        :param snpdf: DataFrame containing SNPs to condition on.
        :type snpdf: pandas.DataFrame
        :param print_call: Whether to print the command call.
        :type print_call: bool, optional
        :param nthread: Number of threads to use.
        :type nthread: str, optional
        :return: DataFrame with conditional analysis results for the single trait and chromosome.
        :rtype: pandas.DataFrame
        """
        trait = snpdf.iloc[0]['trait']#.mode()[0]
        c = snpdf.iloc[0]['Chr']#snpdf['Chr'].mode()[0]
        #bim = self.pbim.set_index('snp')
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

        mlmares = pd.read_csv(f'{self.path}results/gwas/regressedlr_{trait}_chrgwas{self.replacenumstoXYMT(c)}.mlma', sep = '\t')
        snps_of_int = pd.concat([mlmares.query('p<1e-4').query(f'{pos}-5e6<bp<{pos}+5e6') for pos in snpdf2.bp]).drop_duplicates(subset = ['SNP'])
        snps_of_int[['SNP']].to_csv(subset_snp_path,index = False, header = None)

        plink(bfile = self.genotypes_subset, extract = subset_snp_path, make_bed = '', thread_num =  self.threadnum,
              set_missing_var_ids = '@:#', keep_allele_order = '',  set_hh_missing = '' , make_founders = '', 
              out = genotypesCA, chr_set = f'{self.n_autosome} no-xy') #
        
        for num in tqdm(range(snpdf.shape[0])):
            ##### make covarriates dataframe
            covar_snps = covarlist.SNP.to_list()
            # geni = bim.loc[covar_snps, 'i'].to_list()
            #covardf = pd.DataFrame(self.pgen[:, geni], columns =covar_snps , index = self.pfam['iid'])
            covardf = npplink.plink2df(plinkpath=(self.pbim, self.pfam, self.pgen),snplist=covar_snps )
            traitdf = pd.read_csv(f'{self.path}data/pheno/regressedlr_{trait}.txt', sep = r'\s+',  header = None, index_col=0, dtype = {0: str})
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
                                               trait_description = snpdf['trait_description'].fillna('UNK').mode()[0],
                                               qtl_start_bp = f"{r2lis['BP_B'].astype(int).min():,}",
                                               qtl_end_bp   = f"{r2lis['BP_B'].astype(int).max():,}")
                covarlist = pd.concat([covarlist,add2ingsnp])
        return covarlist

    def conditional_analysis_filter_chain(self, qtltable: pd.DataFrame):
        """
        Filter QTL table using a chain of conditional analyses.
    
        This function applies a chain of conditional analyses to each group of QTLs by chromosome and trait.
    
        :param qtltable: DataFrame containing QTL information.
        :type qtltable: pandas.DataFrame
        :return: DataFrame with filtered QTLs after conditional analysis chain.
        :rtype: pandas.DataFrame
        """
        #self.pbim, self.pfam, self.pgen = pandas_plink.read_plink(self.genotypes_subset)
        return qtltable.groupby(['Chr', 'trait'], group_keys=False)\
                       .progress_apply(lambda df: self.conditional_analysis_chain_singletrait( snpdf = df)
                                                 if df.shape[0] > 1 else df).reset_index(drop= True)

    def conditional_analysis_filter_chain_parallel(self, qtltable: pd.DataFrame):
        """
        Filter QTL table using a parallel chain of conditional analyses.
    
        This function applies a chain of conditional analyses in parallel to each group of QTLs by chromosome and trait.
    
        :param qtltable: DataFrame containing QTL information.
        :type qtltable: pandas.DataFrame
        :return: DataFrame with filtered QTLs after parallel conditional analysis chain.
        :rtype: pandas.DataFrame
        """
        #self.pbim, self.pfam, self.pgen = pandas_plink.read_plink(self.genotypes_subset)
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

    def effectsize(self, qtltable: pd.DataFrame = None, display_plots: bool = True):
        """
        Generate effect size plots for QTLs.
    
        This function creates effect size plots for the identified QTLs, showing the distribution of trait values for different genotypes.
    
        :param qtltable: DataFrame containing QTL information. If None, it will be loaded from the final QTL file.
        :type qtltable: pandas.DataFrame, optional
        :param display_plots: Whether to display the plots.
        :type display_plots: bool, optional
        """
        printwithlog(f'starting effect size plot... {self.project_name}') 
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)):
            if os.path.exists(qq:=f'{self.path}results/qtls/finalqtl.csv'):
                qtltable = pd.read_csv(qq)
            else: return
            if not len(qtltable): return
        out = qtltable.reset_index().drop(['index'] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        out = out[out.trait.isin(pd.Series(self.traits).str.replace('regressedlr_', ''))]
        temp = npplink.plink2df(self.genotypes_subset, snplist=out.SNP.unique(), recodeACGT=True ).reset_index(names = 'rfid')
        for col in self.df.filter(regex = 'regressedlr').columns:
            #printwithlog(f'''warning: {col} doesn't have a non regressedout version''')
            if col.replace('regressedlr_', '') not in self.df.columns: self.df[col.replace('regressedlr_', '')] = self.df[col]
        temp = temp.merge(self.df[self.traits + 
                          [t.replace('regressedlr_', '') for t in self.traits] + 
                          ['rfid', 'sex']].rename(lambda x: x.replace('regressedlr_', 'normalized '), axis =1), on = 'rfid')
        
        for name, row in tqdm(list(out.iterrows())):
            isduplicate = (temp[[f'normalized {row.trait}', row.trait]].diff(axis = 1).sum().sum() < 1e-6)
            f, ax = plt.subplots(1, 2 if not isduplicate else 1, figsize = (12 if not isduplicate else 8, 6), dpi=72)
            if len(temp[f'normalized {row.trait}'].dropna().unique()) <= 4:
                for num, ex in enumerate(['normalized '] if isduplicate else ['normalized ', '']):
                    sns.countplot(temp.sort_values(row.SNP), x = row.SNP, hue = ex+ row.trait, ax = ax if isduplicate else ax[num], orient='Vertical')
                    if isduplicate:
                        ax.legend().set_title('')
                        ax.set_ylabel(ex+ row.trait)
                    else:
                        ax[num].legend().set_title('')
                        ax[num].set_ylabel(ex+ row.trait)
            else:    
                figure_colors_box =dict( palette = {'M': 'steelblue', 'F': 'firebrick'},hue="sex",linewidth=1,linecolor='black') \
                                        if str(row.Chr) in ['X', str(self.n_autosome + 1)] else dict(color = 'steelblue')
                figure_colors_strip =dict( palette = {'M': 'steelblue', 'F': 'firebrick'},hue="sex",linewidth=1,edgecolor='black') \
                                          if str(row.Chr) in ['X', str(self.n_autosome + 1)] else dict(color = 'black')
                for num, ex in enumerate(['normalized '] if isduplicate else ['normalized ', '']):
                    sns.boxplot(temp.sort_values(row.SNP), x = row.SNP, y = ex+ row.trait, **figure_colors_box,  ax = ax if isduplicate else ax[num])
                    #cut = 0,  bw= .2, hue="sex", split=True
                    if str(row.Chr) not in ['X', str(self.n_autosome + 1)]:
                        sns.stripplot(temp.sort_values(row.SNP), x = row.SNP, y= ex+row.trait, **figure_colors_strip, jitter = .2, alpha = .4, ax = ax if isduplicate else ax[num] )
                    if isduplicate:
                        ax.hlines(y = 0 if num==0 else temp[ex+ row.trait].mean() , xmin =-.5, xmax=2.5, color = 'black', linewidth = 2, linestyle = '--')
                    else: 
                        ax[num].hlines(y = 0 if num==0 else temp[ex+ row.trait].mean() , xmin =-.5, xmax=2.5, color = 'black', linewidth = 2, linestyle = '--')
            sns.despine()
            os.makedirs(f'{self.path}images/boxplot/', exist_ok=True)
            plt.tight_layout()
            plt.savefig(f'{self.path}images/boxplot/boxplot{row.SNP}__{row.trait}.png'.replace(':', '_'), dpi = 72)
            if display_plots:  plt.show()
            plt.close()   

    def annotQTL(self, qtltable:pd.DataFrame = None, save: bool = False, r2_thresh: float = .65, ld_window:float = 3e6):
        """
        Calculate r2 for nearby SNPs around the top SNPs in QTL analysis.
    
        This function calculates the linkage disequilibrium (r2) values for SNPs near the top SNPs identified in QTL analysis.
    
        :param qtltable: DataFrame containing QTL information. If an empty string, it will be loaded from the final QTL file.
        :type qtltable: pandas.DataFrame or str, optional
        :param save: Whether to save the results to a file.
        :type save: bool, optional
        :param r2_thresh: Threshold for r2 values to consider SNPs as potentially causal.
        :type r2_thresh: float, optional
        """
        printwithlog(f'Annotation QTL: {self.project_name}')
        if qtltable is None: 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv').reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        snp_metadata = pd.read_parquet(f'{self.path}genotypes/snpquality.parquet.gz')
        if 'SNP' not in qtltable.columns:
            qtltable = qtltable.reset_index()
        qtltable['SNP'] = qtltable.SNP.apply(lambda x: self.get_closest_snp(x).iloc[0, 1] if x not in snp_metadata.index.values else x)
        if 'bp' not in qtltable.columns:
            qtltable['bp'] = qtltable.SNP.str.split(':').str[-1].astype(int)
        if 'Chr' not in qtltable.columns:
            qtltable['Chr'] = qtltable.SNP.str.split(':').str[0].map(self.replaceXYMTtonums)
        if 'trait' not in qtltable.columns:
            qtltable['trait'] = 'UNK'
        causal_snps = []
        for name, row in tqdm(list(qtltable.iterrows())):
            ldfilename = f'{self.path}results/lz/temp_qtl_n_@{row.trait}@{row.SNP}'
            r2 = self.plink(bfile = self.genotypes_subset, chr = row.Chr, ld_snp = row.SNP, ld_window_r2 = 0.00001, r2 = 'dprime',\
                            ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb = 7000, nonfounders = '').loc[:, ['SNP_B', 'R2', 'DP']] 
            if len(gwas_files :=  glob(f'{self.path}results/gwas/regressedlr_{row.trait}.loco.mlma') \
                                + glob(f'{self.path}results/gwas/regressedlr_{row.trait}_chrgwas*.mlma')):
                gwas = pd.concat([pd.read_csv(x, sep = '\t') for x in gwas_files]).drop_duplicates(subset = 'SNP')
                tempdf = pd.concat([gwas.set_index('SNP'), r2.rename({'SNP_B': 'SNP'}, axis = 1).drop_duplicates(subset = 'SNP').set_index('SNP')], join = 'inner', axis = 1)
                tempdf = tempdf.combine_first(snp_metadata.loc[tempdf.index])
                tempdf = self.annotate(tempdf.reset_index(), 'SNP', save = False, silent_annotation=True).set_index('SNP').fillna('UNK')
                tempdf.to_csv( f'{self.path}results/lz/lzplottable@{row.trait}@{row.SNP}.tsv', sep = '\t')
            else: 
                tempdf = r2.rename({'SNP_B': 'SNP'}, axis = 1).drop_duplicates(subset = 'SNP').set_index('SNP')
                tempdf = pd.concat([snp_metadata,tempdf], join = 'inner', axis = 1).reset_index()
                tempdf = self.annotate(tempdf, 'SNP', save = False, silent_annotation=True).set_index('SNP').fillna('UNK')
            #subcausal = tempdf.query("putative_impact not in ['UNK', 'MODIFIER']").assign(trait = row.trait, SNP_qtl = row.SNP)
            subcausal = tempdf.query("putative_impact not in ['UNK', 'MODIFIER']").assign(trait = row.trait, SNP_qtl = row.SNP)
            subcausal.columns = subcausal.columns.astype(str)
            if subcausal.shape[0] > 0:
                subcausal = subcausal.query('R2 > @r2_thresh')\
                                     .sort_values('putative_impact', ascending = False).reset_index()\
                                     .drop(['Chr', 'bp', 'se', 'geneid', 'index', 'distancetofeature', 'errors'],errors = 'ignore' ,axis = 1)\
                                     .set_index('SNP_qtl')
                causal_snps += [subcausal]
        if len(causal_snps): 
            causal_snps = pd.concat(causal_snps)
            causal_snps['distance_qtlsnp_annotsnp'] = (causal_snps.index.map(lambda x: int(x.split(':')[1])) - \
                                                       causal_snps.SNP.map(lambda x: int(x.split(':')[1])))\
                                                      .abs()
            causal_snps = causal_snps.query('distance_qtlsnp_annotsnp < @ld_window')
            causal_snps['distance_qtlsnp_annotsnp'] = causal_snps['distance_qtlsnp_annotsnp'].map(bp2str)
        else:
            causal_snps = pd.DataFrame(columns = ['SNP_qtl', 'SNP', 'A1', 'A2', 'CHR', 'DP', 'F_MISS', 'Freq',
                                                   'GENOTYPES', 'HWE', 'MAF', 'PASS', 'PASS_HWE', 'PASS_MAF', 'PASS_MISS',
                                                   'R2', 'b', 'p', 'featureid', 'featureid_type', 'consequence',
                                                   'aminoacids', 'codons', 'refallele', 'putative_impact', 'strand',
                                                   'gene', 'biotype', 'source', 'HGVS.c', 'HGVS.p',
                                                   'position_cdna|cds|protein', 'trait', 'distance_qtlsnp_annotsnp'])
        if save: causal_snps.to_csv(f'{self.path}results/qtls/annotQTL.tsv', sep = '\t')
        return causal_snps
    
    def locuszoom(self, qtltable:pd.DataFrame = None, r2_thresh: float = .65,  padding: float = 2e5, save: bool = True, run_legacy_locuszoom:bool = True,
                   skip_ld_calculation: bool = False, save_causal_table: bool =True, credible_set_threshold: bool = .99, topaxaxis: bool = True, silent=True):
        """
        Generate locuszoom plots for QTLs.
    
        This function creates detailed locuszoom plots for the identified QTLs, showing the linkage disequilibrium (LD) structure,
        nearby genes, and other relevant annotations.
    
        :param qtltable: DataFrame containing QTL information. If None, it will be loaded from the final QTL file.
        :type qtltable: pandas.DataFrame, optional
        :param r2_thresh: Threshold for r2 values to consider SNPs as potentially causal.
        :type r2_thresh: float, optional
        :param padding: Padding around the QTL region in base pairs.
        :type padding: float, optional
        :param save: Whether to save the results to a file.
        :type save: bool, optional
        :param skip_ld_calculation: Whether to skip LD calculation.
        :type skip_ld_calculation: bool, optional
        :param save_causal_table: Whether to save the table of potentially causal SNPs.
        :type save_causal_table: bool, optional
        :param credible_set_threshold: Threshold for credible set inclusion.
        :type credible_set_threshold: float, optional
        :param topaxaxis: Whether to display the x-axis at the top of the plot.
        :type topaxaxis: bool, optional
        """
        printwithlog(f'starting locuszoom... {self.project_name}') 
        from bokeh.models import NumeralTickFormatter
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                         .reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        if not skip_ld_calculation:
            self.annotQTL(save = save_causal_table, r2_thresh= r2_thresh)
        ff_lis = []
        res = {'r2thresh': [], 'minmax': []}
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        printwithlog(f'locuszoom: starting plots {self.project_name}')
        for ((idx, topsnp),boundary) in tqdm(list(itertools.product(qtltable.iterrows(), ['r2thresh','minmax']))):
            if not os.path.isfile(f'{self.path}results/lz/lzplottable@{topsnp.trait}@{topsnp.SNP}.tsv'):
                self.annotQTL(save = save_causal_table, r2_thresh= r2_thresh)
            data = pd.read_csv(f'{self.path}results/lz/lzplottable@{topsnp.trait}@{topsnp.SNP}.tsv', sep = '\t', low_memory = True)\
                     .query('p != "UNK"')\
                     .astype({'HWE': float, 'MAF': float,'R2': float,'b': float,'bp': int,'p': float, 'se': float})
            data['-lg(P)'] = -np.log10(pd.to_numeric(data.p, errors = 'coerce')) 
            if boundary =="r2thresh":
                minval, maxval = data.query(f'R2 > {r2_thresh}').agg({'bp': ['min', 'max']}).values.flatten() + np.array([-padding, padding])
            else:  minval, maxval = (np.array([-3e6, 3e6]) + topsnp.bp).astype(int)
            genes_in_section = self.gtf.query(f'Chr == {topsnp.Chr} and end > {minval} and start < {maxval}')\
                                       .reset_index(drop = True)\
                                       .query("source not in ['cmsearch','tRNAscan-SE']")
            if boundary =="r2thresh":
                ff_lis += [genes_in_section.query('gbkey == "Gene"').sort_values('gene').assign(SNP_origin = topsnp.SNP).drop_duplicates(subset='gene')]
            ngenes = len(genes_in_section.gene.unique())
            causal = pd.read_csv(f'{self.path}results/qtls/annotQTL.tsv' , sep = '\t')\
                       .query(f'SNP_qtl == "{topsnp.SNP}"')
            causal['bp'] = causal.SNP.str.extract(r':(\d+)').astype(int)
            causal = causal.merge(data[['SNP', '-lg(P)']], on='SNP')

            if not os.path.isfile(f'{self.path}results/phewas/pretty_table_both_match.tsv'):
                printwithlog(f' file {self.path}results/phewas/pretty_table_both_match.tsv does not exist...') 
                self.phewas()
            phewas = pd.read_csv(f'{self.path}results/phewas/pretty_table_both_match.tsv' , sep = '\t')\
                       .query(f'SNP_QTL == "{topsnp.SNP}"')
            phewas['bp'] = phewas.SNP_PheDb.str.extract(r':(\d+)').astype(float)
            phewas = phewas.merge(data[['SNP', '-lg(P)']], left_on='SNP_PheDb', right_on='SNP')
            phewas['R2'] = phewas['R2'].map(lambda x: 1 if x == 'Exact match SNP' else float(x))
            phewas['phewas_file'] = phewas['phewas_file'].map(basename)

            if not os.path.isfile(f'{self.path}results/eqtl/pretty_eqtl_table.csv'):
                printwithlog(f' file {self.path}results/eqtl/pretty_eqtl_table.csv does not exist...') 
                self.eQTL()
            eqtl = pd.read_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv' )
            eqtl['SNP'] = eqtl['SNP'].str.replace('chr', '')#\
            eqtl = eqtl.query(f'SNP == "{topsnp.SNP}"')
            eqtl['bp'] = eqtl.SNP_eqtldb.str.extract(r':(\d+)').astype(float)
            
            eqtl = eqtl.merge(data[['SNP', '-lg(P)']], left_on='SNP_eqtldb', right_on='SNP')
            if not os.path.isfile(f'{self.path}results/sqtl/pretty_sqtl_table.csv'):
                printwithlog(f' file {self.path}results/sqtl/pretty_sqtl_table.csv does not exist...') 
                self.sQTL()
            sqtl = pd.read_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv' )
            sqtl['SNP'] = sqtl['SNP'].str.replace('chr', '')#\
            sqtl = sqtl.query(f'SNP == "{topsnp.SNP}"')
            sqtl['bp'] = sqtl.SNP_sqtldb.str.extract(r':(\d+)').astype(float)
            sqtl = sqtl.merge(data[['SNP', '-lg(P)']], left_on='SNP_sqtldb', right_on='SNP')
            tsf = topsnp.to_frame().T.rename({'p': '-lg(P)'}, axis = 1)
            rect = self.make_genetrack_figure_( c=topsnp.Chr, pos_start=minval, pos_end=maxval,  frame_width=1000)
            fontsizes = {'title': 40,  'labels': 25,   'xticks': 15,  'yticks': 15 }
            kw_table = {'data': dict( cmap='Spectral_r', size = 18,alpha = .7, clim= (0,1),\
                                    colorbar=True, line_width = .003, padding=0.05, xlabel = '', line_color = 'Black'),
                        'eqtl': dict( cmap='Blues', size = 23,alpha = .7, clim= (0,1), marker = 'inverted_triangle',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = '', line_color = 'Black'),
                        'sqtl': dict(color='R2', cmap='Blues', size = 23,alpha = .7, clim= (0,1), marker = 'triangle',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = '', line_color = 'Black'),
                        'phewas': dict(color='R2', cmap='Greys', size = 10,alpha = .7, clim= (0,1), marker = 'square',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = ''),
                        'causal': dict(color='R2', cmap='Oranges', size = 23,alpha = .7, clim= (0,1), marker = 'diamond',tools = ['hover'],
                                    colorbar=True,line_width = 1, padding=0.05, xlabel = '', line_color = 'Black'),
                        'tsf': dict(color = 'red', size = 35,alpha = 1, clim= (0,1), marker = 'star',tools = ['hover'],
                                    line_width = 1, padding=0.05, xlabel = '', line_color = 'Black')}
            kw_tabled = {'data': data.query(f'{minval}<bp<{maxval}').sort_values( '-lg(P)'),
                         'eqtl': eqtl.query(f'{minval}<bp<{maxval}').sort_values( '-lg(P)'),
                         'sqtl': sqtl.query(f'{minval}<bp<{maxval}').sort_values( '-lg(P)'),
                         'phewas': phewas.query(f'{minval}<bp<{maxval}').sort_values( '-lg(P)'),
                         'causal': causal.query(f'{minval}<bp<{maxval}').sort_values( '-lg(P)'),
                         'tsf': tsf.query(f'{minval}<bp<{maxval}').sort_values( '-lg(P)')}
            fig = []
            for x in kw_table.keys():
                if x == 'data':
                    a = kw_tabled[x].hvplot.scatter(x = 'bp', y = '-lg(P)', color='R2', **kw_table[x],frame_width=1000, height=400,hover = []).opts(**kw_table[x])
                    b = kw_tabled[x].hvplot.scatter(x = 'bp', y = '-lg(P)', color='black',hover = [])\
                                    .opts(line_width=4, size = 18, padding=0.05, xlabel = '' ,frame_width=1000, height=400, color = 'black')
                    fig += [b*a]
                else:
                    fig += [kw_tabled[x].hvplot.scatter(x = 'bp', y = '-lg(P)',**kw_table[x],frame_width=1000, height=400, hover_cols = list(kw_tabled[x].columns)).opts(**kw_table[x]) ]
            fig = reduce(lambda x,y: x*y, fig)  
            fig = fig*hv.HLine(self.threshold).opts(color='red')
            fig = fig*hv.HLine((self.threshold05)).opts(color='blue')
            
            fig = fig.opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'),ylim = (-.1, max(6, topsnp.p*1.1)), 
                           xticks=5, xrotation = 0, frame_width = 1000, height = 400, fontsize = fontsizes)
            if topaxaxis: fig = fig.opts(xaxis = 'top') #fig = fig.opts(opts.Points( xaxis='top')) #xticks='top'    
            rect = rect.opts(fontsize = fontsizes,xlabel = f'Chr {topsnp.SNP.split(":")[0]}')
            if credible_set_threshold:
                csname = f'{int((credible_set_threshold)*100)}%'
                datacs = data.query(f'({minval}<bp<{maxval})').sort_values('-lg(P)').reset_index(drop=True)
                datacs['ppi'] = bayes_ppi(datacs.p)
                datacs['ppi_s'] = MinMaxScaler(feature_range=(2, 10)).fit_transform(datacs.ppi.values.reshape(-1,1))
                cs = credible_set_idx(datacs.ppi.values,cs_threshold=credible_set_threshold)
                datacs = datacs.assign(**{"CS": 0})
                datacs.loc[cs, "CS"] = 1
                ff1 = hv.Points(datacs, kdims = ['bp', 'CS'] , vdims =['R2', 'ppi_s'] )\
                       .opts(size = 'ppi_s',marker = 'circle', color = 'R2', cmap = 'Spectral_r', frame_width=1000, 
                             fontsize = {'title': 40,  'labels': 25,   'xticks': 15,  'yticks': 8 }, 
                             height=80,alpha = .7, clim= (0,1),yticks =[(0,'∉'+csname),  (1, '∈'+csname)],
                             ylim = (-.5, 1.5),  xaxis=None, labelled=[]) #⊆,⊅  ∉ ∈
                fig = fig+ff1
            finalfig = (fig+rect).cols(1).opts(title = f'locuszoom Chr{topsnp.Chr} {topsnp.trait} {topsnp.SNP}')
            if save: 
                os.makedirs(f'{self.path}images/lz/{boundary}', exist_ok = True)
                hv.save(finalfig, f"{self.path}images/lz/{boundary}/lzi__{topsnp.trait}__{topsnp.SNP}.png".replace(':', '_'))
                hv.save(finalfig, f"{self.path}images/lz/{boundary}/lzi__{topsnp.trait}__{topsnp.SNP}.html".replace(':', '_'))
                #export_png(hv.render(finalfig), filename=f"{self.path}images/lz/{boundary}/lzi__{topsnp.trait}__{topsnp.SNP}.png".replace(':', '_'))
            res[boundary] += [finalfig]
        ff_lis = pd.concat(ff_lis)
        ff_lis['webpage'] = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene=' + ff_lis['gene']
        ff_lis['markdown'] = ff_lis.apply(lambda x: f'[{x.gene}]({x.webpage})', axis = 1)
        if run_legacy_locuszoom: self.legacy_locuszoom(qtltable=qtltable, r2_thresh=r2_thresh, padding=padding, silent = silent)
        if save and save_causal_table: 
            ff_lis.dropna(axis = 1, how = 'all').to_csv(f'{self.path}results/qtls/genes_in_range.csv', index = False)
            genes_in_range2 = self.make_genes_in_range_mk_table()
            genes_in_range2.to_csv(f'{self.path}results/qtls/genes_in_rangemk.csv')
        return res
    
    def legacy_locuszoom(self, qtltable = None, r2_thresh = .65, padding: float = 2e5, print_call = False, updaterefflat = False, silent = True):
        printwithlog(f'generating legacy locuszoom info for project {self.project_name}')
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)):
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')
        os.makedirs(f'{self.path}images/lz/legacyr2', exist_ok = True)
        os.makedirs(f'{self.path}images/lz/legacy6m', exist_ok = True)
        if not glob(f'{self.path}genome_info/ncbi_dataset/data/*/lz.db') or updaterefflat:
            printwithlog(f'generating legacy locuszoom database for project {self.project_name}')
            gtf = self.get_gtf()
            gtf = gtf[gtf.source.str.contains('BestRefSeq')]
            commaj =  lambda x: ','.join(x.astype(str).values)
            def gtfgene2refflatrow(tdf):
                return pd.concat((tdf[['db_xref','Chr', 'strand']].dropna().iloc[0],
                        tdf.agg({'start':'min', 'end' : 'max'}),
                        tdf[tdf.biotype.str.contains('codon$')].agg({'start':'min', 'end' : 'max'}),
                        tdf.query('biotype == "exon"').agg({'Chr': len, 'start':commaj, 'end':commaj})))
            cols2name = ['name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonCount','exonStarts', 'exonEnds']
            refflat = gtf.groupby('gene_id', group_keys = True)\
                         .progress_apply(gtfgene2refflatrow).set_axis(cols2name, axis = 1).dropna(how = 'any')
            refflat[['cdsStart', 'cdsEnd']] = refflat[['cdsStart', 'cdsEnd']].astype(int)
            refflat = refflat.reset_index(names = 'geneName')
            #refflat['chrom'] = refflat['chrom'].astype(str)
            gdatapath = f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/'
            refflat['chrom'] = 'chr'+refflat.chrom.astype(str).map(self.replacenumstoXYMT).str.upper()
            # refflat.to_csv(f'{gdatapath}refflat.txt',index = False, sep = '\t')
            # pd.read_csv(f'{self.all_genotypes}.bim', header = None, 
            #             sep = '\s+', usecols = [1, 0, 3], names = ['chr', 'snp', 'pos'])\
            #             [[ 'snp','chr', 'pos']]\
            #             .to_csv(f'{gdatapath}snppos.txt',index = False, sep = '\t')
            # lzpath = [ y for x in sys.path if os.path.isdir(y := (x.rstrip('/') + '/locuszoom/') ) ][0]
            # bash(f'''conda run -n lzenv {lzpath}bin/dbmeister.py --db {gdatapath}lz.db --refflat {gdatapath}refflat.txt --snp_pos {gdatapath}snppos.txt''', 
            #      shell = True)
            import sqlite3
            with sqlite3.connect(f"{gdatapath}lz.db") as conn:
                refflat.to_sql('refFlat', conn, if_exists="replace", index=False)
                snp_pos= pd.read_csv(f'{self.all_genotypes}.bim', header = None, 
                    sep =r'\s+', usecols = [1, 0, 3], names = ['chr', 'snp', 'pos'])
                snp_pos[[ 'snp','chr', 'pos']].to_sql('snp_pos', conn, if_exists="replace", index=False)
                snp_pos[['snp', 'snp']].set_axis(['rs_orig','rs_current' ], axis = 1).sort_values('rs_orig').to_sql('refsnp_trans', conn, if_exists="replace", index=False)

        ret = []
        for num, (_, qtl_row) in tqdm(list(enumerate(qtltable.reset_index().iterrows()))):
            topsnpchr, topsnpbp = qtl_row.SNP.split(':')
            topsnpchr = self.replaceXYMTtonums(topsnpchr)
            if not os.path.isfile(f'{self.path}results/lz/lzplottable@{qtl_row.trait}@{qtl_row.SNP}.tsv'):
                self.annotQTL(save = True, r2_thresh= r2_thresh)
            test = pd.read_csv(f'{self.path}results/lz/lzplottable@{qtl_row.trait}@{qtl_row.SNP}.tsv', sep = '\t',\
                               dtype = {'p':float, 'bp': int, 'R2': float, 'DP': float}, na_values =  na_values_4_pandas)\
                     .replace([np.inf, -np.inf], np.nan)\
                     .dropna(how = 'any', subset = ['Freq','b','se','p','R2','DP'])   
            test['-log10(P)'] = -np.log10(test.p)
            range_interest = test.query(f'R2> {r2_thresh}')['bp'].agg(['min', 'max'])
            test.SNP = 'chr'+ test.SNP
            os.makedirs(f'{self.path}results/lz/p', exist_ok = True)
            os.makedirs(f'{self.path}results/lz/r2', exist_ok = True)
            lzpvalname, lzr2name = f'{self.path}results/lz/p/{qtl_row.trait}@{qtl_row.SNP}.tsv', f'{self.path}results/lz/r2/{qtl_row.trait}@{qtl_row.SNP}.tsv'
            test.rename({'SNP': 'MarkerName', 'p':"P-value"}, axis = 1)[['MarkerName', 'P-value']].to_csv(lzpvalname, index = False, sep = '\t')
            test.assign(snp1 = qtl_row.SNP).rename({"SNP": 'snp2', 'R2': 'rsquare', 'DP': 'dprime'}, axis = 1)\
                                    [['snp1', 'snp2', 'rsquare', 'dprime']].to_csv(lzr2name, index = False, sep = '\t')
            for filest in glob(f'{self.path}temp/{qtl_row.trait}*{qtl_row.SNP}'): os.system(f'rm -r {filest}')
            def lzcall(c, start, end, snp, title, trait, lzpvalname, lzr2name, lg = ''):
                #import locuszoom_py3
                lzpath = str(Path(locuszoom_py3.__file__).parent)
                #lzpath = [ y for x in sys.path if os.path.isdir(y := (x.rstrip('/') + '/locuszoom_py3/'))][0]
                gdatapath = f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/'
                call = f'''{lzpath}/bin/locuszoom --metal {lzpvalname} --ld {lzr2name}\
                     --refsnp {snp} --chr {int(c)} --start {start} --end {end}\
                     --build manual --db {gdatapath}lz.db\
                     --plotonly showRecomb=FALSE showAnnot=FALSE\
                     --prefix {self.path}temp/{lg}{trait}\
                     signifLine="{self.threshold},{self.threshold05}"\
                     signifLineColor="red,blue"\
                     rfrows=100 showPartialGenes=TRUE geneFontSize=1.2\
                     leftMarginLines=5 rightMarginLines=5 width=20 height=14 axisTextSize=2\
                     smallDot=1 largeDot=2 refsnpTextSize=2\
                     prelude="{lzpath}/prelude.R"
                     title = "{title}"''' #conda run -n lzenv 
                if silent: bash(re.sub(r'\s+',' ' ,call),print_call=False, silent=True, shell=True )
                else: bash(re.sub(r'\s+',' ' ,call),print_call=True, silent=False, shell =True )
                added_p = 'legacyr2' if not lg else 'legacy6m'
                tdstr,s_ = datetime.today().strftime('%y%m%d'), snp.replace(':', '_')
                if os.path.isfile(f'''{self.path}temp/{lg}{trait}_{tdstr}_{s_}/chr{c}_{start}-{end}.pdf'''):
                    bash(f'''cp {self.path}temp/{lg}{trait}_{tdstr}_{s_}/chr{c}_{start}-{end}.pdf \
                         {self.path}images/lz/{added_p}/lz__{trait}__{s_}.pdf''',print_call=False, shell = True )
                    os.system(f'''rm -r {self.path}temp/{lg}{trait}_{tdstr}_{s_}''')
                elif os.path.isfile(f'''{self.path}temp/{lg}{trait}_{tdstr}_{s_}.pdf'''):
                    bash(f'''cp {self.path}temp/{lg}{trait}_{tdstr}_{s_}.pdf \
                         {self.path}images/lz/{added_p}/lz__{trait}__{s_}.pdf''',print_call=False, shell = True )
                    os.system(f'''rm {self.path}temp/{lg}{trait}_{tdstr}_{s_}.pdf''')
                else: 
                    lglist = glob(f'{self.path}temp/{lg}{trait}_{tdstr}_{s_}/chr*_{start}-{end}.pdf') +\
                             glob(f'{self.path}temp/{lg}{trait}_{tdstr}_*.pdf')
                    if not len(lglist):
                        printwithlog(f'could not find the pdf for {lg} {trait} {snp}')
                        return pn.pane.Markdown(f'could not find the pdf for {lg} {trait} {snp}')
                    bash(f'''cp {lglist[0]} \
                         {self.path}images/lz/{added_p}/lz__{trait}__{s_}.pdf''',print_call=False, shell = True )
                    os.system(f'''rm {lglist[0]}''')
                return pn.pane.PDF(f'''{self.path}images/lz/{added_p}/lz__{trait}__{s_}.pdf''',  width = 1040, height = 748)
            _1 = lzcall(topsnpchr, int(range_interest["min"] - padding), int(range_interest["max"] + padding),
                   qtl_row.SNP, f'{qtl_row.trait} SNP {qtl_row.SNP}',  qtl_row.trait,lzpvalname, lzr2name )
            _2 = lzcall(topsnpchr, int(range_interest["min"] - 3e6), int(range_interest["max"] + 3e6),
                   qtl_row.SNP, f'{qtl_row.trait} SNP {qtl_row.SNP}',  qtl_row.trait,lzpvalname, lzr2name, lg = '6m' )
            ret += [pn.Card(_1,_2, title = f'{qtl_row.trait} SNP {qtl_row.SNP}' )]
        from pdf2image import convert_from_path
        for file in tqdm(glob(f'{self.path}images/lz/legacy*/*.pdf'), desc="converting locuszoom pdf to jpeg"):
            images = convert_from_path(file)[0].save(file[:-4]+ '.jpeg', 'JPEG', optimize=True, quality=25)
        printwithlog(f'finished legacy locuszoom info for project {self.project_name}')
        return pn.Card(*ret, title = 'legacy locuszoom', width = 1000)

    @staticmethod
    def get_species_accession(species: str = "rattus norvegicus", return_latest: bool = True, display_options: bool = True):
        """
        Retrieve and display genome assembly accessions for a given species.
    
        This function fetches genome assembly accessions from the NCBI for the specified species. It can display all available
        options and return the latest annotated accession.
    
        :param species: The scientific name of the species.
        :type species: str, optional
        :param return_latest: Whether to return the latest annotated accession.
        :type return_latest: bool, optional
        :param display_options: Whether to display the available accessions.
        :type display_options: bool, optional
        :return: The latest annotated accession if return_latest is True, otherwise the DataFrame of all options.
        :rtype: str or pandas.DataFrame
        """
        srtgi = pd.json_normalize(json.loads(bash(f'''datasets summary genome taxon "{species.lower().replace('_', ' ')}"''', shell = True, silent = True, print_call=False)[0])['reports'])\
              .rename({'paired_accession': 'annotated_accession', 'annotation_info.provider': 'provider', 
                       'annotation_info.release_date': 'release_date', 'annotation_info.report_url': 'report_url',
                       'assembly_info.assembly_name': 'assembly_name', 'assembly_info.biosample.description.organism.organism_name':'organism_name'}, axis = 1)\
              .sort_values('release_date', ascending = False).reset_index(drop = True)
        firstcols = ['assembly_name', 'accession','annotated_accession', 'provider', 'release_date', 'report_url', 'organism_name']
        disp = srtgi[firstcols + list(set(srtgi.columns)-set(firstcols))]
        disp['annotated_accession'] = disp.annotated_accession.str.replace('GCA_', 'GCF_')
        disp['accession'] = disp.annotated_accession.str.replace('GCF_', 'GCA_')
        if display_options: display(fancy_display(disp, download_name='species_accesion.csv')) #.dropna(subset = 'annotated_accession')
        if not return_latest:return disp
        return disp.loc[0, 'annotated_accession']
    
    def ask_user_genome_accession(self): 
        """
        Prompt the user to specify a genome accession.
    
        This function interacts with the user to get the genome accession for a specified species. It fetches available accessions
        and allows the user to select or confirm the latest accession.
        """
        genome_accession = ''
        cnt = 0
        while genome_accession == '' and cnt < 3:
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
            cnt+=1
        self.pull_NCBI_genome_info(GCF_assession_id = genome_accession, redownload = True)
    
    def pull_NCBI_genome_info(self, GCF_assession_id = 'GCF_036323735.1', redownload = False):
        """
        Download and organize NCBI genome information.
    
        This function downloads the genome, GTF, and other relevant files from NCBI for the specified accession ID. It organizes
        these files and creates necessary conversion tables for chromosome names.
    
        :param GCF_assession_id: The genome accession ID.
        :type GCF_assession_id: str, optional
        :param redownload: Whether to force re-download the data.
        :type redownload: bool, optional
        """
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
                #bash(f'''grep -v "#" {i2} | sort -k1,1 -k4,4n -k5,5n -t$'\t' | bgzip -c > {i2}.gz''', shell = True, silent = True, print_call=False)
                bash(f'''grep -v '^#[^#]' {i2} | sort -k1,1V -k4,4n -k5,5n -t$'\t'| bgzip -c > {i2}.gz''' , shell = True, silent = True, print_call=False)
                bash(f"tabix -p gff {i2}.gz", silent = True, print_call=False)
        if not os.path.isfile(f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/sequence_report.jsonl'):
            printwithlog(f'couldnt find the file: {self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/sequence_report.jsonl')
            printwithlog(f'the path is {self.path}')
            printwithlog(f'the genome acession is {self.genome_accession}')
        self.chr_conv_table = pd.read_json(f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/sequence_report.jsonl',  lines=True).rename({'chrName': 'Chr'}, axis = 1)
        self.n_autosome = self.chr_conv_table.Chr.str.extract(r'(\d+)')[0].fillna(-1).astype(int).max()
        self.checkvalidnumchr =  lambda x: True if len(re.findall(r'^chr\d+$|^\d+$|m|mt|x|y', x.lower())) else False
        self.replacenumstoXYMT = lambda x: str(int(float(x))).replace(str(self.n_autosome+1), 'x')\
                                                 .replace(str(self.n_autosome+2), 'y')\
                                                 .replace(str(self.n_autosome+4), 'mt')
        self.replaceXYMTtonums = lambda x: int(float(str(x).lower().replace('chr', '').replace('x', str(self.n_autosome+1))\
                                                 .replace('y', str(self.n_autosome+2))\
                                                 .replace('mt', str(self.n_autosome+4))\
                                                 .replace('m', str(self.n_autosome+4))))
        syntable = self.chr_conv_table[['Chr', 'refseqAccession']]#.query('Chr != "Un"')
        syntable = syntable[syntable.Chr.map(self.checkvalidnumchr)]
        syntable['Chrnum'] = syntable.Chr.map(self.replaceXYMTtonums)
        syntable = syntable.melt(id_vars = 'refseqAccession').drop('variable', axis = 1).astype(str).drop_duplicates().sort_values('value')
        syntable = syntable[~syntable.refseqAccession.str.startswith('NW')]
        syntable = pd.concat([syntable, syntable.assign(value = 'chr'+syntable.value)])
        #syntable = pd.concat([syntable, syntable.set_axis(syntable.columns[::-1], axis = 1)])
        syntable.to_csv(self.chrsyn,  sep = '\t', header = None, index = False)
        
        self.genome_assession_info = pd.read_json(f'{self.path}genome_info/ncbi_dataset/data/assembly_data_report.jsonl', lines = True)
        self.genome_version= self.genome_assession_info.loc[0,'assemblyInfo']['assemblyName']
        self.genomefasta_path = f'{self.path}genome_info/ncbi_dataset/data/{self.genome_accession}/{self.genome_accession}_{self.genome_version}_genomic.fna'
        if needs_dowload or not os.path.isfile(self.genomefasta_path+'.fai'): 
            pysam.faidx(self.genomefasta_path)
        self.species = self.genome_assession_info.loc[0, 'organism']['organismName'].replace(' ', '_').lower()
        self.species_it = f'''*{str.capitalize(self.species.replace('_', ' '))}*'''
        self.species_cname = str(self.genome_assession_info.loc[0, 'organism']['commonName'])
        self.taxid = str(self.genome_assession_info.loc[0, 'organism']['taxId']  )      
    
    def get_gtf(self):
        """
        Retrieve and process the GTF file for the genome.
    
        Steps:
        1. Check if 'gtf_path' and 'chr_conv_table' attributes exist. If not, call pull_NCBI_genome_info to set them up.
        2. Read the GTF file from the specified path, filtering out comment lines and setting column names.
        3. Filter out rows where the 'biotype' is 'transcript' and reset the index.
        4. Replace 'gene' with 'transcript' in the 'biotype' column.
        5. Convert the 'ID' column from a string to a JSON format using the >fjsongtfid2json function.
        6. Normalize the JSON data and merge it with the GTF DataFrame, then drop the original 'ID' column.
        7. Filter out pseudo genes and merge the GTF DataFrame with the chromosome conversion table.
        8. Remove rows with undefined or unwanted chromosome values.
        9. Convert chromosome identifiers using replaceXYMT→νmsreplaceXYMTtonums.
        10. Remove rows with missing or unassigned genes and unnecessary columns.
        11. Add a 'genomic_pos' column combining chromosome, start, and end positions.
        12. Assign the processed DataFrame to the 'gtf' attribute and return it.
    
        :return: Processed GTF DataFrame.
        :rtype: pandas.DataFrame
        """
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
        gtf= gtf[~gtf.Chr.fillna('UNK').isin(['Un', '', 'UNK', 'na'])]
        gtf['Chr'] = gtf['Chr'].map(lambda x: self.replaceXYMTtonums(x.split('_')[0]))
        gtf = gtf.loc[(gtf.gene.fillna('') != '') & ~gtf.transcript_id.astype(str).str.contains('unassigned_transcript') , ~gtf.columns.str.contains(' ') ]
        gtf = gtf.assign(genomic_pos = gtf.Chr.astype(str)+':'+gtf.start.astype(str)+'-'+gtf.end.astype(str))
        self.gtf = gtf
        return gtf

    def get_closest_snp(self, s: str, include_snps_in_gene: bool = False, include_snps_in_ld: bool  = False):
        """
        Find the closest SNP to a given gene or position.
    
        Steps:
        1. Retrieve and process the GTF file if not already done.
        2. Check if the input 's' is a gene name present in the GTF file. If it is:
            a. Get the chromosome and mean position of the gene.
        3. If the input 's' is not a gene, assume it is a chromosome:position string:
            a. Split the string to get the chromosome and position.
            b. Convert chromosome identifier using replaceXYMT→νmsreplaceXYMTtonums.
        4. Load SNP data from the genotypes BIM file, filtering for the relevant chromosome.
        5. If ∈cludesnps∈_≥≠include_snps_in_gene is True and the input 's' is a gene:
            a. Find SNPs within the gene.
            b. If no SNPs are found within the gene, select the closest SNP.
        6. If ∈cludesnps∈_≥≠include_snps_in_gene is False or the input 's' is not a gene, select the closest SNP to the given position.
        7. If ∈cludesnps∈_ldinclude_snps_in_ld is True:
            a. Find SNPs in linkage disequilibrium (LD) with the identified SNPs.
        8. Return the identified SNP(s) as a DataFrame.
    
        :param s: Gene name or chromosome:position string.
        :type s: str
        :param include_snps_in_gene: Whether to include SNPs within the gene, defaults to False.
        :type include_snps_in_gene: bool, optional
        :param include_snps_in_ld: Whether to include SNPs in linkage disequilibrium, defaults to False.
        :type include_snps_in_ld: bool, optional
        :return: DataFrame of the closest SNP(s).
        :rtype: pandas.DataFrame
        """
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
        """
        Create an interactive widget for locuszoom plots.
    
        Steps:
        1. Import necessary modules from ipywidgets.
        2. Retrieve and process the GTF file if not already done.
        3. Create a Combobox widget for selecting a gene or position.
        4. Create a Dropdown widget for selecting a trait.
        5. Create a BoundedFloatText widget for setting the R2 boundary.
        6. Create a BoundedIntText widget for setting the padding.
        7. Define a function locusz∞mmaνalquerylocuszoom_manual_query that:
            a. Finds the closest SNP to the selected location.
            b. Retrieves the corresponding GWAS results for the selected trait.
            c. Displays the temporary QTL table.
            d. Generates and displays the locuszoom plot.
        8. Create an interactive manual widget using the ∫eractmaνalinteract_manual function.
        9. Return the interactive widget.
    
        :return: Interactive widget for locuszoom plots.
        :rtype: ipywidgets.interactive
        """
        from ipywidgets import interact, interact_manual, widgets
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        cloc = widgets.Combobox(placeholder='Choose Gene or position "chr:pos" ',options=tuple(gtf.gene.unique()),description='gene or pos:',ensure_option=False,layout = widgets.Layout(width='500px'))
        ctrait = widgets.Dropdown(description ='trait:',options=[x.replace('regressedlr_', '') for x in self.traits],placeholder='choose a trait',layout = widgets.Layout(width='500px'))
        cr2_thresh = widgets.BoundedFloatText(description = 'R2 boudary', value = .65, max = 1, min = 0, step = 0.01,layout = widgets.Layout(width='150px')) 
        cpadding = widgets.BoundedIntText(description = 'Padding', value = int(.1e6), max = int(10e6), min = 0,format='0,0')
        def locuszoom_manual_query(loc, trait,r2_thresh, padding):
            aa = self.get_closest_snp(loc).loc[0, :]
            tempqtls = pd.read_csv(f'{self.path}results/gwas/regressedlr_{trait}_chrgwas{aa.Chr}.mlma', sep = '\t', index_col = 1).loc[aa.SNP].to_frame().T.reset_index(names = 'SNP')
            tempqtls['p'] = -np.log10(tempqtls['p'].astype(float))
            display(tempqtls)
            res = self.locuszoom(qtltable = tempqtls.assign(QTL = True, trait = trait, interval_size = 'UNK', significance_level = 'UNK') , 
                           save = False, save_causal_table=False, r2_thresh=r2_thresh, padding=padding)
            display(res['minmax'][0])
            print('Done!')
        interact_manual(locuszoom_manual_query,loc = cloc, trait = ctrait, r2_thresh= cr2_thresh, padding = cpadding)
        return 
    
    def phewas(self, qtltable: pd.DataFrame = None, phewas_file = '',
               ld_window: int = int(3e6), save:bool = True, pval_threshold: float = 1e-4, nreturn: int = 1 ,r2_thresh: float = .8,\
              annotate: bool = True, **kwards) -> pd.DataFrame:
        """
        Perform Phenome-Wide Association Study (PheWAS) analysis.
    
        Steps:
        1. Print a log message indicating the start of the PheWAS process.
        2. Determine the base name for saving files based on the provided PheWAS file.
        3. Load the final QTL table if not provided as input.
        4. Read and filter the PheWAS database for significant p-values.
        5. Merge the QTL table with the PheWAS database for exact SNP matches.
        6. Save the table of exact matches to a CSV file.
        7. Identify nearby SNPs within a specified window and LD threshold.
        8. Merge the nearby SNPs with the PheWAS database for window matches.
        9. Save the table of window matches to a CSV file.
        10. Annotate the results if specified.
        11. Group and filter the table of window matches.
        12. Save the filtered table of window matches to a CSV file.
        13. Create and save a pretty table for exact matches.
        14. Create and save a pretty table for window matches.
        15. Combine the exact and window match tables and save the combined table.
        16. Return the combined table.
    
        :param qtltable: DataFrame containing QTL information.
        :param phewas_file: Path to the PheWAS database file.
        :param ld_window: Window size for identifying nearby SNPs.
        :param save: Boolean indicating whether to save the results.
        :param pval_threshold: P-value threshold for filtering the PheWAS database.
        :param nreturn: Number of top results to return for each group.
        :param r2_thresh: R-squared threshold for identifying nearby SNPs.
        :param annotate: Boolean indicating whether to annotate the results.
        :param kwards: Additional keyword arguments.
        :return: DataFrame containing the combined results of the PheWAS analysis.
        """
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
        if 'SNP' in qtltable.columns: qtltable = qtltable.set_index('SNP') 
        if 'trait' not in qtltable.columns: qtltable['trait'] = 'UNK'
        if 'bp' not in qtltable.columns:
            qtltable['bp'] = qtltable.index.str.split(':').str[-1].astype(int)
        if 'Chr' not in qtltable.columns:
            qtltable['Chr'] = qtltable.index.str.split(':').str[0].map(self.replaceXYMTtonums)

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
              .query(f'R2 > {r2_thresh}')\
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
            table_window_match = self.annotate(table_window_match.rename({'A1_phewasdb':'A1', 'A2_phewasdb': 'A2',
                                            'Chr_phewasdb':'Chr', 'bp_phewasdb':'bp'}, axis = 1), \
                                 'NearbySNP', save = False).rename({'gene': 'gene_phewasdb', 'annotation':'annotation_phewasdb'}, axis = 1)
            table_exact_match = self.annotate(table_exact_match.rename({'A1_QTL':'A1', 'A2_QTL': 'A2','Chr_QTL':'Chr', 'bp_QTL':'bp'}, axis = 1), save = False)\
                                    .rename({'A1': 'A1_QTL', 'A2':'A2_QTL','Chr':'Chr_QTL', 'bp':'bp_QTL'}, axis = 1)
            # table_exact_match= table_exact_match.assign(**{i:'' for i in set(['gene', 'annotation'])-set(table_window_match.columns)})\
            #                                     .rename({'gene': 'gene_phewasdb', 'annotation':'annotation_phewasdb'}, axis = 1)
            table_exact_match= table_exact_match.assign(**{i:'' for i in set(['gene', 'annotation'])-set(table_exact_match.columns)})\
                                                 .rename({'gene': 'gene_phewasdb', 'annotation':'annotation_phewasdb'}, axis = 1)
            
        out = table_window_match.groupby([ 'SNP_QTL','project', 'trait_phewasdb'])\
                                .apply(lambda df : df[df.uploadeddate == df.uploadeddate.max()]\
                                                   .nsmallest(n = nreturn, columns = 'p_phewasdb'))\
                                .reset_index(drop = True)\
                                .assign(phewas_r2_thresh = r2_thresh, phewas_p_threshold = pval_threshold ) #, 'annotation_phewasdb'  'gene_phewasdb'
        
        if save: out.to_csv(self.phewas_window_r2, index = False)
        
        
        ##### make prettier tables
        #phewas_info =   pd.read_csv(self.phewas_exact_match_path).drop('QTL', axis = 1).reset_index()
        phewas_info = table_exact_match.drop('QTL', axis = 1,errors = 'ignore').reset_index()
        phewas_info = phewas_info.loc[:, ~phewas_info.columns.str.contains(r'Chr|A\d|bp_|b_|se_|Nearby|research|filename')]\
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
        phewas_info = out.drop('QTL', axis = 1, errors = 'ignore').reset_index()
        phewas_info = phewas_info.loc[:, ~phewas_info.columns.str.contains(r'Chr|A\d|bp_|b_|se_|Nearby|research|filename')]\
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
        """
        Create an interactive PheWAS widget using ipywidgets.
    
        Steps:
        1. Import the necessary widgets from ipywidgets.
        2. Get the GTF data if not already loaded.
        3. Initialize a variable to store the result of the PheWAS widget.
        4. Create a Combobox widget for selecting genes or positions.
        5. Create a Combobox widget for selecting the PheWAS file.
        6. Create a BoundedFloatText widget for setting the R2 threshold.
        7. Define a function for manual PheWAS queries:
            a. Get the closest SNP(s) to the specified gene or position.
            b. Perform the PheWAS analysis using the specified R2 threshold and PheWAS file.
            c. Sort and filter the results.
            d. Store the result in the phewaswtrest̲phewas_widget_result attribute.
            e. Display the result using a fancy display function.
        8. Use the ∫eractmaνalinteract_manual function to create an interactive interface for the PheWAS query function.
        9. Return from the function to render the widget.
    
        """
        from ipywidgets import interact, interact_manual, widgets
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        self.phewas_widget_result = None
        cloc = widgets.Combobox(placeholder='Choose Gene or position "chr:pos" ',options=tuple(gtf.gene.unique()), \
                                description='gene or pos:',ensure_option=False,layout = widgets.Layout(width='500px'))
        phewas_file = widgets.Combobox(placeholder='Choose parquet phewas file" ',
                                       options=[], value = self.phewas_db,
                                       description='phewas_file:',ensure_option=False,layout = widgets.Layout(width='500px'))
        cr2_thresh = widgets.BoundedFloatText(description = 'R2 boudary', value = .9, max = 1, min = 0, step = 0.01,layout = widgets.Layout(width='150px')) 
        def phewas_manual_query(loc,r2_thresh, phewas_file):
            snps = self.get_closest_snp( s= loc, include_snps_in_ld=False, include_snps_in_gene=True) \
                       .assign(trait = 'UNK', p = -1, QTL = True, trait_description = 'UNK', Freq = 'UNK').set_index('SNP')
            out = self.phewas(qtltable =snps,  save= False,r2_thresh = r2_thresh, annotate = True,phewas_file = phewas_file ).drop_duplicates()
            out = out.loc[out.R2.replace('Exact match SNP', 1.1).sort_values(ascending = False).index]\
                     .drop_duplicates(['SNP_PheDb','trait_PheDb', 'project'], keep = 'first').reset_index(drop = True)
            self.phewas_widget_result = out
            pn.extension('tabulator')
            bsname = basename(phewas_file).split('.')[0]
            display(fancy_display(self.phewas_widget_result, download_name = f'phewas_result_{loc}_{bsname}.csv', flexible = True))
            print('Done! result is in self.phewas_widget_result')
            return 
        interact_manual(phewas_manual_query,loc = cloc, r2_thresh= cr2_thresh, phewas_file=phewas_file)
        return 
        
    def eQTL(self, qtltable: pd.DataFrame= None,
             pval_thresh: float = 1e-4, r2_thresh: float = .6, nreturn: int = 1, ld_window: int = 3e6,\
            tissue_list: list = ['Adipose','BLA','Brain','Eye','IL','LHb','Liver','NAcc','OFC','PL','pVTA','RMTg'],\
            annotate = True, **kwards) -> pd.DataFrame:
        """
        Perform eQTL analysis on a given QTL table by iterating over a list of tissues and searching for cis-acting eQTLs in LD with the QTLs.
    
        Steps:
        1. Check if the genome accession is valid for eQTL analysis.
        2. Read in the QTL table if not provided.
        3. Iterate over the list of tissues.
        4. For each tissue, read the eQTL data from a remote file.
        5. Get the nearby SNPs that are in LD with the QTL using plink.
        6. Merge the eQTL data with the nearby SNPs.
        7. Filter the resulting dataframe using the provided p-value threshold and R2 threshold.
        8. Return the n smallest p-values for each QTL.
        9. Concatenate the results from all tissues.
        10. Annotate the eQTL table if the annotate flag is True.
        11. Save the final eQTL table to a file.
        12. Make pretty tables and save them.
    
        :param qtltable: DataFrame containing QTL information.
        :param pval_thresh: Float threshold for the p-value of the eQTLs (default is 1e-4).
        :param r2_thresh: Float threshold for the r-squared value of the eQTLs (default is 0.6).
        :param nreturn: Integer number of eQTLs to return for each QTL (default is 1).
        :param ld_window: Integer size of the window around the QTL to search for eQTLs (default is 3e6).
        :param tissue_list: List of tissues to perform eQTL analysis on (default is a list of specific tissues).
        :param annotate: Boolean indicating whether to annotate the resulting eQTL table (default is True).
        :param kwards: Additional keyword arguments.
        :return: DataFrame containing the annotated eQTL table.
        """
        printwithlog(f'starting eqtl ... {self.project_name}') 
        if self.genome_accession not in [ 'GCF_000001895.5', 'GCF_015227675.2']: 
            res = pd.DataFrame(columns = ['trait', 'SNP', '-log10(P-value)', 'R2', 'SNP_eqtldb', 'tissue', \
                                    '-log10(pval_nominal)', 'DP', 'Ensembl_gene', 'gene_id', 'slope', 'tss_distance', 'af', 'presence_samples'])
            res.to_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv', index = False)
            return res
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            printwithlog(f'reading file from {self.path}results/qtls/finalqtl.csv...') 
            if not os.path.isfile(f'{self.path}results/qtls/finalqtl.csv'):
                printwithlog(f' file {self.path}results/qtls/finalqtl.csv does not exist, consider running callQTLs...') 
                return pd.DataFrame(columns = ['trait', 'SNP', '-log10(P-value)', 'R2', 'SNP_eqtldb', 'tissue', \
                                    '-log10(pval_nominal)', 'DP', 'Ensembl_gene', 'gene_id', 'slope', 'tss_distance', 'af', 'presence_samples'])
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                        .reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                        .set_index('SNP').loc[:, : 'significance_level']
        if 'SNP' in qtltable.columns: qtltable = qtltable.set_index('SNP') 
        if 'trait' not in qtltable.columns: qtltable['trait'] = 'UNK'
        if 'bp' not in qtltable.columns:
            qtltable['bp'] = qtltable.index.str.split(':').str[-1].astype(int)
        if 'Chr' not in qtltable.columns:
            qtltable['Chr'] = qtltable.index.str.split(':').str[0].map(self.replaceXYMTtonums)
        out = []
        genomeacc2rnv = {'GCF_000001895.5': 'rn6', 'GCF_015227675.2': 'rn7' }
        #genomeacc2rnv[self.genome_accession]
        for tissue in tqdm(tissue_list,  position=0, desc="tissue", leave=True):
            #f'https://ratgtex.org/data/eqtl/{tissue}.{}.cis_qtl_signif.txt.gz'
            eqtl_link = f'https://ratgtex.org/data/eqtl/cis_qtl_signif.{tissue}.v3_{genomeacc2rnv[self.genome_accession]}.txt.gz'
            tempdf = pd.read_csv(eqtl_link, sep = '\t').assign(tissue = tissue).rename({'variant_id': 'SNP'}, axis = 1)
            tempdf['SNP'] =tempdf.SNP.str.replace('chr', '')
            out += [pd.concat([ 
                   self.plink(bfile = self.genotypes_subset, chr = row.Chr,ld_snp = row.name,r2 = 'dprime',\
                   ld_window = ld_window, thread_num = int(self.threadnum), nonfounders = '')\
                  .drop(['CHR_A', 'BP_A', 'CHR_B'], axis = 1)\
                  .rename({'SNP_A': 'SNP', 'SNP_B': 'NearbySNP', 'BP_B': 'NearbyBP'}, axis = 1)\
                  .assign(**row.to_dict())\
                  .merge(tempdf, right_on= 'SNP',  left_on='NearbySNP', how = 'inner', suffixes = ('_QTL', '_eqtldb'))\
                  .query(f'R2 > {r2_thresh} and pval_nominal < {pval_thresh}')\
                  .nsmallest(nreturn, 'pval_nominal')
                  for  _, row in qtltable.iterrows() ])]

        out = pd.concat(out).reset_index(drop=True).drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        if annotate:
            out = self.annotate(out, 'NearbySNP', save = False)
        self.eqtl_path = f'{self.path}results/eqtl/eqtl.csv'
        out['presence_samples'] = out.ma_samples.astype(str) + '/'+ out.ma_count.astype(str)
        out.to_csv(self.eqtl_path, index= False)
        
        #### make pretty tables
        eqtl_info = pd.read_csv(self.eqtl_path).rename({'p':'-log10(P-value)'}, axis = 1)
        eqtl_info['-log10(pval_nominal)'] = -np.log10(eqtl_info['pval_nominal'])
        #eqtl_info['Ensembl_gene'] = eqtl_info.gene_id.map(query_gene(eqtl_info.gene_id.unique(), self.taxid)['EnsemblRapid']).fillna('')
        try:
            eqtl_info = eqtl_info.merge(query_gene(eqtl_info.gene_id, self.taxid)\
                 .rename({'EnsemblRapid': 'Ensembl_gene'}, axis =1)[['Ensembl_gene']].fillna(''),
                how ='left', left_on='gene_id', right_index=True)
        except:
            printwithlog('previous query gene is missing "EnsemblRapid" getting this information from the "ensembl" column')
            r = query_gene(eqtl_info.gene_id, self.taxid).ensembl\
                  .map(lambda x: x[0] if isinstance(x, list) else x)\
                  .map(lambda x: np.nan if pd.isna(x) else x['gene'])\
                  .to_frame().set_axis(['Ensembl_gene'], axis = 1)
            r= r.groupby(r.index).agg(lambda x: y[0] if len(y:=x.dropna().unique()) else '')
            eqtl_info = eqtl_info.merge(r, how ='left', left_on='gene_id', right_index=True)
        
        eqtl_info = eqtl_info.loc[:,  ['trait','SNP_QTL','-log10(P-value)','R2','SNP_eqtldb','tissue', '-log10(pval_nominal)','DP' ,
                                       'Ensembl_gene','gene_id', 'slope' ,'tss_distance', 'af', 'presence_samples']].rename(lambda x: x.replace('_QTL', ''), axis = 1)
        eqtl_info.SNP = 'chr' + eqtl_info.SNP
        eqtl_info = eqtl_info.drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        eqtl_info.to_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv', index = False)
        return eqtl_info
    
    
    def sQTL(self, qtltable: pd.DataFrame = None,
             pval_thresh: float = 1e-4, r2_thresh: float = .6, nreturn: int =1, ld_window: int = 3e6, just_cis = True,
             tissue_list: list = ['Adipose','BLA','Brain','Eye','IL','LHb','Liver','NAcc','OFC','PL','pVTA','RMTg'], 
             annotate = True, **kwards) -> pd.DataFrame:
        """
        Perform sQTL analysis on a given QTL table by iterating over a list of tissues and searching for splice QTLs (sQTLs) that are in LD with the QTLs.
    
        Steps:
        1. Check if the genome accession is valid for sQTL analysis.
        2. Read in the QTL table if not provided.
        3. Iterate over the list of tissues and cis/trans combinations.
        4. For each tissue, read the sQTL data from a remote file.
        5. Get the nearby SNPs that are in LD with the QTL using plink.
        6. Merge the sQTL data with the nearby SNPs.
        7. Filter the resulting dataframe using the provided p-value threshold and R2 threshold.
        8. Return the n smallest p-values for each QTL.
        9. Concatenate the results from all tissues.
        10. Annotate the sQTL table if the annotate flag is True.
        11. Save the final sQTL table to a file.
        12. Make pretty tables and save them.
    
        :param qtltable: DataFrame containing QTL information.
        :param pval_thresh: Float threshold for the p-value of the sQTLs (default is 1e-4).
        :param r2_thresh: Float threshold for the r-squared value of the sQTLs (default is 0.6).
        :param nreturn: Integer number of sQTLs to return for each QTL (default is 1).
        :param ld_window: Integer size of the window around the QTL to search for sQTLs (default is 3e6).
        :param just_cis: Boolean indicating whether to only consider cis-acting sQTLs (default is True).
        :param tissue_list: List of tissues to perform sQTL analysis on (default is a list of specific tissues).
        :param annotate: Boolean indicating whether to annotate the resulting sQTL table (default is True).
        :param kwards: Additional keyword arguments.
        :return: DataFrame containing the annotated sQTL table.
        """
        printwithlog(f'starting spliceqtl ... {self.project_name}') 
        if self.genome_accession not in ['GCF_015227675.2', 'GCF_000001895.5']: 
            pd.DataFrame(columns = ['trait', 'SNP', '-log10(P-value)', 'R2', 'SNP_sqtldb', 'tissue',
                                    '-log10(pval_nominal)', 'DP', 'Ensembl_gene', 'gene_id', 'slope', 'tss_distance', 'af', 'presence_samples']).to_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv', index = False)
            return -1
        if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
            printwithlog(f'reading file from {self.path}results/qtls/finalqtl.csv...') 
            qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                         .reset_index().drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                         .set_index('SNP').loc[:, : 'significance_level']
        if 'SNP' in qtltable.columns: qtltable = qtltable.set_index('SNP') 
        if 'trait' not in qtltable.columns: qtltable['trait'] = 'UNK'
        if 'bp' not in qtltable.columns:
            qtltable['bp'] = qtltable.index.str.split(':').str[-1].astype(int)
        if 'Chr' not in qtltable.columns:
            qtltable['Chr'] = qtltable.index.str.split(':').str[0].map(self.replaceXYMTtonums)
        out = []
        loop_str = [('cis','cis_qtl_signif')] if just_cis else [('cis','cis_qtl_signif'), ('trans','trans_qtl_pairs')]
        genomeacc2rnv = {'GCF_000001895.5': 'rn6', 'GCF_015227675.2': 'rn7' }
        
        for tissue, (typ, prefix) in tqdm(list(itertools.product(tissue_list, loop_str)),  position=0, desc="tissue+CisTrans", leave=True):
            #sqtl_link = f'https://ratgtex.org/data/splice/{tissue}.{genomeacc2rnv[self.genome_accession]}.splice.{prefix}.txt.gz'
            sqtl_link = f'https://ratgtex.org/data/splice/splice.{prefix}.{tissue}.v3_{genomeacc2rnv[self.genome_accession]}.txt.gz'
            tempdf = pd.read_csv(sqtl_link, sep = '\t').assign(tissue = tissue).rename({'variant_id': 'SNP', 'pval': 'pval_nominal'}, axis = 1)  
            tempdf['SNP'] =tempdf.SNP.str.replace('chr', '')
            out += [pd.concat([ 
                   self.plink(bfile = self.genotypes_subset, chr = row.Chr,ld_snp = row.name,r2 = 'dprime',\
                   ld_window = ld_window, thread_num = int(self.threadnum), nonfounders = '')\
                  .drop(['CHR_A', 'BP_A', 'CHR_B'], axis = 1)\
                  .rename({'SNP_A': 'SNP', 'SNP_B': 'NearbySNP', 'BP_B': 'NearbyBP'}, axis = 1)\
                  .assign(**row.to_dict())\
                  .merge(tempdf, right_on= 'SNP',  left_on='NearbySNP', how = 'inner', suffixes = ('_QTL', '_sqtldb'))\
                  .query(f'R2 > {r2_thresh} and pval_nominal < {pval_thresh}')\
                  .nsmallest(nreturn, 'pval_nominal').assign(sQTLtype = typ)
                  for  _, row in qtltable.iterrows() ])]

        out = pd.concat(out).reset_index(drop=True)
        out['gene_id'] = out.phenotype_id.str.rsplit(':',n = 1).str[-1]
        if annotate: out = self.annotate(out, 'NearbySNP', save = False)
        out['presence_samples'] = out.ma_samples.astype(str) + '/'+ out.ma_count.astype(str)
        self.sqtl_path = f'{self.path}results/sqtl/sqtl_table.csv'
        out.drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1).to_csv(self.sqtl_path, index= False)
        
        #### make pretty tables
        sqtl_info = pd.read_csv(self.sqtl_path).rename({'p':'-log10(P-value)'}, axis = 1)
        sqtl_info['-log10(pval_nominal)'] = -np.log10(sqtl_info['pval_nominal'])
        # sqtl_info['Ensembl_gene'] = sqtl_info.gene_id.map(query_gene(sqtl_info.gene_id.unique(), self.taxid)['EnsemblRapid']).fillna('')
        try:
            sqtl_info = sqtl_info.merge(query_gene(sqtl_info.gene_id, self.taxid)\
                 .rename({'EnsemblRapid': 'Ensembl_gene'}, axis =1)[['Ensembl_gene']].fillna(''),
                how ='left', left_on='gene_id', right_index=True)
        except:
            printwithlog('previous query gene is missing "EnsemblRapid" getting this information from the "ensembl" column')
            r = query_gene(sqtl_info.gene_id, self.taxid).ensembl\
                  .map(lambda x: x[0] if isinstance(x, list) else x)\
                  .map(lambda x: np.nan if pd.isna(x) else x['gene']).to_frame().set_axis(['Ensembl_gene'], axis = 1)
            r= r.groupby(r.index).agg(lambda x: y[0] if len(y:=x.dropna().unique()) else '')
            sqtl_info = sqtl_info.merge(r, how ='left', left_on='gene_id', right_index=True)
        sqtl_info = sqtl_info.loc[:,  ['trait','SNP_QTL','-log10(P-value)','R2','SNP_sqtldb','tissue', '-log10(pval_nominal)','DP' ,
                                       'Ensembl_gene','gene_id', 'slope' , 'af', 'sQTLtype', 'presence_samples']].rename(lambda x: x.replace('_QTL', ''), axis = 1)
        sqtl_info.SNP = 'chr' + sqtl_info.SNP
        sqtl_info = sqtl_info.drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        sqtl_info.to_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv', index = False)
        return sqtl_info
    
    
    def manhattanplot(self, traitlist: list = [], save_fmt: list = ['html', 'png'], display: bool = True):
        """
        Generate a Manhattan plot for GWAS results.
    
        Steps:
        1. Print a log message indicating the start of the Manhattan plot process.
        2. If no trait list is provided, use all traits.
        3. Iterate over each trait and load the GWAS results.
        4. Combine the GWAS results into a single DataFrame.
        5. Calculate the inverse probability and sort the DataFrame.
        6. Compute the chromosome positions for plotting.
        7. Assign colors to the points based on significance thresholds and chromosome numbers.
        8. Create a scatter plot using Plotly.
        9. Add vertical lines for chromosome boundaries.
        10. Add horizontal lines for significance thresholds.
        11. Update the layout and save the plot in specified formats.
        12. Display the plot if required.
    
        :param traitlist: List of traits to plot.
        :param save_fmt: List of formats to save the plot ('html', 'png').
        :param display: Boolean indicating whether to display the plot.
        :return: Tuple containing the Plotly figure and the GWAS DataFrame.
        """
        printwithlog(f'starting manhattanplot ... {self.project_name}')
        if len(traitlist) == 0: traitlist = self.traits
        for num, t in tqdm(list(enumerate(traitlist))):
            df_gwas,df_date = [], []
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

    def porcupineplot(self, qtltable = '', traitlist: list = [], display_figure = False, skip_manhattan = False, maxtraits = 20, 
                      save = True, color_evens='#87cdea', color_odds='#bfbfbf'):
        """
        Generate an enhanced Porcupine plot for visualizing multiple GWAS traits with additional options.
    
        Steps:
        1. Print a log message indicating the start of the Porcupine plot v2 process.
        2. Load the QTL table if not provided.
        3. Determine the list of traits to plot.
        4. Set color mappings for traits.
        5. Load and combine the GWAS results for each trait.
        6. Calculate the chromosome positions for plotting.
        7. Assign colors to the points based on significance thresholds and chromosome numbers.
        8. Create a scatter plot using Holoviews and Datashader.
        9. Add annotations for QTLs if specified.
        10. Save the plot in specified formats.
        11. Display the plot if required.
    
        :param qtltable: Path to the QTL table or DataFrame containing QTL information.
        :param traitlist: List of traits to plot.
        :param display_figure: Boolean indicating whether to display the plot.
        :param skip_manhattan: Boolean indicating whether to skip the Manhattan plot.
        :param maxtraits: Maximum number of traits to plot.
        :return: Holoviews plot object.
        """
        printwithlog('starting porcupine plot')
        hv.opts.defaults(hv.opts.Points(width=1200, height=600), hv.opts.RGB(width=1200, height=600) )
        if type(qtltable) == str:
            if not len(qtltable): qtltable = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                                               .reset_index()\
                                               .drop(['index' ] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)\
                                               .query('QTL == True')
        if not len(traitlist): traitlist = list(map(lambda x:x.replace('regressedlr_', ''),self.traits))
        else: traitlist = [x.replace('regressedlr_', '') for x in traitlist]
        qtltable = qtltable[qtltable.trait.isin(traitlist)]
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
                if len(df_gwas) == 0:  
                    printwithlog(f'could not open mlma files for {t}')
                    continue
                df_gwas = pd.concat(df_gwas)
                append_position = df_gwas.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
                qtltable['x'] = qtltable.apply(lambda x: x.bp +  append_position[x.Chr], axis = 1)
                df_gwas['-log10p'] = -np.log10(df_gwas.p)
                df_gwas.drop(['A1', 'A2', 'Freq', 'b', 'se', 'p'], axis = 1, inplace = True)
                def mapcolor(c): 
                    if int(str(c).replace('X',str(self.n_autosome+1)).replace('Y', str(self.n_autosome+2)).replace('MT', str(self.n_autosome+4)))%2 == 0: return color_evens
                    return color_odds
                df_gwas = df_gwas.groupby('Chr') \
                                 .apply(lambda df: df.assign(color = mapcolor(df.Chr[0]), x = df.bp + append_position[df.Chr[0]])) \
                                 .reset_index(drop = True)
                df_gwas.loc[df_gwas['-log10p'] > self.threshold, 'color' ] = str(d[t])[1:-1]
                df_gwas.loc[df_gwas['-log10p'] > self.threshold, 'color' ] = df_gwas.loc[df_gwas['-log10p']> self.threshold, 'color' ].str.split(',').map(lambda x: tuple(map(float, x)))
                if not skip_manhattan:
                    yrange = (-.05,max(6, df_gwas['-log10p'].max()+.5))
                    xrange = tuple(df_gwas.x.agg(['min', 'max'])+ np.array([-1e7,+1e7]))
                    fig = []
                    for idx, dfs in df_gwas[df_gwas.color.isin([color_odds, color_evens])].groupby('color'):
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
        for idx, dfs in fdf[fdf.color.isin([color_odds, color_evens])].groupby('color'):
            temp = datashade(hv.Points(dfs, kdims = ['x','-log10p']), pixel_ratio = 2, aggregator = ds.count(), width = 1200,height = 600, y_range = yrange,
                     min_alpha=.7, cmap = [idx], dynamic = False )
            temp = dynspread(temp, max_px=4,threshold= 1 )
            fig += [temp]
        fig = fig[0]*fig[1]
        
        fig = fig*hv.HLine((self.threshold05)).opts(color='blue')
        fig = fig*hv.HLine(self.threshold).opts(color='red')
        
        for idx, dfs in fdf[~fdf.color.isin([color_odds, color_evens])].groupby('color'):
            fig = fig*hv.Points(dfs.drop('color', axis = 1), kdims = ['x','-log10p']).opts(color = idx, size = 5)
        
        for t, dfs in qtltable.groupby('trait'):
            fig = fig*hv.Points(dfs.assign(**{'-log10p': qtltable.p}), kdims = ['x','-log10p'],vdims=[ 'trait','SNP' ,'A1','A2','Freq' ,'b','traitnum'], label = f'({tnum[t]}) {t}' ) \
                                          .opts(size = 17, color = d[t], marker='inverted_triangle', line_color = 'black', tools=['hover']) #
        fig = fig*hv.Labels(qtltable.rename({'p':'-log10p'}, axis = 1)[['x', '-log10p', 'traitnum']], 
                            ['x','-log10p'],vdims=['traitnum']).opts(text_font_size='5pt', text_color='black')
        fig.opts(xticks=[((dfs.x.agg(['min', 'max'])).sum()//2 , self.replacenumstoXYMT(names)) for names, dfs in fdf.groupby('Chr')],
                                   xlim =xrange, ylim=yrange, xlabel='Chromosome', shared_axes=False,
                               width=1200, height=600, title = f'porcupineplot',legend_position='right',show_legend=True)
        if save: hv.save(fig, f'{self.path}images/porcupineplot.png')
        if display_figure: 
            display(fig)
            return
        return fig

    def GWAS_latent_space(self, traitlist: list = [], method: str = 'nmf'):
        """
        Generate a latent space representation of GWAS results using various dimensionality reduction methods.
    
        Steps:
        1. Print a log message indicating the start of the GWAS latent space process.
        2. Set up the plotting environment and color mappings for traits.
        3. Determine the list of traits to process.
        4. Load and combine GWAS results for each trait.
        5. Apply the specified dimensionality reduction method (e.g., PCA, NMF, ICA).
        6. Transform the combined data into the latent space.
        7. Normalize and adjust the latent space components if necessary.
        8. Generate a heatmap to visualize the importance of traits in the latent space.
        9. Merge the latent space components with the GWAS results.
        10. Calculate chromosome positions for plotting.
        11. Assign colors to points based on significance thresholds and chromosome numbers.
        12. Create a scatter plot using Holoviews and Datashader.
        13. Annotate significant QTLs if identified.
        14. Save the plot and annotated QTLs in a deliverable format.
        15. Return the deliverable as a Panel Card object.
    
        :param traitlist: List of traits to include in the analysis.
        :param method: Dimensionality reduction method to use ('nmf', 'pca', 'ica', etc.).
        :return: Tuple containing the Panel Card object and a dictionary of deliverables.
        """
        printwithlog('starting GWAS latent space...')
        pn.extension('tabulator')
        from sklearn.cluster import SpectralCoclustering
        pref = {'pca': 'PC', 'nmf':'NMF', 'spca': 'sPC', 'fa': 'FA', 'ica': 'ic', 'da': 'da'}
        if not len(traitlist): 
            traitlist = [x for x in self.traits if not len(re.findall(r'regressedlr_pc[123]|regressedlr_umap[123]|regressedlr_umap_clusters_\d+|regressedlr_pca_clusters_\d+',x))]
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
        model = SpectralCoclustering(n_clusters=npc).fit(pcadata)
        pcadata = pcadata.iloc[np.argsort(model.row_labels_),np.argsort(model.column_labels_) ]
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
            oo = self.annotate(oo, save= False)
            send2d = oo.drop(['p', 'color', 'x', 'trait_description', 'Chr', 'bp'], axis = 1).rename({'realp': '-log10p'}, axis = 1)
            if 'gene' in send2d: send2d= send2d.loc[:,:'gene']
            deliverable['qtls'] = fancy_display(send2d, download_name='qtls_from_latent_space.csv')
        
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
        description = f'''{method.upper()} decompostion of the GWAS pvalues for all traits. The concept is to find the shared SNPs that influence the GWAS for multiple traits.
The decompositions used also allow to extimate a metric of similarity between the traits used. This model follows the paper: 
[Principal and independent genomic components of brain structure and function]( https://doi.org/10.1111/gbb.12876) but with a key change, we are using P-values while the original paper uses the GWAS z-scored betas'''
        deliverablef = pn.Card(description, *list(deliverable.values())[::-1], title = 'GWAS Latent Space', collapsed = True)#, width=1000
        return deliverablef, deliverable

    # def annotate(self, qtltable: pd.DataFrame(),
    #              snpcol: str = 'SNP', save: bool = False, **kwards) -> pd.DataFrame():
        
    #     '''
    #     This function annotates a QTL table with the snpEff tool, 
    #     which is used to query annotations for QTL and phewas results. 
                
    #     Parameters 
    #     ----------

    #     qtltable: pd.DataFrame
    #         the QTL table to be annotated
    #     genome: str = 'rn7'
    #         the genome to use for annotation (default is 'rn7')
    #     snpcol: str = 'SNP'
    #         the column name for the SNP column in the QTL table (default is 'SNP')
    #     save: bool = True
    #         a boolean indicating whether to save the annotated table to a file (default is True)
            
    #     Design
    #     ------

    #     The function first defines a dictionary that maps the genome input to the corresponding genome version to use in snpEff.
    #     It then creates a temporary VCF file from the QTL table by resetting the index, selecting specific columns, 
    #     and assigning certain values to other columns. The VCF file is then passed through the snpEff tool
    #     and the results are parsed into a new DataFrame.  If save is True, the final annotated table is saved to a file. 
    #     The function returns the final annotated table.
    #     '''
    #     if not (isinstance(qtltable, pd.DataFrame) and len(qtltable)): 
    #         printwithlog('dataframe is empty, returning same dataframe')
    #         return qtltable #qtltable = pd.read_csv(self.allqtlspath).set_index('SNP')
    #     d = {'rn6': 'Rnor_6.0.99', 'rn7':'mRatBN7.2.105', 'cfw': 'GRCm38.99','m38': 'GRCm38.99'}[self.genome]
    #     #bash('java -jar snpEff/snpEff.jar download -v Rnor_6.0.99')
    #     #bash('java -jar snpEff/snpEff.jar download -v mRatBN7.2.105')
    #     #bash('java -jar snpEff/snpEff.jar download -v GRCm39.105') 
    #     #bash('java -jar snpEff/snpEff.jar download -v GRCm38.99') 
    #     qtltable['Chr'] = qtltable['Chr'].map(self.replacenumstoXYMT).map(str.upper)
    #     temp  = qtltable.reset_index()\
    #                     .loc[:,[ 'Chr', 'bp', snpcol, 'A2', 'A1']]\
    #                     .assign(QUAL = 40, FILTER = 'PASS' ,INFO = '', FORMAT = 'GT:GQ:DP:HQ')
    #     temp.columns = ["##CHROM","POS","ID","REF","ALT", 'QUAL', 'FILTER', 'INFO', 'FORMAT']
    #     temp['##CHROM'] = 'chr'+ temp['##CHROM'].astype(str)
    #     vcf_manipulation.pandas2vcf(temp, f'{self.path}temp/test.vcf', metadata='')
    #     #a = bash(f'java -Xmx8g -jar {self.snpeff_path}snpEff.jar {d} -noStats {self.path}temp/test.vcf', print_call = False )# 'snpefftest',  -no-intergenic -no-intron
    #     a = bash(f'$CONDA_PREFIX/share/snpeff-5.2-0/snpEff -Xmx8g {d} -noStats {self.path}temp/test.vcf', shell = True, silent = True, print_call = False )
    #     #a = subprocess.run(f'$CONDA_PREFIX/share/snpeff-5.2-0/snpEff -Xmx8g {d} -noStats {self.path}temp/test.vcf', capture_output = True, shell = True).stdout.decode('ascii').strip().split('\n') 
    #     res = pd.read_csv(StringIO('\n'.join(a)),  comment='#',  sep =r'\s+', 
    #                       header=None, names = temp.columns,  dtype=str).query('INFO != "skipping"')  
    #     ann = res['INFO'].str.replace('ANN=', '').str.split('|',expand=True)
    #     column_dictionary = defaultdict(lambda: 'UNK', {k:v for k,v in enumerate(['alt_temp', 'annotation', 'putative_impact', 'gene', 'geneid', 'featuretype', 'featureid', 'transcriptbiotype',
    #                       'rank', 'HGVS.c', 'HGVS.p', 'cDNA_position|cDNA_len', 'CDS_position|CDS_len', 'Protein_position|Protein_len',
    #                       'distancetofeature', 'errors'])})
    #     ann = ann.rename(column_dictionary, axis = 1)
    #     ann.index = qtltable.index
    #     out = pd.concat([qtltable.loc[:,~qtltable.columns.isin(ann.columns)], ann], axis = 1).replace('', np.nan).dropna(how = 'all', axis = 1).drop('alt_temp', axis = 1, errors ='ignore')
        
    #     if 'geneid' in out.columns:
    #         species = translate_dict(self.genome, {'rn7': 'rat', 'rn8':'rat', 'm38':'mouse', 'rn6': 'rat'})
    #         gene_translation = {x['query']: x['symbol'] for x in mg.querymany(('-'.join(out.geneid)).split('-') ,\
    #                        scopes='ensembl.gene,symbol,RGD', fields='symbol', species=self.taxid, verbose = False, silent = True)  if 'symbol' in x.keys()}
    #         if gene_translation: out['gene'] = out.geneid.map(lambda x: translate_dict(x, gene_translation))
        
    #     if 'errors' in out.columns:  out = out.loc[:, :'errors']
    #     try: 
    #         out['Chr'] = out['Chr'].map(self.replaceXYMTtonums)
    #     except:
    #         print('Chr not in columns, returning with possible errors')
    #         return out
    #     if save:
    #         self.annotatedtablepath = f'{self.path}results/qtls/finalqtlannotated.csv'
    #         out.reset_index().to_csv(self.annotatedtablepath, index= False) 
    #         #out.reset_index().to_csv(f'{self.path}results/qtls/finalqtl.tsv', index= False, sep = '\t')
        
    #     return out 

    def annotate(self, qtltable: pd.DataFrame, snpcol:str = 'SNP',  refcol:str = 'A2',
                altcol:str = 'A1', save: bool = False, adjustchr =False, silent_annotation = False,
                 vep_distance:int = 30000,vep_buffer:int = 1000000, **kwards) -> pd.DataFrame:
        if len(qtltable) == 0: 
            printwithlog('dataframe is empty, returning same dataframe')
            return qtltable 
        qtltable = qtltable.reset_index()
        if not {'Chr', 'bp'}.issubset(qtltable.columns): 
            qtltable[['Chr', 'bp']] = qtltable['SNP'].str.split(':').to_list()
            qtltable.bp = qtltable.bp.astype(int)
        qtltable['Chr'] = qtltable['Chr'].map(self.replaceXYMTtonums)
        qtltable = qtltable.sort_values(['Chr', 'bp'])
        temp  = qtltable.loc[:,[ 'Chr', 'bp', snpcol, refcol, altcol]]\
                        .assign(QUAL = 40, FILTER = 'PASS' ,INFO = '', FORMAT = 'GT:GQ:DP:HQ')\
                        .set_axis(["##CHROM","POS","ID","REF","ALT", 'QUAL', 'FILTER', 'INFO', 'FORMAT'], axis = 1)
        vcf_manipulation.pandas2vcf(temp, f'{self.path}temp/test.vcf', metadata='')
        vdir = bash('echo $CONDA_PREFIX/share/', shell = True, silent=True, print_call=False)[0]
        vdir = glob(f'{vdir}ensembl-vep*')[0] 
        if not hasattr(self, 'gtf_path') or not hasattr(self, 'genomefasta_path'): 
            self.pull_NCBI_genome_info(self.genome_accession, redownload = False)
        gfffile = self.gtf_path.replace('.gtf', f"{'_adjusted' if adjustchr else ''}.gff.gz")
        fafile = self.genomefasta_path.replace('.fna',f"{'_adjusted' if adjustchr else ''}.fna")
        translate_names_dict = {'SYMBOL': 'gene', 'Feature': 'featureid', 'biotype': 'transcriptbiotype', 'Feature_type': 'featuretype',
                                'IMPACT': 'putative_impact','DISTANCE':'distancetofeature', 'HGVSc': 'HGVS.c', 'HGVSp': 'HGVS.p', 'SNP': 'SNP' }
        oo = bash(f'''vep -i {self.path}temp/test.vcf -o STDOUT --gff {gfffile} --species {self.species} --warning_file STDERR \
                              --synonyms {self.chrsyn} -a {self.genome_accession} --no_check_variants_order \
                              --dir {vdir} --dir_cache {vdir} --dir_plugins {vdir} --fasta {fafile} --tab \
                              --regulatory --force_overwrite --domains --per_gene \
                              --appris --mane --biotype --buffer_size {int(vep_buffer)} --hgvs \
                              --distance {int(vep_distance)} --show_ref_allele --sift b \
                              --symbol --transcript_version --tsl --uploaded_allele --refseq \
                              ''', shell = True, print_call=False, silent = silent_annotation) #--sf {self.path}temp/test.html
        oo = pd.read_csv(StringIO(('\n'.join(oo)).replace('-\t','nan\t').replace('\t-','\t-')), sep= '\t', comment='##', engine = 'python').iloc[:, :-1]\
                       .rename(lambda x: x.replace('#Uploaded_variation', 'SNP'), axis = 1)\
                       .drop(['Location','Allele','Gene', 'UPLOADED_ALLELE'], axis = 1)\
                       .replace('-', np.nan).replace('genomic.gff.gz', f'{self.genome_accession}:refseq:gff')\
                       .rename(lambda x: x.replace('_', '').lower() if x not in translate_names_dict else translate_dict(x,translate_names_dict ), axis = 1)
        oo['distancetofeature'] = oo['distancetofeature'].fillna(0)
        oo['putative_impact_inv'] = oo.putative_impact.map(defaultdict(lambda:100, {'HIGH':0,'MODERATE': 1, 'LOW': 2 ,'MODIFIER':3}))
        oo = oo.sort_values(['putative_impact_inv', 'distancetofeature'])\
               .reset_index(drop = True)\
               .drop_duplicates(subset = ['SNP', 'refallele'], keep = 'first')\
               .drop('putative_impact_inv',axis = 1)
        oo.distancetofeature = oo.distancetofeature.astype(int).map(bp2str)
        if oo.columns.str.contains('position$').any() and len(oo):
            poscols = list(oo.columns[oo.columns.str.contains('position$')])
            oo['position_'+ '|'.join(map(lambda x:x.replace('position', ''), poscols))] = oo[poscols].astype(str).apply(lambda c: c.str.cat(sep='|').replace('nan', 'NA'), axis = 1)
            oo = oo.drop(poscols, axis = 1)
        oo = oo.T.dropna(how = 'all').T
        if 'gene' not in oo.columns: oo = oo.assign(gene = np.nan)
            #oo.loc[:, 'gene'] = np.nan
        oo = qtltable.merge(oo.rename({'SNP': snpcol}, axis = 1), on = snpcol, how = 'left')\
                     .drop(['index'] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
        if save:
            self.annotatedtablepath = f'{self.path}results/qtls/finalqtlannotated.csv'
            oo.reset_index().to_csv(self.annotatedtablepath, index= False) 
        return oo

    def qqplot(self, col2use: str = 'trait', add_pval_thresh: bool = True, save: bool = True, contour: bool = True):
        """
        Generate Q-Q and volcano plots for GWAS results.
    
        Steps:
        1. Import necessary libraries and initialize the Dask client.
        2. Load GWAS results from MLMA files and calculate -log10(p-values).
        3. Group GWAS results by trait and calculate ranks.
        4. Merge QTL annotations with the GWAS results.
        5. Define a function to read and sample data from specified paths.
        6. Load and sample data for p-value thresholds if available.
        7. Determine the maximum beta value for the volcano plot.
        8. Set up color mappings for traits.
        9. Initialize the plot ranges.
        10. Create base plots with horizontal lines for p-value thresholds.
        11. Add contour or datashaded points to the plots if p-value thresholds are provided.
        12. Add slope and vertical lines to the plots.
        13. Annotate QTL points on the plots.
        14. Add labels to the plots.
        15. Add confidence intervals to the Q-Q plot if p-value thresholds are provided.
        16. Customize and display the plots.
        17. Save the plots as PNG and HTML files if specified.
        18. Return the combined figure.
    
        :param col2use: Column to use for grouping and labeling (default is 'trait').
        :param add_pval_thresh: Boolean indicating whether to add p-value thresholds (default is True).
        :param save: Boolean indicating whether to save the results (default is True).
        :param contour: Boolean indicating whether to add contour plots (default is True).
        :return: Combined figure containing the Q-Q and volcano plots.
        """
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
        """
        Create a 3D scatter plot of PCA results with HDBSCAN clustering.
    
        Steps:
        1. Define a lambda function to filter out specific columns.
        2. Select columns for the 3D plot and drop rows with all missing values.
        3. Perform PCA on the selected columns using a pipeline with KNN imputation.
        4. Store PCA components and explained variance ratios.
        5. Apply HDBSCAN clustering on the first three principal components.
        6. Return the PCA and clustering results as a DataFrame if retret is set to 'data'.
        7. Create a 3D scatter plot using Plotly, color-coded by HDBSCAN clusters.
        8. Add vectors representing the PCA components to the 3D plot.
        9. Update the layout of the 3D plot.
        10. Return the 3D scatter plot figure.
    
        :param ret: Determines whether to return the figure ('figure') or data ('data'). Defaults to 'figure'.
        :return: 3D scatter plot figure or DataFrame with PCA and clustering results.
        """
        rmcols = lambda x: x not in ['regressedlr_pc1', 'regressedlr_pc2', 'regressedlr_pc3', 'regressedlr_umap1', 'regressedlr_umap2', 
                                     'regressedlr_umap3','regressedlr_pca_clusters', 'regressedlr_umap_clusters']  and  '_just_' not in x
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

    def andrews_curves_plot(self,traits: list = None):
        """
        Create an Andrews Curves plot using the specified traits.
    
        Steps:
        1. Import the necessary function for Andrews Curves plotting.
        2. If no traits are provided, use all traits.
        3. Ensure the traits have the correct prefix.
        4. Perform HDBSCAN clustering on the traits.
        5. Create and return the Andrews Curves plot with clusters.
    
        :param traits: List of traits to include in the plot. Defaults to None, which uses all traits.
        :return: Andrews Curves plot.
        """
        from hvplot import andrews_curves
        if traits is None: traits = self.traits
        else: 
            traits = list(map(lambda x: 'regressedlr_' + x  if ('regressedlr' not in x) else x , traits))
        aa = self.df[self.traits]   
        aa['hdbscan'] = 'class_'+ pd.Series(HDBSCAN().fit_predict(aa)).map(lambda x: x+1 if x>= 0 else 'noclass').astype(str).astype(str)
        return andrews_curves(aa, class_column='hdbscan').opts(width = 1000, height = 800)

    def make_genes_in_range_mk_table(self, path: str = None):
        """
        Create a markdown table of genes in range from a specified path.
    
        Steps:
        1. Check if a path is provided, otherwise use a default path.
        2. Load the genes in range from the specified path.
        3. Retrieve GTF data if not already available.
        4. Define functions to generate markdown links for various gene resources.
        5. Merge gene annotation data with the genes in range data.
        6. Create links for additional gene resources.
        7. Filter out uncharacterized and irrelevant genes.
        8. Calculate the distance from each gene to its associated SNP.
        9. Return a DataFrame with the relevant gene information and links.
    
        :param path: Path to the file containing genes in range. Defaults to None.
        :return: DataFrame containing gene information and links.
        """
        if path is None: 
            if not os.path.isfile(f"{self.path}results/qtls/genes_in_range.csv"): self.locuszoom()
            genes_in_range = pd.read_csv(f"{self.path}results/qtls/genes_in_range.csv")
        else: genes_in_range = pd.read_csv(path)
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        
        #_genecardmk = lambda gene:f'[genecard](https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene})' if not pd.isna(gene) else ''
        _genecardmk = lambda gene: f'''<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene}" target="_blank">🔗</a>''' if not pd.isna(gene) else ''
        # _gwashubmk = lambda gene : f'[gwashub](https://www.ebi.ac.uk/gwas/genes/{gene})' if not pd.isna(gene) else ''
        _gwashubmk = lambda gene : f'''<a href="https://www.ebi.ac.uk/gwas/genes/{gene}" target="_blank">🔗</a>''' if not pd.isna(gene) else ''
        # _twashubmk = lambda gene : f'[twashub](http://twas-hub.org/genes/{gene.upper()})' if not pd.isna(gene) else ''
        _twashubmk = lambda gene :  f'''<a href="http://twas-hub.org/genes/{gene.upper()}" target="_blank">🔗</a>'''  if not pd.isna(gene) else ''
        # _genecupmk = lambda gene : f'[genecup](https://genecup.org/progress?type=brain&type=addiction&type=drug&type=function&type=psychiatric&type=cell&type=stress&type=GWAS&query={gene})'\
        #                            if not pd.isna(gene) else ''
        _genecupmk = lambda gene : f'''<a href="https://genecup.org/progress?type=brain&type=addiction&type=drug&type=function&type=psychiatric&type=cell&type=stress&type=GWAS&query={gene})" target="_blank">🔗</a>''' \
                                   if not pd.isna(gene) else ''
        # _genebassmk = lambda ensid: f'[genebass](https://app.genebass.org/gene/{ensid}?burdenSet=pLoF&phewasOpts=1&resultLayout=full)' if not pd.isna(ensid) else ''
        _genebassmk = lambda ensid: f'''<a href="https://app.genebass.org/gene/{ensid}?burdenSet=pLoF&phewasOpts=1&resultLayout=full" target="_blank">🔗</a>''' if not pd.isna(ensid) else ''
        # _rgdhtmk = lambda gene: f'[rgd](https://rgd.mcw.edu/rgdweb/report/gene/main.html?id={gene.split(":")[-1]}){:target="_blank"}' if not pd.isna(gene) else ''
        _rgdhtmk = lambda gene: f'''<a href="https://rgd.mcw.edu/rgdweb/report/gene/main.html?id={gene.split(":")[-1]}" target="_blank">🔗</a>''' if not pd.isna(gene) else ''
        def tryloc(x):
            try: return f"{x['chr']}:{x['start']}-{x['end']}"
            except: return 'remove'
        def distance2snp(snp, gene):
            s = str(snp).split(':')
            g = str(gene).split(':')
            if self.replaceXYMTtonums(s[0]) != self.replaceXYMTtonums(g[0]) : return 'remove'
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
        #hgenes = hgenes.map(_genebassmk)
        genes_in_range = genes_in_range.merge(hgenes.rename('ensemblh'), left_on = 'symbol', right_on = 'query', how = 'left')
        #genes_in_range['ensemblh'] =genes_in_range['ensemblh'].fillna('')
        list_of_cols =  [('genecard','symbol' ,_genecardmk), ('gwashub', 'symbol' ,_gwashubmk), ('genecup', 'symbol', _genecupmk), 
                       ('twashub', 'symbol' ,_twashubmk), ('genebass', 'ensemblh' ,_genebassmk)]
        if self.species == 'rattus_norvegicus': list_of_cols += [('RGD','AllianceGenome', _rgdhtmk )]
        for idx,_col,fu in list_of_cols:
            genes_in_range[idx] = genes_in_range[_col].map(fu)
        genes_in_range = genes_in_range.drop('ensemblh', axis = 1)
        genes_in_range = genes_in_range.loc[~genes_in_range.name.fillna('').str.contains('uncharacterized LOC|^Trna')]
        genes_in_range = genes_in_range.loc[~(genes_in_range.symbol.fillna('').str.contains('uncharacterized LOC|^Trna') & genes_in_range.name.isna())]
        genes_in_range = genes_in_range.loc[:, ~genes_in_range.columns.str.contains('Unnamed:')]#.drop_duplicates()
        genes_in_range.head()
        genes_in_range['distance'] = genes_in_range.apply(lambda x: distance2snp(x.SNP_origin, x.genomic_pos), axis = 1)
        genes_in_range = genes_in_range.set_index('SNP_origin')\
                      .dropna(subset = ['name'])\
                      .loc[:, ['genomic_pos','distance','symbol', 'name','AllianceGenome','ensembl','entrezgene'] \
                              + [x[0] for x in list_of_cols]]
        genes_in_range = genes_in_range.reset_index().set_index(['SNP_origin', 'genomic_pos', 'distance', 'symbol'])
        dups = genes_in_range.index.duplicated()
        genes_in_range = genes_in_range[~dups].combine_first(genes_in_range[dups]).reset_index().set_index('SNP_origin')
        return genes_in_range

    def report(self, round_version: str = '10.5.2', covariate_explained_var_threshold: float = 0.02, gwas_version: str =None, add_gpt_query: bool = False,
               sorted_gcorr: bool = True, add_gwas_latent_space: str = 'nmf', add_experimental: bool = False, remove_umap_traits_faq: bool = True, 
               qqplot_add_pval_thresh: bool = True, add_qqplot:bool = True, add_cluster_color_heritability: bool = False,
               legacy_locuszoom: bool = True, static: bool = True, traits: bool = None, remove_missing_animals_section: bool = True, headername: str = ''):
        """
        Generate a comprehensive GWAS report.
    
        Steps:
        1. Print a log message indicating the start of the report generation process.
        2. Load and parse the genotype parameter thresholds and log information.
        3. Format the thresholds and genotype information for the report sidebar.
        4. Create a Bootstrap template for the report.
        5. Add the phenotype and genotype information to the sidebar.
        6. Load and display the data dictionary and missing RFID list.
        7. Load and display the explained variances and covariate information.
        8. Load and display the genetic correlation matrix.
        9. Load and display the SNP heritability estimates.
        10. Load and display the summary of QTLs.
        11. Load and display the porcupine plot and QQ plot.
        12. Load and display the Manhattan plots for significant QTLs.
        13. Load and display the regional association plots.
        14. Add experimental sections such as gene enrichment and project graph view.
        15. Load and display the list of traits included in the PheWAS database.
        16. Add references and save the report to an HTML file.
    
        :param round_version: Version of the round for which the report is generated.
        :param covariate_explained_var_threshold: Threshold for the explained variance of covariates.
        :param gwas_version: Version of the GWAS pipeline used.
        :param sorted_gcorr: Boolean indicating whether to sort the genetic correlation matrix.
        :param add_gwas_latent_space: Method to use for adding GWAS latent space ('none' to skip).
        :param add_experimental: Boolean indicating whether to add experimental sections to the report.
        :return: None
        """
        if gwas_version is None: gwas_version = __version__
        if traits is None: 
            traits = self.traits
            redo_figs = False
        else: 
            traits = ['regressedlr_'+x.replace('regressedlr_', '') for x in traits]
            redo_figs = True
        traitfilter = list(map(lambda x:x.replace('regressedlr_', ''),traits))         
        printwithlog('generating report...')
        printwithlog('generating report... making header...')
        with open(f'{self.path}genotypes/parameter_thresholds.txt', 'r') as f: 
            out = f.read()
            params = {x:re.findall(f"--{x} ([^\n]+)", out)[0] for x in ['geno', 'maf', 'hwe']}
        with open(f'{self.path}genotypes/genotypes.log') as f:
            out = f.read()
            params['snpsb4'] = re.findall(r"(\d+) variants loaded from .bim file.", out)[0]
            params['snpsafter'], params['nrats'] = re.findall(r"(\d+) variants and (\d+) samples pass filters and QC.", out)[0]
            params['removed_geno'], params['removedmaf'], params['removedhwe'] = \
                   (~pd.read_parquet(f'{self.path}genotypes/snpquality.parquet.gz')[['PASS_MISS','PASS_MAF','PASS_HWE']])\
                   .sum().astype(str)

        if round(self.threshold, 2) == round(self.threshold05, 2):
            threshtext = f'''* {self.genome_version}:{round_version} 5%: {round(self.threshold, 2)}'''
        else:
            threshtext = f'''* {self.genome_version}:{round_version} 10%: {round(self.threshold, 2)}
* {self.genome_version}:{round_version} 5% : {round(self.threshold05, 2)}'''
                    
        text_sidepanel = f"""# General Information


* Generated on {datetime.today().strftime('%Y-%m-%d')}
* Pipeline version *{gwas_version}*
<hr>

Phenotype Info

* n = *{params['nrats']}*
 
<hr>

Genotype Info

* genotypes version: \n*{self.genome_version}:{round_version}*

* number of snps: \nbefore filter *{format(int(params['snpsb4']), ',')}*, \nafter filter *{format(int(params['snpsafter']), ',')}*

* genotype missing rate filter: < *{params['geno']}* \n(*{format(int( params['removed_geno']),',')}* snps removed)

* minor allele frequency filter: > *{params['maf']}* \n(*{format(int(params['removedmaf']), ',')}* snps removed)

* hardy-weinberg equilibrium filter: < *{params['hwe']}* \n(*{format(int(params['removedhwe']), ',')}* snps removed)
<hr>

Threshold Info

{threshtext}

"""
        #* phenotype statistical descriptions file: [🔗](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/data_distributions.html)
        # '''* phenotype data: [🔗](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/processed_data_ready.csv)

        # * covariate dropboxes: [🔗](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/data_dict_{self.project_name}.csv) '''
        # favicon = icon_path.read_bytes()
        # favicon = base64.b64encode(favicon).decode("ascii")
        # favicon = f"data:image/x-icon;base64,{favicon}"
        template = pn.template.BootstrapTemplate(title=f'GWAS REPORT', favicon = icon_path)
        os.makedirs(f'{self.path}images/report_pieces/',exist_ok=True )
        os.makedirs(f'{self.path}images/report_pieces/regional_assoc/',exist_ok=True )
        add_metadata = lambda x: pn.Column(x, pn.pane.Alert(text_sidepanel, alert_type="primary"))
        #              f"gwas_report_{self.project_name}_round{round_version}_threshold{round(self.threshold,2)}_n{self.df.shape[0]}_date{datetime.today().strftime('%Y-%m-%d')}_gwasversion_{gwas_version}")
        # Add components to the sidebar, main, and header
        template.sidebar.extend([pn.pane.Alert(text_sidepanel, alert_type="primary")])
        ##### adding data dictionary
        dd = pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv').fillna('')\
           [['measure', 'trait_covariate','covariates', 'description']]\
           .query("measure != ''")
        dd =dd[~dd.trait_covariate.isin(['', np.nan])]
        if redo_figs:
            dd = dd[dd.measure.isin(traitfilter) | ~dd.trait_covariate.str.contains('^trait$')]
            dd_covs_covcol = (set(dd.covariates.replace('', np.nan).str.strip(',').dropna().str.split(',').sum()) )
            dd = dd[dd.measure.isin(dd_covs_covcol) | ~dd.trait_covariate.str.contains('covariate')]

        trait_d_card = ['Collaborative data dictionary google document: ', fancy_display(dd, download_name= 'data_dictionary.csv',flexible = True)]
        if os.path.isfile(f'{self.path}missing_rfid_list.txt') and not remove_missing_animals_section: 
            trait_d_card += [pn.Card(fancy_display(pd.read_csv(f'{self.path}missing_rfid_list.txt', dtype = str, header = None, names=['not genotyped RFIDs']), 
                                                   download_name= 'missing_rats.csv', flexible = True), \
                                     title = 'not genotyped samples', collapsed=True)]
        append2 = pn.Card(*trait_d_card , title = 'Trait Descriptions', collapsed=True)
        add_metadata(append2).save(f'{self.path}images/report_pieces/trait_descriptions.html')
        template.main.append(append2)
        
        #explained_vars =  pd.read_csv(f'{self.path}melted_explained_variances.csv').pivot(columns = 'group', values='value', index = 'variable')

        printwithlog('generating report... making covariates section...')
        for col in self.df.filter(regex = 'regressedlr').columns:
            #printwithlog(f'''warning: {col} doesn't have a non regressedout version''')
            if col.replace('regressedlr_', '') not in self.df.columns: self.df[col.replace('regressedlr_', '')] = self.df[col]
        g0 = px.imshow(self.df.set_index('rfid')[traits].rename(lambda x: x.replace('regressedlr_', ''), axis = 1), aspect = 3, color_continuous_scale='RdBu')
        g0.update_layout( width=1000, height=1400,autosize=False,showlegend=False, template='simple_white',plot_bgcolor='black')
        g1df = self.df.set_index('rfid')[list(map(lambda x: x.replace('regressedlr_', ''), traits))]
        g1df.loc[:, :] = StandardScaler().fit_transform(g1df)
        g1 = px.imshow(g1df, aspect = 3, color_continuous_scale='RdBu')
        g1.update_layout(  width=1000, height=1400,autosize=False,showlegend=False, template='simple_white',plot_bgcolor='black')
        
        covariates_list = [f'''Covariates may confound the results of the analysis. Common covariates include “age”, “weight”, “coat color”, “cohort”, and “phenotyping center”. We work with individual PIs to determine which covariates should be considered. In order to “regress out” the part of the phenotypic variance that is related to known covariates, we follow the procedure of fitting a linear model that predicts the desired trait based only on the measured covariates. Then the trait is subtracted by the trait predictions generated by the linear model described above. The resulting subtraction is expected to be independent of the covariates as all the effects caused by those covariates were removed. Since this method utilizes a linear regression to remove those effects, non-linear effects of those covariates onto the traits will not be addressed and assumed to be null. In certain cases, it’s possible that accounting for too many covariates might ‘overcorrect’ the trait. To address this issue, we ‘regress out’ only the covariates that explain more than {covariate_explained_var_threshold} of the variance of the trait. This calculation is often called r^2 or pve (percent explained variance) and is estimated as cov (covariant, trait)/variance(trait). Lastly, the corrected trait is quantile normalized again, as it’s expected to follow a normal distribution. For time series regression we use the prophet package (https://facebook.github.io/prophet/) that uses a generalized additive model to decompose the timewise trend effects and covariates onto the mesurement of animal given its age. Because age might affect differently males and females, we first groupby the animals between genders before using the timeseries regression to remove covariate effects. After removing the covariate effects in with the timeseries regression, we then quantile normalize the residuals to be used for subsequent analysis.'''] 
        if os.path.isfile(f'{self.path}melted_explained_variances.csv'):
            explained_vars = pd.read_csv(f'{self.path}melted_explained_variances.csv')\
                               .rename({'group':'covariate', 'variable': 'trait'}, axis = 1)\
                               .pivot(columns = 'covariate', values='value', index = 'trait')
            for x in set(map(lambda x:  x.replace('regressedlr_', ''), traits)) - set(explained_vars.index.values):
                if len(explained_vars.columns) > 0: explained_vars.loc[x, :] = np.nan
            #explained_vars = explained_vars[explained_vars.index.isin(traitfilter)]
            explained_vars = explained_vars[explained_vars.index.str.contains('|'.join('^'+pd.Series(traitfilter)))]
            fig_exp_vars = explained_vars.dropna(how = 'all').hvplot.heatmap(frame_width = 600, frame_height =600, 
                                                                 cmap= 'reds',line_width =.1, line_color ='black')\
                              .opts(xrotation =45, yrotation = 45)
            fig_exp_vars *= hv.Labels((explained_vars*100).dropna(how = 'all').round(0).fillna(0).astype(int).reset_index().melt(id_vars= 'trait'), 
                                      kdims= [ 'covariate', 'trait'], vdims=['value']).opts(text_color = 'black')
            hv.save(fig_exp_vars, f'{self.path}images/melted_explained_variances.png')
            if static:
                covariates_list += [pn.pane.PNG(f'{self.path}images/melted_explained_variances.png')]
            else:
                covariates_list += [pn.pane.HoloViews(fig_exp_vars)]
        # cov_card = pn.Card( pn.Card(cov_text, pn.pane.Plotly(fig_exp_vars), title = 'Covariate r<sup>2</sup> with traits in percent', collapsed=True),
        #                     pn.Card('Move the divider to see how the preprocessing changes the values of the data *(original - left | regressed out - right)*',\
        #                             pn.Swipe(pn.pane.Plotly(g1),pn.pane.Plotly(g0)), title = 'Changes after regressing out covariates', collapsed=True),
        #                     pn.Card(  \
        #                         pn.Card(pn.pane.Plotly(self._make_eigen3d_figure()), title = 'PCA representation of the data' , collapsed=True),\
        #                         pn.Card(pn.pane.Plotly(self._make_umap3d_figure()), title = 'UMAP representation of the data' , collapsed=True),\
        #                            title = 'EXPERIMENTAL' , collapsed=True),
        #                    title = 'Preprocessing', collapsed=True)
        try:
            self._make_eigen3d_figure().write_html(f'{self.path}images/traitPCA.html')
            self._make_umap3d_figure().write_html(f'{self.path}images/traitUMAP.html')
        except: 
            printwithlog('could not make the PCA and umap figures')

        fulldf = pd.read_csv(f'{self.path}processed_data_ready.csv' ,dtype = {'rfid':str})
        fulldf = fulldf.loc[:, ~fulldf.columns.str.contains('^Unnamed:')]
        fulldf = fulldf.T.dropna(how = 'all').T
        
        regressedtraits_fulldf = list(fulldf.filter(regex = 'regressedlr_').columns)
        remove_reg = list(set(regressedtraits_fulldf) - set(['regressedlr_'+x for x in traitfilter]))    
        remove_reg = remove_reg + [x.replace('regressedlr_', '') for x in remove_reg]
        fulldf = fulldf.drop(remove_reg, axis = 1, errors = 'ignore')
        
        cov_card = pn.Card(pn.Card(*covariates_list,  title = 'r<sup>2</sup> between traits and covariates (%)', collapsed=False),\
                           pn.Card(fancy_display(fulldf, download_name = 'full_dataset.csv', flexible = True,
                                                 cell_font_size=10,  header_font_size=10,max_width=1100, layout = 'fit_data_fill',max_cell_width=120 ),
                                   title = 'Full dataset', collapsed=True),
                           title = 'Preprocessing', collapsed=True)
        add_metadata(cov_card).save(f'{self.path}images/report_pieces/covariates.html')
        template.main.append(cov_card)
        printwithlog('generating report... making genetic correlation section...')
        try:panel_genetic_pca = self.make_panel_genetic_PCA()
        except:
            printwithlog('genetic pca failed')
            panel_genetic_pca = pn.Card('failure calculating genomic PCA', title = 'Genomic PCA', collapsed = True)
        #template.main.append(panel_genetic_pca)
        if len(traits)>1:
            gcorrtext = f'''# *Genetic Correlation Matrix*

Genetic correlation is a statistical concept that quantifies the extent to which two traits share a common genetic basis. The estimation of genetic correlation can be accomplished using Genome-wide Complex Trait Analysis (GCTA), a software tool that utilizes summary statistics from genome-wide association studies (GWAS) to estimate the genetic covariance between pairs of traits. GCTA implements a method that decomposes the total phenotypic covariance between two traits into genetic and environmental components, providing an estimate of the genetic correlation between them. This approach allows researchers to examine the degree of shared genetic architecture between traits of interest and gain insights into the biological mechanisms underlying complex traits and diseases. 

For the figure, the upper triangle represents the genetic correlation (ranges from [-1:1]), while the lower triangle represents the phenotypic correlation. Meanwhile the diagonal displays the heritability (ranges from [0:1]) of the traits. Hierarchical clustering is performed using [scipy's linkage function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) with the genetic correlation. Dendrogram is drawn using [scipy dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html) where color coding for clusters depends on a distance threshold set to 70% of the maximum linkage distance. Asterisks means that test failed, for genetic relationship the main failure point is if the 2 traits being tested are colinear, while for the phenotypic correlation it's due to no overlapping individuals between the 2 traits.'''
            gcorrorder = 'sorted' if sorted_gcorr else 'cluster' 
            bokehgcorrfig = self.make_genetic_correlation_figure(order = gcorrorder, save = True, traits = traits)
            if static:gcorrfig = pn.pane.PNG(f"{self.path}images/genetic_correlation_matrix2{('_'+gcorrorder).replace('_cluster', '')}.png")
            else: gcorrfig = pn.pane.HoloViews(bokehgcorrfig)
            #gcorrfig = pn.pane.PNG(f'{self.path}images/genetic_correlation_matrix2{"_sorted" if sorted_gcorr else "cluster"}.png', max_width=1200, max_height=1200, width = 1200, height = 1200)
            try: gcorr = pd.read_csv(f"{self.path}results/heritability/genetic_correlation_melted_table.csv", index_col=0).map(lambda x: round(x, 3) if type(x) == float else x.replace('regressedlr_', ''))
            except: gcorr = pd.read_csv(f"{self.path}results/heritability/genetic_correlation_melted_table.csv", index_col=0).applymap(lambda x: round(x, 3) if type(x) == float else x.replace('regressedlr_', ''))
            gcorr = fancy_display(gcorr.query('trait1.isin(@traitfilter) and trait2.isin(@traitfilter)'), 
                                  download_name='genetic_correlation.csv')
            # gcorr = fancy_display(gcorr.query('trait1.isin(@traitfilter) and trait2.isin(@traitfilter)'), 
            #                       download_name='genetic_correlation.csv')
            genetic_corr = pn.Card(gcorrtext, gcorrfig, pn.Card(gcorr, title = 'tableView', collapsed=True),title = 'Genetic Correlation', collapsed=True)
            add_metadata(genetic_corr).save(f'{self.path}images/report_pieces/genetic_corr.html')
            template.main.append(genetic_corr)
        heritext = '''# **SNP Heritability Estimates h<sup>2</sup>** 

SNP heritability (often reported as h<sup>2</sup> ) is the fraction of phenotypic variance that can be explained by the genetic variance measured from the Biallelic SNPS called by the genotyping pipeline. It is conceptually similar to heritability estimates that are obtained from panels of inbred strains (or using a twin design in humans), but SNP heritability is expected to be lower.  Specifically, this section shows the SNP heritability (“narrow-sense heritability”) estimated for each trait by GCTA-GREML, which uses the phenotypes and genetic relatedness matrix (GRM) as inputs. Traits with higher SNP heritability are more likely to produce significant GWAS results. It is important to consider both the heritability estimate but also the standard error; smaller sample sizes typically have very large errors, making the results harder to interpret. 
Note that Ns for each trait may differ from trait to trait due to missing data for each trait. 

Column definitions: 

* trait: trait of interest
* N: number of samples containing a non-NA value for this trait
* heritability: quantifies the proportion of phenotypic variance of a trait that can be attributed to genetic variance
* heritability_se: standard error, variance that is affected by N and the distribution of trait values
* pval: probability of observing the estimated heritability under the NULL hypothesis (that the SNP heritability is 0)'''
        
        #
        printwithlog('generating report... making heritability section...')
        herfig = pn.pane.HoloViews(self.make_heritability_figure(add_classes = add_cluster_color_heritability, traitlist = traits))
        if static: herfig = pn.pane.PNG(f'{self.path}images/heritability_sorted.png', width = 1000)
            
        her = pd.read_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t')\
             .set_axis(['trait', 'gen_var', 'env_var', 'phe_var', 'heritability', 'likelihood', 'lrt', 'df', 'pval', 'n', 'heritability_se'], axis = 1).drop(['env_var', 'lrt', 'df'],axis = 1)
        her.trait = her.trait.str.replace('regressedlr_', '')
        her = her[her.trait.isin(traitfilter)]
        her = fancy_display(her, download_name='heritablitity.csv')
        herit_card =  pn.Card(heritext, herfig, her, title = 'Heritability', collapsed=True)
        add_metadata(herit_card).save(f'{self.path}images/report_pieces/heritability.html')
        template.main.append(herit_card)

        qtls = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')\
                 .query('QTL')\
                 .rename({'p':'-Log10(p)', 'b':'beta', 'se': 'betase', 'af': 'Freq', 'SNP': 'TopSNP'}, axis = 1).round(3)\
                 .merge(pd.read_parquet(f'{self.path}genotypes/snpquality.parquet.gz', columns=['F_MISS','GENOTYPES','HWE']), left_on = 'TopSNP', right_index=True)
        qtls = qtls[qtls.trait.isin(traitfilter)]
        founder_ids = set(self.foundersbimfambed[1].fid.sort_values().to_list()) if len(self.foundersbimfambed) else set()
        qtlstext = f'''
# **Summary of QTLs** 

The genome-wide significance threshold (-log10p): 

{threshtext}

The values shown in the table below pass the {self.genome_version}:{round_version} suggestive threshold. 

  Quantitative trait loci (QTLs) are regions in the genome that contain single nucleotide polymorphisms (SNPs) that correlate with a complex trait.
If there are multiple QTLs in a given chromosome, then the top SNP from the most significant QTL is used as a covariate for another GWAS analysis within the chromosome.  If the analysis results in another SNP with a p-value that exceeds the permutation-derived threshold then it is considered an independent locus. This continues until no more QTLs are devoted within a given chromosome. This method is described in details in (Chitre et al., 2020)


Column definitions: 


* TopSNP: SNPs with lowest p-value whithin an independent QTL. SNP name is defined by the location of the top SNP on the chromosome. Read it as follows chromosome: position, so 10:10486551 would be chromosome 10, location on the chromosome at 10486551
* af: frequency of the TopSNP in the population used
* beta: effect size of topSNP
* betase: standard error of effect size of topSNP
* -Log10(p): statistical significance of the association between the trait variability and the top SNP, displayed as -log10(p-value). The log-transformed p-value used in all figures and tables in this report
* trait: trait in which the snp was indentified
{f"* {', '.join(list(founder_ids)[:100])}: genotypes of founders at the topSNP" if len(founder_ids) else ''}'''
        
        printwithlog('generating report... making qtl section...')
        qtl_cols2display = [x for x in ['TopSNP','start_qtl','end_qtl','interval_size','Freq','F_MISS','GENOTYPES','HWE','beta','betase','-Log10(p)','significance_level','trait'] \
                            if x in qtls.columns]
        qtls = qtls[qtl_cols2display + list(founder_ids & set(qtls.columns)) ]
        qtls_card = pn.Card(qtlstext, fancy_display(qtls, 'qtls.csv', flexible = True, cell_font_size=10, 
                                                    header_font_size=10,max_width=1100, layout = 'fit_data_fill',max_cell_width=120 ), title = 'QTL', collapsed=True)
        add_metadata(qtls_card).save(f'{self.path}images/report_pieces/qtls.html')
        template.main.append(qtls_card)
        
        porcupinetext = f'''# **Porcupine Plot**
        
Porcupine plot is a graphical tool that combines multiple Manhattan plots, each representing a single trait, into a single plot. The resulting plot provides a visual representation of the regions of the genome that influence multiple traits, enabling researchers to identify genetic loci that have pleiotropic effects. These plots allow for a quick and efficient analysis of multiple traits simultaneously. For the porcupine plots shown below, only traits with at least one significant QTL are shown.'''
        skipmanhattan = len(set(map(lambda x: x.replace('regressedlr_',''),traits)) \
                - set(map(lambda x: basename(x).replace('.png', ''), glob(f'{self.path}images/manhattan/*.png'))) )
        skipmanhattan = True if not skipmanhattan else False
        if (not static) or redo_figs or (not os.path.isfile(f'{self.path}images/porcupineplot.png')):
            porcfig = pn.pane.HoloViews(self.porcupineplot(display_figure = False, skip_manhattan=skipmanhattan, traitlist=traits ),\
                                        max_width=1200, max_height=600, width = 1200, height = 600)
        else: 
            printwithlog('generating report... loading already ran porcupinefig...')
            porcfig = pn.pane.PNG(f'{self.path}images/porcupineplot.png')
        pcp_o = [porcupinetext,porcfig]
        if add_qqplot and (len(traits)< 20):
            printwithlog('generating report... making qqplot section...')
            if not os.path.isfile(f'{self.path}images/qqplot.png') or not static or redo_figs:
                if len(traits)> 80: qqplot_add_pval_thresh = False
                qqplotfig = pn.pane.HoloViews(self.qqplot(add_pval_thresh= qqplot_add_pval_thresh),
                                              max_width=1200, max_height=1200, width = 900, height = 900)
            if static:
                qqplotfig = pn.pane.PNG(f'{self.path}images/qqplot.png')
            pcp_o += [qqplotfig]
        porcupine_card = pn.Card(*pcp_o,  title = 'Porcupine Plot', collapsed=True)
        add_metadata(porcupine_card).save(f'{self.path}images/report_pieces/porcupine.html')
        template.main.append(porcupine_card)
        
        manhattantext = f'''# **Manhattan plots (for significant QTLS)**
    
These Manhattan plots show QTLs that genome-wide significance threshold of: 

{threshtext}

The Manhattan plot displays the p-values of each SNP sampled, with the aim of finding specific SNPs that pass the significance threshold. The x-axis shows chromosomal position and the y-axis shows -log10 of the p-value. The GWAS analysis uses a linear mixed model implemented by the software package GCTA (function MLMA-LOCO) using dosage and genetic relatedness matrices (GRM) to account for relatedness between individuals in the population. The analysis also employs Leave One Chromosome Out (LOCO) to avoid proximal contamination. 

The genomic significance threshold is the genome-wide significance threshold calculated using permutation test, and the genotypes at the SNPs with p-values exceeding that threshold are considered statistically significantly associated with the trait variance. Since traits are quantile-normalized, the cutoff value is the same across all traits. QTLs are determined by scanning each chromosome for at least a SNP that exceeds the calculated permutation-derived threshold.

To control type I error, we estimated the significance threshold by a permutation test, as described in (Cheng and Palmer, 2013).'''
        
        manhatanfigs = [pn.Card(pn.pane.PNG(f'{self.path}images/manhattan/{trait}.png', max_width=1000, 
                     max_height=600, width = 1000, height = 600), 
                               fancy_display(qtls.query('trait == @trait'), download_name=f'qtls_in_{trait}.csv',
                                             flexible = True, cell_font_size=10, header_font_size=12,max_width=1100, layout = 'fit_data_fill',max_cell_width=120 ),
                                title = trait, collapsed = True) for trait in qtls.trait.unique()]
        
        manhatanfigs2 = [pn.Card(pn.pane.PNG(f'{self.path}images/manhattan/{trait}.png', max_width=1000, 
                     max_height=600, width = 1000, height = 600), title = trait, collapsed = True) \
                        for trait in set(map(lambda x: x.replace('regressedlr_', ''), traits)) - set(qtls.trait)]
        
        manhatan_card = pn.Card(manhattantext, 
                              pn.Card(*manhatanfigs, title='Plots with QTLs', collapsed=True),
                              pn.Card(*manhatanfigs2, title='Plots without QTLs', collapsed=True),
                              title = 'Manhattan Plots', collapsed=True)
        add_metadata(manhatan_card).save(f'{self.path}images/report_pieces/manhattan.html')
        template.main.append( manhatan_card )
        db_vals_t = pd.concat(pd.read_parquet(x).assign(phewas_file = x) for x in self.phewas_db.split(',')).reset_index(drop= True)
        if remove_umap_traits_faq: 
            db_vals_t = db_vals_t[~db_vals_t.trait.str.contains(r'umap\d|^umap_clust|^pc\d$|^pca_clus')]
        PROJECTLIST = '\n'.join(list(map(lambda x: '*  ' + x, db_vals_t['project'].unique())))
        eqtlstext = '' if self.species not in ['rattus_norvegicus'] else f'''## Gene Expression changes:

### expression QTL (eQTLs) 
We examine if the identified SNP does significant alter the gene expression of genes in cis, possibly describing a pathway in which the SNP impact the phenotype. We also examine SNPs within 3 Mb window and a r2  above 0.6 of the topSNP from the current study, in case the selected SNP is in high LD with another SNP that can alter the gene expression in cis.

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
We examine if the identified SNP does significant alter the splicing patterns of genes in cis, possibly describing a pathway in which the SNP impact the phenotype. We also examine SNPs within 3 Mb window and a r2 above 0.6 of the topSNP from the current study, in case the selected SNP is in high LD with another SNP that can alter the splicing in cis.

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

These tables report the correlation between the topSNP and traits from other studies in conducted by the center. Use information from these tables to better understand what additional phenotypes this interval may be associated with. 

The PheWAS table examines the association between the topSNP for this phenotype and all other topSNPs that were mapped within a 3 Mb window of the topSNP from the current study and a r2 above 0.6. Instead of showing association of the topSNP with other traits like in the first table, the second table shows significant association identified for other traits within the nearby chromosomal interval.

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
        
        ann = pd.read_csv(f'{self.path}results/qtls/annotQTL.tsv', sep = '\t').drop(['A1','A2', 'featureid', 'rank', 
                                                                               'cDNA_position|cDNA_len','CDS_position|CDS_len',
                                                                               'Protein_position|Protein_len','distancetofeature'], errors = 'ignore' ,axis = 1)\
                 .query("putative_impact in ['MODERATE', 'HIGH']").sort_values('putative_impact')
        ann['p'] = -np.log10(ann.p)
        ann.rename({'p':'-Log10(p)'},axis=1,  inplace=True)
        ann = ann.set_index(['SNP_qtl', 'SNP',  'R2' , 'gene', 'consequence','putative_impact', 'trait'])\
                 .drop(['PASS','PASS_HWE','PASS_MAF','PASS_MISS', 'source'], errors ='ignore', axis = 1).reset_index()
        
        phewas = pd.read_csv(f'{self.path}results/phewas/pretty_table_both_match.tsv', sep = '\t')
        phewas = phewas.loc[phewas.R2.replace('Exact match SNP', 1.1).astype(float).sort_values(ascending = False).index]\
                       .rename({'p_PheDb': '-Log10(p)PheDb'}, axis =1).drop(['round_version', 'uploadeddate'], axis =1)\
                                 .drop_duplicates(['SNP_QTL', 'SNP_PheDb','trait_QTL','trait_PheDb','project'])
        if remove_umap_traits_faq:
            phewas = phewas[~phewas.trait_PheDb.str.contains(r'umap\d|^umap_clust|^pc\d$|^pca_clus')]
        
        eqtl = pd.read_csv(f'{self.path}results/eqtl/pretty_eqtl_table.csv')\
                 .rename({'-log10(P-value)':'-Log10(p)', '-log10(pval_nominal)': '-Log10(p)_eqtldb' }, axis =1)
        
        sqtl = pd.read_csv(f'{self.path}results/sqtl/pretty_sqtl_table.csv')\
                 .rename({'-log10(P-value)':'-Log10(p)', '-log10(pval_nominal)': '-Log10(p)_sqtldb' }, axis = 1)
        
        genes_in_range = pd.read_csv(f"{self.path}results/qtls/genes_in_range.csv")
        genes_in_range2 = self.make_genes_in_range_mk_table().reset_index().drop_duplicates()
        
        out = [regional_assoc_text]
        printwithlog('generating report... making section per qtl...')
        for index, row in tqdm(list(qtls.iterrows())):
            texttitle = f"Trait: {row.trait} SNP: {row.TopSNP}\n"
            #row_desc = fancy_display(row.to_frame().T)
            row_desc = pn.pane.Markdown(row.to_frame().T.fillna('').set_index('TopSNP').to_markdown())
            snp_doc = row.TopSNP.replace(":", '_')
            c_num  = row.Chr if isinstance(snp_doc.split('_')[0], int) else int(self.replaceXYMTtonums(snp_doc.split('_')[0]))
            if len(ginrange := genes_in_range2.query('SNP_origin.eq(@row.TopSNP)')):
                girantable = fancy_display(ginrange.fillna(''), download_name= f'genes_in_region__{row.trait}__{snp_doc}.csv', 
                                             add_sort=False, wrap_text='wrap', html_cols=['genebass', 'twashub', 'genecup', 'gwashub', 'RGD', 'genecard'], 
                                             page_size = 40, cell_font_size=10, header_font_size=12,max_width=1200,  layout = 'fit_data_fill', flexible = True)
                giran = pn.Card(girantable, title = 'Gene Links', collapsed = False, min_width=500)
                all_genes_string = ', '.join(ginrange.symbol.unique())
            else: 
                giran = pn.Card(f'no Genes in section for SNP {row.TopSNP}', title = 'Gene Links', collapsed = False, min_width=500)
                all_genes_string = 'no genes are present in the region'
            if legacy_locuszoom:
                lzplot = pn.pane.JPG(f'''{self.path}images/lz/legacyr2/lz__{row.trait}__{snp_doc}.jpeg''', width = 900)
                lzplot2 = pn.Card(pn.pane.JPG(f'''{self.path}images/lz/legacy6m/lz__{row.trait}__{snp_doc}.jpeg''', width = 900),
                              title = 'zoomed out locuszoom', collapsed = True)
            else:
                lzplot = pn.pane.PNG(f'{self.path}images/lz/r2thresh/lzi__{row.trait}__{snp_doc}.png')
                lzplot2 = pn.Card(pn.pane.PNG(f'{self.path}images/lz/minmax/lzi__{row.trait}__{snp_doc}.png'),
                                  pn.pane.JPG(f'''{self.path}images/lz/legacyr2/lz__{row.trait}__{snp_doc}.jpeg''', width = 900),
                                  pn.pane.JPG(f'''{self.path}images/lz/legacy6m/lz__{row.trait}__{snp_doc}.jpeg''', width = 900),
                                  title = 'zoomed out locuszoom', collapsed = True) 
            #,  max_width=1200, max_height=4000, width = 1200, height = 800
            # lztext = pn.pane.Markdown(f'[interactive version](https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/images/lz/minmax/lzi__{row.trait}__{snp_doc}.html)')
            # lztext = pn.pane.Markdown(f'''<a href="https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/images/lz/minmax/lzi__{row.trait}__{snp_doc}.html" target="_blank">interactive locuszoom🔗</a>''')
            lztext = pn.widgets.Button(name="interactive locuszoom🔗", button_type="primary",  )
            ilzurl = f'https://palmerlab.s3.sdsc.edu/tsanches_dash_genotypes/gwas_results/{self.project_name}/images/lz/minmax/lzi__{row.trait}__{snp_doc}.html'
            lztext.js_on_click(code=f'window.open("{ilzurl}", "_blank");')
            
            boxplot = pn.pane.PNG(f'{self.path}images/boxplot/boxplot{snp_doc}__{row.trait}.png', max_width=800, max_height=400, width = 800, height = 400)
        
            cau_title = pn.pane.Markdown(f"### Coding variants: {row.trait} {row.TopSNP}\n")
            try:cau = ann.query('SNP_qtl == @row.TopSNP and trait == @row.trait')[['SNP','gene','Freq','b','-Log10(p)','R2','DP',\
                                                                                   'putative_impact','consequence','aminoacids','codons','refallele',\
                                                                                   'distance_qtlsnp_annotsnp','HGVS.c','HGVS.p']]\
                               .drop_duplicates().sort_values('putative_impact')
            except:cau = ann.query('SNP_qtl == @row.TopSNP and trait == @row.trait').drop_duplicates().sort_values('putative_impact')
            caulstemp_string = ', '.join([f'{i[0]} contains {j} {i[1].replace("_variant", "")} variant{"s" if j>1 else "" }' for i,j in cau[['gene', 'consequence']].value_counts().items()])
            if caulstemp_string: caulstemp_string += '\n' + cau.to_markdown() + '\n'
            else: caulstemp_string = 'none contain high impact variants according to VEP annotation'
            if cau.shape[0]: cau = fancy_display(cau.loc[:, ~cau.columns.str.contains('^PASS')].rename({'distance_qtlsnp_annotsnp': 'distance', 'putative_impact': 'impact'}, axis = 1),
                                                 download_name=f'CodingVariants_{row.trait}{row.TopSNP}.csv'.replace(':', '_'), flexible = True, cell_font_size=10, header_font_size=12,max_width=1100,max_cell_width=100 )
            else: cau = pn.pane.Markdown(' \n HIGH or MODERATE impact variants absent \n   \n')

            phewas_section = []
            phewas_string = ''
            for idx, tdf in phewas.groupby('phewas_file'):
                pboth_title = pn.pane.Markdown(f"### PheWAS: Lowest P-values for other phenotypes in a 3Mb window of {row.trait} {row.TopSNP} for {basename(tdf.phewas_file.iloc[0])}\n")
                pbothtemp = tdf.query('SNP_QTL == @row.TopSNP and trait_QTL == @row.trait')[['SNP_PheDb','-Log10(p)PheDb','R2', 'DP' ,'trait_PheDb', 'project', 'trait_description_PheDb']].drop_duplicates()
                if pbothtemp.shape[0]: 
                    phewas_string += ', '.join(pbothtemp.trait_PheDb) + '\n' + pbothtemp.to_markdown() + '\n'
                    pbothtemp = fancy_display(pbothtemp.fillna(''), download_name=f'phewas_{row.trait}{row.TopSNP}.csv'.replace(':', '_'), flexible = True)
                else: pbothtemp = pn.pane.Markdown(f' \n SNPS were not detected for other phenotypes in 3Mb window of topSNP  \n   \n')
                phewas_section += [pboth_title,pbothtemp]
            if phewas_string: 
                phewas_string = 'After performing a PheWAS, this QTL also correlates to the following traits: ' + phewas_string
            else: phewas_string = 'After performing a PheWAS, no other traits were detected in this region'
        
            eqtl_title = pn.pane.Markdown(f"### eQTL: Lowest P-values for eqtls in a 3Mb window of {row.trait} {row.TopSNP}\n")
            eqtltemp = eqtl.query(f'SNP == "{"chr"+row.TopSNP}" and trait == "{row.trait}"').rename({'Ensembl_gene': 'gene'}, axis = 1)\
                           [['SNP_eqtldb', '-Log10(p)_eqtldb', 'tissue', 'R2', 'DP', 'gene', 'gene_id', 'slope', 'af']].drop_duplicates()
            eqtlstemp_string = ', '.join([f'{i} contains {j} expression QTLs' for i,j in eqtltemp.gene_id.value_counts().items()])
            if eqtlstemp_string: eqtlstemp_string += '\n' + eqtltemp.to_markdown() + '\n'
            else: eqtlstemp_string = 'none contain an expression QTL'
            if eqtltemp.shape[0]: eqtltemp = fancy_display(eqtltemp.fillna(''),download_name=f'eqtl_{row.trait}{row.TopSNP}.csv'.replace(':', '_'), flexible = True)
            else:eqtltemp = pn.pane.Markdown(f' \n SNPS were not {"tested" if c_num > self.n_autosome else "detected"} for eQTLs in 3Mb window of trait topSNP  \n   \n')
        
            sqtl_title = pn.pane.Markdown(f"### sQTL: Lowest P-values for splice qtls in a 3Mb window of {row.trait} {row.TopSNP}\n")
            sqtltemp = sqtl.query(f'SNP=="{"chr"+row.TopSNP}" and trait == "{row.trait}"').rename({'Ensembl_gene': 'gene'}, axis = 1)\
                           [['SNP_sqtldb', '-Log10(p)_sqtldb', 'tissue','R2', 'DP', 'gene','gene_id' , 'slope', 'af']].drop_duplicates()
            sqtltemp_string = ', '.join([f'{i} contains {j} splice QTLs' for i,j in sqtltemp.gene_id.value_counts().items()])
            if sqtltemp_string: sqtltemp_string += '\n' + sqtltemp.to_markdown() + '\n'
            else: sqtltemp_string = 'none contain an splice QTL'
            if sqtltemp.shape[0]: sqtltemp = fancy_display(sqtltemp.fillna(''), download_name=f'sqtl_{row.trait}{row.TopSNP}.csv'.replace(':', '_'), flexible = True)
            else: sqtltemp = pn.pane.Markdown(f' \n  SNPS were not {"tested" if c_num > self.n_autosome else "detected"} for sQTLs in 3Mb window of trait topSNP  \n   \n')

            dt2append = [giran, cau_title, cau] + phewas_section #+[phe_title,  phetemp, phew_title, phewtemp]
            if self.genome_accession in ['GCF_015227675.2', 'GCF_000001895.5']:
                dt2append += [eqtl_title, eqtltemp,sqtl_title, sqtltemp]

            question = f'''I performed a GWA study in {self.species.replace("_", " ")} for a certain trait ({row.trait}) and there was \
a significant SNP at {row.TopSNP}. The region near this significant SNP contains the following genes:\
{all_genes_string}. Out of these genes, {caulstemp_string}. Out of these genes, {eqtlstemp_string}.\
Out of these genes, {sqtltemp_string}. {phewas_string}. Given this information and the knowledge about these genes in the literature, \
could you rank the genes from most likely to least likely to cause the phenotype and explain your decision for the 3 most likely genes?\
For the most likely gene could do describe the gene and it's role in embryo development and adult homeostasis and behaviour?\
Last, is there any FDA approved drug known to specifically alter this gene or the pathways in which this gene?\
 Choosing the correct genes is of major importance since the validation with knock-out experiments or with testing the FDA approved drugs will be costly. \
For your answer please write it in long form as a section for a manuscript and not a itemized version. Please be complete in your answers, the more infomation you can provide about how you arrived at your answer the better.'''.replace('    ', '')
            gptanswers = []
            all_models = ["OpenScholar/Llama-3.1_OpenScholar-8B"]
            # "allenai/OLMo-2-1124-13B-Instruct" "meta-llama/Llama-3.2-90B-Vision-Instruct"
            if add_gpt_query:
                from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, MambaForCausalLM
                for model_name in all_models:
                    tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir="/tscc/projects/ps-palmer/gwas/databases/ai_cache")
                    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/tscc/projects/ps-palmer/gwas/databases/ai_cache")
                    chatquery = tokenizer.bos_token + question+ tokenizer.eos_token
                    input_ids = tokenizer(chatquery, return_tensors="pt").input_ids
                    output = model.generate(input_ids,min_new_tokens= 600,  max_new_tokens=700,pad_token_id=tokenizer.pad_token_id,
                                                      bos_token_id=tokenizer.bos_token_id,
                                                      eos_token_id=tokenizer.eos_token_id)
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    response = generated_text.replace(question, '').split('\n\n',1)[-1]
                    gptanswers += [response]

            if len(gptanswers):
                gpt_card = pn.Card(clipboard_button(question),pn.Accordion(*zip(all_models, gptanswers), width = 1100),
                title = "LLM discussion for the Regional Association", collapsed = True)
            else: gpt_card = clipboard_button(question)
        
            reg_card = pn.Card(*[row_desc,lzplot,lzplot2,lztext,boxplot,gpt_card,pn.Card(*dt2append, title = 'tables', collapsed = False)]   ,title = texttitle, collapsed = True)
            add_metadata(reg_card).save(f"{self.path}images/report_pieces/regional_assoc/{row.trait}@{row.TopSNP.replace(':', '__')}.html")
            out += [reg_card]
            #

        reg_assoc = pn.Card(*out, title = 'Regional Association Plots', collapsed = True)
        add_metadata(reg_assoc).save(f'{self.path}images/report_pieces/regional_assoc.html')
        template.main.append(reg_assoc)
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
            experimental_card = pn.Card(*add2card, title = 'Experimental', collapsed=True)
            add_metadata(experimental_card).save(f'{self.path}images/report_pieces/experimental.html')
            template.main.append(experimental_card)
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
        
        template.main.append(pn.Card(faqtext, fancy_display(faqtable, 'list_of_traits.csv', flexible = True), title = 'FAQ', collapsed = True))
        
        reftext = '''* Chitre AS, Polesskaya O, Holl K, Gao J, Cheng R, Bimschleger H, Garcia Martinez A, George T, Gileta AF, Han W, Horvath A, Hughson A, Ishiwari K, King CP, Lamparelli A, Versaggi CL, Martin C, St Pierre CL, Tripi JA, Wang T, Chen H, Flagel SB, Meyer P, Richards J, Robinson TE, Palmer AA, Solberg Woods LC. Genome-Wide Association Study in 3,173 Outbred Rats Identifies Multiple Loci for Body Weight, Adiposity, and Fasting Glucose. Obesity (Silver Spring). 2020 Oct;28(10):1964-1973. doi: 10.1002/oby.22927. Epub 2020 Aug 29. PMID: 32860487; PMCID: PMC7511439.'''
        template.main.append(pn.Card(reftext, title = 'References', collapsed = True))

        if os.path.isfile(f'{self.path}run_notes.md'):
            with open(f'{self.path}run_notes.md') as f: rnotes = f.read()
            template.main.append(pn.Card(pn.pane.Markdown(rnotes), collapsed = True, title ='Manual Notes'))
        
        template.header.append(f'## {self.project_name if not headername else headername}')
        template.save(f'{self.path}results/gwas_report.html', resources=CDN, embed = False, title = f'{self.project_name}_GWAS')
        #template.save(f'{self.path}results/gwas_report.html', resources=INLINE)
        bash(f'''cp {self.path}results/gwas_report.html {self.path}results/gwas_report_{self.project_name}_round{round_version}_threshold{round(self.threshold,2)}_n{self.df.shape[0]}_date{datetime.today().strftime('%Y-%m-%d')}_gwasversion_{gwas_version}.html''')
        #printwithlog(f'{destination.replace("/tscc/projects/ps-palmer/s3/data", "https://palmerlab.s3.sdsc.edu")}/{self.project_name}/results/gwas_report.html')
    
    def store(self, researcher: str , round_version: str, gwas_version: str= None, remove_folders: bool = False):
        """
        Store the project data by compressing it into a zip file.
    
        Steps:
        1. Define the paths of all folders to be included in the zip file.
        2. Create a unique information string based on the project name, researcher, round version, and GWAS version.
        3. Zip all the specified folders into a single zip file named with the unique information string.
        4. Optionally remove the original folders after zipping them.
        5. If specified, remove extra folders such as 'logerr' and 'temp'.
    
        :param researcher: Name of the researcher.
        :param round_version: Version of the round.
        :param gwas_version: Version of the GWAS pipeline.
        :param remove_folders: Boolean indicating whether to remove the original folders after zipping.
        :return: None
        """
        if gwas_version is None: gwas_version = __version__
        base_folder_files = reduce(lambda x, y : x+y, [glob(self.path + x) for x in \
                           ['environment.yml', 'package_versions.txt', 'run_notes.md',
                            f'data_dict_{self.project_name}.csv',\
                            'processed_data_ready.csv','missing_in*',\
                            'missing_rfid_list.txt','pivot_explained_variances.csv',\
                            'data_distributions.html','melted_explained_variances.csv']])
        result_folders = [self.path + x for x in ['data', 'genotypes','images','grm', 'log', 'results']]
        all_folders = ' '.join(base_folder_files+ result_folders)
        info = '_'.join([self.project_name, researcher,round_version,gwas_version])
        bash(f'zip -r {self.path}run_{info}.zip {all_folders}')
        if remove_folders: 
            bash(f'rm -r {all_folders}')
            extra_folders = ' '.join([self.path + x for x in ['logerr', 'temp']])
            bash(f'rm -r {extra_folders}')
            
    def copy_results(self, destination: str = '/tscc/projects/ps-palmer/s3/data/tsanches_dash_genotypes/gwas_results', make_public: bool = True):
        """
        Copy the project results to a specified destination.
    
        Steps:
        1. Print a log message indicating the start of the copy process.
        2. Create the destination directory if it does not exist.
        3. Copy the project directory to the destination.
        4. Wait for a specified time to ensure all files are copied.
        5. Optionally make the copied files public by adjusting permissions.
    
        :param destination: Path to the destination directory.
        :param make_public: Boolean indicating whether to make the copied files public.
        :param tscc: Integer indicating the TSCC setting.
        :return: None
        """
        print(f'copying {self.project_name} to {destination}')
        os.makedirs(f'{destination}', exist_ok = True)
        out_path, pjname = (self.path, '') if self.path else ('.', f'/{self.project_name}')
        
        bash(f'cp -r {out_path} {destination}{pjname}')
        print('waiting 1 min for copying files...')
        sleep(60*1)
        if make_public:
            bash('/tscc/projects/ps-palmer/tsanches/mc anonymous set public /tscc/projects/ps-palmer/s3/data/tsanches_dash_genotypes --recursive')
            printwithlog(f'{destination.replace("/tscc/projects/ps-palmer/s3/data", "https://palmerlab.s3.sdsc.edu")}/{self.project_name}/results/gwas_report.html')


    def GeneEnrichment(self, qtls: pd.DataFrame = None, padding: int = 2e6, r2thresh: float = .6,
                   append_nearby_genes:bool = False, select_nearby_genes_by:str = 'r2'):
        """
        Perform gene enrichment analysis on QTLs.
    
        Steps:
        1. Download and initialize necessary GO and gene association data.
        2. If QTLs are provided as a string, read them from a CSV file.
        3. Get Entrez IDs for the genes associated with the QTLs.
        4. Depending on the selection method, get nearby genes by either distance or R2 threshold.
        5. Group QTLs by trait and aggregate gene information.
        6. Run GO enrichment analysis for closest and nearby genes for each trait.
        7. Save the results to files and generate a gene enrichment figure.
    
        :param qtls: DataFrame containing QTL information.
        :param genome: Genome version.
        :param padding: Padding for selecting nearby genes by distance.
        :param r2thresh: R2 threshold for selecting nearby genes.
        :param append_nearby_genes: Boolean indicating whether to append nearby genes.
        :param select_nearby_genes_by: Method for selecting nearby genes ('r2' or 'distance').
        :return: Gene enrichment figure.
        """
        from goatools.base import download_go_basic_obo
        from goatools.obo_parser import GODag
        from goatools.anno.genetogo_reader import Gene2GoReader
        from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
        os.makedirs(f'{self.path}genome_info/go', exist_ok = True)
        gene2go = download_ncbi_associations(f'{self.path}genome_info/go/gene2go')
        obo_fname = download_go_basic_obo(f'{self.path}genome_info/go/go-basic.obo')
        geneid2gos_rat= Gene2GoReader(gene2go, taxids=[int(self.taxid)])
        obodag = GODag(obo_fname)
        ratassc = geneid2gos_rat.get_ns2assc()

        if isinstance(qtls, str): qtls = pd.read_csv(qtls)
        if not (isinstance(qtls, pd.DataFrame) and len(qtls)) : qtls = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')
        def get_entrez_ids(genes):
            if type(genes) == str: genes = genes.split('-') + [genes]
            o = []
            for i in genes: 
                if not pd.isna(i): o += list(np.unique(i.split('-') + [i]))
            try: res =  list(np.unique([int(y.replace('ENSRNOG', '')) for x  in mg.querymany(o, scopes='ensemblgene,symbol,RGD', fields='all',
                                                                                        species=self.taxid, verbose = False, silent = True, entrezonly=True)\
                                                                          if (len(y := defaultdict(lambda:'', x)['entrezgene']) > 0) * ('ENSRN' not in y)])) #
            except:
                sleep(10)
                res =  list(np.unique([int(y.replace('ENSRNOG', '')) for x  in mg.querymany(o, scopes='ensemblgene,symbol,RGD', fields='all',
                                                                                        species=self.taxid, verbose = False, silent = True, entrezonly=True)\
                                                                          if (len(y := defaultdict(lambda:'', x)['entrezgene']) > 0) * ('ENSRN' not in y)]))
            return res
                
            
        print('getting entrezid per snp...')
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        allgenes = gtf.query('gbkey == "Gene"')[['gene_id', 'Chr', 'start', 'end']].drop_duplicates()
        if select_nearby_genes_by == 'distance':
            if not os.path.isfile(f"{self.path}results/qtls/genes_in_range.csv"): self.locuszoom()
            genes = pd.read_csv(f'{self.path}results/qtls/genes_in_range.csv')
            snpgroups = genes.groupby('SNP_origin')[['gene_id']].agg(lambda x: list(np.unique(x)))
            snpgroups = qtls[['SNP']].rename(lambda x: x+ '_origin', axis = 1)
            snpgroups['gene_id'] = snpgroups.SNP_origin.apply(\
                                    lambda x: list(allgenes.query(f'''Chr == {int(x.split(':')[0])} and ({x.split(':')[1]}-{padding}<value<{x.split(':')[1]}+{padding})''').gene_id.unique()))
            snpgroups = snpgroups.set_index('SNP_origin')
            snpgroups['entrezid'] = snpgroups['gene_id'].progress_apply(get_entrez_ids)
            qtls = qtls.merge(snpgroups.rename(lambda x: x+ '_nearby', axis = 1).reset_index(), left_on = 'SNP', right_on = 'SNP_origin')
        
        qtls['gene'] = qtls['gene'].apply(lambda x: [x] if type(x)==str else x)
        if select_nearby_genes_by == 'r2':
            def get_nearby_genes_r2(rowdata, r2thresh):
                temp = self.plink(bfile = self.genotypes_subset, chr = rowdata.Chr, ld_snp = rowdata.SNP, ld_window_r2 = 0.01, r2 = '',\
                                                    ld_window = 100000, thread_num = int(self.threadnum), ld_window_kb =  6000, nonfounders = '').loc[:, ['SNP_B', 'R2']] 
                minbp,maxbp = pd.Series(temp.query('R2> @r2thresh')['SNP_B'].to_list() + [rowdata.SNP]).str.extract(r':(\d+)').astype(int).agg(['min', 'max']).T.values.flatten()
                out = list(allgenes.query(f'Chr == {rowdata.Chr} and  end > {minbp} and start < {maxbp}').gene_id.unique())
                return [i for i in out ] #if ('LOC' not in i)
            qtls['gene_id_nearby'] = qtls.progress_apply(get_nearby_genes_r2, r2thresh = r2thresh, axis = 1)
            qtls['entrezid_nearby'] = qtls['gene_id_nearby'].progress_apply(get_entrez_ids)
        
        try: merged_qtls = qtls.groupby('trait')[['gene', 'gene_id_nearby','entrezid_nearby']].agg('sum').map(lambda x: [] if x == 0 else x)
        except: merged_qtls = qtls.groupby('trait')[['gene', 'gene_id_nearby','entrezid_nearby']].agg('sum').applymap(lambda x: [] if x == 0 else x)
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
                 for x in goeaobj.run_study(genelist, silent = True, verbose = False, prt=None)],\
                columns = ['GO', 'term', 'class', 'p', 'p_corr', 'n_genes', 'n_study']).sort_values(['p_corr', 'p']).query('p<0.05')
        
        print(f'running GO study for closest genes per trait...')
        merged_qtls['goea'] = merged_qtls.entrezid.progress_apply(goea)
        print(f'running GO study for nearby genes per trait...')
        merged_qtls['goea_nearby'] = merged_qtls.entrezid_nearby.progress_apply(goea)
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

    def GeneEnrichment_figure(self, qtls: str = None, merged_qtls: str = None, append_nearby_genes = False, corrected_p_threshold = 0.05):
        """
        Generate a gene enrichment figure based on QTLs and merged QTLs.
    
        Steps:
        1. Load QTLs and merged QTLs data if provided as file paths.
        2. Initialize a MultiGraph to represent the gene enrichment network.
        3. Add nodes and edges to the graph for SNPs, traits, genes, and nearby genes.
        4. Perform GO enrichment analysis and add GO terms to the graph.
        5. Customize and visualize the graph using HoloViews and NetworkX.
    
        :param qtls: Path to the QTLs pickle file or a DataFrame containing QTLs.
        :param merged_qtls: Path to the merged QTLs pickle file or a DataFrame containing merged QTLs.
        :param append_nearby_genes: Boolean indicating whether to include nearby genes in the graph.
        :return: HoloViews graph object representing the gene enrichment network.
        """
        if qtls is None: 
            qtls = pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichmentqtls.pkl')            
        elif isinstance(qtls, str):  qtls = pd.read_pickle(qtls) 
        else: pass
        if merged_qtls is None: 
            merged_qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichment_mergedqtls.pkl') 
        elif isinstance(merged_qtls, str): merged_qtls = pd.read_pickle(merged_qtls) 
        else: pass
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
            aa = row.goea.query(f'p_corr < {corrected_p_threshold}')
            aa[['p', 'p_corr']] = np.nan_to_num(-np.log10(aa[['p', 'p_corr']]), posinf=30, neginf=0)
            for _, rg in aa.iterrows():
                if not MG.has_node(f'{rg.GO}\n{rg.term}'):
                    MG.add_node(f'{rg.GO}\n{rg.term}' , what = f'Gene Ontology {trait}', 
                                size = 20*rg.p_corr, term = rg.term, cls = rg['class'], p = rg.p, color = 'firebrick')
                MG.add_edges_from([(trait, f'{rg.GO}\n{rg.term}')], weight=1.5*rg.p_corr, type = 'snp2gene')
            bb = row.goea_nearby.query(f'p_corr < {corrected_p_threshold}')
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
        """
        Generate a Steiner tree based on the gene enrichment data and a protein-protein interaction (PPI) network.
    
        Steps:
        1. Load the project graph and gene enrichment data.
        2. Identify all genes involved in the project and nearby genes.
        3. Load and process the PPI network data.
        4. Calculate the shortest paths and reconstruct the path for the Steiner tree.
        5. Generate an edge list and compute the minimum spanning tree.
        6. Create a NetworkX graph and visualize it using HoloViews.
    
        :return: A Panel Card containing the HoloViews graph object representing the Steiner tree.
        """
        import scipy.sparse as sps
        from itertools import chain
        from nltk import ngrams
        mg = self.project_graph_view(obj2return='networkx')
        qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichmentqtls.pkl') 
        merged_qtls =  pd.read_pickle(f'{self.path}results/geneEnrichment/gene_enrichment_mergedqtls.pkl') 
        allgenesproject = list(chain.from_iterable([x.split('-') for x in  mg.nodes  if mg.nodes[x]['color'] == 'seagreen']))
        allgeneswithnearby = list(set(allgenesproject + list(chain.from_iterable(qtls.gene_id_nearby))))
        if not os.path.isfile(f'{self.path}kg.csv'):
            bash(f'wget -O {self.path}kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620', return_stdout=False, silent = True)
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
        st_description = f''' The steiner tree is the approximately the minimum spanning tree that connects  QTL genes found in this study through the Graph of gene interactions. 
This tree is useful to find novel genes that bridge the genes initially found. In the future, we can also add extra connections that can account for drug interaction in the same Graph.
The knowledge Graph used comes from the Harvard's [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/). Examples of papers that use the steiner tree in genomics are:
[A Steiner tree-based method for biomarker discovery and classification in breast cancer metastasis](https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-13-S6-S8)
[Causal network models of SARS-CoV-2 expression and aging to identify candidates for drug repurposing](https://www.nature.com/articles/s41467-021-21056-z)'''
        return pn.Card(st_description, hvg, title='Steiner tree', collapsed=True)

    def project_graph_view(self, append_nearby_genes: bool = False, add_gene_enriched: bool = False, add_phewas: bool = True,
                           add_eqtl: bool = True,add_sqtl: bool = True,add_variants: bool = True, obj2return: str = 'image'):
        """
        Generate a project graph view based on various genetic and genomic data.
    
        Steps:
        1. Load and process various data files, including possible causal SNPs, PheWAS, eQTL, and sQTL data.
        2. Create a MultiGraph object and add nodes and edges based on the data.
        3. Optionally, append nearby genes, gene enrichment, PheWAS, variants, eQTL, and sQTL data to the graph.
        4. Scale node sizes and set attributes for visualization.
        5. Generate and save the graph using HoloViews.
    
        :param append_nearby_genes: Whether to append nearby genes to the graph.
        :param add_gene_enriched: Whether to add gene enrichment data to the graph.
        :param add_phewas: Whether to add PheWAS data to the graph.
        :param add_eqtl: Whether to add eQTL data to the graph.
        :param add_sqtl: Whether to add sQTL data to the graph.
        :param add_variants: Whether to add variant data to the graph.
        :param obj2return: The type of object to return ('image', 'graph', 'networkx').
        :return: The generated graph object based on the specified return type.
        """
        
        ann = pd.read_csv(f'{self.path}results/qtls/annotQTL.tsv', sep = '\t').drop(['A1','A2', 'featureid', 'rank', 
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
            try:tdf.loc[:, tdf.columns.str.contains('SNP')] = tdf.loc[:, tdf.columns.str.contains('SNP')].map(lambda x: x.replace('chr', '') if type(x)== str else x)
            except:tdf.loc[:, tdf.columns.str.contains('SNP')] = tdf.loc[:, tdf.columns.str.contains('SNP')].applymap(lambda x: x.replace('chr', '') if type(x)== str else x)
        
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
        """
        Function to create a conda environment watermark for the project and save it to the project folder.
    
        Steps:
        1. Export the current conda environment details to 'environment.yml' in the project folder.
        2. List all installed packages and their versions, saving the information to 'package_versions.txt' in the project folder.
        """
        with open(f'{self.path}environment.yml', 'w') as f:
            f.write('\n'.join(bash('conda env export --from-history')[1:-1]))
        with open(f'{self.path}package_versions.txt', 'w') as f:
            f.write('\n'.join(bash('conda list')[1:]))

    def make_genetrack_figure_(self, c: str= None, pos_start: int = None, pos_end:int= None,  frame_width = 1000, simplify_genes=True):
        genecolors = {'transcript': 'black', 'exon': 'seagreen', 'CDS': 'steelblue','start_codon': 'firebrick' ,'stop_codon': 'firebrick'}
        genesizes = {'transcript': .03, 'exon': .06, 'CDS': .07,'start_codon': .1 ,'stop_codon': .1}
        gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
        if pos_start is None: pos_start = 0
        if pos_end is None:
            pos_end = self.chr_conv_table.set_index('Chr').loc[self.replacenumstoXYMT(20), 'length'] - 1
        
        genes_in_section = self.gtf.query(f'Chr == {int(c)} and end > {pos_start} and start < {pos_end}')\
                                               .reset_index(drop = True)\
                                               .query("source not in ['cmsearch','tRNAscan-SE']")
        if simplify_genes:
            biotype_str = 'lncRNA|misc_RNA|pseudogene|RNase_|telomerase_RNA|snoRNA|rRNA|_RNA|tRNA|s[cn]RNA|other'
            gbkey_str = 'misc_RNA|ncRNA|precursor_RNA|rRNA|tRNA'
            genes_in_section = genes_in_section[~genes_in_section.gene_biotype.fillna('').str.contains(biotype_str) &\
                ~genes_in_section.gbkey.fillna('').str.contains(gbkey_str)]
        tt = genes_in_section.dropna(how = 'all', axis = 1).reset_index(drop= True).assign(y=1)
        if (not tt.shape[0]) or ('gbkey' not in tt.columns):
            resdf = pd.DataFrame(dict(genes=[0], bp = [(pos_start+pos_end)/2], label = ['no genes in section']))
            res = hv.Labels(resdf, kdims = ['bp','genes'], vdims=['label'])\
                    .opts(yaxis = None,xformatter=NumeralTickFormatter(format='0,0.[0000]a'),\
                          xlabel = '', xlim = ( pos_start, pos_end), frame_width = frame_width, height = 100, text_color = 'Red')
            return res
        tt['mean_pos'] = tt[['start', 'end']].mean(axis = 1)
        tt = tt.assign(color = tt.biotype.map(genecolors), size = tt.biotype.map(genesizes),
                       genestrand = tt.strand.map({'-': '\u20D6'}).fillna('') + tt.gene.str.replace('LOC', '') + tt.strand.map({'+': '\u20D7'}).fillna('') )
        ttgenes = tt.query('gbkey == "Gene"').reset_index(drop = True)#.set_index('end')
        scaling_factor = (pos_end - pos_start) / frame_width
        ogrange = (pos_end - pos_start)
        def scalling_law():
            r = (pos_end - pos_start)
            if r<3e6: return dict(buffer = 10, scaler = 2e6,  min_px=8, max_px = 30)
            if r<6e6: return dict(buffer = 8, scaler = 3e6,  min_px=8, max_px = 60)
            return dict(buffer = 4, scaler = 4e6,  min_px=9, max_px = 80)
        sclaw = scalling_law()
            
        ttgenes["label_width"] = ttgenes["gene"].map(len) * scaling_factor * sclaw['buffer']  # adjust 6 as needed
        ttgenes["start_buffer"] = ttgenes["start"] - ttgenes["label_width"] / 2
        ttgenes["end_buffer"] = ttgenes["end"] + ttgenes["label_width"] / 2
        
        spacing_genes = (pos_end - pos_start)/(20)
        intergene_space = .05
        for idx, row in ttgenes.iterrows():
            if idx == 0: ttgenes.loc[idx, 'stackingy'] = 0
            row_start = ttgenes.loc[idx, "start_buffer"] 
            genes_inreg = set(ttgenes[max(idx-1000, 0): idx].query(f"end_buffer > @row_start").stackingy)
            if idx>0: ttgenes.loc[idx, 'stackingy'] = min(set(range(1000)) - genes_inreg)
            
        yticks3 = (intergene_space*ttgenes[['stackingy','stackingy']]).round(2).agg(lambda x: tuple(x), axis = 1).to_list()
        yticks3l = [(x[0], '') for x in yticks3]
        size_scalers = ttgenes['stackingy'].max()*intergene_space
        stackgenepos = defaultdict(lambda: -1, (ttgenes.set_index('gene')['stackingy']*intergene_space).to_dict())
        tt = tt.assign(stackingy = tt.gene.map(stackgenepos), 
                       stackingy0 = tt.gene.map(stackgenepos) - tt['size']/8, #*size_scalers/10/2
                       stackingy1 = tt.gene.map(stackgenepos) + tt['size']/8)
        
        nrows_genes = int((np.array(yticks3)/intergene_space).max())
        hgenelabels = min(700, 300 + nrows_genes*40)
        fullgene = tt.query('gbkey == "Gene"')
        fullgene = fullgene.assign(ymax = fullgene.stackingy + max(genesizes.values())/8 ) #size_scalers/10/2
        largenames = fullgene.genestrand.str.lower().str.contains(r'^loc\d+') | fullgene.genestrand.map(lambda x: len(x) >= 6 )
        
        labels = hv.Labels(fullgene, kdims=['mean_pos', 'ymax'], vdims=['genestrand']).opts(
            hooks=[js_hook_factory(**sclaw,  font="Arial Narrow")], text_color='Black',text_font_style='normal',
            frame_width=frame_width,
            height=hgenelabels, text_baseline='top', text_align='center',
        )
        
        rect = hv.Rectangles(tt.sort_values('size', ascending = False).rename({'start':'bp'}, axis = 1), kdims=['bp', 'stackingy0', 'end', 'stackingy1'], 
                             vdims=['color', 'gene'])\
                 .opts(xformatter=NumeralTickFormatter(format='0,0.[0000]a') ,xticks=5,xrotation = 0,  frame_width = frame_width, height = hgenelabels, 
                       tools = [],  invert_yaxis=True,
                      ylim= (-intergene_space*.4, max(np.array(yticks3).max()+(.9 if ( len(ttgenes)> 200) else .5)*intergene_space, intergene_space*1.5)), 
                       xlim = ( pos_start, pos_end)) 
        rect = rect.opts(hv.opts.Rectangles(fill_color='color', line_color='color', line_width=2, fill_alpha=0.3))\
                   .opts(ylabel = 'genes', yticks =yticks3l, xformatter=NumeralTickFormatter(format='0,0.[0000]a'), frame_width = frame_width)# yticks =yticks3l, 
        return rect*labels.opts(xformatter=NumeralTickFormatter(format='0,0.[0000]a'), xticks=20)
 
    def make_phewas_figure_single(self, c: str= None, pos_start: int = None, pos_end:int= None, founderbimfambed= None, subsample = None,
                                  decompose_samples = False, add_r2fig= False, return_fig_as_list: bool = False):
        from matplotlib.colors import rgb2hex as mplrgb2hex
        if c is not None: c = int(c)
        chrmin, chrmax = pd.read_csv(f"{self.genotypes_subset}.bim", sep='\t', header = None, 
                        names = ['chr', 'snp', 'cm', 'bp', 'a1', 'a2'], 
                        dtype = {'chr': int, 'bp' : int}, 
                        usecols = ['chr', 'bp']).query(f'chr ==  {int(c)}')\
                        .groupby('chr')['bp'].agg(['min', 'max']).values.flatten()
        if pos_start is None: pos_start = chrmin
        if pos_end is None: pos_end = chrmax
        if founderbimfambed is None: founderbimfambed  = self.foundersbimfambed if len(self.foundersbimfambed) else None
        traitlist = list(map(lambda x:x.replace('regressedlr_', ''),self.traits))
        cmap = sns.color_palette("tab20", len(traitlist))
        d = {t: mplrgb2hex(cmap[v]) for v,t in enumerate(sorted(traitlist))}
        d_inv = {mplrgb2hex(cmap[v]):t for v,t in enumerate(sorted(traitlist))}
        if os.path.exists(f'{self.path}results/qtls/finalqtl.csv'): qtls = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')
        else: qtls =  pd.DataFrame(columns = ['Chr', 'SNP', 'bp', 'A1', 'A2', 'Freq', 'b', 'se', 'p', 'trait'])
        qtlssm = qtls[qtls.bp.between(pos_start, pos_end) & (qtls.Chr == int(c))]
        allmlmas = glob(f'''{self.path}results/gwas/regressedlr*_chrgwas{self.replacenumstoXYMT(f'{c}')}.mlma''')
        if len(allmlmas):
            gwasres = pd.concat([pd.read_csv(x, sep = '\t', dtype={'Chr':int, 'SNP':str, 'bp':int}).assign(trait = re.findall('regressedlr_(.*)_chrgwas', x)[0]) \
                        for x in allmlmas])
            gwasres['-lg(p)'] = -np.log10(gwasres.p)
            gwasres = gwasres[gwasres.bp.between(pos_start, pos_end) & gwasres.trait.isin(qtlssm.trait) ].sort_values('-lg(p)')
            yrange = (-.05,max(6, gwasres['-lg(p)'].max()+.5))
        else: gwasres =  pd.DataFrame(columns = ['Chr', 'SNP', 'bp', 'A1', 'A2', 'Freq', 'b', 'se', 'p', 'trait'])
        xrange = (pos_start, pos_end)
        fig_list = []
        if len(gwasres):
            gwasfig = gwasres.hvplot.scatter(x='bp', y='-lg(p)' ,frame_width = 1000, height = 600, width = 1900,
                                             xaxis='top',line_color='black', line_width = 3, color='black', cmap=d)*\
                      gwasres.hvplot.scatter(x='bp', y='-lg(p)' ,frame_width = 1000, height = 600, width = 1900,
                                             xaxis='top',line_color='black', line_width = .001, color='trait', cmap=d)
            gwasfig = gwasfig*hv.HLine(self.threshold).opts(color='red')*hv.HLine(self.threshold05).opts(color='blue')
            if len(qtlssm):
                gwasfig = gwasfig*qtlssm.rename({'p':'-lg(p)'}, axis = 1)\
                                        .hvplot.scatter(x='bp', y='-lg(p)', marker = 'star',  size = 1000, 
                                                        line_color='Black',  line_width = .5, c= 'trait', cmap=d)
            gwasfig = gwasfig.opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'),xlim = xrange, ylim = yrange, xaxis = 'top' )
            fig_list += [gwasfig]
        pval_threshold: float = 1e-4
        phewas_file, bsname = self.phewas_db, ''
        db_vals = pd.concat(pd.read_parquet(x).query(f'p < {pval_threshold}').assign(phewas_file = x) for x in phewas_file.split(',')).reset_index(drop= True)
        db_vals = db_vals[db_vals.bp.between(pos_start, pos_end) & (db_vals.Chr == int(c)) & ~db_vals.trait.isin(qtlssm.trait) ]
        db_vals['-lg(p)'] = -np.log10(db_vals.p)
        db_vals['size'] = (db_vals['b'].abs()+1)*20
        db_vals = db_vals.sort_values('-lg(p)')
        yticksdb = [(x, x[:10]) for x in db_vals.trait.unique()]
        if len(db_vals):
            db_vals_piv = db_vals.sort_values('-lg(p)')\
                                 .drop_duplicates(subset =  ['project', 'trait', 'SNP'], keep = 'last')\
                                 .pivot(columns = 'SNP', index = ['project', 'trait'], values = '-lg(p)')\
                                 .fillna(0)
            if db_vals_piv.shape[0] < 2: db_vals_piv = pd.concat([db_vals_piv, db_vals_piv.map(lambda x: x+1)])
            db_vals_piv = pd.DataFrame(PCA(n_components=min(10,db_vals_piv.shape[0] )).fit_transform(db_vals_piv),
                                       index = db_vals_piv.index)
            order = leaves_list(linkage(db_vals_piv))
            final_order_dict = {k:v for k,v in zip(db_vals_piv.index, order)}
            db_vals['final_order'] = db_vals.apply(lambda row: final_order_dict[(row.project, row.trait)], axis = 1)
            db_vals = db_vals.sort_values('final_order')
            phewasfig = db_vals.hvplot.scatter(x= 'bp',y= 'trait', s =  'size', c = '-lg(p)', cmap = 'gnuplot2_r', frame_width = 1000, height = 400, width = 1900)\
                               .opts(xlim = xrange, xaxis = None if len(fig_list) else 'top')\
                               .opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'),\
                         yaxis='right',  colorbar_position='left', yrotation=360-45)
            fig_list += [phewasfig]
            #yticks = yticksdb if len(yticksdb)<30 else None,
        #xticks='top'
        UMAP_res = UMAP_HDBSCAN_(plinkpath=self.genotypes_subset, c=c, pos_start=pos_start, decompose_samples = decompose_samples, subsample = None,
                                      pos_end=pos_end, founderbimfambed = founderbimfambed,keep_xaxis = False if len(fig_list) else True )
        hdbclus = UMAP_res['dataframes'][0]
        hdbfig, r2fig = UMAP_res['figures'][:2]
        add_lis = []
        for num, i in enumerate([f'{self.path}results/eqtl/pretty_eqtl_table.csv',
                                 f'{self.path}results/sqtl/sqtl_table.csv',
                                 f'{self.path}results/phewas/pretty_table_both_match.tsv']):
            if os.path.exists(i): 
                _ = pd.read_csv(i, sep = '\t' if i[-3:] == 'tsv' else ',').rename({'NearbySNP': 'SNP', 'SNP_PheDb': 'SNP'}, axis = 1)
            else: _ = pd.DataFrame(columns = ['SNP', "NONE"])
            if len(tdf := hdbclus.merge(_.set_index('SNP'), suffixes = ('', '_D'),
                             left_index = True,  right_index=True, 
                             how = 'inner')):
                ff_ = tdf.reset_index(names = 'SNP').hvplot.scatter(x = 'bp', y= 'clusternum', marker = ['diamond', 'triangle', 'square'][num], size = 200,
                                    color = 'gray', line_color = 'white', hover_cols = _.columns.to_list())
                add_lis += [ff_]
        if len(tdf := hdbclus.merge(qtlssm.set_index('SNP'), suffixes = ('', '_D'),
                             left_index = True,  right_index=True, 
                             how = 'inner')):
            add_lis += [tdf.reset_index(names = 'SNP').hvplot.scatter(x = 'bp', y= 'clusternum', marker = 'star', size = 500,
                                    color = 'black', line_color = 'white', hover_cols = ['SNP', 'trait'])]
        
        a1a2 = npplink.load_plink(self.genotypes_subset)[0].rename(str.upper, axis = 1)[['SNP', 'A0', 'A1']].set_index('SNP')
        a1a2 = pd.concat([hdbclus, a1a2], axis = 1, join = 'inner')
        a1a2 = self.annotate(a1a2.reset_index(names = 'SNP'), refcol= 'A1', altcol='A0')\
                    .query('putative_impact not in ["MODIFIER", "LOW"]')
        a1a2.bp = a1a2.bp.astype(int)
        a1a2fig = a1a2.hvplot.scatter(x = 'bp', y= 'clusternum', marker = 'square', size = 100,
                                    color = 'yellow', line_color = 'black', hover_cols = a1a2.columns.to_list())
        hdbfig *= a1a2fig
        if len(add_lis): hdbfig *= reduce(lambda x, y: x*y, add_lis)
        genesfig = self.make_genetrack_figure_(c=c, pos_start=pos_start, pos_end=pos_end )
        fig_list += [hdbfig.opts(yaxis = 'right')]
        if add_r2fig: fig_list += [r2fig]
        fig_list += [genesfig.opts(yaxis = 'right')]
        if return_fig_as_list: return fig_list 
        fig = pn.Column(*fig_list)
        return fig
    
    def make_phewas_figs(self, qtls = None, padding = 2e6, save = True, decompose_samples_hdbscan = False):
        os.makedirs(f'{self.path}images/phewasfigures/', exist_ok=True)
        if qtls is None: qtls = pd.read_csv(f'{self.path}results/qtls/finalqtl.csv')
        snp_connectivity = pd.DataFrame((spa.distance_matrix(qtls[['bp']],qtls[['bp']], p = 1) < padding) * \
                    (spa.distance_matrix(qtls[['Chr']].astype(int),qtls[['Chr']].astype(int), p = 1) == 0),
                    index = qtls.SNP, columns = qtls.SNP)
        qtls['connectivity'] = [f'Comp{x}' for x in sps.csgraph.connected_components(csgraph=sps.csr_matrix(snp_connectivity), directed=False, return_labels=True)[1]]
        qtlsgroups = qtls.groupby('connectivity')\
            .apply(lambda df: pd.Series(dict (Chr = df.Chr.mode()[0], connect = df.name,
                                              lower = int(df.bp.min() - padding),
                                              upper =int(df.bp.max()+ padding),
                                              snps = ','.join(df.SNP) )))
        figlis = []
        for idx, row in qtlsgroups.iterrows(): 
            print(dict(c = int(row.Chr), pos_start=row.lower,  pos_end=row.upper,   decompose_samples = decompose_samples_hdbscan))
            fig  = self.make_phewas_figure_single(c = int(row.Chr), 
                                                 pos_start=row.lower, 
                                                 pos_end=row.upper,
                                                 decompose_samples = decompose_samples_hdbscan,
                                            founderbimfambed= self.foundersbimfambed if len(self.foundersbimfambed) else None)
            if save: 
                fig.save(f'{self.path}images/phewasfigures/{row.connect}_{row.Chr}_{row.lower}_{row.upper}.html')
                #hv.save(fig, f'{self.path}images/phewasfigures/{row.connect}_{row.Chr}_{row.lower}_{row.upper}.html')
            figlis += [fig]
        qtlsgroups['fig'] = figlis
        #qtlsgroups['fig'] = qtlsgroups.fig.map(lambda x: pn.pane.HoloViews(x))
        return qtlsgroups

    def run(self,
            round_version: str ='genotypes_test', regressout: bool = True, add_gwas_to_db: bool = True,
            gwas_version: str = None,
            researcher: str = 'user', impute_missing_trait: bool = False, clear_directories: bool = False, 
            groupby_animals: list = [], regressout_timeseries: bool = False, add_latent_space: bool = True,
            add_sex_specific_traits:bool = False,
            **kws):
        if gwas_version is None: 
            gwas_version = __version__ #+ datetime.today().strftime(format = '%Y_%m_%d')
        def kw(d, prefix):
            if prefix[-1] != '_': prefix += '_'
            return {k.replace(prefix, '') : typeconverter(v) for k,v in d.items() if (k[:len(prefix)] == prefix)}
        if clear_directories: self.clear_directories()
        if add_sex_specific_traits:
            self.add_sex_specific_traits(save = True)
        if impute_missing_trait: 
            self.impute_traits(groupby_columns=groupby_animals, **kw(kws, 'impute_traits_'))
        if regressout:
            if not regressout_timeseries:  
                if not groupby_animals: self.regressout(data_dictionary= pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv'))
                else: self.regressout_groupby(data_dictionary= pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv'), groupby_columns=groupby_animals)
            elif regressout_timeseries:  
                if not groupby_animals: self.regressout_timeseries(data_dictionary=pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv'))
                else: self.regressout_timeseries(data_dictionary= pd.read_csv(f'{self.path}data_dict_{self.project_name}.csv'), groupby_columns=groupby_animals)
        if add_latent_space: self.add_latent_spaces()
        self.SubsetAndFilter(makefigures = False, **kw(kws, 'SubsetAndFilter_'))
        self.generateGRM(**kw(kws, 'generateGRM_'))
        self.snpHeritability() ###essential
        self.fastGWAS(**kw(kws, 'fastGWAS_')) ###essential
        if add_gwas_to_db:self.addGWASresultsToDb(researcher=researcher, round_version=round_version, gwas_version=gwas_version)
        qtl_add_founder = isinstance(self.foundersbimfambed, tuple) and len(self.foundersbimfambed) == 3
        try:    qtls = self.callQTLs( NonStrictSearchDir=False, add_founder_genotypes = qtl_add_founder, **kw(kws, 'callQTLs_'))
        except: qtls = self.callQTLs( NonStrictSearchDir=True, **kw(kws, 'callQTLs_'))
        self.effectsize(display_plots = False) 
        self.genetic_correlation_matrix(**kw(kws, 'genetic_correlation_matrix_'))
        self.make_heritability_figure()
        self.phewas(**kw(kws, 'phewas_'))  ###essential
        self.eQTL(**kw(kws, 'eQTL_')) ###essential
        self.sQTL(**kw(kws, 'sQTL_')) ###essential
        self.GeneEnrichment() ###essential
        self.locuszoom(**kw(kws, 'locuszoom_'))  ###essential
        self.make_phewas_figs()
        self.report(round_version=round_version, gwas_version=gwas_version, **kw(kws, 'report_'))
        self.store(researcher=researcher, round_version=round_version, gwas_version=gwas_version, remove_folders=False) ###essential
        try:    self.copy_results() ###essential
        except: print('setting up the minio is necessary')
        self.print_watermark()

def get_trait_descriptions_f(data_dic, traits):
    """
    Retrieve trait descriptions from a data dictionary.

    Steps:
    1. Initialize an empty list outout to store descriptions.
    2. Iterate over each trait in the traitstraits list.
    3. For each trait, attempt to find its description in the data dictionary.
    4. Append the found description to the outout list. If not found, append 'UNK'.

    :param data_dic: DataFrame containing trait information.
    :param traits: List of traits for which descriptions are needed.
    :return: List of descriptions for the specified traits. If a description is not found, 'UNK' is returned for that trait.

    Example:
    >>> data_dic = pd.DataFrame({
    ...     'measure': ['trait1', 'trait2'],
    ...     'description': ['Description for trait1', 'Description for trait2']
    ... })
    >>> traits = ['trait1', 'trait3']
    >>> get_trait_descriptions_f(data_dic, traits)
    ['Description for trait1', 'UNK']
    """
    out = []
    for trait in traits:
        try: out +=  [data_dic[data_dic.measure == trait.replace('regressedlr_', '')].description.iloc[0]]
        except: out +=  ['UNK']
    return out
        
def display_graph(G):
    """
    Display a graph using Matplotlib and Graphviz layout.

    Steps:
    1. Create a Matplotlib figure.
    2. Use Graphviz to layout the graph.
    3. Iterate over each connected component of the graph.
    4. Draw the graph with random colors for each connected component.

    :param G: NetworkX Graph to be displayed.
    :return: None

    Example:
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    >>> display_graph(G)
    (Displays the graph)
    """
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
    """
    Find the longest path in a directed acyclic graph (DAG) and assemble a sequence.

    Steps:
    1. Find the longest path in the graph.
    2. Concatenate the nodes in the path to form the sequence.

    :param graph: NetworkX DiGraph representing the directed acyclic graph.
    :return: The assembled sequence.

    Example:
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    >>> assembly(G)
    '1234'
    """
    R = nx.dag_longest_path(nx.DiGraph(graph), weight='weight', default_weight=1)
    return R[0] + ''.join([x[-1] for x in R[1:]])

def assembly_complete(graph):
    """
    Assemble sequences for all connected components of the graph.

    Steps:
    1. Create subgraphs for each connected component.
    2. Assemble sequences for each subgraph.

    :param graph: The graph to be assembled.
    :type graph: networkx.Graph
    :return: Array of assembled sequences for each connected component.
    :rtype: numpy.ndarray

    Example:
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    >>> assembly_complete(G)
    array(['123', '45'])
    """
    S = [nx.DiGraph(graph.subgraph(c).copy()) for c in nx.connected_components(graph.to_undirected())]
    return np.array([assembly(subg) for subg in S])

def tuple_to_graph_edge(graph, kmer1, kmer2,  ide):
    """
    Add an edge between two k-mers in the graph or update the weight if the edge already exists.

    Steps:
    1. Check if an edge exists between kmer1kmer1 and kmer2kmer2.
    2. If the edge exists, increment its weight and update the label.
    3. If the edge does not exist, add it to the graph with an initial weight and label.

    :param graph: The graph to which the edge will be added.
    :type graph: networkx.Graph
    :param kmer1: The first k-mer.
    :param kmer2: The second k-mer.
    :param ide: Identifier for the edge label.
    """
    if graph.has_edge(kmer1, kmer2):  
        graph[kmer1][kmer2]['weight'] += 1 
        graph[kmer1][kmer2]['label'][ide] += 1
    else: graph.add_edge(kmer1, kmer2, weight=1, label = defaultdict(int, {ide:1}))
    return

def dosage_compensation(a, male_indices):
    """
    Apply dosage compensation to the given array for male indices.

    Steps:
    1. Copy the input array aa to a new array bb.
    2. Multiply the values at male indices by 2 in the array bb.

    :param a: Input array.
    :type a: numpy.ndarray
    :param male_indices: Indices corresponding to males.
    :type male_indices: list of int
    :return: Array with dosage compensation applied to male indices.
    :rtype: numpy.ndarray

    Example:
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> male_indices = [1, 3]
    >>> dosage_compensation(a, male_indices)
    array([ 1,  4,  3,  8,  5])
    """
    b = a.copy()
    b[male_indices] *= 2
    return b

def find_overlap(r1, r2):
    """
    Find overlapping indices between two lists.

    Steps:
    1. Iterate over indices and values in r1r1.
    2. Check if the value in r1r1 is greater than or equal to the minimum value in r2r2.
    3. Create a dictionary mapping the overlapping indices.

    :param r1: First list of indices.
    :type r1: list of int
    :param r2: Second list of indices.
    :type r2: list of int
    :return: Dictionary of overlapping indices.
    :rtype: dict

    Example:
    >>> r1 = [1, 2, 3, 4]
    >>> r2 = [3, 4, 5, 6]
    >>> find_overlap(r1, r2)
    {2: 0, 3: 1}
    """
    ol = [num for num, val in enumerate(r1.i_list) if val >= r2.i_min]
    return {int(k):int(v) for v, k in enumerate(ol)}
    
def generate_umap_chunks(chrom, win: int = int(2e6), overlap: float = .5,
                        nsampled_snps: int = 40000,nsampled_rats: int = 20000,
                        random_sample_snps: bool = False,nautosomes=20,
                        founderfile = '/tscc/projects/ps-palmer/gwas/databases/founder_genotypes/founders7.2',
                        latest_round = '/tscc/projects/ps-palmer/gwas/databases/rounds/r10.5.2',
                        pickle_output = False, impute = False,
                        save_path = ''):
    """
    Generate UMAP chunks for a specified chromosome with various parameters.

    Steps:
    1. Print a starting log message.
    2. Read PLINK files for founder and latest round data.
    3. Filter SNPs and determine start and stop positions.
    4. Sample SNPs either randomly or based on regular intervals.
    5. Create positional and SNP-based DataFrames.
    6. Calculate overlapping windows and relationships.
    7. Get genotype data and apply dosage compensation for males if necessary.
    8. Perform KNN imputation and scaling if specified.
    9. Run UMAP and aligned UMAP for the genotypes.
    10. Predict labels using LabelSpreading.
    11. Generate and save 3D and 2D visualizations of the UMAP results.
    12. Save results to disk if specified.

    :param chrom: Chromosome number to process.
    :param win: Window size for each UMAP chunk.
    :param overlap: Overlap percentage between consecutive windows.
    :param nsampled_snps: Number of SNPs to sample.
    :param nsampled_rats: Number of rats to sample.
    :param random_sample_snps: Whether to sample SNPs randomly.
    :param nautosomes: Number of autosomes.
    :param founderfile: Path to the founder genotype PLINK files.
    :param latest_round: Path to the latest round genotype PLINK files.
    :param pickle_output: Whether to save the output as a pickle file.
    :param impute: Whether to perform KNN imputation on the genotype data.
    :param save_path: Path to save the output files.
    :return: DataFrame containing UMAP chunk information and embeddings.
    :rtype: pandas.DataFrame

    Example:
    >>> result = generate_umap_chunks(chrom=1, win=2e6, overlap=0.5,
    ...                               nsampled_snps=40000, nsampled_rats=20000,
    ...                               random_sample_snps=False, nautosomes=20,
    ...                               founderfile='/path/to/founderfile',
    ...                               latest_round='/path/to/latest_round',
    ...                               pickle_output=False, impute=True,
    ...                               save_path='/save/path/')
    """
    print(f'starting umap chunks for {chrom}, nsnps = {nsampled_snps}, maxrats = {nsampled_rats}')
    if type(founderfile) == str: bimf, famf, genf = npplink.load_plink(founderfile)
    else: bimf, famf, genf = founderfile
    if type(latest_round) == str: bim, fam, gen = npplink.load_plink(latest_round)
    else: bim, fam, gen = latest_round
    bimf = bimf.assign(i = bimf.index)
    
    bim1 = bim[bim.chrom.isin([chrom, str(chrom)]) & bim.snp.isin(bimf.snp)]#.query('chrom == "12"')
    start, stop = bim1['i'].agg(['min', 'max']).values
    if random_sample_snps: 
        sampled_snps = sorted(np.random.choice(range(start, stop), size = min(nsampled_snps, gen.shape[1] ) , replace = False))
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
        allowed_rats = range(gen.shape[0])
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
    out['genotypes'] = out.snp_list.progress_apply(lambda x: np.vstack([gen[sampled_rats][:, bim1isnp.loc[x].i ], 
                                                                  genf[:, bimf1isnp.loc[x].i]]).astype(np.float16).compute())#

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
    else:  
        metr = nan_euclidean_distances
        out['genotypes'] = out['genotypes'].progress_apply(lambda x: np.nan_to_num(make_pipeline( StandardScaler() )\
                                                           .fit_transform(x)))
    
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
    
    founder_colors =  defaultdict(lambda: 'white', {'BN': '#000000', 'ACI':'#D32F2F', 'MR': '#8E24AA', 'M520':'#FBC02D',
                      'F344': '#388E3C', 'BUF': '#1976D2', 'WKY': '#F57C00', 'WN': '#02D3D3'})
    
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

from sklearn.model_selection import KFold, cross_validate
from sklearn.base import clone
class LGBMImputer:
    def __init__(self, window = 100, qc:bool = True, device = 'cpu', classifier = None, regressor = None, silent = False) :
        #force_col_wise=True,
        self.map_score = {True:[ 'explained_variance', 'max_error',  'neg_mean_absolute_error', 'r2'],
            False:['accuracy', 'balanced_accuracy',  'f1_weighted'] } # 'precision_weighted',  'recall_weighted', 'roc_auc_ovo'
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42 )
        self.model_table = pd.DataFrame()
        self.window = window
        self.device = device
        self.qc= qc
        self.needs_rename = True if classifier is None else False
        self.classifier = classifier if classifier is not None else LGBMClassifier( device=self.device, feature_fraction=.5,verbosity= -1, class_weight ='balanced', n_jobs =-1)
        self.regressor = regressor if regressor is not None else LGBMRegressor( device=self.device, feature_fraction=1.,verbosity= -1, n_jobs =-1)
        self.silent = silent
    def _compute(self, X, ret, columns_subset=None):
        Xnew = X.copy()
        if (empty_cols := Xnew.isna().all(axis = 0)).any(): 
            if not self.silent: print('[warning] dropping fully empty columns: ' + '|'.join(Xnew.columns[empty_cols]))
            Xnew = Xnew.loc[:, ~empty_cols]
        if self.needs_rename and (Xnew.columns == Xnew.columns.map(lambda x: re.sub(r'\W', '_', x))).all():
            self.needs_rename = False
        else: 
            if not self.silent: print(r'[warning] \W characters in column names will be replaced with _ when fitting the model')
        self.model_table = pd.DataFrame()
        if columns_subset is not None:
            columns_subset = Xnew.columns[Xnew.columns.isin(columns_subset)]
            if not len(columns_subset): 
                if not self.silent: print(r'[warning] subset of columns not present in dataset, using the whole set of columns')
                columns_subset = Xnew.columns
        else: columns_subset = Xnew.columns
        self.model_table.index = pd.Series(Xnew.columns, name = 'features')
        self.model_table['isnumeric'] = Xnew.dtypes.map(pd.api.types.is_numeric_dtype)
        self.model_table['model'] = self.model_table.isnumeric.map({True: clone(self.regressor), False:clone(self.classifier)})
        self.model_table['num'] = range(self.model_table.shape[0])
        self.model_table['columns'] = self.model_table.num.map(lambda x: list(self.model_table.index[x-self.window:x]) \
                                                                         +list(self.model_table.index[x+1:x+self.window]))
        self.model_table = self.model_table.loc[columns_subset]
        for yname, row in tqdm(self.model_table.iterrows(), total=len(self.model_table), disable = self.silent):
            cols = row.columns
            mod = row.model
            scorers = self.map_score[row.isnumeric]
            toimpute = X[yname].isna()
            X2fit = X.loc[~toimpute,cols]
            y2fit = X.loc[~toimpute,yname]
            Xpredict = X.loc[toimpute,cols]
            if self.needs_rename:
                X2fit.columns = X2fit.columns.str.replace(r'\W', '_', regex = True)
            if toimpute.sum():
                imputed_vals = pd.DataFrame(mod.fit(X2fit, y2fit).predict(Xpredict), columns =[ yname], index = X.index[toimpute] )
                Xnew.update(imputed_vals)
            if self.qc: 
                    row_qc = pd.DataFrame(cross_validate(mod, X = X2fit, y = y2fit, scoring = scorers, cv =self.kf, return_train_score= True))\
                               .mean().to_frame().T.set_axis([yname]).assign(n = X2fit.shape[0] )
                    self.model_table.loc[row_qc.index, row_qc.columns] = row_qc
            else:
                if not toimpute.sum(): mod.fit(X2fit, y2fit)
                self.model_table.loc[yname, scorers + ['n'] ] = [get_scorer(name)(mod, X2fit, y2fit) for name in scorers] + [X2fit.shape[0]]
                    
        if ret == 'qc': return self.model_table
        if ret == 'imputed': return Xnew
        return self
    def fit(self, X, columns_subset=None):
        return self._compute(X, ret = 'self',columns_subset = columns_subset)
    def fit_transform(self, X, columns_subset=None):
        return self._compute(X, ret = 'imputed',columns_subset = columns_subset)
    def transform(X):
        Xnew = X.copy()
        for yname, row in tqdm(self.model_table.iterrows(), total=len(self.model_table),  disable = self.silent):
            cols = row.columns
            mod = row.model
            toimpute = X[yname].isna()
            if toimpute.sum():
                Xpredict = X.loc[toimpute,cols]
                if self.needs_rename:
                    Xpredict.columns = Xpredict.columns.str.replace(r'\W', '_', regex = True)
                imputed_vals = pd.DataFrame(mod.predict(Xpredict), columns =[ yname], index = X.index[toimpute] )
                newX.update(imputed_vals)
        return Xnew




def double_sided_randomized_svd(X, n_components=100, oversample=10, random_state=None):
    """
    Compute an approximate rank-k SVD of X using double-sided random projections.
    
    Parameters
    ----------
    X : ndarray
        Input matrix of shape (n_samples, n_features).
    k : int
        Target rank (number of singular values/vectors to retain).
    oversample : int, default 10
        Additional oversampling dimension to improve the quality of the approximation.
    random_state : int or None
        Seed for reproducibility.
        
    Returns
    -------
    U_hat : ndarray, shape (n_samples, k)
        Approximate left singular vectors.
    sigma : ndarray, shape (k,)
        Approximate singular values.
    Vt_hat : ndarray, shape (k, n_features)
        Approximate right singular vectors transposed.
        
    Such that:
        X ≈ U_hat @ diag(sigma) @ Vt_hat
    """
    k = min(n_components, min(X.shape) )
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    L = k + oversample  # effective projection dimension
    
    # Generate random Gaussian projection matrices:
    # T projects the column space (size d x L)
    T = rng.standard_normal(size=(d, L))
    # S projects the row space (size n x L)
    S = rng.standard_normal(size=(n, L))
    
    # Form the sketch matrix:
    # Y = S^T X T has shape (L, L) and captures the main structure of X.
    Y = S.T @ X @ T  # shape (L, L)
    
    # Compute the SVD of the small sketch matrix Y:
    U_y, sigma_all, V_yT = np.linalg.svd(Y, full_matrices=False)
    
    # Truncate to the desired rank k.
    sigma = sigma_all[:k]
    U_y = U_y[:, :k]      # shape (L, k)
    V_y = V_yT.T[:, :k]   # shape (L, k)
    
    # "Lift" the factors to approximate the singular vectors of X:
    # Approximate left singular vectors:
    U_hat = X @ T @ V_y  # shape (n, k)
    U_hat = U_hat / sigma[np.newaxis, :]  # scale each column by its singular value
    
    # Approximate right singular vectors:
    V_hat = X.T @ S @ U_y  # shape (d, k)
    V_hat = V_hat / sigma[np.newaxis, :]  # scale each column by its singular value
    
    return U_hat, sigma, V_hat.T  # Vt_hat = V_hat.T


def corrSVD(df, doublesided = True, rank=50, random_state=42):
    """
    Computes an approximate correlation matrix (features x features) 
    using randomized SVD while handling missing values without external imputation.
    
    This function computes column means and standard deviations ignoring NaNs,
    centers and scales the data, then replaces missing values (which become zero
    after centering) with zero before computing the SVD. In parallel, it computes
    an effective sample size for each pair of features based on non-missing overlaps.
    The low-rank approximation of the covariance is then normalized elementwise
    by these effective counts, and finally converted to a correlation matrix.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples as rows and features as columns.
    rank : int, optional
        Target rank for the randomized SVD approximation.
    random_state : int, optional
        Random state for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        A (C x C) DataFrame containing the approximate correlation matrix.
    """
    # Convert the DataFrame to a numpy array
    X = df.values.astype(np.float64)
    
    # Compute an indicator matrix: 1 if not nan, 0 if nan
    indicator = (~np.isnan(X)).astype(np.float64)
    
    # Compute column means and standard deviations ignoring NaNs
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant columns
    
    # Center and scale (standardize) the data
    X_centered = X - means
    X_std = X_centered / stds
    
    # For SVD, replace NaNs with 0 (which is equivalent to mean imputation after centering)
    X_std_imputed = np.where(np.isnan(X_std), 0.0, X_std)
    
    # Compute effective sample counts for each pair of features:
    # This is the dot product of the indicator matrix with its transpose.
    effective_n = np.dot(indicator.T, indicator)
    effective_n[effective_n == 0] = np.nan  # avoid division by zero
    
    # Compute a low-rank randomized SVD of the standardized, imputed matrix.
    if not doublesided: U, S, Vt = randomized_svd(X_std_imputed, n_components=rank, random_state=random_state)
    else:  U, S, Vt = double_sided_randomized_svd(X_std_imputed, n_components=rank, random_state=random_state)
    
    # The full covariance (for standardized data) would be X_std.T @ X_std.
    # Here we approximate that by: Vt.T @ diag(S**2) @ Vt,
    # then we normalize elementwise by the effective sample count.
    approx_cov = np.dot(Vt.T * (S**2), Vt)
    approx_cov = approx_cov / effective_n  # elementwise division for each pair
    
    # Convert the covariance matrix to a correlation matrix by normalizing each entry.
    diag = np.sqrt(np.diag(approx_cov))
    norm_matrix = np.outer(diag, diag)
    approx_corr = approx_cov / norm_matrix
    
    # Clip values to [-1, 1] for numerical stability.
    approx_corr = np.clip(approx_corr, -1, 1)
    
    # Return a DataFrame with the same feature labels.
    return pd.DataFrame(approx_corr, index=df.columns, columns=df.columns)

def set_nan_color(plot, element):
    color_mapper = plot.handles.get('color_mapper', None)
    if color_mapper is not None:
        color_mapper.nan_color = 'black'




import numpy as np, pandas as pd, os, re, math
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (r2_score, accuracy_score,
                             explained_variance_score,
                             max_error, mean_absolute_error,
                             balanced_accuracy_score, f1_score)
from lightgbm import LGBMRegressor, LGBMClassifier

# turbo_lgbm_imputer.py
import numpy as np, pandas as pd, os, re, math
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import (r2_score, explained_variance_score, max_error,
                             mean_absolute_error, accuracy_score,
                             balanced_accuracy_score, f1_score)
from lightgbm import LGBMRegressor, LGBMClassifier


class TurboLGBMImputer:
    """
    Fast LightGBM-based imputer with:
      • early-stopping       • row subsampling        • column bagging
      • multi-output blocks  • prediction-only CV     • *iterative passes*

    Parameters
    ----------
    n_iter : int, default 1
        Number of complete imputation sweeps.  If >1, the freshly
        imputed matrix is fed back into the next sweep.
    atol : float, default 1e-4
        Absolute tolerance for early convergence between passes.
    window : int
        Predictor window size each side of the target column.
    qc : bool
        If True, compute “prediction-only” fold scores (cheap).
    max_rows : int or None
        Cap on rows used per target/block (after removing missing).
    block_size : int
        Number of numeric targets fitted together in a MultiOutputRegressor.
    early_stopping_rounds, feature_fraction, n_estimators, n_jobs_outer,
    device_type, **lgbm_kw
        Passed through to LightGBM learners.
    """

    _KF = KFold(n_splits=5, shuffle=True, random_state=42)

    _NUM_SCORERS = dict(explained_variance=explained_variance_score,
                        max_error=max_error,
                        neg_mean_absolute_error=lambda y, p:
                            -mean_absolute_error(y, p),
                        r2=r2_score)

    _CAT_SCORERS = dict(accuracy=accuracy_score,
                        balanced_accuracy=balanced_accuracy_score,
                        f1_weighted=lambda y, p:
                            f1_score(y, p, average="weighted"))

    def __init__(self,
                 *,
                 n_iter: int = 1,
                 atol: float = 1e-4,
                 window: int = 100,
                 qc: bool = False,
                 max_rows: int | None = None,
                 block_size: int = 10,
                 early_stopping_rounds: int = 50,
                 feature_fraction: float = 0.6,
                 n_estimators: int = 400,
                 n_jobs_outer: int | None = None,
                 device_type: str = "cpu",
                 **lgbm_kw):

        if n_iter < 1:
            raise ValueError("n_iter must be ≥ 1")

        self.n_iter   = n_iter
        self.atol     = atol
        self.window   = window
        self.qc       = qc
        self.max_rows = max_rows
        self.block_sz = max(1, block_size)
        self.es_rounds= early_stopping_rounds
        self.n_jobs_outer = n_jobs_outer or os.cpu_count()

        # base LightGBM estimators
        self._reg = LGBMRegressor(device_type=device_type,
                                  n_estimators=n_estimators,
                                  n_jobs=1,
                                  feature_fraction=feature_fraction,
                                  early_stopping_rounds=self.es_rounds,
                                  verbosity=-1,
                                  **lgbm_kw)

        self._clf = LGBMClassifier(device_type=device_type,
                                   n_estimators=n_estimators,
                                   n_jobs=1,
                                   class_weight='balanced',
                                   feature_fraction=feature_fraction,
                                   early_stopping_rounds=self.es_rounds,
                                   verbosity=-1,
                                   **lgbm_kw)

        # artefacts filled during last pass
        self.models_: dict[str, object] = {}
        self.model_table: pd.DataFrame = pd.DataFrame()

    def fit_transform(self, X: pd.DataFrame, columns_subset=None) -> pd.DataFrame:
        if columns_subset is not None:
            missing = set(columns_subset) - set(X.columns)
            if missing:
                raise KeyError(f"columns_subset not in DataFrame: {missing}")
            target_mask = X.columns.isin(columns_subset)
        else:
            target_mask = np.repeat(True, X.shape[1])

        imputed = X.copy()
        prev    = None
        for it in range(self.n_iter):
            if it:
                print(f"[TurboLGBMImputer] pass {it+1}/{self.n_iter}")
            imputed = self._single_pass(imputed, target_mask) 
            if prev is not None and np.allclose(prev.values,
                                                imputed.values,
                                                atol=self.atol,
                                                equal_nan=True):
                print(f"[TurboLGBMImputer] converged at pass {it+1}")
                break
            prev = imputed.copy()
        return imputed

    def fit(self, X: pd.DataFrame, columns_subset=None):
        self.fit_transform(X, columns_subset)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xnew = X.copy()
        for feat, row in self.model_table.dropna(how="all").iterrows():
            miss = Xnew[feat].isna()
            if miss.any():
                cols = row["columns"]
                Xnew.loc[miss, feat] = self.models_[feat].predict(
                    Xnew.loc[miss, cols])
        return Xnew

    def _wins(self, p: int) -> list[np.ndarray]:
        return [np.r_[max(0, i-self.window):i,
                      i+1:min(p, i+self.window+1)] for i in range(p)]

    @staticmethod
    def _train_val_split(idx: np.ndarray):
        if len(idx) < 10:
            return idx, np.array([], int)
        cut = math.ceil(0.8 * len(idx))
        return idx[:cut], idx[cut:]

    def _fit_numeric_block(self, X, miss, targ_idx, cols_idx):
        rows = np.flatnonzero(~miss[:, targ_idx].any(axis=1))
        if rows.size == 0:
            return None
        if self.max_rows and len(rows) > self.max_rows:
            rows = np.random.choice(rows, self.max_rows, replace=False)

        Xtr = X[rows][:, cols_idx]
        Ytr = X[rows][:, targ_idx]

        tr, val = self._train_val_split(np.arange(len(rows)))
        eval_set = [(Xtr[val], Ytr[val])] if val.size else None

        est = MultiOutputRegressor(clone(self._reg))
        est.fit(Xtr[tr], Ytr[tr], eval_set=eval_set, verbose=False)

        preds_tr = est.predict(Xtr)
        train_scores = [r2_score(Ytr[:, k], preds_tr[:, k])
                        for k in range(len(targ_idx))]

        cv_scores = None
        if self.qc:
            cv_scores = {nm: np.zeros(len(targ_idx))
                         for nm in self._NUM_SCORERS}
            for tr_idx, te_idx in self._KF.split(Xtr):
                pr = est.predict(Xtr[te_idx])
                for k in range(len(targ_idx)):
                    for nm, fn in self._NUM_SCORERS.items():
                        cv_scores[nm][k] += fn(Ytr[te_idx, k], pr[:, k])
            cv_scores = {k: v/self._KF.get_n_splits() for k, v in cv_scores.items()}

        return est, train_scores, cv_scores

    def _single_pass(self, Xdf: pd.DataFrame) -> pd.DataFrame:
        Xnp   = Xdf.to_numpy(np.float32, copy=True)
        miss  = np.isnan(Xnp)
        n, p  = Xnp.shape
        wins  = self._wins(p)
        isnum = Xdf.dtypes.map(pd.api.types.is_numeric_dtype).to_numpy()
        names = Xdf.columns.to_numpy()

        # build blocks
        num_cols = [i for i, b in enumerate(isnum) if b]
        cat_cols = [i for i, b in enumerate(isnum) if not b]
        blocks   = [num_cols[i:i+self.block_sz]
                    for i in range(0, len(num_cols), self.block_sz)]
        blocks  += [[j] for j in cat_cols]          # categorical singletons

        def _process_block(tidx):
            cols_idx = np.unique(np.concatenate([wins[j] for j in tidx]))
            if len(tidx) > 1:                       # numeric group
                res = self._fit_numeric_block(Xnp, miss, tidx, cols_idx)
                return tidx, cols_idx, res
            # single target (num or cat)
            j = tidx[0]
            rows = np.flatnonzero(~miss[:, j])
            if rows.size == 0:
                return tidx, cols_idx, None
            if self.max_rows and len(rows) > self.max_rows:
                rows = np.random.choice(rows, self.max_rows, replace=False)

            Xtr, ytr = Xnp[rows][:, cols_idx], Xnp[rows, j]
            tr, val  = self._train_val_split(np.arange(len(rows)))
            eval_set = [(Xtr[val], ytr[val])] if val.size else None
            base = clone(self._reg if isnum[j] else self._clf)
            base.fit(Xtr[tr], ytr[tr], eval_set=eval_set, verbose=False)
            train_score = (r2_score if isnum[j] else accuracy_score)(
                ytr, base.predict(Xtr))

            cv_scores = None
            if self.qc:
                scorers = (self._NUM_SCORERS if isnum[j]
                           else self._CAT_SCORERS)
                sums = {k: 0. for k in scorers}
                for tr_idx, te_idx in self._KF.split(Xtr):
                    pr = base.predict(Xtr[te_idx])
                    for nm, fn in scorers.items():
                        sums[nm] += fn(ytr[te_idx], pr)
                cv_scores = {k: v/self._KF.get_n_splits() for k, v in sums.items()}

            return tidx, cols_idx, (base, [train_score], cv_scores)

        results = Parallel(self.n_jobs_outer, verbose=10)(
                      delayed(_process_block)(blk) for blk in blocks)

        records = []
        for tidx, cols_idx, payload in results:
            if payload is None: continue
            est, tr_scores, cv_scores = payload
            preds_full = est.predict(Xnp[:, cols_idx]) \
                if len(tidx) > 1 else est.predict(Xnp[:, cols_idx]).reshape(-1, 1)
            for k, j in enumerate(tidx):
                msk = miss[:, j]
                if msk.any():
                    Xnp[msk, j] = preds_full[msk, k]
                name = names[j]
                self.models_[name] = (est.estimators_[k]
                                      if hasattr(est, "estimators_")
                                      else est)
                rec = dict(feature=name,
                           isnumeric=bool(isnum[j]),
                           model=self.models_[name],
                           num=int(j),
                           columns=list(names[cols_idx]),
                           train_score=tr_scores[k])
                if cv_scores:
                    for nm, arr in cv_scores.items():
                        rec[nm] = arr[k] if isinstance(arr, np.ndarray) else arr
                records.append(rec)

        self.model_table = (pd.DataFrame(records)
                              .set_index("feature")
                              .reindex(Xdf.columns))

        return pd.DataFrame(Xnp, index=Xdf.index, columns=Xdf.columns)


def make_genetrack_figure_(self, c: str= None, pos_start: int = None, pos_end:int= None,  frame_width = 1000, simplify_genes=True):
    genecolors = {'transcript': 'black', 'exon': 'seagreen', 'CDS': 'steelblue','start_codon': 'firebrick' ,'stop_codon': 'firebrick'}
    genesizes = {'transcript': .05, 'exon': .25, 'CDS': .35,'start_codon': .45 ,'stop_codon': .45}
    gtf = self.get_gtf() if not hasattr(self, 'gtf') else self.gtf
    genes_in_section = self.gtf.query(f'Chr == {int(c)} and end > {pos_start} and start < {pos_end}')\
                                           .reset_index(drop = True)\
                                           .query("source not in ['cmsearch','tRNAscan-SE']")
    if simplify_genes:
        biotype_str = 'lncRNA|misc_RNA|pseudogene|RNase_|telomerase_RNA|snoRNA|rRNA|_RNA|tRNA|s[cn]RNA|other'
        gbkey_str = 'misc_RNA|ncRNA|precursor_RNA|rRNA|tRNA'
        genes_in_section = genes_in_section[~genes_in_section.gene_biotype.fillna('').str.contains(biotype_str) &\
            ~genes_in_section.gbkey.fillna('').str.contains(gbkey_str)]
    tt = genes_in_section.dropna(how = 'all', axis = 1).reset_index(drop= True).assign(y=1)
    if (not tt.shape[0]) or ('gbkey' not in tt.columns):
        resdf = pd.DataFrame(dict(genes=[0], bp = [(pos_start+pos_end)/2], label = ['no genes in section']))
        res = hv.Labels(resdf, kdims = ['bp','genes'], vdims=['label'])\
                .opts(yaxis = None,xformatter=NumeralTickFormatter(format='0,0.[0000]a'),\
                      xlabel = '', xlim = ( pos_start, pos_end), frame_width = frame_width, height = 100, text_color = 'Red')
        return res
    tt['mean_pos'] = tt[['start', 'end']].mean(axis = 1)
    tt = tt.assign(color = tt.biotype.map(genecolors), size = tt.biotype.map(genesizes), genestrand = tt.gene.str.replace('LOC', '') ) #+ tt.strand
    ttgenes = tt.query('gbkey == "Gene"').reset_index(drop = True)#.set_index('end')
    spacing_genes = (pos_end - pos_start)/(20 if len(ttgenes)<= 15 else 120)
    intergene_space = .36 if len(ttgenes)<= 15 else  .11*ttgenes.gene.map(len).max()
    for idx, row in ttgenes.iterrows():
        if idx == 0: ttgenes.loc[idx, 'stackingy'] = 0
        genes_inreg = set(ttgenes[min(idx-1000, 0): idx].query(f"end + {spacing_genes} > @row.start").stackingy)
        if idx>0: ttgenes.loc[idx, 'stackingy'] = min(set(range(1000)) - genes_inreg)
    yticks3 = (intergene_space*ttgenes[['stackingy','stackingy']]).round(2).agg(lambda x: tuple(x), axis = 1).to_list()
    yticks3l = [(x[0], '') for x in yticks3]
    size_scalers = ttgenes['stackingy'].max()*intergene_space
    stackgenepos = defaultdict(lambda: -1, (ttgenes.set_index('gene')['stackingy']*intergene_space).to_dict())
    tt = tt.assign(stackingy = tt.gene.map(stackgenepos), 
                   stackingy0 = tt.gene.map(stackgenepos) - tt['size']/5, #*size_scalers/10/2
                   stackingy1 = tt.gene.map(stackgenepos) + tt['size']/5)
    
    nrows_genes = int((np.array(yticks3)/intergene_space).max())
    hgenelabels = 300 + nrows_genes*70
    fullgene = tt.query('gbkey == "Gene"')
    fullgene = fullgene.assign(ymax = fullgene.stackingy + max(genesizes.values())/5 ) #size_scalers/10/2
    largenames = fullgene.genestrand.str.lower().str.contains(r'^loc\d+') | fullgene.genestrand.map(lambda x: len(x) >= 8 )
    labels1 = hv.Labels(fullgene[~largenames], kdims = ['mean_pos','ymax'], vdims=['genestrand'])\
                .opts( opts.Labels(text_color='Black', text_font_style='normal',
                                  text_font_size='14px' if (nrows_genes > 6 or len(ttgenes)> 15) else '25px', 
                                  angle=90 if (nrows_genes > 6 or len(ttgenes)> 15) else 0,
                                  text_baseline='middle' if (nrows_genes > 6 or len(ttgenes)> 15) else 'top',
                                  text_align='right' if (nrows_genes > 6 or len(ttgenes)> 15) else 'center',
                                    frame_width = frame_width, height = hgenelabels) )
    labels2 = hv.Labels(fullgene[largenames], kdims = ['mean_pos','ymax'], vdims=['genestrand'])\
                .opts( opts.Labels(text_color='Black', text_font_style='normal', 
                                  text_font_size='14px' if (nrows_genes > 6 or len(ttgenes)> 15) else '18px',
                                  text_baseline='middle' if (nrows_genes > 6 or len(ttgenes)> 15) else 'top',
                                  text_align='right' if (nrows_genes > 6 or len(ttgenes)> 15) else 'center',
                                  angle=90 if (nrows_genes > 6 or len(ttgenes)> 15) else 0, 
                                    frame_width = frame_width, height = hgenelabels) )
    #text_font_size='8px' if (nrows_genes > 6 or len(ttgenes)> 15) else '12px',
    labels = (labels1 * labels2).opts(xaxis = None)
    rect = hv.Rectangles(tt.sort_values('size', ascending = False).rename({'start':'bp'}, axis = 1), kdims=['bp', 'stackingy0', 'end', 'stackingy1'], 
                         vdims=['color', 'gene'])\
             .opts(xformatter=NumeralTickFormatter(format='0,0.[0000]a') ,xticks=5,xrotation = 0,  frame_width = frame_width, height = hgenelabels,
                   tools = ['hover'],  invert_yaxis=True,
                  ylim= (-intergene_space*.4, max(np.array(yticks3).max()+(.9 if (nrows_genes > 6 or len(ttgenes)> 15) else .5)*intergene_space, intergene_space*1.5)), 
                   xlim = ( pos_start, pos_end)) 
    rect = rect.opts(hv.opts.Rectangles(fill_color='color', line_color='color', line_width=2, fill_alpha=0.3))\
               .opts(ylabel = 'genes', yticks =yticks3l, xformatter=NumeralTickFormatter(format='0,0.[0000]a'), frame_width = frame_width)# yticks =yticks3l, 
    return rect*labels

def make_zip_comparison_report(zip1, zip2, nauto = 20, save = True):
    zips2run =[zip1, zip2]
    traits  = read_csv_zip('.*data/pheno/regressedlr_.*txt', zippath=zips2run, sep = '\s+', header = None, names = ['rfid', 'iid', 'value'], 
                           file_func= lambda x: basename(x.replace('regressedlr_', ''))[:-4],
                           zipfile_func=lambda x: basename(x).split('_tsanches_')[-1][:-4] )
    traits_names = traits.groupby('zipfile').file.agg(lambda x: set(x))
    shared_traits = traits_names.agg(lambda x: reduce(lambda a,b: a&b, x)  )
    traits = traits[traits.file.isin(shared_traits)]
    herall = read_csv_zip('.*results/heritability/heritability.tsv', zippath=zips2run,
                           file_func= lambda x: basename(x).split('_chrgwas')[0].replace('regressedlr_', ''),
                           zipfile_func=lambda x: basename(x).split('_tsanches_')[-1][:-4] , sep = '\t', index_col = 0)\
            .reset_index(names = 'trait')
    herall = herall.pivot_table(index = 'trait', columns='zipfile', values=['V(G)/Vp', 'heritability_SE', 'n'])\
                  .rename(lambda x: x.split('tsanches_')[-1].replace('.zip', ''), axis = 1)
    herall_zips = list(set([x[1] for x in herall.columns]))
    herall.columns = ['__'.join(x) for x in herall.columns]
    herall = herall.reset_index()
    p1 = herall.hvplot.scatter(x = f'n__{herall_zips[0]}',  y= f'n__{herall_zips[1]}', frame_height = 500, frame_width = 500,
                                line_width = 2, line_color = 'black', size = 400, 
                                alpha = .7, hover_cols =herall.columns.to_list())*hv.Slope(1, 0).opts( color = 'red')
    p2 = (herall.hvplot.scatter(x = f'V(G)/Vp__{herall_zips[0]}',  y= f'V(G)/Vp__{herall_zips[1]}', frame_height = 500, frame_width = 500,
                                line_width = 2, line_color = 'black', size = 400,  alpha = .7, hover_cols =herall.columns.to_list())*\
        hv.ErrorBars(herall, kdims=[f'V(G)/Vp__{herall_zips[0]}'],
                            vdims=  [f'V(G)/Vp__{herall_zips[1]}', f'heritability_SE__{herall_zips[1]}'])*\
        hv.ErrorBars(herall, kdims=[f'V(G)/Vp__{herall_zips[0]}'],
                            vdims=  [f'V(G)/Vp__{herall_zips[1]}', f'heritability_SE__{herall_zips[0]}'], horizontal = True)\
        *hv.Slope(1, 0).opts( color = 'red')\
    ).opts(xrotation = 45,)
    corrall = read_csv_zip('.*results/heritability/genetic_correlation_melted_table.csv', zippath=zips2run,
                           file_func= lambda x: basename(x),
                           zipfile_func=lambda x: basename(x).split('_tsanches_')[-1][:-4], index_col = 0)\
            .reset_index(names = 'trait')
    corrall['trait'] = corrall.trait1 + '-' +corrall.trait2
    corrall['rG_SE'] =  corrall['rG_SE'].clip(upper = 1)
    corrall = corrall.pivot_table(index = 'trait', columns='zipfile', values=['genetic_correlation', 'rG_SE'])\
                  .rename(lambda x: x.split('tsanches_')[-1].replace('.zip', ''), axis = 1)
    corrall_zips = list(set([x[1] for x in corrall.columns]))
    corrall.columns = ['__'.join(x) for x in corrall.columns]
    corrall = corrall.reset_index()
    corrall['trait1'] = corrall.trait.str.split('regressedlr_').str[-1]
    corrall['trait'] = corrall['trait'].str.replace('regressedlr_', '')
    pgpcorr = corrall.hvplot.scatter(x = f'genetic_correlation__{corrall_zips[0]}',  y= f'genetic_correlation__{corrall_zips[1]}', frame_height = 500, frame_width = 500,\
                                line_width = 2, line_color = 'black', size = 400,  alpha = .7, hover_cols =corrall.columns.to_list(), color = 'trait1').opts(show_legend = False)*\
    hv.ErrorBars(corrall, kdims=[f'genetic_correlation__{corrall_zips[0]}'], 
                 vdims=[f'genetic_correlation__{corrall_zips[1]}', f'rG_SE__{corrall_zips[1]}'])*\
    hv.ErrorBars(corrall, kdims=[f'genetic_correlation__{corrall_zips[0]}'],
                 vdims=[f'genetic_correlation__{corrall_zips[1]}', f'rG_SE__{corrall_zips[0]}'], horizontal = True)\
    *hv.Slope(1, 0).opts( color = 'red')
    n_her_figure = (p1.opts(title = 'sample size')+p2.opts(title = 'heritability')+pgpcorr.opts(title = 'genetic correlation'))
    #########################
    def make_figure_scaterline(tt):
        keys_zipfile2num = {v:k for k,v in enumerate(sorted(traits.zipfile.unique()))}
        tt = tt.assign(zipencoded = tt.zipfile.map(keys_zipfile2num))
        paths = hv.Path(tt.groupby('iid').apply(lambda x: x.sort_values('zipencoded')[['zipencoded', 'value']].values, include_groups=False).to_list())
        shade = datashade(paths, aggregator=ds.count(), precompute=True)
        points = tt.hvplot.scatter(x = 'zipencoded', y = 'value',  line_width = 2, line_color = 'black', size = 300, frame_width = 900, frame_height = 600,
                                alpha = .3, hover_cols =traits.columns.to_list()).opts(xticks = [(j, i) for i,j in keys_zipfile2num.items()],
                                                                                      ylabel = 'value', xlabel = '', xlim = (-.2, len(keys_zipfile2num)-1+.2))
        return shade*points
    tpiv = traits.pivot_table(index = ['iid', 'file'], columns = 'zipfile', values = 'value').reset_index()
    tab = tpiv.groupby('file').agg('count').rename(lambda x: 'n_'+x if x != 'iid' else 'n_total', axis = 1)
    tab['n_shared'] = tpiv.groupby('file').apply(lambda x:(~x.iloc[:, -2:].isna()).all(axis = 1).sum(), include_groups=False)
    tab['R2_shared'] = tpiv.groupby('file').apply(lambda x: x.iloc[:, -2:].corr().iloc[0, 1], include_groups=False)
    tab.reset_index().iloc[:, :]
    def marginal_scatter(df, x, y, **kws):
        h = df.hvplot.scatter(x = x, y = y, frame_width = 250, frame_height = 250,shared_axes = False,
                                line_width = 1, line_color = 'black', alpha = .5, **kws).opts(show_legend= False)
        hx = df.hvplot.hist(x, bins = 100, frame_width = 250, frame_height = 20, normed  = True, alpha = .5, **kws).opts(show_legend= False, xaxis = None, yaxis = None )
        hy = df.hvplot.hist(y, bins = 100, frame_width = 20, frame_height = 250, normed  = True, alpha = .5, **kws).opts( xaxis = None, yaxis = None)
        fig = (h<<hy<<hx).opts(toolbar='left')
        fig.spacing = 0
        fig.margin = (0,0)
        return pn.Column(fig, height =350, width = 350)
    gpt = tpiv.groupby('file').apply(lambda df: marginal_scatter(df, df.columns[-1], df.columns[-2], hover_cols = ['iid']),  include_groups=False)
    pn.extension('tabulator')
    numeric_cols = tab.select_dtypes(include=np.number).columns.to_list()
    d = {x : {'type': 'number', 'func': '>=', 'placeholder': 'Enter minimum'} for x in numeric_cols} | \
        {x : {'type': 'input', 'func': 'like', 'placeholder': 'Similarity'} for x in tab.columns[~tab.columns.isin(numeric_cols)]} 
    fmts = {'keep': {'type': 'tickCross'}}
    tbed = {'keep': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None}}
    download_table = pn.widgets.Tabulator( tab.reset_index(),layout='fit_columns', formatters=fmts, editors=tbed, header_filters=d, selectable='checkbox',show_index=False,
        embed_content=True, row_content=lambda x: gpt[x['file']],expanded=[], pagination='local' ,page_size= 10, height = None, width = 1400) # row_height=600 pagination='local' ,page_size= 15,
    filename, button = download_table.download_menu(text_kwargs={'name': 'Enter filename', 'value': 'download'},button_kwargs={'name': 'Download table'})
    download_table.expanded = list(range(len(tab)))
    download_table.style.background_gradient(cmap="Blues", subset=list(tab.columns[1:-1]))
    download_table.style.background_gradient(cmap="Greens", subset=list(tab.columns[-1:]))
    table_fig = pn.Column(pn.Row(filename, button), download_table, width = 1500)
    #########################
    shared_traits = list(shared_traits)
    qtl_all  = read_csv_zip('.*results/qtls/finalqtl.csv',zippath=zips2run,  
                            file_func= lambda x: basename(x).split('_chrgwas')[0].replace('regressedlr_', ''),
                           zipfile_func=lambda x: basename(x).split('_tsanches_')[-1][:-4] )\
               .query('trait.isin(@shared_traits)')
    qtl_all['pos'] = qtl_all.bp.map(lambda x: str(y:=round(x, -5)/1e6)+ 'M-' +str(y+.1)+'M')
    qtl_all['snp_trait'] = qtl_all.SNP + '_' + qtl_all.trait
    qtl_allp = qtl_all.pivot(index=['Chr','pos', 'trait'], columns='zipfile', values=['p', 'SNP']).sort_index(level = ['trait','Chr' ,'pos'])
    qtl_allp.columns = ['__'.join(x) for x in qtl_allp.columns]
    qtl_allp.loc[:, qtl_allp.columns.str.startswith('p__')] = qtl_allp.loc[:, qtl_allp.columns.str.startswith('p__')].astype(float).round(2)
    qtl_allp = qtl_allp.reset_index().query('trait.isin(@shared_traits)')
    #qtl_fig = hv.Table(qtl_allp.replace(np.nan, '')).opts(width = 700, height = 600)
    qtl_fig = fancy_display(qtl_allp.replace(np.nan, ''), max_width = 1000, max_height=600, layout = 'fit_columns', download_name='qtls_pval_comparison.csv', flexible = True)
    #########################
    traits2look = r'|'.join('.*results/gwas/regressedlr.*' + qtl_all['trait'] + '.*mlma')
    df_gwas = read_csv_zip(traits2look, zippath=zips2run, sep = '\t',
                           usecols = ['Chr', 'SNP', 'bp', 'p'], 
                           dtype= {'Chr': int, 'SNP': str, 'bp':int, 'p':float},
                           query_string='p<0.1',
                           file_func= lambda x: basename(x).split('_chrgwas')[0].replace('regressedlr_', ''),
                           zipfile_func=lambda x: basename(x).split('_tsanches_')[-1][:-4] )\
              .query('file.isin(@shared_traits)')
    df_gwas['p'] = -np.log10(df_gwas['p']) 
    df_gwas['snp_trait'] = df_gwas.SNP + '_' + df_gwas.file
    #### making pvalue scatter comparison
    df_all = df_gwas.pivot(index=['SNP', 'file'], columns='zipfile', values=['p'])
    df_all.columns = ['__'.join(x) for x in df_all.columns]
    qlts_from_mlma =  df_gwas[df_gwas.snp_trait.isin(qtl_all.snp_trait)]\
       .pivot(index=['SNP', 'file'], columns='zipfile', values='p')\
       .sort_index(level =['SNP', 'file']).fillna(1).reset_index()
    qtls_figs = qlts_from_mlma\
       .hvplot.scatter(y= df_all.columns[1].split('__')[1], x =df_all.columns[0].split('__')[1], frame_width = 400, frame_height = 400, size = 200,
                       line_width = 1, line_color = 'black', hover_cols = ['SNP', 'file'], color = 'file', alpha = .5)
    table_pvalf =  fancy_display(qlts_from_mlma.round(2).replace(1, '<1').astype(str), max_width = 600, max_height=600, layout = 'fit_columns', flexible = True)
    scatter_pvalf = df_all.reset_index().hvplot.scatter(x = df_all.columns[0], y =df_all.columns[1], by='file', frame_width = 600,dynspread = 1, frame_height = 600, 
                          xlim = (.9,df_all.iloc[:, 0].max()+.5), ylim=(.9,df_all.iloc[:, 1].max()+.5), datashade = True, 
                          max_px=3,threshold= 1,)*hv.HLine(5.39).opts(color = 'red')*hv.VLine(5.39).opts(color = 'red')*qtls_figs
    qtl_pval_fig = pn.Row(table_pvalf, scatter_pvalf)
    #########################
    def customize_yaxis_ticks_and_labels(plot, element):
        for ax in plot.state.yaxis:
            ax.axis_label = ''                 # Remove y-axis label
            ax.major_tick_in = 0              # Tick pointing inside
            ax.major_tick_out = 0             # No tick outside
            ax.axis_line_width = 0
            ax.axis_line_color = "black"
            ax.major_label_text_font_size = '0pt'
            ax.major_label_standoff = -15      # Move labels inside
            ax.minor_tick_line_color = None
            ax.axis_line_color = "white"
        plot.state.outline_line_color = None  # removes entire frame outline
        if hasattr(plot.state, 'xgrid'): plot.state.xgrid[0].grid_line_color = None
        if hasattr(plot.state, 'ygrid'): plot.state.ygrid[0].grid_line_color = None
        plot.state.border_fill_color = 'white'   # background if needed
        plot.state.min_border_top = 0
        plot.state.min_border_right = 0
    def remove_top_right_borders(plot, element):
        plot.state.outline_line_color = None  # removes entire frame outline
        if hasattr(plot.state, 'xgrid'): plot.state.xgrid[0].grid_line_color = None
        if hasattr(plot.state, 'ygrid'): plot.state.ygrid[0].grid_line_color = None
        plot.state.border_fill_color = 'white'   # background if needed
        plot.state.min_border_top = 0
        plot.state.min_border_right = 0
    # cmap = ['black'] if int(df.name)%2 else ['gray']
    yrange = (.99,max(6, df_gwas['p'].max()+.5))
    cmap_ = {i:j for i, j in zip(df_gwas.zipfile.unique(), ['steelblue', 'firebrick', 'teal'])}
    length_genome = df_gwas.groupby('Chr')['bp'].agg('max').sum()
    fig_set = df_gwas.groupby('Chr').apply(lambda df: df.rename({'bp': f'{df.name}'}, axis = 1)\
                                    .hvplot.scatter(x =  f'{df.name}', y = 'p', by = 'zipfile',  aggregator= ds.by('zipfile', ds.count()),
                                                       datashade = True, pixel_ratio = 1,dynamic = False,
                                                       dynspread=True, max_px=4, threshold = 1, color_key = cmap_
                                                   )*hv.HLines([5.39]).opts(color ='black',line_dash = 'dotted')\
      .opts(xformatter= NumeralTickFormatter(format='0,0.[0000]a'), 
            yformatter= NumeralTickFormatter(format='0,0.[0]') , 
            frame_width =max(20, int(1700*df.bp.max()/length_genome)), 
            frame_height = 600, border=0, xrotation = 90,ylim=yrange,tools=['hover'],hooks = [remove_top_right_borders]), include_groups=False)
    # aggregator = ds.max('-log10p'), 
    fig_set = fig_set.sort_index()
    pcp_fig = (fig_set.iloc[0]\
               +reduce(lambda x,y: x+y, fig_set.iloc[1:]\
                       .map(lambda x: x.opts(ylabel = '', hooks=[customize_yaxis_ticks_and_labels, ])))\
              ).cols(len(fig_set)).opts(shared_axes=False)
    
    num2xymt =  lambda x: str(int(float(x))).replace(str(nauto+1), 'x')\
                                                    .replace(str(nauto+2), 'y')\
                                                    .replace(str(nauto+4), 'mt')
    
    def mapcolor(c): 
        if int(str(c).replace('X',str(nauto+1)).replace('Y', str(nauto+2)).replace('MT', str(nauto+4)))%2 == 0: return 'black'
        return 'gray'
    
    append_position = df_gwas.groupby('Chr').bp.agg('max').sort_index().cumsum().shift(1,fill_value=0)
    df_gwas = df_gwas.groupby('Chr',group_keys=True)\
                     .apply(lambda df: df.assign(color = mapcolor(df.name), \
                                                 x = df.bp + append_position[df.name]), \
                            include_groups=False) \
                     .reset_index(drop = False)
    df_gwas['p value'] =  df_gwas['zipfile'].map(dict(zip([qlts_from_mlma.columns[2],qlts_from_mlma.columns[3]], (-1,1))))*df_gwas.p
    base = reduce(lambda x,y:x*y, df_gwas.groupby('color').apply(lambda df: \
          df.hvplot.scatter(x ='x', y = 'p value', rasterize = True, dynspread=True, max_px=2,threshold= 1, pixel_ratio= 2,
                            cmap = [df.name], frame_width = 1200, frame_height = 600, colorbar = False),include_groups=False))
    trait_dict = dict(zip(df_gwas.file.unique(), \
                          [rgb2hex(int(x[0]*255), int(x[1]*255), int(x[2]*255)) for x in sns.palettes.color_palette('tab10')*5]))
    traitnum_dict = dict(zip(df_gwas.file.unique(), range(100)))
    temp = df_gwas.query('p>4')
    abv4 = temp.hvplot.scatter(x ='x', y = 'p value',color ='file', hover_cols = ['bp', 'file'], cmap = trait_dict)
    qlts_from_mlma[['Chr', 'pos']] = qlts_from_mlma.SNP.str.split(':').to_list()
    qlts_from_mlma['color'] = qlts_from_mlma.file.map(trait_dict)
    qlts_from_mlma['traitlab'] = qlts_from_mlma.file.map(traitnum_dict)
    qlts_from_mlma['x'] = qlts_from_mlma.pos.astype(int) +  qlts_from_mlma.Chr.astype(int).map(append_position)
    qq = qlts_from_mlma.hvplot.scatter(x = 'x', color = 'file', y = qlts_from_mlma.columns[3], marker = 'inverted_triangle', size = 400, line_width = 1, line_color = 'black',
                                 hover_cols = list(qlts_from_mlma.columns), cmap = trait_dict)
    qq = qq*qlts_from_mlma.assign(y = -qlts_from_mlma[qlts_from_mlma.columns[2]])\
            .hvplot.scatter(x = 'x', color = 'file', y = 'y', marker = 'triangle', size = 400, line_width = 1, line_color = 'black',
                                 hover_cols = list(qlts_from_mlma.columns),cmap = trait_dict)
    qq =qq*hv.Labels(qlts_from_mlma, kdims=['x',  qlts_from_mlma.columns[3]], vdims='traitlab').opts(text_color = 'black', text_font_size = '8px')\
          *hv.Labels(qlts_from_mlma.assign(y = -qlts_from_mlma[qlts_from_mlma.columns[2]]), kdims=['x',  'y'], vdims='traitlab').opts(text_color = 'black', text_font_size = '8px')
    pcp = base*abv4*hv.HLines([-5.3, 5.3]).opts(color = 'red', line_dash = 'dashed')*qq*hv.HLine(0)
    names_f = hv.HLine(0).opts(color = 'black', line_width = 4)*\
    hv.Text(df_gwas.x.max()/2, 0.5, '\u2191 '+ qlts_from_mlma.columns[3], fontsize=15)*\
    hv.Text(df_gwas.x.max()/2, -0.5, '\u2193 ' +qlts_from_mlma.columns[2],  fontsize=15)
    yrange = tuple(np.array([-1,1])*(temp.p.max()+0.5))
    pcp = pcp*names_f
    pcp = pcp.opts(xticks=[((dfs.x.agg(['min', 'max'])).sum()//2 , num2xymt(names)) for names, dfs in df_gwas.groupby('Chr')],
                                       ylim =yrange, xlabel='Chromosome', shared_axes=False,
                                   frame_width = 1200, frame_height = 600, title = f'porcupineplot',legend_position='right',show_legend=True)
    tdy = datetime.today()
    lng_prefix = longest_prefix(zips2run[0],zips2run[1], suffix=False, return_pos = True)
    lng_suffix = longest_prefix(zips2run[0],zips2run[1], suffix=True, return_pos = True)
    side_text = f"""
# General Information
<hr>
    
* generated on : \n <b>{tdy.strftime('%Y-%B-%d')}</b> \n
* data from : \n<b>{zips2run[0]}</b> \n<b>{zips2run[1]}</b>
* shared prefix: <b>{zips2run[0][:lng_prefix]}</b>
* shared suffix: <b>{zips2run[0][lng_suffix:]}</b>
* unique names : \n<b>{zips2run[0][lng_prefix:lng_suffix]}</b> \n<b>{zips2run[1][lng_prefix:lng_suffix]}</b>
"""
    template = pn.template.BootstrapTemplate(title=f'GWAS comparison',
                                             favicon='/tscc/projects/ps-palmer/gwas/GWAS-pipeline/gwas/rat.ico',
                                             collapsed_sidebar=True)
    template.sidebar.extend([pn.pane.Alert(side_text, alert_type="primary")])
    template.main.append(pn.Card(table_fig, collapsed = False, title = 'traits', width = 1800 ))
    template.main.append(pn.Card(n_her_figure, collapsed = False, title = 'heritability', width = 1800 ))
    template.main.append(pn.Card(qtl_pval_fig, collapsed = False, title = 'QTLs pval figure', width = 1800 ))
    # template.main.append(pn.Card(qtl_fig,  collapsed = False, title = 'QTL regions',width = 1800 ))
    template.main.append(pn.Card(pcp, collapsed = False, title = 'porcupine',width = 1800 ))
    template.header.append(f'## {zips2run[0][lng_prefix:lng_suffix]} vs {zips2run[1][lng_prefix:lng_suffix]} {tdy.strftime("%Y-%B-%d")}')
    template.save(f'report_comparision{zip1.split("/")[-1]}vs{zip2.split("/")[-1]}_{tdy.strftime("%Y%b%d")}.html', resources=INLINE,  title='GWAS comparison')
    return template

from alphagenome.models import dna_client
from scipy.spatial.distance import hamming
from alphagenome.models.dna_client import OutputType
class AlphaGenome:
    def __init__(self, gwas_pipe, alpha_genome_key_path = "ALPHAGENOME_KEY"):
        agkey = os.environ.get(alpha_genome_key_path)
        self.dna_model = dna_client.create(agkey)
        self.FA  = pysam.FastaFile(gwas_pipe.genomefasta_path)
        self.supported_sequences = dna_client.SUPPORTED_SEQUENCE_LENGTHS
        self.gtf = gwas_pipe.get_gtf() if not hasattr(gwas_pipe, 'gtf') else gwas_pipe.gtf
        self.chrsyn = gwas_pipe.chrsyn
        self.gwas_pipe = gwas_pipe
        self.supported_predictions = ['ATAC','DNASE','CAGE', 'PROCAP','RNA_SEQ','CHIP_HISTONE', 'CHIP_TF',
                                      'SPLICE_SITES','SPLICE_SITE_USAGE','SPLICE_JUNCTIONS','CONTACT_MAPS']
        self.supported_species = ['MUS_MUSCULUS','HOMO_SAPIENS']
        
    def get_gene_region(self,genepos, sequence_length = 'SEQUENCE_LENGTH_100KB', region_align = 'center'):
        if sequence_length not in self.supported_sequences:
            raise ValueError('sequence length has to be in ' + str(list(dna_client.SUPPORTED_SEQUENCE_LENGTHS.keys())) )
        length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[sequence_length]       
        if ':' in genepos: chrom, pos = genepos.split(':')
        elif len(gene:=self.gtf[self.gtf.gene_id.eq(genepos)]): 
            pos = np.mean(gene.agg({'start':'min', 'end':'max'}))
            chrom = gene.Chr.iloc[0]
        else: raise ValueError('genepos is not in gtf file and does not follow the {chr}:{pos} format')
        bounds = {'center':  1/2*np.array([-length, length]), 'end': np.array([-length, 0]), 'start': np.array([0, length])}
        fstart, fend = (float(pos) + bounds[region_align]).round().astype(int)
        chr_conv_ncbi = pd.read_csv(self.chrsyn, sep = '\t', header = None, index_col = 1, names = ['ncbi', 'num'])['ncbi']
        seq = self.FA.fetch(chr_conv_ncbi.loc[str(chrom)], start = fstart-1, end = fend-1).upper()
        return fstart, fend, seq, chrom
        
    def get_genotypes_in_region(self, genepos, min_count_haplotypes = 10,  sequence_length = 'SEQUENCE_LENGTH_100KB', 
                                use_raw_genotypes = False, species = 'HOMO_SAPIENS', snplist = None, region_align = 'center',
                                topSNP = None, subset_snps_from_topsnp_r2 = .8, skip_founder_alphagenome = False):
        fstart, fend, seq, c = self.get_gene_region(genepos=genepos, sequence_length = sequence_length, region_align = region_align )
        length = fend - fstart
        seq_set = []
        if topSNP is not None:
            if not use_raw_genotypes:
                snps = npplink.plink2df(plinkpath=self.gwas_pipe.genotypes_subset, c = c, pos_end=fend, pos_start=fstart, snplist=snplist )
            if use_raw_genotypes or (snps.shape[1]==0):
                snpsrecoded = npplink.plink2df(plinkpath=self.gwas_pipe.all_genotypes, c = c, pos_end=fend, pos_start=fstart,snplist=snplist)
            snplist = sorted(npplink.R2(snps[[topSNP]], snps, return_square=False).query(f'value>={subset_snps_from_topsnp_r2}').bp2.to_list())
            print(f'subsetting snps with R2>={subset_snps_from_topsnp_r2} from {topSNP}, keeping {len(snplist)} out of {snps.shape[1]} in this region')
            
        if not use_raw_genotypes:
            snpsrecoded = npplink.plink2df(plinkpath=self.gwas_pipe.genotypes_subset, c = c, pos_end=fend, pos_start=fstart, recodeACGT=True, snplist=snplist)
            if snpsrecoded.shape[1]==0:print('no variants in genotype subset, using all genotypes')
        if use_raw_genotypes or (snpsrecoded.shape[1]==0):
            snpsrecoded = npplink.plink2df(plinkpath=self.gwas_pipe.all_genotypes, c = c, pos_end=fend, pos_start=fstart, recodeACGT=True, snplist=snplist)
        if snpsrecoded.shape[1]==0:
            print('no variants in either genotype subset, or all genotypes')
        else:
            snpsrecodedHom  = snpsrecoded.copy()
            snpsrecodedHom[snpsrecodedHom.apply(lambda x:x.str[0]) != snpsrecodedHom.apply(lambda x:x.str[1])] = np.nan
            snpsrecodedHom = snpsrecodedHom.loc[:, ~snpsrecodedHom.isna().all()]
            variants_used = ','.join(snpsrecodedHom.columns)
            snpsrecodedHom.columns = snpsrecodedHom.columns.str.split(':').str[1].astype(int) - fstart
            snpsrecodedHom = snpsrecodedHom.loc[:, snpsrecodedHom.columns< fend]
            snpsrecodedHom_haps = snpsrecodedHom.value_counts().to_frame().reset_index().set_index('count')#.applymap(lambda x: x[0])
            seqlis = snpsrecodedHom_haps.apply(AlphaGenome.var2seq,seq=seq ,axis = 1).to_frame(name = 'seq').reset_index()
            seqlis['isref'], seqlis['size'] = seqlis.seq.eq(seq), seqlis.seq.map(len)
            seq_set += [seqlis.rename(lambda x: f'genotypes_{x}').assign(variants = variants_used, founders = False)]
        if len(self.gwas_pipe.foundersbimfambed):
            snpsrecodedf = npplink.plink2df(plinkpath=self.gwas_pipe.foundersbimfambed, c = c, pos_end=fend, pos_start=fstart, recodeACGT=True, snplist=snplist )
            seqlis_founders = snpsrecodedf.rename(lambda x:int(x.split(':')[-1]) - fstart, axis = 1)\
                                .apply(AlphaGenome.var2seq,seq=seq ,axis = 1).to_frame(name = 'seq').reset_index()
            seqlis_founders['isref'], seqlis_founders['size'] = seqlis_founders.seq.eq(seq), seqlis_founders.seq.map(len)
            seq_set += [seqlis_founders.set_index('index').assign(variants = ','.join(snpsrecodedf.columns), founders = True)]
        allseqs = pd.concat(seq_set)
        if not len(allseqs): print('no variants in genotypes or founders'); return pd.DataFrame()
        allseqs['count'] = allseqs['count'].fillna(np.inf if not skip_founder_alphagenome else (min_count_haplotypes-1))
        for a,b in tqdm(itertools.combinations(allseqs.index, 2)):
            allseqs.loc[a, b] = int(hamming(list(allseqs.loc[a, 'seq']),list(allseqs.loc[b, 'seq']) )*length)
        allseqs.loc[allseqs['count'].ge(min_count_haplotypes), 'alphagout'] = \
            allseqs.loc[allseqs['count'].ge(min_count_haplotypes), 'seq'].progress_apply(self.predict_sequence, species = species)
        allseqs = allseqs.assign(fstart = fstart, fend = fend, length = length,c = c)
        return allseqs
    
    @staticmethod
    def alphagenome2pd(pos_start, len_seq, alpha_output, modality, numeric_pos = True):
        modality = modality.lower()
        vals, meta = getattr(alpha_output, modality).values, getattr(alpha_output, modality).metadata
        binsize = int(len_seq/vals.shape[0])
        pos = (np.linspace(pos_start,  pos_start+len_seq,vals.shape[0]+1) + binsize/2).astype(int)[:-1]
        pos_cols = str(c)+':'+pd.Series(pos).astype(str) if not numeric_pos else pos
        return pd.concat([meta.assign(binsize = binsize), pd.DataFrame(data = vals.T, columns =pos_cols)], axis = 1)\
                 .set_index(meta.columns.to_list()+ ['binsize'], drop = True)
        
    @staticmethod
    def var2seq(s, seq):
        s = s[s.index<len(seq)].fillna('NN').str.upper()
        seqarray = np.array(list(seq), dtype = 'U1')
        seqarray[s.index.astype(int)] = s.str.upper().str[0].values
        mask = ~np.isin(seqarray, np.array(['A', 'C', 'T', 'G', 'N'],  dtype = 'U1'))
        if mask.any(): 
            print('found a character not in ACGTN, replacing it with N, please check it before moving forward')
            print(seqarray[mask])
            svals = s.str.upper().str[0]
            display(s[~svals.isin(list('ACTGN'))])
        seqarray[mask] = 'N'
        return ''.join(seqarray).upper()
        
    def predict_sequence(self, seq, species = 'HOMO_SAPIENS'):
        organism=dna_client.Organism.MUS_MUSCULUS if species == 'MUS_MUSCULUS' else dna_client.Organism.HOMO_SAPIENS
        sleep(5)
        return self.dna_model.predict_sequence( sequence=seq.upper(), organism = organism, 
            requested_outputs=[ OutputType.ATAC,OutputType.DNASE,OutputType.CAGE,
            OutputType.PROCAP,OutputType.RNA_SEQ,OutputType.CHIP_HISTONE,
            OutputType.CHIP_TF,OutputType.SPLICE_SITES,OutputType.SPLICE_SITE_USAGE,
            OutputType.SPLICE_JUNCTIONS,OutputType.CONTACT_MAPS ],ontology_terms=[] )
        
    def get_genetrack_seqlis(seqlis, seqlis_index=None, frame_width = 1000):
        if seqlis_index is None: seqlis_index = seqlis.index[0]
        return self.gwas_pipe.make_genetrack_figure_(c = seqlis.loc[seqlis_index, 'c'], 
                                                        pos_start=seqlis.loc[seqlis_index, 'fstart'], 
                                                        pos_end=seqlis.loc[seqlis_index, 'fend'], frame_width=frame_width)
    
    @staticmethod
    def twoway_quantile_filter(df, row_quantile=.8, col_quantile = .8, absolute = True):
        dfcp = df.abs() if absolute else df
        row_sum, row_max, sum_pos, max_pos = dfcp.sum(axis = 1), dfcp.max(axis = 1), dfcp.sum(), dfcp.max()
        return df.loc[row_sum.gt(row_sum.quantile(row_quantile)) | row_max.gt(row_max.quantile(row_quantile)),
                     sum_pos.gt(sum_pos.quantile(col_quantile))  | max_pos.gt(max_pos.quantile(col_quantile))]

    @staticmethod
    def remove_odd_celltypes(df):
        mask = (~df.index.get_level_values("biosample_name").str.contains(r"[A-Z\d]") |
                 df.index.get_level_values("biosample_name").str.contains(r"cell|CD\d")) & \
                ~df.index.get_level_values("biosample_name").str.contains(r"endothelial cell of umbilical")     
        return df[mask]

    def simplify_ag_df(self, seqlis, seqlis_index = 'genotypes_0', metric = 'ATAC', quantile_filter_celltype = .8, 
                       quantile_filter_pos = .5, absolute = True, return_melted = True, remove_odd_celltypes= True):
        mydf = self.alphagenome2pd(seqlis.loc[seqlis_index,'fstart'],
                                   seqlis.loc[seqlis_index,'length'], 
                                   seqlis.loc[seqlis_index,'alphagout'], metric)
        if remove_odd_celltypes:
            mydf = self.remove_odd_celltypes(mydf)
        if (quantile_filter_celltype<1) or (quantile_filter_pos<1):
            mydf = self.twoway_quantile_filter(mydf, quantile_filter_celltype, quantile_filter_pos, absolute = absolute)
        if return_melted:
            return  mydf.reset_index().melt(id_vars=list(set(mydf.index.names) & \
                                                         set(['biosample_name', 'biosample_life_stage', 'biosample_type',
                                                              'biosample_life_stage','transcription_factor', 'histone_mark'])), 
                                     value_vars=mydf.columns, var_name='bp', value_name=metric)\
                                    .astype({'bp': float, metric : float})
        return mydf
    def get_genetrack_seqlis(self, seqlis, seqlis_index=None, frame_width = 1000):
        if seqlis_index is None: seqlis_index = seqlis.index[0]
        return self.gwas_pipe.make_genetrack_figure_(c = seqlis.loc[seqlis_index, 'c'], 
                                                        pos_start=seqlis.loc[seqlis_index, 'fstart'], 
                                                        pos_end=seqlis.loc[seqlis_index, 'fend'], frame_width=frame_width)

    def snp_lines(self, seqlis, seqlis_index=None, frame_width = 1000):
        if seqlis_index is None: seqlis_index = seqlis.index
        snp_pos = [hv.VLines( [int(x.split(':')[-1]) for x in snps_lis.split(',')]  )\
                     .opts(color = 'red' if isfounder else 'blue', alpha = .5,line_dash = 'dotted',
                           xlabel='bp', line_width = .2, frame_width= frame_width)\
                  for isfounder, snps_lis in seqlis.loc[seqlis_index].value_counts(['founders', 'variants']).index]
        return reduce(lambda x,y: x*y, snp_pos).opts( xformatter=NumeralTickFormatter(format='0,0.[0000]a'))

    def melted_metric_all_haplotypes(self, seqlis, metric = 'ATAC', indices = None, agg = 'max', 
                                     quantile_filter_celltype = 1, quantile_filter_pos = 1, remove_odd_celltypes=True ):
        if indices is None: indices = seqlis.dropna(subset = 'alphagout').index
        return pd.concat([(y:=self.simplify_ag_df(seqlis, i, metric = metric, return_melted=True, remove_odd_celltypes = remove_odd_celltypes,
                                                  quantile_filter_celltype = quantile_filter_celltype, 
                                                  quantile_filter_pos = quantile_filter_pos))\
                           .groupby(list(y.columns[:-1])).agg(agg).rename(lambda x: f'{x}_{i}', axis = 1)
                  for i in tqdm(indices)], axis = 1)

    def delta2genotypes(self, seqlis, idx0, idx1, metric='CHIP_TF', quantile_filter_celltype=1, quantile_filter_pos=1, return_melted = True,  remove_odd_celltypes=True ):
        deltadf = self.simplify_ag_df(seqlis=seqlis, metric=metric, return_melted=False, seqlis_index = idx0, 
                                      quantile_filter_celltype=quantile_filter_celltype, quantile_filter_pos=quantile_filter_pos, 
                                      remove_odd_celltypes = remove_odd_celltypes) - \
                  self.simplify_ag_df(seqlis=seqlis, metric=metric, return_melted=False, seqlis_index = idx1, 
                                      quantile_filter_celltype=quantile_filter_celltype, quantile_filter_pos=quantile_filter_pos,
                                      remove_odd_celltypes = remove_odd_celltypes)
        if not return_melted: return deltadf.reset_index()
        return deltadf.reset_index().melt(id_vars=list(set(deltadf.index.names) & 
                                                       set(['biosample_name', 'biosample_life_stage', 'biosample_type', 'histone_mark',
                                                                     'biosample_life_stage','transcription_factor'])), 
                                             value_vars=deltadf.columns, var_name='bp', value_name=f'delta_{metric}')\
                                            .astype({'bp': float, f'delta_{metric}' : float})

    def make_fig_delta(self, seqlis, idx0 = 'genotypes_0', idx1 = 'genotypes_1'):
        totrnadelta = self.delta2genotypes(seqlis, idx0, idx1, metric = 'RNA_SEQ', )
        tottfdelta = self.delta2genotypes(seqlis, idx0, idx1, metric = 'CHIP_TF', )
        tothistdelta = self.delta2genotypes(seqlis, idx0, idx1, metric = 'CHIP_HISTONE', )
        totsplicedelta = self.delta2genotypes(seqlis, idx0, idx1, metric = 'SPLICE_SITE_USAGE', )
    
        snplines = self.snp_lines(seqlis)
        genetrack = self.get_genetrack_seqlis(seqlis)
        return (
        (totrnadelta[totrnadelta.delta_RNA_SEQ.abs()>totrnadelta.delta_RNA_SEQ.abs().quantile(.5)]\
                    .hvplot.line(x = 'bp', y = 'delta_RNA_SEQ', by = 'biosample_name', rasterize=True, downsample=True, 
                                 frame_width =1000, frame_height =100, legend = False, xaxis = 'top', shared_axes=False)*snplines)\
            .opts( xaxis = 'top', xformatter=NumeralTickFormatter(format='0,0.[0000]a'))+\
        totsplicedelta.hvplot.line(x = 'bp', y = 'delta_SPLICE_SITE_USAGE', by = 'biosample_name', rasterize=True, downsample=True, 
                                 frame_width =1000, frame_height =100, legend = False, xaxis = None, shared_axes=False)*snplines+\
        tothistdelta.hvplot.line(x = 'bp', y = 'delta_CHIP_HISTONE', by = [ 'histone_mark'],
                                rasterize=True, downsample=True, frame_width =1000, legend = False, frame_height =100,xaxis = None,)*snplines+\
        tottfdelta[tottfdelta.transcription_factor.str.contains('POL') &  ~tottfdelta.biosample_name.str.contains("liver")]\
                          .rename({'delta_CHIP_TF': 'deltaPOL2|CTCF'}, axis = 1)\
                          .hvplot.line(x = 'bp', y = 'deltaPOL2|CTCF', by = ['biosample_name', 'transcription_factor'],
                               rasterize=True, downsample=True, frame_width =1000, legend = False,frame_height =100, xaxis = None,  )*snplines\
                             .opts( axiswise=True)+\
        tottfdelta[~tottfdelta.transcription_factor.str.contains('CTCF|POL')&  ~tottfdelta.biosample_name.str.contains("liver")]\
                           .hvplot.line(x = 'bp', y = 'delta_CHIP_TF', by = ['biosample_name', 'transcription_factor'],
                               rasterize=True, downsample=True, frame_width =1000, legend = False, frame_height =100,xaxis = None,)*snplines+\
        genetrack
        ).cols(1)
    
        
    def make_fig(self, allseqs, allseqs_idxa,allseqs_idxb, by = 'biosample_name' ):
        fstart, fend, length, c = allseqs.fstart.iloc[0], allseqs.fend.iloc[0], allseqs.length.iloc[0], allseqs.c.iloc[0]
        genetrack =  self.gwas_pipe.make_genetrack_figure_(c = c, pos_start=fstart, pos_end=fend, frame_width=1000)
        snp_pos = [hv.VLines( [int(x.split(':')[-1]) for x in snps_lis.split(',')]  ).opts(color = 'red' if isfounder else 'blue', 
                                                                                         alpha = .5,line_dash = 'dotted', xlabel='bp', line_width = .2)\
                  for isfounder, snps_lis in allseqs.value_counts(['founders', 'variants']).index]
        snp_pos = reduce(lambda x,y: x*y, snp_pos)
        
        delta_seq = (AlphaGenome.alphagenome2pd(fstart, length, allseqs.loc[allseqs_idxa]['alphagout'], 'RNA_SEQ')-\
                     AlphaGenome.alphagenome2pd(fstart, length, allseqs.loc[allseqs_idxb]['alphagout'], 'RNA_SEQ'))
        sum_celltype, max_celltype = delta_seq.abs().sum(axis = 1), delta_seq.abs().max(axis = 1)
        delta_seq = delta_seq[sum_celltype.gt(sum_celltype.quantile(.95)) | max_celltype.gt(max_celltype.quantile(.95)) ]
        delta_seqm = delta_seq.reset_index().melt(id_vars=['biosample_name', 'biosample_life_stage'], 
                                     value_vars=delta_seq.columns, var_name='bp', value_name='delta_exp')\
                                    .astype({'bp': float, 'delta_exp' : float})
        quants = delta_seqm.delta_exp.quantile([.1,.9])
        criteria_1 = delta_seqm.delta_exp.lt(quants.iloc[0])|delta_seqm.delta_exp.gt(quants.iloc[1])
        criteria =  ~delta_seqm.biosample_life_stage.str.contains('unknown')
        biosample_subset = delta_seqm.loc[criteria&criteria_1].biosample_name.value_counts()#.hvplot.hist(bins = 50)
        tissues2use = set(biosample_subset[biosample_subset>biosample_subset.quantile(.95)].index)
        tissue_extreme =  delta_seqm.loc[criteria&criteria_1].groupby('biosample_name')\
                                    .delta_exp.agg(lambda x: x.abs().max()).sort_values(ascending = False)#.max()#.quantile(.9)
        tissue_extreme2use  = set(tissue_extreme[tissue_extreme>tissue_extreme.quantile(.95)].index)
        criteria_2 = delta_seqm.biosample_name.isin(tissue_extreme2use|tissues2use)
        #groupby= 'biosample_life_stage','biosample_life_stage'  groupby =  'biosample_life_stage',
        scatter_ex = delta_seqm.loc[:].hvplot.scatter(x= 'bp', y = 'delta_exp', by = by,  datashade = True, dynspread = True, threshold = .9,
                                                             hover_cols=['biosample_life_stage'] , frame_width =1000).opts(xaxis = None)
        line_ex = delta_seqm.loc[:].hvplot.line(x= 'bp', y = 'delta_exp', by = by,
                                                 rasterize=True, downsample=True, frame_width =1000, legend = None).opts(xaxis = None)
        ###############################
        delta_tf = (AlphaGenome.alphagenome2pd(fstart, length, allseqs.loc[allseqs_idxa]['alphagout'], 'CHIP_TF')-\
                    AlphaGenome.alphagenome2pd(fstart, length, allseqs.loc[allseqs_idxb]['alphagout'], 'CHIP_TF'))
        delta_tf = delta_tf[~(delta_tf.index.get_level_values('transcription_factor') == 'CTCF')]
        delta_tf = delta_tf[~(delta_tf.index.get_level_values('transcription_factor').to_series().str.contains('^POLR')).values]
        sum_celltype, max_celltype = delta_tf.abs().sum(axis = 1), delta_tf.abs().max(axis = 1)
        delta_tf = delta_tf[sum_celltype.gt(sum_celltype.quantile(.5)) | max_celltype.gt(max_celltype.quantile(.5)) ]
        delta_tfm = delta_tf.reset_index().melt(id_vars=['biosample_name', 'biosample_life_stage', 'transcription_factor'], 
                                     value_vars=delta_tf.columns, var_name='bp', value_name='TF_binding')\
                                    .astype({'bp': float, 'TF_binding' : float})
        line_tf = delta_tfm.loc[:].hvplot.line(x= 'bp', y = 'TF_binding', by = 'transcription_factor' , downsample=True,
                                               rasterize=True, frame_width =1000, legend = None).opts(xaxis = None) 
        ###############################
        return (line_ex*snp_pos+line_tf*snp_pos+genetrack).cols(1)

def longest_prefix(a, b, suffix = False, return_pos = False):
    if suffix: a,b = a[::-1], b[::-1]
    for c in range(min(len(a), len(b))):
        if a[c]!=b[c]: break
    if return_pos: return (-1 if suffix else 1)*c
    return a[:c][::-1 if suffix else 1]

def parse_gcta_time(file ):
    ttime = 0
    with open(file) as f:
        time_str  = re.findall('computational time:([^\n]+)', f.read().lower())
        if not len(time_str):  return pd.Timedelta(0, unit='s')
    time_str = time_str[0].strip()
    min_match = re.search(r'(\d+(?:\.\d+)?)\s*min', time_str)
    if min_match: ttime += 60*float(min_match.group(1))
    sec_match = re.search(r'(\d+(?:\.\d+)?)\s*sec', time_str)
    if sec_match: ttime += float(sec_match.group(1))
    h_match = re.search(r'(\d+(?:\.\d+)?)\s*h', time_str)
    if h_match: ttime += 3600*float(h_match.group(1))
    return pd.Timedelta(ttime, unit='s')

def cost_calculator(ncores, mem, time_in_s, gpu=None, SU2dollar=0.000587, return_monetary = True):
    SU2dollar = defaultdict(lambda: SU2dollar,{'condo':0.00019705151837500393, 'hotel': 0.000587 })[SU2dollar]
    gpu_c = defaultdict(int,{'A100': 60,'RTX3090': 20, 'A40': 10, 'RTXA6000': 30})
    SU = time_in_s/60*(ncores+.2*mem+gpu_c[gpu])*.97
    if return_monetary: return f'${round(SU*SU2dollar, 2)}'
    return f'{int(SU):,}'+ 'SU'