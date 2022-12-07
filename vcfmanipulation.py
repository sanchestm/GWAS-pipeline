import pandas as pd
import subprocess
from glob import glob
from datetime import datetime
import numpy as np
import re
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 
import gzip
import dask.dataframe as dd
from tqdm import tqdm
import gc
from collections import Counter
from sklearn.decomposition import PCA
from umap import UMAP
from pathlib import Path
import os
import inspect
from time import sleep
import sys
import itertools
from IPython.utils import io


class vcftools:
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
                               names = vcftools.get_vcf_header(filename),blocksize=None,  dtype=str, ).repartition(npartitions = 100000)
        # usecols=['#CHROM', 'POS']
        return pd.read_csv(filename,  compression='gzip', comment='#',  delim_whitespace=True,
                           header=None, names = vcftools.get_vcf_header(filename),  dtype=str )

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
        if not metadata:
            header = """##fileformat=VCFv4.1
            ##fileDate=20090805
            ##source=myImputationProgramV3.1
            ##reference=file:///seq/references/
            """
        if metadata[-4:] == '.vcf': get_vcf_metadata(metadata)
        else: header = metadata
        with open(filename, 'w') as vcf: vcf.write(header)
        df.to_csv(filename, sep="\t", mode='a', index=False)

    def get_vcf_header(vcf_path):
        with gzip.open(vcf_path, "rt") as ifile:
            for num, line in enumerate(ifile):
                if line.startswith("#CHROM"): return line.strip().split('\t')
                if num > 10000: return '-1'
        return '-1'


    
