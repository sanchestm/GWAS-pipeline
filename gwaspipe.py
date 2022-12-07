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
from vcfmanipulation import vcftools


def bash(call, verbose = 0, return_stdout = True, print_call = True):
    if print_call: print(call+'\n')
    out = subprocess.run(call.split(' '), capture_output = True) 
    if verbose and not return_stdout: print(out.stdout)
    
    if out.stderr: print(out.stderr)
    if return_stdout: return out.stdout.decode('ascii').strip().split('\n')
    return out

def qsub(call: str, queue = 'condo' ,walltime = 3, ppn = 1, out = 'log/', err = 'logerr/' , project_dir = ''):
    err_path = f'{project_dir}{err}$PBS_JOBNAME.err'
    out_path = f'{project_dir}{out}$PBS_JOBNAME.out'
    call_path = f'{project_dir}{call}'
    return bash(f'qsub -q {queue} -l nodes=1:ppn={ppn} -j oe -o {out_path} -e {err_path} -l walltime={walltime}:00:00 {call_path}')
  
  
 class gwas_pipe:
    def __init__(self, 
                 path: str = f'{Path().absolute()}/', 
                 use_tscc_modules: list = [],
                 all_genotypes: str = '/projects/ps-palmer/apurva/riptide/genotypes/round9_1',
                 gtca_path: str = '',
                 data: pd.DataFrame() = pd.DataFrame(),
                 traits: list = [],
                 threads: int = 12):
        
        if use_tscc_modules: bash(f'module load {" ".join(use_tscc_modules)}')
        self.gtca = 'gcta64' if not gtca_path else gtca_path
        self.path = path
        self.all_genotypes = all_genotypes
        df = data
        df.columns = df.columns.str.lower()
        self.df = df
        self.traits = [x.lower() for x in traits]
        
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.genotypes_subset = f'{self.path}genotypes/genotypes'
        self.genotypes_subset_vcf = f'{self.path}genotypes/genotypes_subset_vcf.vcf.gz'
        
        self.autoGRM = f'{self.path}grm/AllchrGRM'
        self.xGRM = f'{path}grm/xchrGRM'
        self.log = pd.DataFrame( columns = ['function', 'call', 'out'])
        self.thrflag = f'--thread-num {threads}'
        self.print_call = True
        self.chrList = ['x' if x == 21 else x for x in range(1,22)]
        
    def plink2Df(self, call, temp_out_filename = 'temp/temp', dtype = '.ld'):
        full_call = re.sub(r' +', ' ', call + f'--out {self.path}{temp_out_filename}')
        bash(full_call, print_call = False)
        try: out = pd.read_csv(f'{self.path}{temp_out_filename}{dtype}', sep = '\s+')
        except:
            print(f"file not found")
            out = pd.DataFrame()
        return out 
    
    def plink(self, return_file = 'ld', outfile = '',  **kwargs):
        call = 'plink ' + ' '.join([f'--{k.replace("_", "-")} {v}'
                                    for k,v in kwargs.items()])
        if not outfile and return_file:
            return self.Plink2Df(call ,dtype = f'.{return_file}' ) 
        else:
            return bash(call + f' --out {outfile}', print_call=False)
        
    def bashLog(self, call, func, print_call = True):
        self.append2log(func, call , bash(re.sub(r' +', ' ', call), print_call = print_call))
        
        
    def append2log(self, func, call, out):
        self.log.loc[len(self.log)] = [func, call, out]
        with open(f'{self.path}/log/{func}.log', 'w') as f:
            f.write('\n'.join(out))
            
    def make_dir_structure(self,folders: list = ['data', 'genotypes', 'grm', 'log', 'logerr' , 
                                            'results', 'temp', 'data/pheno', 'results/heritability', 
                                             'results/gwas',  'results/loco', 'results/qtls', 'temp/r2']):
        for folder in folders:
            os.makedirs(f'{self.path}{folder}', exist_ok = True)
            
            
    def subsetSamplesFromAllGenotypes(self,samplelist: list = [], 
                                      use_rfid_from_df = True, sourceFormat = 'vcf', 
                                      geno: float = .1, maf: float = .005, hwe: float = 1e-10 ):
        
        funcName = inspect.getframeinfo(inspect.currentframe()).function
        
        os.makedirs(f'{self.path}genotypes', exist_ok = True)
        
        df = self.df[['rfid', 'rfid']] if use_rfid_from_df else pd.DataFrame([samplelist,samplelist]).T.astype(str)
        df.to_csv(f'{self.path}genotypes/sample_rfids.txt', index = False, header = None, sep = ' ')
        self.sample_names = samplelist
        
        fmt_call = {'vcf': 'vcf', 'plink': 'bfile'}[sourceFormat]        
        
        extra_params = f'--geno {geno} --maf {maf} --hwe {hwe} --double-id --set-missing-var-ids @:#'
        
        self.bashLog(f'plink --{fmt_call} {self.all_genotypes} --keep {self.sample_path} {extra_params} {self.thrflag} --make-bed --out {self.genotypes_subset}',
                    funcName)
        
    def SubsampleMissMafHweFilter(self, sexfmt: str = 'M|F',  sexColumn: str = 'sex',  
                                  geno: float = .1, maf: float = .005, hwe: float = 1e-10,
                                  sourceFormat = 'vcf', remove_dup: bool = True, print_call: bool = False):
        funcName = inspect.getframeinfo(inspect.currentframe()).function
        
               
        fmt_call = {'vcf': 'vcf', 'plink': 'bfile'}[sourceFormat]  
        rmv = f' --double-id --set-missing-var-ids @:# ' if remove_dup else ' '
        sub = self.genotypes_subset
        
        mfList = sorted(sexfmt.split('|'))
        self.samples = {}
        for num, sx in enumerate(tqdm(mfList)): 
            
            dff = self.df[(self.df[sexColumn] == sx) & 
                          (self.df.astype(str).rfid.isin(vcftools.get_vcf_header(self.all_genotypes)))]
            dff[['rfid', 'rfid']].to_csv(f'{self.path}genotypes/sample_rfids_{sx}.txt', index = False, header = None, sep = ' ')
            self.samples[sx] = f'{self.path}genotypes/sample_rfids_{sx}.txt'
            
            filtering_flags = f' --geno {geno} --maf {maf} --hwe {hwe}'
            filtering_flags_justx = f''
            extra_flags = f'--not-chr X'  if num == 1 else ''
            
            self.bashLog(f'plink --{fmt_call} {self.all_genotypes} --keep {self.samples[sx]} {filtering_flags} {rmv} {extra_flags} {self.thrflag} --make-bed --out {sub}_{sx}',
                        f'{funcName}_subseting_{sx}')
            
            if num == 1:
                self.bashLog(f'plink --bfile {sub}_{sx} --chr x {self.thrflag} --make-bed --out {sub}_{sx}_xchr',
                        f'{funcName}_maleXsubset{sx}') #--out {sub}_{sx}_xchr
                male_1_x_filenames = [aa for aa in [f'{sub}_{sx}', f'{sub}_{sx}_xchr'] if len(glob(aa+'.*')) >= 5]
                male_gen_filenames = f'{self.path}/genotypes/temp_male_filenames'
                pd.DataFrame(male_1_x_filenames).to_csv(male_gen_filenames, index = False, header = None)
            else: female_hwe = f'{sub}_{sx}'
                
        print('merging sexes')        
        self.bashLog(f'plink --bfile {female_hwe} --merge-list {male_gen_filenames} {self.thrflag} {filtering_flags} --make-bed --out {sub}_hwe',
                        f'{funcName}_mergeSexes')
        
        self.genotypes_subset = f'{sub}_hwe'
        
    def generateGRM(self, autosome_list: list = list(range(1,21)), print_call: bool = False,
                    extra_chrs: list = ['xchr']):
        
        funcName = inspect.getframeinfo(inspect.currentframe()).function
                
        self.bashLog(f'{self.gtca} {self.thrflag} --bfile {self.genotypes_subset} --autosome-num 21 --autosome --make-grm-bin --out {self.autoGRM}',
            funcName, print_call = print_call)
        
        if 'xchr' in extra_chrs:
            self.bashLog(f'{self.gtca} {self.thrflag} --bfile {self.genotypes_subset} --make-grm-xchr --out {self.xGRM}',
                        f'{funcName}_chrX', print_call = print_call)
        
        for c in tqdm(autosome_list):
            self.bashLog(f'{self.gtca} {self.thrflag} --bfile {self.genotypes_subset} --chr {c} --make-grm-bin --out {self.path}grm/{c}chrGRM',
                        f'{funcName}_chr{c}',  print_call = print_call)

    def snpHeritability(self, print_call: bool = False):
        h2table = pd.DataFrame()
        for trait in tqdm(self.traits):
            trait_file = f'{self.path}data/pheno/{trait}.txt'
            out_file = f'{self.path}results/heritability/{trait}' 
            df.dropna(subset = ['rfid'])[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file, index = False, sep = ' ', header = None)
            
            self.bashLog(f'{self.gtca} --reml {self.thrflag} --pheno {trait_file} --grm {self.autoGRM} --out {out_file}',
                        f'snpHeritability_{trait}', print_call = print_call)
            
            a = pd.read_csv(f'{out_file}.hsq', skipfooter=6, sep = '\t',engine='python')
            b = pd.read_csv(f'{out_file}.hsq', skiprows=6, sep = '\t', header = None, index_col = 0).T.rename({1: trait})
            newrow = pd.concat(
                [a[['Source','Variance']].T[1:].rename({i:j for i,j in enumerate(a.Source)}, axis = 1).rename({'Variance': trait}),
                b],axis =1 )
            h2table= pd.concat([h2table,newrow])
            
        h2table.to_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t')
        return h2table
        
    def gwasPerChr(self, nchr: int = 21, print_call: bool = False):
        for trait, chrom in tqdm(list(itertools.product(gwas.traits, range(1,nchr+1)))):
            if chrom == 21: chrom = 'x'
            self.bashLog(f'{self.gtca} {self.thrflag} --pheno {self.path}data/pheno/{trait}.txt --bfile {self.genotypes_subset} \
                                       --grm {self.path}grm/AllchrGRM   \
                                       --chr {chrom} \
                                       --mlma-subtract-grm {self.path}grm/{chrom}chrGRM  \
                                       --mlma --out {self.path}results/gwas/gwas_{chrom}_{trait}',
                        f'GWAS_{chrom}_{trait}', print_call = print_call)
    
    def GWAS(self, subtract_grm: bool = False, loco: bool = True , print_call: bool = False):
        grm_flag = f'--grm {self.path}grm/AllchrGRM --mlma-subtract-grm {self.path}grm/AllchrGRM' if subtract_grm else ''
        grm_name = 'sub_grm' if subtract_grm else 'with_grm'
        loco_flag = '-loco' if loco else ''
        for trait in tqdm(self.traits):
            self.bashLog(f'{self.gtca} {self.thrflag} --pheno {self.path}data/pheno/{trait}.txt --bfile {self.genotypes_subset}\
                                       {grm_flag}  \
                                       --mlma{loco_flag} --out {self.path}results/loco/{trait}',
                        f'GWAS_{grm_name}_{loco_flag[1:]}_{trait}',  print_call = print_call)
        
    def callQTLs(self, threshold: float = 5.3, window: int = 1e6, subterm: int = 2,
                 ldwin = 1e6, ldkb = 11000, ldr2 = .4, qtl_dist = 2*1e6, nchr: int = 21, NonStrictSearchDir = ''):
        thresh = 10**(-threshold)
        chr_list = ['x' if x == 21 else x for x in range(1,nchr+1)]
        
        if not NonStrictSearchDir:
            topSNPs = pd.DataFrame()
            for t, c in tqdm(list(itertools.product(self.traits, chr_list))):
                filename = f'{self.path}results/gwas/gwas_{c}_{t}.mlma'
                try:
                    topSNPs = pd.concat([topSNPs, pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=t)])
                except:
                    print(f"didn't open {filename}, does it exist?")
        else:
            topSNPs = pd.concat([pd.read_csv(filename, sep = '\t').query(f'p < {thresh}').assign(trait=re.findall('/([^/]*).mlma', filename)[0]) for
                                 filename in tqdm(glob(f"{NonStrictSearchDir}/*.mlma"))])

        out = pd.DataFrame()

        for (t, c), df in tqdm(topSNPs.groupby(['trait','Chr'])):
            df = df.set_index('bp')
            df.p = -np.log10(df.p)

            while df.query('p > @threshold').shape[0]:
                idx = df.p.idxmax()
                maxp = df.loc[idx]
                correlated_snps = df.loc[idx- window//2: idx + window//2].query('p > @maxp.p - @subterm')
                qtl = True if correlated_snps.shape[0] > 2 else False

                out = pd.concat([out,
                                 maxp.to_frame().T.assign(QTL= qtl)],
                                 axis = 0)

                ldfilename = f'{self.path}temp/r2/temp_qtl_n_{t}'
                self.bashLog(f'plink --bfile {self.genotypes_subset} --chr {c}  --ld-snp {maxp.SNP} \
                                     --ld-window {ldwin} {self.thrflag} -- \
                                     --nonfounders --r2  \
                                     --ld-window-r2 {ldr2} --out {ldfilename}',
                             f'qlt_{t}', False )#--ld_window_kb {ldkb}

                try: 
                    ldSNPS = pd.read_csv(f'{ldfilename}.ld', sep = r'\s+').SNP_B.to_list() + [maxp.SNP]
                    df = df.query('~(@idx - @qtl_dist//2 < index < @idx + @qtl_dist//2) and (SNP not in @ldSNPS)')
                except:
                    ldSNPS = [maxp.SNP]
                    df = df.query('(SNP not in @ldSNPS)') ##### should this be different than the one above?
                #if sum(cnt.values()) % 10 == 0: print(cnt)
                
        def get_kb_range(row):
            #### add things for phewas here
            pass
            

        out =  out.reset_index().rename({'index': 'bp'}, axis = 1).sort_values('trait')
        out.to_csv(f'{self.path}/results/qtls/allQTLS.csv', index = False)
        return out.set_index('SNP')            

    def annSnpEff(self):
        pass 
                

    def print_watermark():
        pass
            
            
            
            
    
