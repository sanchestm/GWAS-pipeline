
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
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from umap import UMAP
from pathlib import Path
import os
import inspect
from time import sleep
import sys
import itertools
from IPython.utils import io
import psycopg2
import warnings
import requests
from io import StringIO
import requests
#import sleep
warnings.filterwarnings('ignore')
#conda create --name gpipe -c conda-forge openjdk=17 ipykernel pandas seaborn scikit-learn umap-learn psycopg2 dask
#conda activate gpipe
#conda install -c bioconda gcta plink snpeff
#wget https://snpeff.blob.core.windows.net/versions/snpEff_latest_core.zip


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
    if print_call: print(call+'\n')
    out = subprocess.run(call.split(' '), capture_output = True) 
    if verbose and not return_stdout: print(out.stdout)
    
    if out.stderr: print(out.stderr.decode('ascii'))
    if return_stdout: return out.stdout.decode('ascii').strip().split('\n')
    return out

def qsub(call: str, queue = 'condo', walltime = 8, ppn = 12, out = 'log/', err = 'logerr/', project_dir = ''):
    err_path = f'{project_dir}{err}$PBS_JOBNAME.err'
    out_path = f'{project_dir}{out}$PBS_JOBNAME.out'
    call_path = f'{project_dir}{call}'
    return bash(f'qsub -q {queue} -l nodes=1:ppn={ppn} -j oe -o {out_path} -e {err_path} -l walltime={walltime}:00:00 {call_path}')

def vcf2plink(vcf = 'round9_1.vcf.gz', out_path = 'zzplink_genotypes/allgenotypes_r9.1'):
    bash(f'plink --thread-num 16 --vcf {vcf} --chr-set 20 no-xy --set-missing-var-ids @:# --make-bed --out {out_path}')
    
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

    use_tscc_modules: list = []
        list of strings to load modules from a HPC using 'module load'

    all_genotypes: str = '/projects/ps-palmer/apurva/riptide/genotypes/round9_1'
        path to file with all genotypes, we are currently at the round 9.1 

    gtca_path: str = ''
        path to gcta64 program for GWAS, if not provided will use the gcta64 dowloaded with conda

    data: pd.DataFrame() = pd.DataFrame()
        pandas dataframe that contains all the phenotypic data necessary to run gcta, 
        genomic identity of a sample has to be provided in the 'rfid' column

    traits: list = []
        list of strings of the phenotype columns that will be used for the GWAS analysis
        this could be integrated with the data dictionaries of the project in the future

    project_name: str = 'test'
        name of the project being ran, 
        do not include date, run identifiers 
        this will be used for the phewas db, so remember to follow the naming conventions in
        https://docs.google.com/spreadsheets/d/1S7_ZIGpMkNmIjhAKUjHBAmcieDihFS-47-_XsgGTls8/edit#gid=1440209927

    phewas_db: str = 'phewasdb.parquet.gz'
        path to the phewas database file 
        The phewas db will be maintained in the parquet format but we are considering using apache feather for saving memory
        this could be integrated with tscc by using scp of the dataframe
        curretly we are using just one large file, but we should consider breaking it up in the future

    threads: int = os.cpu_count()
        number of threads when running multithread code
        default is the number of cpus in the machine

    Atributes
    ---------
    gcta
    path
    all_genotypes
    df
    traits
    phewas_db
    project_name
    sample_path
    genotypes_subset
    genotypes_subset_vcf
    autoGRM
    xGRM
    log
    thrflag
    chrList

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
                 use_tscc_modules: list = [],
                 all_genotypes: str = 'round10.vcf.gz',
                 gtca_path: str = '',
                 data: pd.DataFrame() = pd.DataFrame(),
                 traits: list = [],
                 trait_descriptions: list = [],
                 project_name: str = 'test',
                 phewas_db: str = 'phewasdb.parquet.gz',
                 threads: int = os.cpu_count()):

        
        if use_tscc_modules: bash(f'module load {" ".join(use_tscc_modules)}')
        self.gtca = 'gcta64' if not gtca_path else gtca_path
        self.path = path
        self.all_genotypes = all_genotypes
        
        df = data
        df.columns = df.columns.str.lower()
        if 'vcf' in self.all_genotypes:
            sample_list_inside_genotypes = vcf_manipulation.get_vcf_header(self.all_genotypes)
        else:
            sample_list_inside_genotypes = pd.read_csv(self.all_genotypes+'.fam', header = None, sep=' ', dtype = str)[0].to_list()
        df = df.sort_values('rfid').reset_index(drop = True).drop_duplicates(subset = ['rfid']).dropna(subset = 'rfid')
        self.df = df[df.astype(str).rfid.isin(sample_list_inside_genotypes)].copy()
        
        if self.df.shape[0] != df.shape[0]:
            missing = set(df.rfid.astype(str).unique()) - set(self.df.rfid.astype(str).unique())
            print(f"missing {len(missing)} rfids for project {project_name}")
            
        
        self.traits = [x.lower() for x in traits]
        self.make_dir_structure()
        for trait in self.traits:
            trait_file = f'{self.path}data/pheno/{trait}.txt'            
            self.df[['rfid', 'rfid', trait]].fillna('NA').astype(str).to_csv(trait_file,  index = False, sep = ' ', header = None)
            
        self.get_trait_descriptions = defaultdict(lambda: 'UNK', {k:v for k,v in zip(traits, trait_descriptions)})
        
        self.phewas_db = phewas_db
        self.project_name = project_name
        
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.genotypes_subset = f'{self.path}genotypes/genotypes'
        self.genotypes_subset_vcf = f'{self.path}genotypes/genotypes_subset_vcf.vcf.gz'
        
        self.autoGRM = f'{self.path}grm/AllchrGRM'
        self.xGRM = f'{path}grm/xchrGRM'
        self.log = pd.DataFrame( columns = ['function', 'call', 'out'])
        self.thrflag = f'--thread-num {threads}'
        self.print_call = True
        self.chrList = ['x' if x == 21 else x for x in range(1,22)]
        self.failed_full_grm = False
        
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.sample_sex_path = f'{self.path}genotypes/sample_rfids_sex_info.txt'
        self.sample_sex_path_gcta = f'{self.path}genotypes/sample_rfids_sex_info_gcta.txt'
        
        
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
            
    def make_dir_structure(self, folders: list = ['data', 'genotypes', 'grm', 'log', 'logerr', 
                                            'results', 'temp', 'data/pheno', 'results/heritability', 
                                             'results/gwas',  'results/loco', 'results/qtls','results/eqtl',
                                                  'results/phewas', 'temp/r2', 'results/lz/']):
        
        '''
        creates the directory structure for the project
        '''
        for folder in folders:
            os.makedirs(f'{self.path}{folder}', exist_ok = True)
            
            
    def subsetSamplesFromAllGenotypes(self,samplelist: list = [], sexfmt: str = 'M|F',  sexColumn: str = 'sex',
                                      use_rfid_from_df = True, sourceFormat = 'plink', 
                                      geno: float = .1, maf: float = .005, hwe: float = 1e-10,hwex: float = 1e-20, **kwards ):
        
        '''
        this function will get the large round vcf (or plink) file, subset it based of the rfid from the dataframe
        (or a sample list if wanted), and filter based on missingness, minor allele frequency and hwe equilibrium
        '''
        
        
        funcName = inspect.getframeinfo(inspect.currentframe()).function
        
        os.makedirs(f'{self.path}genotypes', exist_ok = True)
        
        mfList = sorted(sexfmt.split('|'))
        if sourceFormat == 'vcf':
            sample_list_inside_genotypes = vcf_manipulation.get_vcf_header(self.all_genotypes)
        elif sourceFormat in ['bfile', 'plink']:
            sample_list_inside_genotypes = pd.read_csv(self.all_genotypes+'.fam', header = None, sep=' ', dtype = str)[0].to_list()
        
        dff = self.df[self.df.astype(str).rfid.isin(sample_list_inside_genotypes)].copy()
        dff.rfid = dff.rfid.astype(str)
        dff['plink_sex'] = dff[sexColumn].apply(lambda x: self.plink_sex_encoding(x, female_code=mfList[0] ,male_code=mfList[1]))
        
        if len(samplelist) > 0:
            dff = dff.loc[dff.rfid.isin(samplelist)]
            
        self.sample_path = f'{self.path}genotypes/sample_rfids.txt'
        self.sample_sex_path = f'{self.path}genotypes/sample_rfids_sex_info.txt'
        
        dff[['rfid', 'rfid']].to_csv(self.sample_path, index = False, header = None, sep = ' ')
        dff[['rfid', 'plink_sex']].to_csv(self.sample_sex_path, index = False, header = None, sep = ' ')
        self.sample_names = dff.rfid.to_list()
        
        
        fmt_call = {'vcf': 'vcf', 'plink': 'bfile'}[sourceFormat]        
        
        extra_params = f'--geno {geno} --maf {maf} --hwe {hwe} --set-missing-var-ids @:# --chr-set 20 no-xy' #--double-id 
        
        sex_flags = f'--update-sex {self.sample_sex_path}'
        
        self.bashLog(f'plink --{fmt_call} {self.all_genotypes} --keep {self.sample_path} {extra_params} {sex_flags} {self.thrflag} --make-bed --out {self.genotypes_subset}',
                    funcName)
        
        tempfam = pd.read_csv(self.genotypes_subset+ '.fam', header = None, sep = '\s+', dtype = str)
        tempfam = tempfam.merge(dff[['rfid','plink_sex']], left_on = 0, right_on='rfid', how = 'left').fillna(0)
        tempfam.iloc[:, 4] = tempfam['plink_sex']
        tempfam.drop(['rfid', 'plink_sex'], axis = 1).to_csv(self.genotypes_subset+ '.fam', header = None, sep = ' ', index = False)
        
        
    def SubsampleMissMafHweFilter(self, sexfmt: str = 'M|F',  sexColumn: str = 'sex',  
                                  geno: float = .1, maf: float = .005, hwe: float = 1e-10,
                                  sourceFormat = 'vcf', remove_dup: bool = True, print_call: bool = False, **kwards):
        
        '''
        Function to subset and refilter the genotypes given a set of geno, maf and hwe.
        
        
        Parameters
        ----------
        sexfmt: str = 'M|F'
            format of the sex column it should be written as the options for describing sex in the self.df
            M -> male, F -> female becomes "M|F"
            0 -> male, 1 -> female becomes "0|1"
            male -> male, fem -> female becomes "male|fem"
        sexColumn: str = 'sex'
            which column of self.df contains the sex information
        geno: float = .1
            missingness threshold
        maf: float = .005
        hwe: float = 1e-10
        sourceFormat = 'vcf'
        remove_dup: bool = True
        print_call: bool = False
        
        Design
        ------
        1.It cross validates the rfids from self.df and the genotypes file
        2.Then it generates 4 files based on self.df 
            for female samples  one containing the rfids and another containing the rfids 
            and the other sex information encoded' on the plink format (0 is missing, 
            1 is male 2 is female)
            for male samples  one containing the rfids and another containing the rfids 
            and the other sex information encoded' on the plink format (0 is missing, 
            1 is male 2 is female)
        
        3.For female samples we apply all the filters
        4.For males we apply the all the filters on the autossomes but for the X we do not apply the HWE
        5.then the resulting plink files are merged and the path is assigned to self.genotypes_subset
        '''
        print(f'starting subsample plink ... {self.project_name}') 
        funcName = inspect.getframeinfo(inspect.currentframe()).function
        
               
        fmt_call = {'vcf': 'vcf', 'plink': 'bfile'}[sourceFormat]  
        rmv = f' --set-missing-var-ids @:# ' if remove_dup else ' '#--double-id 
        sub = self.genotypes_subset
        
        mfList = sorted(sexfmt.split('|'))
        
        dff = self.df.copy()
        dff.rfid = dff.rfid.astype(str)
        dff['plink_sex'] = dff[sexColumn].apply(lambda x: self.plink_sex_encoding(x, female_code=mfList[0] ,male_code=mfList[1]))
        
        dff[['rfid', 'rfid']].to_csv(self.sample_path, index = False, header = None, sep = ' ')
        dff[['rfid', 'plink_sex']].to_csv(self.sample_sex_path, index = False, header = None, sep = ' ')
        dff[['rfid', 'rfid', 'plink_sex']].to_csv(self.sample_sex_path_gcta, index = False, header = None, sep = ' ')
        self.sample_names = dff.rfid.to_list()
        
        
        self.samples = {}
        self.samples_sex = {}
        for num, sx in enumerate(tqdm(mfList)): 
            
            dff_sm = dff[dff[sexColumn] == sx]
            dff_sm['plink_sex'] = dff_sm[sexColumn].apply(lambda x: self.plink_sex_encoding(x, female_code=mfList[0] ,male_code=mfList[1]))
            
            self.samples[sx] = f'{self.path}genotypes/sample_rfids_{sx}.txt'
            dff_sm[['rfid', 'rfid']].to_csv(self.samples[sx], index = False, header = None, sep = ' ')
            
            self.samples_sex[sx] = f'{self.path}genotypes/sample_rfids_sex_info_{sx}.txt'
            dff_sm[['rfid', 'plink_sex']].to_csv(self.samples_sex[sx], index = False, header = None, sep = ' ')
            
            filtering_flags = f' --geno {geno} --maf {maf} --hwe {hwe}'
            filtering_flags_justx = f''
            extra_flags = f'--not-chr X'  if num == 1 else ''
            sex_flags = f'--update-sex {self.samples_sex[sx]}'
            
            self.bashLog(f'plink --{fmt_call} {self.all_genotypes} --keep {self.samples[sx]}  --chr-set 20 no-xy \
                          {sex_flags} {filtering_flags} {rmv} {extra_flags} {self.thrflag} --make-bed --out {sub}_{sx}',
                        f'{funcName}_subseting_{sx}', print_call=print_call)#--autosome-num 20
            
            if num == 1:
                self.bashLog(f'plink --bfile {self.all_genotypes} --keep {self.samples[sx]}  --chr-set 20 no-xy\
                                --chr x {self.thrflag} --geno {geno} --maf {maf} --make-bed --out {sub}_{sx}_xchr',
                        f'{funcName}_maleXsubset{sx}',  print_call=print_call) #--out {sub}_{sx}_xchr {sub}_{sx} 
                male_1_x_filenames = [aa for aa in [f'{sub}_{sx}', f'{sub}_{sx}_xchr'] if len(glob(aa+'.*')) >= 5]
                male_gen_filenames = f'{self.path}genotypes/temp_male_filenames'
                pd.DataFrame(male_1_x_filenames).to_csv(male_gen_filenames, index = False, header = None)
            else: female_hwe = f'{sub}_{sx}'
                
        print('merging sexes')        
        self.bashLog(f'plink --bfile {female_hwe} --merge-list {male_gen_filenames} {self.thrflag} \
                       --geno {geno} --maf {maf} --chr-set 20 no-xy --make-bed --out {sub}',
                        f'{funcName}_mergeSexes', print_call=print_call) # {filtering_flags}
        
        self.genotypes_subset = f'{sub}'
        
        tempfam = pd.read_csv(self.genotypes_subset+ '.fam', header = None, sep = '\s+', dtype = str)
        tempfam = tempfam.merge(dff[['rfid','plink_sex']], left_on = 0, right_on='rfid', how = 'left').fillna(0)
        tempfam.iloc[:, 4] = tempfam['plink_sex']
        tempfam.drop(['rfid', 'plink_sex'], axis = 1).to_csv(self.genotypes_subset+ '.fam', header = None, sep = ' ', index = False)
        
        
    def generateGRM(self, autosome_list: list = list(range(1,21)), print_call: bool = True, allatonce: bool = False,
                    extra_chrs: list = ['xchr'], just_autosomes: bool = True, just_full_grm: bool = True,
                   full_grm: bool = True, **kwards):
        
        '''
        generates the grms, one per chromosome and one with all the chromossomes
        
        Parameters
        ----------
        autosome_list: list = list(range(1,21))
            list of chromosomes that will be used
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
        
        if allatonce:
            auto_flags = f'--autosome-num {int(len(autosome_list))} --autosome' if just_autosomes else ''
            sex_flags = f'--update-sex {self.sample_sex_path_gcta} --dc 1' #f' --sex {self.sample_sex_path}'
            
            self.bashLog(f'{self.gtca} {self.thrflag} --bfile {self.genotypes_subset} {sex_flags}\
                           --make-grm-bin {auto_flags} --out {self.autoGRM}_allatonce',
                           funcName, print_call = print_call) # 
            
        all_filenames_partial_grms = pd.DataFrame(columns = ['filename'])
        
        if 'xchr' in extra_chrs:
            self.bashLog(f'{self.gtca} {self.thrflag} --bfile {self.genotypes_subset} --autosome-num 20 \
                           --make-grm-xchr --out {self.xGRM}',
                        f'{funcName}_chrX', print_call = False)

        all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = self.xGRM

        for c in tqdm(autosome_list):
            self.bashLog(f'{self.gtca} {self.thrflag} --bfile {self.genotypes_subset} --chr {c} --autosome-num 20\
                         --make-grm-bin --out {self.path}grm/{c}chrGRM',
                        f'{funcName}_chr{c}',  print_call = False)

            all_filenames_partial_grms.loc[len(all_filenames_partial_grms), 'filename'] = f'{self.path}grm/{c}chrGRM'

        all_filenames_partial_grms.to_csv(f'{self.path}grm/listofchrgrms.txt', index = False, sep = ' ', header = None)

        self.bashLog(f'{self.gtca} {self.thrflag} --mgrm {self.path}grm/listofchrgrms.txt \
                       --make-grm-bin --out {self.autoGRM}', f'{funcName}_mergedgrms',  print_call = False )

        #if os.path.exists(f'{self.autoGRM}.grm.bin'): 
        #    self.failed_full_grm = False 
        
        return 1
            
                

    def snpHeritability(self, print_call: bool = False, save: bool = True, **kwards):
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
                self.bashLog(f'{self.gtca} --reml {self.thrflag} --reml-no-constrain --autosome-num 20\
                                           --pheno {trait_file} --mgrm {self.path}grm/listofchrgrms.txt --out {out_file}',
                            f'snpHeritability_{trait}', print_call = print_call) 
            else:
                self.bashLog(f'{self.gtca} --reml {self.thrflag}  --autosome-num 20\
                                           --pheno {trait_file} --grm {self.autoGRM} --out {out_file}',
                            f'snpHeritability_{trait}', print_call = print_call) #--autosome
            
            a = pd.read_csv(f'{out_file}.hsq', skipfooter=6, sep = '\t',engine='python')
            b = pd.read_csv(f'{out_file}.hsq', skiprows=6, sep = '\t', header = None, index_col = 0).T.rename({1: trait})
            newrow = pd.concat(
                [a[['Source','Variance']].T[1:].rename({i:j for i,j in enumerate(a.Source)}, axis = 1).rename({'Variance': trait}),
                b],axis =1 )
            h2table= pd.concat([h2table,newrow])
            
        if save: h2table.to_csv(f'{self.path}results/heritability/heritability.tsv', sep = '\t')
        return h2table
        
    
    def GWAS(self, traitlist: list = [] ,subtract_grm: bool = True, loco: bool = True , run_per_chr: bool = False,
             print_call: bool = False, nchr:int = 21, **kwards):
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
                self.bashLog(f"{self.gtca} {self.thrflag} {grm_flag} \
                --autosome-num 20\
                --pheno {self.path}data/pheno/{trait}.txt \
                --bfile {self.genotypes_subset} \
                --mlma{loco_flag} \
                --out {self.path}results/gwas/{trait}",\
                            f'GWAS_{loco_flag[1:]}_{trait}',  print_call = print_call)
                if not os.path.exists(f'{self.path}results/gwas/{trait}.loco.mlma')+os.path.exists(f'{self.path}results/gwas/{trait}.mlma'):
                    print(f"couldn't run trait: {trait}")
                    self.GWAS(traitlist = traitlist, run_per_chr = True, print_call= print_call)
                    return 2
            ranges = [21]
        else:
            print('running gwas per chr per trait...')
            ranges =  range(1,nchr+1)
        
        for trait, chrom in tqdm(list(itertools.product(traitlist,ranges))):
            if chrom == 21: chromp2 = 'x'
            else: chromp2 = chrom
            self.bashLog(f'{self.gtca} {self.thrflag} --pheno {self.path}data/pheno/{trait}.txt --bfile {self.genotypes_subset} \
                                       --grm {self.path}grm/AllchrGRM \
                                       --autosome-num 20\
                                       --chr {chrom} \
                                       --mlma-subtract-grm {self.path}grm/{chromp2}chrGRM \
                                       --mlma \
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
        
        
    def callQTLs(self, threshold: float = 5.3, window: int = 1e6, subterm: int = 2,  annotate_genome: str = 'rn7',
                 ldwin = 1e6, ldkb = 11000, ldr2 = .4, qtl_dist = 2*1e6, nchr: int = 21, NonStrictSearchDir = True, **kwards):
        
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
         nchr: int = 21
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
        chr_list = ['x' if x == 21 else x for x in range(1,nchr+1)]
        
        if not NonStrictSearchDir:
            topSNPs = pd.DataFrame()
            for t, chrom in tqdm(list(itertools.product(self.traits, range(1,nchr+1)))):
                    if chrom == 21 : chrom = 'x'
                    filename = f'{self.path}results/gwas/{t}_chrgwas{chrom}.mlma'
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
                correlated_snps = df.loc[idx- window//2: idx + window//2].query('p > @maxp.p - @subterm')
                qtl = True if correlated_snps.shape[0] > 2 else False

                

                ldfilename = f'{self.path}temp/r2/temp_qtl_n_{t}'
                self.bashLog(f'plink --bfile {self.genotypes_subset} --chr {c} --ld-snp {maxp.SNP} \
                                     --ld-window {ldwin} {self.thrflag} \
                                     --nonfounders --r2  \
                                     --ld-window-r2 {ldr2} --out {ldfilename}',\
                             f'qlt_{t}', False )#--ld_window_kb {ldkb} --nonfounders might be able to remove

                try: 
                    plinkrestemp =  pd.read_csv(f'{ldfilename}.ld', sep = r'\s+')
                    ldSNPS = plinkrestemp.SNP_B.to_list() + [maxp.SNP]
                    ldSNPS_LEN = plinkrestemp.query('R2 > @ldr2').BP_B.agg(lambda x: (x.max()-x.min())/1e6)
                    df = df.query('~(@idx - @qtl_dist//2 < index < @idx + @qtl_dist//2) and (SNP not in @ldSNPS)')
                except:
                    ldSNPS = [maxp.SNP]
                    ldSNPS_LEN = 0
                    df = df.query('(SNP not in @ldSNPS)') ##### should this be different than the one above?
                #if sum(cnt.values()) % 10 == 0: print(cnt)
                            
                out = pd.concat([out,
                                 maxp.to_frame().T.assign(QTL= qtl, interval_size = '{:.2f} Mb'.format(ldSNPS_LEN))],
                                 axis = 0)

        out =  out.reset_index().rename({'index': 'bp'}, axis = 1).sort_values('trait')#.assign(project = self.project_name)
        out['trait_description'] = out.trait.apply(lambda x: self.get_trait_descriptions[x])
        out['trait'] = out.trait.apply(lambda x: x.replace('regressedlr_', ''))
        self.allqtlspath = f'{self.path}/results/qtls/allQTLS.csv'
        out.to_csv(self.allqtlspath, index = False)
        
        print(f'generating locuszoom info for project {self.project_name}')
        for name, row in tqdm(list(out.iterrows())):
            ldfilename = f'{self.path}results/lz/temp_qtl_n_@{row.trait}@{row.SNP}'
            r2 = self.plink(bfile = self.genotypes_subset, chr = row.Chr, ld_snp = row.SNP, ld_window_r2 = 0.001, r2 = 'dprime',\
                                    ld_window = 100000, thread_num = 12, ld_window_kb =  6000, nonfounders = '').loc[:, ['SNP_B', 'R2', 'DP']] 
            gwas = pd.concat([pd.read_csv(x, sep = '\t') for x in glob(f'{self.project_name}/results/gwas/*{row.trait}.loco.mlma') \
                                + glob(f'{self.project_name}/results/gwas/*{row.trait}.mlma')])
            tempdf = pd.concat([gwas.set_index('SNP'), r2.rename({'SNP_B': 'SNP'}, axis = 1).set_index('SNP')], join = 'inner', axis = 1)
            tempdf = self.annotate(tempdf.reset_index(), annotate_genome, 'SNP', save = False).set_index('SNP')
            tempdf.to_csv( f'{self.project_name}/results/lz/lzplottable@{row.trait}@{row.SNP}.tsv', sep = '\t')
        
        
        return out.set_index('SNP')   
    
    def phewas(self, qtltable: pd.DataFrame(), ld_window: int = int(3e6), pval_threshold: float = 1e-4, nreturn: int = 30 ,r2_threshold: float = .8,\
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
        db_vals = pd.read_parquet(self.phewas_db).query(f'p < {pval_threshold} and project != "{self.project_name}"')  #, compression='gzip'      
        
        table_exact_match = db_vals.merge(qtltable.reset_index(), on = 'SNP', how = 'inner', suffixes = ('_phewasdb', '_QTL'))
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
        
        
        table_window_match = db_vals.merge(nearby_snps.reset_index(), left_on= 'SNP', 
                                                         right_on='NearbySNP', how = 'inner', suffixes = ('_phewasdb', '_QTL'))
        
        
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
        return out
        

        
        
    def eQTL(self, qtltable: pd.DataFrame(), pval_thresh: float = 1e-4, r2_thresh: float = .8, nreturn: int =30, ld_window: int = 3e6,\
            tissue_list: list = ['Adipose', 'BLA','Brain','Eye','IL','LHb','Liver','NAcc','NAcc2','OFC','PL','PL2'],\
            annotate = True, annotate_genome = 'rn7', **kwards) -> pd.DataFrame():
        
        '''
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
        if qtltable.shape == (0,0): qtltable = pd.read_csv(self.annotatedtablepath).set_index('SNP')
        out = []
        for tissue in tqdm(tissue_list,  position=0, desc="tissue", leave=True):

            tempdf = pd.read_csv(f'https://ratgtex.org/data/eqtl/{tissue}.cis_qtl_signif.txt.gz', sep = '\t').assign(tissue = tissue)\
                                                                                                             .rename({'variant_id': 'SNP'}, axis = 1)
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
            out = self.annotate(out, annotate_genome, 'NearbySNP', save = False)
        self.eqtl_path = f'{self.path}results/eqtl/eqtl.csv'
        out.to_csv(self.eqtl_path, index= False)
        
        return out

    def annotate(self, qtltable: pd.DataFrame(), genome: str = 'rn7', 
                 snpcol: str = 'SNP', save: bool = True, **kwards) -> pd.DataFrame():
        
        '''
        This function annotates a QTL table with the snpEff tool, 
        which is used to query annotations for QTL and phewas results. 
                
        Parameters 
        ----------

        qtltable: pd.DataFrame
            the QTL table to be annotated
        genome: str = 'rn6'
            the genome to use for annotation (default is 'rn6')
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
        d = {'rn6': 'Rnor_6.0.99', 'rn7':'mRatBN7.2.105'}[genome]
        #bash('java -jar snpEff/snpEff.jar download -v Rnor_6.0.99')
        #bash('java -jar snpEff/snpEff.jar download -v mRatBN7.2.105')    
        
        temp  = qtltable.reset_index()\
                        .loc[:,[ 'Chr', 'bp', snpcol, 'A1', 'A2']]\
                        .assign(QUAL = 40, FILTER = 'PASS' ,INFO = '', FORMAT = 'GT:GQ:DP:HQ')
        temp.columns = ["##CHROM","POS","ID","REF","ALT", 'QUAL', 'FILTER', 'INFO', 'FORMAT']
        temp['##CHROM'] = 'chr'+ temp['##CHROM'].astype(str)
        vcf_manipulation.pandas2vcf(temp, f'{self.path}temp/test.vcf', metadata='')
        a = bash(f'java -Xmx8g -jar snpEff/snpEff.jar {d} -no-intergenic -no-intron -noStats {self.path}temp/test.vcf', print_call = False )# 'snpefftest',
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
        
        return out 
    
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
            bash('../../mc anonymous set public myminio/tsanches_dash_genotypes --recursive')


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
        


    
    
