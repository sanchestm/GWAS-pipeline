#!/usr/bin/env python 
from gwas_class_auto import *
import sys
import pandas as pd

dictionary = defaultdict(lambda: 0, {k.replace('-', ''):v for k,v in [(x + '=1' if '=' not in x else x).split('=') for x in sys.argv[1:]] })

def typeconverter(s):
    s= str(s)
    if s.lower() in ['1', 'true']: return 1
    if s.lower() in ['0', 'false']: return 0
    try: return int(s)
    except: pass
    try: return float(s)
    except: return s
        
def kw(d, prefix):
    if prefix[-1] != '_': prefix += '_'
    return {k.replace(prefix, ''):typeconverter(v) for k,v in d.items() if (k[:len(prefix)] == prefix)}

path = dictionary['path'].rstrip('/') + '/' if (dictionary['path'] ) else ''

pj = dictionary['project'].rstrip('/')  if (dictionary['project'] ) else 'test'

if not dictionary['genotypes']:
    dictionary['genotypes'] = '/tscc/projects/ps-palmer/gwas/databases/rounds/r10.2.1'
    
if not dictionary['threshold']: dictionary['threshold'] = 5.58
if dictionary['threshold'] != 'auto':  dictionary['threshold']= float(dictionary['threshold'])

if not dictionary['threshold05']: dictionary['threshold05'] = 5.58
dictionary['threshold05'] = float(dictionary['threshold05'])
    
if not dictionary['genome']: dictionary['genome'] = 'rn7'
    
if not dictionary['researcher']: dictionary['researcher'] = 'tsanches'
if not dictionary['n_autosome']: dictionary['n_autosome'] = 20
if not dictionary['round']: dictionary['round'] = '10.1.0'

if not dictionary['gwas_version']: dictionary['gwas_version'] = '0.1.2'
    
if not dictionary['snpeff_path']: dictionary['snpeff_path'] = 'snpEff/'
    
if not dictionary['phewas_path']: dictionary['phewas_path'] = 'phewasdb.parquet.gz'
    
if not dictionary['regressout']:
    df = pd.read_csv(f'{path}{pj}/processed_data_ready.csv', 
                     dtype = {'rfid': str}).drop_duplicates(subset = 'rfid') 
    if not dictionary['traits']:
        traits_ = df.columns[df.columns.str.startswith('regressedlr_')]#cluster_bysex
    elif 'prefix_' in dictionary['traits']: 
        pref = dictionary['traits'].replace('prefix_', '')
        traits_ = df.columns[df.columns.str.startswith(f'regressedlr_{pref}')]
    else: traits_ = dictionary['traits'].split(',')
    try: traits_d = get_trait_descriptions_f(pd.read_csv(f'{path}{pj}/data_dict_{pj}.csv'), traits_)
    except: traits_d = ['UNK' for x in range(len(traits_))]
else:
    rawdata = dictionary['regressout'] if (len(dictionary['regressout']) > 1) else f'{path}{pj}/raw_data.csv'
    df = pd.read_csv(rawdata, dtype = {'rfid': str}).drop_duplicates(subset = 'rfid') 
    traits_, traits_d = [], []
#sys.stdout = open(f'{path}{pj}/GWASpipelineCLIout.txt', 'w')
gwas = gwas_pipe(path = f'{path}{pj}/',
             all_genotypes = dictionary['genotypes'], #'round9_1.vcf.gz',
             data = df,
             project_name = pj.split('/')[-1],
             n_autosome = int(dictionary['n_autosome']),
             traits = traits_,
             genome = dictionary['genome'],
             founderfile = dictionary['founder_genotypes'],
             snpeff_path= dictionary['snpeff_path'],
             phewas_db = dictionary['phewas_path'],
             trait_descriptions= traits_d,
             threshold = dictionary['threshold'],
             threshold05 = dictionary['threshold05'],
             threads = dictionary['threads'])
printwithlog(path)
printwithlog(pj)
for k,v in dictionary.items(): printwithlog(f'--{k} : {v}')
if dictionary['clear_directories'] and not dictionary['skip_already_present_gwas']: gwas.clear_directories()
if dictionary['impute']: 
    gbcols =  [] if not dictionary['groupby'] else dictionary['groupby'].split(',')
    gwas.impute_traits(crosstrait_imputation = dictionary['crosstrait_imputation'],  groupby_columns=dictionary['groupby'].split(','))
if dictionary['regressout']: 
    if not dictionary['timeseries']:  
        if not dictionary['groupby']: gwas.regressout(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'))
        else: gwas.regressout_groupby(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'), groupby_columns=dictionary['groupby'].split(','))
    else:  
        if not dictionary['groupby']: gwas.regressout_timeseries(data_dictionary=pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'))
        else: gwas.regressout_timeseries(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'), groupby_columns=dictionary['groupby'].split(','))

if dictionary['latent_space']: gwas.add_latent_spaces()
if dictionary['subset']:  ###essential
    kws = kw(dictionary, 'subset_')
    if dictionary['subset_make_figures'] : gwas.SubsetAndFilter(makefigures = True, **kws)
    else: gwas.SubsetAndFilter(makefigures = False, **kws)
if dictionary['grm']: 
    gwas.generateGRM(**kw(dictionary, 'grm_')) ###essential
if dictionary['h2']:  gwas.snpHeritability() ###essential
if dictionary['BLUP']: gwas.BLUP()
if dictionary['BLUP_predict']: gwas.BLUP_predict(dictionary['BLUP_predict']);
if dictionary['gwas']: gwas.fastGWAS(skip_already_present=dictionary['skip_already_present_gwas']) ###essential
if dictionary['db']: gwas.addGWASresultsToDb(researcher=dictionary['researcher'],
                                             round_version=dictionary['round'], 
                                             gwas_version=dictionary['gwas_version'])
if dictionary['qtl']: ###essential
    qtl_add_founder = True if (dictionary['founder_genotypes'] not in [ 'none', 'None', 0]) else False
    try: qtls = gwas.callQTLs( NonStrictSearchDir=False,   add_founder_genotypes = qtl_add_founder )
    except: qtls = gwas.callQTLs( NonStrictSearchDir=True)
    #gwas.annotate(qtls)
    gwas.effectsize() 
if dictionary['effect']: gwas.effectsize() 
if dictionary['gcorr']: ###essential
    gwas.genetic_correlation_matrix_old()
    gwas.make_heritability_figure(display = False)
if dictionary['manhattanplot'] or dictionary['porcupineplot']: gwas.porcupineplotv2() ###essential
if dictionary['phewas']:gwas.phewas(annotate=True, pval_threshold = 1e-4, nreturn = 1, r2_threshold = .4)  ###essential
if dictionary['eqtl']:gwas.eQTL(annotate= True) ###essential
if dictionary['sqtl']:gwas.sQTL() ###essential
if dictionary['goea']:gwas.GeneEnrichment() ###essential
if dictionary['locuszoom']: gwas.locuszoom2(**kw(dictionary, 'locuszoom_'))  ###essential
if dictionary['h2fig']: gwas.make_heritability_figure(display = False) 
if dictionary['report']:
    kws = kw(dictionary, 'report_')
    gwas.report(round_version=dictionary['round'], gwas_version=dictionary['gwas_version'], **kws)
    gwas.copy_results() ###essential
if dictionary['store']:gwas.store(researcher=dictionary['researcher'],
                                  round_version=dictionary['round'], 
                                  gwas_version=dictionary['gwas_version'],  
                                  remove_folders=False) ###essential
try: 
    if dictionary['publish']:gwas.copy_results() ###essential
except: 
    print('setting up the minio is necessary')
gwas.print_watermark()
