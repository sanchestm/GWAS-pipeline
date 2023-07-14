#!/usr/bin/env python 
from gwas_class_auto import *
import sys
import pandas as pd

dictionary = defaultdict(lambda: 'dont', {k.replace('-', ''):v for k,v in [(x + '=1' if '=' not in x else x).split('=') for x in sys.argv[1:]] })

path = dictionary['path'].rstrip('/') + '/' if (dictionary['path'] != 'dont') else ''
print(path)
pj = dictionary['project'].rstrip('/')  if (dictionary['project'] != 'dont') else 'test'
print(pj)
if dictionary['genotypes'] == 'dont' :
    dictionary['genotypes'] = '/projects/ps-palmer/tsanches/gwaspipeline/gwas/zzplink_genotypes/round10'

if dictionary['genome'] == 'dont' : dictionary['genome'] = 'rn7'
    
if dictionary['researcher'] == 'dont': dictionary['researcher'] = 'tsanches'
    
if dictionary['round'] == 'dont': dictionary['round'] = '10.0.0'

if dictionary['gwas_version'] == 'dont': dictionary['gwas_version'] = '0.1.2'
    
if dictionary['snpeff_path'] == 'dont': dictionary['snpeff_path'] = 'snpEff/'
    
if dictionary['phewas_path'] == 'dont': dictionary['phewas_path'] = 'phewasdb.parquet.gz'
    
if dictionary['regressout'] == 'dont':
    df = pd.read_csv(f'{path}{pj}/processed_data_ready.csv', dtype = {'rfid': str}).drop_duplicates(subset = 'rfid') 
    if dictionary['traits'] == 'dont':
        traits_ = df.columns[df.columns.str.startswith('regressedlr_')]#cluster_bysex
    elif 'prefix_' in dictionary['traits']: 
        pref = dictionary['traits'].replace('prefix_', '')
        traits_ = df.columns[df.columns.str.startswith(f'regressedlr_{pref}')]
    else: traits_ = dictionary['traits'].split(',')
    traits_d = get_trait_descriptions_f(pd.read_csv(f'{path}{pj}/data_dict_{pj}.csv'), traits_)
else:
    rawdata = dictionary['regressout'] if (len(dictionary['regressout']) > 1) else f'{path}{pj}/raw_data.csv'
    df = pd.read_csv(rawdata, dtype = {'rfid': str}).drop_duplicates(subset = 'rfid') 
    traits_, traits_d = [], []

gwas = gwas_pipe(path = f'{path}{pj}/',
             all_genotypes = dictionary['genotypes'], #'round9_1.vcf.gz',
             data = df,
             project_name = pj.split('/')[-1],
             traits = traits_,
             founder_genotypes = dictionary['founder_genotypes'],
             snpeff_path= dictionary['snpeff_path'],
             phewas_db = dictionary['phewas_path'],
             trait_descriptions= traits_d,
             threads = dictionary['threads'])

if dictionary['regressout']!= 'dont': 
    if dictionary['timeseries']== 'dont':  gwas.regressout(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'))
    else:  gwas.regressout_timeseries(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'))
if dictionary['subset']!= 'dont': gwas.subsetSamplesFromAllGenotypes(sourceFormat = 'plink')
if dictionary['grm']!= 'dont':gwas.generateGRM()
if dictionary['h2']!= 'dont': gwas.snpHeritability()
if dictionary['BLUP'] != 'dont': gwas.BLUP()
if dictionary['BLUP_predict'] != 'dont': gwas.BLUP_predict(dictionary['BLUP_predict']);
if dictionary['gwas']!= 'dont': gwas.GWAS()
if dictionary['db']!= 'dont': gwas.addGWASresultsToDb(researcher=dictionary['researcher'], round_version=dictionary['round'], gwas_version=dictionary['gwas_version'])
if dictionary['qtl']!= 'dont': 
    qtl_add_founder = True if ((dictionary['genome'] in ['rn7', 'rn6']) 
                               and (dictionary['founder_genotypes'] not in ['dont', 'none'])) else False
    try: qtls = gwas.callQTLs( NonStrictSearchDir=False, add_founder_genotypes = qtl_add_founder)
    except: qtls = gwas.callQTLs( NonStrictSearchDir=True)
    gwas.annotate(qtls, genome = dictionary['genome'] )
if dictionary['locuszoom'] != 'dont': gwas.locuszoom(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv'), annotate_genome = dictionary['genome']) 
if dictionary['effect'] != 'dont': gwas.effectsize(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv')) 
if dictionary['gcorr'] != 'dont': gwas.genetic_correlation_matrix()
if dictionary['manhattanplot'] != 'dont': gwas.manhattanplot(display = False)
if dictionary['porcupineplot'] != 'dont': gwas.porcupineplot(pd.read_csv(f'{gwas.path}/results/qtls/finalqtl.csv'), display = False)
if dictionary['phewas']!= 'dont':gwas.phewas(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv').set_index('SNP'), annotate=True, pval_threshold = 1e-4, nreturn = 1, r2_threshold = .4, annotate_genome = dictionary['genome']) 
if dictionary['eqtl']!= 'dont':gwas.eQTL(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv').set_index('SNP'), annotate= True, genome = dictionary['genome'])
if dictionary['sqtl']!= 'dont':gwas.sQTL(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv').set_index('SNP'), genome = dictionary['genome'])
if dictionary['report']!= 'dont':gwas.report(round_version=dictionary['round'], gwas_version=dictionary['gwas_version'])
if dictionary['store']!= 'dont':gwas.store(researcher=dictionary['researcher'],round_version=dictionary['round'], gwas_version=dictionary['gwas_version'],  remove_folders=False)
try: 
    if dictionary['publish']!= 'dont':gwas.copy_results()
except: 
    print('setting up the minio is necessary')
gwas.print_watermark()
