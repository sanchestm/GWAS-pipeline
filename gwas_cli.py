#!/usr/bin/env python 
from gwas_class_auto import *
import sys
import pandas as pd

dictionary = {k.replace('-', ''):v for k,v in [(x + '=1' if '=' not in x else x).split('=') for x in sys.argv[1:]] }
dictionary = defaultdict(lambda: 'dont', dictionary)
pj = dictionary['project']
if dictionary['genotypes'] == 'dont' :
    dictionary['genotypes'] = '/projects/ps-palmer/tsanches/gwaspipeline/gwas/zzplink_genotypes/round10'

if dictionary['researcher'] == 'dont': dictionary['researcher'] = 'tsanches'
    
if dictionary['round'] == 'dont': dictionary['round'] = '10.0.0'
if dictionary['gwas_version'] == 'dont': dictionary['gwas_version'] = '0.1.2'
    
if dictionary['snpeff_path'] == 'dont':
    dictionary['snpeff_path'] = '/projects/ps-palmer/tsanches/gwaspipeline/gwas/snpEff/'
    
if dictionary['phewas_path'] == 'dont': dictionary['phewas_path'] = 'phewasdb.parquet.gz'
    
if dictionary['regressout'] == 'dont':
    df = pd.read_csv(f'{pj}/processed_data_ready.csv', dtype = {'rfid': str}).drop_duplicates(subset = 'rfid') 
    if dictionary['traits'] == 'dont':
        traits_ = df.columns[df.columns.str.startswith('regressedlr_')]#cluster_bysex
    elif 'prefix_' in dictionary['traits']: 
        pref = dictionary['traits'].replace('prefix_', '')
        traits_ = df.columns[df.columns.str.startswith(f'regressedlr_{pref}')]
    else: traits_ = dictionary['traits'].split(',')
    traits_d = get_trait_descriptions_f(pd.read_csv(f'{pj}/data_dict_{pj}.csv'), traits_)
else:
    rawdata = dictionary['regressout'] if dictionary['regressout'] != '1' else f'{pj}/raw_data.csv'
    df = pd.read_csv(rawdata, dtype = {'rfid': str}).drop_duplicates(subset = 'rfid') 
    traits_, traits_d = [], []

gwas = gwas_pipe(path = f'{pj}/',
             all_genotypes = dictionary['genotypes'], #'round9_1.vcf.gz',
             data = df,
             project_name = pj,
             traits = traits_ ,
             snpeff_path= dictionary['snpeff_path'],
             phewas_db = dictionary['phewas_path'],
             trait_descriptions= traits_d,
             threads = dictionary['threads'])

if dictionary['regressout']!= 'dont': gwas.regressout(data_dictionary= pd.read_csv(f'{pj}/data_dict_{pj}.csv'))
if dictionary['subset']!= 'dont': gwas.subsetSamplesFromAllGenotypes(sourceFormat = 'plink')
if dictionary['grm']!= 'dont':gwas.generateGRM()
if dictionary['h2']!= 'dont': gwas.snpHeritability()
if dictionary['BLUP'] != 'dont': gwas.BLUP()
if dictionary['BLUP_predict'] != 'dont': gwas.BLUP_predict(dictionary['BLUP_predict']);
if dictionary['gwas']!= 'dont': gwas.GWAS()
if dictionary['db']!= 'dont': gwas.addGWASresultsToDb(researcher=dictionary['researcher'], round_version=dictionary['round'], gwas_version=dictionary['gwas_version'])
if dictionary['qtl']!= 'dont': 
    qtls = gwas.callQTLs( NonStrictSearchDir=False)
    gwas.annotate(qtls)
if dictionary['locuszoom'] != 'dont': gwas.locuszoom(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv')) 
if dictionary['effect'] != 'dont': gwas.effectsize(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv')) 
if dictionary['gcorr'] != 'dont': gwas.genetic_correlation_matrix()
if dictionary['manhattanplot'] != 'dont': gwas.manhattanplot(display = False)
if dictionary['porcupineplot'] != 'dont': gwas.porcupineplot(pd.read_csv(f'{gwas.path}/results/qtls/finalqtl.csv'), display = False)
if dictionary['phewas']!= 'dont':gwas.phewas(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv').set_index('SNP'), annotate=True, pval_threshold = 1e-4, nreturn = 1, r2_threshold = .4) 
if dictionary['eqtl']!= 'dont':gwas.eQTL(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv').set_index('SNP'), annotate= True)
if dictionary['sqtl']!= 'dont':gwas.sQTL(pd.read_csv(f'{gwas.path}results/qtls/finalqtl.csv').set_index('SNP'))
if dictionary['report']!= 'dont':gwas.report(round_version=dictionary['round'])
if dictionary['store']!= 'dont':gwas.store(researcher=dictionary['researcher'],round_version=dictionary['round'] , gwas_version=dictionary['gwas_version'],  remove_folders=False)
try: if dictionary['publish']!= 'dont':gwas.copy_results()
gwas.print_watermark()
