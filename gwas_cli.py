#!/usr/bin/env python 
from gwas_class_auto import *
import sys
import pandas as pd

dictionary = {k.replace('-', ''):v for k,v in [(x + '=1' if '=' not in x else x).split('=') for x in sys.argv[1:]] }
dictionary = defaultdict(lambda: 'dont', dictionary)
pj = dictionary['project']

def get_trait_descriptions_f(project, traits):
    qq =  pd.read_csv(f'{project}/data_dict_{project}.csv')
    out = []
    for trait in traits:
        try: 
            out +=  [qq[qq.measure == trait.replace('regressedlr_', '')].description.iloc[0]]
        except:
            out +=  ['UNK']
    return out

df = pd.read_csv(f'{pj}/processed_data_ready.csv').drop_duplicates(subset = 'rfid') 
traits_ = df.columns[df.columns.str.contains('regressedlr_')]#cluster_bysex
traits_d = get_trait_descriptions_f(pj, traits_)
gwas = gwas_pipe(path = f'{pj}/',
             all_genotypes = 'zzplink_genotypes/round10', #'round9_1.vcf.gz',
             data = df,
             project_name = pj,
             traits = traits_ ,
             trait_descriptions= traits_d,
             threads = dictionary['threads'])
if dictionary['subset']!= 'dont': gwas.subsetSamplesFromAllGenotypes(sourceFormat = 'plink')
if dictionary['grm']!= 'dont':gwas.generateGRM()
if dictionary['h2']!= 'dont': gwas.snpHeritability()
if dictionary['BLUP'] != 'dont': gwas.BLUP()
if dictionary['BLUP_predict'] != 'dont': gwas.BLUP_predict(dictionary['BLUP_predict']);
if dictionary['gwas']!= 'dont': gwas.GWAS()
if dictionary['db']!= 'dont': gwas.addGWASresultsToDb(researcher='tsanches', round_version='10.0.0', gwas_version='0.1.0')
if dictionary['qtl']!= 'dont': 
    qtls = gwas.callQTLs()
    gwas.annotate(qtls)
if dictionary['locuszoom'] != 'dont': gwas.locuszoom(pd.read_csv(f'{pj}/results/qtls/finalqtl.csv')) 
if dictionary['gcorr'] != 'dont': gwas.genetic_correlation_matrix()
if dictionary['manhattanplot'] != 'dont': gwas.manhattanplot(display = False)
if dictionary['porcupineplot'] != 'dont': gwas.porcupineplot(pd.read_csv(f'{pj}/results/qtls/finalqtl.csv'), display = False)
if dictionary['phewas']!= 'dont':gwas.phewas(pd.read_csv(f'{pj}/results/qtls/finalqtl.csv').set_index('SNP'), annotate=True, pval_threshold = 1e-4, nreturn = 1, r2_threshold = .4) 
if dictionary['eqtl']!= 'dont':gwas.eQTL(pd.read_csv(f'{pj}/results/qtls/finalqtl.csv').set_index('SNP'), annotate= True)
if dictionary['sqtl']!= 'dont':gwas.sQTL(pd.read_csv(f'{pj}/results/qtls/finalqtl.csv').set_index('SNP'))
if dictionary['store']!= 'dont':gwas.store(researcher='tsanches',round_version='10.0.0', gwas_version='0.1.0',  remove_folders=False)
if dictionary['publish']!= 'dont':gwas.copy_results()
gwas.print_watermark()
