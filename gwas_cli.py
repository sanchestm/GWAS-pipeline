#!/usr/bin/env python 
from gwas_class_auto import *
import sys
import pandas as pd

dictionary = {k.replace('-', ''):v for k,v in [x.split('=') for x in sys.argv[1:]] }
dictionary = defaultdict(lambda: 1, dictionary)
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
if dictionary['subset']!= '0': gwas.subsetSamplesFromAllGenotypes(sourceFormat = 'plink')
if dictionary['grm']!= '0':gwas.generateGRM()
if dictionary['h2']!= '0': gwas.snpHeritability()
if dictionary['BLUP'] != '0': gwas.BLUP()
if dictionary['BLUP_predict'] != 1: gwas.BLUP_predict(dictionary['BLUP_predict']);
if dictionary['gwas']!= '0': gwas.GWAS()
if dictionary['db']!= '0': gwas.addGWASresultsToDb(researcher='tsanches', round_version='10.0.0', gwas_version='0.1.0')
if dictionary['qtl']!= '0': 
    qtls = gwas.callQTLs()
    gwas.annotate(qtls)
if dictionary['phewas']!= '0':gwas.phewas(qtls, annotate=True, pval_threshold = 1e-4, nreturn = 1, r2_threshold = .4) 
if dictionary['eqtl']!= '0':gwas.eQTL(qtls, annotate= True)
if dictionary['store']!= '0':gwas.store(researcher='tsanches',round_version='10.0.0', gwas_version='0.1.0',  remove_folders=False)
if dictionary['publish']!= '0':gwas.copy_results()
gwas.print_watermark()


#### example call: python gwas_cli.py project=u01_tom_jhou threads=30 phewas=0 store=0 publish=0 BLUP=0 eqtl=0 
