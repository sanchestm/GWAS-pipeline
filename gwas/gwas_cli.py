#!/usr/bin/env python
import numpy as np
import re
from collections import defaultdict
import pandas as pd
import sys

try:
    from . import core as gg
except ImportError as e:
    print('Failed to import gwas.gwas; trying local import...')
    try:
        from gwas import core as gg
    except ImportError:
        raise ImportError("Could not import gwas.gwasgwas.gwas. Make sure your package is installed or the structure is correct.") from e

runall = 'regressout subset grm h2 gwas db qtl gcorr phewas eqtl sqtl goea locuszoom h2fig report store publish porcupineplot'.replace(' ', '|||')
run2phewas = 'regressout subset grm h2 gwas db qtl gcorr eqtl sqtl goea locuszoom h2fig'.replace(' ', '|||')
def typeconverter(s):
    s= str(s)
    if s.lower() in ['1', 'true']: return 1
    if s.lower() in ['0', 'false']: return 0
    try: return int(s)
    except: pass
    if s[-2:] == '()':
        try: return eval(np.random.choice( re.split(r'(\s|\,|\]|\[)', s)))
        except: pass
    try: return float(s)
    except: return s
        
def kw(d, prefix):
    if prefix[-1] != '_': prefix += '_'
    return {k.replace(prefix, ''):typeconverter(v) for k,v in d.items() if (k[:len(prefix)] == prefix)}

def main():
    
    allargs = '|||'.join(sys.argv[1:]).replace('runall', runall).replace('run2phewas', run2phewas).split('|||')
    dictionary = defaultdict(lambda: 0, {k.replace('-', ''):v for k,v in [(x + '=1' if '=' not in x else x).split('=') for x in allargs] })
   
    
    path = dictionary['path'].rstrip('/') + '/' if (dictionary['path'] ) else ''
    
    pj = dictionary['project'].rstrip('/')  if (dictionary['project'] ) else 'test'
    
    if not dictionary['genotypes']:
        dictionary['genotypes'] = '/tscc/projects/ps-palmer/gwas/databases/rounds/r10.5.2'
    
    if not dictionary['gwas_version']: dictionary['gwas_version'] = gg.__version__
        
    if not dictionary['threshold']: dictionary['threshold'] = 5.39
    if dictionary['threshold'] != 'auto':  dictionary['threshold']= float(dictionary['threshold'])
    
    if not dictionary['threshold05']: dictionary['threshold05'] = 5.64
    dictionary['threshold05'] = float(dictionary['threshold05'])
        
    if not dictionary['genome_accession']: dictionary['genome_accession'] = 'GCF_015227675.2'
        
    if not dictionary['researcher']: dictionary['researcher'] = 'tsanches'
    if not dictionary['round']: dictionary['round'] = '10.5.2'
            
    if not dictionary['phewas_path']: dictionary['phewas_path'] = 'phewasdb.parquet.gz'
        
    if not dictionary['regressout']:
        df = pd.read_csv(f'{path}{pj}/processed_data_ready.csv',
                         dtype={'rfid': str}).drop_duplicates(subset='rfid')
        if not dictionary['traits']:
            traits_ = df.columns[df.columns.str.startswith('regressedlr_')]
        elif 'prefix_' in dictionary['traits']: 
            pref = dictionary['traits'].replace('prefix_', '')
            traits_ = df.columns[df.columns.str.startswith(f'regressedlr_{pref}')]
        else: traits_ = dictionary['traits'].split(',')
        try: traits_d = gg.get_trait_descriptions_f(pd.read_csv(f'{path}{pj}/data_dict_{pj}.csv'), traits_)
        except: traits_d = ['UNK' for x in range(len(traits_))]
    else:
        rawdata = dictionary['regressout'] if (len(dictionary['regressout']) > 1) else f'{path}{pj}/raw_data.csv'
        df = pd.read_csv(rawdata, dtype = {'rfid': str}).drop_duplicates(subset = 'rfid') 
        traits_, traits_d = [], []
    
    for k,v in dictionary.items(): gg.printwithlog(f'--{k} : {v}')
    gwas = gg.gwas_pipe(path = f'{path}{pj}/',
                 all_genotypes = dictionary['genotypes'], #'round9_1.vcf.gz',
                 data = df,
                 project_name = pj.split('/')[-1],
                 traits = traits_,
                 genome_accession = dictionary['genome_accession'],
                 founderfile = dictionary['founder_genotypes'],
                 phewas_db = dictionary['phewas_path'],
                 trait_descriptions= traits_d,
                 threshold = dictionary['threshold'],
                 threshold05 = dictionary['threshold05'],
                 threads = dictionary['threads'])
    gg.printwithlog(path)
    gg.printwithlog(pj)
    if dictionary['clear_directories'] and not dictionary['skip_already_present_gwas']: gwas.clear_directories()
    if dictionary['impute']: 
        gbcols =  [] if not dictionary['groupby'] else dictionary['groupby'].split(',')
        gwas.impute_traits(crosstrait_imputation = dictionary['crosstrait_imputation'],  groupby_columns=dictionary['groupby'].split(','))
    if dictionary['add_sex_specific_traits']: 
        gwas.add_sex_specific_traits(save = True)
    if dictionary['regressout'] : 
        kws = kw(dictionary, 'regressout_')
        if not dictionary['timeseries']:  
            if not dictionary['groupby']: gwas.regressout(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'), **kws)
            else: gwas.regressout_groupby(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'), groupby_columns=dictionary['groupby'].split(','),**kws)
        else:  
            if not dictionary['groupby']: gwas.regressout_timeseries(data_dictionary=pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'),**kws)
            else: gwas.regressout_timeseries(data_dictionary= pd.read_csv(f'{gwas.path}data_dict_{pj}.csv'), groupby_columns=dictionary['groupby'].split(','), **kws)
    
    if dictionary['latent_space']: gwas.add_latent_spaces()
    if dictionary['subset'] :  ###essential
        kws = kw(dictionary, 'subset_')
        if dictionary['subset_make_figures'] : gwas.SubsetAndFilter(makefigures = True, **kws)
        else: gwas.SubsetAndFilter(makefigures = False, **kws)
    if dictionary['grm']: 
        gwas.generateGRM(**kw(dictionary, 'grm_')) ###essential
    if dictionary['h2'] :  gwas.snpHeritability() ###essential
    if dictionary['BLUP'] : gwas.BLUP()
    if dictionary['BLUP_predict']: gwas.BLUP_predict(dictionary['BLUP_predict']);
    if dictionary['gwas']: gwas.fastGWAS(skip_already_present=dictionary['skip_already_present_gwas']) ###essential
    if dictionary['db'] and not dictionary['nodb']: gwas.addGWASresultsToDb(researcher=dictionary['researcher'],
                                                 round_version=dictionary['round'], 
                                                 gwas_version=dictionary['gwas_version'])
    if dictionary['qtl']: ###essential
        qtl_add_founder = True if (dictionary['founder_genotypes'] not in ['none', 'None', 0]) else False
        try: qtls = gwas.callQTLs( NonStrictSearchDir=False, add_founder_genotypes = qtl_add_founder)
        except: qtls = gwas.callQTLs( NonStrictSearchDir=True)
        #gwas.annotate(qtls)
        gwas.effectsize() 
    if dictionary['effect']: gwas.effectsize() 
    if dictionary['gcorr']: ###essential
        kws = kw(dictionary, 'gcorr_')
        gwas.genetic_correlation_matrix(**kws)
        gwas.make_heritability_figure()
    if dictionary['manhattanplot'] or dictionary['porcupineplot']: gwas.porcupineplot() ###essential
    if dictionary['phewas']:gwas.phewas(annotate=True, pval_threshold = 1e-4, nreturn = 1, r2_thresh = .65)  ###essential
    if dictionary['eqtl']:gwas.eQTL(annotate= True) ###essential
    if dictionary['sqtl']:gwas.sQTL(annotate= True) ###essential
    if dictionary['goea']:gwas.GeneEnrichment() ###essential
    if dictionary['locuszoom']: gwas.locuszoom(**kw(dictionary, 'locuszoom_'))  ###essential
    if dictionary['h2fig']: gwas.make_heritability_figure() 
    if dictionary['report']:
        kws = kw(dictionary, 'report_')
        gwas.report(round_version=dictionary['round'], gwas_version=dictionary['gwas_version'], **kws)
        gwas.copy_results() ###essential
    if dictionary['store']: gwas.store(researcher=dictionary['researcher'],
                                      round_version=dictionary['round'], 
                                      gwas_version=dictionary['gwas_version'],  
                                      remove_folders=False) ###essential
    try: 
        if dictionary['publish']: gwas.copy_results() ###essential
    except: 
        print('setting up the minio is necessary')
    if dictionary['phewas_fig']: gwas.make_phewas_figs()
    gwas.print_watermark()

if __name__ == "__main__":
    main()
