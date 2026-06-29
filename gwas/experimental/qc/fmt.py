import pandas as pd

def generate_datadic(rawdata, trait_prefix = ['pr', 'sha', 'lga', 'shock'], main_cov = 'sex,cohort,weight_surgery,box,coatcolor', project_name = '', save = False):
    dd_new = rawdata.describe(include = 'all').T.reset_index(names= ['measure'])
    dd_new['trait_covariate'] = dd_new['unique'].apply(lambda x: 'covariate_categorical' if not np.isnan(x) else 'covariate_continuous')
    dd_new.loc[dd_new.measure.str.contains('^'+'|^'.join(trait_prefix), regex = True), ['trait_covariate', 'covariates']] = ['trait', main_cov]
    dd_new.loc[dd_new.measure.str.endswith('_age') , ['trait_covariate', 'covariates']] = ['covariate_continuous', '']
    dd_new['description'] = dd_new['measure']
    for pref in trait_prefix:
        addcov = ','.join(dd_new.loc[dd_new.measure.str.endswith('_age') &  dd_new.measure.str.startswith(pref)].description)
        if len(addcov): 
            dd_new.loc[dd_new.measure.str.startswith(pref), 'covariates'] = dd_new.loc[dd_new.measure.str.startswith(pref), 'covariates'] + ',' + addcov
    dd_new.loc[dd_new.measure.isin(['rfid', 'labanimalid']), 'trait_covariate'] = 'metadata'
    dd_new = dd_new.loc[~dd_new.measure.str.startswith('Unnamed: ')]
    def remove_outofboundages(s, trait):
       try:
           min, max = pd.Series(list(map(int , re.findall('\d{2}', trait)))).agg(['min', 'max'])
           #print(min, max)
           aa =  "|".join( [str(x).zfill(2) for x in range(min, max+1)])
       except: aa = '-9999999'
       #print(aa)
       return ','.join([x for x in s.split(',') if re.findall(f'({aa}).*(age)', x) or ('age' not in x)])
    dd_new.loc[dd_new.trait_covariate == 'trait', 'covariates'] = dd_new.loc[dd_new.trait_covariate == 'trait']\
          .apply(lambda row: remove_outofboundages(row.covariates,row.measure), axis = 1)
    if save:
        dd_new.to_csv(f'data_dict_{project_name}.csv')
    return dd_new