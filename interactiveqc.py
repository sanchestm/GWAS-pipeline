import pandas as pd
from ipywidgets import interact, interact_manual, widgets
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class interactive_QC:
    def __init__(self, raw_data: pd.DataFrame, data_dictionary: pd.DataFrame):
        if type(raw_data) == str: raw_data = pd.read_csv(raw_data, dtype={'rfid': str})
        if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
        self.uncurated_data = raw_data.copy()
        self.uncurated_data.to_csv(f'uncurated_data_N{self.uncurated_data.shape[0]}_{datetime.today().strftime("%Y%m%d")}.csv')
        self.dfog = raw_data.copy()
        self.dd = data_dictionary.copy()

        self.traits = self.dd.query('trait_covariate == "trait"').measure.to_list()
        self.covs = self.dd[self.dd['trait_covariate'].str.contains('covariate')].measure.to_list()

        self.dfog[self.traits] = self.dfog[self.traits].applymap(lambda x: (str(x).replace('#DIV/0!', 'nan').replace('#VALUE!', 'nan')  
                                                .replace('n/a ','nan' ).replace('#REF!','nan' ))).astype(float)

        self.dffinal = ''
    
        self.value_dict = pd.DataFrame(columns = ['wid_range', 'wid_exclude_rfid', 'wid_trait', 'wid_extra'])
        for t in self.traits:
            minv, maxv = self.dfog[t].astype(float).agg(['min', 'max']).to_list()
            wid = widgets.FloatRangeSlider(value=[minv, maxv],min=minv,max=maxv,step=0.0001,disabled=False,
                continuous_update=False, orientation='horizontal',readout=True,layout = widgets.Layout(width='1000px'),
                readout_format='.2f', description = 'min_max_threshold:')
            tags2 = widgets.TagsInput(value=[], allowed_tags=list(self.dfog.rfid),
                allow_duplicates=False, description = 'rm rfids:', layout = widgets.Layout(width='1000px'),)
            self.value_dict.loc[t] = [wid, tags2, widgets.Text(value = t, disable = True), widgets.Text(value = '', 
                                                                                                   description = 'extra code',  
                                                                                           layout = widgets.Layout(width='1000px'))]
        self.outdfname = f'raw_data_curated_n{self.dfog.shape[0]}_{datetime.today().strftime("%Y%m%d")}.csv'

    def quick_view(self):
        if type(self.dffinal) == str: sr = statsReport.stat_check(self.dfog)
        else: sr = statsReport.stat_check(self.dffinal)
            
        for group, dd1 in self.dd.query('trait_covariate == "trait"').groupby('covariates'):
            traits = dd1.measure.to_list()
            covs = dd1.covariates.iloc[0].split(',')
            display(sr.do_anova(targets = traits, covariates=covs).loc[:, (slice(None), 'qvalue')].set_axis(covs, axis = 1)) #
            for subtraits in [traits[15*i:15*i+15] for i in range(len(traits)//15+1)]:
                sr.plot_var_distribution(subtraits, covs)

    def _f(self, trait, ranger, remove_rfids, extra):
        todo = '|'.join(sorted(set(self.traits) - set(self.dffinal.columns) - set([trait])))
        if todo: print(f'traits still to do...\n{todo}')
        else: print('curation was performed for all traits')
        df = self.dfog.copy()
        df.loc[df.rfid.isin(remove_rfids) | (df[trait] < ranger[0]) | (df[trait] > ranger[1]), trait] = np.nan
        if len(extra):
            for ex_ in extra.split(';'):
                if '->' not in ex_: 
                    df.loc[~(eval(ex_)), trait] = np.nan
                else:
                    a_,b_ = ex_.split('->')[:2]
                    df.loc[(eval(a_)), trait] = eval(b_)
            
        describer = df[[trait]].describe(percentiles = [0.01, 0.05, 0.1,.9,.95, .99]).T
        ppfr =  abs(norm.ppf(np.array([0.00001, 0.0001, 0.001, .01, .05][::-1])/2))[:, None]*np.array([-1, 1])*describer['std'][0] + describer['mean'][0]
        describer.loc[trait, [f'z{x}%' for x in 100*np.array([0.00001, 0.0001, 0.001, .01, .05][::-1])]] = [' – '.join(x) for x in ppfr.round(2).astype(str)]
        display(describer)
        covariates = set(self.dd.set_index('measure').loc[trait, 'covariates'].split(',')) 
        fig = px.histogram(df, x = trait, color = 'sex')
        fig.update_layout(template='simple_white',width = 800, height = 500, coloraxis_colorbar_x=1.05,
                                      coloraxis_colorbar_y = .3,coloraxis_colorbar_len = .8,hovermode='x unified')
        fig.add_vline(x=ranger, line_width=3, line_dash="dash", line_color="red")
        for rangeri in ranger: fig.add_vline(x=rangeri, line_width=3, line_dash="dash", line_color="red")
        display(fig)
        for cv in covariates:
            cvtype = self.dd.set_index('measure').loc[cv, 'trait_covariate']
            if cvtype !='covariate_continuous':
                dffigs = df.sort_values(cv)
                fig  = px.strip(dffigs,y=trait, x=cv, hover_data=['rfid']+ list(covariates))
                fig.data[0]['marker']['color'] = 'black'
                fig.add_violin(x=dffigs[cv], y=dffigs[trait], hoverinfo=[],box_visible=True, line_color='black', hoveron='points',
                                               meanline_visible=True, fillcolor='gray', opacity=0.4,hovertext=[] )
                fig.update_layout(legend_visible = False,hovermode='y unified')
            else: 
                fig = px.density_contour(df, y=trait, x=cv, trendline='ols')
                for i in fig.data: i['line']['color'] = 'gray'
                fig.add_scatter(x = df[cv], y = df[trait], mode='markers', marker= dict(line_width=1, size=7, color = 'black', line_color = 'white'),
                                text=df[['rfid']+ list(covariates)].apply(lambda x: '<br>'.join(map(lambda m: ":".join(m), zip(x.index, x.astype(str).values))), axis = 1), ) 
            for rangeri in ranger: fig.add_hline(y=rangeri, line_width=3, line_dash="dash", line_color="red")
            fig.update_layout(template='simple_white',width = 800, height = 600, coloraxis_colorbar_x=1.05, legend_visible = False, 
                                      coloraxis_colorbar_y = .3,coloraxis_colorbar_len = .8)
            display(fig)
            #print('ye')
        self.dffinal.loc[:, trait] = df.loc[:, trait].values
        display(self.dffinal.drop(self.covs, axis = 1).set_index('rfid'))
        if not todo: 
            print('saving boundaries to "QC_set_boundaries.csv"')
            print(f'saving data to "{self.outdfname}"')
            self.value_dict.applymap(lambda x: x.value).to_csv('QC_set_boundaries.csv')
            self.dffinal.set_index('rfid').sort_index().to_csv(self.outdfname)
        #return self.dffinal
        #return df.loc[:, trait]

    def QC(self):
        self.dffinal = self.dfog[['rfid']+self.covs]
        interact(lambda single_trait : interact_manual(self._f, 
                        trait = self.value_dict.loc[single_trait,'wid_trait'], 
                        ranger = self.value_dict.loc[single_trait,'wid_range'], 
                        remove_rfids= self.value_dict.loc[single_trait,'wid_exclude_rfid'], 
                        extra= self.value_dict.loc[single_trait,'wid_extra']),
                 single_trait =  widgets.Dropdown(options=sorted(self.traits)))

    def save_curated(self):
        print('saving boundaries to "QC_set_boundaries.csv"')
        print(f'saving data to "{self.outdfname}"')
        self.value_dict.applymap(lambda x: x.value).to_csv('QC_set_boundaries.csv')
        self.dffinal.set_index('rfid').sort_index().to_csv(self.outdfname)

    def get_df(self):
        return self.dffinal

    def get_boundaries(self):
        return self.value_dict.applymap(lambda x: x.value).sort_index()

