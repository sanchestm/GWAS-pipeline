import pandas as pd
from ipywidgets import interact, interact_manual, widgets
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
from holoviews import opts
import holoviews as hv
import hvplot.pandas
import hvplot.dask  
from functools import reduce, wraps
pn.extension('plotly')
pn.extension()

class interactive_QC:
    def __init__(self, raw_data: pd.DataFrame, data_dictionary: pd.DataFrame):
        if type(raw_data) == str: raw_data = pd.read_csv(raw_data, dtype={'rfid': str})
        if type(data_dictionary) == str: data_dictionary = pd.read_csv(data_dictionary)
        self.uncurated_data = raw_data.copy()
        self.uncurated_data.to_csv(f'uncurated_data_N{self.uncurated_data.shape[0]}_{datetime.today().strftime("%Y%m%d")}.csv')
        self.dfog = raw_data.copy()
        self.dd = data_dictionary.copy()
        self.dd.loc[self.dd.measure =="rfid", 'trait_covariate']='metadata'
        self.align_w_cols = lambda l: list(set(l) & set(self.dfog.columns))
        self.traits = self.align_w_cols(self.dd.query('trait_covariate == "trait"').measure.to_list())
        self.covs = self.align_w_cols(self.dd[self.dd['trait_covariate'].fillna('').str.contains('covariate')].measure.to_list())
        self.covs_cat = self.align_w_cols(self.dd[self.dd['trait_covariate'].fillna('').str.contains('covariate_categorical')].measure.to_list())
        self.covs_cont = self.align_w_cols(self.dd[self.dd['trait_covariate'].fillna('').str.contains('covariate_continuous')].measure.to_list())
        self.cols_metadata = self.align_w_cols(self.dd[self.dd['trait_covariate'].fillna('').str.contains('metadata')].measure.to_list())
        self.dfog[self.traits] = self.dfog[self.traits].apply(pd.to_numeric, errors = 'coerce')
        self.dfog[self.covs_cat] = self.dfog[self.covs_cat].astype(str).fillna('UNK')
        self.dfog[self.covs_cont] = self.dfog[self.covs_cont].apply(pd.to_numeric, errors = 'coerce').fillna(-9999)
        self.dfog[self.cols_metadata] = self.dfog[self.cols_metadata].astype(str)
        self.dffinal = ''
    
        self.value_dict = pd.DataFrame(columns = ['wid_range', 'wid_exclude_rfid', 'wid_trait', 'wid_extra'])
        for t in tqdm(self.traits):
            minv, maxv = self.dfog[t].astype(float).agg(['min', 'max']).to_list()
            wid = widgets.FloatRangeSlider(value=[minv, maxv],min=minv,max=maxv,step=0.0001,disabled=False,
                continuous_update=False, orientation='horizontal',readout=True,layout = widgets.Layout(width='1000px'),
                readout_format='.2f', description = 'min_max_threshold:')
            tags2 = widgets.TagsInput(value=[], allowed_tags=list(self.dfog.rfid),
                allow_duplicates=False, description = 'rm rfids:', layout = widgets.Layout(width='1000px'),)
            self.value_dict.loc[t] = [wid, tags2, widgets.Text(value = t, disable = True), widgets.Text(value = '', 
                                                                                                   description = 'extra code',  
                                                                                           layout = widgets.Layout(width='1000px'))]
        self.value_dict_rat = pd.DataFrame(index = self.dfog.reset_index().rfid.to_list())
        self.value_dict_rat['wid_text'] =  [widgets.Text(value = '', description = f'extra code rat {i}',  layout = widgets.Layout(width='1000px')) \
                                             for i in self.dfog.reset_index().rfid]
        self.outdfname = f'raw_data_curated_n{self.dfog.shape[0]}_{datetime.today().strftime("%Y%m%d")}.csv'

    def quick_view(self):
        from gwas import statsReport
        if type(self.dffinal) == str: sr = statsReport.stat_check(self.dfog)
        else: sr = statsReport.stat_check(self.dffinal)
        tabs = pn.Tabs(tabs_location='left') 
        for group, dd1 in self.dd.query('trait_covariate == "trait"').groupby('covariates'):
            traits = dd1.measure.to_list()
            covs = dd1.covariates.iloc[0].split(',')
            for subtraits in [traits[15*i:15*i+15] for i in range(len(traits)//15+1)]:
                anova_res = sr.do_anova(targets = subtraits, covariates=covs).loc[:, (slice(None), 'qvalue')].set_axis(covs, axis = 1) #
                pane = sr.plot_var_distribution(subtraits, covs)
                tabs.append((f"{len(covs)}",pn.Column(anova_res,pane)))    #traits:{','.join(traits)}\ncovariates:{','.join(covs)}
        return tabs

    def _f(self, trait, ranger, remove_rfids, extra):
        todo = '|'.join(sorted(set(self.traits) - set(self.dffinal.columns) - set([trait])))
        if todo: print(f'traits still to do...\n{todo}')
        else: print('curation was performed for all traits')
        df = self.dfog.copy().reset_index()#.set_index('rfid')#  if not len(self.dffinal) else self.dffinal.copy().reset_index()
        df.loc[df.rfid.isin(remove_rfids) | (df[trait] < ranger[0]) | (df[trait] > ranger[1]), trait] = np.nan
        if len(extra):
            for ex_ in extra.split(';'):
                if '->' not in ex_: 
                    df.loc[~(eval(ex_)), trait] = np.nan
                else:
                    a_,b_ = ex_.split('->')[:2]
                    df.loc[(eval(a_)), trait] = eval(b_)
            
        describer = df[[trait]].describe(percentiles = [0.01, 0.05, 0.1,.9,.95, .99]).T
        ppfr =  abs(norm.ppf(np.array([0.00001, 0.0001, 0.001, .01, .05][::-1])/2))[:, None]*np.array([-1, 1])*describer['std'].iloc[0] + describer['mean'].iloc[0]
        describer.loc[trait, [f'z{x}%' for x in 100*np.array([0.00001, 0.0001, 0.001, .01, .05][::-1])]] = [' – '.join(x) for x in ppfr.round(2).astype(str)]
        display(describer)
        covariates = set(self.align_w_cols(self.dd.set_index('measure').loc[trait, 'covariates'].split(','))) 
        tabs = pn.Tabs(tabs_location='left')
        if covariates != {'passthrough'}:
            fig = px.histogram(df, x = trait, color = 'sex')
            fig.update_layout(template='simple_white',width = 800, height = 500, coloraxis_colorbar_x=1.05,
                                          coloraxis_colorbar_y = .3,coloraxis_colorbar_len = .8,hovermode='x unified')
            fig.add_vline(x=ranger, line_width=3, line_dash="dash", line_color="red")
            for rangeri in ranger: fig.add_vline(x=rangeri, line_width=3, line_dash="dash", line_color="red")
            tabs.append(('histogram', pn.pane.Plotly(fig)))
            #display(fig)
            for cv in covariates:
                cvtype = self.dd.set_index('measure').loc[cv, 'trait_covariate']
                if cvtype !='covariate_continuous':
                    dffigs = df.sort_values(cv)
                    fig  = px.strip(dffigs,y=trait, x=cv, hover_data=['rfid']+ list(covariates))
                    fig.data[0]['marker']['color'] = 'black'
                    fig.add_violin(x=dffigs[cv], y=dffigs[trait], hoverinfo=[],box_visible=True, line_color='black', hoveron='points',
                                                   meanline_visible=True, fillcolor='gray', opacity=0.4,hovertext=[] )
                    fig.update_layout(legend_visible = False,hovermode='y unified')
                    plotname = f'{cv}:violin'
                else: 
                    plotname = f'{cv}:scatter'
                    dfsset = df.dropna(subset=trait)
                    fig = px.density_contour(dfsset, y=trait, x=cv, trendline='ols')
                    for i in fig.data: i['line']['color'] = 'gray'
                    fig.add_scatter(x = dfsset[cv], y = dfsset[trait], mode='markers', marker= dict(line_width=1, size=7, color = 'black', line_color = 'white'),
                                    text=dfsset[['rfid']+ list(covariates)].apply(lambda x: '<br>'.join(map(lambda m: ":".join(m), zip(x.index, x.astype(str).values))), axis = 1), ) 
                for rangeri in ranger: fig.add_hline(y=rangeri, line_width=3, line_dash="dash", line_color="red")
                fig.update_layout(template='simple_white',width = 800, height = 600, coloraxis_colorbar_x=1.05, legend_visible = False, 
                                          coloraxis_colorbar_y = .3,coloraxis_colorbar_len = .8) #xaxis_range=[dfsset.query().cv.replace(-9999), xmax]
                tabs.append((plotname, pn.pane.Plotly(fig)))
            display(tabs)
            self.dffinal.loc[:, trait] = df.loc[:, trait].values
            display(self.dffinal.drop(self.covs, axis = 1).set_index('rfid'))
        if not todo: 
            print('saving boundaries to "QC_set_boundaries.csv"')
            print(f'saving data to "{self.outdfname}"')
            self.value_dict.applymap(lambda x: x.value).to_csv('QC_set_boundaries.csv')
            self.dffinal.set_index('rfid').sort_index().to_csv(self.outdfname)

    def QC(self):
        if not len(self.dffinal): self.dffinal = self.dfog[['rfid']+self.covs].copy() 
        if 'rfid' not in self.dffinal.columns: self.dffinal = self.dffinal.reset_index()
        interact(lambda single_trait : interact_manual(self._f, 
                        trait = self.value_dict.loc[single_trait,'wid_trait'], 
                        ranger = self.value_dict.loc[single_trait,'wid_range'], 
                        remove_rfids= self.value_dict.loc[single_trait,'wid_exclude_rfid'], 
                        extra= self.value_dict.loc[single_trait,'wid_extra']),
                 single_trait =  widgets.Dropdown(options=sorted(self.traits)))

    def _fi(self, rfid, extra):
        df = self.dfog.reset_index().set_index('rfid') #if not len(self.dffinal) else self.dffinal.reset_index().set_index('rfid')
        rat = df.loc[rfid]
        #self.value_dict_rat.loc[rfid, 'wid_text'].value = extra
        if 'rfid' in self.dffinal.columns: self.dffinal = self.dffinal.set_index('rfid')
        if len(extra):
            for ex_ in extra.replace('=', '->').replace(' ','').split(';'):
                if '->' not in ex_: 
                    if ex_ not in df.columns: ex_ = df.columns[df.columns.str.startswith(ex_)].to_list()
                    df.loc[rat.name, ex_] = np.nan
                    self.dffinal.loc[rat.name, ex_] = np.nan 
                else:
                    a_,b_ = ex_.split('->')[:2]
                    if a_ not in df.columns: a_ = df.columns[df.columns.str.startswith(a_)].to_list()
                    df.loc[rat.name, a_] = eval(b_)
                    self.dffinal.loc[rat.name, a_] = eval(b_)
    
        from sklearn.preprocessing import LabelEncoder
        df[self.covs_cat] = df[self.covs_cat].astype(str)
        describedall = df.describe(include = 'all', percentiles = [.05, .1, .9, .95])
        describedall.index = describedall.index + '_all'
        describecoh = df.query('cohort == @rat.cohort').describe(include = 'all')
        describecoh.index = describecoh.index + f"_c{rat['cohort']}"
        df = pd.concat([df, describedall, describecoh])
        df = df[~df.index.to_series().str.contains('count|std|50%')]
        df[self.covs_cat] = df[self.covs_cat].astype(str)
        for i in self.covs_cat: df[f'LE{i}'] = LabelEncoder().fit_transform(df[i])
    
        ncols = 2
        categories = set(map(lambda x: x.split('_')[0] + '_', self.traits))
        fig = make_subplots(rows=(len(categories)+2)//ncols +1, cols=ncols)
        hm1 = [rat.name] + df.index[df.index.to_series().str.contains(f'_all|_c{rat.cohort}')].to_list() 
        hm1df =  df.loc[hm1,self.covs_cat].replace('nan', np.nan).dropna(how = 'all').sort_index(axis = 1)
        fig.add_trace(go.Heatmap(z = df.loc[hm1df.index, 'LE'+hm1df.columns][::-1].fillna(-10), texttemplate="%{text}", y = hm1df.index[::-1], x = hm1df.columns,
                   textfont={"size":10},text = hm1df[::-1], colorscale='Jet'), row = 1, col = 1 )
        #display(df.loc[hm1df.index, 'LE'+hm1df.columns][::-1].fillna(-10))
        
        hm1df =  df.loc[hm1,self.covs_cont].dropna(how = 'all').sort_index(axis = 1)
        fig.add_trace(go.Heatmap(z = hm1df[::-1], texttemplate="%{text}", y = hm1df.index[::-1], x = hm1df.columns,
                   textfont={"size":10},text = hm1df[::-1].round(2), colorscale='RdBu'), row = 1, col = 2 )
        
        cnt = 2
        for cattemp in sorted(categories):
            hm1df =  df.loc[hm1,df.columns.str.startswith(cattemp)].dropna(how = 'all').sort_index(axis = 1)
            if rat.name in hm1df.index:
                fig.add_trace(go.Heatmap(z = hm1df[::-1], texttemplate="%{text}", y = hm1df.index[::-1], x = hm1df.columns,
                           textfont={"size":10 if len(hm1df.columns)< 10 else 7},text = hm1df[::-1].round(2) if len(hm1df.columns)< 15 else hm1df.applymap(lambda x: ''),
                                         colorscale='RdBu'), row = cnt//ncols +1, col = cnt%ncols +1)
            else:
                fig.add_trace(go.Heatmap(z = [[0]], textfont={"size":30}, texttemplate="%{text}",
                                         text = [[f'No {cattemp} values for <br> rat {rat.name}']]), 
                              row = cnt//ncols +1, col = cnt%ncols +1)
            cnt += 1
        
        fig.update_layout(width = 1200, height = 400*(len(categories)+2)//ncols +1, title = f'Rat: {rat.name} sex: {rat.sex} cohort: {rat.cohort}', template = 'simple_white')
        fig.update_traces(showscale=False)
        display(fig)
        self.dffinal = self.dffinal.reset_index()
    
    def QCiid(self):
        if not len(self.dffinal): self.dffinal = self.dfog[['rfid']+self.covs].copy() 
        interact(lambda ratid : interact_manual(self._fi,
                                rfid = ratid,
                                extra= self.value_dict_rat.loc[ratid,'wid_text']),
                         ratid =  widgets.Dropdown(options=sorted(self.dfog.reset_index()['rfid'].to_list())))
        

    def save_curated(self):
        print('saving boundaries to "QC_set_boundaries.csv"')
        print(f'saving data to "{self.outdfname}"')
        self.value_dict.applymap(lambda x: x.value).to_csv('QC_set_boundaries.csv')
        self.value_dict_rat.applymap(lambda x: x.value).to_csv('QC_set_boundaries_rat.csv')
        self.dffinal.set_index('rfid').sort_index().to_csv(self.outdfname)

    def get_df(self):
        return self.dffinal

    def get_boundaries(self):
        return self.value_dict.applymap(lambda x: x.value).sort_index()


def hv2strmlit(file,width = None, height = 500 ,**kws):
    import streamlit as st
    if type(file)!=str:
        html_buffer = StringIO()
        hv.save(file,html_buffer)
        t = html_buffer.getvalue()
    else:
        with open(file, 'r', encoding='utf-8') as f: 
            t = f.read()
    return  st.components.v1.html(t, width=width, height=height, scrolling=True, **kws)


def fancy_display(df: pd.DataFrame, download_name: str = 'default.csv', max_width = 1400, max_height = 600) -> pn.widgets.Tabulator:
    pn.extension('tabulator')
    df = df.drop(['index'] + [f'level_{x}' for x in range(100)], errors = 'ignore', axis = 1)
    numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
    try:df[numeric_cols] = df[numeric_cols].map(round, ndigits=3)
    except: df[numeric_cols] = df[numeric_cols].applymap(round, ndigits=3)
    d = {x : {'type': 'number', 'func': '>=', 'placeholder': 'Enter minimum'} for x in numeric_cols} | \
        {x : {'type': 'input', 'func': 'like', 'placeholder': 'Similarity'} for x in df.columns[~df.columns.isin(numeric_cols)]}
    download_table = pn.widgets.Tabulator(df,pagination='local',page_size= 20, header_filters=d, layout = 'fit_data_fill', max_width = max_width, max_height = max_height)
    filename, button = download_table.download_menu(text_kwargs={'name': 'Enter filename', 'value': download_name},button_kwargs={'name': 'Download table'})
    return pn.Column(pn.Row(filename, button), download_table)

import html
def pn_Iframe(file,width = 600, height = 600 ,return_html_iframe=False,**kws):
    if type(file)!=str:
        html_buffer = StringIO()
        hv.save(file,html_buffer)
        t = html.escape(html_buffer.getvalue())
    else:
        with open(file) as f: t = html.escape(f.read())
    iframe_code = f'<iframe srcdoc="{t}" style="height:100%; width:100%" frameborder="0"></iframe>'
    if return_html_iframe: return iframe_code
    return pn.pane.HTML(iframe_code, max_width = 1000, max_height = 1000, height = height, width=width, **kws)

def regex_plt(df, rg, max_cols = 10, full = True):
    if full:
        seq = '|'.join(rg) if not isinstance(rg, str) else rg
        table = fancy_display(df.loc[:, ~df.columns.str.contains(seq)], max_height=400, max_width=1000)
        return pn.Column(table,  regex_plt(df,rg, max_cols = max_cols, full = False))
    if not isinstance(rg, str):
        fig = reduce(lambda x,y: x+y, [regex_plt(df, x, full = False).opts(shared_axes = False) for x in rg]).opts(shared_axes = False).cols(max_cols)
        return fig
    sset = df.filter(regex = rg).T
    return (sset.hvplot(kind='line', rot= 45, grid =True) \
           * sset.hvplot(kind='scatter', marker='o', size=50, rot= 45,line_width = 1, line_color='black', alpha = .7 ,  legend = False, title = rg, grid =True))\
           .opts(frame_width = 300, frame_height = 300,show_legend = False, title = rg)

def megaHeatmap(df, sets, cpallete = ['Greys','OrRd', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'BuGn', 'YlGn'], scale_column=True, row_clusters = False, **bokeh_opts):
    figstack = []
    figstackna = []
    bokeh_opts = dict(width = 1450, height = 2000)|bokeh_opts
    df = df.copy()
    if row_clusters:
        from scipy.cluster.hierarchy import ward, dendrogram, leaves_list, linkage
        from scipy.spatial import distance
        dfn = df.filter(regex = '|'.join(sets)).select_dtypes(include = 'number')
        if dfn.shape[1] and dfn.shape[0]:     
            dist_array = distance.pdist(dfn.fillna(dfn.mean()).fillna(1))
            if dist_array.size > 0:
                hieg = linkage(dist_array + 1e-5)
                df = df.iloc[leaves_list(hieg)]
            else:
                print("Warning: Empty distance matrix after pdist; skipping clustering.")
        elif dfn.shape[0] <= 1:
            print("Warning: Skipping clustering because there is only one row.")
        elif dfn.shape[1] == 0:
            print("Warning: No numeric columns matched the provided regex in this group.")
    for  i, cmap in zip(sets,cpallete):
        tempdf = df.filter(regex = i).select_dtypes(include = 'number')
        if not scale_column or not tempdf.shape[1]:
            fig = tempdf.hvplot.heatmap(rot = 90, cmap = cmap, colorbar=False, shared_axes = False)
            labels = hv.Labels(fig).opts( text_font_size="5pt", text_color="black" )
            fig = fig*labels
        else: 
            fig_lis = (tempdf[[tcol]].hvplot.heatmap(rot = 90, cmap = cmap, colorbar=False, shared_axes = False)\
                         for tcol in tempdf.columns)
            fig_lis  = (fig*hv.Labels(fig).opts( text_font_size="5pt", text_color="black" ) for fig in fig_lis)
            fig = reduce(lambda x,y: x*y, fig_lis  )
        figstack += [fig.opts(xaxis = 'top')]
        figstackna += [df.filter(regex = i).isna().hvplot.heatmap(cmap = 'greys', alpha = 1, rot = 90, colorbar=False, xaxis = 'top')]
    
    bigfig = reduce(lambda x,y: x*y, figstack).opts( **bokeh_opts )#"zoom_box",active_tools=['hover']
    bigfigna = reduce(lambda x,y: x*y, figstackna).opts( **bokeh_opts)#active_tools=['hover']
    return bigfig, bigfigna

def megaHeatmapGroupby(df, sets, gb_column, scale_column = True,  **bokeh_opts):
    figs = df.groupby(gb_column).progress_apply(lambda x: megaHeatmap(x, sets, scale_column=scale_column))
    tabs = pn.Tabs()
    tabsna = pn.Tabs()
    for i, v in figs.items():
        tabs.append((i,  pn.pane.HoloViews(v[0].opts(**bokeh_opts), linked_axes=False)))
        tabsna.append((i,pn.pane.HoloViews(v[1].opts(**bokeh_opts), linked_axes=False)))
    return tabs, tabsna
