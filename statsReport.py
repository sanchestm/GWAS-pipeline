import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests,fdrcorrection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

class stat_check:
    def __init__(self, dataframe: pd.DataFrame):
        self.categorical_columns = dataframe.select_dtypes(exclude=np.number).columns
        self.numerical_columns = dataframe.select_dtypes(include=np.number).columns
        self.df = dataframe
        
    def do_anova(self, targets: list = [] , covariates: list = []) -> pd.DataFrame:
        test_variables = list(set(self.numerical_columns) & set(targets if len(targets) > 0 else self.df.columns))
        out = []
        for cov in covariates:
            statsdf = pd.DataFrame([stats.f_oneway(*lis)\
                                   if len(lis:=[z for x,y in self.df.groupby(cov)[i] if len(z:=y.dropna()) > 0 ]) >1\
                                   else ['not enough groups', 1]\
                                   for i in test_variables] \
                                   ,columns = ['F stat', 'pval'], index = test_variables)
            
            statsdf['qvalue'] = -np.log10(fdrcorrection(statsdf['pval'])[1])
            statsdf['pval'] = statsdf['pval'].apply(lambda x: -np.log10(x))
            out += [statsdf]

        out = pd.concat(out, axis = 1, keys = covariates)
        newindex = out.xs('pval', level=1, axis = 1).max(axis = 1).sort_values(ascending = False).index
        return out.loc[newindex, :].round(4)
    
    def explained_variance(self, targets: list = [] , covariates: list = []) -> pd.DataFrame:
        return self.df[covariates+targets].corr().loc[covariates, targets]**2
        
    
    def plot_var_distribution(self, targets: list = [] , covariates: list = []) -> pd.DataFrame:
        numeric_columns = list(set(self.numerical_columns) & set(targets if len(targets) > 0 else self.df.columns))
        categorical_columns = list(set(self.categorical_columns) & set(covariates if len(covariates) > 0 else self.df.columns))
        print(categorical_columns)
        dfsm = self.df[numeric_columns + categorical_columns].copy()
        dfsm.loc[:,numeric_columns] = StandardScaler().fit_transform(dfsm[numeric_columns])
        melted = pd.melt(dfsm, id_vars=categorical_columns, value_vars=numeric_columns, value_name = 'normalized values')
        sns.set(rc={'figure.figsize':(2*11.7,2*8.27),"font.size":50,"axes.titlesize":30,"axes.labelsize":20, 
                    "lines.linewidth": 3,"legend.fontsize":20},style="white", context='paper',font_scale=3)
        stats_data = self.do_anova(numeric_columns, categorical_columns)

        for cat_col in categorical_columns:
            g = sns.boxenplot(data = melted, x = 'variable', y = 'normalized values', hue = cat_col) # palette=['steelblue', 'firebrick'] ,hue_order = ['M', 'F'],
            plt.ylabel('Normalized values')
            plt.xlabel('Parameter')
            plt.xticks(rotation=75)
            max_val = g.get_ylim()[1]
            sns.despine()
            g.plot(g.get_xlim(), [0,0], linestyle = 'dashed', color = 'black', label = 'average')
            plt.legend('') if len(dfsm[cat_col].unique()) > 10 else plt.legend(bbox_to_anchor=(1.01, 0.5)) 
            for num,col in enumerate(numeric_columns):
                p = stats_data.loc[col, (cat_col, 'pval')]
                if -np.log10(.01)>p>-np.log10(.05):
                    g.plot([num-0.2, num-.2, num +.2, num +.2], [max_val - .2, max_val,max_val, max_val-.2], lw=3, color = 'black')
                    g.text(num, max_val -.2 , '*', ha='center', va='bottom', color='black')
                if p>-np.log10(.01):
                    g.plot([num-0.2, num-.2, num +.2, num +.2], [max_val - .2, max_val,max_val, max_val-.2], lw=3, color = 'black')
                    g.text(num, max_val -.2, '*', ha='center', va='bottom', color='red')
            plt.show()
            
    def var_distributions_for_plotly(self, targets: list = [] , covariates: list = []) -> pd.DataFrame:
        numeric_columns = list(set(self.numerical_columns) & set(targets if len(targets) > 0 else self.df.columns))
        categorical_columns = list(set(self.categorical_columns) & set(covariates if len(covariates) > 0 else self.df.columns))
        dfsm = self.df[numeric_columns + categorical_columns].copy()
        dfsm.loc[:,numeric_columns] = StandardScaler().fit_transform(dfsm[numeric_columns])
        melted = pd.melt(dfsm, id_vars=categorical_columns, value_vars=numeric_columns, value_name = 'normalized values')
        return melted

    def make_report(self,filename: str):
        from ydata_profiling import ProfileReport
        ProfileReport(self.df,  interactions=None, correlations={"cramers": {"calculate": False}}).to_file(filename) #
        
    def get_outliers(self, subset: list = [], threshold: float = 3) -> pd.DataFrame:
        numeric_columns = list(set(self.numerical_columns) & set(subset if len(subset) > 0 else self.df.columns))
        scaled_values = np.abs(StandardScaler().fit_transform(self.df[numeric_columns]))
        return self.df[( scaled_values > threshold).any(axis=1)]
    
    def plot_clusters(self ,subset = None, dim_red_func = UMAP(n_components=2) , hue = HDBSCAN(), return_ = 'outliers') -> pd.DataFrame:
        if not subset: subset = self.numerical_columns
        data = dim_red_func.fit_transform(self.df.dropna()[subset])
        labels = LabelEncoder().fit_transform(self.df.dropna()[hue]) if (type(hue) == str) else hue.fit_predict(data)
        palette = sns.color_palette('deep', len(np.unique(labels)) + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data.T[0], data.T[1], c=colors, **{'alpha' : 0.25, 's' : 80, 'linewidths':0})
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        sns.despine()
        plt.show()
        if return_ == 'outliers':
            return self.df.dropna().loc[labels < 0,:]
        return self.df.dropna().loc[labels > 0,:]
    
    def run_all(self, clustering_hue = HDBSCAN(min_cluster_size = 30), targets: list = [], covariates: list = [], outlier_thrs: float = 4 ):
        print('doing clustering over the target variables')
        display(self.plot_clusters(hue = clustering_hue)) #
        print('anova test for all variables and all covariates')
        display(self.do_anova(targets = targets, covariates = covariates))
        print('visualizing distribution of variables per covariate')
        self.plot_var_distribution(targets = targets, covariates = covariates)
        print(f'printing outliers > {outlier_thrs} std')
        display(self.get_outliers(subset = targets, threshold= outlier_thrs))
        

        

def regress_out_covariates(df: pd.DataFrame, covariates: list, variates: list, \
                           preprocessing = make_pipeline(KNNImputer(), 
                                                         QuantileTransformer(n_quantiles = 100)) 
                          ):
    df2 = pd.DataFrame(preprocessing.fit_transform(df[covariates + variates]), 
                       columns = covariates + variates)#
    regression = MultiOutputRegressor(LinearRegression()).fit(df2[covariates], df2[variates])
    regressedOut = df2[variates] - regression.predict(df2[covariates])
    #### apply quantile transform again
    regressedOut[:] = QuantileTransformer(n_quantiles = 100).fit_transform(regressedOut)
    out = df.copy()
    out.loc[:, variates] = regressedOut.values
    return out


def regress_out_per_group(df: pd.DataFrame,groups: list, variates: list = [], **kwds):
    if not variates:
        variates = df.select_dtypes(include=np.number).columns
        
    return  df.groupby(groups).apply(lambda x:regress_out_covariates(x, variates))


def regress_out(df, variable, covariates, model = LinearRegression()):
    covariates = list(set(covariates))
    subdf = df[variable + covariates].dropna(axis = 0)
    regression = model.fit(subdf[covariates], subdf[variable])
    regressedOut = df[variable] - regression.predict(df[covariates].fillna(df[covariates].mean()))
    return regressedOut.add_prefix('regressedLR_')


def regress_out_v2(df, variable, covariates, model = LinearRegression()):
    subdf = df[variable + covariates].dropna(axis = 0)
    def tempfunc():pass
    subdf.groupby(subdf.columns.str.contains('OHE')).apply()
    regression = model.fit(subdf[covariates], subdf[variable])
    regressedOut = df[variable] - regression.predict(df[covariates])
    return regressedOut.add_prefix('regressedLR_')


def quantileTrasformEdited(df, columns):
    return (df[columns].rank(axis = 0, method = 'first')/(df[columns].count()+1)).apply(norm.ppf)

