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
from sklearn.preprocessing import QuantileTransformer,PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import panel as pn

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
        
    
    def plot_var_distribution(self, targets: list = [] , covariates: list = []) -> pn.Tabs:
        numeric_columns = list(set(self.numerical_columns) & set(targets if len(targets) > 0 else self.df.columns))
        categorical_columns = list(set(self.categorical_columns) & set(covariates if len(covariates) > 0 else self.df.columns))
        #print(categorical_columns)
        dfsm = self.df[numeric_columns + categorical_columns].copy()
        if len(numeric_columns):
            dfsm.loc[:,numeric_columns] = StandardScaler().fit_transform(dfsm[numeric_columns])
        melted = pd.melt(dfsm, id_vars=categorical_columns, value_vars=numeric_columns, value_name = 'normalized values')
        sns.set(rc={'figure.figsize':(2*11.7/4,2*8.27/4),"font.size":5,"axes.titlesize":5,"axes.labelsize":5, 
                    "lines.linewidth": 1,"legend.fontsize":5},style="white", context='paper',font_scale=1)
        stats_data = self.do_anova(numeric_columns, categorical_columns)
        tabs = pn.Tabs(tabs_location='above', width = 1000)
        for cat_col in categorical_columns:
            fig, ax = plt.subplots(figsize=(5, 5))
            g = sns.boxenplot(data = melted, x = 'variable', y = 'normalized values', hue = cat_col, ax = ax, flier_kws=dict(facecolor=".7", linewidth=.5, s = 3)) 
            # palette=['steelblue', 'firebrick'] ,hue_order = ['M', 'F'],
            # plt.ylabel('Normalized values')
            # plt.xlabel('Parameter')
            # plt.xticks(rotation=75)
            ax.set_ylabel('Normalized values')
            ax.set_xlabel('Parameter')
            ax.set_xticks(ax.get_xticks())  # Ensure proper xticks
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, size = 4)  # Rotate labels .tick_params(labelsize=14)
            max_val = ax.get_ylim()[1]
            sns.despine(ax=ax)  # Despine using the explicit Axes
            # max_val = g.get_ylim()[1]
            # sns.despine()
            #g.plot(g.get_xlim(), [0,0], linestyle = 'dashed', color = 'black', label = 'average')
            ax.plot(ax.get_xlim(), [0, 0], linestyle='dashed', color='black', label='average')
            # plt.legend('') if len(dfsm[cat_col].unique()) > 10 else plt.legend(bbox_to_anchor=(1.01, 0.5)) 
            if len(dfsm[cat_col].unique()) > 10: ax.legend('')
            else: pass
            # Add significance markers
            for num, col in enumerate(numeric_columns):
                p = stats_data.loc[col, (cat_col, 'pval')]
                if -np.log10(.01) > p > -np.log10(.05):
                    ax.plot([num - 0.2, num - .2, num + .2, num + .2], 
                            [max_val - .2, max_val, max_val, max_val - .2], 
                            lw=2, color='black')
                    ax.text(num, max_val - .2, '*', ha='center', va='bottom', color='black', fontsize = 20)
                if p > -np.log10(.01):
                    ax.plot([num - 0.2, num - .2, num + .2, num + .2], 
                            [max_val - .2, max_val, max_val, max_val - .2], 
                            lw=2, color='black')
                    ax.text(num, max_val - .2, '*', ha='center', va='bottom', color='red', fontsize = 20)
            ax.legend(bbox_to_anchor=(1.01, 0.5))
            fig.tight_layout()
            tabs.append((cat_col, pn.pane.Matplotlib(fig, max_width = 500, max_height = 500)))
        return tabs


            
            # for num,col in enumerate(numeric_columns):
            #     p = stats_data.loc[col, (cat_col, 'pval')]
            #     if -np.log10(.01)>p>-np.log10(.05):
            #         g.plot([num-0.2, num-.2, num +.2, num +.2], [max_val - .2, max_val,max_val, max_val-.2], lw=3, color = 'black')
            #         g.text(num, max_val -.2 , '*', ha='center', va='bottom', color='black')
            #     if p>-np.log10(.01):
            #         g.plot([num-0.2, num-.2, num +.2, num +.2], [max_val - .2, max_val,max_val, max_val-.2], lw=3, color = 'black')
            #         g.text(num, max_val -.2, '*', ha='center', va='bottom', color='red')
            #tabs = pn.Tabs(tabs_location='left')
            #tabs.append((cat_col, pn.pane.Matplotlib(g.figure, max_width = 1000, max_height = 700)))
            #plt.close()
            
    def var_distributions_for_plotly(self, targets: list = [] , covariates: list = []) -> pd.DataFrame:
        numeric_columns = list(set(self.numerical_columns) & set(targets if len(targets) > 0 else self.df.columns))
        categorical_columns = list(set(self.categorical_columns) & set(covariates if len(covariates) > 0 else self.df.columns))
        dfsm = self.df[numeric_columns + categorical_columns].copy()
        if len(numeric_columns):
            dfsm.loc[:,numeric_columns] = StandardScaler().fit_transform(dfsm[numeric_columns])
        melted = pd.melt(dfsm, id_vars=categorical_columns, value_vars=numeric_columns, value_name = 'normalized values')
        return melted

    def make_report(self,filename: str):
        from ydata_profiling import ProfileReport
        try:
            ProfileReport(self.df,  interactions=None, correlations={"cramers": {"calculate": False}}).to_file(filename) #
        except: 
            print('could not run profile report, data frame might have too many columns')
        
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
                           preprocessing = make_pipeline(KNNImputer(), QuantileTransformer(n_quantiles = 100)) 
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
    return df.groupby(groups).apply(lambda x:regress_out_covariates(x, variates))

def regress_out(df, variable, covariates, model = LinearRegression()):
    covariates = list(set(covariates))
    subdf = df[variable + covariates].dropna(axis = 0)
    regression = model.fit(subdf[covariates], subdf[variable])
    regressedOut = df[variable] - regression.predict(df[covariates].fillna(df[covariates].mean())).flatten()[:, None]
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

def ScaleTransformer(df, columns, method= 'quantile'):
    res = df.copy()
    if isinstance(method, str):
        if method == 'passthrough': return res
        elif method == 'quantile_noties': res = quantileTrasformEdited(res, columns)
        elif method == 'boxcox': res.loc[:, columns] = PowerTransformer().fit_transform(res.loc[:, columns])
        elif method == 'quantile': res.loc[:, columns] = QuantileTransformer(n_quantiles = res.shape[0], output_distribution='normal').fit_transform(res.loc[:, columns])
        else: print('not normalizing after regressing out covariates options are ("quantile", "boxcox", "passthrough")')
        return res
    else:
        try: res.loc[:, columns] = method.fit_transform(res.loc[:, columns])
        except: raise TypeError('normalize does not contain a fit_transform method')
        return res
    
class StepWiseRegression():
    def __init__(self, threshold: float = 0.02,estimator=LinearRegression()):
        self.threshold = threshold
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.metrics import r2_score
        r2_score_pearson = lambda x, y : np.corrcoef(x.flatten(), y.flatten())[0,1]**2
        ### convert to np.array initalize residuals
        self.resid = np.array(y)
        X = np.array(X)
        ### set model stach and order of X columns
        self.modelstack = []
        self.x_order = []
        ### auxiliary function to get sort a list of tuples by the first value and get the maximum value 
        topsort = lambda y: sorted(y,key=lambda x:x[0], reverse=True)[0]
        #### while the X covariate is larger than threshold
        while (best_trait := \
               topsort([(r2_score_pearson(self.resid , X[:, [col]]),  col) for col in range(X.shape[1])]))[0] \
               > self.threshold + 1e-10: 
            ### add covariate idx to self.x_order
            self.x_order += [best_trait[1]]
            ### add fitted linear regression 
            self.modelstack += [ self.estimator.fit(X[:, [best_trait[1]]], self.resid )]
            self.resid -= self.modelstack[-1].predict(X[:, [ best_trait[1]]])
        return self

    def predict(self, X):
        ### convert to np.array
        if not len(self.modelstack): return np.zeros((X.shape[0], 1))
        X = np.array(X)
        ### add the effect of each predictors as stack
        ad = np.array([model.predict(X[:, [col]]) \
                      for col, model\
                      in zip(self.x_order, self.modelstack)])
        #return sum of the effects
        return ad.sum(axis = 0)

    def fit_predict(self,X, y):
        return self.fit(X,y).predict(X)
