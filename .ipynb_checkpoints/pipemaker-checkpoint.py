import pandas as pd
import statsReport
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import re
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.base import is_classifier, is_regressor
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["figure.figsize"] = (15,15)
import math
import seaborn as sns
import shap
from datetime import datetime
import time
import umap
from pandas_profiling import ProfileReport
from sklearn.neighbors import kneighbors_graph
from prophet import Prophet
from umap import UMAP
from lightgbm import LGBMRegressor,LGBMClassifier, plot_tree
from sklearn.preprocessing import Binarizer,FunctionTransformer, KBinsDiscretizer, KernelCenterer, LabelBinarizer, LabelEncoder, MinMaxScaler,MaxAbsScaler,\
                                  QuantileTransformer, Normalizer, OneHotEncoder, OrdinalEncoder,PowerTransformer, RobustScaler, SplineTransformer,StandardScaler, PolynomialFeatures
from sklearn.decomposition import DictionaryLearning,FastICA, IncrementalPCA, KernelPCA, MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF,PCA,SparsePCA, FactorAnalysis,\
                                  TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding, TSNE
from sklearn.pipeline import make_pipeline
from sklearn.utils import estimator_html_repr
#import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import  BaseEnsemble,RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding, ExtraTreesClassifier, ExtraTreesRegressor,\
                          BaggingClassifier, BaggingRegressor, IsolationForest, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier,\
                          AdaBoostRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, HistGradientBoostingClassifier,\
                          HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, Hinge, Huber, HuberRegressor, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC,\
                             LinearRegression, Log, LogisticRegression, LogisticRegressionCV, ModifiedHuber,MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso,\
                             MultiTaskLassoCV, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, \
                             QuantileRegressor, Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier, SGDRegressor, SGDOneClassSVM, SquaredLoss,TheilSenRegressor, \
                             RANSACRegressor, PoissonRegressor,GammaRegressor,TweedieRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB,ComplementNB, CategoricalNB
from scipy.stats import ttest_ind, ttest_1samp
from  plotly.offline  import plot_mpl
import plotly.tools as ptools
import networkx as nx
from prophet.plot import plot_plotly, plot_components_plotly
import calendar
from prophet.utilities import regressor_coefficients 
import plotly.express as px
import base64
import numpy as np
import pandas as pd
from io import StringIO
import io
#from keplergl import KeplerGl
import hdbscan
import datetime
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
#from jupyter_dash.comms import _send_jupyter_config_comm_request
#_send_jupyter_config_comm_request()
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from joblib import Memory
from shutil import rmtree
from sklearn import svm, datasets
from sklearn.metrics import auc,confusion_matrix,classification_report
from sklearn.metrics import RocCurveDisplay,ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from skopt import BayesSearchCV, gp_minimize, forest_minimize, gbrt_minimize
from skopt.searchcv import BayesSearchCV as BSCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from sklearn.feature_selection import RFECV
set_config(display='diagram') 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib
from sklearn.base import clone
import plotly.figure_factory as ff
import statsReport
from glob import glob
import gzip


def optionize(lis):
    return [{'label':x, 'value': x} for x in lis]

def flatten_cols(input_list):
    output_list = []
    for element in input_list:
        if type(element) == list: output_list.extend(flatten_cols(element))
        else: output_list.append(element)
    return output_list

class pipemaker2:
    def __init__(self, df,ipt_pipe, target ,*, height = 'auto', width = 'auto'):
        self.pipe_list = []
        self.df = df
        self.TG = target
        self.check = 0
        self.cached_pipe = 0
        self.location = 0
        self.memory = 0
        self.optimized_pipe = (0, 0)
        self.input_pipe = ipt_pipe
        
    def Pipe(self):
        return clone(self.input_pipe)
    
    def Cache_pipe(self):
        self.location = 'cachedir'
        self.memory = Memory(location=self.location, verbose=0)
        self.cached_pipe = self.Pipe().set_params(memory = self.memory)
    
    def release_cache(self):
        self.memory.clear(warn=True)
        rmtree(self.location)
        del self.memory
        
    def export_kwards(self):
        return self.Pipe().get_params()
    def fit_transform(self):
        return self.ColumnTransform().fit_transform(self.df)
    def fit_predict(self):
        return self.Pipe().fit_predict(self.df, self.df[self.TG])
    def fit(self):
        return self.Pipe().fit(self.df, self.df[self.TG])
    
    def RFECV(self):
        preprocessed_df = pd.DataFrame(self.Pipe()['preprocessing'].fit_transform(self.df))
        
        if self.optimized_pipe[1] == 0:
            selector = RFECV(self.Pipe()['classifier'], step=1, cv=KFold(10, shuffle= True)).fit(preprocessed_df, self.df[self.TG])
        else:
            selector = RFECV(self.optimized_pipe[0]['classifier'], step=1, cv=KFold(10, shuffle= True)).fit(preprocessed_df, self.df[self.TG])
            
        hX = np.array( range(1, len(selector.grid_scores_) + 1))
        hY= selector.grid_scores_
        H = pd.DataFrame(np.array([hX, hY]).T, columns = ['Number of parameters', 'Cross Validation Score'])
        
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(hX, hY)
        plt.show()
        return pd.DataFrame([selector.ranking_, selector.support_], columns = preprocessed_df.columns, index = ['Ranking', 'support'])
    
    def make_skpot_var(self, param, temperature = 3, distribution = 'uniform', just_classifier = False): #'log-uniform'
        value = self.export_kwards()[param]
        if just_classifier == True: name = param.split('__')[1]
        else: name = param
        
        
        if name == 'alpha': return Real(1e-8, 1, 'log-uniform',name = name)
        
        if value == 0 or value ==1: return
        
        if type(value) == int: 
            if value == -1: return Integer(1, 200, name = name)
            lower_bondary = int(value/temperature)
            if lower_bondary < 2: lower_bondary = 2
            upper_bondary = int(value*temperature) + lower_bondary
            #if value <= 1: return Real(1e-3, 1, distribution ,name = name)
            return Integer(lower_bondary, upper_bondary, distribution ,name = name)

        if type(value) == float:
            if value == -1: return Real(1, 200, name = name)
            if value <= 1: return Real(1e-3, 1, distribution ,name = name)
            lower_bondary = value/temperature
            if lower_bondary < 2: lower_bondary = 2
            upper_bondary = value*temperature + lower_bondary
            return Real(lower_bondary, upper_bondary, distribution ,name = name)

    def skopt_classifier_space(self, just_classifier = False):
        dic = self.export_kwards()
        classifier_params = [x for x in  dic.keys() 
                             if x.find('classifier__') != -1 
                             and  x.find('silent') == -1 
                             and  x.find('n_jobs') == -1
                             and x != 'classifier__tol'
                             and x != 'classifier__max_iter'
                             and x.find('bagging_fraction') == -1 
                             and x != 'classifier__subsample'
                             and x != 'classifier__validation_fraction'] # and
        SPACE = [self.make_skpot_var(i, just_classifier = just_classifier) for i in classifier_params]
        SPACE = [x for x in SPACE if x if x != None ]
        return SPACE

    def objective(self, params):
        classifier = self.Pipe().set_params(**{dim.name: val for dim, val in zip(self.skopt_classifier_space(), params)})
        return -np.mean(cross_val_score(classifier, self.df, self.df[self.TG], cv = StratifiedKFold(n_splits = 5, shuffle=True)))
    
    def objective_just_classifier(self, params, metric , cv_method ):
        return -np.mean(cross_val_score(self.cached_pipe['classifier'].set_params(**{dim.name: val for dim, val in zip(self.skopt_classifier_space(just_classifier = 1), params)}), 
                                        self.transformed_opt, 
                                        self.target_opt,
                                        scoring = metric,
                                        cv = cv_method, 
                                        n_jobs = -1))
    
    def objective_cached(self, params):
        return -np.mean(cross_val_score(self.cached_pipe.set_params(**{dim.name: val for dim, val in zip(self.skopt_classifier_space(), params)}),
                                        self.df, 
                                        self.df[self.TG], 
                                        cv = StratifiedKFold(n_splits = 5, shuffle=True)))
    
    
    def optimize_classifier(self, n_calls = 50, cache = False):
        if cache: 
            self.Cache_pipe()
            result = gp_minimize(self.objective_cached, self.skopt_classifier_space() , n_calls=n_calls)
            self.release_cache()
        else: result = gp_minimize(self.objective, self.skopt_classifier_space() , n_calls=n_calls)
        #plot_convergence(result)
        #_ = plot_objective(result, n_points=n_calls)
        #print(result.fun)
        return {'result': result, 'best_params': self.get_params(result, self.skopt_classifier_space() )} 
    
    def fast_optimize_classifier(self, n_calls = 50,  is_classifier = True):
        self.Cache_pipe()
        
        self.transformed_opt = self.cached_pipe['preprocessing'].fit_transform(self.df)
        self.target_opt = self.df[self.TG]
        
        if is_classifier: 
            cv_method = StratifiedKFold(n_splits = 5, shuffle=True)
            metric    = 'f1_weighted'
        else:      
            cv_method = KFold(n_splits = 5, shuffle=True)
            metric    = 'r2'
        
        result = gp_minimize(lambda x: self.objective_just_classifier(x, metric, cv_method), self.skopt_classifier_space(just_classifier = True) , n_calls=n_calls)
        self.release_cache()
        
        best_params = self.get_params(result, self.skopt_classifier_space(just_classifier = True))
        best_params = {'classifier__'+ i[0]:i[1] for i in best_params.items()}
        
        self.optimized_pipe = (self.Pipe().set_params(**best_params), 1)

        return {'result': result, 'best_params':best_params} 

    def get_params(self, result_object, space):
        try:
            return { i.name: result_object.x[num] for  num, i in enumerate(space) }
        except:
            raise
             
    def Vis_Cluster(self, method):
        transformed = UMAP().fit_transform(self.Pipe()['preprocessing'].fit_transform(self.df))
        classsification = method.fit_predict(transformed)  #(*args, **kwds)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(classsification).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in classsification]
        plt.scatter(transformed.T[0], transformed.T[1], c=colors, s = MinMaxScaler(feature_range=(30, 300)).fit_transform(self.df[self.TG].values.reshape(-1, 1)) , **{'alpha' : 0.3,  'linewidths':1})
        frame = plt.gca() 
        for num, spine in enumerate(frame.spines.values()):
            if num == 1 or num == 3: spine.set_visible(False)
        plt.title('Clusters found by {}'.format(str(method)), fontsize=24)
        plt.show()
        return 
    
    def Evaluate_model(self):
        tprs = []
        aucs = []
        prd = []
        tru = []
        mean_fpr = np.linspace(0, 1, 100)
        X = self.df.copy()
        y = self.df[self.TG]
        if self.optimized_pipe[1] == 0: clf = self.Pipe()
        else: clf = self.optimized_pipe[0]
        fig, ax = plt.subplots(1, 2, figsize = (20,10))
        try:
            for i, (train, test) in enumerate(StratifiedKFold(n_splits=5, shuffle=True).split(X, y)):
                clf.fit(X.iloc[train], y.iloc[train])
                viz = RocCurveDisplay.from_estimator(clf, X.iloc[test], y.iloc[test],
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3, lw=1, ax=ax[0])
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            ax[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax[0].plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax[0].fill_between(mean_fpr, tprs_lower, tprs_upper, color='steelblue', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')

            ax[0].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
            #       title="Receiver operating characteristic example")
            ax[0].legend(loc="lower right")
        except: 
            print('non-binary classifier')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        try:
            ConfusionMatrixDisplay.from_estimator(clf.fit(X_train, y_train), X_test, y_test,
                                         display_labels=['negative detection', 'positive detection'],
                                         cmap=plt.cm.Blues, ax = ax[1])
            ax[1].grid(False)
            #fig.tight_layout()
        except: 
            print('is it a regressor?')
            plt.close()
        try: 
            report = classification_report(clf.predict(X_test), y_test, output_dict=True) # target_names=['Negative detection', 'Positive detection']
        except: #### report for regression
            if self.optimized_pipe[1] == 0: clf = self.Pipe()
            else: clf = self.optimized_pipe[0]
            report = cross_validate(clf, X, y, cv=5,  scoring=('neg_mean_absolute_percentage_error','r2','explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'), return_train_score = True)
            fig, ax = plt.subplots(1, 1, figsize = (10,10))
            #fig.tight_layout()
        return pd.DataFrame(report), fig
        
    def named_preprocessor(self):  
        naming_features = []
        for transformer in self.Pipe()['preprocessing'].transformers:
            transformed = ColumnTransformer(transformers = [transformer]).fit_transform(self.df)
            if transformed.shape[1] == len(transformer[2]):
                naming_features += list(transformer[2])
            else:
                naming_features += [transformer[0] +'.'+ str(i) for i in range(transformed.shape[1]) ]
        if self.optimized_pipe[1] == 0: clf = self.Pipe()
        else: clf = self.optimized_pipe[0]
        return pd.DataFrame(clf['preprocessing'].fit_transform(self.df), columns = naming_features)

    def Shapley_feature_importance(self, clustering_cutoff = -1, forceplot = 'matplotlib', do_force_plot=False):
        if self.optimized_pipe[1] == 0: clf = self.Pipe()
        else: clf = self.optimized_pipe[0]
        shap.initjs()
        dat_trans = self.named_preprocessor()
        #explainer = shap.TreeExplainer(clf['classifier'].fit(dat_trans, self.df[self.TG])) #,feature_perturbation = "tree_path_dependent"
        #shap_values = explainer.shap_values(dat_trans)
        
        #try: 
        #    explainer = shap.TreeExplainer(clf['classifier'].fit(dat_trans, self.df[self.TG]), dat_trans) 
        #    shap_values = explainer.shap_values(dat_trans, check_additivity=False) 
        #    failhere
        #except: 
        #    explainer = shap.Explainer(clf['classifier'].fit(dat_trans, self.df[self.TG]), dat_trans) 
        #    shap_values = explainer.shap_values(dat_trans)   
        try: 
            explainer = shap.TreeExplainer(clf['classifier'].fit(dat_trans, self.df[self.TG]), dat_trans) 
            shap_values = explainer.shap_values(dat_trans, check_additivity=False) 
        except: 
            explainer = shap.Explainer(clf['classifier'].fit(dat_trans, self.df[self.TG]), dat_trans) 
            shap_values = explainer.shap_values(dat_trans)

        try : shap.plots.heatmap(explainer(dat_trans), show= False) 
        except : shap.plots.heatmap(explainer(dat_trans), show= False, check_additivity=False)
        fig1 = plt.gcf()
        fig1.set_figheight(15)
        fig1.set_figwidth(15)
        fig1.tight_layout()
        fig1.show()   
        
        #### force-plot
        print('doing forceplot')
        if do_force_plot:
            try: a = [shap.force_plot(explainer.expected_value[i], shap_values[i], dat_trans, matplotlib=False, figsize=(18, 18)) \
                      for i in range(len(shap_values))]
            except: a = [shap.force_plot(explainer.expected_value, shap_values, dat_trans, matplotlib=False, figsize=(18, 18)) ]
        else: a = []
        
        ### dependence matrix
        #figdm, axdm = plt.subplots(len( dat_trans.columns),  len(dat_trans.columns), figsize=(15, 15))
        #try:
        #    ivalues = explainer.shap_interaction_values(dat_trans)
        #    d = {i: name for i,name in enumerate(dat_trans.columns)}
        #    for i in d.keys():
        #        for j in d.keys():
        #            shap.dependence_plot((d[i], d[j]), ivalues[1], dat_trans, ax = axdm[i,j], show = False)
        #except: print('failed at dependence matrix')
                
        if clustering_cutoff < 0:
            fig_summary, ax = plt.subplots(figsize=(15, 15))
            shap.summary_plot(shap_values,dat_trans,plot_size=(10,10), max_display=40,show= True)
            
            fig_summary, ax = plt.subplots(figsize=(15, 15))
            shap.summary_plot(shap_values,dat_trans, plot_type='bar',plot_size=(10,10), max_display=40,show= True)
        if clustering_cutoff > 0 :
            clustering = shap.utils.hclust(dat_trans, self.df[self.TG])
            shap.plots.bar(explainer(dat_trans),  clustering=clustering_cutoff,  clustering_cutoff=0.9, check_additivity=False)
            
        try:
            shap.plots.scatter(explainer(dat_trans[:])[:,abs(interaction_shap_values.values).sum(axis=0).argsort()[-8:][::-1]])
        except:
            print('didnt make shap scatterplot')
             
                
        return a #fig,
