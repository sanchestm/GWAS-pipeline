import pandas as pd
import pandas_plink
import numpy as np

def merge_duplicates_fancy(df):
    if not len(df): return
    if len(df) == 1: df.iloc[0]
    return pd.Series({y:'|'.join(df[y].astype(str).unique()) for y in df.columns})

def groupby_no_loss_merge(df, groupcol):
    return df.groupby(col).progress_apply(merge_duplicates_fancy).reset_index(drop = True)

def decompose_grm(grm_path, n_comp = 50, verbose = True):
    (grmxr, ar) =  pandas_plink.read_grm(grm_path)
    eigval, eigvec = np.linalg.eig(grmxr)
    eig_pairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(len(eigval))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    explained_var = np.array([eig_pairs[i][0] for i in range(n_comp)])/sum(list(zip(*eig_pairs))[0])
    if verbose: print(f'explained_variances:{explained_var}\
          \n total explained var:{sum(explained_var)}' )
    return pd.DataFrame(np.vstack((eig_pairs[i][1] for i in range(n_comp))).T,
             columns = [f'GRM_PC{x}' for x in range(1, n_comp+1)],
             index= grmxr.sample_0.astype(str))

def decompose_grm_pca(grm_path, n_comp = 5, verbose = True):
    (grmxr, ar) =  pandas_plink.read_grm(grm_path)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp)
    decomp = pca.fit_transform(grmxr)
    if verbose: print(f'explained_variances:{pca.explained_variance_ratio_}\
          \ntotal explained var:{sum(pca.explained_variance_ratio_)}' )
    return pd.DataFrame(decomp, columns = [f'GRM_PC{x}' for x in range(1, decomp.shape[1]+1)],
             index= grmxr.sample_0.astype(str))

def get_topk_values(s, k): 
    return s.groupby(s.values).agg(lambda x: '|'.join(x.index) ).sort_index()[::-1].values[:k]