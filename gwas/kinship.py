import numpy as np
import scipy.sparse as sps
from scipy.sparse import identity
from tqdm import tqdm
import numpy as np
import itertools
import dask.bag as db
from scipy.sparse import csr_matrix, identity
from collections import defaultdict
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
import nltk

def sps2fig(sparse_matrix, ids):
    res = pd.DataFrame(sparse_matrix.todense(), columns=ids, index = ids )
    res = res.loc[~res.index.str.contains('^rndgenerated@'), ~res.columns.str.contains('^rndgenerated@')]
    return (hm:=res.hvplot.heatmap(cmap = 'Blues',colorbar = False ).opts(invert_yaxis = True)) \
            * hv.Labels(hm).opts(padding=0, text_color = 'red', text_font_size = '5pt', frame_width = 500, frame_height = 500, xrotation =90)

def pedigree2sparse(pedigree: pd.DataFrame,parent1_col = 'dam', parent2_col = 'sire', offspring_col = 'rfid', 
                    na_ids = ['NaN', 'NA', 'na', 0, 'nan', 'NAN', 'n/a', 'N/A'] ):
    melted = pedigree.melt(id_vars=offspring_col, value_vars=[parent1_col, parent2_col], value_name = 'parent', var_name='relatioship')\
                     .replace(na_ids, np.nan)\
                     .dropna(how='any').assign(distance = 1/2)
    num2id = melted[[offspring_col, 'parent']].melt().drop_duplicates(subset='value').reset_index().value.to_dict()
    id2num = {v:k for k,v in num2id.items()}
    melted['offspring_id'] = melted[offspring_col].map(id2num)
    melted['parent_id'] = melted.parent.map(id2num)
    sped = sps.coo_matrix((np.ones_like(melted.offspring_id)/2, (melted.offspring_id, melted.parent_id)) ,  shape = (len(id2num),len(id2num)) ).tocsr()
    return sped, melted, id2num

def cutoff_hist_valleys(x, bins=1000, sigma=1.5, rang =  [0,1], total_density = .98):
    hist, edges = np.histogram(x, bins=bins, density=True, range = rang )
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_filter1d(hist, sigma=sigma)
    mins = argrelextrema(smooth, np.less)[0]
    smooth = pd.Series(smooth, index = np.linspace(rang[0],rang[1],bins))
    cutoffs = pd.DataFrame([[x,y] for x,y in nltk.ngrams([0]+ list(centers[mins])+ [1], 2)],  columns = ['start', 'end'])
    cutoffs['density'] = cutoffs.apply(lambda x: smooth[x.start: x.end].sum(), axis = 1)
    cutoffs = cutoffs.sort_values('density', ascending = True)
    cutoffs['density_cumsum'] = cutoffs.density.cumsum()
    cutoffs['density_pass'] = cutoffs['density_cumsum'].gt(total_density)
    cutoffs = cutoffs.sort_index()    
    return cutoffs, centers, smooth

def kinship_from_pedigree(pedigree: pd.core.frame.DataFrame, parent1_col='dam', parent2_col='sire', offspring_col='rfid',
                         threshold_approx = 0.01, na_ids = ['NaN', 'NA', 'na', 0, 'nan', 'NAN', 'n/a', 'N/A'], density_bins = 1000, density_sigma = 2):
    sped, melted, id2num = pedigree2sparse(pedigree,  parent1_col=parent1_col, parent2_col=parent2_col, offspring_col=offspring_col)
    if len(set(np.unique(np.array(sped.sum(axis = 1)))) - {0,1, .5}): raise ValueError('some individuals have more than 2 parents')
    phi = kinship_from_child_parent_fast(sped,threshold_approx=threshold_approx )
    phicoo = phi.tocoo()
    phidf = pd.DataFrame(dict(id1 = phicoo.row, id2 = phicoo.col, phi = phicoo.data) )
    del(phicoo)
    id2num_r = pd.Series(data = id2num.keys(), index = id2num.values()).to_dict()
    phidf['iid1'] = phidf.id1.map(id2num_r)
    phidf['iid2'] = phidf.id2.map(id2num_r)
    phidf = phidf[['id1', 'iid1', 'id2', 'iid2', 'phi']]
    phidf['phi_bin']= pd.cut(phidf.phi, bins =[0, 0.0255,0.0535, 0.1125, 0.2065 , .3, .45, .55, 1  ] )
    cutoffs, centers, smooth = cutoff_hist_valleys(phidf.phi, bins =density_bins, sigma= density_sigma)
    bins2 = cutoffs.query('density_pass').melt(value_vars=['start', 'end']).drop_duplicates(subset= 'value').value.to_list()+ [1]
    phidf['phi_autobin']= pd.cut(phidf.phi, bins =sorted(bins2))
    return phi, phidf, sped, melted, id2num, cutoffs

def chrYMT_from_pedigree(ped, parent_female_col='dam', parent_male_col='sire',  offspring_col='rfid',):
    spedy, meltedy, id2numy = pedigree2sparse(ped.reset_index().assign(**{parent_female_col : np.nan}))
    spedmt, meltedmt, id2numt = pedigree2sparse(ped.reset_index().assign(**{parent_male_col : np.nan}))
    spedmt, spedy = spedmt + spedmt.T , spedy + spedy.T
    ccy = sps.csgraph.connected_components(spedy)[1]
    ccmt = sps.csgraph.connected_components(spedmt)[1]
    ytm_version = pd.DataFrame({'rfid': id2numy.keys(), 'Y':ccy[[x for x in id2numy.values() if x<ccy.shape[0]]]}).merge(
                     pd.DataFrame({'rfid': id2numt.keys(), 'MT':ccmt[[x for x in id2numt.values() if x<ccmt.shape[0]]]}),on = 'rfid')\
                      .reset_index(drop = True)
    ytm_version['Y'] = 'connected_component_'+ ytm_version['Y'].astype(str)
    ytm_version['MT'] = 'connected_component_'+ ytm_version['MT'].astype(str)
    return ytm_version, (spedy, spedy)

def sum_powers(P, max_iter = 10):
    """S = I + P + P^2 + ...  (stops when the next power is empty)."""
    n = P.shape[0]
    S = identity(n, format="csr", dtype=float)
    Pk = P.copy()
    for i in tqdm(range(max_iter)):
        if Pk.nnz == 0: break
        S = (S + Pk).tocsr()
        Pk = (Pk @ P).tocsr()
    S.eliminate_zeros()
    return S

def parents_of(A):
    """list of parent index lists for each individual (row i has the parents)."""
    return [A.getrow(i).indices.tolist() for i in range(A.shape[0])]


def topo_order(P):
    """Parents before children (DAG required)."""
    par = parents_of(P)
    n = P.shape[0]
    founders = np.where(P.getnnz(axis=1) == 0)[0].tolist()
    seen = set(founders)
    order = founders[:]
    while len(order) < n:
        added = False
        for i in range(n):
            if i in seen: continue
            if set(par[i]).issubset(seen):
                order.append(i); seen.add(i); added = True
        if not added:  # graph not a DAG or malformed
            # fall back: append remaining
            order.extend([i for i in range(n) if i not in seen])
            break
    return order, founders, par

def kinship_from_child_parent(P, max_iter = 20):
    """
    P: CSR with P[i,j]=0.5 if j is a parent of i, else 0.
    Returns Phi (kinship) and A (relationship).
    Handles consanguineous matings via D_ii update using parental inbreeding.
    """
    n = P.shape[0]

    # 1) S = I + P + P^2 + ...  (finite because DAG)
    S = sum_powers(P, max_iter = max_iter)

    # 2) Build D diagonals topologically (parents before children).
    #    A simple order that works with S: founders first (rows with 0 parents).
    par = parents_of(P)
    founders = np.where(P.getnnz(axis=1) == 0)[0]
    order = list(founders) + [i for i in range(n) if i not in founders]  # ok for small pedigrees

    d = np.zeros(n, dtype=float)   # D_ii
    F = np.zeros(n, dtype=float)   # inbreeding

    # founders
    for i in founders:
        d[i] = 1.0
        F[i] = 0.0

    # helper for row-weighted dot: (S[p] * d) · S[q]^T
    Sd = S.multiply(d).tocsr()  # row-by-row scales each column by d[col]
    for i in tqdm(order):
        if i in founders:  continue
        ps = par[i]
        if len(ps) == 1:
            p, = ps
            # other parent unknown ⇒ F_i = 0
            F[i] = 0.0
            d[i] = 0.5 - 0.25*(F[p] + 0.0)
        elif len(ps) == 2:
            p, q = ps
            # φ_pq = 0.5 * (S[p,:] D S[q,:]^T)
            Apq = Sd.getrow(p).dot(S.getrow(q).T)[0, 0]
            phi_pq = 0.5 * Apq
            F[i] = phi_pq
            d[i] = 0.5 - 0.25*(F[p] + F[q])
        else:
            # more than two parents not allowed; treat as founders if none
            d[i] = 1.0
            F[i] = 0.0
        # keep Sd in sync after setting d[i]
        Sd[:, i] = S[:, i].multiply(d[i])

    # 3) A = S D S^T  (all sparse)
    D = sps.diags(d, format="csr")
    A = (S @ D @ S.T).tocsr()
    A.eliminate_zeros()
    Phi = (0.5 * A).tocsr()
    return Phi, A


# ---------- fast S via forward sweep (S = I + P S) ----------
def compute_S_forward(P):
    """
    P: CSR child->parent with 0.5 entries.
    Returns S in CSR with rows built once in topological order.
    """
    n = P.shape[0]
    order, founders, par = topo_order(P)

    # We'll build S in LIL (fast row writes), then convert to CSR.
    S = [dict() for _ in range(n)]  # row as {col: val}

    for i in order:
        row = S[i]
        row[i] = 1.0                    # identity contribution
        for p in par[i]:                # add parents' rows scaled by 0.5
            for k, v in S[p].items():
                row[k] = row.get(k, 0.0) + 0.5 * v

    # convert to CSR
    indptr = [0]
    indices = []
    data = []
    for i in range(n):
        # keep sparsity tidy
        items = [(j, v) for j, v in S[i].items() if v != 0.0]
        items.sort()
        indices.extend([j for j, _ in items])
        data.extend([v for _, v in items])
        indptr.append(len(indices))
    return csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(n, n))

    
# ---------- weighted row intersection (phi_pq core) ----------
def weighted_row_dot(S, p, q, d):
    """
    Compute sum_k S[p,k]*d[k]*S[q,k] quickly by intersecting the column indices.
    S must be CSR. p and q are row indices. d is 1D array.
    """
    rp_start, rp_end = S.indptr[p], S.indptr[p+1]
    rq_start, rq_end = S.indptr[q], S.indptr[q+1]
    idx_p = S.indices[rp_start:rp_end]; val_p = S.data[rp_start:rp_end]
    idx_q = S.indices[rq_start:rq_end]; val_q = S.data[rq_start:rq_end]

    # two-pointer merge over sorted column indices
    i = j = 0
    acc = 0.0
    while i < len(idx_p) and j < len(idx_q):
        a, b = idx_p[i], idx_q[j]
        if a == b:
            acc += val_p[i] * d[a] * val_q[j]
            i += 1; j += 1
        elif a < b: i += 1
        else: j += 1
    return acc


def kinship_from_child_parent_fast(P, return_S = False, threshold_approx = .01):
    """
    P: CSR with P[i,j]=0.5 if j is a parent of i, else 0.
    Returns Phi (kinship) and A (relationship) using:
      1) S via forward sweep (S = I + P S)
      2) D diagonals via topological pass and weighted row intersections
      3) A = (S * sqrt(d)) @ (S * sqrt(d)).T
    """
    n = P.shape[0]
    order, founders, par = topo_order(P)
    # 1) Build S once (fast)
    S = compute_S_forward(P)   # CSR
    # 2) Build D diagonals (Mendelian sampling variances) and inbreeding
    d = np.zeros(n, dtype=float)
    F = np.zeros(n, dtype=float)
    for i in founders:
        d[i] = 1.0; F[i] = 0.0
    for i in tqdm(order):
        if i in founders: continue
        ps = par[i]
        if len(ps) == 1:
            p = ps[0]
            F[i] = 0.0
            d[i] = 0.5 - 0.25*(F[p] + 0.0)
        elif len(ps) == 2:
            p, q = ps
            # φ_pq = 0.5 * sum_k S[p,k]*d[k]*S[q,k]
            phi_pq = 0.5 * weighted_row_dot(S, p, q, d)
            F[i] = phi_pq
            d[i] = 0.5 - 0.25*(F[p] + F[q])
        else:
            d[i] = 1.0; F[i] = 0.0
    # 3) A = S D S^T via column-scaling by sqrt(d)
    rootd = np.sqrt(d, dtype=float)
    Sc = S.tocsc(copy=True)              # scale columns in-place cheaply
    for j in tqdm(range(n)):
        if rootd[j] != 1.0:
            start, end = Sc.indptr[j], Sc.indptr[j+1]
            if end > start: Sc.data[start:end] *= rootd[j]
    if return_S: return Sc
    Sc.data[Sc.data < threshold_approx] = 0
    Sc.eliminate_zeros()
    Sc = Sc.tocsr()
    A = (Sc @ Sc.T)
    A.data[A.data < threshold_approx] = 0
    A.eliminate_zeros()
    return .5 * A
    # Phi = (0.5 * A)
    # return Phi, A





def sum_powers_pruned(
    P: csr_matrix,
    *,
    max_iter: int | None = 30,          # hard cap on generations
    min_weight: float | None = 1e-9,    # prune very small contributions each iter
    min_increment: float | None = 1e-9, # stop if this iter adds < tol (L1 mass)
    stall_tol: float = 1.02,            # stop if nnz grows by <2% for a few iters
    stall_patience: int = 2,            # how many stalls before stopping
    prune_S_each_iter: bool = True      # also prune S after adding Pk
):
    """
    Compute S = I + P + P^2 + ... using sparse matmuls with:
      - generation cap,
      - per-iter pruning of tiny values,
      - early stop when new contribution is negligible (L1 mass),
      - early stop when nnz growth stalls.

    P is CSR (child->parent), typically strictly lower-triangular with 0.5 edges.
    """
    if not sps.isspmatrix_csr(P):  P = P.tocsr()
    n = P.shape[0]
    S = sps.eye(n, format='csr', dtype=float)
    Pk = P.copy()
    # bookkeeping for early stop
    prev_nnz = float(Pk.nnz)
    stalls = 0
    it = 1
    T = max_iter if max_iter is not None else (n - 1)
    # small helpers
    def _prune_inplace(X: csr_matrix, tol: float):
        if tol is None: return X
        # zero out small data then coalesce/cleanup
        data = X.data
        if data.size:
            mask = np.abs(data) >= tol
            if mask.sum() < data.size:
                X.data = data[mask]
                X.indices = X.indices[mask]
                X.eliminate_zeros()
        return X
    for it in tqdm(range(T)):
        if Pk.nnz == 0: break
        # optional: prune current power
        if min_weight is not None:
            _prune_inplace(Pk, min_weight)
            if Pk.nnz == 0:  break
        # add this generation
        S = (S + Pk)
        if prune_S_each_iter and (min_weight is not None):
            _prune_inplace(S, min_weight)

        # contribution mass check (L1 of added block)
        if min_increment is not None:
            inc_mass = float(np.abs(Pk.data).sum())
            if inc_mass < min_increment: break
        # next power
        Pk_next = (Pk @ P)
        if Pk_next.nnz == 0: break
        # stall detection (nnz growth)
        growth = (Pk_next.nnz + 1e-12) / (prev_nnz + 1e-12)
        if growth < stall_tol:
            stalls += 1
            if stalls >= stall_patience:break
        else: stalls = 0
        prev_nnz = float(Pk_next.nnz)
        Pk = Pk_next
    S.eliminate_zeros()
    return S


# Phi, A = kinship_from_child_parent(sped2) 
# sps2fig(Phi, ids).opts(frame_width = 600,frame_height = 600)