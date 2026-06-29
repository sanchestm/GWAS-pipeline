import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, spsolve_triangular
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, depth_first_order
from tqdm import tqdm

def topological_sort(adj):
    # Ensure CSR format for efficient traversal
    adj = csr_matrix(adj, dtype=bool)
    n_nodes = adj.shape[0]
    # --- Step 1: Detect cycles using strongly connected components ---
    n_components, labels = connected_components(adj, directed=True, connection='strong')
    if np.any(np.bincount(labels) > 1):
        raise ValueError("Graph contains at least one cycle; topological order undefined.")
    # --- Step 2: Compute DFS finishing order using csgraph ---
    visited = np.zeros(n_nodes, dtype=bool)
    finish_order = []
    for start in range(n_nodes):
        if not visited[start]:
            # depth_first_order returns nodes in visitation order
            nodes, predecessors = depth_first_order(adj, i_start=start, directed=True)
            visited[nodes] = True
            # We append them in reverse to emulate finishing order
            finish_order.extend(nodes[::-1])
    # Remove duplicates while preserving last occurrence (like DFS finishing)
    # np.unique(..., return_index=True) returns first occurrence; we reverse logic
    _, idx = np.unique(finish_order[::-1], return_index=True)
    topo_order = np.array(finish_order[::-1])[np.sort(idx)]
    return topo_order

# ---------- graph utilities (SciPy csgraph) ----------
def _permute_by_components(P: csr_matrix):
    n = P.shape[0]
    n_comp, labels = connected_components(P + P.T, directed=False, return_labels=True)
    perm = np.argsort(labels)
    inv  = np.empty_like(perm); inv[perm] = np.arange(n)
    Pp   = P[perm][:, perm]
    labels = labels[perm]
    # block ranges (each family)
    cuts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    spans = list(zip(cuts[:-1], cuts[1:]))  # [start,end)
    return Pp, perm, inv, spans

def _topo_order_block(Pb: csr_matrix):
    # parents-before-children: call topo on transpose
    ord_blk = topological_sort(Pb.T)
    inv_blk = np.empty_like(ord_blk); inv_blk[ord_blk] = np.arange(Pb.shape[0])
    return Pb[ord_blk][:, ord_blk], ord_blk, inv_blk

# ---------- S via forward sweep (no SpGEMM; minimal Python) ----------
def _compute_S_forward(P_topo: csr_matrix) -> csr_matrix:
    n = P_topo.shape[0]
    parents = [P_topo.indices[P_topo.indptr[i]:P_topo.indptr[i+1]] for i in range(n)]  # CSR row indices
    rows = [dict() for _ in range(n)]
    for i in range(n):                   # topo order guarantees parents done
        ri = rows[i]; ri[i] = 1.0
        for p in parents[i]:
            for k, v in rows[p].items():
                ri[k] = ri.get(k, 0.0) + 0.5*v
    indptr=[0]; idx=[]; data=[]
    for i in tqdm(range(n)):
        it = sorted((j,v) for j,v in rows[i].items() if v)
        idx.extend(j for j,_ in it); data.extend(v for _,v in it); indptr.append(len(idx))
    return csr_matrix((np.array(data), np.array(idx), np.array(indptr)), shape=(n,n))

# Optional: truncated Neumann series if you prefer pure SpMM
def _compute_S_series(P_topo: csr_matrix, max_gens=20, min_weight=None) -> csr_matrix:
    n = P_topo.shape[0]
    S = sps.eye(n, format="csr", dtype=float)
    Pk = P_topo.copy()
    for _ in range(max_gens):
        if Pk.nnz == 0: break
        if min_weight is not None:
            m = np.abs(Pk.data) >= min_weight
            if m.sum() == 0: break
            Pk = csr_matrix((Pk.data[m], Pk.indices[m], Pk.indptr), shape=Pk.shape)
            Pk.eliminate_zeros()
        S = (S + Pk).tocsr()
        Pk = (Pk @ P_topo).tocsr()
    S.eliminate_zeros()
    return S

# ---------- batched D (two SpMM + Hadamard per generation) ----------





def _compute_D_batched(P_topo: csr_matrix, S: csr_matrix):
    n = P_topo.shape[0]
    parents = [P_topo.indices[P_topo.indptr[i]:P_topo.indptr[i+1]] for i in range(n)]
    founders = np.where(P_topo.getnnz(axis=1) == 0)[0]
    level = np.full(n, -1, int); level[founders] = 0
    for i in range(n):  # topo order
        if level[i] >= 0: continue
        ps = parents[i]
        level[i] = 1 + (0 if len(ps)==0 else max(level[p] for p in ps))
    d = np.zeros(n); F = np.zeros(n); d[founders]=1.0; F[founders]=0.0
    S_csr = S.tocsr()
    Lmax = int(level.max())
    for L in tqdm(range(1, Lmax+1)):
        kids = np.where(level==L)[0]
        if kids.size==0: continue
        two = [(i, parents[i]) for i in kids if len(parents[i])==2]
        if two:
            idx  = np.fromiter((i for i,_ in two), int)
            pidx = np.fromiter((pp[0] for _,pp in two), int)
            qidx = np.fromiter((pp[1] for _,pp in two), int)
            m = len(idx)
            Rp = csr_matrix((np.ones(m), (np.arange(m), pidx)), shape=(m, n))
            Rq = csr_matrix((np.ones(m), (np.arange(m), qidx)), shape=(m, n))
            M1 = (Rp @ S_csr) @ sps.diags(d, format="csr")
            M2 = (Rq @ S_csr)
            phi_vec = 0.5 * np.asarray(M1.multiply(M2).sum(axis=1)).ravel()
            F[idx] = phi_vec
            d[idx] = 0.5 - 0.25*(F[pidx] + F[qidx])
        singles = [i for i in kids if len(parents[i]) == 1]
        if singles:
            pidx = np.fromiter((parents[i][0] for i in singles), int)
            # Unknown parent assumed base/unrelated => F_i = 0
            F[singles] = 0.0
            # Correct Mendelian sampling variance for one known parent
            d[singles] = 0.75 - 0.25 * F[pidx]
    return d, F

# ---------- explicit Φ per block ----------
def _phi_block(P_block: csr_matrix, *, use_series=False, max_gens=20, min_weight=None):
    P_topo, ord_blk, inv_blk = _topo_order_block(P_block)
    S = _compute_S_series(P_topo, max_gens, min_weight) if use_series else _compute_S_forward(P_topo)
    d, F = _compute_D_batched(P_topo, S)
    S_csc = S.tocsc(copy=True)
    rootd = np.sqrt(d, dtype=float)
    for j in range(S_csc.shape[1]):
        if rootd[j] != 1.0:
            S_csc.data[S_csc.indptr[j]:S_csc.indptr[j+1]] *= rootd[j]
    A_topo = (S_csc @ S_csc.T).tocsr()
    Phi_topo = 0.5 * A_topo
    # unpermute block
    A_blk   = A_topo[inv_blk][:, inv_blk].tocsr()
    Phi_blk = Phi_topo[inv_blk][:, inv_blk].tocsr()
    return Phi_blk, A_blk, d[inv_blk], F[inv_blk]

def kinship_explicit(P: csr_matrix, *, use_series=False, max_gens=20, min_weight=None):
    """
    Full Φ via sparse ops only; splits by families and topo-sorts per family.
    Returns Φ (CSR), A (CSR), d, F in ORIGINAL ordering.
    """
    P = P.tocsr(copy=True); P.sum_duplicates(); P.sort_indices()
    Pp, perm, inv, spans = _permute_by_components(P)
    n = P.shape[0]
    Phi_p = sps.csr_matrix((n, n)); A_p = sps.csr_matrix((n, n))
    d_all = np.zeros(n); F_all = np.zeros(n)
    for s,e in spans:
        Pb = Pp[s:e, s:e]
        Phi_b, A_b, db, Fb = _phi_block(Pb, use_series=use_series, max_gens=max_gens, min_weight=min_weight)
        Phi_p[s:e, s:e] = Phi_b; A_p[s:e, s:e] = A_b
        d_all[s:e] = db; F_all[s:e] = Fb
    # unpermute to original
    Phi = Phi_p[inv][:, inv].tocsr()
    A   = A_p[inv][:, inv].tocsr()
    d   = d_all[inv]; F = F_all[inv]
    return Phi, A, d, F

# ---------- implicit Φ operator (for large n) ----------
class KinshipLinearOperator(LinearOperator):
    """
    y = Φ x = 0.5 * (I-P)^(-1) D (I-P)^(-T) x
    Assumes P is topo-ordered (parents before children).
    """
    def __init__(self, P_topo: csr_matrix, d: np.ndarray):
        self.P = P_topo.tocsr(); self.n = self.P.shape[0]
        self.d = d.astype(float, copy=False)
        self.L = (sps.eye(self.n, format="csr") - self.P).tocsr()   # unit-lower
        super().__init__(dtype=float, shape=(self.n, self.n))
    def _matvec(self, x):
        u = spsolve_triangular(self.L.T, np.asarray(x), lower=False, unit_diagonal=True)
        v = self.d * u
        y = spsolve_triangular(self.L, v, lower=True, unit_diagonal=True)
        return 0.5 * y
    _rmatvec = _matvec

def build_phi_operator(P: csr_matrix, *, use_series=False, max_gens=20, min_weight=None):
    """
    Returns (Phi_op, d, F, perm_to_topo, inv_topo).
    Builds d via the same batched scheme; P is returned topo-ordered for the operator.
    """
    P = P.tocsr(copy=True); P.sum_duplicates(); P.sort_indices()
    # Single block build of d,F on topo-ordered P (if you have many families, do per family and concatenate)
    P_topo, _ord, _inv = _topo_order_block(P)
    S = _compute_S_series(P_topo, max_gens, min_weight) if use_series else _compute_S_forward(P_topo)
    d, F = _compute_D_batched(P_topo, S)
    return KinshipLinearOperator(P_topo, d), d, F, _ord, _inv