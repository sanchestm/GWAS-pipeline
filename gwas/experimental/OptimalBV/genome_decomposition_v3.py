import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erfc
from tqdm import tqdm
try:
    from gwas import npplink
except ImportError:
    try:
        import npplink
    except ImportError:
        npplink = None
from glob import glob
from itertools import pairwise
import psutil
import gc
import os
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, solve_triangular

def corr_se_matrix_fisher(df):
    """Corr se matrix fisher.

Allowed inputs
--------------
df : Any
    Accepted as provided to the function.

Returns
-------
tuple
    Inferred return type(s) from the implementation."""
    r = df.corr(method='pearson')
    n = df.notna().astype(int).T @ df.notna().astype(int)
    se_z = 1 / np.sqrt(n - 3)
    se_r = (1 - r ** 2) * se_z
    se_r = se_r.where(n >= 4)
    return (r, se_r, n)

def _wrap_mvblup_output(Uhat, traits_is_df, row_index, col_index, return_numpy):
    """Wrap mvblup output.

Allowed inputs
--------------
Uhat : Any
    Accepted as provided to the function.
traits_is_df : Any
    Accepted as provided to the function.
row_index : Any
    Accepted as provided to the function.
col_index : Any
    Accepted as provided to the function.
return_numpy : Any
    Accepted as provided to the function.

Returns
-------
object | pandas.DataFrame
    Inferred return type(s) from the implementation."""
    if traits_is_df and (not return_numpy): return pd.DataFrame(Uhat, index=row_index, columns=col_index)
    return Uhat

def _invsqrt_psd(M, eps=1e-10):
    """Symmetric inverse square root of a PSD matrix.

Allowed inputs
--------------
M : Any
    Accepted as provided to the function.
eps : float
    default=1e-10.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    M = _symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return V * (1.0 / np.sqrt(w)) @ V.T

def _project_psd(M, eps=1e-10):
    """Project psd.

Allowed inputs
--------------
M : Any
    Accepted as provided to the function.
eps : float
    default=1e-10.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    M = _symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return V * w @ V.T

def _cov_to_corr(C, eps=1e-10):
    """Cov to corr.

Allowed inputs
--------------
C : Any
    Accepted as provided to the function.
eps : float
    default=1e-10.

Returns
-------
tuple
    Inferred return type(s) from the implementation."""
    d = np.sqrt(np.clip(np.diag(C), eps, None))
    R = C / np.outer(d, d)
    R = _symmetrize(R)
    return (R, d)

def _corr_to_cov(R, scale):
    """Corr to cov.

Allowed inputs
--------------
R : Any
    Accepted as provided to the function.
scale : Any
    Accepted as provided to the function.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    return _symmetrize(R * np.outer(scale, scale))

def decompose_covariance(C, rank=None, method='fa', *, standardize=False, preserve_diag=True, jitter=1e-08, maxiter=1000, tol=1e-08):
    """Low-dimensional decomposition of a covariance matrix.

Allowed inputs
--------------
C : Any
    Accepted as provided to the function.
rank : Any | None
    default=None.
method : str
    allowed values: 'eigen', 'fa', 'factor', 'factor_analysis', 'pca', 'spectral'; default='fa'.
standardize : Any
    default=False.
preserve_diag : Any
    default=True.
jitter : Any
    default=1e-08.
maxiter : Any
    default=1000.
tol : Any
    default=1e-08.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    C = _symmetrize(np.asarray(C, dtype=float))
    p = C.shape[0]
    if C.shape != (p, p):
        raise ValueError('C must be square')
    rank = p if rank is None else int(rank)
    rank = max(1, min(rank, p))
    method = method.lower()
    if method in {'factor', 'factor_analysis'}:
        method = 'fa'
    if method in {'spectral', 'pca'}:
        method = 'eigen'
    if method not in {'fa', 'eigen'}:
        raise ValueError("method must be one of {'fa','eigen','spectral','pca','factor','factor_analysis'}")
    if standardize:
        C_work, scale = _cov_to_corr(C, eps=jitter)
    else:
        C_work = C.copy()
        scale = None
    C_work = _project_psd(C_work, eps=jitter)
    if method == 'eigen':
        w, V = np.linalg.eigh(C_work)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]
        w_r = np.clip(w[:rank], 0.0, None)
        L = V[:, :rank] * np.sqrt(w_r)[None, :]
        lowrank = L @ L.T
        if preserve_diag:
            psi = np.clip(np.diag(C_work - lowrank), 0.0, None)
        else:
            psi = np.zeros(p, dtype=float)
        recon = _symmetrize(lowrank + np.diag(psi))
        success = True
        message = 'eigen decomposition'
    else:
        w0, V0 = np.linalg.eigh(C_work)
        idx = np.argsort(w0)[::-1]
        w0 = w0[idx]
        V0 = V0[:, idx]
        w_r = np.clip(w0[:rank], 0.0, None)
        L0 = V0[:, :rank] * np.sqrt(w_r)[None, :]
        psi0 = np.clip(np.diag(C_work - L0 @ L0.T), jitter, None)
        x0 = np.concatenate([L0.ravel(), np.log(psi0)])

        def unpack(x):
            L = x[:p * rank].reshape(p, rank)
            psi = np.exp(x[p * rank:])
            return (L, psi)

        def objective_and_grad(x):
            L, psi = unpack(x)
            M = L @ L.T + np.diag(psi) - C_work
            M = _symmetrize(M)
            f = 0.5 * np.sum(M * M)
            grad_L = 2.0 * M @ L
            grad_logpsi = np.diag(M) * psi
            g = np.concatenate([grad_L.ravel(), grad_logpsi])
            return (float(f), g)
        res = minimize(objective_and_grad, x0=x0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol})
        L, psi = unpack(res.x)
        psi = np.clip(psi, jitter, None)
        lowrank = L @ L.T
        recon = _symmetrize(lowrank + np.diag(psi))
        success = bool(res.success)
        message = str(res.message)
    if standardize:
        L = scale[:, None] * L
        psi = scale ** 2 * psi
        lowrank = L @ L.T
        recon = _symmetrize(lowrank + np.diag(psi))
    residual = _symmetrize(C - recon)
    tr_total = np.trace(C)
    tr_lowrank = np.trace(lowrank)
    explained_trace = tr_lowrank / tr_total if tr_total > 0 else np.nan
    fro_total = np.sum(C * C)
    fro_resid = np.sum(residual * residual)
    explained_fro = 1.0 - fro_resid / fro_total if fro_total > 0 else np.nan
    eigvals_fit = np.linalg.eigvalsh(recon)
    eigvals_fit = np.sort(eigvals_fit)[::-1]
    return {'method': method, 'rank': rank, 'loadings': L, 'specific_var': psi, 'lowrank': _symmetrize(lowrank), 'reconstructed': recon, 'residual': residual, 'explained_trace': explained_trace, 'explained_fro': explained_fro, 'success': success, 'message': message, 'eigenvalues': eigvals_fit}

def decompose_GE(G, E, rank_G=None, rank_E=None, method_G='fa', method_E='fa', *, standardize=False, preserve_diag=True, jitter=1e-08, maxiter=1000, tol=1e-08):
    """Decompose both G and E into lower-dimensional representations.

Allowed inputs
--------------
G : Any
    Accepted as provided to the function.
E : Any
    Accepted as provided to the function.
rank_G : Any | None
    default=None.
rank_E : Any | None
    default=None.
method_G : str
    default='fa'.
method_E : str
    default='fa'.
standardize : Any
    default=False.
preserve_diag : Any
    default=True.
jitter : Any
    default=1e-08.
maxiter : Any
    default=1000.
tol : Any
    default=1e-08.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    dec_G = decompose_covariance(G, rank=rank_G, method=method_G, standardize=standardize, preserve_diag=preserve_diag, jitter=jitter, maxiter=maxiter, tol=tol)
    dec_E = decompose_covariance(E, rank=rank_E, method=method_E, standardize=standardize, preserve_diag=preserve_diag, jitter=jitter, maxiter=maxiter, tol=tol)
    return {'G': dec_G, 'E': dec_E, 'G_reconstructed': dec_G['reconstructed'], 'E_reconstructed': dec_E['reconstructed']}

def _prepare_traits(traits: pd.DataFrame, dtype=np.float64):
    """Parameters.

Allowed inputs
--------------
traits : pd.DataFrame
    Accepted as provided to the function.
dtype : Any
    default=np.float64.

Returns
-------
tuple
    Inferred return type(s) from the implementation."""
    Y = traits.to_numpy(dtype=dtype, copy=True)
    obs = np.isfinite(Y)
    n_obs = obs.sum(axis=0, keepdims=True)
    if np.any(n_obs == 0):
        bad = list(traits.columns[n_obs.ravel() == 0])
        raise ValueError(f'Traits with all values missing: {bad}')
    mu = np.divide(np.nansum(Y, axis=0, keepdims=True), n_obs, out=np.zeros((1, Y.shape[1]), dtype=dtype), where=n_obs > 0)
    Y -= mu
    Y[~obs] = 0.0
    Y0 = np.asarray(Y, dtype=dtype, order='C')
    R = np.asarray(obs, dtype=dtype, order='C')
    return (Y0, R)

def _pack_lower(M):
    """Pack lower.

Allowed inputs
--------------
M : Any
    Accepted as provided to the function.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    tri = np.tril_indices(M.shape[0])
    return M[tri]

def _unpack_lower(v, T, dtype=np.float64):
    """Unpack lower.

Allowed inputs
--------------
v : Any
    Accepted as provided to the function.
T : Any
    Accepted as provided to the function.
dtype : Any
    default=np.float64.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    tri = np.tril_indices(T)
    M = np.zeros((T, T), dtype=dtype)
    M[tri] = v
    M = M + np.tril(M, -1).T
    return M

def _pack_symmetric_grad(G):
    """Gradient wrt packed lower-triangle parameters of a symmetric matrix.

Allowed inputs
--------------
G : Any
    Accepted as provided to the function.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    tri = np.tril_indices(G.shape[0])
    out = G[tri].copy()
    offdiag = tri[0] > tri[1]
    out[offdiag] *= 2.0
    return out

def _psd_factor_from_sym(A, rank=None, eps=1e-12):
    """Build F such that F F^T is the PSD projection (optionally rank-truncated).

Allowed inputs
--------------
A : Any
    Accepted as provided to the function.
rank : Any | None
    default=None.
eps : float
    default=1e-12.

Returns
-------
numpy.ndarray | object
    Inferred return type(s) from the implementation."""
    A = _symmetrize(np.asarray(A))
    w, V = np.linalg.eigh(A)
    w[w < 0] = 0.0
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    keep = w > eps
    w = w[keep]
    V = V[:, keep]
    T = A.shape[0]
    if rank is None:
        rank = T
    rank = min(rank, T)
    if w.size == 0:
        return np.zeros((T, rank), dtype=A.dtype)
    w = w[:rank]
    V = V[:, :rank]
    F = V * np.sqrt(w)[None, :]
    if F.shape[1] < rank:
        pad = np.zeros((T, rank - F.shape[1]), dtype=A.dtype)
        F = np.hstack([F, pad])
    return F

def _rg_from_G_pairwise(G, eps=1e-12, dtype=np.float64):
    """Safe rg computation for unconstrained pairwise G.

Allowed inputs
--------------
G : Any
    Accepted as provided to the function.
eps : float
    default=1e-12.
dtype : Any
    default=np.float64.

Returns
-------
tuple
    Inferred return type(s) from the implementation."""
    vg = np.diag(G).astype(dtype, copy=True)
    valid_vg = np.isfinite(vg) & (vg > eps)
    sdg = np.full_like(vg, np.nan, dtype=dtype)
    sdg[valid_vg] = np.sqrt(vg[valid_vg])
    scale = sdg[:, None] * sdg[None, :]
    Rg = np.divide(G, scale, out=np.full_like(G, np.nan, dtype=dtype), where=np.isfinite(scale) & (scale > 0))
    Rg[np.diag_indices_from(Rg)] = np.where(valid_vg, 1.0, np.nan)
    return (Rg, vg, valid_vg)

def _rg_from_G_psd(G, eps=1e-12, dtype=np.float64):
    """Safe rg computation for PSD G.

Allowed inputs
--------------
G : Any
    Accepted as provided to the function.
eps : float
    default=1e-12.
dtype : Any
    default=np.float64.

Returns
-------
tuple
    Inferred return type(s) from the implementation."""
    vg = np.diag(G).astype(dtype, copy=True)
    vg = np.where(np.isfinite(vg), vg, np.nan)
    vg_clip = np.clip(vg, 0.0, None)
    sdg = np.full_like(vg_clip, np.nan, dtype=dtype)
    finite = np.isfinite(vg_clip)
    sdg[finite] = np.sqrt(vg_clip[finite])
    scale = sdg[:, None] * sdg[None, :]
    Rg = np.divide(G, scale, out=np.full_like(G, np.nan, dtype=dtype), where=np.isfinite(scale) & (scale > 0))
    valid_vg = np.isfinite(vg) & (vg_clip > eps)
    Rg[np.diag_indices_from(Rg)] = np.where(valid_vg, 1.0, np.nan)
    return (Rg, vg, valid_vg)

def _nanstd_stack(arrays):
    """Standard deviation across a list of same-shaped arrays, ignoring NaNs.

Allowed inputs
--------------
arrays : Any
    Accepted as provided to the function.

Returns
-------
None | numpy.ndarray
    Inferred return type(s) from the implementation."""
    if len(arrays) == 0:
        return None
    X = np.stack(arrays, axis=0).astype(np.float64, copy=False)
    valid = np.isfinite(X)
    n = valid.sum(axis=0)
    mean = np.divide(np.nansum(X, axis=0), n, out=np.full(X.shape[1:], np.nan, dtype=np.float64), where=n > 0)
    sq = np.where(valid, (X - mean) ** 2, 0.0)
    var = np.divide(sq.sum(axis=0), n - 1, out=np.full(X.shape[1:], np.nan, dtype=np.float64), where=n > 1)
    return np.sqrt(var)

def grm_moment_matrices_einsum_dense(grm, traits: pd.DataFrame, *, accum_dtype=np.float64, square_dtype=None):
    """Exact dense GRM moment matrices using full-matrix np.einsum.

Allowed inputs
--------------
grm : Any
    Accepted as provided to the function.
traits : pd.DataFrame
    Accepted as provided to the function.
accum_dtype : Any
    default=np.float64.
square_dtype : Any
    default=None.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    grm = np.asarray(grm)
    n, n2 = grm.shape
    if n != n2:
        raise ValueError('grm must be square')
    if len(traits) != n:
        raise ValueError('traits and grm must have the same number of rows')
    Y0, R = _prepare_traits(traits, dtype=accum_dtype)
    diagK = np.asarray(np.diag(grm), dtype=accum_dtype)
    K_use = np.asarray(grm, dtype=accum_dtype, order='C')
    if square_dtype is None:
        square_dtype = accum_dtype
    K2 = np.asarray(grm, dtype=square_dtype, order='C').copy()
    np.square(K2, out=K2)
    A11 = np.einsum('it,ij,js->ts', R, K2, R, optimize=True)
    A12 = np.einsum('it,i,is->ts', R, diagK, R, optimize=True)
    A22 = np.einsum('it,is->ts', R, R, optimize=True)
    B1 = np.einsum('it,ij,js->ts', Y0, K_use, Y0, optimize=True)
    B2 = np.einsum('it,is->ts', Y0, Y0, optimize=True)
    A11 = _symmetrize(A11)
    A12 = _symmetrize(A12)
    A22 = _symmetrize(A22)
    B1 = _symmetrize(B1)
    B2 = _symmetrize(B2)
    return {'A11': A11, 'A12': A12, 'A22': A22, 'B1': B1, 'B2': B2, 'Y0': Y0, 'R': R}

def grm_moment_matrices_einsum_blocked(grm, traits: pd.DataFrame, *, block_size=1024, block_dtype=np.float32, accum_dtype=np.float64):
    """Exact GRM moment matrices using blocked np.einsum.

Allowed inputs
--------------
grm : Any
    Accepted as provided to the function.
traits : pd.DataFrame
    Accepted as provided to the function.
block_size : Any
    default=1024.
block_dtype : Any
    default=np.float32.
accum_dtype : Any
    default=np.float64.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    grm = np.asarray(grm)
    n, n2 = grm.shape
    if n != n2:
        raise ValueError('grm must be square')
    if len(traits) != n:
        raise ValueError('traits and grm must have the same number of rows')
    Y0, R = _prepare_traits(traits, dtype=accum_dtype)
    t = Y0.shape[1]
    A11 = np.zeros((t, t), dtype=accum_dtype)
    B1 = np.zeros((t, t), dtype=accum_dtype)
    diagK = np.asarray(np.diag(grm), dtype=accum_dtype)
    A12 = np.einsum('it,i,is->ts', R, diagK, R, optimize=True)
    A22 = np.einsum('it,is->ts', R, R, optimize=True)
    B2 = np.einsum('it,is->ts', Y0, Y0, optimize=True)
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        sl = slice(i0, i1)
        Kb = np.array(grm[sl, :], dtype=block_dtype, order='C', copy=True)
        B1 += np.einsum('it,ij,js->ts', Y0[sl], Kb, Y0, optimize=True)
        np.square(Kb, out=Kb)
        A11 += np.einsum('it,ij,js->ts', R[sl], Kb, R, optimize=True)
    A11 = _symmetrize(A11)
    A12 = _symmetrize(A12)
    A22 = _symmetrize(A22)
    B1 = _symmetrize(B1)
    B2 = _symmetrize(B2)
    return {'A11': A11, 'A12': A12, 'A22': A22, 'B1': B1, 'B2': B2, 'Y0': Y0, 'R': R}

def grm_moment_matrices_einsum(grm, traits: pd.DataFrame, *, mode='auto', block_size=1024, block_dtype=np.float32, accum_dtype=np.float64, square_dtype=None, dense_threshold_gb=4.0):
    """Dispatch to dense or blocked exact moment computation.

Allowed inputs
--------------
grm : Any
    Accepted as provided to the function.
traits : pd.DataFrame
    Accepted as provided to the function.
mode : Any
    allowed values: 'auto', 'blocked', 'dense'; default='auto'.
block_size : Any
    default=1024.
block_dtype : Any
    default=np.float32.
accum_dtype : Any
    default=np.float64.
square_dtype : Any
    default=None.
dense_threshold_gb : Any
    default=4.0.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    grm = np.asarray(grm)
    n = grm.shape[0]
    if mode not in {'auto', 'dense', 'blocked'}:
        raise ValueError("mode must be 'auto', 'dense', or 'blocked'")
    if mode == 'auto':
        sdtype = accum_dtype if square_dtype is None else square_dtype
        est_gb = n * n * np.dtype(sdtype).itemsize / 1024 ** 3
        mode = 'dense' if est_gb <= dense_threshold_gb else 'blocked'
    if mode == 'dense':
        return grm_moment_matrices_einsum_dense(grm, traits, accum_dtype=accum_dtype, square_dtype=square_dtype)
    return grm_moment_matrices_einsum_blocked(grm, traits, block_size=block_size, block_dtype=block_dtype, accum_dtype=accum_dtype)

def _bootstrap_standard_errors(fit_func, grm, traits, *, n_boot=100, random_state=None):
    """Animal bootstrap. Resamples rows/cols of the GRM and rows of the trait table.

Allowed inputs
--------------
fit_func : Any
    Accepted as provided to the function.
grm : Any
    Accepted as provided to the function.
traits : Any
    Accepted as provided to the function.
n_boot : Any
    default=100.
random_state : Any
    default=None.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    if n_boot <= 0:
        return {'G_se': None, 'E_se': None, 'rg_se': None, 'n_boot': 0, 'n_boot_success': 0}
    rng = np.random.default_rng(random_state)
    n = len(traits)
    G_samples = []
    E_samples = []
    Rg_samples = []
    for _ in tqdm(range(n_boot), desc='bootstrapping', total=n_boot):
        idx = rng.integers(0, n, size=n)
        grm_b = np.asarray(grm)[np.ix_(idx, idx)]
        traits_b = traits.iloc[idx].reset_index(drop=True)
        try:
            out_b = fit_func(grm_b, traits_b)
            G_b = np.asarray(out_b['G'], dtype=np.float64)
            E_b = np.asarray(out_b['E'], dtype=np.float64)
            Rg_b = np.asarray(out_b['Rg'], dtype=np.float64)
            G_samples.append(G_b)
            E_samples.append(E_b)
            Rg_samples.append(Rg_b)
        except Exception:
            continue
    return {'G_se': _nanstd_stack(G_samples), 'E_se': _nanstd_stack(E_samples), 'rg_se': _nanstd_stack(Rg_samples), 'n_boot': int(n_boot), 'n_boot_success': int(len(G_samples))}

def score_grm_pairwise_einsum(grm, traits: pd.DataFrame, *, mode='auto', block_size=1024, block_dtype=np.float32, accum_dtype=np.float64, square_dtype=None, dense_threshold_gb=4.0, eps=1e-12, n_boot=0, random_state=None):
    """Pairwise-separable multivariate method-of-moments estimator from one GRM.

Allowed inputs
--------------
grm : Any
    Accepted as provided to the function.
traits : pd.DataFrame
    Accepted as provided to the function.
mode : Any
    default='auto'.
block_size : Any
    default=1024.
block_dtype : Any
    default=np.float32.
accum_dtype : Any
    default=np.float64.
square_dtype : Any
    default=None.
dense_threshold_gb : Any
    default=4.0.
eps : Any
    default=1e-12.
n_boot : Any
    default=0.
random_state : Any
    default=None.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    mom = grm_moment_matrices_einsum(grm, traits, mode=mode, block_size=block_size, block_dtype=block_dtype, accum_dtype=accum_dtype, square_dtype=square_dtype, dense_threshold_gb=dense_threshold_gb)
    A11, A12, A22 = (mom['A11'], mom['A12'], mom['A22'])
    B1, B2 = (mom['B1'], mom['B2'])
    den = A11 * A22 - A12 * A12
    G = np.full_like(B1, np.nan, dtype=accum_dtype)
    E = np.full_like(B2, np.nan, dtype=accum_dtype)
    ok = np.abs(den) > eps
    G[ok] = (B1[ok] * A22[ok] - B2[ok] * A12[ok]) / den[ok]
    E[ok] = (A11[ok] * B2[ok] - A12[ok] * B1[ok]) / den[ok]
    no_overlap = (A22 <= eps) & (np.abs(A12) <= eps) & (A11 > eps)
    G[no_overlap] = B1[no_overlap] / A11[no_overlap]
    E[no_overlap] = 0.0
    G = _symmetrize(G)
    E = _symmetrize(E)
    Rg, vg, valid_vg = _rg_from_G_pairwise(G, eps=eps, dtype=accum_dtype)
    out = {'G': G, 'E': E, 'Rg': Rg, 'vg': vg, 'valid_vg': valid_vg, 'invalid_var_mask': ~valid_vg, 'n_invalid_variances': int((~valid_vg).sum()), 'invalid_traits': list(traits.columns[~valid_vg]), 'G_se': None, 'E_se': None, 'rg_se': None, 'n_boot': 0, 'n_boot_success': 0, **mom}
    if n_boot > 0:
        boot = _bootstrap_standard_errors(lambda Kb, Tb: score_grm_pairwise_einsum(Kb, Tb, mode=mode, block_size=block_size, block_dtype=block_dtype, accum_dtype=accum_dtype, square_dtype=square_dtype, dense_threshold_gb=dense_threshold_gb, eps=eps, n_boot=0), grm, traits, n_boot=n_boot, random_state=random_state)
        out.update(boot)
    return out

def genetic_varcov_PSD(grm, traits: pd.DataFrame, *, mode='auto', block_size=1024, block_dtype=np.float32, accum_dtype=np.float64, square_dtype=None, dense_threshold_gb=4.0, rank_G=None, rank_E=None, psd_E=True, ridge_G=0.0, ridge_E=0.0, method='L-BFGS-B', maxiter=1000, gtol=1e-08, ftol=1e-12, init='pairwise', eps=1e-12, n_boot=0, random_state=None):
    """PSD-constrained multivariate method-of-moments estimator using scipy.optimize.

Allowed inputs
--------------
grm : Any
    Accepted as provided to the function.
traits : pd.DataFrame
    Accepted as provided to the function.
mode : Any
    default='auto'.
block_size : Any
    default=1024.
block_dtype : Any
    default=np.float32.
accum_dtype : Any
    default=np.float64.
square_dtype : Any
    default=None.
dense_threshold_gb : Any
    default=4.0.
rank_G : Any
    default=None.
rank_E : Any
    default=None.
psd_E : Any
    default=True.
ridge_G : Any
    default=0.0.
ridge_E : Any
    default=0.0.
method : Any
    default='L-BFGS-B'.
maxiter : Any
    default=1000.
gtol : Any
    default=1e-08.
ftol : Any
    default=1e-12.
init : Any
    default='pairwise'.
eps : Any
    default=1e-12.
n_boot : Any
    default=0.
random_state : Any
    default=None.

Returns
-------
pandas.Series
    Inferred return type(s) from the implementation."""
    n_traits = len(traits.columns)
    pw = score_grm_pairwise_einsum(grm, traits, mode=mode, block_size=block_size, block_dtype=block_dtype, accum_dtype=accum_dtype, square_dtype=square_dtype, dense_threshold_gb=dense_threshold_gb, eps=eps, n_boot=0)
    A11, A12, A22 = (pw['A11'], pw['A12'], pw['A22'])
    B1, B2 = (pw['B1'], pw['B2'])
    T = A11.shape[0]
    if rank_G is None:
        rank_G = T
    if rank_E is None:
        rank_E = T
    if init == 'pairwise':
        G0 = np.nan_to_num(pw['G'], nan=0.0)
        E0 = np.nan_to_num(pw['E'], nan=0.0)
    elif init == 'zero':
        G0 = np.zeros((T, T), dtype=accum_dtype)
        E0 = np.zeros((T, T), dtype=accum_dtype)
    else:
        raise ValueError("init must be 'pairwise' or 'zero'")
    Fg0 = _psd_factor_from_sym(G0, rank=rank_G)
    if psd_E:
        Fe0 = _psd_factor_from_sym(E0, rank=rank_E)
        x0 = np.concatenate([Fg0.ravel(), Fe0.ravel()])
    else:
        E0 = _symmetrize(E0)
        x0 = np.concatenate([Fg0.ravel(), _pack_lower(E0)])
    nFg = T * rank_G

    def unpack(x):
        Fg = x[:nFg].reshape(T, rank_G)
        G = _symmetrize(Fg @ Fg.T)
        if psd_E:
            Fe = x[nFg:].reshape(T, rank_E)
            E = _symmetrize(Fe @ Fe.T)
            return (Fg, G, E, Fe)
        E = _symmetrize(_unpack_lower(x[nFg:], T, dtype=accum_dtype))
        return (Fg, G, E, None)

    def objective_and_grad(x):
        Fg, G, E, Fe = unpack(x)
        MG = _symmetrize(A11 * G + A12 * E - B1 + ridge_G * G)
        ME = _symmetrize(A22 * E + A12 * G - B2 + ridge_E * E)
        f = np.sum(A11 * (G * G)) + 2.0 * np.sum(A12 * (G * E)) + np.sum(A22 * (E * E)) - 2.0 * np.sum(B1 * G) - 2.0 * np.sum(B2 * E) + ridge_G * np.sum(G * G) + ridge_E * np.sum(E * E)
        grad_Fg = 4.0 * (MG @ Fg)
        if psd_E:
            grad_Fe = 4.0 * (ME @ Fe)
            grad = np.concatenate([grad_Fg.ravel(), grad_Fe.ravel()])
        else:
            grad_E = _pack_symmetric_grad(2.0 * ME)
            grad = np.concatenate([grad_Fg.ravel(), grad_E])
        return (float(f), grad)
    options = {'maxiter': maxiter}
    if method.upper() in {'L-BFGS-B', 'BFGS', 'CG'}:
        options['gtol'] = gtol
    if method.upper() in {'L-BFGS-B', 'TNC', 'SLSQP'}:
        options['ftol'] = ftol
    res = minimize(objective_and_grad, x0=x0, jac=True, method=method, options=options)
    success = bool(res.success)
    message = str(res.message)
    if np.any(~np.isfinite(res.x)):
        success = False
        message = f'{message}; non-finite optimizer parameters returned'
    try:
        _, G, E, _ = unpack(res.x)
    except Exception as exc:
        G = np.full((T, T), np.nan, dtype=accum_dtype)
        E = np.full((T, T), np.nan, dtype=accum_dtype)
        success = False
        message = f'{message}; unpack failed: {exc}'
    if np.any(~np.isfinite(G)):
        success = False
        message = f'{message}; estimated G contains non-finite values'
    if np.any(~np.isfinite(E)):
        success = False
        message = f'{message}; estimated E contains non-finite values'
    Rg, vg, valid_vg = _rg_from_G_psd(G, eps=eps, dtype=accum_dtype)
    out = pd.Series({'G': pd.DataFrame(G, columns=traits.columns, index=traits.columns), 
                     'E': pd.DataFrame(E, columns=traits.columns, index=traits.columns), 
                     'Rg': pd.DataFrame(Rg, columns=traits.columns, index=traits.columns), 
                     'vg': pd.Series(data=vg, index=traits.columns), 
                     'valid_vg': valid_vg, 'invalid_var_mask': ~valid_vg, 
                     'n_invalid_variances': int((~valid_vg).sum()), 
                     'invalid_traits': list(traits.columns[~valid_vg]), 
                     'G_se': pd.DataFrame(np.full((n_traits, n_traits), np.nan), columns=traits.columns, index=traits.columns), 
                     'E_se': pd.DataFrame(np.full((n_traits, n_traits), np.nan), columns=traits.columns, index=traits.columns), 
                     'rg_se': pd.DataFrame(np.full((n_traits, n_traits), np.nan), columns=traits.columns, index=traits.columns), 
                     'n_boot': 0, 'n_boot_success': 0, 'result': res, 'success': success, 'message': message, 
                     'nit': getattr(res, 'nit', None), 'fun': getattr(res, 'fun', np.nan), 'rank_G': rank_G, 
                     'rank_E': rank_E if psd_E else None, 'psd_E': psd_E, 'ridge_G': ridge_G, 'ridge_E': ridge_E})
    if n_boot > 0:
        boot = _bootstrap_standard_errors(lambda Kb, Tb: genetic_varcov_PSD(Kb, Tb, mode=mode, block_size=block_size, block_dtype=block_dtype, accum_dtype=accum_dtype, square_dtype=square_dtype, dense_threshold_gb=dense_threshold_gb, rank_G=rank_G, rank_E=rank_E, psd_E=psd_E, ridge_G=ridge_G, ridge_E=ridge_E, method=method, maxiter=maxiter, gtol=gtol, ftol=ftol, init=init, eps=eps, n_boot=0), grm, traits, n_boot=n_boot, random_state=random_state)
        out.update(boot)
        for i in ['G_se', 'E_se', 'rg_se']:
            out[i] = pd.DataFrame(out[i], columns=traits.columns, index=traits.columns)
    return out

def _symmetrize(M):
    """Symmetrize.

Allowed inputs
--------------
M : Any
    Accepted as provided to the function.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    return 0.5 * (M + M.T)

def _solve_spd(A, b, jitter=1e-08, max_tries=6):
    """Solve A x = b for symmetric PSD/SPD A, adding diagonal jitter if needed.

Allowed inputs
--------------
A : Any
    Accepted as provided to the function.
b : Any
    Accepted as provided to the function.
jitter : float
    default=1e-08.
max_tries : int
    default=6.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    A = _symmetrize(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float)
    eye = np.eye(A.shape[0], dtype=A.dtype)
    cur_jitter = float(jitter)
    for _ in range(max_tries):
        try:
            c, lower = cho_factor(A + cur_jitter * eye, lower=True, check_finite=False)
            return cho_solve((c, lower), b, check_finite=False)
        except np.linalg.LinAlgError:
            cur_jitter *= 10.0
    return np.linalg.pinv(A) @ b

def _diag_cov(M):
    """Diag cov.

Allowed inputs
--------------
M : Any
    Accepted as provided to the function.

Returns
-------
numpy.ndarray
    Inferred return type(s) from the implementation."""
    M = np.asarray(M)
    return np.diag(np.diag(M))

def _normalize_grm_spec(spec, n=None, dtype=np.float64):
    """Return a dict with keys:.

Allowed inputs
--------------
spec : Any
    Accepted as provided to the function.
n : Any | None
    default=None.
dtype : Any
    default=np.float64.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    if isinstance(spec, dict):
        typ = spec['type'].lower()
        if typ == 'eigh':
            lam = np.asarray(spec['values'], dtype=dtype).reshape(-1)
            Q = np.asarray(spec['vectors'], dtype=dtype)
            if Q.shape[0] != Q.shape[1]:
                raise ValueError('GRM eigenvectors must be square')
            if len(lam) != Q.shape[0]:
                raise ValueError('GRM eigenvalues and eigenvectors shape mismatch')
            lam = np.clip(lam, 0.0, None)
            if n is not None and Q.shape[0] != n:
                raise ValueError('GRM eigendecomposition size does not match number of animals')
            return {'type': 'eigh', 'Q': Q, 'lam': lam, 'K': None}
        if typ == 'full':
            K = _symmetrize(np.asarray(spec['matrix'], dtype=dtype))
            if K.shape[0] != K.shape[1]:
                raise ValueError('GRM must be square')
            if n is not None and K.shape[0] != n:
                raise ValueError('GRM size does not match number of animals')
            lam, Q = np.linalg.eigh(K)
            return {'type': 'full', 'Q': Q, 'lam': np.clip(lam, 0.0, None), 'K': K}
        raise ValueError(f'Unknown GRM spec type: {typ}')
    if isinstance(spec, tuple):
        typ = spec[0].lower()
        if typ == 'eigh':
            lam = np.asarray(spec[1], dtype=dtype).reshape(-1)
            Q = np.asarray(spec[2], dtype=dtype)
            if Q.shape[0] != Q.shape[1]:
                raise ValueError('GRM eigenvectors must be square')
            if len(lam) != Q.shape[0]:
                raise ValueError('GRM eigenvalues and eigenvectors shape mismatch')
            lam = np.clip(lam, 0.0, None)
            if n is not None and Q.shape[0] != n:
                raise ValueError('GRM eigendecomposition size does not match number of animals')
            return {'type': 'eigh', 'Q': Q, 'lam': lam, 'K': None}
        if typ == 'full':
            K = _symmetrize(np.asarray(spec[1], dtype=dtype))
            if K.shape[0] != K.shape[1]:
                raise ValueError('GRM must be square')
            if n is not None and K.shape[0] != n:
                raise ValueError('GRM size does not match number of animals')
            lam, Q = np.linalg.eigh(K)
            return {'type': 'full', 'Q': Q, 'lam': np.clip(lam, 0.0, None), 'K': K}
        raise ValueError(f'Unknown GRM tuple type: {typ}')
    K = _symmetrize(np.asarray(spec, dtype=dtype))
    if K.shape[0] != K.shape[1]:
        raise ValueError('GRM must be square')
    if n is not None and K.shape[0] != n:
        raise ValueError('GRM size does not match number of animals')
    lam, Q = np.linalg.eigh(K)
    return {'type': 'full', 'Q': Q, 'lam': np.clip(lam, 0.0, None), 'K': K}

def _reconstruct_grm(grm_spec, dtype=np.float64):
    """Reconstruct grm.

Allowed inputs
--------------
grm_spec : Any
    Accepted as provided to the function.
dtype : Any
    default=np.float64.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    if grm_spec['K'] is not None:
        return grm_spec['K']
    Q = grm_spec['Q']
    lam = grm_spec['lam']
    return _symmetrize(Q * lam @ Q.T)

def _normalize_traitcov_spec(spec, size, dtype=np.float64, diag_only=False, allow_none=False):
    """Return a normalized trait covariance spec dict with keys:.

Allowed inputs
--------------
spec : Any
    Accepted as provided to the function.
size : Any
    Accepted as provided to the function.
dtype : Any
    default=np.float64.
diag_only : bool
    default=False.
allow_none : bool
    default=False.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    if spec is None:
        if not allow_none:
            return {'mode': 'diag', 'diag': np.zeros(size, dtype=dtype), 'loadings': None, 'full': None}
        return {'mode': 'diag', 'diag': np.zeros(size, dtype=dtype), 'loadings': None, 'full': None}
    if isinstance(spec, dict):
        typ = spec['type'].lower()
        if typ == 'diag':
            d = np.asarray(spec['diag'], dtype=dtype).reshape(-1)
            if len(d) != size:
                raise ValueError('Diagonal covariance length mismatch')
            return {'mode': 'diag', 'diag': d, 'loadings': None, 'full': None}
        if typ == 'factor':
            L = np.asarray(spec['loadings'], dtype=dtype)
            psi = np.asarray(spec.get('diag', np.zeros(size)), dtype=dtype).reshape(-1)
            if L.shape[0] != size or len(psi) != size:
                raise ValueError('Factor covariance shape mismatch')
            if diag_only:
                d = np.einsum('ij,ij->i', L, L) + psi
                return {'mode': 'diag', 'diag': d, 'loadings': None, 'full': None}
            return {'mode': 'factor', 'diag': psi, 'loadings': L, 'full': None}
        if typ == 'eigh':
            w = np.asarray(spec['values'], dtype=dtype).reshape(-1)
            Q = np.asarray(spec['vectors'], dtype=dtype)
            if Q.shape[0] != size or Q.shape[1] != len(w):
                raise ValueError('Eigh covariance shape mismatch')
            w = np.clip(w, 0.0, None)
            L = Q * np.sqrt(w)[None, :]
            if diag_only:
                d = np.einsum('ij,ij->i', L, L)
                return {'mode': 'diag', 'diag': d, 'loadings': None, 'full': None}
            return {'mode': 'factor', 'diag': np.zeros(size, dtype=dtype), 'loadings': L, 'full': None}
        if typ == 'full':
            M = _symmetrize(np.asarray(spec['matrix'], dtype=dtype))
            if M.shape != (size, size):
                raise ValueError('Full covariance shape mismatch')
            if diag_only:
                return {'mode': 'diag', 'diag': np.diag(M), 'loadings': None, 'full': None}
            return {'mode': 'full', 'diag': None, 'loadings': None, 'full': M}
        raise ValueError(f'Unknown covariance spec type: {typ}')
    if isinstance(spec, tuple):
        typ = spec[0].lower()
        if typ == 'diag':
            d = np.asarray(spec[1], dtype=dtype).reshape(-1)
            if len(d) != size:
                raise ValueError('Diagonal covariance length mismatch')
            return {'mode': 'diag', 'diag': d, 'loadings': None, 'full': None}
        if typ == 'factor':
            L = np.asarray(spec[1], dtype=dtype)
            psi = np.asarray(spec[2], dtype=dtype).reshape(-1)
            if L.shape[0] != size or len(psi) != size:
                raise ValueError('Factor covariance shape mismatch')
            if diag_only:
                d = np.einsum('ij,ij->i', L, L) + psi
                return {'mode': 'diag', 'diag': d, 'loadings': None, 'full': None}
            return {'mode': 'factor', 'diag': psi, 'loadings': L, 'full': None}
        if typ == 'eigh':
            w = np.asarray(spec[1], dtype=dtype).reshape(-1)
            Q = np.asarray(spec[2], dtype=dtype)
            if Q.shape[0] != size or Q.shape[1] != len(w):
                raise ValueError('Eigh covariance shape mismatch')
            w = np.clip(w, 0.0, None)
            L = Q * np.sqrt(w)[None, :]
            if diag_only:
                d = np.einsum('ij,ij->i', L, L)
                return {'mode': 'diag', 'diag': d, 'loadings': None, 'full': None}
            return {'mode': 'factor', 'diag': np.zeros(size, dtype=dtype), 'loadings': L, 'full': None}
        if typ == 'full':
            M = _symmetrize(np.asarray(spec[1], dtype=dtype))
            if M.shape != (size, size):
                raise ValueError('Full covariance shape mismatch')
            if diag_only:
                return {'mode': 'diag', 'diag': np.diag(M), 'loadings': None, 'full': None}
            return {'mode': 'full', 'diag': None, 'loadings': None, 'full': M}
        raise ValueError(f'Unknown covariance tuple type: {typ}')
    arr = np.asarray(spec, dtype=dtype)
    if arr.ndim == 1:
        if len(arr) != size:
            raise ValueError('Diagonal covariance length mismatch')
        return {'mode': 'diag', 'diag': arr.copy(), 'loadings': None, 'full': None}
    M = _symmetrize(arr)
    if M.shape != (size, size):
        raise ValueError('Full covariance shape mismatch')
    if diag_only:
        return {'mode': 'diag', 'diag': np.diag(M), 'loadings': None, 'full': None}
    return {'mode': 'full', 'diag': None, 'loadings': None, 'full': M}

def _traitcov_to_full(spec):
    """Traitcov to full.

Allowed inputs
--------------
spec : Any
    Accepted as provided to the function.

Returns
-------
numpy.ndarray | object
    Inferred return type(s) from the implementation."""
    if spec['mode'] == 'full':
        return spec['full']
    if spec['mode'] == 'diag':
        return np.diag(spec['diag'])
    return _symmetrize(spec['loadings'] @ spec['loadings'].T + np.diag(spec['diag']))

def _traitcov_diag(spec):
    """Traitcov diag.

Allowed inputs
--------------
spec : Any
    Accepted as provided to the function.

Returns
-------
numpy.ndarray | object
    Inferred return type(s) from the implementation."""
    if spec['mode'] == 'diag':
        return spec['diag']
    if spec['mode'] == 'factor':
        return np.einsum('ij,ij->i', spec['loadings'], spec['loadings']) + spec['diag']
    return np.diag(spec['full'])

def _apply_traitcov(spec, x):
    """Compute C @ x without necessarily materializing C.

Allowed inputs
--------------
spec : Any
    Accepted as provided to the function.
x : Any
    Accepted as provided to the function.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    x = np.asarray(x, dtype=float)
    if spec['mode'] == 'diag':
        return spec['diag'] * x
    if spec['mode'] == 'factor':
        L = spec['loadings']
        return spec['diag'] * x + L @ (L.T @ x)
    return spec['full'] @ x

def _can_use_factor_speed(G_spec, E_spec):
    """True if both are diagonal/factor representations so Woodbury / low-rank.

Allowed inputs
--------------
G_spec : Any
    Accepted as provided to the function.
E_spec : Any
    Accepted as provided to the function.

Returns
-------
bool
    Inferred return type(s) from the implementation."""
    return G_spec['mode'] in {'diag', 'factor'} and E_spec['mode'] in {'diag', 'factor'}

def _row_diag_lowrank(li, G_spec, E_spec):
    """Build A = li*G + E in low-rank+diag form:.

Allowed inputs
--------------
li : Any
    Accepted as provided to the function.
G_spec : Any
    Accepted as provided to the function.
E_spec : Any
    Accepted as provided to the function.

Returns
-------
tuple
    Inferred return type(s) from the implementation."""
    d = li * _traitcov_diag(G_spec) + _traitcov_diag(E_spec)
    U_parts = []
    if G_spec['mode'] == 'factor' and G_spec['loadings'] is not None and (G_spec['loadings'].shape[1] > 0) and (li > 0):
        U_parts.append(np.sqrt(li) * G_spec['loadings'])
    if E_spec['mode'] == 'factor' and E_spec['loadings'] is not None and (E_spec['loadings'].shape[1] > 0):
        U_parts.append(E_spec['loadings'])
    if len(U_parts) == 0:
        U = np.zeros((len(d), 0), dtype=float)
    else:
        U = np.concatenate(U_parts, axis=1)
    return (d, U)

def _solve_diag_plus_lowrank(d, U, b, jitter=1e-08):
    """Solve (diag(d) + U U^T) x = b using Woodbury.

Allowed inputs
--------------
d : Any
    Accepted as provided to the function.
U : Any
    Accepted as provided to the function.
b : Any
    Accepted as provided to the function.
jitter : float
    default=1e-08.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    d = np.asarray(d, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.clip(d, jitter, None)
    if U.shape[1] == 0:
        return b / d
    DinvU = U / d[:, None]
    M = np.eye(U.shape[1], dtype=float) + U.T @ DinvU
    rhs = U.T @ (b / d)
    tmp = _solve_spd(M, rhs, jitter=jitter)
    return b / d - DinvU @ tmp

def _cholupdate_lower(L, x):
    """Rank-1 update of lower-triangular Cholesky factor:.

Allowed inputs
--------------
L : Any
    Accepted as provided to the function.
x : Any
    Accepted as provided to the function.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    x = x.copy().astype(float)
    n = len(x)
    for k in range(n):
        r = np.hypot(L[k, k], x[k])
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k + 1:, k] = (L[k + 1:, k] + s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]
    return L

def _chol_diag_plus_lowrank(d, U, jitter=1e-08):
    """Compute lower Cholesky L of A = diag(d) + U U^T using rank-1 updates.

Allowed inputs
--------------
d : Any
    Accepted as provided to the function.
U : Any
    Accepted as provided to the function.
jitter : float
    default=1e-08.

Returns
-------
numpy.ndarray
    Inferred return type(s) from the implementation."""
    d = np.clip(np.asarray(d, dtype=float), jitter, None)
    L = np.diag(np.sqrt(d))
    for j in range(U.shape[1]):
        _cholupdate_lower(L, U[:, j])
    return L

def _prepare_mvblup_inputs(traits, GRM, G, E=None, center=True, dtype=np.float64, diag_only=False):
    """Prepare mvblup inputs.

Allowed inputs
--------------
traits : Any
    Accepted as provided to the function.
GRM : Any
    Accepted as provided to the function.
G : Any
    Accepted as provided to the function.
E : Any | None
    default=None.
center : bool
    default=True.
dtype : Any
    default=np.float64.
diag_only : bool
    default=False.

Returns
-------
dict
    Inferred return type(s) from the implementation."""
    traits_is_df = isinstance(traits, pd.DataFrame)
    if traits_is_df:
        row_index = traits.index
        col_index = traits.columns
        Y = traits.to_numpy(dtype=dtype, copy=True)
    else:
        Y = np.asarray(traits, dtype=dtype)
        if Y.ndim != 2:
            raise ValueError('traits must be a 2D array or DataFrame')
        row_index = None
        col_index = None
    n, t = Y.shape
    grm_spec = _normalize_grm_spec(GRM, n=n, dtype=dtype)
    Q, lam, K_full = (grm_spec['Q'], grm_spec['lam'], grm_spec['K'])
    G_spec = _normalize_traitcov_spec(G, size=t, dtype=dtype, diag_only=diag_only)
    E_spec = _normalize_traitcov_spec(E, size=t, dtype=dtype, diag_only=diag_only, allow_none=True)
    means = np.zeros(t, dtype=dtype)
    if center:
        means = np.nanmean(Y, axis=0)
        Y = Y - means[None, :]
    return {'Y': Y, 'Q': Q, 'lam': lam, 'K': K_full, 'G_spec': G_spec, 'E_spec': E_spec, 'means': means, 'traits_is_df': traits_is_df, 'row_index': row_index, 'col_index': col_index, 'n': n, 't': t, 'diag_only': diag_only}

def _wrap_mv_output(M, traits_is_df, row_index, col_index, return_numpy):
    """Wrap mv output.

Allowed inputs
--------------
M : Any
    Accepted as provided to the function.
traits_is_df : Any
    Accepted as provided to the function.
row_index : Any
    Accepted as provided to the function.
col_index : Any
    Accepted as provided to the function.
return_numpy : Any
    Accepted as provided to the function.

Returns
-------
object | pandas.DataFrame
    Inferred return type(s) from the implementation."""
    if traits_is_df and (not return_numpy):
        return pd.DataFrame(M, index=row_index, columns=col_index)
    return M

def mvBLUP_complete(traits, GRM, G, E=None, eigen: bool=True, center: bool=True, dtype=np.float64, jitter: float=1e-08, diag_only: bool=False, return_numpy: bool=False, return_aux: bool=False):
    """Complete-data BLUP.

Allowed inputs
--------------
traits : Any
    Accepted as provided to the function.
GRM : Any
    Accepted as provided to the function.
G : Any
    Accepted as provided to the function.
E : Any | None
    default=None.
eigen : bool
    default=True.
center : bool
    default=True.
dtype : Any
    default=np.float64.
jitter : float
    default=1e-08.
diag_only : bool
    default=False.
return_numpy : bool
    default=False.
return_aux : bool
    default=False.

Returns
-------
object
    Inferred return type(s) from the implementation."""
    inp = _prepare_mvblup_inputs(traits, GRM, G, E=E, center=center, dtype=dtype, diag_only=diag_only)
    Y, Q, lam = (inp['Y'], inp['Q'], inp['lam'])
    if np.isnan(Y).any():
        raise ValueError('mvBLUP_complete does not accept NaN. Use mvBLUP(..., missing=...).')
    n, t = Y.shape
    G_spec, E_spec = (inp['G_spec'], inp['E_spec'])
    if np.all(_traitcov_diag(E_spec) == 0) and E_spec['mode'] == 'diag':
        Uhat = Y.copy()
        aux = {'method': 'complete_no_residual', 'means': inp['means'], 'diag_only': diag_only}
        Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
        return (Uhat, aux) if return_aux else Uhat
    if diag_only:
        vg = _traitcov_diag(G_spec)
        ve = _traitcov_diag(E_spec)
        Y_tilde = Q.T @ Y
        denom = lam[:, None] * vg[None, :] + ve[None, :]
        shrink = np.divide(lam[:, None] * vg[None, :], denom, out=np.zeros_like(denom), where=denom > 0)
        Uhat = Q @ (shrink * Y_tilde)
        aux = {'method': 'complete_diag_eigen_fast', 'means': inp['means'], 'diag_only': True}
        Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
        return (Uhat, aux) if return_aux else Uhat
    Y_tilde = Q.T @ Y
    U_tilde = np.empty_like(Y_tilde)
    use_factor_speed = _can_use_factor_speed(G_spec, E_spec)
    if use_factor_speed:
        for i in range(n):
            li = lam[i]
            d, U = _row_diag_lowrank(li, G_spec, E_spec)
            z = _solve_diag_plus_lowrank(d, U, Y_tilde[i, :], jitter=jitter)
            U_tilde[i, :] = li * _apply_traitcov(G_spec, z)
        method = 'complete_factor_woodbury'
    else:
        Gf = _traitcov_to_full(G_spec)
        Ef = _traitcov_to_full(E_spec)
        for i in range(n):
            li = lam[i]
            A = li * Gf + Ef
            z = _solve_spd(A, Y_tilde[i, :], jitter=jitter)
            U_tilde[i, :] = li * (Gf @ z)
        method = 'complete_full_dense'
    Uhat = Q @ U_tilde
    aux = {'method': method, 'means': inp['means'], 'diag_only': diag_only}
    Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
    return (Uhat, aux) if return_aux else Uhat

def mvBLUP(traits, GRM, G, E=None, eigen: bool=True, missing: str='exact', center: bool=True, dtype=np.float64, jitter: float=1e-08, init: str='blup0', max_iter: int=50, tol: float=1e-06, relaxation: float=0.7, diag_only: bool=False, return_numpy: bool=False, return_aux: bool=False):
    """BLUP with missing-data handling.

Allowed inputs
--------------
traits : Any
    Accepted as provided to the function.
GRM : Any
    Accepted as provided to the function.
G : Any
    Accepted as provided to the function.
E : Any | None
    default=None.
eigen : bool
    default=True.
missing : str
    allowed values: 'exact', 'iterative'; default='exact'.
center : bool
    default=True.
dtype : Any
    default=np.float64.
jitter : float
    default=1e-08.
init : str
    allowed values: 'blup0', 'mean', 'zero'; default='blup0'.
max_iter : int
    default=50.
tol : float
    default=1e-06.
relaxation : float
    default=0.7.
diag_only : bool
    default=False.
return_numpy : bool
    default=False.
return_aux : bool
    default=False.

Returns
-------
object | tuple
    Inferred return type(s) from the implementation."""
    if missing not in {'exact', 'iterative'}:
        raise ValueError("missing must be 'exact' or 'iterative'")
    inp = _prepare_mvblup_inputs(traits, GRM, G, E=E, center=center, dtype=dtype, diag_only=diag_only)
    Y, Q, lam = (inp['Y'], inp['Q'], inp['lam'])
    K_full = inp['K']
    n, t = (inp['n'], inp['t'])
    G_spec, E_spec = (inp['G_spec'], inp['E_spec'])
    mask_obs = np.isfinite(Y)
    mask_mis = ~mask_obs
    if mask_obs.all():
        return mvBLUP_complete(traits=Y if not inp['traits_is_df'] else pd.DataFrame(Y, index=inp['row_index'], columns=inp['col_index']), GRM=('eigh', lam, Q), G=G_spec if G_spec['mode'] == 'full' else G, E=E_spec if E_spec['mode'] == 'full' else E, center=False, dtype=dtype, jitter=jitter, diag_only=diag_only, return_numpy=return_numpy, return_aux=return_aux)
    if K_full is None:
        K_full = _reconstruct_grm({'Q': Q, 'lam': lam, 'K': None, 'type': 'eigh'}, dtype=dtype)
    if diag_only and missing == 'exact':
        vg = _traitcov_diag(G_spec)
        ve = _traitcov_diag(E_spec)
        Uhat = np.full((n, t), np.nan, dtype=dtype)
        for k in range(t):
            idx = np.flatnonzero(mask_obs[:, k])
            if len(idx) == 0:
                Uhat[:, k] = 0.0
                continue
            yk = Y[idx, k]
            A = vg[k] * K_full[np.ix_(idx, idx)] + ve[k] * np.eye(len(idx), dtype=dtype)
            alpha = _solve_spd(A, yk, jitter=jitter)
            Uhat[:, k] = vg[k] * (K_full[:, idx] @ alpha)
        aux = {'method': 'missing_exact_diag_fast', 'means': inp['means'], 'diag_only': True, 'n_obs': int(mask_obs.sum()), 'n_total': int(n * t), 'missing_fraction': float(1.0 - mask_obs.mean())}
        Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
        return (Uhat, aux) if return_aux else Uhat
    if missing == 'exact':
        Gf = _traitcov_to_full(G_spec)
        Ef = _traitcov_to_full(E_spec)
        obs_idx_by_trait = []
        y_obs_parts = []
        trait_slices = []
        a_obs = []
        tr_obs = []
        start = 0
        for b in range(t):
            idx_b = np.flatnonzero(mask_obs[:, b])
            obs_idx_by_trait.append(idx_b)
            yb = Y[idx_b, b]
            y_obs_parts.append(yb)
            m_b = len(idx_b)
            trait_slices.append(slice(start, start + m_b))
            start += m_b
            a_obs.append(idx_b)
            tr_obs.append(np.full(m_b, b, dtype=int))
        y_obs = np.concatenate(y_obs_parts).astype(dtype, copy=False)
        a_obs = np.concatenate(a_obs)
        tr_obs = np.concatenate(tr_obs)
        m_obs = len(y_obs)
        K_obs = K_full[np.ix_(a_obs, a_obs)]
        G_obs = Gf[np.ix_(tr_obs, tr_obs)]
        if np.any(Ef != 0):
            same_animal = a_obs[:, None] == a_obs[None, :]
            E_obs = Ef[np.ix_(tr_obs, tr_obs)]
            Sigma_oo = K_obs * G_obs + same_animal * E_obs
        else:
            Sigma_oo = K_obs * G_obs
        alpha = _solve_spd(Sigma_oo, y_obs, jitter=jitter)
        B = np.zeros((n, t), dtype=dtype)
        for b in range(t):
            sl = trait_slices[b]
            idx_b = obs_idx_by_trait[b]
            if len(idx_b) == 0:
                continue
            B[:, b] = K_full[:, idx_b] @ alpha[sl]
        Uhat = B @ Gf.T
        aux = {'method': 'missing_exact_full', 'means': inp['means'], 'diag_only': diag_only, 'n_obs': int(m_obs), 'n_total': int(n * t), 'missing_fraction': float(1.0 - m_obs / (n * t)), 'alpha': alpha}
        Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
        return (Uhat, aux) if return_aux else Uhat
    if relaxation <= 0 or relaxation > 1:
        raise ValueError('relaxation must be in (0, 1]')
    if init not in {'mean', 'zero', 'blup0'}:
        raise ValueError("init must be one of {'mean', 'zero', 'blup0'}")
    Y_filled = Y.copy()
    Y_filled[mask_mis] = 0.0
    if init == 'blup0':
        U0 = mvBLUP_complete(traits=Y_filled, GRM=('eigh', lam, Q), G=G, E=E, center=False, dtype=dtype, jitter=jitter, diag_only=diag_only, return_numpy=True)
        Y_filled[mask_mis] = U0[mask_mis]
    history = []
    converged = False
    for it in range(1, max_iter + 1):
        old_missing = Y_filled[mask_mis].copy()
        Uhat = mvBLUP_complete(traits=Y_filled, GRM=('eigh', lam, Q), G=G, E=E, center=False, dtype=dtype, jitter=jitter, diag_only=diag_only, return_numpy=True)
        proposal = Uhat[mask_mis]
        Y_filled[mask_mis] = (1.0 - relaxation) * Y_filled[mask_mis] + relaxation * proposal
        delta = Y_filled[mask_mis] - old_missing
        rmse = np.sqrt(np.mean(delta ** 2)) if delta.size else 0.0
        max_abs = np.max(np.abs(delta)) if delta.size else 0.0
        history.append({'iter': it, 'rmse_missing_update': float(rmse), 'maxabs_missing_update': float(max_abs)})
        if rmse < tol:
            converged = True
            break
    Uhat = mvBLUP_complete(traits=Y_filled, GRM=('eigh', lam, Q), G=G, E=E, center=False, dtype=dtype, jitter=jitter, diag_only=diag_only, return_numpy=True)
    aux = {'method': 'missing_iterative', 'means': inp['means'], 'diag_only': diag_only, 'n_iter': len(history), 'converged': converged, 'history': history, 'Y_filled': Y_filled}
    Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
    if return_aux:
        if isinstance(aux.get('Y_filled'), np.ndarray) and inp['traits_is_df'] and (not return_numpy):
            aux['Y_filled'] = pd.DataFrame(aux['Y_filled'], index=inp['row_index'], columns=inp['col_index'])
        return (Uhat, aux)
    return Uhat

def mvWhiten(df, grm, G, E, *, only_observed: bool=True, center: bool=True, fill_missing: str='exact', diag_only: bool=False, eps: float=1e-10, return_info: bool=False, **mvblup_kwargs):
    """Whitening under vec(Y) ~ N(0, grm ⊗ G + I ⊗ E).

Allowed inputs
--------------
df : Any
    Accepted as provided to the function.
grm : Any
    Accepted as provided to the function.
G : Any
    Accepted as provided to the function.
E : Any
    Accepted as provided to the function.
only_observed : bool
    default=True.
center : bool
    default=True.
fill_missing : str
    default='exact'.
diag_only : bool
    default=False.
eps : float
    default=1e-10.
return_info : bool
    default=False.
**mvblup_kwargs : Any
    Accepted as provided to the function.

Returns
-------
pandas.DataFrame | tuple
    Inferred return type(s) from the implementation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('mvWhiten currently expects df to be a pandas DataFrame.')
    inp = _prepare_mvblup_inputs(df, grm, G, E=E, center=center, diag_only=diag_only, dtype=np.float64)
    Y, Q, lam = (inp['Y'], inp['Q'], inp['lam'])
    K_full = inp['K']
    n, t = (inp['n'], inp['t'])
    G_spec, E_spec = (inp['G_spec'], inp['E_spec'])
    mask = np.isfinite(Y)
    if only_observed:
        if diag_only:
            if K_full is None:
                K_full = _reconstruct_grm({'Q': Q, 'lam': lam, 'K': None, 'type': 'eigh'}, dtype=np.float64)
            vg = _traitcov_diag(G_spec)
            ve = _traitcov_diag(E_spec)
            Z = np.full_like(Y, np.nan)
            for k in range(t):
                idx = np.flatnonzero(mask[:, k])
                if len(idx) == 0:
                    continue
                A = vg[k] * K_full[np.ix_(idx, idx)] + ve[k] * np.eye(len(idx))
                L = np.linalg.cholesky(_symmetrize(A + eps * np.eye(len(idx))))
                Z[idx, k] = solve_triangular(L, Y[idx, k], lower=True, check_finite=False)
            out = pd.DataFrame(Z, index=df.index, columns=df.columns)
            if not return_info:
                return out
            info = {'method': 'observed_only_diag_fast', 'means': inp['means'], 'diag_only': True, 'n_obs': int(mask.sum()), 'n_total': int(n * t), 'missing_fraction': float(1.0 - mask.mean())}
            return (out, info)
        if K_full is None:
            K_full = _reconstruct_grm({'Q': Q, 'lam': lam, 'K': None, 'type': 'eigh'}, dtype=np.float64)
        yvec = Y.reshape(-1, order='F')
        obs_mask = np.isfinite(yvec)
        obs_idx = np.flatnonzero(obs_mask)
        a_obs = obs_idx % n
        tr_obs = obs_idx // n
        y_obs = yvec[obs_mask]
        Gf = _traitcov_to_full(G_spec)
        Ef = _traitcov_to_full(E_spec)
        K_obs = K_full[np.ix_(a_obs, a_obs)]
        G_obs = Gf[np.ix_(tr_obs, tr_obs)]
        E_obs = Ef[np.ix_(tr_obs, tr_obs)]
        same_animal = a_obs[:, None] == a_obs[None, :]
        Sigma_oo = K_obs * G_obs + same_animal * E_obs
        L = np.linalg.cholesky(_symmetrize(Sigma_oo + eps * np.eye(len(y_obs))))
        z_obs = solve_triangular(L, y_obs, lower=True, check_finite=False)
        zvec = np.full(n * t, np.nan, dtype=float)
        zvec[obs_mask] = z_obs
        Z = zvec.reshape((n, t), order='F')
        out = pd.DataFrame(Z, index=df.index, columns=df.columns)
        if not return_info:
            return out
        info = {'method': 'observed_only_exact', 'means': inp['means'], 'diag_only': False, 'n_obs': int(obs_mask.sum()), 'n_total': int(n * t), 'missing_fraction': float(1.0 - obs_mask.mean())}
        return (out, info)
    if mask.all():
        Y_filled = Y.copy()
    else:
        Y_centered_df = pd.DataFrame(Y, index=df.index, columns=df.columns)
        Uhat = mvBLUP(Y_centered_df, ('eigh', lam, Q), G, E, missing=fill_missing, center=False, diag_only=diag_only, return_numpy=True, **mvblup_kwargs)
        Y_filled = Y.copy()
        Y_filled[~mask] = Uhat[~mask]
    Y_rot = Q.T @ Y_filled
    if diag_only:
        vg = _traitcov_diag(G_spec)
        ve = _traitcov_diag(E_spec)
        denom = lam[:, None] * vg[None, :] + ve[None, :]
        Z_rot = np.divide(Y_rot, np.sqrt(np.clip(denom, eps, None)), out=np.zeros_like(Y_rot), where=denom > 0)
        Z = Q @ Z_rot
        Z[~mask] = np.nan
        out = pd.DataFrame(Z, index=df.index, columns=df.columns)
        if not return_info:
            return out
        info = {'method': f'full_fill_{fill_missing}_diag_fast', 'means': inp['means'], 'diag_only': True, 'Y_filled': pd.DataFrame(Y_filled, index=df.index, columns=df.columns), 'Q': Q, 'lambda': lam}
        return (out, info)
    use_factor_speed = _can_use_factor_speed(G_spec, E_spec)
    Z_rot = np.empty_like(Y_rot)
    row_covs = [] if return_info else None
    if use_factor_speed:
        for i in range(n):
            li = lam[i]
            d, U = _row_diag_lowrank(li, G_spec, E_spec)
            L = _chol_diag_plus_lowrank(d, U, jitter=eps)
            Z_rot[i, :] = solve_triangular(L, Y_rot[i, :], lower=True, check_finite=False)
            if return_info:
                row_covs.append(np.diag(d) + U @ U.T)
        method = f'full_fill_{fill_missing}_factor_fast'
    else:
        Gf = _traitcov_to_full(G_spec)
        Ef = _traitcov_to_full(E_spec)
        for i in range(n):
            Vi = _symmetrize(lam[i] * Gf + Ef)
            L = np.linalg.cholesky(Vi + eps * np.eye(t))
            Z_rot[i, :] = solve_triangular(L, Y_rot[i, :], lower=True, check_finite=False)
            if return_info:
                row_covs.append(Vi)
        method = f'full_fill_{fill_missing}_full_dense'
    Z = Q @ Z_rot
    Z[~mask] = np.nan
    out = pd.DataFrame(Z, index=df.index, columns=df.columns)
    if not return_info:
        return out
    info = {'method': method, 'means': inp['means'], 'diag_only': diag_only, 'Y_filled': pd.DataFrame(Y_filled, index=df.index, columns=df.columns), 'Q': Q, 'lambda': lam, 'row_covs': row_covs}
    return (out, info)

def organize_scores(df, SCORE_output):
    """Organize scores.

Allowed inputs
--------------
df : Any
    Accepted as provided to the function.
SCORE_output : Any
    Accepted as provided to the function.

Returns
-------
pandas.Series
    Inferred return type(s) from the implementation."""
    Ecov = pd.DataFrame(SCORE_output['E'], columns=df.columns, index=df.columns)
    Gcov = pd.DataFrame(SCORE_output['G'], columns=df.columns, index=df.columns)
    RG = pd.DataFrame(SCORE_output['Rg'], index=df.columns, columns=df.columns)
    RG_se = pd.DataFrame(SCORE_output['rg_se'], index=df.columns, columns=df.columns)
    phenocorr = corr_se_matrix_fisher(df)
    gcorr_melted = pd.concat([i.reset_index(names='trait1').melt(id_vars='trait1', value_name=j, var_name='trait2').set_index(['trait1', 'trait2']) for i, j in zip([phenocorr[0], phenocorr[1], RG, RG_se, erfc(np.abs(RG.divide(RG_se).where(RG_se > 0)) / np.sqrt(2))], ['phenotypic_correlation', 'rP_SE', 'genetic_correlation', 'rG_SE', 'pval'])], axis=1).query('trait1 < trait2')
    vg, ve, svg, sve = (np.diag(SCORE_output['G']), np.diag(SCORE_output['E']), np.diag(SCORE_output['G_se']), np.diag(SCORE_output['E_se']))
    h2 = vg / (vg + ve)
    dh_dvg, dh_dve = (ve / (vg + ve) ** 2, -vg / (vg + ve) ** 2)
    h2_se = np.sqrt(dh_dvg ** 2 * svg ** 2 + dh_dve ** 2 * sve ** 2)
    h2_pvalue = 0.5 * erfc(np.abs(h2 / h2_se) / np.sqrt(2))
    her = pd.DataFrame(data=np.stack([vg, ve, h2, svg, sve, h2_se, h2_pvalue, h2_pvalue < 0.05]).T, index=df.columns, columns=['vG', 'vE', 'h2', 'vG_se', 'vE_se', 'h2_se', 'pval', 'significance'])
    return pd.Series({'heritability_table': her, 'g_corr_table': gcorr_melted, 'G': Gcov, 'rG_SE': RG_se, 'E': Ecov})

def mvGWAS(traitdf, genotypes='genotypes/genotypes', grms_folder='grm', save_path='results/gwas_parquet/', save=False, y_correction='blup_resid', y_correction_multivariate=False, return_table=True, stat='ttest', chrset=None, dtype='pandas', dof='correct', snp_block_size=50000000.0, gwa_center=True, gwa_scale=False, regression_mode='blas', covariance_estimator='psd', npplink_h2_ncomponents = 3000 ):
    """Mvgwas.

Allowed inputs
--------------
traitdf : Any
    Accepted as provided to the function.
genotypes : str
    default='genotypes/genotypes'.
grms_folder : str
    default='grm'.
save_path : str
    default='results/gwas_parquet/'.
save : bool
    default=False.
y_correction : str
    allowed values: 'ystar'; default='blup_resid'.
y_correction_multivariate : bool
    default=False.
return_table : bool
    default=True.
stat : str
    default='ttest'.
chrset : Any | None
    default=None.
dtype : str
    default='pandas'.
dof : str
    default='correct'.
snp_block_size : float
    default=50000000.0.
gwa_center : bool
    default=True.
gwa_scale : bool
    default=False.
regression_mode : str
    default='blas'.
covariance_estimator: str
    allowed values: 'npplink_svd'; default='psd'.
npplink_h2_ncomponents: int
    default=3000.

Returns
-------
None | object | pandas.DataFrame
    Inferred return type(s) from the implementation."""
    current_mem = lambda: str(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, 2)) + 'Gb'
    if isinstance(grms_folder, str):
        grms_folder = glob(f'{grms_folder}/*.grm.bin')
    read_gen = npplink.load_plink(genotypes)
    chrunique = [str(x) for x in read_gen[0].chrom.unique()]
    grms_folder = pd.DataFrame(grms_folder, columns=['path'])
    grms_folder.index = grms_folder['path'].str.extract('([\\d\\w_]+)chrGRM.', expand=False)
    grms_folder = grms_folder.sort_index(key=lambda idx: idx.str.lower().map({str(i): int(i) for i in range(1000)} | {i: int(i) for i in range(1000)} | {'all': -1000, 'x': 1001, 'y': 1002, 'mt': 1003, 'm': 1003}))
    grms_folder = grms_folder[~grms_folder.index.isna()]
    allGRM = npplink.read_grm(grms_folder.loc['All', 'path'].replace('.bin', ''))
    grms_folder['in_chrunique'] = grms_folder.index.isin(chrunique)
    grms_folder['isnum'] = grms_folder.index.str.isnumeric()
    max_grm_chr = grms_folder.query('isnum').index.astype(int).max()
    if grms_folder.in_chrunique.eq(False).sum() > 1:
        print('some chromosomes in the grms folder does not align with the genotypes, trying to convert x,y,mt to +1, +2, +4')
        grms_folder = grms_folder.rename({'x': str(max_grm_chr + 1), 'y': str(max_grm_chr + 2), 'mt': str(max_grm_chr + 4), 'X': str(max_grm_chr + 1), 'Y': str(max_grm_chr + 2), 'MT': str(max_grm_chr + 4)})
        grms_folder['in_chrunique'] = grms_folder.index.isin(chrunique)
        if grms_folder.in_chrunique.eq(False).sum() > 1:
            print('not solved')
            raise ValueError('cannot match chromosomes in grm folder and genotype flies')
    chrom_sizes = read_gen[0].astype({'chrom': str}).groupby('chrom').pos.max()
    sumstats = []
    if save:
        os.makedirs(save_path, exist_ok=True)
    for c, row in (pbar := tqdm(list(grms_folder.drop(['All']).iterrows()))):
        pbar.set_description(f' GWAS-Chr{c}-reading {c}GRM')
        c_grm = npplink.read_grm(row.path.replace('.bin', ''))
        subgrm = allGRM['grm'].to_pandas() if not row.isnum else ((allGRM['grm'] * allGRM['w'] - c_grm['grm'] * c_grm['w']) / (allGRM['w'] - c_grm['w'])).to_pandas()
        pbar.set_description(f'GWAS-Chr{c}-estimating var/covarmatrix')
        if covariance_estimator == 'psd':
            pbar.set_description(f'GWAS-Chr{c}-PSD: estimating var/covarmatrix')
            GErG = genetic_varcov_PSD(subgrm, traitdf, n_boot=0)
            gen_var_cov, env_var_cov = GErG['G'], GErG['E']
        elif covariance_estimator == 'npplink_svd':
            pbar.set_description(f'GWAS-Chr{c}-npplink: estimating h2 and setting y_correction to diagonal')
            h2 = npplink.heritability(traitdf, GRM = subgrm, n_components=min(npplink_h2_ncomponents, traitdf.shape[0]-1))['h2']
            gen_var_cov, env_var_cov = h2 * traitdf.std(), (1-h2) * traitdf.std()
            gen_var_cov = pd.DataFrame(np.diag(gen_var_cov),index = gen_var_cov.index, columns = gen_var_cov.index)
            env_var_cov = pd.DataFrame(np.diag(env_var_cov),index = env_var_cov.index, columns = env_var_cov.index)
            y_correction_multivariate = False
        if y_correction == 'blup_resid':
            pbar.set_description(f'GWAS-Chr{c}-BLUPresiduals-MEM:{current_mem()}')
            traits = traitdf - mvBLUP(traitdf, subgrm, gen_var_cov, E=env_var_cov, missing='exact', diag_only=not y_correction_multivariate)
        elif y_correction in ['ystar']:
            pbar.set_description(f'GWAS-Chr{c}-whitenmatrix-MEM:{current_mem()}')
            traits = mvWhiten(traitdf, subgrm, gen_var_cov, E=env_var_cov, only_observed=False, diag_only=not y_correction_multivariate)
        if snp_block_size < chrom_sizes[c]:
            snp_blocks = list(pairwise(np.arange(0, chrom_sizes[c] + snp_block_size, snp_block_size)))
            if return_table and (not save):
                pbar.set_description(f'GWAS-Chr{c}-GWAS-MEM:{current_mem()}')
                sumstats.extend((npplink.GWA(traits, npplink.plink2df(read_gen, c=c, rfids=traits.index, pos_start=start, pos_end=stop), dtype=dtype, stat=stat, dof='correct', center=gwa_center, scale=gwa_scale, regression_mode=regression_mode) for start, stop in snp_blocks))
            elif save and (not return_table):
                for start, stop in snp_blocks:
                    pbar.set_description(f'GWAS-Chr{c}-GWAS[{start / 1000000.0:.0f}-{stop / 1000000.0:.0f}]Mb-MEM:{current_mem()}')
                    npplink.GWA(traits, npplink.plink2df(read_gen, c=c, rfids=traits.index, pos_start=start, pos_end=stop), dtype=dtype, stat=stat, dof='correct', center=gwa_center, scale=gwa_scale, regression_mode=regression_mode).to_parquet(f'{save_path}gwas{c}_{int(start)}_{int(stop)}.parquet.gz', compression='gzip', engine='pyarrow', compression_level=1, use_dictionary=True)
            else:
                for start, stop in snp_blocks:
                    pbar.set_description(f'GWAS-Chr{c}-GWAS[{start / 1000000.0:.0f}-{stop / 1000000.0:.0f}]Mb-MEM:{current_mem()}')
                    block = npplink.GWA(traits, npplink.plink2df(read_gen, c=c, rfids=traits.index, pos_start=start, pos_end=stop), dtype=dtype, stat=stat, dof='correct', center=gwa_center, scale=gwa_scale, regression_mode=regression_mode)
                    if save:
                        block.to_parquet(f'{save_path}gwas{c}_{int(start)}_{int(stop)}.parquet.gz', compression='gzip', engine='pyarrow', compression_level=1, use_dictionary=True)
                    if return_table:
                        sumstats.append(block)
        elif save and (not return_table):
            pbar.set_description(f'GWAS-Chr{c}-GWAS&save-MEM:{current_mem()}')
            npplink.GWA(traits, npplink.plink2df(read_gen, c=c, rfids=traits.index), dtype=dtype, stat=stat, dof='correct', center=gwa_center, scale=gwa_scale, regression_mode=regression_mode).to_parquet(f'{save_path}gwas{c}.parquet.gz', compression='gzip', engine='pyarrow', compression_level=1, use_dictionary=True)
        else:
            pbar.set_description(f'GWAS-Chr{c}-GWAS-MEM:{current_mem()}')
            sumstats.append(npplink.GWA(traits, npplink.plink2df(read_gen, c=c, rfids=traits.index), dtype=dtype, stat=stat, dof='correct', center=gwa_center, scale=gwa_scale, regression_mode=regression_mode))
            if save:
                pbar.set_description(f'GWAS-Chr{c}-GWASsaving-MEM:{current_mem()}')
                sumstats[-1].to_parquet(f'{save_path}gwas{c}.parquet.gz', compression='gzip', engine='pyarrow', compression_level=1, use_dictionary=True)
        del c_grm, subgrm, traits
        gc.collect()
    if return_table:
        if 'pandas' not in dtype:
            return res
        return pd.concat(sumstats)
    return

# =============================================================================
# v3 optimized numerical core
# =============================================================================
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import LinearOperator, cg

_V3_VERSION = '3.0.0'


def _prepare_mvblup_inputs_v3(traits, GRM, G, E=None, center=True, dtype=np.float64, diag_only=False, need_eigen=False):
    traits_is_df = isinstance(traits, pd.DataFrame)
    if traits_is_df:
        row_index, col_index = traits.index, traits.columns
        Y = traits.to_numpy(dtype=dtype, copy=True)
    else:
        Y = np.asarray(traits, dtype=dtype)
        if Y.ndim != 2: raise ValueError('traits must be a 2D array or DataFrame')
        row_index = col_index = None
    n, t = Y.shape
    if isinstance(GRM, dict) and str(GRM.get('type', '')).lower() == 'eigh':
        lam = np.clip(np.asarray(GRM['values'], dtype=dtype).reshape(-1), 0.0, None)
        Q = np.asarray(GRM['vectors'], dtype=dtype)
        K = None
    elif isinstance(GRM, tuple) and str(GRM[0]).lower() == 'eigh':
        lam = np.clip(np.asarray(GRM[1], dtype=dtype).reshape(-1), 0.0, None)
        Q = np.asarray(GRM[2], dtype=dtype)
        K = None
    else:
        raw = GRM['matrix'] if isinstance(GRM, dict) and str(GRM.get('type', '')).lower() == 'full' else (GRM[1] if isinstance(GRM, tuple) and str(GRM[0]).lower() == 'full' else GRM)
        K = _symmetrize(np.asarray(raw, dtype=dtype))
        if K.shape != (n, n): raise ValueError('GRM size does not match number of animals')
        Q = lam = None
    if need_eigen and Q is None:
        lam, Q = np.linalg.eigh(K)
        lam = np.clip(lam, 0.0, None)
    if Q is not None and Q.shape != (n, n): raise ValueError('GRM eigenvectors must be square and aligned to traits')
    G_spec = _normalize_traitcov_spec(G, size=t, dtype=dtype, diag_only=diag_only)
    E_spec = _normalize_traitcov_spec(E, size=t, dtype=dtype, diag_only=diag_only, allow_none=True)
    means = np.zeros(t, dtype=dtype)
    if center:
        means = np.nanmean(Y, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        Y -= means[None, :]
    return {'Y':Y, 'Q':Q, 'lam':lam, 'K':K, 'G_spec':G_spec, 'E_spec':E_spec, 'means':means,
            'traits_is_df':traits_is_df, 'row_index':row_index, 'col_index':col_index, 'n':n, 't':t, 'diag_only':diag_only}


def _generalized_trait_blup(Y_tilde, lam, G, E, jitter=1e-8):
    """Exact row-wise BLUP after one generalized trait eigendecomposition."""
    t = G.shape[0]
    Ereg = _symmetrize(E) + float(jitter) * np.eye(t)
    rho, V = scipy_eigh(_symmetrize(G), Ereg, check_finite=False)
    rho = np.clip(rho, 0.0, None)
    EV = Ereg @ V
    Z = Y_tilde @ V
    lr = lam[:, None] * rho[None, :]
    shrink = np.divide(lr, 1.0 + lr, out=np.zeros_like(lr), where=np.isfinite(lr))
    return (Z * shrink) @ EV.T


def mvBLUP_complete(traits, GRM, G, E=None, eigen: bool=True, center: bool=True, dtype=np.float64, jitter: float=1e-08, diag_only: bool=False, return_numpy: bool=False, return_aux: bool=False):
    inp = _prepare_mvblup_inputs_v3(traits, GRM, G, E=E, center=center, dtype=dtype, diag_only=diag_only, need_eigen=True)
    Y, Q, lam = inp['Y'], inp['Q'], inp['lam']
    if np.isnan(Y).any(): raise ValueError('mvBLUP_complete does not accept NaN. Use mvBLUP(..., missing=...).')
    G_spec, E_spec = inp['G_spec'], inp['E_spec']
    if np.all(_traitcov_diag(E_spec) == 0) and E_spec['mode'] == 'diag':
        Uhat, method = Y.copy(), 'complete_no_residual'
    elif diag_only:
        vg, ve = _traitcov_diag(G_spec), _traitcov_diag(E_spec)
        Yt = Q.T @ Y
        denom = lam[:, None] * vg[None, :] + ve[None, :]
        shrink = np.divide(lam[:, None] * vg[None, :], denom, out=np.zeros_like(denom), where=denom > 0)
        Uhat, method = Q @ (shrink * Yt), 'complete_diag_eigen_fast'
    else:
        Gf, Ef = _traitcov_to_full(G_spec), _traitcov_to_full(E_spec)
        Uhat = Q @ _generalized_trait_blup(Q.T @ Y, lam, Gf, Ef, jitter=jitter)
        method = 'complete_full_generalized_eigen'
    aux = {'method':method, 'means':inp['means'], 'diag_only':diag_only, 'version':_V3_VERSION}
    Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
    return (Uhat, aux) if return_aux else Uhat


def _observed_plan(Y):
    mask = np.isfinite(Y)
    idx, slices, yparts, animals, traits = [], [], [], [], []
    start = 0
    for b in range(Y.shape[1]):
        ib = np.flatnonzero(mask[:, b]); m = len(ib)
        idx.append(ib); slices.append(slice(start, start + m)); yparts.append(Y[ib, b])
        animals.append(ib); traits.append(np.full(m, b, dtype=np.int32)); start += m
    return {'mask':mask, 'idx':idx, 'slices':slices, 'y':np.concatenate(yparts),
            'animals':np.concatenate(animals), 'traits':np.concatenate(traits), 'm':start}


def _observed_cov_operator(K, G, E, plan, dtype=np.float64):
    n, t = K.shape[0], G.shape[0]
    A = np.zeros((n, t), dtype=dtype)
    out = np.empty(plan['m'], dtype=dtype)
    def matvec(v):
        A.fill(0.0)
        for b, (ib, sl) in enumerate(zip(plan['idx'], plan['slices'])): A[ib, b] = v[sl]
        Z = (K @ A) @ G.T
        if np.any(E): Z += A @ E.T
        for b, (ib, sl) in enumerate(zip(plan['idx'], plan['slices'])): out[sl] = Z[ib, b]
        return out.copy()
    return LinearOperator((plan['m'], plan['m']), matvec=matvec, dtype=np.dtype(dtype))


def _trait_block_preconditioner(K, G, E, plan, jitter=1e-8, dtype=np.float64):
    factors = []
    for b, ib in enumerate(plan['idx']):
        if len(ib) == 0:
            factors.append(None); continue
        P = G[b, b] * np.asarray(K[np.ix_(ib, ib)], dtype=dtype)
        P.flat[::len(ib)+1] += E[b, b] + float(jitter)
        try: factors.append(cho_factor(_symmetrize(P), lower=True, check_finite=False))
        except np.linalg.LinAlgError: factors.append(None)
    def matvec(v):
        z = np.empty_like(v, dtype=dtype)
        for b, sl in enumerate(plan['slices']):
            if sl.stop == sl.start: continue
            fac = factors[b]
            if fac is None:
                d = max(float(G[b,b] + E[b,b]), float(jitter)); z[sl] = v[sl] / d
            else: z[sl] = cho_solve(fac, v[sl], check_finite=False)
        return z
    return LinearOperator((plan['m'], plan['m']), matvec=matvec, dtype=np.dtype(dtype))


def _diag_observed_preconditioner(K, G, E, plan, jitter=1e-8, dtype=np.float64):
    dK = np.diag(K)
    d = dK[plan['animals']] * np.diag(G)[plan['traits']] + np.diag(E)[plan['traits']] + float(jitter)
    d = np.clip(d, float(jitter), None).astype(dtype, copy=False)
    return LinearOperator((plan['m'], plan['m']), matvec=lambda v: v / d, dtype=np.dtype(dtype))


def _solve_observed_pcg(K, G, E, plan, *, jitter=1e-8, rtol=1e-7, atol=0.0, maxiter=500, preconditioner='trait_block', dtype=np.float64):
    A = _observed_cov_operator(K, G, E, plan, dtype=dtype)
    if preconditioner == 'trait_block': M = _trait_block_preconditioner(K, G, E, plan, jitter=jitter, dtype=dtype)
    elif preconditioner == 'diag': M = _diag_observed_preconditioner(K, G, E, plan, jitter=jitter, dtype=dtype)
    elif preconditioner in {None, 'none'}: M = None
    else: raise ValueError("preconditioner must be 'trait_block', 'diag', or 'none'")
    nit = [0]
    def cb(_): nit[0] += 1
    alpha, info = cg(A, plan['y'].astype(dtype, copy=False), M=M, rtol=float(rtol), atol=float(atol), maxiter=int(maxiter), callback=cb)
    residual = np.linalg.norm(A @ alpha - plan['y']) / max(np.linalg.norm(plan['y']), np.finfo(float).eps)
    return alpha, {'cg_info':int(info), 'cg_iterations':int(nit[0]), 'cg_relative_residual':float(residual), 'preconditioner':preconditioner}


def _dense_observed_solve(K, G, E, plan, jitter=1e-8):
    a, tr = plan['animals'], plan['traits']
    Sigma = K[np.ix_(a, a)] * G[np.ix_(tr, tr)]
    if np.any(E): Sigma += (a[:, None] == a[None, :]) * E[np.ix_(tr, tr)]
    return _solve_spd(Sigma, plan['y'], jitter=jitter)


def mvBLUP(traits, GRM, G, E=None, eigen: bool=True, missing: str='exact', center: bool=True, dtype=np.float64, jitter: float=1e-08,
           init: str='blup0', max_iter: int=50, tol: float=1e-06, relaxation: float=0.7, diag_only: bool=False,
           return_numpy: bool=False, return_aux: bool=False, solver: str='auto', solver_tol: float=1e-7,
           solver_maxiter: int=500, preconditioner: str='diag', dense_obs_threshold: int=3000):
    if missing not in {'exact', 'iterative'}: raise ValueError("missing must be 'exact' or 'iterative'")
    need_eigen = missing == 'iterative'
    inp = _prepare_mvblup_inputs_v3(traits, GRM, G, E=E, center=center, dtype=dtype, diag_only=diag_only, need_eigen=need_eigen)
    Y, K, n, t = inp['Y'], inp['K'], inp['n'], inp['t']
    mask = np.isfinite(Y)
    if mask.all():
        GRM2 = ('eigh', inp['lam'], inp['Q']) if inp['Q'] is not None else K
        return mvBLUP_complete(Y if not inp['traits_is_df'] else pd.DataFrame(Y,index=inp['row_index'],columns=inp['col_index']), GRM2, G, E=E,
                               center=False, dtype=dtype, jitter=jitter, diag_only=diag_only, return_numpy=return_numpy, return_aux=return_aux)
    if K is None: K = _reconstruct_grm({'Q':inp['Q'], 'lam':inp['lam'], 'K':None, 'type':'eigh'}, dtype=dtype)
    G_spec, E_spec = inp['G_spec'], inp['E_spec']
    if diag_only and missing == 'exact':
        vg, ve = _traitcov_diag(G_spec), _traitcov_diag(E_spec)
        Uhat = np.zeros((n, t), dtype=dtype)
        for b in range(t):
            ib = np.flatnonzero(mask[:, b])
            if not len(ib): continue
            A = vg[b] * K[np.ix_(ib, ib)] + ve[b] * np.eye(len(ib), dtype=dtype)
            alpha = _solve_spd(A, Y[ib, b], jitter=jitter)
            Uhat[:, b] = vg[b] * (K[:, ib] @ alpha)
        aux = {'method':'missing_exact_diag_fast', 'means':inp['means'], 'diag_only':True, 'n_obs':int(mask.sum()),
               'n_total':int(n*t), 'missing_fraction':float(1-mask.mean()), 'version':_V3_VERSION}
    elif missing == 'exact':
        Gf, Ef = _traitcov_to_full(G_spec), _traitcov_to_full(E_spec)
        plan = _observed_plan(Y); m = plan['m']
        use_solver = 'dense' if solver == 'auto' and m <= int(dense_obs_threshold) else ('pcg' if solver == 'auto' else solver)
        if use_solver == 'dense':
            alpha = _dense_observed_solve(K, Gf, Ef, plan, jitter=jitter); sinfo = {}
        elif use_solver == 'pcg':
            alpha, sinfo = _solve_observed_pcg(K, Gf, Ef, plan, jitter=jitter, rtol=solver_tol, maxiter=solver_maxiter,
                                               preconditioner=preconditioner, dtype=dtype)
            if sinfo['cg_info'] != 0 or sinfo['cg_relative_residual'] > max(10*solver_tol, 1e-5):
                raise RuntimeError(f"Exact mvBLUP PCG did not converge: {sinfo}")
        else: raise ValueError("solver must be 'auto', 'dense', or 'pcg'")
        Aalpha = np.zeros((n, t), dtype=dtype)
        for b, (ib, sl) in enumerate(zip(plan['idx'], plan['slices'])): Aalpha[ib, b] = alpha[sl]
        Uhat = (K @ Aalpha) @ Gf.T
        aux = {'method':f'missing_exact_{use_solver}', 'means':inp['means'], 'diag_only':False, 'n_obs':int(m), 'n_total':int(n*t),
               'missing_fraction':float(1-m/(n*t)), 'alpha':alpha, 'version':_V3_VERSION, **sinfo}
    else:
        if relaxation <= 0 or relaxation > 1: raise ValueError('relaxation must be in (0, 1]')
        if init not in {'mean','zero','blup0'}: raise ValueError("init must be one of {'mean', 'zero', 'blup0'}")
        Q, lam = inp['Q'], inp['lam']; Yfill = Y.copy(); mis = ~mask; Yfill[mis] = 0.0
        if init == 'blup0': Yfill[mis] = mvBLUP_complete(Yfill, ('eigh',lam,Q), G, E=E, center=False, dtype=dtype, jitter=jitter, diag_only=diag_only, return_numpy=True)[mis]
        hist=[]; converged=False
        for it in range(1,max_iter+1):
            old=Yfill[mis].copy(); Uhat=mvBLUP_complete(Yfill,('eigh',lam,Q),G,E=E,center=False,dtype=dtype,jitter=jitter,diag_only=diag_only,return_numpy=True)
            Yfill[mis]=(1-relaxation)*Yfill[mis]+relaxation*Uhat[mis]; d=Yfill[mis]-old; rmse=float(np.sqrt(np.mean(d*d))) if d.size else 0.0
            hist.append({'iter':it,'rmse_missing_update':rmse,'maxabs_missing_update':float(np.max(np.abs(d))) if d.size else 0.0})
            if rmse < tol: converged=True; break
        Uhat=mvBLUP_complete(Yfill,('eigh',lam,Q),G,E=E,center=False,dtype=dtype,jitter=jitter,diag_only=diag_only,return_numpy=True)
        aux={'method':'missing_iterative','means':inp['means'],'diag_only':diag_only,'n_iter':len(hist),'converged':converged,'history':hist,'Y_filled':Yfill,'version':_V3_VERSION}
    Uhat = _wrap_mv_output(Uhat, inp['traits_is_df'], inp['row_index'], inp['col_index'], return_numpy)
    if return_aux:
        if isinstance(aux.get('Y_filled'), np.ndarray) and inp['traits_is_df'] and not return_numpy: aux['Y_filled']=pd.DataFrame(aux['Y_filled'],index=inp['row_index'],columns=inp['col_index'])
        return Uhat, aux
    return Uhat


# Keep the v2 estimator as the exact compatibility core, and add optional v3 regularization.
_genetic_varcov_PSD_v2 = genetic_varcov_PSD

def _shrink_covariance_offdiag(C, alpha):
    C = np.asarray(C, dtype=float); D = np.diag(np.diag(C)); a = float(alpha)
    if not 0 <= a <= 1: raise ValueError('covariance shrinkage must be in [0,1]')
    return _symmetrize(D + a * (C - D))


def genetic_varcov_PSD(grm, traits: pd.DataFrame, *, mode='auto', block_size=1024, block_dtype=np.float32, accum_dtype=np.float64,
                       square_dtype=None, dense_threshold_gb=4.0, rank_G=None, rank_E=None, psd_E=True, ridge_G=0.0, ridge_E=0.0,
                       method='L-BFGS-B', maxiter=1000, gtol=1e-08, ftol=1e-12, init='pairwise', eps=1e-12, n_boot=0, random_state=None,
                       shrink_G='auto', shrink_E='auto', E_structure='auto', min_E_pair_overlap=2):
    out = _genetic_varcov_PSD_v2(grm, traits, mode=mode, block_size=block_size, block_dtype=block_dtype, accum_dtype=accum_dtype,
        square_dtype=square_dtype, dense_threshold_gb=dense_threshold_gb, rank_G=rank_G, rank_E=rank_E, psd_E=psd_E,
        ridge_G=ridge_G, ridge_E=ridge_E, method=method, maxiter=maxiter, gtol=gtol, ftol=ftol, init=init, eps=eps,
        n_boot=n_boot, random_state=random_state)
    G0, E0 = out['G'].to_numpy(dtype=float), out['E'].to_numpy(dtype=float)
    overlap = traits.notna().astype(np.int32).T @ traits.notna().astype(np.int32)
    nobs = traits.notna().sum(axis=0).to_numpy(float); iu = np.triu_indices(len(nobs), 1); ov = overlap.to_numpy(dtype=float)
    alpha_G = float(np.median(nobs)/(np.median(nobs)+50.0)) if isinstance(shrink_G,str) and shrink_G.lower()=='auto' else float(shrink_G)
    med_overlap = float(np.median(ov[iu])) if len(iu[0]) else float(np.median(nobs))
    alpha_E = float(med_overlap/(med_overlap+400.0)) if isinstance(shrink_E,str) and shrink_E.lower()=='auto' else float(shrink_E)
    G1, E1 = _shrink_covariance_offdiag(G0, alpha_G), _shrink_covariance_offdiag(E0, alpha_E)
    E_mode = str(E_structure).lower()
    if E_mode == 'diag': E1 = np.diag(np.diag(E1)); E_resolved = 'diag'
    elif E_mode == 'auto':
        ov = overlap.to_numpy(); iu = np.triu_indices_from(ov, 1); zero_fraction = float(np.mean(ov[iu] < int(min_E_pair_overlap))) if len(iu[0]) else 0.0
        if zero_fraction >= 0.5:
            E1 = np.diag(np.diag(E1)); E_resolved = 'diag_sparse_overlap'
        else:
            keep = ov >= int(min_E_pair_overlap); np.fill_diagonal(keep, True); E1 = np.where(keep, E1, 0.0); E1 = _project_psd(E1, eps=max(eps,1e-10)); E_resolved = 'supported_full'
    elif E_mode == 'full': E_resolved = 'full'
    else: raise ValueError("E_structure must be 'full', 'diag', or 'auto'")
    cols = traits.columns; Rg, vg, valid = _rg_from_G_psd(G1, eps=eps)
    out['G'] = pd.DataFrame(G1,index=cols,columns=cols); out['E'] = pd.DataFrame(E1,index=cols,columns=cols)
    out['Rg'] = pd.DataFrame(Rg,index=cols,columns=cols); out['vg'] = pd.Series(vg,index=cols); out['valid_vg']=valid
    out['invalid_var_mask']=~valid; out['n_invalid_variances']=int((~valid).sum()); out['invalid_traits']=list(cols[~valid])
    out['shrink_G']=alpha_G; out['shrink_E']=alpha_E; out['shrink_G_requested']=shrink_G; out['shrink_E_requested']=shrink_E; out['E_structure']=E_mode; out['E_structure_resolved']=E_resolved; out['E_pair_overlap']=overlap
    return out


def auxiliary_impute(traits, blup, method='lightgbm', *, random_state=0, n_components=8, model_kwargs=None):
    """Optional phenotype imputation stacked on BLUP; never changes the EBV itself."""
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
    from sklearn.preprocessing import StandardScaler
    Y = traits.copy() if isinstance(traits,pd.DataFrame) else pd.DataFrame(traits)
    B = blup.reindex_like(Y) if isinstance(blup,pd.DataFrame) else pd.DataFrame(blup,index=Y.index,columns=Y.columns)
    Xobs = Y.copy(); Xobs = Xobs.fillna(B)
    X = np.concatenate([B.to_numpy(float), Xobs.to_numpy(float)], axis=1)
    X = np.nan_to_num(X, nan=0.0); out = B.copy(); model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    models = {}; requested_method = method; resolved_method = method
    if method == 'multitask_elasticnet':
        complete = Y.notna().all(axis=1).to_numpy()
        if complete.sum() >= max(20, Y.shape[1] + 2):
            pars=dict(alpha=0.01,l1_ratio=0.2,max_iter=5000,random_state=random_state); pars.update(model_kwargs); mdl=MultiTaskElasticNet(**pars)
            # Use BLUPs only: putting observed phenotypes in a shared multitask feature matrix leaks each target into itself.
            mdl.fit(B.to_numpy(float)[complete],Y.to_numpy(float)[complete]); out.loc[:,:]=mdl.predict(B.to_numpy(float)); models['multitask']=mdl
            no_pheno=Y.notna().sum(axis=1).eq(0); out.loc[no_pheno]=B.loc[no_pheno]
            return {'imputed':out,'imputed_ml':out.copy(),'blup':B,'models':models,'method':requested_method,'method_resolved':'multitask_elasticnet','n_genomic_only':int(no_pheno.sum())}
        resolved_method = 'elasticnet_masked_fallback'; method = 'elasticnet'
    for j,c in enumerate(Y.columns):
        ok = Y[c].notna().to_numpy()
        if ok.sum() < 20: continue
        Xj = X.copy()
        Xj[:, B.shape[1] + j] = B.iloc[:, j].to_numpy(float)  # never leak the target phenotype into its own features
        if method == 'lightgbm':
            from lightgbm import LGBMRegressor
            pars = dict(n_estimators=96,num_leaves=15,max_depth=5,learning_rate=0.04,verbosity=-1,random_state=random_state,n_jobs=1)
            pars.update(model_kwargs); mdl=LGBMRegressor(**pars); Xdf=pd.DataFrame(Xj,index=Y.index,columns=[f'x{i}' for i in range(Xj.shape[1])]); mdl.fit(Xdf.loc[ok],Y.loc[ok,c]); pred=mdl.predict(Xdf)
        elif method == 'pcr':
            k=min(int(n_components),X.shape[1],max(1,ok.sum()-1)); pca=PCA(k,random_state=random_state); Z=pca.fit_transform(Xj[ok]); coef=np.linalg.lstsq(np.c_[np.ones(len(Z)),Z],Y.loc[ok,c],rcond=None)[0]; pred=np.c_[np.ones(len(Xj)),pca.transform(Xj)]@coef; mdl=(pca,coef)
        elif method == 'pls':
            k=min(int(n_components),X.shape[1],max(1,ok.sum()-1)); mdl=PLSRegression(n_components=k,scale=True,max_iter=500); mdl.fit(Xj[ok],Y.loc[ok,c]); pred=mdl.predict(Xj).ravel()
        elif method in {'elasticnet','multitask_elasticnet'}:
            # With arbitrary element-wise missingness sklearn MultiTaskElasticNet cannot use the masked target matrix correctly;
            # fit one regularized model per trait instead of silently dropping nearly all animals.
            pars=dict(alpha=0.01,l1_ratio=0.2,max_iter=5000,random_state=random_state); pars.update(model_kwargs); mdl=ElasticNet(**pars); mdl.fit(Xj[ok],Y.loc[ok,c]); pred=mdl.predict(Xj)
        else: raise ValueError("method must be 'lightgbm', 'pcr', 'pls', or 'elasticnet'")
        out[c]=pred; models[c]=mdl
    ml_out = out.copy()
    no_pheno = Y.notna().sum(axis=1).eq(0)
    out.loc[no_pheno] = B.loc[no_pheno]  # genomic-only candidates keep the BLUP; ML is for phenotype-cell imputation
    return {'imputed':out,'imputed_ml':ml_out,'blup':B,'models':models,'method':requested_method,'method_resolved':resolved_method,'n_genomic_only':int(no_pheno.sum())}

# --- v3 LOCO GWAS -----------------------------------------------------------
_mvGWAS_v2 = mvGWAS

def _require_npplink():
    global npplink
    if npplink is None:
        import importlib
        npplink = importlib.import_module('npplink')
    return npplink


def _lmm_score_gwa(traits, snps, K, G, E, *, dtype='pandas', stat='ttest', center=True, scale=False, jitter=1e-8):
    """Trait-wise exact GLS/score scan conditional on LOCO G/E estimates."""
    from scipy.stats import t as student_t
    Y = traits.to_numpy(float); X0 = snps.to_numpy(float); nvar, nt = X0.shape[1], Y.shape[1]
    beta=np.full((nvar,nt),np.nan); se=np.full_like(beta,np.nan); stats=np.full_like(beta,np.nan); nlp=np.full_like(beta,np.nan); dof=np.full_like(beta,np.nan)
    vg, ve = np.diag(np.asarray(G,float)), np.diag(np.asarray(E,float))
    for t in range(nt):
        obs=np.flatnonzero(np.isfinite(Y[:,t])); n=len(obs)
        if n < 4: continue
        y=Y[obs,t]; X=X0[obs].copy(); valid=np.isfinite(X); counts=valid.sum(0)
        means=np.divide(np.nansum(X,axis=0),counts,out=np.zeros(nvar),where=counts>0); X-=means[None,:]; X[~valid]=0.0
        if scale:
            sd=np.sqrt(np.divide(np.sum(X*X,axis=0),np.maximum(counts-1,1))); X=np.divide(X,sd[None,:],out=np.zeros_like(X),where=sd[None,:]>0)
        V=vg[t]*np.asarray(K[np.ix_(obs,obs)],float); V.flat[::n+1]+=ve[t]+float(jitter); fac=cho_factor(_symmetrize(V),lower=True,check_finite=False)
        Vy=cho_solve(fac,y,check_finite=False)
        if center:
            one=np.ones(n); V1=cho_solve(fac,one,check_finite=False); den1=float(one@V1); Py=Vy-V1*((one@Vy)/den1)
        else: V1=None; den1=None; Py=Vy
        VX=cho_solve(fac,X,check_finite=False)
        if center: PX=VX-V1[:,None]*((one@VX)/den1)[None,:]
        else: PX=VX
        U=X.T@Py; I=np.sum(X*PX,axis=0); ok=(I>0)&(counts>(2 if center else 1)); b=np.divide(U,I,out=np.full(nvar,np.nan),where=ok); s=np.divide(1.0,np.sqrt(I),out=np.full(nvar,np.nan),where=ok); z=b/s
        beta[:,t],se[:,t]=b,s; dof[:,t]=counts-(2 if center else 1)
        if stat=='ttest':
            stats[:,t]=z; p=2*student_t.sf(np.abs(z),df=dof[:,t])
        elif stat in {'wald','score'}:
            stats[:,t]=z*z; p=erfc(np.abs(z)/np.sqrt(2))
        else: raise ValueError("stat must be 'ttest', 'wald', or 'score'")
        nlp[:,t]=-np.log10(np.clip(p,np.finfo(float).tiny,1.0))
    results=(beta,se,stats,nlp,dof); metrics=['beta','beta_se','stat','neglog_p','dof']; snp_names=list(snps.columns); trait_names=list(traits.columns)
    if dtype=='tuple': return results
    if dtype=='xarray_dataset':
        import xarray as xr
        return xr.Dataset({m:(('snp','trait'),a) for m,a in zip(metrics,results)},coords={'snp':snp_names,'trait':trait_names})
    if dtype=='xarray':
        import xarray as xr
        return xr.DataArray(np.stack(results),dims=['metric','snp','trait'],coords={'metric':metrics,'snp':snp_names,'trait':trait_names})
    if dtype=='pandas_highmem':
        import xarray as xr
        A=xr.DataArray(np.stack(results),dims=['metric','snp','trait'],coords={'metric':metrics,'snp':snp_names,'trait':trait_names}); return A.to_dataset(dim='metric').to_dataframe().reset_index()
    if dtype=='pandas':
        return pd.DataFrame({'snp':np.repeat(snp_names,nt),'trait':np.tile(trait_names,nvar),**{m:a.ravel(order='C') for m,a in zip(metrics,results)}})
    raise ValueError("dtype must be 'tuple', 'xarray', 'xarray_dataset', 'pandas', or 'pandas_highmem'")


def mvGWAS(traitdf, genotypes='genotypes/genotypes', grms_folder='grm', save_path='results/gwas_parquet/', save=False, y_correction='blup_resid',
           y_correction_multivariate=False, return_table=True, stat='ttest', chrset=None, dtype='pandas', dof='correct', snp_block_size=50000000.0,
           gwa_center=True, gwa_scale=False, regression_mode='blas', covariance_estimator='psd', npplink_h2_ncomponents=3000,
           gwas_method='loco_score', covariance_kwargs=None):
    """v3 mvGWAS. `loco_score` is the statistically correct trait-wise LMM scan; `legacy` reproduces v2."""
    if gwas_method in {'legacy','residualized_approx'}:
        return _mvGWAS_v2(traitdf,genotypes=genotypes,grms_folder=grms_folder,save_path=save_path,save=save,y_correction=y_correction,
            y_correction_multivariate=y_correction_multivariate,return_table=return_table,stat=stat,chrset=chrset,dtype=dtype,dof=dof,
            snp_block_size=snp_block_size,gwa_center=gwa_center,gwa_scale=gwa_scale,regression_mode=regression_mode,
            covariance_estimator=covariance_estimator,npplink_h2_ncomponents=npplink_h2_ncomponents)
    if gwas_method != 'loco_score': raise ValueError("gwas_method must be 'loco_score', 'legacy', or 'residualized_approx'")
    npl=_require_npplink(); covariance_kwargs={} if covariance_kwargs is None else dict(covariance_kwargs)
    paths=glob(f'{grms_folder}/*.grm.bin') if isinstance(grms_folder,str) else list(grms_folder); gf=pd.DataFrame(paths,columns=['path']); gf.index=gf.path.str.extract(r'([\d\w_]+)chrGRM.',expand=False); gf=gf[~gf.index.isna()]
    allg=read_grm_lowmem(gf.loc['All','path'],with_weights=True,dtype=np.float32); read_gen=npl.load_plink(genotypes); chroms=[str(x) for x in read_gen[0].chrom.unique()]; sizes=read_gen[0].astype({'chrom':str}).groupby('chrom').pos.max(); out=[]
    rows=[(c,r) for c,r in gf.drop(index='All').iterrows() if chrset is None or str(c) in set(map(str,chrset))]
    if save: os.makedirs(save_path,exist_ok=True)
    for c,row in tqdm(rows,desc='mvGWAS LOCO score'):
        cg=read_grm_lowmem(row.path,with_weights=True,dtype=np.float32); isnum=str(c).isdigit()
        if not isnum: K=allg['grm']
        else:
            K=allg['grm']*allg['w']; K-=cg['grm']*cg['w']; den=allg['w']-cg['w']; np.divide(K,den,out=K,where=den!=0); K[den==0]=0.0
        ids=allg['ids'].iid; Y=traitdf.reindex(ids)
        ge=genetic_varcov_PSD(K,Y,n_boot=0,**covariance_kwargs); G,E=ge['G'],ge['E']; cc=str(c)
        if cc not in chroms:
            match=[x for x in chroms if str(x).lower()==cc.lower()]; cc=match[0] if match else cc
        stops=[(0,float(sizes[cc])+1)] if snp_block_size>=sizes[cc] else list(pairwise(np.arange(0,float(sizes[cc])+snp_block_size,snp_block_size)))
        for start,stop in stops:
            X=npl.plink2df(read_gen,c=cc,rfids=Y.index,pos_start=int(start),pos_end=int(stop)); block=_lmm_score_gwa(Y,X,K,G,E,dtype=dtype,stat=stat,center=gwa_center,scale=gwa_scale)
            if save:
                suffix='' if len(stops)==1 else f'_{int(start)}_{int(stop)}'; block.to_parquet(f'{save_path}gwas{c}{suffix}.parquet.gz',compression='gzip',engine='pyarrow',compression_level=1,use_dictionary=True)
            if return_table: out.append(block)
    if not return_table: return None
    if 'pandas' in dtype: return pd.concat(out,ignore_index=True) if out else pd.DataFrame()
    return out

# --- packed GCTA GRM I/O used internally by v3 -----------------------------
def read_grm_lowmem(path, *, with_weights=True, dtype=np.float32):
    """Read GCTA packed lower-triangle GRM without allocating np.tril_indices arrays."""
    path = str(path).replace('.N.bin','').replace('.bin','')
    ids = pd.read_csv(path + '.id', sep=r'\s+', header=None, names=['fid','iid'], dtype=str)
    n=len(ids); packed=np.memmap(path+'.bin',mode='r',dtype=np.float32); expected=n*(n+1)//2
    if packed.size < expected: raise ValueError(f'GRM file {path}.bin is shorter than expected')
    K=np.empty((n,n),dtype=dtype); off=0
    for i in range(n):
        row=np.asarray(packed[off:off+i+1],dtype=dtype); K[i,:i+1]=row; K[:i,i]=row[:i]; off+=i+1
    out={'grm':K,'ids':ids,'path':path}
    if with_weights:
        wp=np.memmap(path+'.N.bin',mode='r',dtype=np.float32); W=np.empty((n,n),dtype=dtype); off=0
        for i in range(n):
            row=np.asarray(wp[off:off+i+1],dtype=dtype); W[i,:i+1]=row; W[:i,i]=row[:i]; off+=i+1
        out['w']=W
    return out

# --- BLAS/GEMM moment kernels (same exact equations, less einsum overhead) ---
def grm_moment_matrices_einsum_dense(grm, traits: pd.DataFrame, *, accum_dtype=np.float64, square_dtype=None):
    grm=np.asarray(grm); n=grm.shape[0]
    if grm.shape!=(n,n) or len(traits)!=n: raise ValueError('traits and square grm must have the same number of rows')
    Y0,R=_prepare_traits(traits,dtype=accum_dtype); K=np.asarray(grm,dtype=accum_dtype,order='C'); sd=accum_dtype if square_dtype is None else square_dtype
    K2=np.asarray(grm,dtype=sd,order='C').copy(); np.square(K2,out=K2); d=np.asarray(np.diag(grm),dtype=accum_dtype)
    A11=R.T@(K2@R); A12=R.T@(d[:,None]*R); A22=R.T@R; B1=Y0.T@(K@Y0); B2=Y0.T@Y0
    return {'A11':_symmetrize(A11),'A12':_symmetrize(A12),'A22':_symmetrize(A22),'B1':_symmetrize(B1),'B2':_symmetrize(B2),'Y0':Y0,'R':R}


def grm_moment_matrices_einsum_blocked(grm, traits: pd.DataFrame, *, block_size=1024, block_dtype=np.float32, accum_dtype=np.float64):
    grm=np.asarray(grm); n=grm.shape[0]
    if grm.shape!=(n,n) or len(traits)!=n: raise ValueError('traits and square grm must have the same number of rows')
    Y0,R=_prepare_traits(traits,dtype=accum_dtype); t=Y0.shape[1]; A11=np.zeros((t,t),dtype=accum_dtype); B1=np.zeros_like(A11); d=np.asarray(np.diag(grm),dtype=accum_dtype)
    A12=R.T@(d[:,None]*R); A22=R.T@R; B2=Y0.T@Y0
    for i0 in range(0,n,int(block_size)):
        sl=slice(i0,min(i0+int(block_size),n)); Kb=np.array(grm[sl,:],dtype=block_dtype,order='C',copy=True); B1+=Y0[sl].T@(Kb@Y0); np.square(Kb,out=Kb); A11+=R[sl].T@(Kb@R)
    return {'A11':_symmetrize(A11),'A12':_symmetrize(A12),'A22':_symmetrize(A22),'B1':_symmetrize(B1),'B2':_symmetrize(B2),'Y0':Y0,'R':R}

# ============================================================================
# v3.1 production extensions: fast GWAS + cross-fitted BLUP residual experts
# ============================================================================
_V3_VERSION = '3.1.0'
_mvGWAS_v30 = mvGWAS
_auxiliary_impute_v30 = auxiliary_impute


def _format_gwa_results(results,snp_names,trait_names,dtype='pandas'):
    metrics=['beta','beta_se','stat','neglog_p','dof']; nvar=len(snp_names); nt=len(trait_names)
    if dtype=='tuple': return results
    if dtype=='xarray_dataset':
        import xarray as xr
        return xr.Dataset({m:(('snp','trait'),a) for m,a in zip(metrics,results)},coords={'snp':list(snp_names),'trait':list(trait_names)})
    if dtype=='xarray':
        import xarray as xr
        return xr.DataArray(np.stack(results),dims=['metric','snp','trait'],coords={'metric':metrics,'snp':list(snp_names),'trait':list(trait_names)})
    if dtype=='pandas_highmem':
        import xarray as xr
        A=xr.DataArray(np.stack(results),dims=['metric','snp','trait'],coords={'metric':metrics,'snp':list(snp_names),'trait':list(trait_names)})
        return A.to_dataset(dim='metric').to_dataframe().reset_index()
    if dtype=='pandas': return pd.DataFrame({'snp':np.repeat(list(snp_names),nt),'trait':np.tile(list(trait_names),nvar),**{m:a.ravel(order='C') for m,a in zip(metrics,results)}})
    raise ValueError("dtype must be 'tuple', 'xarray', 'xarray_dataset', 'pandas', or 'pandas_highmem'")


def _center_genotype_block(X0,scale=False):
    X=np.asarray(X0,dtype=float).copy(); valid=np.isfinite(X); counts=valid.sum(0); means=np.divide(np.nansum(X,axis=0),counts,out=np.zeros(X.shape[1]),where=counts>0); X-=means[None,:]; X[~valid]=0
    if scale:
        sd=np.sqrt(np.divide(np.sum(X*X,axis=0),np.maximum(counts-1,1))); X=np.divide(X,sd[None,:],out=np.zeros_like(X),where=sd[None,:]>0)
    return X,counts,valid


def _trait_null_state(y,K,vg,ve,*,center=True,jitter=1e-8):
    obs=np.flatnonzero(np.isfinite(y)); n=len(obs)
    if n<4:return None
    yy=np.asarray(y[obs],float);V=float(vg)*np.asarray(K[np.ix_(obs,obs)],float);V.flat[::n+1]+=float(ve)+float(jitter);fac=cho_factor(_symmetrize(V),lower=True,check_finite=False);L=fac[0];zy=solve_triangular(L,yy,lower=fac[1],check_finite=False)
    if center:
        z1=solve_triangular(L,np.ones(n),lower=fac[1],check_finite=False);den=float(z1@z1);zy=zy-z1*((z1@zy)/den)
    else:z1=den=None
    Vy=cho_solve(fac,yy,check_finite=False)
    if center:
        one=np.ones(n);V1=cho_solve(fac,one,check_finite=False);Py=Vy-V1*((one@Vy)/(one@V1))
    else:Py=Vy
    return {'obs':obs,'n':n,'fac':fac,'L':L,'zy':zy,'z1':z1,'den':den,'Py':Py}


def _trait_exact_block_from_state(state,X,counts,*,center=True,stat='ttest'):
    from scipy.stats import t as student_t
    ZX=solve_triangular(state['L'],X,lower=state['fac'][1],check_finite=False)
    if center:
        z1=state['z1'];ZX-=z1[:,None]*((z1@ZX)/state['den'])[None,:]
    U=ZX.T@state['zy'];I=np.sum(ZX*ZX,axis=0);dof=counts-(2 if center else 1);ok=(I>0)&(dof>0);beta=np.divide(U,I,out=np.full(len(I),np.nan),where=ok);se=np.divide(1,np.sqrt(I),out=np.full(len(I),np.nan),where=ok);z=beta/se
    if stat=='ttest':st=z;p=2*student_t.sf(np.abs(z),df=dof)
    elif stat in {'wald','score'}:st=z*z;p=erfc(np.abs(z)/np.sqrt(2))
    else:raise ValueError("stat must be 'ttest', 'wald', or 'score'")
    return beta,se,st,-np.log10(np.clip(p,np.finfo(float).tiny,1)),dof,I,U


def _prepare_trait_score_null(traits,K,G,E,*,center=True,jitter=1e-8):
    Y=traits.to_numpy(float);n,t=Y.shape;vg,ve=np.diag(np.asarray(G,float)),np.diag(np.asarray(E,float));states=[];A=np.zeros((n,t),dtype=np.float32);M=np.isfinite(Y).astype(np.float32)
    for j in range(t):
        st=_trait_null_state(Y[:,j],K,vg[j],ve[j],center=center,jitter=jitter);states.append(st)
        if st is not None:A[st['obs'],j]=st['Py'].astype(np.float32,copy=False)
    return {'states':states,'A':A,'mask':M,'trait_names':list(traits.columns),'center':center,'jitter':jitter}


def _lmm_score_gwa_exact_shared(null,snps,*,dtype='pandas',stat='ttest',scale=False):
    nt=len(null['trait_names']);X0=snps.to_numpy(float);nvar=X0.shape[1];out=[np.full((nvar,nt),np.nan) for _ in range(5)]
    for t,st in enumerate(null['states']):
        if st is None:continue
        X,counts,_=_center_genotype_block(X0[st['obs']],scale=scale);vals=_trait_exact_block_from_state(st,X,counts,center=null['center'],stat=stat)
        for k in range(5):out[k][:,t]=vals[k]
    return _format_gwa_results(tuple(out),snps.columns,null['trait_names'],dtype)


def _calibration_bins(q,ratio,n_bins=4):
    q=np.asarray(q,float);ratio=np.asarray(ratio,float);ok=np.isfinite(q)&np.isfinite(ratio)&(q>0)&(ratio>0)
    if ok.sum()<4:return np.array([-np.inf,np.inf]),np.array([float(np.nanmedian(ratio[ok])) if ok.any() else 1.])
    nb=max(1,min(int(n_bins),ok.sum()//4));edges=np.unique(np.quantile(q[ok],np.linspace(0,1,nb+1)))
    if len(edges)<2:return np.array([-np.inf,np.inf]),np.array([float(np.nanmedian(ratio[ok]))])
    edges[0],edges[-1]=-np.inf,np.inf;vals=[]
    for b in range(len(edges)-1):
        z=ok&(q>edges[b])&(q<=edges[b+1]);vals.append(float(np.nanmedian(ratio[z])) if z.any() else float(np.nanmedian(ratio[ok])))
    return edges,np.asarray(vals)


def _apply_calibration_bins(q,edges,vals):
    b=np.searchsorted(edges[1:-1],q,side='right');return vals[np.clip(b,0,len(vals)-1)]


def _lmm_score_gwa_fast_shared(null,snps,*,dtype='pandas',stat='ttest',scale=False,calibration_n=96,calibration_bins=4,rescore_p=1e-4,rescore_top=64):
    from scipy.stats import t as student_t
    Xraw=snps.to_numpy(np.float32);valid=np.isfinite(Xraw);nvar=Xraw.shape[1];cnt=valid.sum(0);mu=np.divide(np.nansum(Xraw,axis=0,dtype=np.float64),cnt,out=np.zeros(nvar),where=cnt>0).astype(np.float32);X=Xraw-mu[None,:];X[~valid]=0
    if scale:
        sd=np.sqrt(np.divide(np.sum(X*X,axis=0,dtype=np.float64),np.maximum(cnt-1,1)));X=np.divide(X,sd[None,:],out=np.zeros_like(X),where=sd[None,:]>0)
    U=(X.T@null['A']).astype(np.float64);M=null['mask'];s1=(X.T@M).astype(np.float64);s2=((X*X).T@M).astype(np.float64);nobs=M.sum(0).astype(float);q=np.maximum(s2-np.divide(s1*s1,nobs[None,:],out=np.zeros_like(s1),where=nobs[None,:]>0),0);nt=U.shape[1];out=[np.full((nvar,nt),np.nan) for _ in range(5)]
    for t,st0 in enumerate(null['states']):
        if st0 is None:continue
        eligible=np.flatnonzero(q[:,t]>0);nc=min(int(calibration_n),len(eligible))
        if nc:
            cal=eligible[np.unique(np.linspace(0,len(eligible)-1,nc,dtype=int))];Xc,cc,_=_center_genotype_block(Xraw[st0['obs']][:,cal],scale=scale);ex=_trait_exact_block_from_state(st0,Xc,cc,center=null['center'],stat='score');ratio=np.divide(ex[5],q[cal,t],out=np.full(len(cal),np.nan),where=q[cal,t]>0);edges,gam=_calibration_bins(q[cal,t],ratio,calibration_bins);I=q[:,t]*_apply_calibration_bins(q[:,t],edges,gam)
        else:I=q[:,t].copy()
        dofv=np.full(nvar,max(int(nobs[t])-(2 if null['center'] else 1),0),float);ok=(I>0)&(dofv>0);b=np.divide(U[:,t],I,out=np.full(nvar,np.nan),where=ok);se=np.divide(1,np.sqrt(I),out=np.full(nvar,np.nan),where=ok);z=b/se
        if stat=='ttest':stv=z;p=2*student_t.sf(np.abs(z),df=dofv)
        elif stat in {'wald','score'}:stv=z*z;p=erfc(np.abs(z)/np.sqrt(2))
        else:raise ValueError("stat must be 'ttest', 'wald', or 'score'")
        cand=np.flatnonzero(np.isfinite(p)&(p<float(rescore_p)))
        if int(rescore_top)>0:
            finite=np.flatnonzero(np.isfinite(p));cand=np.unique(np.r_[cand,finite[np.argsort(p[finite])[:min(int(rescore_top),len(finite))]]])
        if len(cand):
            Xc,cc,_=_center_genotype_block(Xraw[st0['obs']][:,cand],scale=scale);ex=_trait_exact_block_from_state(st0,Xc,cc,center=null['center'],stat=stat);b[cand],se[cand],stv[cand],nlp,dofv[cand]=ex[:5];p[cand]=10**(-nlp)
        out[0][:,t],out[1][:,t],out[2][:,t],out[3][:,t],out[4][:,t]=b,se,stv,-np.log10(np.clip(p,np.finfo(float).tiny,1)),dofv
    return _format_gwa_results(tuple(out),snps.columns,null['trait_names'],dtype)


# Full multivariate observed-space score null
def _mv_obs_scatter(v,plan,n,t):
    A=np.zeros((n,t),float)
    for b,(ib,sl) in enumerate(zip(plan['idx'],plan['slices'])):A[ib,b]=v[sl]
    return A


def _mv_solver_state(K,G,E,plan,*,jitter=1e-8,tol=2e-6,maxiter=400,preconditioner='diag'):
    A=_observed_cov_operator(K,G,E,plan,dtype=np.float64)
    if preconditioner=='trait_block':M=_trait_block_preconditioner(K,G,E,plan,jitter=jitter,dtype=np.float64)
    elif preconditioner=='diag':M=_diag_observed_preconditioner(K,G,E,plan,jitter=jitter,dtype=np.float64)
    elif preconditioner in {None,'none'}:M=None
    else:raise ValueError("preconditioner must be 'trait_block', 'diag', or 'none'")
    def solve(v):
        nit=[0]
        def cb(_):nit[0]+=1
        x,info=cg(A,np.asarray(v,float),M=M,rtol=float(tol),atol=0,maxiter=int(maxiter),callback=cb);res=np.linalg.norm(A@x-v)/max(np.linalg.norm(v),np.finfo(float).eps)
        if info!=0 or res>max(20*tol,2e-4):raise RuntimeError(f'multivariate null PCG failed: info={info}, relres={res}, iter={nit[0]}')
        return x,{'cg_info':int(info),'cg_iterations':int(nit[0]),'cg_relative_residual':float(res)}
    return {'solve':solve,'preconditioner':preconditioner,'tol':tol,'maxiter':maxiter}


def _prepare_mv_score_null(traits,K,G,E,*,center=True,jitter=1e-8,solver_tol=2e-6,solver_maxiter=400,preconditioner='diag',project_intercepts=False):
    Y=traits.to_numpy(float);means=np.nanmean(Y,axis=0) if center else np.zeros(Y.shape[1]);means=np.where(np.isfinite(means),means,0);Yc=Y-means[None,:];plan=_observed_plan(Yc);n,t=Y.shape;Gf=np.asarray(G,float);Ef=np.asarray(E,float);solver=_mv_solver_state(K,Gf,Ef,plan,jitter=jitter,tol=solver_tol,maxiter=solver_maxiter,preconditioner=preconditioner);vy,info=solver['solve'](plan['y']);VinvC=Hinv=None
    if project_intercepts:
        VinvC=np.empty((plan['m'],t),float)
        for j,sl in enumerate(plan['slices']):
            c=np.zeros(plan['m']);c[sl]=1;VinvC[:,j],_=solver['solve'](c)
        H=np.zeros((t,t))
        for j,sl in enumerate(plan['slices']):H[j]=VinvC[sl].sum(axis=0)
        Hinv=np.linalg.pinv(_symmetrize(H),rcond=1e-10);q=np.array([vy[sl].sum() for sl in plan['slices']]);py=vy-VinvC@(Hinv@q)
    else:py=vy
    return {'Y':Y,'Yc':Yc,'means':means,'plan':plan,'K':K,'G':Gf,'E':Ef,'py':py,'A':_mv_obs_scatter(py,plan,n,t),'VinvC':VinvC,'Hinv':Hinv,'project_intercepts':project_intercepts,'solver':solver,'solve_info':info,'jitter':jitter}


def _mv_apply_P(v,null):
    w,_=null['solver']['solve'](v)
    if null['VinvC'] is None:return w
    q=np.array([w[sl].sum() for sl in null['plan']['slices']]);return w-null['VinvC']@(null['Hinv']@q)


def _mv_exact_information_for_snp(x,null):
    plan=null['plan'];t=null['G'].shape[0];D=np.zeros((plan['m'],t))
    for j,(ib,sl) in enumerate(zip(plan['idx'],plan['slices'])):D[sl,j]=x[ib]
    I=np.empty((t,t))
    for j in range(t):I[:,j]=D.T@_mv_apply_P(D[:,j],null)
    return _symmetrize(I)


def _robust_score_cov(U,ss,calibration_n=1024):
    eligible=np.flatnonzero(np.isfinite(ss)&(ss>0)&np.all(np.isfinite(U),axis=1))
    if not len(eligible):return np.eye(U.shape[1]),np.array([],int)
    nc=min(int(calibration_n),len(eligible));idx=eligible[np.unique(np.linspace(0,len(eligible)-1,nc,dtype=int))];Z=U[idx]/np.sqrt(ss[idx])[:,None];norms=np.sum(Z*Z,axis=1)
    if len(Z)>20:Z=Z[norms<=np.quantile(norms,.98)]
    M=np.cov(Z,rowvar=False,ddof=1) if len(Z)>1 else np.eye(U.shape[1]);return _project_psd(M,eps=max(1e-10,1e-8*np.trace(M)/max(1,M.shape[0]))),idx


def _mv_score_block_fast(null,snps,*,dtype='pandas',calibration_n=2048,exact_rescore_p=5e-7,exact_rescore_top=0):
    from scipy.stats import chi2
    Xraw=snps.to_numpy(np.float32);valid=np.isfinite(Xraw);cnt=valid.sum(0);mu=np.divide(np.nansum(Xraw,axis=0,dtype=np.float64),cnt,out=np.zeros(Xraw.shape[1]),where=cnt>0).astype(np.float32);X=Xraw-mu[None,:];X[~valid]=0;ss=np.sum(X*X,axis=0,dtype=np.float64);U=(X.T@np.asarray(null['A'],np.float32)).astype(np.float64);M,cal=_robust_score_cov(U,ss,calibration_n);Mi=np.linalg.pinv(M,rcond=1e-10);Z=np.divide(U,np.sqrt(ss)[:,None],out=np.zeros_like(U),where=np.sqrt(ss)[:,None]>0);Q=np.einsum('ij,jk,ik->i',Z,Mi,Z,optimize=True);rank=max(1,int(np.linalg.matrix_rank(M)));qcal=Q[cal] if len(cal) else Q[np.isfinite(Q)];lam=max(float(np.nanmedian(qcal)/chi2.ppf(.5,rank)),1e-6);Q/=lam;p=chi2.sf(Q,rank);idiag=np.clip(np.diag(M)[None,:]*ss[:,None],1e-16,None);beta=U/idiag;se=1/np.sqrt(idiag);z=beta/se;cand=np.flatnonzero(np.isfinite(p)&(p<float(exact_rescore_p)))
    if int(exact_rescore_top)>0:
        finite=np.flatnonzero(np.isfinite(p));cand=np.unique(np.r_[cand,finite[np.argsort(p[finite])[:min(int(exact_rescore_top),len(finite))]]])
    exact={}
    for j in cand:
        I=_mv_exact_information_for_snp(X[:,j].astype(float),null);Ii=np.linalg.pinv(I,rcond=1e-10);r=max(1,int(np.linalg.matrix_rank(I)));b=Ii@U[j];s=np.sqrt(np.clip(np.diag(Ii),1e-30,None));zz=b/s;q=float(U[j]@Ii@U[j]);beta[j],se[j],z[j],Q[j],p[j]=b,s,zz,q,chi2.sf(q,r);exact[int(j)]={'rank':r}
    nt=U.shape[1];nvar=len(ss);nobs=np.asarray([len(i) for i in null['plan']['idx']],float);dof=np.broadcast_to(np.maximum(nobs-2,1)[None,:],(nvar,nt)).copy();rows=_format_gwa_results((beta,se,z*z,-np.log10(np.clip(erfc(np.abs(z)/np.sqrt(2)),np.finfo(float).tiny,1)),dof),snps.columns,null.get('trait_names',list(range(nt))),'pandas');odof=np.full(nvar,rank,float)
    for j,v in exact.items():odof[j]=v['rank']
    omni=pd.DataFrame({'snp':list(snps.columns),'trait':'__omnibus__','beta':np.nan,'beta_se':np.nan,'stat':Q,'neglog_p':-np.log10(np.clip(p,np.finfo(float).tiny,1)),'dof':odof});return pd.concat([rows,omni],ignore_index=True),{'score_cov':M,'n_calibration':len(cal),'n_exact_rescored':len(cand),'lambda_gc':lam,'exact':exact}

def _mv_score_block_exact(null,snps,*,dtype='pandas'):
    from scipy.stats import chi2
    X,_,_=_center_genotype_block(snps.to_numpy(float));U=X.T@null['A'];nt=U.shape[1];nvar=X.shape[1];beta=np.full((nvar,nt),np.nan);se=np.full_like(beta,np.nan);stat=np.full_like(beta,np.nan);nlp=np.full_like(beta,np.nan);dof=np.full_like(beta,np.nan);oq=np.full(nvar,np.nan);op=np.full(nvar,np.nan);odf=np.full(nvar,np.nan)
    for j in range(nvar):
        I=_mv_exact_information_for_snp(X[:,j],null);Ii=np.linalg.pinv(I,rcond=1e-10);r=max(1,int(np.linalg.matrix_rank(I)));b=Ii@U[j];s=np.sqrt(np.clip(np.diag(Ii),1e-30,None));z=b/s;beta[j],se[j],stat[j],nlp[j],dof[j]=b,s,z*z,-np.log10(np.clip(erfc(np.abs(z)/np.sqrt(2)),np.finfo(float).tiny,1)),r;oq[j]=float(U[j]@Ii@U[j]);op[j]=chi2.sf(oq[j],r);odf[j]=r
    rows=_format_gwa_results((beta,se,stat,nlp,dof),snps.columns,null.get('trait_names',list(range(nt))),'pandas');omni=pd.DataFrame({'snp':list(snps.columns),'trait':'__omnibus__','beta':np.nan,'beta_se':np.nan,'stat':oq,'neglog_p':-np.log10(np.clip(op,np.finfo(float).tiny,1)),'dof':odf});return pd.concat([rows,omni],ignore_index=True)


# BLUP + residual learning
class MaskedMultiTaskElasticNet:
    def __init__(self,alpha=.02,l1_ratio=.25,max_iter=1000,tol=1e-5):self.alpha=alpha;self.l1_ratio=l1_ratio;self.max_iter=max_iter;self.tol=tol
    def fit(self,X,Y):
        X=np.asarray(X,float);Y=np.asarray(Y,float);self.xmean_=X.mean(0);self.xscale_=X.std(0);self.xscale_[self.xscale_<1e-10]=1;Z=(X-self.xmean_)/self.xscale_;M=np.isfinite(Y);self.ymean_=np.divide(np.nansum(Y,axis=0),M.sum(0),out=np.zeros(Y.shape[1]),where=M.sum(0)>0);Y0=np.where(M,Y-self.ymean_[None,:],0);n,p=Z.shape;t=Y.shape[1];B=np.zeros((p,t));A=B.copy();q=1.;L=np.linalg.norm(Z,2)**2/max(n,1)+1e-8;step=1/L;l1=self.alpha*self.l1_ratio*step;lg=self.alpha*(1-self.l1_ratio)*step
        for _ in range(int(self.max_iter)):
            R=(Z@A-Y0)*M;C=A-step*(Z.T@R/max(n,1));C=np.sign(C)*np.maximum(np.abs(C)-l1,0);nr=np.linalg.norm(C,axis=1);C*=np.maximum(0,1-lg/np.maximum(nr,1e-30))[:,None];q2=(1+np.sqrt(1+4*q*q))/2;A2=C+((q-1)/q2)*(C-B);rel=np.linalg.norm(C-B)/max(np.linalg.norm(B),1e-8);B,A,q=C,A2,q2
            if rel<self.tol:break
        self.coef_=B;return self
    def predict(self,X):return ((np.asarray(X,float)-self.xmean_)/self.xscale_)@self.coef_+self.ymean_[None,:]


def _genetic_canonical_basis(G,E,jitter=1e-6):
    G=np.asarray(G,float);E=np.asarray(E,float);scale=max(float(np.nanmedian(np.diag(E))),1e-6);Er=_symmetrize(E)+float(jitter)*scale*np.eye(E.shape[0])
    try:rho,V=scipy_eigh(_symmetrize(G),Er,check_finite=False);o=np.argsort(rho)[::-1];return V[:,o],np.clip(rho[o],0,None)
    except Exception:w,V=np.linalg.eigh(_project_psd(G));o=np.argsort(w)[::-1];return V[:,o],np.clip(w[o],0,None)


def _expert_features(Y,B,G,E,target=None,*,phenotype_assisted=False,genomic_features=None,n_ge_components=8):
    Ya=Y.to_numpy(float) if isinstance(Y,pd.DataFrame) else np.asarray(Y,float);Ba=B.to_numpy(float) if isinstance(B,pd.DataFrame) else np.asarray(B,float);t=Ba.shape[1];V,_=_genetic_canonical_basis(G,E);k=min(int(n_ge_components),t);parts=[Ba,Ba@V[:,:k]]
    if genomic_features is not None:parts.append(np.asarray(genomic_features,float))
    if phenotype_assisted:
        R=Ya-Ba;M=np.isfinite(R).astype(float);R=np.nan_to_num(R,nan=0)
        if target is not None:R[:,int(target)]=0;M[:,int(target)]=0
        Re=R@_invsqrt_psd(np.asarray(E,float)+1e-6*np.eye(t))
        if target is not None:Re[:,int(target)]=0
        parts.extend([R,Re,M])
    return np.concatenate(parts,axis=1)


def _fit_single_residual_model(method,X,y,*,random_state=0,n_components=8,model_kwargs=None):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet,LinearRegression
    model_kwargs={} if model_kwargs is None else dict(model_kwargs);method=str(method).lower();n=len(y)
    if method=='pcr':mdl=make_pipeline(StandardScaler(),PCA(max(1,min(int(n_components),X.shape[1],n-2)),random_state=random_state),LinearRegression())
    elif method=='pls':mdl=make_pipeline(StandardScaler(),PLSRegression(n_components=max(1,min(int(n_components),X.shape[1],n-2)),scale=False,max_iter=500))
    elif method=='elasticnet':
        pars=dict(alpha=.02,l1_ratio=.25,max_iter=5000,random_state=random_state);pars.update(model_kwargs);mdl=make_pipeline(StandardScaler(),ElasticNet(**pars))
    elif method=='lightgbm':
        from lightgbm import LGBMRegressor
        pars=dict(n_estimators=72,num_leaves=15,max_depth=5,learning_rate=.04,verbosity=-1,random_state=random_state,n_jobs=1,min_child_samples=15);pars.update(model_kwargs);mdl=LGBMRegressor(**pars)
    else:raise ValueError("method must be 'pcr', 'pls', 'elasticnet', or 'lightgbm'")
    mdl.fit(X,y);return mdl


def _predict_model(mdl,X):
    if mdl.__class__.__module__.startswith('lightgbm'):return np.asarray(mdl.predict(X,validate_features=False)).reshape(-1)
    return np.asarray(mdl.predict(X)).reshape(-1)


def _corr_rmse(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);ok=np.isfinite(y)&np.isfinite(p);n=int(ok.sum())
    return {'n':n,'r':float(np.corrcoef(y[ok],p[ok])[0,1]) if n>1 else np.nan,'rmse':float(np.sqrt(np.mean((y[ok]-p[ok])**2))) if n else np.nan}


def _paired_rmse_gain(y,base,pred):
    y=np.asarray(y,float);base=np.asarray(base,float);pred=np.asarray(pred,float);eb=(y-base)**2;ep=(y-pred)**2;d=eb-ep;rb=np.sqrt(np.mean(eb));rp=np.sqrt(np.mean(ep));gain=float((rb-rp)/max(rb,1e-12));se=np.std(d,ddof=1)/np.sqrt(len(d)) if len(d)>1 else np.inf;z=float(np.mean(d)/se) if se>0 and np.isfinite(se) else (np.inf if np.mean(d)>0 else -np.inf);return gain,z


def _split_rows(n,n_splits,seed):
    rng=np.random.default_rng(seed);o=rng.permutation(n);f=np.empty(n,int)
    for k,z in enumerate(np.array_split(o,int(n_splits))):f[z]=k
    return f


def _split_cells(Y,n_splits,seed):
    rng=np.random.default_rng(seed);F=np.full(Y.shape,-1,np.int16)
    for j in range(Y.shape[1]):
        idx=np.flatnonzero(np.isfinite(Y[:,j]));rng.shuffle(idx)
        for k,z in enumerate(np.array_split(idx,int(n_splits))):F[z,j]=k
    return F


def _fit_global_residual_models(Xlist,ylist,methods,*,random_state=0,n_components=8,model_min_gain=.005,model_min_z=1.96,stack_min_gain=.02,stack_min_z=1.96,selection_cv=3,max_stack_methods=2,single_tie_tolerance=.005,model_kwargs=None):
    from sklearn.model_selection import KFold
    from scipy.optimize import nnls
    methods=tuple(methods);cvs=[];ys=[]
    for j,(X,y) in enumerate(zip(Xlist,ylist)):
        if X is None or y is None or len(y)<20:cvs.append(None);continue
        X=np.asarray(X,float);y=np.asarray(y,float);P=np.full((len(y),len(methods)),np.nan);kf=KFold(max(2,min(int(selection_cv),max(2,len(y)//20))),shuffle=True,random_state=random_state+j)
        for mi,m in enumerate(methods):
            try:
                for tr,va in kf.split(X):P[va,mi]=_predict_model(_fit_single_residual_model(m,X[tr],y[tr],random_state=random_state+100+j,n_components=n_components,model_kwargs=(model_kwargs or {}).get(m)),X[va])
            except Exception:pass
        cvs.append(P);ys.append((j,y))
    if not ys:return {},{'mode':'blup','weights':{},'scores':{}}
    yall=np.concatenate([y for _,y in ys]);Pall=np.vstack([cvs[j] for j,_ in ys]);scores={'blup':_corr_rmse(yall,np.zeros(len(yall)))};good=[]
    for mi,m in enumerate(methods):
        p=Pall[:,mi];ok=np.isfinite(p)
        if ok.sum()<20:scores[m]={'n':int(ok.sum()),'r':np.nan,'rmse':np.inf,'gain_vs_blup':0,'gain_z':-np.inf};continue
        scores[m]=_corr_rmse(yall[ok],p[ok]);gain,z=_paired_rmse_gain(yall[ok],np.zeros(ok.sum()),p[ok]);scores[m].update(gain_vs_blup=gain,gain_z=z)
        if gain>=model_min_gain and z>=model_min_z:good.append(m)
    if not good:return {},{'mode':'blup','weights':{},'scores':scores,'best_single':'blup','stack_gain':0}
    best_rmse=min(scores[m]['rmse'] for m in good);near=[m for m in good if scores[m]['rmse']<=best_rmse*(1+float(single_tie_tolerance))];best=max(near,key=lambda m:scores[m]['gain_z']);cols=[methods.index(m) for m in good];P=Pall[:,cols];ok=np.isfinite(P).all(1);y=yall[ok];P=P[ok];rng=np.random.default_rng(random_state+7001);o=rng.permutation(len(y));nv=max(20,len(y)//3);va=o[:nv];tr=o[nv:];w,_=nnls(P[tr],y[tr]);w=w/w.sum() if w.sum()>0 else np.zeros(len(good));ki=np.argsort(w)[::-1][:max(1,min(int(max_stack_methods),len(w)))];w2,_=nnls(P[tr][:,ki],y[tr]);w2=w2/w2.sum() if w2.sum()>0 else np.zeros(len(ki));w=np.zeros(len(good));w[ki]=w2;stack=P[va]@w;sg,sz=_paired_rmse_gain(y[va],P[va,good.index(best)],stack)
    if sg>=stack_min_gain and sz>=stack_min_z:mode='stack';weights={m:float(a) for m,a in zip(good,w) if a>1e-6};selected=list(weights)
    else:mode='single';weights={best:1.};selected=[best]
    models={}
    for j,(X,y) in enumerate(zip(Xlist,ylist)):
        if X is None or y is None or len(y)<20:continue
        models[j]={'mode':mode,'weights':weights.copy(),'models':{m:_fit_single_residual_model(m,X,y,random_state=random_state+900+j,n_components=n_components,model_kwargs=(model_kwargs or {}).get(m)) for m in selected},'best_single':best,'global':True}
    return models,{'mode':mode,'weights':weights,'scores':scores,'best_single':best,'stack_gain':float(sg),'stack_gain_z':float(sz),'selected':selected}


def fit_blup_residual_experts(traits,GRM,G,E,*,blup=None,n_splits=3,methods=('pcr','pls','elasticnet','lightgbm'),n_components=8,random_state=0,stack_min_gain=.02,genomic_min_gain=.03,phenotype_min_gain=.005,genomic_min_z=2.33,phenotype_min_z=1.96,genomic_features=None,model_kwargs=None,solver='pcg',solver_tol=2e-6,solver_maxiter=500,preconditioner='diag',crossfit_refit_covariance=True,covariance_kwargs=None,allow_blup_only_nonlinear=False,store_oof=False):
    Y=traits.copy() if isinstance(traits,pd.DataFrame) else pd.DataFrame(traits);n,t=Y.shape;Gf=np.asarray(G,float);Ef=np.asarray(E,float);K=GRM;covariance_kwargs={} if covariance_kwargs is None else dict(covariance_kwargs);B=mvBLUP(Y,K,Gf,Ef,missing='exact',solver=solver,solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner) if blup is None else (blup.reindex_like(Y) if isinstance(blup,pd.DataFrame) else pd.DataFrame(blup,index=Y.index,columns=Y.columns))
    def fold_blup(Yk):
        if crossfit_refit_covariance:
            ge=genetic_varcov_PSD(K,Yk,n_boot=0,**covariance_kwargs);Gk,Ek=ge['G'],ge['E']
        else:Gk,Ek=Gf,Ef
        return mvBLUP(Yk,K,Gk,Ek,missing='exact',solver=solver,solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner),np.asarray(Gk,float),np.asarray(Ek,float)
    rf=_split_rows(n,n_splits,random_state);gx=[[] for _ in range(t)];gy=[[] for _ in range(t)];Xo=None;Ro=np.full((n,t),np.nan)
    for k in range(int(n_splits)):
        val=rf==k;Yk=Y.copy();Yk.iloc[val,:]=np.nan;Bk,Gk,Ek=fold_blup(Yk);Xg=_expert_features(Yk,Bk,Gk,Ek,phenotype_assisted=False,genomic_features=genomic_features,n_ge_components=n_components)
        if Xo is None:Xo=np.full((n,Xg.shape[1]),np.nan)
        Xo[val]=Xg[val];Ro[val]=Y.to_numpy(float)[val]-Bk.to_numpy(float)[val]
        for j in range(t):
            rows=np.flatnonzero(val&Y.iloc[:,j].notna().to_numpy())
            if len(rows):gx[j].append(Xg[rows]);gy[j].append(Y.iloc[rows,j].to_numpy(float)-Bk.iloc[rows,j].to_numpy(float))
    cf=_split_cells(Y.to_numpy(float),n_splits,random_state+101);px=[[] for _ in range(t)];py=[[] for _ in range(t)]
    for k in range(int(n_splits)):
        hold=cf==k;Yk=Y.mask(hold);Bk,Gk,Ek=fold_blup(Yk)
        for j in range(t):
            rows=np.flatnonzero(hold[:,j])
            if len(rows):
                Xp=_expert_features(Yk,Bk,Gk,Ek,target=j,phenotype_assisted=True,genomic_features=genomic_features,n_ge_components=n_components);px[j].append(Xp[rows]);py[j].append(Y.iloc[rows,j].to_numpy(float)-Bk.iloc[rows,j].to_numpy(float))
    gx=[np.vstack(z) if z else None for z in gx];gy=[np.concatenate(z) if z else None for z in gy];px=[np.vstack(z) if z else None for z in px];py=[np.concatenate(z) if z else None for z in py];cols=list(Y.columns)
    if genomic_features is not None or allow_blup_only_nonlinear:
        gm,gdiag=_fit_global_residual_models(gx,gy,methods,random_state=random_state,n_components=n_components,model_min_gain=genomic_min_gain,model_min_z=genomic_min_z,stack_min_gain=stack_min_gain,model_kwargs=model_kwargs);genmodels={cols[j]:gm.get(j,{'mode':'blup','models':{},'weights':{},'global':True}) for j in range(t)}
    else:gdiag={'mode':'blup','weights':{},'scores':{},'best_single':'blup','reason':'no additional genomic features beyond BLUP'};genmodels={c:{'mode':'blup','models':{},'weights':{},'global':True} for c in cols}
    pm,pdiag=_fit_global_residual_models(px,py,methods,random_state=random_state+1000,n_components=n_components,model_min_gain=phenotype_min_gain,model_min_z=phenotype_min_z,stack_min_gain=stack_min_gain,model_kwargs=model_kwargs);phenmodels={cols[j]:pm.get(j,{'mode':'blup','models':{},'weights':{},'global':True}) for j in range(t)}
    mt=None
    try:
        ok=np.isfinite(Xo).all(1)&np.isfinite(Ro).any(1)
        if (genomic_features is not None or allow_blup_only_nonlinear) and ok.sum()>30:mt=MaskedMultiTaskElasticNet().fit(Xo[ok],Ro[ok])
    except Exception:pass
    out={'blup':B,'G':pd.DataFrame(Gf,index=cols,columns=cols),'E':pd.DataFrame(Ef,index=cols,columns=cols),'genomic_models':genmodels,'phenotype_models':phenmodels,'multitask_genomic':mt,'methods':tuple(methods),'n_splits':int(n_splits),'diagnostics':{'genomic_global':gdiag,'phenotype_global':pdiag},'genomic_features':genomic_features,'n_components':int(n_components),'version':_V3_VERSION}
    if store_oof:out['oof']={'genomic_X':gx,'genomic_y':gy,'phenotype_X':px,'phenotype_y':py}
    return out


def _predict_stacked_target(fit,X):
    if not fit.get('models'):return np.zeros(X.shape[0])
    return sum(float(fit['weights'].get(m,0))*_predict_model(model,X) for m,model in fit['models'].items())


def predict_blup_residual_experts(fit,traits,*,blup=None,GRM=None,genomic_features=None,include_multitask=False):
    Y=traits.copy() if isinstance(traits,pd.DataFrame) else pd.DataFrame(traits,columns=fit['blup'].columns);G=fit['G'].to_numpy();E=fit['E'].to_numpy();B=fit['blup'].reindex(Y.index) if blup is None and GRM is None else (mvBLUP(Y,GRM,G,E,missing='exact',solver='pcg',preconditioner='diag') if blup is None else (blup.reindex_like(Y) if isinstance(blup,pd.DataFrame) else pd.DataFrame(blup,index=Y.index,columns=Y.columns)));gf=genomic_features if genomic_features is not None else fit.get('genomic_features');Xg=_expert_features(Y,B,G,E,phenotype_assisted=False,genomic_features=gf,n_ge_components=fit['n_components']);gen=B.copy();phen=B.copy()
    for j,c in enumerate(Y.columns):
        if c in fit['genomic_models']:gen[c]=B[c].to_numpy()+_predict_stacked_target(fit['genomic_models'][c],Xg)
        if c in fit['phenotype_models']:
            Xp=_expert_features(Y,B,G,E,target=j,phenotype_assisted=True,genomic_features=gf,n_ge_components=fit['n_components']);phen[c]=B[c].to_numpy()+_predict_stacked_target(fit['phenotype_models'][c],Xp)
    mt=pd.DataFrame(B.to_numpy()+fit['multitask_genomic'].predict(Xg),index=Y.index,columns=Y.columns) if include_multitask and fit.get('multitask_genomic') is not None else None;has=Y.notna().any(axis=1);hyb=gen.copy();hyb.loc[has]=phen.loc[has];return {'prediction':hyb,'genomic_only':gen,'phenotype_assisted':phen,'blup':B,'multitask_genomic':mt,'has_phenotype':has}


def auxiliary_impute(traits,blup,method='stacked',*,GRM=None,G=None,E=None,random_state=0,n_components=8,model_kwargs=None,n_splits=3,methods=('pcr','pls','elasticnet','lightgbm'),stack_min_gain=.02,genomic_features=None,solver='pcg',solver_tol=2e-6,solver_maxiter=500,preconditioner='diag'):
    if GRM is None or G is None or E is None:return _auxiliary_impute_v30(traits,blup,method='lightgbm' if method=='stacked' else method,random_state=random_state,n_components=n_components,model_kwargs=model_kwargs)
    fit=fit_blup_residual_experts(traits,GRM,G,E,blup=blup,n_splits=n_splits,methods=methods if method=='stacked' else (method,),n_components=n_components,random_state=random_state,stack_min_gain=stack_min_gain,genomic_features=genomic_features,model_kwargs=model_kwargs,solver=solver,solver_tol=solver_tol,solver_maxiter=solver_maxiter,preconditioner=preconditioner);pred=predict_blup_residual_experts(fit,traits,blup=blup,genomic_features=genomic_features);return {'imputed':pred['prediction'],'imputed_ml':pred['prediction'].copy(),'blup':pred['blup'],'models':fit,'method':method,'method_resolved':'crossfit_two_expert_residual','n_genomic_only':int((~pred['has_phenotype']).sum()),'experts':pred}


def mvGWAS(traitdf,genotypes='genotypes/genotypes',grms_folder='grm',save_path='results/gwas_parquet/',save=False,y_correction='blup_resid',y_correction_multivariate=False,return_table=True,stat='ttest',chrset=None,dtype='pandas',dof='correct',snp_block_size=50000000.,gwa_center=True,gwa_scale=False,regression_mode='blas',covariance_estimator='psd',npplink_h2_ncomponents=3000,gwas_method='loco_score_fast',covariance_kwargs=None,calibration_n=96,calibration_bins=4,rescore_p=1e-4,rescore_top=64,mv_calibration_n=2048,mv_exact_rescore_p=5e-7,mv_exact_rescore_top=0,mv_project_intercepts=False,mv_solver_tol=2e-6,mv_solver_maxiter=400,mv_preconditioner='diag',random_state=0):
    if gwas_method in {'legacy','residualized_approx'}:return _mvGWAS_v30(traitdf,genotypes=genotypes,grms_folder=grms_folder,save_path=save_path,save=save,y_correction=y_correction,y_correction_multivariate=y_correction_multivariate,return_table=return_table,stat=stat,chrset=chrset,dtype=dtype,dof=dof,snp_block_size=snp_block_size,gwa_center=gwa_center,gwa_scale=gwa_scale,regression_mode=regression_mode,covariance_estimator=covariance_estimator,npplink_h2_ncomponents=npplink_h2_ncomponents)
    gwas_method={'loco_score':'loco_score_exact'}.get(gwas_method,gwas_method)
    if gwas_method not in {'loco_score_exact','loco_score_fast','loco_mv_score_fast','loco_mv_score_exact'}:raise ValueError('unsupported gwas_method')
    npl=_require_npplink();covariance_kwargs={} if covariance_kwargs is None else dict(covariance_kwargs);paths=glob(f'{grms_folder}/*.grm.bin');gf=pd.DataFrame(paths,columns=['path']);gf.index=gf.path.str.extract(r'([\d\w_]+)chrGRM.',expand=False);gf=gf[~gf.index.isna()];allg=read_grm_lowmem(gf.loc['All','path'],with_weights=True,dtype=np.float32);read_gen=npl.load_plink(genotypes);chroms=[str(x) for x in read_gen[0].chrom.unique()];sizes=read_gen[0].astype({'chrom':str}).groupby('chrom').pos.max();out=[];rows=[(c,r) for c,r in gf.drop(index='All').iterrows() if chrset is None or str(c) in set(map(str,chrset))]
    if save:os.makedirs(save_path,exist_ok=True)
    for c,row in tqdm(rows,desc=f'mvGWAS {gwas_method}'):
        cg=read_grm_lowmem(row.path,with_weights=True,dtype=np.float32)
        if not str(c).isdigit():K=allg['grm']
        else:
            num=allg['grm']*allg['w']-cg['grm']*cg['w'];den=allg['w']-cg['w'];K=np.divide(num,den,out=np.zeros_like(num),where=den!=0)
        ids=allg['ids'].iid;Y=traitdf.reindex(ids);ge=genetic_varcov_PSD(K,Y,n_boot=0,**covariance_kwargs);G,E=ge['G'],ge['E'];cc=str(c)
        if cc not in chroms:
            z=[x for x in chroms if x.lower()==cc.lower()];cc=z[0] if z else cc
        if gwas_method in {'loco_score_exact','loco_score_fast'}:null=_prepare_trait_score_null(Y,K,G,E,center=gwa_center)
        else:
            null=_prepare_mv_score_null(Y,K,G,E,center=gwa_center,solver_tol=mv_solver_tol,solver_maxiter=mv_solver_maxiter,preconditioner=mv_preconditioner,project_intercepts=(True if gwas_method=='loco_mv_score_exact' else mv_project_intercepts));null['trait_names']=list(Y.columns)
        stops=[(0,float(sizes[cc])+1)] if snp_block_size>=sizes[cc] else list(pairwise(np.arange(0,float(sizes[cc])+snp_block_size,snp_block_size)))
        for start,stop in stops:
            X=npl.plink2df(read_gen,c=cc,rfids=Y.index,pos_start=int(start),pos_end=int(stop))
            if gwas_method=='loco_score_exact':block=_lmm_score_gwa_exact_shared(null,X,dtype=dtype,stat=stat,scale=gwa_scale)
            elif gwas_method=='loco_score_fast':block=_lmm_score_gwa_fast_shared(null,X,dtype=dtype,stat=stat,scale=gwa_scale,calibration_n=calibration_n,calibration_bins=calibration_bins,rescore_p=rescore_p,rescore_top=rescore_top)
            elif gwas_method=='loco_mv_score_fast':block,_=_mv_score_block_fast(null,X,dtype=dtype,calibration_n=mv_calibration_n,exact_rescore_p=mv_exact_rescore_p,exact_rescore_top=mv_exact_rescore_top)
            else:block=_mv_score_block_exact(null,X,dtype=dtype)
            if save:
                suffix='' if len(stops)==1 else f'_{int(start)}_{int(stop)}';block.to_parquet(f'{save_path}gwas{c}{suffix}.parquet.gz',compression='gzip',engine='pyarrow',compression_level=1,use_dictionary=True)
            if return_table:out.append(block)
    if not return_table:return None
    return pd.concat(out,ignore_index=True) if 'pandas' in dtype and out else out
