import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erfc
from tqdm import tqdm
from gwas import npplink
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
