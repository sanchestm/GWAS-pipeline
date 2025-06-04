import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import chi2
# from jax.scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import jax
from jax.scipy.stats import t

@jit
def scale_with_mask(X):
    X = jnp.asarray(X, dtype=jnp.float32)
    M = ~jnp.isnan(X)
    mean = jnp.nanmean(X, axis=0)
    X_centered = jnp.where(M, X - mean, 0.0)
    sum_sq = jnp.einsum("ij,ij->j", X_centered, X_centered, optimize=True)
    std = jnp.sqrt(sum_sq / M.sum(axis=0))
    std = jnp.where(std == 0, jnp.nan, std)
    X_scaled = X_centered / std
    return X_scaled, std, M.astype(jnp.float32)


@jit
def regression_with_einsum(ssnps, straits, snps_mask, traits_mask, dof='correct'):
    XtY = jnp.einsum("ij,ik->jk", ssnps, straits, optimize=True)  # shape: (num_snps, num_traits)
    diag_XtX = jnp.einsum("ij,ik->jk", ssnps**2, traits_mask, optimize=True)
    term1 = jnp.einsum("ij,ik->jk", snps_mask, traits_mask * (straits**2), optimize=True)
    if dof != 'incorrect':
        df = jnp.einsum("ij,ik->jk", snps_mask, traits_mask, optimize=True) - 1
    else:
        df = jnp.broadcast_to(traits_mask.sum(axis=0) - 1, (ssnps.shape[1], traits_mask.shape[1]))
    df = jnp.where(df <= 0, jnp.nan, df)
    beta = XtY / diag_XtX
    SSR = term1 - 2 * beta * XtY + beta**2 * diag_XtX
    sigma2 = SSR / df
    se_beta = jnp.sqrt(sigma2 / diag_XtX)
    t_stats = beta / se_beta
    p_values = 2 * (1 - t.cdf(jnp.abs(t_stats), df))
    neg_log10_p_values = -jnp.log10(p_values)
    return beta, se_beta, t_stats, neg_log10_p_values, df


@jit
def GRM(X, scale=True, return_weights=False, nan_policy='ignore', correlation_matrix=False):
    x = jnp.asarray(X)
    z = x - jnp.nanmean(x, axis=0)
    if scale:
        z = z / jnp.nanstd(x, axis=0)
    z = jnp.nan_to_num(z, nan=0.0)
    zzt = jnp.dot(z, z.T)

    if nan_policy == 'mean':
        zzt_w = x.shape[1] - 1
    elif nan_policy in ['ignore', 'per_iid']:
        zna = (~jnp.isnan(x)).astype(jnp.float32)
        zzt_w = jnp.dot(zna, zna.T)
        zzt_w = jnp.clip(zzt_w - 1, a_min=1, a_max=jnp.inf)
        if nan_policy == 'per_iid':
            zzt_w = jnp.max(zzt_w, axis=1, keepdims=True)

    grm = zzt / zzt_w

    if correlation_matrix:
        sig = jnp.sqrt(jnp.diag(grm))
        grm = grm / jnp.outer(sig, sig)
        grm = grm.at[jnp.diag_indices(grm.shape[0])].set(1.0)

    if return_weights:
        return {'zzt': zzt, 'weights': zzt_w, 'grm': grm}
    else:
        return grm


@jit
def remove_relatedness_transformation(G=None, U=None, s=None, h2=0.5, yvar=1, tol=1e-8, n_components=None, return_eigen=False):
    if s is None and U is None and G is not None:
        G = yvar * (h2 * G + (1 - h2) * jnp.eye(G.shape[0]))
        if n_components is None:
            n_components = G.shape[0]
        U, s, _ = jnp.linalg.svd(G, full_matrices=False)
        U = U[:, :n_components]
        s = s[:n_components]
    elif s is not None and U is not None and G is None:
        U = jnp.asarray(U)
        s = jnp.asarray(s)
    else:
        raise ValueError('cannot submit both G and U s at the same time')
    eigs_fG = yvar * (h2 * s + (1 - h2))
    eigs_fG = jnp.where(eigs_fG < tol, tol, eigs_fG)
    D_inv_sqrt = jnp.diag(1 / jnp.sqrt(eigs_fG))
    if return_eigen:
        return U, D_inv_sqrt
    W = U @ D_inv_sqrt @ U.T
    return W


@jit
def H2SVD(y, grm=None, s=None, U=None, l='REML', n_components=None, return_SVD=False, tol=1e-8):
    y = jnp.asarray(y).flatten()
    notnan = ~jnp.isnan(y)
    obs = jnp.where(notnan)[0]
    y = y[obs]
    sp = jnp.nanvar(y, ddof=1)
    m = jnp.nanmean(y)
    y -= m
    N = notnan.sum()
    if s is None and U is None and grm is not None:
        grm = jnp.asarray(grm)[jnp.ix_(obs, obs)]
        if n_components is None:
            n_components = grm.shape[0]
        U, s, _ = jnp.linalg.svd(grm, full_matrices=False)
        U = U[:, :n_components]
        s = s[:n_components]
    elif s is not None and U is not None and grm is None:
        U = jnp.asarray(U)[obs, :]
        s = s * jnp.sum(U**2, axis=0)
        if n_components is not None:
            U = U[:, :n_components]
            s = s[:n_components]
    else:
        raise ValueError('cannot submit both grm and U s at the same time')
    Ur2 = jnp.dot(U.T, y)**2
    s = jnp.maximum(s, tol)

    def _L(h2):
        sg = h2 * sp
        se = sp - sg
        sgsse = sg * s + se
        log_det = jnp.sum(jnp.log(sgsse))
        quad = jnp.sum(Ur2 / sgsse)
        ll = -0.5 * (quad + log_det + N * jnp.log(2 * jnp.pi))
        if l == 'REML':
            Xt_cov_inv_X = jnp.sum(1.0 / sgsse)
            ll -= 0.5 * jnp.log(Xt_cov_inv_X)
        return -ll

    result = minimize(_L, bounds=(0.0, 1.0), method='bounded')
    h2 = result.x
    likelihood = result.fun
    if return_SVD:
        sg = h2 * sp
        se = sp * (1 - h2)
        sgsse = sg * s + se
        log_det = jnp.sum(jnp.log(sgsse))
        quad = jnp.sum(Ur2 / sgsse)
        return {'U': U, 's': s, 'h2': h2, 'L': likelihood, 'quad': quad, 'log_det': log_det}
    return h2

@jit
def R2(X, Y=None):
    x = jnp.asarray(X, dtype=jnp.float32)
    x_mask = ~jnp.isnan(x)
    x_n = jnp.sum(x_mask, axis=0)
    x_mean = jnp.nanmean(x, axis=0)
    x_centered = jnp.where(x_mask, x - x_mean, 0.0)
    x_std = jnp.sqrt(jnp.sum(x_centered**2, axis=0) / x_n)
    x_std = jnp.where(x_std == 0, jnp.nan, x_std)

    if Y is None:
        y, y_mask, y_std = x_centered, x_mask, x_std
    else:
        y = jnp.asarray(Y, dtype=jnp.float32)
        y_mask = ~jnp.isnan(y)
        y_n = jnp.sum(y_mask, axis=0)
        y_mean = jnp.nanmean(y, axis=0)
        y_centered = jnp.where(y_mask, y - y_mean, 0.0)
        y_std = jnp.sqrt(jnp.sum(y_centered**2, axis=0) / y_n)
        y_std = jnp.where(y_std == 0, jnp.nan, y_std)
        y = y_centered

    weights = jnp.dot(x_mask.T, y_mask)
    weights = jnp.where(weights == 0, jnp.nan, weights)
    cov = jnp.dot(x_centered.T, y) / weights
    corr_sq = jnp.clip((cov / jnp.outer(x_std, y_std))**2, a_min=0.0, a_max=1.0)
    return corr_sq

def off_diagonalR2(snps, snp_dist=1000, min_snp_dist=0, return_square=True):
    import nltk
    import dask.bag as db
    import dask
    step = 200
    step_list = [list(snps.columns[step*x:step*x+step]) for x in range(int(snps.shape[1]/step)+1)]
    off_diagonal_range = (int(min_snp_dist/step)+1, int(snp_dist/step)+1)
    all_ngrams = db.from_sequence(nltk.everygrams(step_list, min_len=off_diagonal_range[0], max_len=off_diagonal_range[1]))

    def _dr2(tup):
        if not len(tup):
            res = R2(snps.loc[:, tup])
        else:
            res = R2(snps.loc[:, tup[0]], snps.loc[:, tup[-1]])
        res = res.reset_index(names='bp1').melt(id_vars='bp1', var_name='bp2')
        return pd.concat([res, res.rename({'bp1': 'bp2', 'bp2': 'bp1'}, axis=1)])

    rr2 = pd.concat(db.map(_dr2, all_ngrams).compute())\
             .astype({'bp1': str, 'bp2': str, 'value': float})\
             .drop_duplicates(['bp1', 'bp2'])

    if return_square:
        rr2 = rr2.pivot(index='bp1', columns='bp2', values='value')\
                 .loc[list(snps.columns), list(snps.columns)]
    else:
        pos = len(rr2['bp1'].iloc[0].split(':')[0]) + 1
        rr2['distance'] = (rr2.bp1.str.slice(start=pos).astype(int) - rr2.bp2.str.slice(start=pos).astype(int)).abs()
    return rr2


def GWA(traits, snps, dtype='pandas'):
    # Apply scaling and masking
    ssnps, snps_std, snps_mask = scale_with_mask(snps)
    straits, traits_std, traits_mask = scale_with_mask(traits)

    # Run regression
    beta, beta_se, t_stat, dof = regression_with_einsum(ssnps, straits, snps_mask, traits_mask, dof='correct')

    # Stack results
    result_array = jnp.stack([beta, beta_se, t_stat, neglog_p, dof], axis=0)

    # Convert to xarray
    res = xr.DataArray(
        np.array(result_array),  # convert JAX to NumPy for xarray compatibility
        dims=["metric", "snp", "trait"],
        coords={
            "metric": np.array(['beta', 'beta_se', 't_stat', 'neglog_p', 'dof']),
            "snp": list(snps.columns),
            "trait": traits.columns.map(lambda x: x.split('__subtractgrm')[0]).to_list()
        }
    )

    if dtype == 'pandas':
        return res.to_dataset(dim="metric").to_dataframe().reset_index()
    else:
        return res



