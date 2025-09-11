import numpy as np
from scipy import integrate
from scipy.special import comb
from scipy.interpolate import interp1d


import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

# ---- closed-form eventual fixation at p=1 (t→∞) ----
def wf_fixation_prob(p0, Ne, s):
    p0 = np.asarray(p0, dtype=float)
    if np.isclose(s, 0.0):
        return p0
    x = -2.0 * Ne * s
    return (1.0 - np.exp(x * p0)) / (1.0 - np.exp(x))

# ---- small helpers ----
def _initial_bumps_matrix(p, p0s, width=1e-3):
    """Stacked, normalized Gaussian bumps centered at each p0 (columns)."""
    p0s = np.asarray(p0s, dtype=float).ravel()
    P, P0 = np.meshgrid(p, p0s, indexing="ij")       # shapes: (M+1, K)
    G = np.exp(-0.5 * ((P - P0) / width) ** 2)
    G[(P <= 0) | (P >= 1)] = 0.0
    # column-wise normalization
    area = np.trapz(G, p, axis=0) + 1e-300
    G /= area
    return G  # (M+1, K)

def _build_implicit_matrix(p, mu, D, dt):
    """(I - dt L) with upwind drift and central diffusion (absorbing BCs)."""
    M = p.size - 1
    h = p[1] - p[0]
    pf = 0.5 * (p[:-1] + p[1:])
    muf = np.interp(pf, p, mu)
    Df  = np.interp(pf, p, D)
    mup = np.maximum(muf, 0.0)
    mum = np.minimum(muf, 0.0)

    Dl = Df[:-1] / h
    Dr = Df[1:]  / h
    mup_l, mum_l = mup[:-1], mum[:-1]
    mup_r, mum_r = mup[1:],  mum[1:]

    a = (Dl + mup_l) / h                 # sub-diagonal
    c = -(Dr - mum_r) / h                # super-diagonal
    b = ((mum_l - Dl) - (mup_r + Dr)) / h

    main  = 1.0 - dt * b
    lower = -dt * a[1:]
    upper = -dt * c[:-1]
    A = diags([lower, main, upper], offsets=[-1, 0, 1], format="csc")
    return A, (mup, mum, Df, h)

def _boundary_fluxes_matrix(F, mup, mum, Df, h):
    """
    Outward fluxes at p=0 and p=1 for all columns.
    F shape: (M+1, K)
    Returns vectors J0, J1 of length K.
    """
    # left face 1/2 (index 0): f0 = 0
    F_left  = (mup[0] * 0.0 + mum[0] * F[1, :]) - Df[0] * (F[1, :] - 0.0) / h
    # right face M-1/2 (index -1): fM = 0
    F_right = (mup[-1] * F[-2, :] + mum[-1] * 0.0) - Df[-1] * (0.0 - F[-2, :]) / h
    J0 = np.maximum(0.0, -F_left)      # outward to the left
    J1 = np.maximum(0.0,  F_right)     # outward to the right
    return J0, J1

# ---- vectorized solver ----
def wf_fp_solve_many(
    p0s,
    Ne=1e4,
    s=0.0,
    T=100.0,
    M=800,
    dt=None,
    bump_width=1e-3,
    checkpoints=(),
    return_snapshots=False,
):
    """
    Vectorized WF Fokker–Planck (absorbing 0/1) for multiple initial p0s.
    Evolves all columns simultaneously reusing one sparse LU.

    Parameters
    ----------
    p0s : array-like, shape (K,)
        Initial frequencies.
    Ne, s : floats
        Effective size and selection coefficient per generation.
    T : float
        Total generations to evolve.
    M : int
        Number of intervals (grid has M+1 nodes).
    dt : float or None
        Time step; if None, chosen heuristically.
    bump_width : float
        Width of initial Gaussian bump approximating δ(p0).
    checkpoints : iterable of floats
        Times at which to store interior densities (optional).
    return_snapshots : bool
        If True, return a dict of snapshots (time -> (p, F_at_time)).

    Returns
    -------
    result : dict with
        grid            : (M+1,) grid in p
        density         : (M+1, K) interior density at final time T
        M0, M1          : (K,) boundary masses at time T
        mass_interior   : (K,) interior masses at time T
        mass_total      : (K,) total mass ≈ 1
        P_fix1_infty    : (K,) closed-form fixation prob at p=1 (t→∞)
        P_fix0_infty    : (K,)
        dt              : float time step used
        snapshots       : optional {t: (p, F_t)} with F_t shape (M+1, K)
    """
    p0s = np.asarray(p0s, dtype=float).ravel()
    K = p0s.size

    # grid and coefficients
    p = np.linspace(0.0, 1.0, M + 1)
    h = p[1] - p[0]
    mu = s * p * (1.0 - p)
    D  = (p * (1.0 - p)) / (2.0 * Ne)

    # initial stacked bumps (columns)
    F = _initial_bumps_matrix(p, p0s, width=bump_width)  # (M+1, K)

    # time step
    if dt is None:
        dt = 0.2 * h**2 / (np.max(D) + 1e-12)

    # operator & LU
    A, aux = _build_implicit_matrix(p, mu, D, dt)
    lu = splu(A)

    # run
    t = 0.0
    M0 = np.zeros(K)
    M1 = np.zeros(K)
    checkpoints = np.sort(np.array(checkpoints, dtype=float))
    ck_idx = 0
    snaps = {} if return_snapshots and len(checkpoints) else None

    while t < T - 1e-12:
        # boundary fluxes for all columns
        J0, J1 = _boundary_fluxes_matrix(F, *aux)
        M0 += J0 * dt
        M1 += J1 * dt

        # implicit step for all columns at once
        RHS = F[1:-1, :]                        # shape (M-1, K)
        F_new = np.zeros_like(F)
        # splu.solve supports 2D RHS (solves each column)
        F_new[1:-1, :] = lu.solve(RHS)
        F_new[F_new < 0] = 0.0
        F = F_new
        t += dt

        # snapshots
        while ck_idx < len(checkpoints) and t >= checkpoints[ck_idx] - 1e-12:
            if snaps is not None:
                snaps[float(checkpoints[ck_idx])] = (p.copy(), F.copy())
            ck_idx += 1

    # diagnostics per column
    mass_interior = np.trapz(F, p, axis=0)
    mass_total    = mass_interior + M0 + M1
    Pfix1 = wf_fixation_prob(p0s, Ne, s)

    out = dict(
        grid=p,
        density=F,
        M0=M0, M1=M1,
        mass_interior=mass_interior,
        mass_total=mass_total,
        P_fix1_infty=Pfix1,
        P_fix0_infty=1.0 - Pfix1,
        dt=dt,
    )
    if snaps is not None:
        out["snapshots"] = snaps
    return out

import numpy as np
from scipy import integrate
from scipy.special import comb
from scipy.interpolate import interp1d

def loglike_single_time_quad(k, n, p0, Ne, s, T,
                             M=800, dt=None, bump_width=1e-3):
    """
    Log-likelihood of observing k derived out of n at time T,
    starting at p0, under WF diffusion with selection s.
    PDE solved by wf_fp_solve_many; interior integral via scipy.integrate.quad.
    """
    out = wf_fp_solve_many([p0], Ne=Ne, s=s, T=T, M=M, dt=dt, bump_width=bump_width)
    p = out["grid"]
    f = out["density"][:, 0]
    M0, M1 = out["M0"][0], out["M1"][0]

    # boundary binomial pmf
    pmf0 = 1.0 if k == 0 else 0.0
    pmf1 = 1.0 if k == n else 0.0

    # continuous f(p) via interpolation
    f_interp = interp1d(p, f, kind="linear", bounds_error=False, fill_value=0.0)

    def integrand(x):
        if x <= 0.0 or x >= 1.0:
            return 0.0
        # Binomial pmf × density
        return comb(n, k) * (x**k) * ((1-x)**(n-k)) * float(f_interp(x))

    val, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-8)
    like = M0*pmf0 + M1*pmf1 + val
    return np.log(np.clip(like, 1e-300, None))


def fit_s_grid_quad(k, n, p0, Ne, T, s_grid,
                    M=800, dt=None, bump_width=1e-3):
    """
    Evaluate log-likelihood over s_grid using loglike_single_time_quad,
    return MLE s_hat, log-likelihoods, LR statistic and p-value vs H0: s=0.
    """
    s_grid = np.asarray(s_grid, dtype=float)
    ll = np.array([loglike_single_time_quad(k, n, p0, Ne, s, T,
                                            M=M, dt=dt, bump_width=bump_width)
                   for s in s_grid])

    # MLE on the grid
    imax = int(np.argmax(ll))
    s_hat = float(s_grid[imax])

    # LR test vs s = 0
    ll0 = float(loglike_single_time_quad(k, n, p0, Ne, 0.0, T,
                                         M=M, dt=dt, bump_width=bump_width))
    lr  = 2.0 * (ll[imax] - ll0)
    pval = 1.0 - chi2.cdf(max(lr, 0.0), df=1)

    return s_hat, ll, lr, pval

def posterior_prob_s_positive_quad(k, n, p0, Ne, T, s_grid,
                                   prior=None, M=800, dt=None, bump_width=1e-3):
    """
    Posterior P(s>0 | data) on a discrete grid of s.
    prior: array-like same shape as s_grid or None for uniform.
    Returns (prob_pos, post_weights) where post_weights sum to 1 over s_grid.
    """
    s_grid = np.asarray(s_grid, dtype=float)
    ll = np.array([loglike_single_time_quad(k, n, p0, Ne, s, T,
                                            M=M, dt=dt, bump_width=bump_width)
                   for s in s_grid])

    if prior is None:
        prior = np.ones_like(s_grid, dtype=float)
    prior = np.asarray(prior, dtype=float)
    prior = prior / np.sum(prior)

    logpost = ll + np.log(prior + 1e-300)
    m = np.max(logpost)
    w = np.exp(logpost - m)
    w = w / np.sum(w)

    prob_pos = float(w[s_grid > 0].sum())
    return prob_pos, w