"""
wf_temporal_selection.py
========================

Mass-conservative Wright-Fisher Fokker-Planck solver and scalable temporal
selection scan for genome-wide SNP-frequency time series.

The module supports two complementary analyses:

1. Exact-ish finite-volume Fokker-Planck propagation for a small/moderate number
   of initial allele frequencies. This returns the continuous allele-frequency
   density plus sticky absorbing point masses at p=0 and p=1.

2. A genome-wide, vectorized temporal scan for millions of SNPs. This estimates
   effective population size (Ne) from genome-wide drift and then estimates a
   per-SNP selection coefficient, direction, likelihood-ratio statistic, p-value,
   LOD score, and Benjamini-Hochberg q-value.

Conventions
-----------
Allele frequency is p in [0, 1]. Selection acts through the drift term

    mu(p) = s * p * (1 - p)

so s > 0 means that the tracked allele is favored/increases, and s < 0 means
that the tracked allele is disfavored/decreases.

The Fokker-Planck equation is written in conservative flux form:

    d f / dt = - dJ / dp
    J        = mu(p) f - d[D(p) f] / dp
    D(p)     = p(1-p) / (diffusion_scale * Ne)

The classical diploid Wright-Fisher diffusion convention corresponds to
`diffusion_scale=4`, because the forward equation uses one-half of the
per-generation binomial variance. The earlier prototype used `diffusion_scale=2`;
that option remains available for reproducibility.

For genome-wide temporal scans, the default Ne convention is the discrete diploid
heterozygosity decay model:

    F_t = 1 - (1 - 1/(2*Ne))**t

which is usually the clearest model for generation-to-generation colony analyses.

Dependencies: numpy, scipy, pandas only for dataframe scan functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import interp1d
from scipy.special import gammaln
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.stats import chi2

ArrayLike = Union[np.ndarray, Sequence[float], float]


# =============================================================================
# Basic numerical helpers
# =============================================================================


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _broadcast_optional_array(x: Optional[ArrayLike], size: int, name: str) -> Optional[np.ndarray]:
    """Broadcast a scalar/array optional input to a 1D float array of length size."""
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=float)
    arr = arr.ravel()
    if arr.size != size:
        raise ValueError(f"{name} must be scalar or have length {size}.")
    return arr


def _logit_clipped(p: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable logit after clipping frequencies away from 0 and 1."""
    p = np.asarray(p, dtype=float)
    pc = np.clip(p, eps, 1.0 - eps)
    return np.log(pc) - np.log1p(-pc)


def _inv_logit(z: ArrayLike) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _log_expm1_pos(z: ArrayLike) -> np.ndarray:
    """Return log(exp(z)-1) accurately for z >= 0."""
    z = np.asarray(z, dtype=float)
    return z + np.log1p(-np.exp(-z))


def _benjamini_hochberg(pvalues: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values, preserving NaNs."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not np.any(ok):
        return q
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1, dtype=float)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    q[ok] = tmp
    return q


# =============================================================================
# Closed-form population-genetic quantities
# =============================================================================


def wf_fixation_prob(
    p0: ArrayLike,
    Ne: float,
    s: float,
    diffusion_scale: float = 2.0,
) -> np.ndarray:
    """Closed-form eventual fixation probability at p=1.

    Parameters
    ----------
    p0 : array-like
        Initial allele frequency/frequencies.
    Ne : float
        Effective population size.
    s : float
        Selection coefficient in mu(p)=s*p*(1-p).
    diffusion_scale : float, default=2.0
        Must match D=p(1-p)/(diffusion_scale*Ne). The prototype used 2.0.
        Classical diploid diffusion usually uses 4.0.

    Returns
    -------
    numpy.ndarray
        Probability that the tracked allele eventually fixes at p=1.

    Notes
    -----
    For the diffusion used here,

        u(p0) = [1 - exp(-alpha*p0)] / [1 - exp(-alpha)]
        alpha = diffusion_scale * Ne * s

    and u(p0)=p0 when s=0.
    """
    if Ne <= 0:
        raise ValueError("Ne must be positive.")
    p = np.clip(np.asarray(p0, dtype=float), 0.0, 1.0)
    alpha = float(diffusion_scale) * float(Ne) * float(s)

    if np.isclose(alpha, 0.0):
        return p.copy()

    out = np.empty_like(p, dtype=float)
    out[p <= 0.0] = 0.0
    out[p >= 1.0] = 1.0
    mid = (p > 0.0) & (p < 1.0)
    x = p[mid]

    if alpha > 0:
        out[mid] = np.expm1(-alpha * x) / np.expm1(-alpha)
    else:
        beta = -alpha
        out[mid] = np.exp(_log_expm1_pos(beta * x) - _log_expm1_pos(beta))

    return np.clip(out, 0.0, 1.0)


def deterministic_selected_freq(
    p0: ArrayLike,
    s: float,
    generations: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Closed-form deterministic allele-frequency trajectory under selection.

    This solves dp/dt = s*p*(1-p), ignoring drift:

        logit(p_t) = logit(p0) + s*t

    It is the closed-form backbone used to estimate per-SNP s in the fast scan.
    """
    if generations < 0:
        raise ValueError("generations must be non-negative.")
    z = _logit_clipped(p0, eps=eps) + float(s) * float(generations)
    return _inv_logit(z)


def estimate_s_closed_form(
    p_before: ArrayLike,
    p_after: ArrayLike,
    generations: float,
    eps: float = 1e-6,
    s_bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Closed-form per-SNP selection coefficient from two frequencies.

    Uses

        s_hat = [logit(p_after) - logit(p_before)] / generations.

    This is vectorized and is appropriate for millions of SNPs. Frequencies are
    clipped by `eps` to avoid infinite estimates at 0 or 1.
    """
    if generations <= 0:
        raise ValueError("generations must be positive.")
    p0 = np.asarray(p_before, dtype=float)
    p1 = np.asarray(p_after, dtype=float)
    s_hat = (_logit_clipped(p1, eps=eps) - _logit_clipped(p0, eps=eps)) / float(generations)
    if s_bounds is not None:
        low, high = map(float, s_bounds)
        if low >= high:
            raise ValueError("s_bounds must be an increasing (low, high) tuple.")
        s_hat = np.clip(s_hat, low, high)
    return s_hat


# =============================================================================
# Conservative finite-volume Wright-Fisher Fokker-Planck solver
# =============================================================================


def _initial_density_and_boundary_masses(
    p: np.ndarray,
    p0s: np.ndarray,
    width: float = 1e-3,
    boundary_tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create initial interior density plus sticky boundary masses."""
    p0s = np.asarray(p0s, dtype=float).ravel()
    K = p0s.size
    F = np.zeros((p.size, K), dtype=float)
    M0 = np.zeros(K, dtype=float)
    M1 = np.zeros(K, dtype=float)

    at0 = p0s <= boundary_tol
    at1 = p0s >= (1.0 - boundary_tol)
    interior = ~(at0 | at1)
    M0[at0] = 1.0
    M1[at1] = 1.0

    if np.any(interior):
        pp = p[:, None]
        centers = p0s[interior][None, :]
        G = np.exp(-0.5 * ((pp - centers) / width) ** 2)
        G[(p <= 0.0) | (p >= 1.0), :] = 0.0
        area = np.trapezoid(G, p, axis=0)

        # If the Gaussian is narrower than the grid, put mass on the nearest
        # interior node instead of losing the initial condition.
        bad = area <= 1e-300
        if np.any(bad):
            G[:, bad] = 0.0
            bad_cols = np.flatnonzero(bad)
            idx = np.clip(np.searchsorted(p, centers.ravel()[bad]), 1, p.size - 2)
            G[idx, bad_cols] = 1.0 / (p[1] - p[0])
            area = np.trapezoid(G, p, axis=0)

        G /= area[None, :]
        F[:, interior] = G

    return F, M0, M1


def _build_implicit_matrix_ito(
    p: np.ndarray,
    mu: np.ndarray,
    D: np.ndarray,
    dt: float,
) -> Tuple[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Build I - dt*L for J=mu*f - d(D*f)/dp."""
    h = p[1] - p[0]
    pf = 0.5 * (p[:-1] + p[1:])
    muf = np.interp(pf, p, mu)
    mup = np.maximum(muf, 0.0)
    mum = np.minimum(muf, 0.0)

    D_l = D[:-2]
    D_i = D[1:-1]
    D_r = D[2:]
    mup_l, mum_l = mup[:-1], mum[:-1]
    mup_r, mum_r = mup[1:], mum[1:]

    lower_L = (mup_l + D_l / h) / h
    main_L = (mum_l - D_i / h - mup_r - D_i / h) / h
    upper_L = (D_r / h - mum_r) / h

    A = diags(
        [-dt * lower_L[1:], 1.0 - dt * main_L, -dt * upper_L[:-1]],
        offsets=[-1, 0, 1],
        format="csc",
    )
    return A, (mup, mum, D, h)


def _boundary_fluxes_ito(
    F: np.ndarray,
    mup: np.ndarray,
    mum: np.ndarray,
    D: np.ndarray,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Outward sticky-boundary fluxes for J=mu*f - d(D*f)/dp."""
    J_left = (mup[0] * F[0, :] + mum[0] * F[1, :]) - (
        D[1] * F[1, :] - D[0] * F[0, :]
    ) / h
    J_right = (mup[-1] * F[-2, :] + mum[-1] * F[-1, :]) - (
        D[-1] * F[-1, :] - D[-2] * F[-2, :]
    ) / h
    return np.maximum(0.0, -J_left), np.maximum(0.0, J_right)


def _build_implicit_matrix_fickian(
    p: np.ndarray,
    mu: np.ndarray,
    D: np.ndarray,
    dt: float,
) -> Tuple[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Build I - dt*L for J=mu*f - D*df/dp.

    This reproduces the prototype flux form, but with a conservative corrected
    finite-volume discretization. The Itô form is the default for WF diffusion.
    """
    h = p[1] - p[0]
    pf = 0.5 * (p[:-1] + p[1:])
    muf = np.interp(pf, p, mu)
    Df = np.interp(pf, p, D)
    mup = np.maximum(muf, 0.0)
    mum = np.minimum(muf, 0.0)

    Dl = Df[:-1] / h
    Dr = Df[1:] / h
    mup_l, mum_l = mup[:-1], mum[:-1]
    mup_r, mum_r = mup[1:], mum[1:]

    lower_L = (Dl + mup_l) / h
    main_L = ((mum_l - Dl) - (mup_r + Dr)) / h
    upper_L = (Dr - mum_r) / h

    A = diags(
        [-dt * lower_L[1:], 1.0 - dt * main_L, -dt * upper_L[:-1]],
        offsets=[-1, 0, 1],
        format="csc",
    )
    return A, (mup, mum, Df, h)


def _boundary_fluxes_fickian(
    F: np.ndarray,
    mup: np.ndarray,
    mum: np.ndarray,
    Df: np.ndarray,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Outward sticky-boundary fluxes for J=mu*f - D*df/dp."""
    J_left = (mup[0] * F[0, :] + mum[0] * F[1, :]) - Df[0] * (F[1, :] - F[0, :]) / h
    J_right = (mup[-1] * F[-2, :] + mum[-1] * F[-1, :]) - Df[-1] * (F[-1, :] - F[-2, :]) / h
    return np.maximum(0.0, -J_left), np.maximum(0.0, J_right)


def wf_fp_solve_many(
    p0s: ArrayLike,
    Ne: float = 1e4,
    s: float = 0.0,
    T: float = 100.0,
    M: int = 800,
    dt: Optional[float] = None,
    bump_width: float = 1e-3,
    checkpoints: Iterable[float] = (),
    return_snapshots: bool = False,
    diffusion_scale: float = 4.0,
    flux_form: str = "ito",
    boundary_tol: float = 0.0,
) -> Dict[str, Any]:
    """Solve the WF Fokker-Planck equation for many initial frequencies.

    Parameters
    ----------
    p0s : array-like, shape (K,)
        Initial allele frequencies. Values exactly 0 or 1 start as sticky
        boundary masses.
    Ne : float
        Effective population size used in D=p(1-p)/(diffusion_scale*Ne).
    s : float
        Selection coefficient in mu=s*p*(1-p).
    T : float
        Number of generations elapsed.
    M : int
        Number of grid intervals; the grid has M+1 nodes.
    dt : float or None
        Time step. If None, a conservative heuristic is used. The solver rescales
        dt so that an integer number of steps ends exactly at T.
    bump_width : float
        Width of the Gaussian approximation to a point initial condition.
    checkpoints : iterable of float
        Optional times for storing snapshots.
    return_snapshots : bool
        If True, store density and boundary masses at requested checkpoints.
    diffusion_scale : float, default=4.0
        Classical diploid diffusion convention. Use 2.0 to reproduce the older
        prototype scaling.
    flux_form : {'ito', 'fickian'}
        'ito' is the WF forward equation: J=mu*f-d(D*f)/dp. 'fickian' is the
        corrected conservative form of the prototype J=mu*f-D*df/dp.
    boundary_tol : float
        Frequencies <= boundary_tol or >= 1-boundary_tol are initialized as
        boundary masses.

    Returns
    -------
    dict
        Keys include grid, density, M0, M1, mass_interior, mass_total,
        P_fix1_infty, P_fix0_infty, dt, n_steps.
    """
    p0s_arr = _as_1d_float(p0s, "p0s")
    if Ne <= 0:
        raise ValueError("Ne must be positive.")
    if M < 3:
        raise ValueError("M must be at least 3.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if diffusion_scale <= 0:
        raise ValueError("diffusion_scale must be positive.")
    if bump_width <= 0:
        raise ValueError("bump_width must be positive.")
    if not np.all(np.isfinite(p0s_arr)):
        raise ValueError("p0s contains NaN or infinite values.")
    if np.any((p0s_arr < 0.0) | (p0s_arr > 1.0)):
        raise ValueError("All p0s must lie in [0, 1].")

    p = np.linspace(0.0, 1.0, int(M) + 1)
    h = p[1] - p[0]
    mu = float(s) * p * (1.0 - p)
    D = p * (1.0 - p) / (float(diffusion_scale) * float(Ne))

    F, M0, M1 = _initial_density_and_boundary_masses(
        p, p0s_arr, width=bump_width, boundary_tol=boundary_tol
    )

    if T == 0:
        mass_interior = h * np.sum(F[1:-1, :], axis=0)
        Pfix1 = wf_fixation_prob(p0s_arr, Ne=Ne, s=s, diffusion_scale=diffusion_scale)
        return dict(
            grid=p,
            density=F,
            M0=M0,
            M1=M1,
            mass_interior=mass_interior,
            mass_total=mass_interior + M0 + M1,
            P_fix1_infty=Pfix1,
            P_fix0_infty=1.0 - Pfix1,
            dt=0.0,
            n_steps=0,
            flux_form=flux_form,
            diffusion_scale=float(diffusion_scale),
        )

    if dt is None:
        # Fully implicit is stable for larger dt, but this keeps boundary flux
        # and transient densities reasonably accurate.
        dt = 0.2 * h * h / (np.max(D) + 1e-300)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    n_steps = int(np.ceil(float(T) / float(dt)))
    dt_eff = float(T) / n_steps

    flux_form_l = str(flux_form).lower()
    if flux_form_l == "ito":
        A, aux = _build_implicit_matrix_ito(p, mu, D, dt_eff)
        flux_fun = _boundary_fluxes_ito
    elif flux_form_l == "fickian":
        A, aux = _build_implicit_matrix_fickian(p, mu, D, dt_eff)
        flux_fun = _boundary_fluxes_fickian
    else:
        raise ValueError("flux_form must be 'ito' or 'fickian'.")

    lu = splu(A)
    checkpoints_arr = np.sort(np.asarray(list(checkpoints), dtype=float))
    ck_idx = 0
    snaps = {} if return_snapshots and checkpoints_arr.size else None
    t = 0.0

    for _ in range(n_steps):
        F_new = np.zeros_like(F)
        F_new[1:-1, :] = lu.solve(F[1:-1, :])

        tiny_negative = (F_new < 0.0) & (F_new > -1e-13)
        F_new[tiny_negative] = 0.0
        if np.any(F_new < -1e-10):
            raise FloatingPointError(
                "Large negative density encountered. Refine M/dt or check parameters."
            )

        # Boundary masses are updated using the same implicit-time flux that
        # generated the interior update. This is what prevents probability loss.
        J0, J1 = flux_fun(F_new, *aux)
        M0 += J0 * dt_eff
        M1 += J1 * dt_eff
        F = F_new
        t += dt_eff

        while ck_idx < checkpoints_arr.size and t >= checkpoints_arr[ck_idx] - 1e-12:
            if snaps is not None:
                snaps[float(checkpoints_arr[ck_idx])] = (
                    p.copy(),
                    F.copy(),
                    M0.copy(),
                    M1.copy(),
                )
            ck_idx += 1

    mass_interior = h * np.sum(F[1:-1, :], axis=0)
    mass_total = mass_interior + M0 + M1
    Pfix1 = wf_fixation_prob(p0s_arr, Ne=Ne, s=s, diffusion_scale=diffusion_scale)

    out = dict(
        grid=p,
        density=F,
        M0=M0,
        M1=M1,
        mass_interior=mass_interior,
        mass_total=mass_total,
        P_fix1_infty=Pfix1,
        P_fix0_infty=1.0 - Pfix1,
        dt=dt_eff,
        n_steps=n_steps,
        flux_form=flux_form_l,
        diffusion_scale=float(diffusion_scale),
    )
    if snaps is not None:
        out["snapshots"] = snaps
    return out


def wf_fp_solve_many_chunked(
    p0s: ArrayLike,
    chunk_size: int = 5000,
    return_density: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Chunked wrapper for large panels of initial frequencies.

    A dense (M+1) x K density matrix is not feasible for millions of SNPs. This
    function processes `p0s` in chunks and returns final summaries. Set
    `return_density=True` only for small K.
    """
    p0s_arr = _as_1d_float(p0s, "p0s")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    M0_list, M1_list, mi_list, mt_list = [], [], [], []
    densities = []
    grid = None
    meta = None

    for start in range(0, p0s_arr.size, int(chunk_size)):
        stop = min(start + int(chunk_size), p0s_arr.size)
        out = wf_fp_solve_many(p0s_arr[start:stop], **kwargs)
        grid = out["grid"]
        meta = out
        M0_list.append(out["M0"])
        M1_list.append(out["M1"])
        mi_list.append(out["mass_interior"])
        mt_list.append(out["mass_total"])
        if return_density:
            densities.append(out["density"])

    result = dict(
        grid=grid,
        M0=np.concatenate(M0_list),
        M1=np.concatenate(M1_list),
        mass_interior=np.concatenate(mi_list),
        mass_total=np.concatenate(mt_list),
        dt=meta["dt"],
        n_steps=meta["n_steps"],
        flux_form=meta["flux_form"],
        diffusion_scale=meta["diffusion_scale"],
    )
    if return_density:
        result["density"] = np.concatenate(densities, axis=1)
    return result


def wf_density_cdf(out: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return CDF of the mixed sticky-boundary distribution from solver output.

    The returned CDF includes M0 at p=0, the continuous density over (0,1), and
    is forced to 1 at p=1.
    """
    p = out["grid"]
    F = out["density"]
    M0 = np.asarray(out["M0"], dtype=float)
    cdf_cont = integrate.cumulative_trapezoid(F, p, axis=0, initial=0.0)
    cdf = M0[None, :] + cdf_cont
    cdf[-1, :] = 1.0
    return p, cdf


# =============================================================================
# PDE observation likelihoods for top-locus validation
# =============================================================================


def _binom_pmf_grid(k: int, n: int, p: np.ndarray) -> np.ndarray:
    """Binomial PMF for observing k of n on each grid p, stable in log-space."""
    k = int(k)
    n = int(n)
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n.")
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    if k == 0:
        out[p == 0.0] = 1.0
    if k == n:
        out[p == 1.0] = 1.0
    mid = (p > 0.0) & (p < 1.0)
    logpmf = (
        gammaln(n + 1)
        - gammaln(k + 1)
        - gammaln(n - k + 1)
        + k * np.log(p[mid])
        + (n - k) * np.log1p(-p[mid])
    )
    out[mid] = np.exp(logpmf)
    return out


def loglike_single_time_grid(
    k: int,
    n: int,
    p0: float,
    Ne: float,
    s: float,
    T: float,
    M: int = 800,
    dt: Optional[float] = None,
    bump_width: float = 1e-3,
    diffusion_scale: float = 4.0,
    flux_form: str = "ito",
) -> float:
    """Likelihood of observing k derived chromosomes out of n at time T.

    This is the preferred PDE likelihood because the PDE density already lives
    on a grid; direct grid integration is faster than adaptive quadrature.
    """
    out = wf_fp_solve_many(
        [p0],
        Ne=Ne,
        s=s,
        T=T,
        M=M,
        dt=dt,
        bump_width=bump_width,
        diffusion_scale=diffusion_scale,
        flux_form=flux_form,
    )
    p = out["grid"]
    pmf = _binom_pmf_grid(k, n, p)
    like = (
        out["M0"][0] * (1.0 if k == 0 else 0.0)
        + out["M1"][0] * (1.0 if k == n else 0.0)
        + np.trapezoid(pmf * out["density"][:, 0], p)
    )
    return float(np.log(np.clip(like, 1e-300, None)))


def loglike_single_time_quad(
    k: int,
    n: int,
    p0: float,
    Ne: float,
    s: float,
    T: float,
    M: int = 800,
    dt: Optional[float] = None,
    bump_width: float = 1e-3,
    diffusion_scale: float = 4.0,
    flux_form: str = "ito",
) -> float:
    """Adaptive-quadrature version of the single-time likelihood.

    This is slower than `loglike_single_time_grid`; it is kept for numerical
    cross-checks only.
    """
    out = wf_fp_solve_many(
        [p0],
        Ne=Ne,
        s=s,
        T=T,
        M=M,
        dt=dt,
        bump_width=bump_width,
        diffusion_scale=diffusion_scale,
        flux_form=flux_form,
    )
    p = out["grid"]
    f = out["density"][:, 0]
    M0, M1 = out["M0"][0], out["M1"][0]
    pmf0 = 1.0 if k == 0 else 0.0
    pmf1 = 1.0 if k == n else 0.0
    f_interp = interp1d(p, f, kind="linear", bounds_error=False, fill_value=0.0)

    def integrand(x: float) -> float:
        if x <= 0.0 or x >= 1.0:
            return 0.0
        logpmf = (
            gammaln(n + 1)
            - gammaln(k + 1)
            - gammaln(n - k + 1)
            + k * np.log(x)
            + (n - k) * np.log1p(-x)
        )
        return float(np.exp(logpmf) * f_interp(x))

    val, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-8)
    like = M0 * pmf0 + M1 * pmf1 + val
    return float(np.log(np.clip(like, 1e-300, None)))


def fit_s_pde_mle(
    k: int,
    n: int,
    p0: float,
    Ne: float,
    T: float,
    s_bounds: Tuple[float, float] = (-1.0, 1.0),
    use_quad: bool = False,
    **kwargs: Any,
) -> Dict[str, float]:
    """PDE likelihood MLE for s at one SNP; intended for top-locus validation."""
    low, high = map(float, s_bounds)
    if low >= high:
        raise ValueError("s_bounds must be an increasing (low, high) tuple.")
    ll_fun = loglike_single_time_quad if use_quad else loglike_single_time_grid

    def nll(s: float) -> float:
        return -ll_fun(k=k, n=n, p0=p0, Ne=Ne, s=float(s), T=T, **kwargs)

    res = optimize.minimize_scalar(nll, bounds=(low, high), method="bounded", options={"xatol": 1e-5})
    s_hat = float(res.x)
    ll_alt = -float(res.fun)
    ll_null = ll_fun(k=k, n=n, p0=p0, Ne=Ne, s=0.0, T=T, **kwargs)
    lr = max(0.0, 2.0 * (ll_alt - ll_null))
    return dict(
        s_hat=s_hat,
        loglike_alt=ll_alt,
        loglike_null=ll_null,
        lr_stat=lr,
        p_value=float(chi2.sf(lr, df=1)),
        lod_score=float(lr / (2.0 * np.log(10.0))),
        optimizer_success=bool(res.success),
    )


# =============================================================================
# Temporal Ne estimation from genome-wide drift
# =============================================================================


def temporal_F_to_Ne(
    F: float,
    generations: float,
    model: str = "discrete_diploid",
    diffusion_scale: float = 4.0,
) -> float:
    """Convert standardized temporal drift variance F to Ne."""
    F = float(F)
    generations = float(generations)
    if generations <= 0:
        raise ValueError("generations must be positive.")
    if F <= 0.0:
        return float("inf")
    if F >= 1.0:
        return 0.5

    model_l = str(model).lower()
    if model_l in {"discrete", "diploid", "discrete_diploid", "wf"}:
        one_minus_per_gen = np.exp(np.log1p(-F) / generations)
        return float(1.0 / (2.0 * (1.0 - one_minus_per_gen)))
    if model_l in {"diffusion", "continuous", "pde"}:
        return float(-2.0 * generations / (float(diffusion_scale) * np.log1p(-F)))
    raise ValueError("model must be 'discrete_diploid' or 'diffusion'.")


def temporal_Ne_to_F(
    Ne: float,
    generations: float,
    model: str = "discrete_diploid",
    diffusion_scale: float = 4.0,
) -> float:
    """Convert Ne to temporal standardized drift variance F."""
    if generations <= 0:
        raise ValueError("generations must be positive.")
    if Ne is None or not np.isfinite(Ne):
        return 0.0
    Ne = float(Ne)
    if Ne <= 0.0:
        return 1.0 - 1e-12

    model_l = str(model).lower()
    if model_l in {"discrete", "diploid", "discrete_diploid", "wf"}:
        return float(1.0 - np.exp(float(generations) * np.log1p(-1.0 / (2.0 * Ne))))
    if model_l in {"diffusion", "continuous", "pde"}:
        return float(1.0 - np.exp(-2.0 * float(generations) / (float(diffusion_scale) * Ne)))
    raise ValueError("model must be 'discrete_diploid' or 'diffusion'.")


def _prepare_sample_sizes(
    n_before: Optional[np.ndarray],
    n_after: Optional[np.ndarray],
    unit: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    unit_l = str(unit).lower()
    n0 = None if n_before is None else np.asarray(n_before, dtype=float).copy()
    n1 = None if n_after is None else np.asarray(n_after, dtype=float).copy()
    if unit_l in {"diploid", "diploids", "diploid_individuals", "individuals", "rats"}:
        if n0 is not None:
            n0 *= 2.0
        if n1 is not None:
            n1 *= 2.0
    elif unit_l in {"chromosome", "chromosomes", "alleles", "haploid"}:
        pass
    else:
        raise ValueError("sample_sizes_are must be 'diploid_individuals' or 'chromosomes'.")
    return n0, n1


def estimate_ne_from_temporal_freqs(
    p_before: ArrayLike,
    p_after: ArrayLike,
    generations: float,
    n_before: Optional[ArrayLike] = None,
    n_after: Optional[ArrayLike] = None,
    sample_sizes_are: str = "diploid_individuals",
    method: str = "normal_mle",
    model: str = "discrete_diploid",
    diffusion_scale: float = 4.0,
    min_initial_het: float = 1e-4,
    min_maf: Optional[float] = None,
    trim_quantile: Optional[float] = None,
    weights: Optional[ArrayLike] = None,
    n_boot: int = 0,
    bootstrap_size: Optional[int] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, Any]:
    """Estimate effective population size from genome-wide temporal drift.

    Parameters
    ----------
    p_before, p_after : array-like
        Frequencies of the same allele before and after the interval.
    generations : float
        Number of generations between the two time points.
    n_before, n_after : scalar/array-like, optional
        Sample sizes for frequency estimates. By default interpreted as diploid
        individuals and converted to chromosome counts.
    sample_sizes_are : {'diploid_individuals', 'chromosomes'}
        Unit for n_before/n_after.
    method : {'normal_mle', 'moment'}
        `moment` uses a corrected standardized temporal variance.
        `normal_mle` solves the normal-approximation score equation with
        scipy.optimize.brentq when possible.
    model : {'discrete_diploid', 'diffusion'}
        Mapping from temporal variance F to Ne.
    diffusion_scale : float
        Used only when model='diffusion'.
    min_initial_het : float
        Exclude SNPs with p0*(1-p0) below this threshold.
    min_maf : float or None
        Optional initial MAF filter.
    trim_quantile : float or None
        Optional trimming of largest standardized changes before estimating Ne.
        Useful to reduce influence of selected/error-prone SNPs.
    weights : scalar/array-like, optional
        SNP weights. Consider LD-pruned or block weights for final analyses.
    n_boot : int
        Optional nonparametric bootstrap over retained SNPs.

    Returns
    -------
    dict with Ne, F, diagnostic counts, and optional bootstrap CIs.
    """
    p0 = _as_1d_float(p_before, "p_before")
    p1 = _as_1d_float(p_after, "p_after")
    if p0.size != p1.size:
        raise ValueError("p_before and p_after must have the same length.")
    if generations <= 0:
        raise ValueError("generations must be positive.")

    n0 = _broadcast_optional_array(n_before, p0.size, "n_before")
    n1 = _broadcast_optional_array(n_after, p0.size, "n_after")
    n0, n1 = _prepare_sample_sizes(n0, n1, sample_sizes_are)
    w = _broadcast_optional_array(weights, p0.size, "weights")

    finite = np.isfinite(p0) & np.isfinite(p1)
    in_bounds = (p0 >= 0.0) & (p0 <= 1.0) & (p1 >= 0.0) & (p1 <= 1.0)
    h0 = p0 * (1.0 - p0)
    mask = finite & in_bounds & (h0 >= float(min_initial_het))

    if min_maf is not None:
        mask &= np.minimum(p0, 1.0 - p0) >= float(min_maf)
    if n0 is not None:
        mask &= np.isfinite(n0) & (n0 > 0.0)
    if n1 is not None:
        mask &= np.isfinite(n1) & (n1 > 0.0)
    if w is not None:
        mask &= np.isfinite(w) & (w > 0.0)

    delta = p1 - p0
    standardized_change = np.full(p0.size, np.nan, dtype=float)
    valid_h = h0 > 0.0
    standardized_change[valid_h] = delta[valid_h] ** 2 / h0[valid_h]

    if trim_quantile is not None:
        q = float(trim_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("trim_quantile must lie between 0 and 1.")
        if np.any(mask):
            cutoff = np.nanquantile(standardized_change[mask], q)
            mask &= standardized_change <= cutoff

    if not np.any(mask):
        raise ValueError("No SNPs remained after filtering. Relax filters.")

    p0m = p0[mask]
    p1m = p1[mask]
    dm = p1m - p0m
    h = p0m * (1.0 - p0m)
    ww = np.ones_like(p0m) if w is None else w[mask].astype(float)

    sampling_var = np.zeros_like(p0m)
    if n0 is not None:
        sampling_var += p0m * (1.0 - p0m) / n0[mask]
    if n1 is not None:
        sampling_var += p1m * (1.0 - p1m) / n1[mask]

    def _estimate_F(indices: Optional[np.ndarray] = None) -> float:
        if indices is None:
            d, hh, sv, weights_i = dm, h, sampling_var, ww
        else:
            d, hh, sv, weights_i = dm[indices], h[indices], sampling_var[indices], ww[indices]

        method_l = str(method).lower()
        if method_l in {"moment", "moments", "variance"}:
            Fi = (d * d - sv) / hh
            return float(np.clip(np.sum(weights_i * Fi) / np.sum(weights_i), 0.0, 1.0 - 1e-12))

        if method_l not in {"normal_mle", "mle", "normal"}:
            raise ValueError("method must be 'normal_mle' or 'moment'.")

        # Negative log-likelihood in F: delta ~ N(0, F*h + sampling_var).
        def nll(F: float) -> float:
            V = F * hh + sv
            if np.any(V <= 0.0):
                return np.inf
            return float(0.5 * np.sum(weights_i * (np.log(V) + (d * d) / V)))

        # Analytic score. Root-find is usually faster than minimize_scalar.
        def score(F: float) -> float:
            V = F * hh + sv
            return float(0.5 * np.sum(weights_i * hh * (1.0 / V - (d * d) / (V * V))))

        lo = 1e-12
        hi = 1.0 - 1e-12
        if np.allclose(d, 0.0) and np.all(sv == 0.0):
            return 0.0
        try:
            slo = score(lo)
            shi = score(hi)
            if np.isfinite(slo) and np.isfinite(shi) and slo < 0.0 and shi > 0.0:
                return float(optimize.brentq(score, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100))
        except Exception:
            pass

        # Robust fallback if the score root is not bracketed.
        res = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        return float(np.clip(res.x, 0.0, 1.0 - 1e-12))

    Fhat = _estimate_F()
    Ne_hat = temporal_F_to_Ne(Fhat, generations=generations, model=model, diffusion_scale=diffusion_scale)

    result: Dict[str, Any] = dict(
        Ne=float(Ne_hat),
        F=float(Fhat),
        generations=float(generations),
        method=str(method),
        model=str(model),
        diffusion_scale=float(diffusion_scale),
        n_snps_input=int(p0.size),
        n_snps_used=int(mask.sum()),
        n_snps_excluded=int(p0.size - mask.sum()),
        mean_delta=float(np.average(dm, weights=ww)),
        mean_abs_delta=float(np.average(np.abs(dm), weights=ww)),
        mean_standardized_delta2=float(np.average(dm * dm / h, weights=ww)),
        mean_sampling_variance=float(np.average(sampling_var, weights=ww)),
        min_initial_het=float(min_initial_het),
        min_maf=None if min_maf is None else float(min_maf),
        trim_quantile=None if trim_quantile is None else float(trim_quantile),
    )

    if n_boot and int(n_boot) > 0:
        n_boot = int(n_boot)
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        B = int(mask.sum() if bootstrap_size is None else bootstrap_size)
        if B <= 0:
            raise ValueError("bootstrap_size must be positive.")
        n = int(mask.sum())
        F_boot = np.empty(n_boot, dtype=float)
        Ne_boot = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=B)
            Fb = _estimate_F(idx)
            F_boot[b] = Fb
            Ne_boot[b] = temporal_F_to_Ne(Fb, generations=generations, model=model, diffusion_scale=diffusion_scale)
        result.update(
            F_bootstrap=F_boot,
            Ne_bootstrap=Ne_boot,
            F_ci95=tuple(np.nanpercentile(F_boot, [2.5, 97.5])),
            Ne_ci95=tuple(np.nanpercentile(Ne_boot, [2.5, 97.5])),
        )

    return result


# =============================================================================
# DataFrame-level temporal Ne + per-SNP selection scan
# =============================================================================


def _get_df_vector_or_scalar(
    df: Any,
    value: Optional[ArrayLike] = None,
    col: Optional[str] = None,
    size: Optional[int] = None,
    name: str = "value",
) -> Optional[np.ndarray]:
    """Return scalar/column/array as a vector matching the dataframe length."""
    if size is None:
        size = len(df)
    if col is not None:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} was not found in the dataframe.")
        return np.asarray(df[col], dtype=float).ravel()
    return _broadcast_optional_array(value, size, name)


def temporal_selection_scan_dataframe(
    df: Any,
    generations: float,
    snp_col: str = "SNP_ID",
    freq_a_col: str = "frequency generation a",
    freq_b_col: str = "frequency generation b",
    Ne: Optional[float] = None,
    n_a: Optional[ArrayLike] = None,
    n_b: Optional[ArrayLike] = None,
    n_a_col: Optional[str] = None,
    n_b_col: Optional[str] = None,
    sample_sizes_are: str = "diploid_individuals",
    ne_method: str = "normal_mle",
    ne_model: str = "discrete_diploid",
    diffusion_scale: float = 4.0,
    ne_min_initial_het: float = 1e-4,
    ne_min_maf: Optional[float] = None,
    ne_trim_quantile: Optional[float] = 0.995,
    ne_weights: Optional[ArrayLike] = None,
    ne_weights_col: Optional[str] = None,
    n_boot_ne: int = 0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    test_min_initial_het: float = 1e-6,
    test_min_maf: Optional[float] = None,
    logit_eps: float = 1e-6,
    s_bounds: Tuple[float, float] = (-5.0, 5.0),
    add_fdr: bool = True,
    alpha: float = 0.05,
    return_invalid_rows: bool = True,
) -> Dict[str, Any]:
    """Estimate temporal Ne and scan SNPs for directional selection.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain SNP ID and two allele-frequency columns. Frequencies must
        be for the same allele orientation at both time points.
    generations : float
        Number of generations between time A and time B.
    snp_col, freq_a_col, freq_b_col : str
        Column names. Defaults match the user-requested names.
    Ne : float or None
        If None, estimate Ne from the genome-wide drift distribution. If given,
        use this value for all per-SNP tests.
    n_a, n_b or n_a_col, n_b_col : optional
        Sample sizes used to estimate the frequencies. When supplied, binomial
        sampling variance is added to the neutral drift variance.
    sample_sizes_are : {'diploid_individuals', 'chromosomes'}
        Unit for n_a/n_b.
    ne_trim_quantile : float or None
        Trim largest standardized allele-frequency changes before estimating Ne.
        Default 0.995 is robust for scans where a small fraction of SNPs may be
        selected or error-prone.
    test_min_maf : float or None
        Optional MAF filter for per-SNP tests at generation A.
    logit_eps : float
        Frequencies are clipped to [eps, 1-eps] for the closed-form s estimate.
    s_bounds : tuple
        Clip s estimates to finite bounds. This avoids infinite estimates for
        near-boundary frequencies.
    add_fdr : bool
        Add Benjamini-Hochberg q-values.
    alpha : float
        Alpha used for the convenience column `significant_direction`.
    return_invalid_rows : bool
        If True, output has one row per input row with NaNs for untestable SNPs.

    Returns
    -------
    dict
        {'Ne': float, 'Ne_summary': dict, 'results': pandas.DataFrame}

    The results DataFrame includes: SNP_ID, p_generation_a, p_generation_b,
    delta_p, selection_coefficient_s, selection_direction, p_value, lod_score,
    q_value_bh, and diagnostics.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("temporal_selection_scan_dataframe requires pandas.") from exc

    if generations <= 0:
        raise ValueError("generations must be positive.")
    for col in (snp_col, freq_a_col, freq_b_col):
        if col not in df.columns:
            raise ValueError(f"Required column {col!r} was not found in the dataframe.")
    n = len(df)
    if n == 0:
        raise ValueError("df is empty.")

    snp_id = np.asarray(df[snp_col])
    p0 = np.asarray(df[freq_a_col], dtype=float).ravel()
    p1 = np.asarray(df[freq_b_col], dtype=float).ravel()
    n0 = _get_df_vector_or_scalar(df, value=n_a, col=n_a_col, size=n, name="n_a")
    n1 = _get_df_vector_or_scalar(df, value=n_b, col=n_b_col, size=n, name="n_b")
    ne_w = _get_df_vector_or_scalar(df, value=ne_weights, col=ne_weights_col, size=n, name="ne_weights")

    if Ne is None:
        Ne_summary = estimate_ne_from_temporal_freqs(
            p_before=p0,
            p_after=p1,
            generations=generations,
            n_before=n0,
            n_after=n1,
            sample_sizes_are=sample_sizes_are,
            method=ne_method,
            model=ne_model,
            diffusion_scale=diffusion_scale,
            min_initial_het=ne_min_initial_het,
            min_maf=ne_min_maf,
            trim_quantile=ne_trim_quantile,
            weights=ne_w,
            n_boot=n_boot_ne,
            random_state=random_state,
        )
        Ne_used = float(Ne_summary["Ne"])
    else:
        Ne_used = float(Ne)
        if Ne_used <= 0:
            raise ValueError("Ne must be positive when supplied.")
        Ne_summary = dict(
            Ne=Ne_used,
            F=temporal_Ne_to_F(Ne_used, generations=generations, model=ne_model, diffusion_scale=diffusion_scale),
            generations=float(generations),
            method="supplied",
            model=str(ne_model),
            diffusion_scale=float(diffusion_scale),
            n_snps_input=int(n),
            n_snps_used=None,
            n_snps_excluded=None,
        )

    F_drift = temporal_Ne_to_F(Ne_used, generations=generations, model=ne_model, diffusion_scale=diffusion_scale)
    n0_chr, n1_chr = _prepare_sample_sizes(n0, n1, sample_sizes_are)

    finite = np.isfinite(p0) & np.isfinite(p1)
    in_bounds = (p0 >= 0.0) & (p0 <= 1.0) & (p1 >= 0.0) & (p1 <= 1.0)
    h0 = p0 * (1.0 - p0)
    testable = finite & in_bounds & (h0 >= float(test_min_initial_het))
    if test_min_maf is not None:
        testable &= np.minimum(p0, 1.0 - p0) >= float(test_min_maf)
    if n0_chr is not None:
        testable &= np.isfinite(n0_chr) & (n0_chr > 0.0)
    if n1_chr is not None:
        testable &= np.isfinite(n1_chr) & (n1_chr > 0.0)

    delta = p1 - p0
    s_hat = np.full(n, np.nan)
    expected_alt = np.full(n, np.nan)
    se_s = np.full(n, np.nan)
    z_s = np.full(n, np.nan)
    ll0 = np.full(n, np.nan)
    ll1 = np.full(n, np.nan)
    lr = np.full(n, np.nan)
    pval = np.full(n, np.nan)
    lod = np.full(n, np.nan)
    var_delta = np.full(n, np.nan)

    idx = np.flatnonzero(testable)
    if idx.size:
        p0m = p0[idx]
        p1m = p1[idx]
        h0m = h0[idx]
        dm = delta[idx]

        sampling_var = np.zeros_like(p0m)
        if n0_chr is not None:
            sampling_var += p0m * (1.0 - p0m) / n0_chr[idx]
        if n1_chr is not None:
            sampling_var += p1m * (1.0 - p1m) / n1_chr[idx]

        V = F_drift * h0m + sampling_var
        V = np.maximum(V, 1e-300)
        var_delta[idx] = V

        sm = estimate_s_closed_form(p0m, p1m, generations=generations, eps=logit_eps, s_bounds=s_bounds)
        s_hat[idx] = sm
        m_alt = deterministic_selected_freq(p0m, s=0.0, generations=0.0)  # allocate shape safely
        m_alt = _inv_logit(_logit_clipped(p0m, eps=logit_eps) + sm * float(generations))
        expected_alt[idx] = m_alt

        # Delta-method SE for s under H0. d logit(p)/dp = 1/[p(1-p)].
        se = np.sqrt(V) / (float(generations) * h0m)
        se_s[idx] = se
        z_s[idx] = sm / se

        ll_null = -0.5 * (np.log(2.0 * np.pi * V) + ((p1m - p0m) ** 2) / V)
        ll_alt = -0.5 * (np.log(2.0 * np.pi * V) + ((p1m - m_alt) ** 2) / V)
        lr_m = np.maximum(0.0, 2.0 * (ll_alt - ll_null))

        ll0[idx] = ll_null
        ll1[idx] = ll_alt
        lr[idx] = lr_m
        pval[idx] = chi2.sf(lr_m, df=1)
        lod[idx] = lr_m / (2.0 * np.log(10.0))

    direction = np.full(n, "not_tested", dtype=object)
    direction[testable & (s_hat > 0.0)] = "positive_increase"
    direction[testable & (s_hat < 0.0)] = "negative_decrease"
    direction[testable & np.isclose(s_hat, 0.0, atol=1e-15)] = "neutral_no_change"

    significant_direction = np.full(n, "not_significant", dtype=object)
    sig = np.isfinite(pval) & (pval < float(alpha))
    significant_direction[sig & (s_hat > 0.0)] = "significant_positive_increase"
    significant_direction[sig & (s_hat < 0.0)] = "significant_negative_decrease"

    qval = _benjamini_hochberg(pval) if add_fdr else np.full(n, np.nan)

    results = pd.DataFrame(
        {
            snp_col: snp_id,
            "p_generation_a": p0,
            "p_generation_b": p1,
            "delta_p": delta,
            "testable": testable,
            "Ne_used": Ne_used,
            "F_drift_used": F_drift,
            "var_delta_used": var_delta,
            "selection_coefficient_s": s_hat,
            "selection_direction": direction,
            "expected_p_b_under_s_hat": expected_alt,
            "se_s": se_s,
            "z_s": z_s,
            "loglike_null_s0": ll0,
            "loglike_alt_s_hat": ll1,
            "lr_stat": lr,
            "p_value": pval,
            "lod_score": lod,
            "q_value_bh": qval,
            "significant_direction": significant_direction,
        }
    )

    if not return_invalid_rows:
        results = results.loc[results["testable"]].reset_index(drop=True)

    Ne_summary = dict(Ne_summary)
    Ne_summary.update(
        Ne_used_for_selection=Ne_used,
        F_drift_used_for_selection=F_drift,
        n_snps_tested=int(np.sum(testable)),
        n_snps_not_tested=int(n - np.sum(testable)),
        alpha=float(alpha),
    )
    return {"Ne": Ne_used, "Ne_summary": Ne_summary, "results": results}


scan_temporal_selection_dataframe = temporal_selection_scan_dataframe
estimate_ne_and_selection_from_dataframe = temporal_selection_scan_dataframe


# =============================================================================
# Validation / smoke tests
# =============================================================================


def check_mass_conservation() -> float:
    """Return maximum absolute mass error across a small parameter grid."""
    errs = []
    for flux_form in ("ito", "fickian"):
        for s in (0.0, 0.02, -0.02):
            out = wf_fp_solve_many(
                [0.0, 0.1, 0.5, 0.9, 1.0],
                Ne=100,
                s=s,
                T=100,
                M=200,
                bump_width=0.005,
                diffusion_scale=4.0,
                flux_form=flux_form,
            )
            errs.append(np.max(np.abs(out["mass_total"] - 1.0)))
    return float(np.max(errs))


def check_temporal_ne_estimator(random_state: int = 1) -> Dict[str, float]:
    """Simulate neutral drift and check whether temporal Ne is recovered."""
    rng = np.random.default_rng(random_state)
    true_Ne = 190.0
    generations = 10
    p0 = rng.uniform(0.05, 0.95, 50_000)
    p1 = p0.copy()
    for _ in range(generations):
        p1 = rng.binomial(int(2 * true_Ne), p1) / (2.0 * true_Ne)
    est = estimate_ne_from_temporal_freqs(p0, p1, generations=generations, method="normal_mle")
    return {"true_Ne": true_Ne, "estimated_Ne": float(est["Ne"]), "F": float(est["F"])}


def check_temporal_selection_scan_dataframe(random_state: int = 2) -> Dict[str, Any]:
    """Simulate drift plus selected SNPs and check that selected loci rank highly."""
    import pandas as pd

    rng = np.random.default_rng(random_state)
    true_Ne = 200.0
    generations = 8
    n_snps = 20_000
    p0 = rng.uniform(0.05, 0.95, n_snps)
    p1 = p0.copy()
    for _ in range(generations):
        p1 = rng.binomial(int(2 * true_Ne), p1) / (2.0 * true_Ne)

    selected = rng.choice(n_snps, size=50, replace=False)
    s_true = 0.20
    p1[selected] = deterministic_selected_freq(p0[selected], s=s_true, generations=generations)

    df = pd.DataFrame(
        {
            "SNP_ID": [f"snp_{i}" for i in range(n_snps)],
            "frequency generation a": p0,
            "frequency generation b": p1,
        }
    )
    out = temporal_selection_scan_dataframe(
        df,
        generations=generations,
        ne_trim_quantile=0.995,
        test_min_maf=0.01,
        alpha=0.05,
    )
    top = out["results"].nsmallest(10, "p_value")
    selected_ids = {f"snp_{i}" for i in selected}
    return {
        "true_Ne": true_Ne,
        "estimated_Ne": float(out["Ne"]),
        "top10_contains_selected": int(np.isin(top["SNP_ID"].values, list(selected_ids)).sum()),
        "best_snp": str(top.iloc[0]["SNP_ID"]),
        "best_snp_is_selected": bool(top.iloc[0]["SNP_ID"] in selected_ids),
        "best_p_value": float(top.iloc[0]["p_value"]),
        "best_lod": float(top.iloc[0]["lod_score"]),
    }


def validate_module() -> Dict[str, Any]:
    """Run all built-in validation checks."""
    return {
        "max_mass_error": check_mass_conservation(),
        "temporal_ne": check_temporal_ne_estimator(),
        "temporal_selection_scan": check_temporal_selection_scan_dataframe(),
    }


if __name__ == "__main__":
    checks = validate_module()
    print("Validation checks")
    print("-----------------")
    print(f"max |mass_total - 1|: {checks['max_mass_error']:.3e}")
    print(f"temporal Ne check:     {checks['temporal_ne']}")
    print(f"selection scan check:  {checks['temporal_selection_scan']}")
