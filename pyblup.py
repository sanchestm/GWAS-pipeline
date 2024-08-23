import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve, svd, qr, eig

def pyblup(y, Z=None, K=None, X=None, method="REML", bounds=(1e-09, 1e+09), SE=False, return_Hinv=False):
    pi = np.pi
    n = len(y)
    y = np.array(y).reshape((n, 1))
    not_NA = np.where(~np.isnan(y))[0]

    if X is None:
        X = np.ones((n, 1))
    else:
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((len(X), 1))

    if Z is None:
        Z = np.eye(n)
    else:
        Z = np.array(Z)
        if len(Z.shape) == 1:
            Z = Z.reshape((len(Z), 1))

    if K is not None:
        K = np.array(K)

    Z = Z[not_NA, :]
    X = X[not_NA, :]
    y = y[not_NA, :]
    n = len(not_NA)

    XtX = X.T @ X
    rank_X = np.linalg.matrix_rank(XtX)
    if rank_X < X.shape[1]:
        raise ValueError("X not full rank")
    XtXinv = np.linalg.inv(XtX)
    S = np.eye(n) - X @ XtXinv @ X.T

    if n <= Z.shape[1] + X.shape[1]:
        spectral_method = "eigen"
    else:
        spectral_method = "cholesky"
        if K is not None:
            np.fill_diagonal(K, K.diagonal() + 1e-6)
            try:
                B = cholesky(K)
            except np.linalg.LinAlgError:
                raise ValueError("K not positive semi-definite")

    if spectral_method == "cholesky":
        ZBt = Z if K is None else Z @ B.T
        svd_ZBt = svd(ZBt, full_matrices=False)
        U = svd_ZBt[0]
        phi = np.concatenate([svd_ZBt[1] ** 2, np.zeros(n - Z.shape[1])])
        SZBt = S @ ZBt
        try:
            svd_SZBt = svd(SZBt, full_matrices=False)
        except np.linalg.LinAlgError:
            svd_SZBt = svd(SZBt + np.eye(SZBt.shape[0]) * 1e-10, full_matrices=False)
        Q, R = qr(np.hstack([X, svd_SZBt[0]]))
        Q = Q[:, X.shape[1]:]
        R = R[X.shape[1]:, X.shape[1]:]
        try:
            theta = solve(R.T @ R, svd_SZBt[1] ** 2)
            theta = np.concatenate([theta, np.zeros(n - X.shape[1] - Z.shape[1])])
        except np.linalg.LinAlgError:
            spectral_method = "eigen"

    if spectral_method == "eigen":
        offset = np.sqrt(n)
        Hb = Z @ K @ Z.T + offset * np.eye(n) if K is not None else Z @ Z.T + offset * np.eye(n)
        Hb_system = eig(Hb)
        phi = Hb_system[0] - offset
        # if min(phi) < -1e-6:
        #     raise ValueError("K not positive semi-definite")
        U = Hb_system[1]
        SHbS = S @ Hb @ S
        SHbS_system = eig(SHbS)
        theta = SHbS_system[0][:n - X.shape[1]] - offset
        Q = SHbS_system[1][:, :n - X.shape[1]]

    omega = Q.T @ y
    omega_sq = omega ** 2

    if method == "ML":
        def f_ML(lambda_, n, theta, omega_sq, phi):
            return n * np.log(np.sum(omega_sq / (theta + lambda_))) + np.sum(np.log(phi + lambda_))
        soln = minimize(f_ML, x0=1, args=(n, theta, omega_sq, phi), bounds=[bounds])
    else:
        def f_REML(lambda_, n_p, theta, omega_sq):
            return n_p * np.log(np.sum(omega_sq / (theta + lambda_))) + np.sum(np.log(theta + lambda_))
        soln = minimize(f_REML, x0=.5, args=(n - X.shape[1], theta, omega_sq), bounds=[bounds])

    lambda_opt = soln.x[0]
    df = n if method == "ML" else n - X.shape[1]
    Vu_opt = np.sum(omega_sq / (theta + lambda_opt)) / df
    Ve_opt = lambda_opt * Vu_opt
    Hinv = U @ np.linalg.inv(U.T @ np.diag(phi + lambda_opt) @ U)

    W = X.T @ Hinv @ X
    beta = np.linalg.solve(W, X.T @ Hinv @ y)
    
    if K is None:
        KZt = Z.T
    else:
        KZt = K @ Z.T

    KZt_Hinv = KZt @ Hinv
    u = KZt_Hinv @ (y - X @ beta)

    LL = -0.5 * (soln.fun + df + df * np.log(2 * pi / df))

    if SE:
        Winv = np.linalg.inv(W)
        beta_SE = np.sqrt(Vu_opt * np.diag(Winv))
        WW = KZt_Hinv @ KZt_Hinv.T
        WWW = KZt_Hinv @ X @ Winv @ X.T @ KZt_Hinv.T
        if K is None:
            u_SE = np.sqrt(Vu_opt * (np.ones(Z.shape[1]) - np.diag(WW) + np.diag(WWW)))
        else:
            u_SE = np.sqrt(Vu_opt * (np.diag(K) - np.diag(WW) + np.diag(WWW)))
        
        if return_Hinv:
            return {'Vu': np.float64(Vu_opt), 'Ve': np.float64(Ve_opt), 
                    'beta': np.float64(beta), 'beta_SE': np.float64(beta_SE), 
                    'u': np.float64(u), 'u_SE': np.float64(u_SE), 'LL': np.float64(LL), 'Hinv': np.float64(Hinv)}
        else:
            return {'Vu': np.float64(Vu_opt), 'Ve': np.float64(Ve_opt), 
                    'beta': np.float64(beta), 'beta_SE': np.float64(beta_SE), 
                    'u': np.float64(u), 'u_SE': np.float64(u_SE), 'LL': np.float64(LL)}
    else:
        if return_Hinv:
            return {'Vu': np.float64(Vu_opt), 'Ve': np.float64(Ve_opt), 
                    'beta': np.float64(beta), 'u': np.float64(u), 'LL': np.float64(LL), 'Hinv': np.float64(Hinv)}
        else:
            return {'Vu': np.float64(Vu_opt), 'Ve': np.float64(Ve_opt), 
                    'beta': np.float64(beta), 'u': np.float64(u), 'LL': np.float64(LL)}

