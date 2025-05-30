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


from sklearn.utils.extmath import randomized_svd
from scipy.optimize import minimize
from scipy.linalg import solve
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class BLUPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, method="REML", bounds=(1e-9, 1e9), return_Hinv=False, n_components=None, random_state=42, alpha=1e-6):
        self.method = method
        self.bounds = bounds
        self.return_Hinv = return_Hinv
        self.n_components = n_components
        self.random_state = random_state
        self.alpha = alpha

    def _preprocess_X(self, X):
        if isinstance(X, pd.DataFrame):
            snpcols = list(X.columns[(X.dtypes != 'category') & X.columns.str.contains(r'\w{1,3}[:_]\d+')])
            catcols = list(X.columns[(X.dtypes == 'category') & ~X.columns.isin(snpcols)])
            catdf = pd.get_dummies(X, columns=catcols, dummy_na=True, drop_first=True) if catcols else pd.DataFrame(index=X.index)
            fixedcols = X.columns[~X.columns.isin(snpcols + catcols)]
            Z_f = pd.concat([X[snpcols].astype(float), catdf], axis=1) if len(snpcols + catcols) else None
            X_f = X[fixedcols] if len(fixedcols) else None
            self._snpcol, self._catcols, self._fixedcols = snpcols, catcols, fixedcols
            if Z_f is not None:
                self.rngeffcols = list(Z_f.columns)
        else:
            Z_f = np.array(X)
            X_f = None
        return X_f, Z_f

    def fit(self, y, Z=None, K=None, X=None):
        y = np.asarray(y).reshape(-1, 1)
        n = len(y)
        if self.n_components is None: self.n_components = n//2
        if K is None and Z is None:
            X, Z = self._preprocess_X(X)
        if X is None:
            X = np.ones((n, 1))
        if Z is None:
            Z = np.eye(n)
        if K is None:
            K = np.eye(Z.shape[1])

        K += np.eye(K.shape[0]) * self.alpha  # Regularize K
        XtX = X.T @ X
        XtXinv = np.linalg.inv(XtX)
        S = np.eye(n) - X @ XtXinv @ X.T
        SZKZS = S @ Z @ K @ Z.T @ S

        # Randomized SVD
        U, s, _ = randomized_svd(SZKZS, n_components=self.n_components, random_state=self.random_state)
        theta = s
        Q = U

        omega = Q.T @ y
        omega_sq = omega**2

        if self.method == "ML":
            def likelihood(lambda_):
                return n * np.log(np.sum(omega_sq / (theta + lambda_))) + np.sum(np.log(theta + lambda_))
        else:
            def likelihood(lambda_):
                df = n - X.shape[1]
                return df * np.log(np.sum(omega_sq / (theta + lambda_))) + np.sum(np.log(theta + lambda_))

        soln = minimize(likelihood, x0=1.0, bounds=[self.bounds], method="L-BFGS-B")
        lambda_opt = soln.x[0]

        df = n if self.method == "ML" else n - X.shape[1]
        self.Vu_ = np.sum(omega_sq / (theta + lambda_opt)) / df
        self.Ve_ = lambda_opt * self.Vu_

        phi = np.concatenate([theta, np.ones(n - len(theta))])
        self.Hinv_ = U @ np.diag(1 / (phi + lambda_opt)) @ U.T
        W = X.T @ self.Hinv_ @ X
        self.beta_ = np.linalg.solve(W, X.T @ self.Hinv_ @ y)
        KZt = K @ Z.T
        self.u_ = KZt @ self.Hinv_ @ (y - X @ self.beta_)

        Winv = np.linalg.inv(W)
        self.b_SE = np.sqrt(self.Vu_ * np.diag(Winv))
        WW = KZt @ self.Hinv_ @ KZt.T
        WWW = KZt @ self.Hinv_ @ X @ Winv @ X.T @ self.Hinv_ @ KZt.T
        u_var = np.diag(K) - np.diag(WW) + np.diag(WWW)
        self.u_SE = np.sqrt(self.Vu_ * u_var)
        return self

    def predict(self, X=None, Z=None):
        if X is None: X = np.ones((len(Z), 1))
        else:  X, Z = self._preprocess_X(X)
        if Z is None: Z = np.eye(len(X))
        y_pred = X @ self.beta_
        if self.u_ is not None: y_pred += Z @ self.u_
        return y_pred

class BLUPRegressorSVD(BaseEstimator, RegressorMixin):
    def __init__(self, method="REML", bounds=(1e-9, 1e9), return_Hinv=False, 
                 n_components=None, random_state=42, alpha=1e-6):
        self.method = method
        self.bounds = bounds
        self.return_Hinv = return_Hinv
        self.n_components = n_components
        self.random_state = random_state
        self.alpha = alpha

    def _preprocess_X(self, X):
        # (Your original _preprocess_X code goes here.)
        if isinstance(X, pd.DataFrame):
            snpcols = list(X.columns[(X.dtypes != 'category') & X.columns.str.contains(r'\w{1,3}[:_]\d+')])
            catcols = list(X.columns[(X.dtypes == 'category') & ~X.columns.isin(snpcols)])
            catdf = pd.get_dummies(X, columns=catcols, dummy_na=True, drop_first=True) if catcols else pd.DataFrame(index=X.index)
            fixedcols = X.columns[~X.columns.isin(snpcols + catcols)]
            Z_f = pd.concat([X[snpcols].astype(float), catdf], axis=1) if len(snpcols + catcols) else None
            X_f = X[fixedcols] if len(fixedcols) else None
            self._snpcol, self._catcols, self._fixedcols = snpcols, catcols, fixedcols
            if Z_f is not None:
                self.rngeffcols = list(Z_f.columns)
        else:
            Z_f = np.array(X)
            X_f = None
        return X_f, Z_f

    def fit(self, y, Z=None, K=None, X=None):
        y = np.asarray(y).reshape(-1, 1)
        n = len(y)
        if self.n_components is None: self.n_components = n // 2
        if K is None and Z is None: X, Z = self._preprocess_X(X)
        if X is None: X = np.ones((n, 1))
        if Z is None: Z = np.eye(n)
        if K is None: K = np.eye(Z.shape[1])
        K += np.eye(K.shape[0]) * self.alpha  
        XtX = X.T @ X
        tol_eig = 1e-9
        w, V = np.linalg.eigh(XtX)
        w[w < tol_eig] = tol_eig
        XtXinv = V @ np.diag(1.0 / w) @ V.T
        S = np.eye(n) - X @ XtXinv @ X.T
        SZKZS = S @ Z @ K @ Z.T @ S
        U, s, _ = randomized_svd(SZKZS, n_components=self.n_components, random_state=self.random_state)
        theta = s
        Q = U
        omega = Q.T @ y
        omega_sq = omega**2
        if self.method == "ML":
            def likelihood(lambda_):
                return n * np.log(np.sum(omega_sq / (theta + lambda_))) + np.sum(np.log(theta + lambda_))
        else:
            def likelihood(lambda_):
                df = n - X.shape[1]
                return df * np.log(np.sum(omega_sq / (theta + lambda_))) + np.sum(np.log(theta + lambda_))
        soln = minimize(likelihood, x0=1.0, bounds=[self.bounds], method="L-BFGS-B")
        lambda_opt = soln.x[0]
        df_effective = n if self.method == "ML" else n - X.shape[1]
        self.Vu_ = np.sum(omega_sq / (theta + lambda_opt)) / df_effective
        self.Ve_ = lambda_opt * self.Vu_
        phi = np.concatenate([theta, np.ones(n - len(theta))])
        self.Hinv_ = U @ np.diag(1 / (phi + lambda_opt)) @ U.T
        W = X.T @ self.Hinv_ @ X
        wW, VW = np.linalg.eigh(W)
        wW[wW < tol_eig] = tol_eig
        W_inv = VW @ np.diag(1.0 / wW) @ VW.T
        self.beta_ = W_inv @ (X.T @ self.Hinv_ @ y)
        
        KZt = K @ Z.T
        self.u_ = KZt @ self.Hinv_ @ (y - X @ self.beta_)
        
        self.b_SE = np.sqrt(self.Vu_ * np.diag(W_inv))
        WW = KZt @ self.Hinv_ @ KZt.T
        WWW = KZt @ self.Hinv_ @ X @ W_inv @ X.T @ self.Hinv_ @ KZt.T
        u_var = np.diag(K) - np.diag(WW) + np.diag(WWW)
        self.u_SE = np.sqrt(self.Vu_ * u_var)
        return self

    def predict(self, X=None, Z=None):
        if X is None: X = np.ones((len(Z), 1))
        else: X, Z = self._preprocess_X(X)
        if Z is None: Z = np.eye(len(X))
        y_pred = X @ self.beta_
        if self.u_ is not None: y_pred += Z @ self.u_
        return y_pred