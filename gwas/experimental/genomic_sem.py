from . import npplink
import pandas as pd
import numpy as np
import pymc as pm
import networkx as nx
import holoviews as hv
import hvplot.networkx as hvnx
from scipy import sparse, linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import pandas as pd
import networkx as nx
import pymc as pm
import pytensor.tensor as pt
from scipy import sparse
from scipy.sparse.linalg import eigsh
import panel as pn
import hvplot.pandas
import hvplot.dask 
import pytensor
from functools import reduce

class CausalGraph:
    """
    DAG -> PyMC compiler with:
      - Exogenous nodes (pm.Data)
      - GRM nodes (low-rank X = U*sqrt(S))
      - Latent nodes (random variables)
      - Endogenous nodes (observed: normal / poisson / bernoulli)

    Alignment to your model1 efficiency:
      GRM -> target uses:  contrib = (X_grm @ Z @ L.T)
        - X_grm: N x r  (U*sqrt(S))
        - Z:     r x d  (PC-space loadings)
        - L:     d x d  (LKJ Cholesky over target dim)
      This avoids dense beta matrices for GRM edges.

    Trait blocking:
      Do it by creating multiple endo nodes (blocks). Shared residual structure across blocks must be via
      an explicit node (e.g., Residual_Factor) and edges.
    """

    # -----------------------------
    # init / basic utilities
    # -----------------------------
    def __init__(self, df_main: pd.DataFrame):
        self.df_main = df_main
        self.obs_ids = df_main.index.tolist()
        self.G = nx.DiGraph()

    @staticmethod
    def _pc_labels(r: int, prefix: str = "Pc") -> list[str]: return [f"{prefix}{i}" for i in range(1, r + 1)]

    @staticmethod
    def _latent_labels(name: str, d: int) -> list[str]: return [f"{name}{i}" for i in range(1, d + 1)]

    @staticmethod
    def _as_str_coords(coords) -> list[str]: return [str(c) for c in coords]

    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1: x = x.reshape(-1, 1)
        return x

    def _align_to_obs(self, data):
        if isinstance(data, (pd.Series, pd.DataFrame)): return data.reindex(self.df_main.index)
        return data

    @staticmethod
    def _masked_or_subset(y: np.ndarray, *, impute_missing: bool):
        """
        For observed likelihoods:
          - If impute_missing: return masked array (works well for Normal and often for discrete too).
          - Else: return (y_obs, obs_idx) where obs_idx indexes observed rows.
        """
        y = np.asarray(y)
        if not np.isnan(y).any(): return y, None
        if impute_missing: return np.ma.masked_invalid(y), None
        # subset rows that have any observed values (for multivariate y) if we want to skip imputation
        if y.ndim == 1:
            obs_idx = np.where(~np.isnan(y))[0]
            return y[obs_idx], obs_idx
        obs_idx = np.where(~np.isnan(y).any(axis=1))[0]
        return y[obs_idx], obs_idx

    def _safe_py_scalar(x, default=None):
        """Convert numpy/pandas scalars to plain Python scalars, else fallback."""
        try:
            if x is None: return default
            if isinstance(x, (np.generic,)):  return x.item()
            if isinstance(x, (str, int, float, bool)): return x
            return x
        except Exception: return default
    
    def _shape_str(shape):
        """Turn (N,d) etc into a safe string."""
        try:
            if shape is None:  return "?"
            if isinstance(shape, (tuple, list)): return "x".join(str(int(_safe_py_scalar(s, s))) for s in shape)
            return str(shape)
        except Exception: return "?"

    # -----------------------------
    # eigendecomposition helper
    # -----------------------------
    @staticmethod
    def decompose_grm(K, *, r: int, jitter: float = 1e-6, use_sparse: bool = True, random_state: int = 42):
        n = K.shape[0]
        K = 0.5 * (K + K.T)
        r = max(1, min(int(r), n - 1))
        if sparse.issparse(K):
            K += jitter * sparse.eye(n, format="csr")
            if use_sparse:
                v0 = np.random.RandomState(random_state).uniform(size=n)
                evals, evecs = eigsh(K, k=r, which="LA", v0=v0)
                order = np.argsort(evals)[::-1]
                evals, evecs = evals[order], evecs[:, order]
            else:
                Kd = K.toarray()
                evals, evecs = np.linalg.eigh(Kd)
                evals, evecs = evals[::-1], evecs[:, ::-1]
                evals, evecs = evals[:r], evecs[:, :r]
        else:
            K = np.asarray(K, dtype=float)
            K += jitter * np.eye(n)
            evals, evecs = np.linalg.eigh(K)
            evals, evecs = evals[::-1], evecs[:, ::-1]
            evals, evecs = evals[:r], evecs[:, :r]
        evals = np.clip(evals, 0.0, np.inf)
        return evecs, evals

    @staticmethod
    def _resolve_r(n: int, n_components):
        """
        n_components can be:
          - int: keep that many
          - float in (0,1): keep that fraction of n (lightweight heuristic)
        """
        if isinstance(n_components, int): r = n_components
        elif isinstance(n_components, float) and 0 < n_components < 1: r = int(np.ceil(n_components * n))
        else: raise ValueError("n_components must be int or float in (0,1).")
        return max(1, min(int(r), n - 1))

    # -----------------------------
    # node creation
    # -----------------------------
    def add_input(self, name, data, *, fillna_for_exo: float = 0.0):
        """
        Exogenous predictors (pm.Data). NaNs are filled (predictors can't be 'missing' here).
        """
        data = self._align_to_obs(data)
        if isinstance(data, pd.Series):
            arr = data.to_numpy(dtype=float).reshape(-1, 1)
            coord_names = [data.name if data.name is not None else f"{name}1"]
        elif isinstance(data, pd.DataFrame):
            arr = data.to_numpy(dtype=float)
            coord_names = list(data.columns)
        else:
            arr = np.asarray(data, dtype=float)
            arr = self._ensure_2d(arr)
            coord_names = self._pc_labels(arr.shape[1], prefix=f"{name}_")
        if np.isnan(arr).any(): arr = np.nan_to_num(arr, nan=float(fillna_for_exo))
        self.G.add_node( name, type="exo", data=arr, shape=arr.shape, dim_name=f"{name}_dim",
                         coords=self._as_str_coords(coord_names), track_values=False, track_values_per_path=False)

    def add_grm(
        self,
        name,
        grm_matrix,
        *,
        n_components=300,
        top_eigen2fixed: float | None = None,     # fraction of variance to move into fixed node
        subtract_fixed: str | None = None,        # project out an exo node from this GRM
        add_UMAP: bool = False,
        umap_n_components: int = 2,
        umap_kwargs: dict | None = None,
        jitter: float = 1e-6,
        use_sparse: bool = False,
        random_state: int = 42,
    ):
        """
        Adds a GRM node as low-rank X = U*sqrt(S).

        Optional: split top PCs into an exo "Fixed" node:
          - choose cutoff by cumulative variance ratio until it reaches top_eigen2fixed
          - fixed node data = [U_fixed * sqrt(S_fixed), UMAP(U_fixed)] if add_UMAP
          - remaining PCs become the random GRM node

        Optional: subtract_fixed projects out an existing exo node from X.

        Notes:
          - if top_eigen2fixed is provided, we decompose to r_total = n_components and then split
          - UMAP is computed on U_fixed (unscaled) by default, matching your old intent.
        """
        K = grm_matrix
        if isinstance(K, pd.DataFrame):
            idx = self.df_main.index.intersection(K.index).intersection(K.columns)
            K = K.loc[idx, idx].to_numpy(dtype=float)
        else: K = np.asarray(K, dtype=float)
        n = K.shape[0]
        r_total = self._resolve_r(n, n_components)
        U, S = self.decompose_grm( K, r=r_total, jitter=jitter, use_sparse=use_sparse, random_state=random_state)
        X_full = U * np.sqrt(S)[None, :]
        if top_eigen2fixed is not None:
            if not (0 < float(top_eigen2fixed) < 1): raise ValueError("top_eigen2fixed must be a float in (0,1) representing variance fraction.")
            total = np.sum(S) if np.sum(S) > 0 else 1.0
            cum = np.cumsum(S) / total
            cutoff = int(np.searchsorted(cum, float(top_eigen2fixed)) + 1)
            cutoff = max(1, min(cutoff, r_total - 1))  # ensure at least 1 fixed and 1 random
            U_fixed = U[:, :cutoff]
            S_fixed = S[:cutoff]
            X_fixed = U_fixed * np.sqrt(S_fixed)[None, :]
            if add_UMAP:
                try: import umap
                except Exception as e: raise ImportError("add_UMAP=True requires umap-learn to be installed.") from e
                umap_kwargs = {} if umap_kwargs is None else dict(umap_kwargs)
                reducer = umap.UMAP(n_components=int(umap_n_components), random_state=random_state, **umap_kwargs)
                umap_emb = reducer.fit_transform(U_fixed)  # use unscaled eigenvectors
                fixed_data = np.hstack([X_fixed, umap_emb])
            else: fixed_data = X_fixed
            self.G.add_node(f"{name}_fixed", type="exo",
                            data=fixed_data.astype(float), shape=fixed_data.shape,
                            dim_name=f"{name}_fixed_dim", 
                            coords=self._as_str_coords(self._pc_labels(cutoff, prefix="Pc") + ([f"UMAP{x}" for x in range(1, umap_n_components+1)] if add_UMAP else [])),
                            track_values=False,track_values_per_path=False )
            X_full = X_full[:, cutoff:]
        # Optional orthogonalization: subtract_fixed (must be exo node)
        if subtract_fixed is not None:
            if subtract_fixed not in self.G.nodes: raise ValueError(f"subtract_fixed='{subtract_fixed}' not found.")
            fix = self.G.nodes[subtract_fixed]
            if fix["type"] != "exo": raise ValueError(f"subtract_fixed must be an exo node; got {fix['type']}.")
            C = fix["data"]  # (N, p)
            beta, *_ = np.linalg.lstsq(C, X_full, rcond=None)
            X_full = X_full - C @ beta
            norms = np.linalg.norm(X_full, axis=0)
            norms[norms < 1e-12] = 1.0
            X_full = X_full / norms[None, :]
            
        fixed_node_name = f"{name}_fixed" if top_eigen2fixed is not None else None
        self.G.add_node(name, type="grm", data=X_full.astype(float),
                        shape=X_full.shape, dim_name=f"{name}_pc", 
                        coords=self._as_str_coords(self._pc_labels(X_full.shape[1], prefix="Pc")),
                        track_values=False, track_values_per_path=False,
                        fixed_node=fixed_node_name)

    def add_latent(self,name,*,n_dim=1,prior_sigma=1.0,
                   track_values=True,track_values_per_path=False,variance_budget = True,
                  budget_w_eps=1e-6, budget_concentration =20):
        n_dim = int(n_dim)
        shape = (len(self.obs_ids), n_dim)
        self.G.add_node(name,type="latent",data=None,shape=shape,dim_name=f"{name}_dim",
                        coords=self._as_str_coords(self._latent_labels(name, n_dim)),
                        prior_sigma=float(prior_sigma),track_values=track_values,track_values_per_path=track_values_per_path,variance_budget = variance_budget,
                       budget_w_eps=budget_w_eps, budget_concentration =budget_concentration)

    def add_trait(
        self,
        name,
        *,
        data_matrix=None,
        data_cols=None,
        likelihood="normal",          # "normal" | "poisson" | "bernoulli"
        link="identity",              # normal: identity; poisson: log; bernoulli: logit
        residual="diag",              # normal only: "diag" | "lkj"
        eta_E=4.0,
        var_E=1.0,
        impute_missing=True,
        track_values=True,
        track_values_per_path=False,
        variance_budget = True,
        budget_w_eps=1e-6, budget_concentration =20
    ):
        """
        Endogenous observed node.
          - NaNs allowed; handled via masked arrays (if impute_missing=True) or subsetting rows (fast/simple).
          - If you want shared residuals across blocks, do it via a shared latent node + edges.
        """
        if data_matrix is None and data_cols is None:raise ValueError("Provide data_matrix or data_cols.")
        if data_matrix is not None: data = data_matrix
        else:
            if isinstance(data_cols, str):data = self.df_main[data_cols]
            else: data = self.df_main[list(data_cols)]

        data = self._align_to_obs(data)
        arr = data.to_numpy(dtype=float) if isinstance(data, (pd.Series, pd.DataFrame)) else np.asarray(data, dtype=float)
        arr = self._ensure_2d(arr)
        if isinstance(data, pd.Series): coord_names = [data.name if data.name is not None else f"{name}1"]
        elif isinstance(data, pd.DataFrame): coord_names = list(data.columns)
        else: coord_names = [f"{name}{i}" for i in range(1, arr.shape[1] + 1)]
        
        likelihood = str(likelihood).lower()
        link = str(link).lower()
        residual = str(residual).lower()

        # reasonable defaults
        if likelihood == "normal" and link == "identity":pass
        elif likelihood == "poisson" and link == "identity": link = "log"
        elif likelihood == "bernoulli" and link == "identity": link = "logit"

        self.G.add_node(
            name,
            type="endo",
            data=arr,
            shape=arr.shape,
            dim_name=f"{name}_dim",
            coords=coord_names,
            likelihood=likelihood,
            link=link,
            residual=residual,
            eta_E=float(eta_E),
            sd_E=float(np.sqrt(var_E)),
            impute_missing=bool(impute_missing),
            track_values=track_values,
            track_values_per_path=track_values_per_path,
            variance_budget = variance_budget,
            budget_w_eps=budget_w_eps, 
            budget_concentration =budget_concentration
        )
    def add_edge(
        self,
        source,
        target,
        *,
        kind="auto",           # "auto" | "dense" | "grm"
        prior_variance=1.0,    # used for dense regression sigma
        eta=1.0,               # LKJ eta for GRM edges
        track_variance=False,
        adjust_variance_by_shape = True):
        if source not in self.G.nodes or target not in self.G.nodes: raise ValueError(f"Missing nodes for edge {source} -> {target}")

        src,tgt = self.G.nodes[source], self.G.nodes[target]
        if kind == "auto": kind = "grm" if src["type"] == "grm" else "dense"
        p_dim = int(src['shape'][1])
        n_dim = int(tgt['shape'][1])
        n_parameters = p_dim*n_dim
        if kind == "grm": n_parameters += n_dim*(n_dim+1)//2
        if adjust_variance_by_shape and kind!='grm': prior_variance /= src['shape'][1]
        prior_variance = float(np.sqrt(max(float(prior_variance), 1e-12)))
        self.G.add_edge(source, target, kind=str(kind),
                        prior_sigma=prior_variance, n_parameters=int(n_parameters),
                        eta=float(eta), sd_G=prior_variance, adjust_variance_by_shape= adjust_variance_by_shape,
                        track_variance=bool(track_variance))
    # -----------------------------
    # build model
    # -----------------------------
    def build_model(self):
        order = list(nx.topological_sort(self.G))
        coords = {"obs_id": self._as_str_coords(self.obs_ids)}
        for node, attr in self.G.nodes(data=True): 
            coords[attr["dim_name"]] =  self._as_str_coords(attr["coords"])
        with pm.Model(coords=coords) as model:
            model_vars = {}
            # 1) create pm.Data for exo & grm nodes
            for node in order:
                attr = self.G.nodes[node]
                if attr["type"] in ("exo", "grm"):
                    model_vars[node] = pm.Data( node, attr["data"], dims=("obs_id", attr["dim_name"])   )

            # 2) now create latents & endo likelihoods in topo order
            for node in order:
                attr = self.G.nodes[node]
                node_type = attr["type"]
                dim_name = attr["dim_name"]
                n_dim = int(attr["shape"][1])

                if node_type in ("exo", "grm"): continue
                parents = list(self.G.predecessors(node))
                mu = pt.zeros((len(self.obs_ids), n_dim), dtype=pytensor.config.floatX)
                
                for parent in parents:
                    p_attr = self.G.nodes[parent]
                    edge = self.G.edges[parent, node]                        
                    kind = edge["kind"]
                    parent_data = model_vars[parent]
                    p_dim = int(p_attr["shape"][1])
                    p_dim_name = p_attr["dim_name"]
                    
                    if kind == "grm":
                        # contrib = X_grm @ Z @ L.T
                        r = p_dim
                        Z = pm.Normal( f"Z_{parent}->{node}", mu=0.0,  sigma=1.0,dims=(p_dim_name, dim_name))
                        # L  = pm.LKJCholeskyCov(f"cholG:{parent}->{node}", n=n_dim, eta=edge.get("eta", 1.0),
                        #                        sd_dist=pm.TruncatedNormal.dist(mu=0.05, sigma=edge.get("sd_G", 1.0), lower=0.05),
                        #                        compute_corr=False)
                        L  = pm.LKJCholeskyCov(f"cholG:{parent}->{node}", n=n_dim, eta=edge['eta'],
                                               sd_dist=pm.TruncatedNormal.dist(mu= edge['sd_G'], sigma= .4*edge['sd_G'], lower=0.01),
                                               compute_corr=False)               
                        if edge.get("track_variance", False) or attr.get("track_values_per_path", False):
                            model.add_coords({dim_name+'_t': coords[dim_name]})
                            L = pm.Deterministic(f"L:{parent}->{node}", 
                                                 pm.expand_packed_triangular(n_dim, L, lower=True),
                                                 dims=(dim_name, dim_name+'_t'))
                        else : L = pm.expand_packed_triangular(n_dim, L, lower=True)
                        contrib = pt.dot(pt.dot(parent_data, Z), L.T)
                    elif kind == "dense":
                        beta = pm.Normal(f"beta:{parent}->{node}",  mu=0.0, 
                                         sigma=edge.get("prior_sigma", 1.0),dims=(p_dim_name, dim_name))
                        contrib = pt.dot(parent_data, beta)
                    else: raise ValueError(f"Unknown edge kind '{kind}' for {parent}->{node}")
                    if attr.get("track_values_per_path", False):  pm.Deterministic( f"prediction:{parent}->{node}", contrib, dims=("obs_id", dim_name))
                    if edge.get("track_variance", False): pm.Deterministic(f"variance:{parent}->{node}", contrib.var(axis=0))
                    mu = mu + contrib

                if attr.get("track_values", False) and len(parents) > 0:
                    pm.Deterministic(f"prediction:{node}", mu, dims=("obs_id", dim_name))

                # define node variable
                if node_type == "latent":
                    sigma_lat = float(attr.get("prior_sigma", 1.0))
                    offset = pm.Normal(
                        f"{node}:offset",
                        mu=0.0,
                        sigma=sigma_lat,
                        dims=("obs_id", dim_name),
                    )
                    model_vars[node] = pm.Deterministic(node, mu + offset, dims=("obs_id", dim_name))

                elif node_type == "endo":
                    lik = attr.get("likelihood", "normal")
                    link = attr.get("link", "identity")
                    impute_missing = bool(attr.get("impute_missing", True))
                    Y = attr["data"] 
                    if lik == "normal":
                        # Normal with either diag sigma or LKJ MvNormal
                        if n_dim == 1:
                            y_obs, obs_idx = self._masked_or_subset(Y[:, 0], impute_missing=impute_missing)
                            sigma = pm.HalfNormal(f"{node}:sigma", 1.0)
                            if obs_idx is None: pm.Normal(node, mu=mu[:, 0], sigma=sigma, observed=y_obs, dims=("obs_id",))
                            else:  pm.Normal(node, mu=mu[obs_idx, 0], sigma=sigma, observed=y_obs, dims=("obs_id",))

                        else:
                            residual = attr.get("residual", "diag")
                            y_obs, obs_idx = self._masked_or_subset(Y, impute_missing=impute_missing)
                            if residual == "lkj":
                                # sigmae = pm.TruncatedNormal.dist(mu=edge["sd_E"], sigma=edge["sd_E"]*.3, lower=0.005)
                                sigmae = pm.TruncatedNormal.dist(mu=attr["sd_E"], sigma=.4*attr["sd_E"], lower=0.005)
                                L_E  = pm.LKJCholeskyCov(f"{node}:cholE", n=n_dim,eta=float(attr["eta_E"]),
                                                         sd_dist=sigmae,compute_corr=False )
                                L_E = pm.expand_packed_triangular(n_dim, L_E, lower=True)
                                if obs_idx is None: pm.MvNormal(node, mu=mu, chol=L_E, observed=y_obs, dims=("obs_id", dim_name))
                                else: pm.MvNormal(node, mu=mu[obs_idx], chol=L_E, observed=y_obs, dims=("obs_id", dim_name))
                            elif residual == "diag":
                                sigma = pm.HalfNormal(f"{node}:sigma", 1.0, dims=(dim_name,))
                                if obs_idx is None: pm.Normal(node, mu=mu, sigma=sigma, observed=y_obs, dims=("obs_id", dim_name))
                                else: pm.Normal(node, mu=mu[obs_idx], sigma=sigma, observed=y_obs, dims=("obs_id", dim_name))
                            else:  raise ValueError(f"Unknown residual='{residual}' for normal endo node '{node}'")

                    elif lik == "poisson":
                        if link == "log": rate = pm.math.exp(mu)
                        elif link == "identity": rate = pm.math.maximum(mu, 1e-12)
                        else: raise ValueError(f"Unsupported link='{link}' for poisson")

                        # observed handling
                        y = Y
                        if n_dim == 1:
                            y1 = y[:, 0]
                            if np.isnan(y1).any() and not impute_missing:
                                obs_idx = np.where(~np.isnan(y1))[0]
                                y_obs = y1[obs_idx].astype(int)
                                pm.Poisson(node, mu=rate[obs_idx, 0], observed=y_obs, dims=("obs_id",))
                            else:
                                y_obs = np.ma.masked_invalid(y1).astype(float) if np.isnan(y1).any() else y1.astype(int)
                                pm.Poisson(node, mu=rate[:, 0], observed=y_obs, dims=("obs_id",))
                        else:
                            for j in range(n_dim):
                                yj = y[:, j]
                                nm = f"{node}:{j}"
                                if np.isnan(yj).any() and not impute_missing:
                                    obs_idx = np.where(~np.isnan(yj))[0]
                                    y_obs = yj[obs_idx].astype(int)
                                    pm.Poisson(nm, mu=rate[obs_idx, j], observed=y_obs, dims=("obs_id",))
                                else:
                                    y_obs = np.ma.masked_invalid(yj).astype(float) if np.isnan(yj).any() else yj.astype(int)
                                    pm.Poisson(nm, mu=rate[:, j], observed=y_obs, dims=("obs_id",))

                    elif lik == "bernoulli":
                        if link == "logit": p = pm.math.sigmoid(mu)
                        elif link == "identity": p = pm.math.clip(mu, 0.0, 1.0)
                        else: raise ValueError(f"Unsupported link='{link}' for bernoulli")
                        y = Y
                        if n_dim == 1:
                            y1 = y[:, 0]
                            if np.isnan(y1).any() and not impute_missing:
                                obs_idx = np.where(~np.isnan(y1))[0]
                                y_obs = y1[obs_idx].astype(int)
                                pm.Bernoulli(node, p=p[obs_idx, 0], observed=y_obs, dims=("obs_id",))
                            else:
                                y_obs = np.ma.masked_invalid(y1).astype(float) if np.isnan(y1).any() else y1.astype(int)
                                pm.Bernoulli(node, p=p[:, 0], observed=y_obs, dims=("obs_id",))
                        else:
                            for j in range(n_dim):
                                yj = y[:, j]
                                nm = f"{node}:{j}"
                                if np.isnan(yj).any() and not impute_missing:
                                    obs_idx = np.where(~np.isnan(yj))[0]
                                    y_obs = yj[obs_idx].astype(int)
                                    pm.Bernoulli(nm, p=p[obs_idx, j], observed=y_obs, dims=("obs_id",))
                                else:
                                    y_obs = np.ma.masked_invalid(yj).astype(float) if np.isnan(yj).any() else yj.astype(int)
                                    pm.Bernoulli(nm, p=p[:, j], observed=y_obs, dims=("obs_id",))

                    else: raise ValueError(f"Unsupported likelihood='{lik}' for node '{node}'")
                    model_vars[node] = Y
                else: raise ValueError(f"Unknown node type: {node_type}")
        return model

    def build_model_predictive_vc(self):
            """
            Predictive-first model:
              - No Dirichlet budgets
              - GRM edges: low-rank random effect via X_grm @ Z @ L.T with LKJCholeskyCov sd_dist ~ HalfNormal(sd_G)
              - Dense edges: beta ~ Normal(0, prior_sigma)
              - Normal endo residual: sigma ~ HalfNormal(sd_E) (diag) or LKJCholeskyCov with sd_dist ~ HalfNormal(sd_E) (lkj)
            """
            order = list(nx.topological_sort(self.G))
            coords = {"obs_id": self._as_str_coords(self.obs_ids)}
            obs_ids_np = np.asarray(self.obs_ids)
    
            for node, attr in self.G.nodes(data=True):
                coords[attr["dim_name"]] = self._as_str_coords(attr["coords"])
    
            with pm.Model(coords=coords) as model:
                model_vars = {}
    
                # 1) pm.Data for exo & grm nodes
                for node in order:
                    attr = self.G.nodes[node]
                    if attr["type"] in ("exo", "grm"):
                        model_vars[node] = pm.Data(node, attr["data"], dims=("obs_id", attr["dim_name"]))
    
                # 2) latents & endo likelihoods
                for node in order:
                    attr = self.G.nodes[node]
                    node_type = attr["type"]
                    dim_name = attr["dim_name"]
                    n_dim = int(attr["shape"][1])
    
                    if node_type in ("exo", "grm"):
                        continue
    
                    parents = list(self.G.predecessors(node))
                    mu = pt.zeros((len(self.obs_ids), n_dim), dtype=pytensor.config.floatX)
    
                    # ---------- parent contributions ----------
                    for parent in parents:
                        p_attr = self.G.nodes[parent]
                        edge = self.G.edges[parent, node]
                        kind = str(edge.get("kind", "dense"))
                        parent_data = model_vars[parent]
                        p_dim_name = p_attr["dim_name"]
    
                        if kind == "grm":
                            Z = pm.Normal(
                                f"Z_{parent}->{node}",
                                mu=0.0,
                                sigma=1.0,
                                dims=(p_dim_name, dim_name),
                            )
    
                            L = pm.LKJCholeskyCov(
                                f"cholG:{parent}->{node}",
                                n=n_dim,
                                eta=float(edge.get("eta", 1.0)),
                                sd_dist=pm.HalfNormal.dist(float(edge.get("sd_G", 1.0)), shape=n_dim),
                                compute_corr=False,
                            )
                            if edge.get("track_variance", False) or attr.get("track_values_per_path", False):
                                model.add_coords({dim_name + "_t": coords[dim_name]})
                                L = pm.Deterministic(
                                    f"L:{parent}->{node}",
                                    pm.expand_packed_triangular(n_dim, L, lower=True),
                                    dims=(dim_name, dim_name + "_t"),
                                )
                            else:
                                L = pm.expand_packed_triangular(n_dim, L, lower=True)
    
                            contrib = pt.dot(pt.dot(parent_data, Z), L.T)
    
                        elif kind == "dense":
                            beta = pm.Normal(
                                f"beta:{parent}->{node}",
                                mu=0.0,
                                sigma=float(edge.get("prior_sigma", 1.0)),
                                dims=(p_dim_name, dim_name),
                            )
                            contrib = pt.dot(parent_data, beta)
    
                        else:
                            raise ValueError(f"Unknown edge kind '{kind}' for {parent}->{node}")
    
                        if attr.get("track_values_per_path", False):
                            pm.Deterministic(f"prediction:{parent}->{node}", contrib, dims=("obs_id", dim_name))
                        if edge.get("track_variance", False):
                            pm.Deterministic(f"variance:{parent}->{node}", contrib.var(axis=0))
    
                        mu = mu + contrib
    
                    if attr.get("track_values", False) and len(parents) > 0:
                        pm.Deterministic(f"prediction:{node}", mu, dims=("obs_id", dim_name))
    
                    # ---------- define node variable ----------
                    if node_type == "latent":
                        sigma_lat = float(attr.get("prior_sigma", 1.0))
                        offset = pm.Normal(
                            f"{node}:offset",
                            mu=0.0,
                            sigma=sigma_lat,
                            dims=("obs_id", dim_name),
                        )
                        model_vars[node] = pm.Deterministic(node, mu + offset, dims=("obs_id", dim_name))
    
                    elif node_type == "endo":
                        lik = str(attr.get("likelihood", "normal")).lower()
                        link = str(attr.get("link", "identity")).lower()
                        impute_missing = bool(attr.get("impute_missing", True))
                        Y = attr["data"]
    
                        if lik == "normal":
                            if n_dim == 1:
                                y_obs, obs_idx = self._masked_or_subset(Y[:, 0], impute_missing=impute_missing)
                                sigma = pm.HalfNormal(f"{node}:sigma", float(attr.get("sd_E", 1.0)))
    
                                if obs_idx is None:
                                    pm.Normal(node, mu=mu[:, 0], sigma=sigma, observed=y_obs, dims=("obs_id",))
                                else:
                                    obs_dim = f"{node}_obs_id"
                                    model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
                                    pm.Normal(node, mu=mu[obs_idx, 0], sigma=sigma, observed=y_obs, dims=(obs_dim,))
    
                            else:
                                residual = str(attr.get("residual", "diag")).lower()
                                y_obs, obs_idx = self._masked_or_subset(Y, impute_missing=impute_missing)
    
                                if residual == "lkj":
                                    L_E = pm.LKJCholeskyCov(
                                        f"{node}:cholE",
                                        n=n_dim,
                                        eta=float(attr.get("eta_E", 4.0)),
                                        sd_dist=pm.HalfNormal.dist(float(attr.get("sd_E", 1.0)), shape=n_dim),
                                        compute_corr=False,
                                    )
                                    L_E = pm.expand_packed_triangular(n_dim, L_E, lower=True)
    
                                    if obs_idx is None:
                                        pm.MvNormal(node, mu=mu, chol=L_E, observed=y_obs, dims=("obs_id", dim_name))
                                    else:
                                        obs_dim = f"{node}_obs_id"
                                        model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
                                        pm.MvNormal(node, mu=mu[obs_idx], chol=L_E, observed=y_obs, dims=(obs_dim, dim_name))
    
                                elif residual == "diag":
                                    sigma = pm.HalfNormal(
                                        f"{node}:sigma",
                                        float(attr.get("sd_E", 1.0)),
                                        dims=(dim_name,),
                                    )
                                    if obs_idx is None:
                                        pm.Normal(node, mu=mu, sigma=sigma, observed=y_obs, dims=("obs_id", dim_name))
                                    else:
                                        obs_dim = f"{node}_obs_id"
                                        model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
                                        pm.Normal(node, mu=mu[obs_idx], sigma=sigma, observed=y_obs, dims=(obs_dim, dim_name))
    
                                else:
                                    raise ValueError(f"Unknown residual='{residual}' for normal endo node '{node}'")
    
                        elif lik == "poisson":
                            if link == "log":
                                rate = pm.math.exp(mu)
                            elif link == "identity":
                                rate = pm.math.maximum(mu, 1e-12)
                            else:
                                raise ValueError(f"Unsupported link='{link}' for poisson")
    
                            y = Y
                            if n_dim == 1:
                                y1 = y[:, 0]
                                if np.isnan(y1).any() and not impute_missing:
                                    obs_idx = np.where(~np.isnan(y1))[0]
                                    y_obs = y1[obs_idx].astype(int)
                                    obs_dim = f"{node}_obs_id"
                                    model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
                                    pm.Poisson(node, mu=rate[obs_idx, 0], observed=y_obs, dims=(obs_dim,))
                                else:
                                    y_obs = np.ma.masked_invalid(y1).astype(float) if np.isnan(y1).any() else y1.astype(int)
                                    pm.Poisson(node, mu=rate[:, 0], observed=y_obs, dims=("obs_id",))
                            else:
                                for j in range(n_dim):
                                    yj = y[:, j]
                                    nm = f"{node}:{j}"
                                    if np.isnan(yj).any() and not impute_missing:
                                        obs_idx = np.where(~np.isnan(yj))[0]
                                        y_obs = yj[obs_idx].astype(int)
                                        obs_dim = f"{nm}_obs_id"
                                        model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
                                        pm.Poisson(nm, mu=rate[obs_idx, j], observed=y_obs, dims=(obs_dim,))
                                    else:
                                        y_obs = np.ma.masked_invalid(yj).astype(float) if np.isnan(yj).any() else yj.astype(int)
                                        pm.Poisson(nm, mu=rate[:, j], observed=y_obs, dims=("obs_id",))
    
                        elif lik == "bernoulli":
                            if link == "logit":
                                p = pm.math.sigmoid(mu)
                            elif link == "identity":
                                p = pm.math.clip(mu, 0.0, 1.0)
                            else:
                                raise ValueError(f"Unsupported link='{link}' for bernoulli")
    
                            y = Y
                            if n_dim == 1:
                                y1 = y[:, 0]
                                if np.isnan(y1).any() and not impute_missing:
                                    obs_idx = np.where(~np.isnan(y1))[0]
                                    y_obs = y1[obs_idx].astype(int)
                                    obs_dim = f"{node}_obs_id"
                                    model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
                                    pm.Bernoulli(node, p=p[obs_idx, 0], observed=y_obs, dims=(obs_dim,))
                                else:
                                    y_obs = np.ma.masked_invalid(y1).astype(float) if np.isnan(y1).any() else y1.astype(int)
                                    pm.Bernoulli(node, p=p[:, 0], observed=y_obs, dims=("obs_id",))
                            else:
                                for j in range(n_dim):
                                    yj = y[:, j]
                                    nm = f"{node}:{j}"
                                    if np.isnan(yj).any() and not impute_missing:
                                        obs_idx = np.where(~np.isnan(yj))[0]
                                        y_obs = yj[obs_idx].astype(int)
                                        obs_dim = f"{nm}_obs_id"
                                        model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
                                        pm.Bernoulli(nm, p=p[obs_idx, j], observed=y_obs, dims=(obs_dim,))
                                    else:
                                        y_obs = np.ma.masked_invalid(yj).astype(float) if np.isnan(yj).any() else yj.astype(int)
                                        pm.Bernoulli(nm, p=p[:, j], observed=y_obs, dims=("obs_id",))
    
                        else:
                            raise ValueError(f"Unsupported likelihood='{lik}' for node '{node}'")
    
                        model_vars[node] = Y
    
                    else:
                        raise ValueError(f"Unknown node type: {node_type}")
    
            return model


    # -----------------------------
    # qc / debugging
    # -----------------------------
    def graph_qc(self):
        print("--- Causal Graph QC ---")
        print(f"Nodes: {self.G.number_of_nodes()} | Edges: {self.G.number_of_edges()}")
        try:
            cycles = list(nx.simple_cycles(self.G))
            if cycles: print(f"⚠️ cycles detected: {cycles}")
            else: print("✅ DAG: acyclic")
        except Exception:  print("cycle check skipped (graph may be large)")
        isolates = list(nx.isolates(self.G))
        if isolates: print(f"⚠️ isolates: {isolates}")
        print("\nNode summary:")
        edges = nx.to_pandas_edgelist(self.G)
        nodes = pd.DataFrame.from_dict(dict(self.G.nodes(data=True)), orient="index")
        nodes.index.name = "node"
        nodes["n_parents"]  = [self.G.in_degree(n)  for n in nodes.index]
        nodes["n_children"] = [self.G.out_degree(n) for n in nodes.index]
        def node_missingness(a):
            x = a.get("data", None)
            if x is None:return np.nan
            try:
                x = np.asarray(x)
                return float(np.isnan(x).mean()) * 100.0
            except Exception:  return np.nan
        
        nodes["missing_pct"] = [node_missingness(dict(self.G.nodes[n])) for n in nodes.index]
        if "shape" in nodes.columns:
            nodes["shape_str"] = nodes["shape"].apply( lambda s: "x".join(map(str, s)) if isinstance(s, (tuple, list)) else str(s))
        
        nodes_md = (nodes.reset_index()
                         .sort_values(["type", "node"])
                         .loc[:, [c for c in ["node","type","shape_str","n_parents","n_children","missing_pct",
                                              "likelihood","link","residual","fixed_node"] if c in nodes.columns]])
        edges_md = edges.sort_values(["source","target"]).reset_index(drop=True)
        print("## Nodes\n")
        print(nodes_md.to_markdown(index=False))
        print("\n\n## Edges\n")
        print(edges_md.to_markdown(index=False))

    def plot_graph(self, *, show_edge_param_labels: bool = True):
        P = nx.DiGraph()
        for n, attr in self.G.nodes(data=True):
            ntype = str(attr.get("type", "unknown"))
            shape = attr.get("shape", None)
            shape_str = "x".join(map(str, shape)) if shape is not None else "?"
            likelihood = str(attr.get("likelihood", "N/A"))
            P.add_node(n, type=ntype, shape_str=shape_str, likelihood=likelihood)
        for u, v, attr in self.G.edges(data=True):
            raw_sigma = attr.get("prior_sigma", 1.0)
            if attr['adjust_variance_by_shape'] and attr['kind']!='grm': raw_sigma *= np.sqrt(self.G.nodes[u]['shape'][1])
            conf = min(4, max(raw_sigma, .1))
            P.add_edge( u, v, confidence=conf,variance = raw_sigma**2,
                n_params=int(attr.get("n_parameters", 0)),
                kind=str(attr.get("kind", "dense")))
        for n in P.nodes():
            out_degree = P.out_degree(n)
            in_degree = P.in_degree(n)
            subset = 0 if in_degree == 0 else (2 if out_degree == 0 else 1)
            P.nodes[n]["subset"] = int(subset)
        pos = nx.multipartite_layout(P, subset_key="subset", scale=2)
        color_map  = {"exo": "lightsteelblue", "grm": "salmon", "latent": "palegoldenrod", "endo": "darkseagreen"}
        marker_map = {"exo": "circle",  "grm": "hex", "latent": "square", "endo": "square"}
        edge_color_map = {"dense": "steelblue",  "grm":   "firebrick"}
        for n in P.nodes():
            t = P.nodes[n]["type"]
            P.nodes[n]["color"]  = str(color_map.get(t, "white"))
            P.nodes[n]["marker"] = str(marker_map.get(t, "circle"))
    
        # 1) Draw edges/nodes separately and style them directly
        edge_tooltips = [("Edge", "@start -> @end"), ("Kind", "@kind"), ("Params", "@n_params"), ("Confidence", "@confidence")]
        node_tooltips = [("Node", "@index"), ("Type", "@type"), ("Shape", "@shape_str"), ("Likelihood", "@likelihood")]
    
        edge_plot = hvnx.draw_networkx_edges( P, pos,
            edge_color="kind", alpha=0.6,  arrow_style="-|>", arrow_size=5,
            edge_width=4+ hv.dim("confidence").norm() * 15, cmap = edge_color_map,
        ).opts( tools=["hover"],   hover_tooltips=edge_tooltips )
        node_plot = hvnx.draw_networkx_nodes(P, pos,
            node_color="type",node_marker="marker", alpha = 1,
            node_size=5200,cmap=color_map ).opts( tools=["hover"],  hover_tooltips=node_tooltips)
        label_df = pd.DataFrame( {"x": [pos[n][0] for n in P.nodes()],
                                  "y": [pos[n][1] for n in P.nodes()],
                                  "text": [f"{n}\n{P.nodes[n]['shape_str']}" for n in P.nodes()]} )
        labels = hv.Labels(label_df, kdims=["x", "y"], vdims=["text"]).opts( text_color="black",text_font_size="8pt",  yoffset=0.0,
                                                                            text_align="center", text_baseline="middle")
    
        # 3) Optional edge midpoint labels
        edge_param_labels = None
        if show_edge_param_labels and P.number_of_edges() > 0:
            rows = []
            for u, v, ed in P.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                rows.append({"x": float((x0 + x1) / 2.0), "y": float((y0 + y1) / 2.0), "text": f"n_params: {int(ed['n_params'])}\nvariance:  {round(ed['variance'],3)}"})
            edge_param_labels = hv.Labels(pd.DataFrame(rows), kdims=["x", "y"], vdims=["text"])\
                                  .opts(text_font_size="9pt", text_color="black", text_align="left", text_baseline="middle" )
        graph = edge_plot * node_plot * labels
        if edge_param_labels is not None: graph = graph * edge_param_labels
        return graph.opts(
            title="GraphView", width=900, height=650, xaxis=None, yaxis=None,
            tools=["tap", "box_select", 'hover'])


def QC_FIGS(imputed_df_fit, imputed_df_g, imputed_df_g_e, df_full, test_rows, test_elems, name):
    tabs = pn.Tabs()
    meta = dict(frame_width = 270, frame_height = 270,alpha = .7, xlabel = '',ylabel = '',  line_width = 1, line_color = 'black', hover = False)
    for df_imputed,df_f,c,names in zip([imputed_df_fit, imputed_df_g, imputed_df_g_e], 
                                       [df_full,test_rows,test_elems],
                                       ['steelblue', 'firebrick', 'gray'], 
                                       ['seen samples', 'unseen samples', 'unseen measurement of seen samples'] ):
        dfppc = pd.concat([df_f,  df_imputed.rename(lambda x: x+'_ppc', axis = 1)], axis = 1)\
                  .rename(lambda x: x.replace('regressedlr_', ''), axis = 1).rename(str)
        ppc_out_of_sample = reduce(lambda x, y : x+y, [dfppc.hvplot.scatter(x = i, y = f'{i}_ppc',color = c,
                                                        title = "r:"+"r2:"+ str(round(dfppc[[i, f'{i}_ppc']].corr().iloc[0, 1]**2, 2))+ ' ' +i, **meta )*\
                                   hv.Slope(1,0).opts(color = 'red') for i in imputed_df_fit.columns.str.replace('regressedlr_', '')]\
                                  ).cols(4)
        tabs.append((f'{name} {names}', ppc_out_of_sample))
    return tabs

def chol2cov(L, return_correlation = False):
    import xarray as xr
    variance = (L**2).sum(dim="traits_dim_t")
    stds = np.sqrt(variance)
    L_transposed = L.rename({"traits_dim": "traits_dim_2"})
    covariance = xr.dot(L, L_transposed, dims="traits_dim_t")
    stds_2 = stds.rename({"traits_dim": "traits_dim_2"})
    if not return_correlation: return covariance
    return covariance / (stds * stds_2)

def chol2covMAP(trace, name, columns, return_correlation = False):
    mapl = trace[name]
    mapcov = mapl.dot(mapl.T) 
    if not return_correlation: return pd.DataFrame(mapcov, columns = columns, index = columns)
    mapsig = np.sqrt(np.diag(mapcov))
    return pd.DataFrame(mapcov/np.outer(mapsig, mapsig), columns = columns, index = columns)

def create_dual_masking_scheme(df: pd.DataFrame, row_pct=0.10, elem_pct=0.20, seed=42, holdout_row_idx: [np.array,None] = None):
    """
    Creates a training dataset with two types of artificial missingness:
    1. Row Mask (Hold-out Set): 'row_pct' of rows are fully masked (all traits hidden).
    2. Element Mask (Imputation Set): For the remaining rows, 'elem_pct' of available data 
       is masked per column.
    """
    rng = np.random.default_rng(seed)
    n, t = df.shape
    valid_data = ~df.isna().to_numpy()
    mask_rows, mask_elems = np.zeros((n, t), dtype=bool), np.zeros((n, t), dtype=bool)    
    row_indices = np.arange(n)
    if holdout_row_idx is None: holdout_row_idx = rng.choice(row_indices, round(n*row_pct), replace=False)
    mask_rows[holdout_row_idx, :] = True
    training_row_bool = ~np.isin(row_indices, holdout_row_idx)
    training_row_idx = row_indices[training_row_bool]
    for col_idx in range(t):
        candidates = np.intersect1d(training_row_idx, np.where(valid_data[:, col_idx])[0] )
        if len(candidates) == 0: continue
        n_to_drop = int(len(candidates) * elem_pct)
        if n_to_drop == 0: continue
        drop_idx = rng.choice(candidates, size=n_to_drop, replace=False)
        mask_elems[drop_idx, col_idx] = True
    combined_mask = mask_rows | mask_elems    
    return mask_rows, mask_elems, combined_mask


## examples
## simple greml 
# mgreml = genomic_sem.CausalGraph(df_pheno)
# mgreml.add_grm('FullGRM', fgrm,top_eigen2fixed=None, n_components=7600)
# mgreml.add_trait('traits', data_cols=df_pheno.columns, track_values=True, track_values_per_path=True) #  residual = 'lkj', var_E = .55, eta_E=6
# mgreml.add_edge('FullGRM', 'traits',eta = 1.2, track_variance=True,prior_variance =.55, adjust_variance_by_shape=False)
# mgreml.graph_qc()
# mgreml.plot_graph()
# %time model = mgreml.build_model()
# import pymc as pm
# with model:
#     map_trace = pm.find_MAP(maxeval = 20000)
#     advi_EXP = pd.DataFrame(map_trace['prediction:traits'], index = df_train.index, columns = traits)
#     trace_g =pd.DataFrame( map_trace['prediction:FullGRM->traits'], index = df_train.index, columns = traits)
#     ppc = pm.sample_posterior_predictive([map_trace], var_names=["traits"])
#     ppc_df = ppc['posterior_predictive']['traits'].isel(draw=0, chain = 0).to_pandas()
#     display(genomic_sem.QC_FIGS(advi_EXP, advi_EXP, advi_EXP, df_pheno, test_rows, test_elems, 'mgreml'))
#     mapcorr = chol2covMAP(map_trace, 'L:FullGRM->traits', df_train.columns, return_correlation=True)






    
