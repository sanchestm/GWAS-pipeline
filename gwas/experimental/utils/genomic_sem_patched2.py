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
from functools import reduce
import copy

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
        store_full: bool = False,
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
            umap_labels = self._pc_labels(int(umap_n_components), prefix="UMAP") if add_UMAP else []
            self.G.add_node(f"{name}_fixed", type="exo",
                            data=fixed_data.astype(float), shape=fixed_data.shape,
                            dim_name=f"{name}_fixed_dim", 
                            coords=self._as_str_coords(self._pc_labels(cutoff, prefix="Pc") + umap_labels),
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
        K_full = None
        if bool(store_full):
            # Keep a symmetric copy for specialized exact LMM compilers.
            # NOTE: For large N this is memory heavy (N^2). Default is False.
            K_full = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)

        self.G.add_node(name, type="grm", data=X_full.astype(float),
                        shape=X_full.shape, dim_name=f"{name}_pc", 
                        coords=self._as_str_coords(self._pc_labels(X_full.shape[1], prefix="Pc")),
                        track_values=False, track_values_per_path=False,
                        fixed_node=fixed_node_name,
                        K_full=K_full)

    def add_latent(
        self,
        name,
        *,
        n_dim=1,
        prior_sigma=1.0,
        track_values=True,
        track_values_per_path=False,
        variance_budget=True,
        budget_w_eps=1e-6,
        budget_concentration=20,
        is_factor: bool = False,
    ):
        n_dim = int(n_dim)
        shape = (len(self.obs_ids), n_dim)
        self.G.add_node(
            name,
            type="latent",
            data=None,
            shape=shape,
            dim_name=f"{name}_dim",
            coords=self._as_str_coords(self._latent_labels(name, n_dim)),
            prior_sigma=float(prior_sigma),
            track_values=track_values,
            track_values_per_path=track_values_per_path,
            variance_budget=variance_budget,
            budget_w_eps=budget_w_eps,
            budget_concentration=budget_concentration,
            is_factor=bool(is_factor),
        )

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
        is_series = isinstance(data, pd.Series)
        is_df = isinstance(data, pd.DataFrame)
        arr = data.to_numpy(dtype=float) if (is_series or is_df) else np.asarray(data, dtype=float)
        arr = self._ensure_2d(arr)

        if is_series: coord_names = [data.name if data.name is not None else f"{name}1"]
        elif is_df: coord_names = list(data.columns)
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
            coords=self._as_str_coords(coord_names),
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
        kind="auto",           # "auto" | "dense" | "grm" | "factor"
        prior_variance=1.0,    # used for dense/factor regression sigma (on loadings)
        eta=1.0,               # LKJ eta for GRM edges
        grm_trait_cov: str = "lkj",   # for GRM edges: "lkj" | "diag" | "scalar"
        factor_identifiable: bool = True,  # enforce lower-triangular + positive diagonal loadings
        loading_diag_scale: float | None = None,  # scale for positive diagonal loadings (defaults to sqrt(prior_variance))
        track_variance=False,
        adjust_variance_by_shape = True):
        if source not in self.G.nodes or target not in self.G.nodes: raise ValueError(f"Missing nodes for edge {source} -> {target}")

        src,tgt = self.G.nodes[source], self.G.nodes[target]
        if kind == "auto":
            if src["type"] == "grm":
                kind = "grm"
            elif src["type"] == "latent" and bool(src.get("is_factor", False)) and tgt["type"] == "endo":
                kind = "factor"
            else:
                kind = "dense"
        p_dim = int(src['shape'][1])
        n_dim = int(tgt['shape'][1])

        kind = str(kind).lower()
        grm_trait_cov = str(grm_trait_cov).lower()

        if adjust_variance_by_shape and kind not in ("grm",):
            prior_variance = float(prior_variance) / float(max(src['shape'][1], 1))
        prior_sigma = float(np.sqrt(max(float(prior_variance), 1e-12)))

        if kind == "factor":
            if src["type"] != "latent":
                raise ValueError(f"factor edges require latent source; got {source} type={src['type']}")
            if int(n_dim) < int(p_dim):
                raise ValueError(
                    f"factor edges require target dim >= source dim (traits >= factors). "
                    f"Got {source} dim={p_dim} -> {target} dim={n_dim}."
                )
            # free loadings in a (p x k) lower-triangular block for first k rows + full rows thereafter
            k = int(p_dim)
            p = int(n_dim)
            n_parameters = int(p * k - (k * (k - 1)) // 2)
            self.G.add_edge(
                source,
                target,
                kind="factor",
                prior_sigma=prior_sigma,
                loading_diag_scale=float(loading_diag_scale) if loading_diag_scale is not None else prior_sigma,
                factor_identifiable=bool(factor_identifiable),
                n_parameters=n_parameters,
                adjust_variance_by_shape=adjust_variance_by_shape,
                track_variance=bool(track_variance),
            )
            return

        # dense / grm
        n_parameters = int(p_dim * n_dim)
        if kind == "grm":
            if grm_trait_cov == "lkj":
                n_parameters += int(n_dim * (n_dim + 1) // 2)
            elif grm_trait_cov == "diag":
                n_parameters += int(n_dim)
            elif grm_trait_cov == "scalar":
                n_parameters += 1
            else:
                raise ValueError(f"Unknown grm_trait_cov='{grm_trait_cov}'. Use 'lkj'|'diag'|'scalar'.")

        self.G.add_edge(
            source,
            target,
            kind=kind,
            prior_sigma=prior_sigma,
            n_parameters=n_parameters,
            eta=float(eta),
            sd_G=prior_sigma,
            grm_trait_cov=grm_trait_cov,
            adjust_variance_by_shape=adjust_variance_by_shape,
            track_variance=bool(track_variance),
        )
    # -----------------------------
    # build model
    # -----------------------------
    def build_model(
        self,
        *,
        sd_prior: str = "lognormal",
        sd_prior_sigma: float = 0.5,
        poisson_clip: float = 20.0,
        discrete_imputation: bool = False,
        diag_missing_per_trait: bool = True,
    ):
        """Compile the DAG into a single PyMC model.

        Stability / identifiability features:
          - factor edges (kind='factor') implement lower-triangular loadings with positive diagonal
          - GRM edges support trait covariance structures: 'lkj'|'diag'|'scalar' (edge attr grm_trait_cov)
          - no forced float32 in the graph (uses PyTensor floatX)
          - Poisson(log) uses exp(clip(mu))
          - Bernoulli(logit) uses logit_p
          - correct dims for subsetting observed rows
        """
        order = list(nx.topological_sort(self.G))
        obs_ids_np = np.asarray(self.obs_ids)

        # ---- coords ----
        coords = {"obs_id": self._as_str_coords(self.obs_ids)}
        for n, a in self.G.nodes(data=True):
            coords[a["dim_name"]] = self._as_str_coords(a["coords"])

        def _sd_dist(scale: float, *, shape=None):
            scale = max(float(scale), 1e-8)
            prior = str(sd_prior).lower()
            if prior == "lognormal":
                return pm.LogNormal.dist(mu=np.log(scale), sigma=float(sd_prior_sigma), shape=shape)
            if prior == "halfnormal":
                return pm.HalfNormal.dist(sigma=scale, shape=shape)
            if prior == "exponential":
                return pm.Exponential.dist(lam=1.0 / scale, shape=shape)
            raise ValueError(f"Unknown sd_prior='{sd_prior}'. Use 'lognormal'|'halfnormal'|'exponential'.")

        def _subset_coords(model: pm.Model, name: str, obs_idx: np.ndarray):
            obs_dim = f"{name}_obs_id"
            model.add_coords({obs_dim: self._as_str_coords(obs_ids_np[obs_idx])})
            return obs_dim

        with pm.Model(coords=coords) as model:
            model_vars: dict[str, pt.TensorVariable] = {}

            # 1) pm.Data for exo & grm nodes
            for node in order:
                attr = self.G.nodes[node]
                if attr["type"] in ("exo", "grm"):
                    model_vars[node] = pm.Data(node, attr["data"], dims=("obs_id", attr["dim_name"]))

            # 2) latents & likelihoods in topo order
            for node in order:
                attr = self.G.nodes[node]
                node_type = attr["type"]
                dim_name = attr["dim_name"]
                n_dim = int(attr["shape"][1])

                if node_type in ("exo", "grm"):
                    continue

                parents = list(self.G.predecessors(node))
                mu = pt.zeros((len(self.obs_ids), n_dim))

                # ---------- parent contributions ----------
                for parent in parents:
                    p_attr = self.G.nodes[parent]
                    edge = self.G.edges[parent, node]
                    kind = str(edge.get("kind", "dense")).lower()
                    parent_data = model_vars[parent]
                    p_dim = int(p_attr["shape"][1])
                    p_dim_name = p_attr["dim_name"]

                    if kind == "grm":
                        Z = pm.Normal(
                            f"Z_{parent}->{node}",
                            mu=0.0,
                            sigma=1.0,
                            dims=(p_dim_name, dim_name),
                        )

                        trait_cov = str(edge.get("grm_trait_cov", "lkj")).lower()
                        sd_G = float(edge.get("sd_G", edge.get("prior_sigma", 1.0)))

                        if trait_cov == "lkj":
                            chol_packed = pm.LKJCholeskyCov(
                                f"cholG:{parent}->{node}",
                                n=n_dim,
                                eta=float(edge.get("eta", 1.0)),
                                sd_dist=_sd_dist(sd_G, shape=n_dim),
                                compute_corr=False,
                            )
                            L = pm.expand_packed_triangular(n_dim, chol_packed, lower=True)

                        elif trait_cov == "diag":
                            sd_vec = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sd_G, 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(dim_name,),
                            )
                            L = pt.diag(sd_vec)

                        elif trait_cov == "scalar":
                            sd_s = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sd_G, 1e-8)),
                                sigma=float(sd_prior_sigma),
                            )
                            L = sd_s * pt.eye(n_dim)

                        else:
                            raise ValueError(
                                f"Unknown grm_trait_cov='{trait_cov}' for edge {parent}->{node}. "
                                "Use 'lkj'|'diag'|'scalar'."
                            )

                        if edge.get("track_variance", False) or attr.get("track_values_per_path", False):
                            model.add_coords({dim_name + "_t": coords[dim_name]})
                            L_det = pm.Deterministic(
                                f"L:{parent}->{node}",
                                L,
                                dims=(dim_name, dim_name + "_t"),
                            )
                            contrib = pt.dot(pt.dot(parent_data, Z), L_det.T)
                        else:
                            contrib = pt.dot(pt.dot(parent_data, Z), L.T)

                    elif kind == "dense":
                        beta = pm.Normal(
                            f"beta:{parent}->{node}",
                            mu=0.0,
                            sigma=float(edge.get("prior_sigma", 1.0)),
                            dims=(p_dim_name, dim_name),
                        )
                        contrib = pt.dot(parent_data, beta)

                    elif kind == "factor":
                        # Identifiable loadings: Lambda (p x k) lower-triangular (first k rows) + positive diagonal.
                        if int(n_dim) < int(p_dim):
                            raise ValueError(
                                f"factor edge requires target dim >= source dim (traits >= factors). "
                                f"Got {parent} dim={p_dim} -> {node} dim={n_dim}."
                            )
                        k = int(p_dim)
                        p = int(n_dim)

                        # mask is 1 for free entries, 0 for constrained zeros (upper triangle of first k rows)
                        mask = np.ones((p, k), dtype=float)
                        for i in range(min(p, k)):
                            if i + 1 < k:
                                mask[i, (i + 1) :] = 0.0
                        mask_t = pt.constant(mask)

                        loading_sigma = float(edge.get("prior_sigma", 1.0))
                        Lambda_raw = pm.Normal(
                            f"Lambda_raw:{parent}->{node}",
                            mu=0.0,
                            sigma=loading_sigma,
                            dims=(dim_name, p_dim_name),  # (traits_dim, factors_dim)
                        )
                        Lambda = Lambda_raw * mask_t

                        if bool(edge.get("factor_identifiable", True)):
                            diag_scale = float(edge.get("loading_diag_scale", loading_sigma))
                            diag = pm.LogNormal(
                                f"Lambda_diag:{parent}->{node}",
                                mu=np.log(max(diag_scale, 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(p_dim_name,),
                            )
                            idx = np.arange(k)
                            Lambda = pt.set_subtensor(Lambda[idx, idx], diag)

                        if edge.get("track_variance", False) or attr.get("track_values_per_path", False):
                            Lambda_det = pm.Deterministic(
                                f"Lambda:{parent}->{node}",
                                Lambda,
                                dims=(dim_name, p_dim_name),
                            )
                            contrib = pt.dot(parent_data, Lambda_det.T)
                        else:
                            contrib = pt.dot(parent_data, Lambda.T)

                    else:
                        raise ValueError(f"Unknown edge kind '{kind}' for {parent}->{node}")

                    if attr.get("track_values_per_path", False):
                        pm.Deterministic(f"prediction:{parent}->{node}", contrib, dims=("obs_id", dim_name))
                    if edge.get("track_variance", False):
                        pm.Deterministic(f"variance:{parent}->{node}", contrib.var(axis=0))

                    mu = mu + contrib

                if attr.get("track_values", False) and len(parents) > 0:
                    pm.Deterministic(f"prediction:{node}", mu, dims=("obs_id", dim_name))

                # ---------- define node ----------
                if node_type == "latent":
                    sigma_lat = float(attr.get("prior_sigma", 1.0))
                    offset = pm.Normal(
                        f"{node}:offset",
                        mu=0.0,
                        sigma=sigma_lat,
                        dims=("obs_id", dim_name),
                    )
                    model_vars[node] = pm.Deterministic(node, mu + offset, dims=("obs_id", dim_name))
                    continue

                if node_type != "endo":
                    raise ValueError(f"Unknown node type: {node_type}")

                lik = str(attr.get("likelihood", "normal")).lower()
                link = str(attr.get("link", "identity")).lower()
                impute_missing = bool(attr.get("impute_missing", True))
                Y = np.asarray(attr["data"])

                # ---- NORMAL ----
                if lik == "normal":
                    residual = str(attr.get("residual", "diag")).lower()
                    sd_E = float(attr.get("sd_E", 1.0))

                    if n_dim == 1:
                        y_obs, obs_idx = self._masked_or_subset(Y[:, 0], impute_missing=impute_missing)
                        sigma = pm.LogNormal(f"{node}:sigma", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma))
                        if obs_idx is None:
                            rv = pm.Normal(node, mu=mu[:, 0], sigma=sigma, observed=y_obs, dims=("obs_id",))
                        else:
                            obs_dim = _subset_coords(model, node, obs_idx)
                            rv = pm.Normal(node, mu=mu[obs_idx, 0], sigma=sigma, observed=y_obs, dims=(obs_dim,))
                        model_vars[node] = rv
                        continue

                    if residual == "lkj":
                        y_obs, obs_idx = self._masked_or_subset(Y, impute_missing=impute_missing)
                        chol_packed = pm.LKJCholeskyCov(
                            f"{node}:cholE",
                            n=n_dim,
                            eta=float(attr.get("eta_E", 4.0)),
                            sd_dist=_sd_dist(sd_E, shape=n_dim),
                            compute_corr=False,
                        )
                        chol = pm.expand_packed_triangular(n_dim, chol_packed, lower=True)
                        if obs_idx is None:
                            rv = pm.MvNormal(node, mu=mu, chol=chol, observed=y_obs, dims=("obs_id", dim_name))
                        else:
                            obs_dim = _subset_coords(model, node, obs_idx)
                            rv = pm.MvNormal(node, mu=mu[obs_idx], chol=chol, observed=y_obs, dims=(obs_dim, dim_name))
                        model_vars[node] = rv
                        continue

                    if residual == "diag":
                        sigma = pm.LogNormal(
                            f"{node}:sigma",
                            mu=np.log(max(sd_E, 1e-8)),
                            sigma=float(sd_prior_sigma),
                            dims=(dim_name,),
                        )

                        # element-wise missingness: fit each trait separately if we are NOT imputing
                        if (not impute_missing) and np.isnan(Y).any() and diag_missing_per_trait:
                            for j in range(n_dim):
                                yj = Y[:, j]
                                nm = f"{node}:{j}"
                                y_obs, obs_idx = self._masked_or_subset(yj, impute_missing=False)
                                if obs_idx is None:
                                    pm.Normal(nm, mu=mu[:, j], sigma=sigma[j], observed=y_obs, dims=("obs_id",))
                                else:
                                    obs_dim = _subset_coords(model, nm, obs_idx)
                                    pm.Normal(nm, mu=mu[obs_idx, j], sigma=sigma[j], observed=y_obs, dims=(obs_dim,))
                            model_vars[node] = mu
                            continue

                        y_obs, obs_idx = self._masked_or_subset(Y, impute_missing=impute_missing)
                        if obs_idx is None:
                            rv = pm.Normal(node, mu=mu, sigma=sigma, observed=y_obs, dims=("obs_id", dim_name))
                        else:
                            obs_dim = _subset_coords(model, node, obs_idx)
                            rv = pm.Normal(node, mu=mu[obs_idx], sigma=sigma, observed=y_obs, dims=(obs_dim, dim_name))
                        model_vars[node] = rv
                        continue

                    raise ValueError(f"Unknown residual='{residual}' for normal endo node '{node}'")

                # ---- POISSON ----
                if lik == "poisson":
                    if link == "log":
                        rate = pm.math.exp(pm.math.clip(mu, -float(poisson_clip), float(poisson_clip)))
                    elif link == "identity":
                        rate = pm.math.maximum(mu, 1e-12)
                    else:
                        raise ValueError(f"Unsupported link='{link}' for poisson")

                    y = np.asarray(Y)
                    has_nan = np.isnan(y).any()
                    do_impute = bool(impute_missing) and bool(discrete_imputation)

                    if (not has_nan) or do_impute:
                        y_obs = np.ma.masked_invalid(y).astype(float) if (has_nan and do_impute) else y.astype(int)
                        if n_dim == 1:
                            rv = pm.Poisson(node, mu=rate[:, 0], observed=y_obs[:, 0], dims=("obs_id",))
                        else:
                            rv = pm.Poisson(node, mu=rate, observed=y_obs, dims=("obs_id", dim_name))
                        model_vars[node] = rv
                        continue

                    # subset observed only (no discrete imputation by default)
                    if n_dim == 1:
                        y1 = y[:, 0]
                        obs_idx = np.where(~np.isnan(y1))[0]
                        y_obs = y1[obs_idx].astype(int)
                        obs_dim = _subset_coords(model, node, obs_idx)
                        rv = pm.Poisson(node, mu=rate[obs_idx, 0], observed=y_obs, dims=(obs_dim,))
                        model_vars[node] = rv
                        continue

                    for j in range(n_dim):
                        yj = y[:, j]
                        nm = f"{node}:{j}"
                        obs_idx = np.where(~np.isnan(yj))[0]
                        y_obs = yj[obs_idx].astype(int)
                        obs_dim = _subset_coords(model, nm, obs_idx)
                        pm.Poisson(nm, mu=rate[obs_idx, j], observed=y_obs, dims=(obs_dim,))
                    model_vars[node] = rate
                    continue

                # ---- BERNOULLI ----
                if lik == "bernoulli":
                    y = np.asarray(Y)
                    has_nan = np.isnan(y).any()
                    do_impute = bool(impute_missing) and bool(discrete_imputation)

                    if link == "logit":
                        use_logit = True
                    elif link == "identity":
                        use_logit = False
                        p = pm.math.clip(mu, 0.0, 1.0)
                    else:
                        raise ValueError(f"Unsupported link='{link}' for bernoulli")

                    if (not has_nan) or do_impute:
                        y_obs = np.ma.masked_invalid(y).astype(float) if (has_nan and do_impute) else y.astype(int)
                        if n_dim == 1:
                            if use_logit:
                                rv = pm.Bernoulli(node, logit_p=mu[:, 0], observed=y_obs[:, 0], dims=("obs_id",))
                            else:
                                rv = pm.Bernoulli(node, p=p[:, 0], observed=y_obs[:, 0], dims=("obs_id",))
                        else:
                            if use_logit:
                                rv = pm.Bernoulli(node, logit_p=mu, observed=y_obs, dims=("obs_id", dim_name))
                            else:
                                rv = pm.Bernoulli(node, p=p, observed=y_obs, dims=("obs_id", dim_name))
                        model_vars[node] = rv
                        continue

                    # subset observed only (no discrete imputation by default)
                    if n_dim == 1:
                        y1 = y[:, 0]
                        obs_idx = np.where(~np.isnan(y1))[0]
                        y_obs = y1[obs_idx].astype(int)
                        obs_dim = _subset_coords(model, node, obs_idx)
                        if use_logit:
                            rv = pm.Bernoulli(node, logit_p=mu[obs_idx, 0], observed=y_obs, dims=(obs_dim,))
                        else:
                            rv = pm.Bernoulli(node, p=p[obs_idx, 0], observed=y_obs, dims=(obs_dim,))
                        model_vars[node] = rv
                        continue

                    for j in range(n_dim):
                        yj = y[:, j]
                        nm = f"{node}:{j}"
                        obs_idx = np.where(~np.isnan(yj))[0]
                        y_obs = yj[obs_idx].astype(int)
                        obs_dim = _subset_coords(model, nm, obs_idx)
                        if use_logit:
                            pm.Bernoulli(nm, logit_p=mu[obs_idx, j], observed=y_obs, dims=(obs_dim,))
                        else:
                            pm.Bernoulli(nm, p=p[obs_idx, j], observed=y_obs, dims=(obs_dim,))
                    model_vars[node] = mu
                    continue

                raise ValueError(f"Unsupported likelihood='{lik}' for node '{node}'")

            return model

    # -----------------------------
    # specialized compilers (fast paths)
    # -----------------------------
    def _simple_grm_to_traits_spec(self, *, traits_node: str | None = None):
        """Detect the pattern: (exo/dense)* + (single GRM/grm edge) -> single normal endo.

        Returns
        -------
        dict or None
            {
              'traits': <endo node>,
              'grm': <grm node>,
              'exo_parents': [list of exo parents],
              'edge_grm': edge attrs dict,
            }
        """
        endos = [n for n, a in self.G.nodes(data=True) if a.get("type") == "endo"]
        if traits_node is None:
            if len(endos) != 1:
                return None
            traits_node = endos[0]
        if traits_node not in self.G.nodes:
            return None
        if self.G.nodes[traits_node].get("type") != "endo":
            return None

        # no latents anywhere
        if any(a.get("type") == "latent" for _, a in self.G.nodes(data=True)):
            return None

        # single endo only
        if len(endos) != 1:
            return None

        # traits must be a leaf
        if len(list(self.G.successors(traits_node))) != 0:
            return None

        parents = list(self.G.predecessors(traits_node))
        grm_parents = [p for p in parents if self.G.nodes[p].get("type") == "grm"]
        if len(grm_parents) != 1:
            return None
        grm_node = grm_parents[0]
        e_grm = dict(self.G.edges[grm_node, traits_node])
        if str(e_grm.get("kind", "dense")).lower() != "grm":
            return None

        exo_parents = []
        for p in parents:
            if p == grm_node:
                continue
            p_type = self.G.nodes[p].get("type")
            if p_type != "exo":
                return None
            e = self.G.edges[p, traits_node]
            if str(e.get("kind", "dense")).lower() != "dense":
                return None
            exo_parents.append(p)

        # no other edges anywhere besides parents -> traits
        ok_edges = set([(p, traits_node) for p in parents])
        for u, v in self.G.edges:
            if (u, v) not in ok_edges:
                return None

        # only exo/grm/endo nodes
        for n, a in self.G.nodes(data=True):
            if a.get("type") not in ("exo", "grm", "endo"):
                return None

        return {
            "traits": traits_node,
            "grm": grm_node,
            "exo_parents": exo_parents,
            "edge_grm": e_grm,
        }

    @staticmethod
    def _chol_psd(K: np.ndarray, *, jitter: float = 1e-6):
        K = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
        n = K.shape[0]
        try:
            return np.linalg.cholesky(K + float(jitter) * np.eye(n))
        except np.linalg.LinAlgError:
            # robust fallback: clip eigenvalues then cholesky
            evals, evecs = np.linalg.eigh(K)
            evals = np.clip(evals, 0.0, np.inf) + float(jitter)
            K_pd = (evecs * evals[None, :]) @ evecs.T
            return np.linalg.cholesky(K_pd)

    def build_model_grm_traits_kron(
        self,
        *,
        traits_node: str | None = None,
        jitter: float = 1e-6,
        sd_prior_sigma: float = 0.5,
        eta_G: float = 2.0,
        sd_G: float | None = None,
        sd_E: float = 1.0,
        trait_cov: str = "lkj",     # genetic trait cov: 'lkj'|'diag'|'scalar'
        require_store_full: bool = True,
        impute_missing: bool = True,
    ):
        """Specialized compiler for the simple pattern: GRM -> traits (+ dense exo).

        This uses `pm.KroneckerNormal` for the likelihood:

          vec(Y) ~ N(vec(mu),  K ⊗ Σ_G  + σ_E^2 I)

        Pros
        ----
        * Efficient logp for Kronecker covariance + iid noise.
        * Great as a fast baseline / debugging target.

        Cons
        ----
        * Residual noise is iid across *all* entries (individuals x traits). If you need Σ_E != σ^2 I,
          use `build_model_grm_traits_matrixnormal` instead.

        Requirements
        ------------
        * `add_grm(..., store_full=True)` so the GRM node carries `K_full`.
        """
        spec = self._simple_grm_to_traits_spec(traits_node=traits_node)
        if spec is None:
            raise ValueError("Graph is not a simple (exo,dense)* + (single GRM,grm) -> single normal traits pattern.")

        t = spec["traits"]
        g = spec["grm"]
        exo_parents = spec["exo_parents"]
        edge_grm = spec["edge_grm"]

        attr_t = self.G.nodes[t]
        if str(attr_t.get("likelihood", "normal")).lower() != "normal":
            raise ValueError("build_model_grm_traits_kron only supports normal likelihood.")

        Y = np.asarray(attr_t["data"], dtype=float)
        Y = self._ensure_2d(Y)
        N, T = Y.shape

        if np.isnan(Y).any() and (not bool(impute_missing)):
            raise ValueError("KroneckerNormal likelihood requires elementwise missingness to be imputed (use impute_missing=True) or removed.")

        # masked arrays are the only practical option here (element-wise subsetting breaks Kronecker structure)
        Y_obs = np.ma.masked_invalid(Y) if (np.isnan(Y).any() and bool(impute_missing)) else Y
        y_vec = np.asarray(Y_obs).reshape(-1)
        if isinstance(Y_obs, np.ma.MaskedArray):
            y_vec = np.ma.MaskedArray(y_vec, mask=np.asarray(Y_obs.mask).reshape(-1))

        # GRM full covariance
        K = self.G.nodes[g].get("K_full")
        if K is None and bool(require_store_full):
            raise ValueError(
                f"GRM node '{g}' does not have K_full. Re-add it with add_grm(..., store_full=True)."
            )
        if K is None:
            # fallback: low-rank approximation (dense), may be singular; the iid noise term still makes K ⊗ Σ + σ^2I PD.
            X = np.asarray(self.G.nodes[g]["data"], dtype=float)
            K = X @ X.T

        rowchol = self._chol_psd(K, jitter=float(jitter))

        # coords
        coords = {
            "obs_id": self._as_str_coords(self.obs_ids),
            attr_t["dim_name"]: self._as_str_coords(attr_t["coords"]),
        }
        for p in exo_parents:
            a = self.G.nodes[p]
            coords[a["dim_name"]] = self._as_str_coords(a["coords"])

        def _sd_dist(scale: float, *, shape=None):
            scale = max(float(scale), 1e-8)
            return pm.LogNormal.dist(mu=np.log(scale), sigma=float(sd_prior_sigma), shape=shape)

        trait_cov = str(trait_cov).lower()
        if sd_G is None:
            sd_G = float(edge_grm.get("sd_G", edge_grm.get("prior_sigma", 1.0)))

        with pm.Model(coords=coords) as model:
            # pm.Data for exo nodes
            X_exo = {}
            for p in exo_parents:
                a = self.G.nodes[p]
                X_exo[p] = pm.Data(p, a["data"], dims=("obs_id", a["dim_name"]))

            # fixed mean
            mu = pt.zeros((N, T))
            for p in exo_parents:
                a = self.G.nodes[p]
                e = self.G.edges[p, t]
                beta = pm.Normal(
                    f"beta:{p}->{t}",
                    mu=0.0,
                    sigma=float(e.get("prior_sigma", 1.0)),
                    dims=(a["dim_name"], attr_t["dim_name"]),
                )
                mu = mu + pt.dot(X_exo[p], beta)

            pm.Deterministic(f"prediction:{t}", mu, dims=("obs_id", attr_t["dim_name"]))

            # Σ_G
            if trait_cov == "lkj":
                chol_packed = pm.LKJCholeskyCov(
                    f"cholG:{g}->{t}",
                    n=T,
                    eta=float(eta_G),
                    sd_dist=_sd_dist(float(sd_G), shape=T),
                    compute_corr=False,
                )
                colchol = pm.expand_packed_triangular(T, chol_packed, lower=True)
            elif trait_cov == "diag":
                sd_vec = pm.LogNormal(
                    f"sdG:{g}->{t}",
                    mu=np.log(max(float(sd_G), 1e-8)),
                    sigma=float(sd_prior_sigma),
                    dims=(attr_t["dim_name"],),
                )
                colchol = pt.diag(sd_vec)
            elif trait_cov == "scalar":
                sd_s = pm.LogNormal(
                    f"sdG:{g}->{t}",
                    mu=np.log(max(float(sd_G), 1e-8)),
                    sigma=float(sd_prior_sigma),
                )
                colchol = sd_s * pt.eye(T)
            else:
                raise ValueError("trait_cov must be 'lkj'|'diag'|'scalar'.")

            sigma_e = pm.LogNormal(f"{t}:sigma", mu=np.log(max(float(sd_E), 1e-8)), sigma=float(sd_prior_sigma))
            mu_vec = pt.reshape(mu, (N * T,))

            pm.KroneckerNormal(
                t,
                mu=mu_vec,
                chols=[rowchol, colchol],
                sigma=sigma_e,
                observed=y_vec,
                shape=N * T,
            )

            return model

    def build_model_grm_traits_matrixnormal(
        self,
        *,
        traits_node: str | None = None,
        jitter: float = 1e-6,
        sd_prior_sigma: float = 0.5,
        eta_G: float = 2.0,
        eta_E: float = 4.0,
        sd_G: float | None = None,
        sd_E: float = 1.0,
        trait_cov_G: str = "lkj",   # Σ_G: 'lkj'|'diag'|'scalar'
        residual: str = "diag",     # Σ_E: 'diag'|'lkj' (rowcov=I)
        require_store_full: bool = True,
        impute_missing: bool = True,
    ):
        """Specialized compiler for the simple pattern: GRM -> traits (+ dense exo).

        This uses a latent matrix-normal genetic effect:

          G ~ MatrixNormal(0, rowcov=K, colcov=Σ_G)
          Y ~ MatrixNormal(mu+G, rowcov=I, colcov=Σ_E)

        This is the cleanest exact multi-trait LMM in this framework.
        """
        spec = self._simple_grm_to_traits_spec(traits_node=traits_node)
        if spec is None:
            raise ValueError("Graph is not a simple (exo,dense)* + (single GRM,grm) -> single normal traits pattern.")

        t = spec["traits"]
        g = spec["grm"]
        exo_parents = spec["exo_parents"]
        edge_grm = spec["edge_grm"]

        attr_t = self.G.nodes[t]
        if str(attr_t.get("likelihood", "normal")).lower() != "normal":
            raise ValueError("build_model_grm_traits_matrixnormal only supports normal likelihood.")

        Y = np.asarray(attr_t["data"], dtype=float)
        Y = self._ensure_2d(Y)
        N, T = Y.shape

        if np.isnan(Y).any() and (not bool(impute_missing)):
            raise ValueError("MatrixNormal likelihood requires missingness to be imputed (use impute_missing=True) or removed.")
        Y_obs = np.ma.masked_invalid(Y) if (np.isnan(Y).any() and bool(impute_missing)) else Y

        K = self.G.nodes[g].get("K_full")
        if K is None and bool(require_store_full):
            raise ValueError(
                f"GRM node '{g}' does not have K_full. Re-add it with add_grm(..., store_full=True)."
            )
        if K is None:
            X = np.asarray(self.G.nodes[g]["data"], dtype=float)
            K = X @ X.T
        rowchol = self._chol_psd(K, jitter=float(jitter))

        coords = {
            "obs_id": self._as_str_coords(self.obs_ids),
            attr_t["dim_name"]: self._as_str_coords(attr_t["coords"]),
        }
        for p in exo_parents:
            a = self.G.nodes[p]
            coords[a["dim_name"]] = self._as_str_coords(a["coords"])

        def _sd_dist(scale: float, *, shape=None):
            scale = max(float(scale), 1e-8)
            return pm.LogNormal.dist(mu=np.log(scale), sigma=float(sd_prior_sigma), shape=shape)

        trait_cov_G = str(trait_cov_G).lower()
        residual = str(residual).lower()
        if sd_G is None:
            sd_G = float(edge_grm.get("sd_G", edge_grm.get("prior_sigma", 1.0)))

        with pm.Model(coords=coords) as model:
            X_exo = {}
            for p in exo_parents:
                a = self.G.nodes[p]
                X_exo[p] = pm.Data(p, a["data"], dims=("obs_id", a["dim_name"]))

            mu = pt.zeros((N, T))
            for p in exo_parents:
                a = self.G.nodes[p]
                e = self.G.edges[p, t]
                beta = pm.Normal(
                    f"beta:{p}->{t}",
                    mu=0.0,
                    sigma=float(e.get("prior_sigma", 1.0)),
                    dims=(a["dim_name"], attr_t["dim_name"]),
                )
                mu = mu + pt.dot(X_exo[p], beta)
            pm.Deterministic(f"prediction:{t}", mu, dims=("obs_id", attr_t["dim_name"]))

            # Σ_G
            if trait_cov_G == "lkj":
                cholG_packed = pm.LKJCholeskyCov(
                    f"cholG:{g}->{t}",
                    n=T,
                    eta=float(eta_G),
                    sd_dist=_sd_dist(float(sd_G), shape=T),
                    compute_corr=False,
                )
                colcholG = pm.expand_packed_triangular(T, cholG_packed, lower=True)
            elif trait_cov_G == "diag":
                sd_vec = pm.LogNormal(
                    f"sdG:{g}->{t}",
                    mu=np.log(max(float(sd_G), 1e-8)),
                    sigma=float(sd_prior_sigma),
                    dims=(attr_t["dim_name"],),
                )
                colcholG = pt.diag(sd_vec)
            elif trait_cov_G == "scalar":
                sd_s = pm.LogNormal(
                    f"sdG:{g}->{t}",
                    mu=np.log(max(float(sd_G), 1e-8)),
                    sigma=float(sd_prior_sigma),
                )
                colcholG = sd_s * pt.eye(T)
            else:
                raise ValueError("trait_cov_G must be 'lkj'|'diag'|'scalar'.")

            Gmat = pm.MatrixNormal(
                f"G:{g}->{t}",
                mu=pt.zeros((N, T)),
                rowchol=rowchol,
                colchol=colcholG,
                dims=("obs_id", attr_t["dim_name"]),
            )

            if residual == "diag":
                sigma = pm.LogNormal(
                    f"{t}:sigma",
                    mu=np.log(max(float(sd_E), 1e-8)),
                    sigma=float(sd_prior_sigma),
                    dims=(attr_t["dim_name"],),
                )
                pm.Normal(t, mu=mu + Gmat, sigma=sigma, observed=Y_obs, dims=("obs_id", attr_t["dim_name"]))
            elif residual == "lkj":
                cholE_packed = pm.LKJCholeskyCov(
                    f"{t}:cholE",
                    n=T,
                    eta=float(eta_E),
                    sd_dist=_sd_dist(float(sd_E), shape=T),
                    compute_corr=False,
                )
                colcholE = pm.expand_packed_triangular(T, cholE_packed, lower=True)
                pm.MatrixNormal(
                    t,
                    mu=mu + Gmat,
                    rowcov=pt.eye(N),
                    colchol=colcholE,
                    observed=Y_obs,
                    dims=("obs_id", attr_t["dim_name"]),
                )
            else:
                raise ValueError("residual must be 'diag'|'lkj'.")

            return model

    def clone(self):
        """Deep-copy the causal graph structure (nodes/edges + attrs), sharing df_main."""
        cg = CausalGraph(self.df_main)
        cg.obs_ids = list(self.obs_ids)
        cg.G = copy.deepcopy(self.G)
        return cg

    def convexish_copy(
        self,
        *,
        force_normal_residual: str = "diag",
        force_grm_trait_cov: str = "scalar",
        drop_factor_edges: bool = False,
        drop_edges_from: tuple[str, ...] = ("residual_factor",),
    ):
        """Return a simplified graph intended for MAP/ADVI debugging.

        The goal is to reduce curvature pathologies (LKJ, rotations) while keeping most
        parameter names stable (dense betas, GRM Z matrices, latent offsets, etc.).
        """
        cg = self.clone()

        # 1) normal residual structure
        force_normal_residual = str(force_normal_residual).lower()
        if force_normal_residual not in ("diag", "lkj"):
            raise ValueError("force_normal_residual must be 'diag' or 'lkj'.")
        for n, a in cg.G.nodes(data=True):
            if a.get("type") == "endo" and str(a.get("likelihood", "normal")).lower() == "normal":
                a["residual"] = force_normal_residual

        # 2) GRM trait covariance structure
        force_grm_trait_cov = str(force_grm_trait_cov).lower()
        for u, v, e in cg.G.edges(data=True):
            if str(e.get("kind", "dense")).lower() == "grm":
                e["grm_trait_cov"] = force_grm_trait_cov

        # 3) optionally drop known-rotational edges
        drop_from = set(drop_edges_from)
        for u, v, e in list(cg.G.edges(data=True)):
            if u in drop_from:
                cg.G.remove_edge(u, v)
            elif drop_factor_edges and str(e.get("kind", "")).lower() == "factor":
                cg.G.remove_edge(u, v)

        return cg

    @staticmethod
    def initvals_from_point(model: pm.Model, point: dict):
        """Build a valid initvals dict for `model` from a (possibly partial) point.

        Useful for staged workflows where MAP/ADVI was fit on a simplified model.
        Only keys that exist in `model.initial_point()` are retained.
        """
        init = model.initial_point()
        if point is None:
            return init
        for k, v in dict(point).items():
            if k in init:
                init[k] = v
        return init

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
                    mu = pt.zeros((len(self.obs_ids), n_dim), dtype="float32")
    
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
        edge_color_map = {"dense": "steelblue",  "grm": "firebrick", "factor": "purple"}
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

def chol2cov(L, return_correlation = False, eps: float = 1e-12):
    import xarray as xr
    variance = (L**2).sum(dim="traits_dim_t")
    stds = np.sqrt(variance)
    L_transposed = L.rename({"traits_dim": "traits_dim_2"})
    covariance = xr.dot(L, L_transposed, dims="traits_dim_t")
    stds_2 = stds.rename({"traits_dim": "traits_dim_2"})
    if not return_correlation: return covariance
    denom = (stds * stds_2)
    return covariance / (denom + eps)

def chol2covMAP(trace, name, columns, return_correlation = False, eps: float = 1e-12):
    mapl = trace[name]
    mapcov = mapl.dot(mapl.T) 
    if not return_correlation: return pd.DataFrame(mapcov, columns = columns, index = columns)
    mapsig = np.sqrt(np.diag(mapcov))
    denom = np.outer(mapsig, mapsig)
    return pd.DataFrame(mapcov/(denom + eps), columns = columns, index = columns)

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






    
