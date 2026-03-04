
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import networkx as nx

# Optional plotting deps (only required if you call plot_graph)
import holoviews as hv
import hvplot.networkx as hvnx


def softplus(x: pt.TensorVariable) -> pt.TensorVariable:
    # numerically stable softplus
    return pt.log1p(pt.exp(-pt.abs(x))) + pt.maximum(x, 0.0)


def _corr_from_cov(Sigma: pt.TensorVariable, eps: float = 1e-12) -> pt.TensorVariable:
    d = pt.sqrt(pt.clip(pt.diag(Sigma), eps, np.inf))
    denom = d[:, None] * d[None, :]
    return Sigma / (denom + eps)


def _pack_pattern_mask(M: np.ndarray) -> np.ndarray:
    """
    Pack boolean missing mask (n x t) into bytes per row so we can count unique patterns fast.
    M: True where missing
    Returns: uint8 packed, shape (n, ceil(t/8))
    """
    M = np.asarray(M, dtype=np.uint8)
    return np.packbits(M, axis=1)


@dataclass
class _ModelBuildInfo:
    engine: str
    obs_ids: list[str]
    trait_nodes: list[str]


class CausalGraph:
    """
    DAG -> PyMC compiler for multivariate genetic models with:

    - Exogenous nodes: pm.Data, optionally standardized and QR-reduced.
    - GRM nodes: low-rank representation X = U * sqrt(S), where K ≈ X X^T.
    - Latent nodes: per-individual latent variables (e.g., genomicSEM-style factors).
    - Endogenous nodes: observed nodes with Normal/Poisson/Bernoulli likelihoods.

    Key design points for speed + missingness + interpretability
    -----------------------------------------------------------
    1) Missingness:
       - missing='mask' uses a flattened observed-only likelihood (no imputation RV explosion).
       - missing='pattern' is used only for full MVN residual models (lkj/fa) when pattern count is small.
       - missing='impute' uses PyMC masked arrays (can explode dimensionality; keep for small problems).
       - missing='row' is for collapsed marginal-eVD fast path (requires row-complete after filtering).

    2) Residual covariance:
       - residual='diag' is fastest and works naturally with missing='mask'.
       - residual='factor' implements a generative low-rank residual covariance (exact under masking),
         with identifiable loadings (lower-triangular + positive diagonal).
       - residual='lkj'/'fa' are marginal MVN residual models; masking requires missing='pattern' or 'impute'.

    3) Genetic correlations:
       - GRM edges can use grm_trait_cov='lkj' and we store deterministics:
         SigmaG:<edge>, RG:<edge>, V_G_param:<edge>.

    4) State-of-the-art latent spaces:
       - factor edges (latent->traits) use identifiable loadings by default.

    Notes:
      - This is designed to be NUTS-viable for moderate N (e.g., 1k–5k) and r (e.g., 200–800),
        and to have practical MAP/ADVI behavior by avoiding missing-value latent variables by default.
    """

    def __init__(self, df_main: pd.DataFrame):
        self.df_main = df_main
        self.obs_ids = [str(x) for x in df_main.index.tolist()]
        self.G = nx.DiGraph()
        self._last_build: _ModelBuildInfo | None = None

    # -----------------------------
    # basic utilities
    # -----------------------------
    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    @staticmethod
    def _as_str_coords(coords) -> list[str]:
        return [str(c) for c in coords]

    @staticmethod
    def _center_scale(arr: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(arr, dtype=float)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
        std = np.where(std < eps, 1.0, std)
        out = (arr - mean) / std
        return out, mean, std

    @staticmethod
    def decompose_grm(K: np.ndarray, *, r: int, jitter: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        K = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
        n = K.shape[0]
        r = max(1, min(int(r), n - 1))
        K = K + float(jitter) * np.eye(n)
        evals, evecs = np.linalg.eigh(K)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, np.inf)
        evecs = evecs[:, order]
        evals, evecs = evals[:r], evecs[:, :r]
        return evecs, evals

    # -----------------------------
    # node creation
    # -----------------------------
    def add_input(
        self,
        name: str,
        data: pd.DataFrame | pd.Series | np.ndarray,
        *,
        fillna_for_exo: float = 0.0,
        standardize: bool = True,
        qr: bool = False,
    ) -> None:
        data = data.reindex(self.df_main.index) if isinstance(data, (pd.Series, pd.DataFrame)) else data
        if isinstance(data, pd.Series):
            arr = data.to_numpy(dtype=float).reshape(-1, 1)
            coords = [data.name if data.name else f"{name}1"]
        elif isinstance(data, pd.DataFrame):
            arr = data.to_numpy(dtype=float)
            coords = list(map(str, data.columns))
        else:
            arr = self._ensure_2d(np.asarray(data, dtype=float))
            coords = [f"{name}{i}" for i in range(1, arr.shape[1] + 1)]

        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=float(fillna_for_exo))

        mean = std = None
        if standardize and arr.shape[1] > 0:
            arr, mean, std = self._center_scale(arr)

        R = None
        if qr and arr.shape[1] > 0:
            Q, R = np.linalg.qr(arr, mode="reduced")
            arr = Q

        self.G.add_node(
            name,
            type="exo",
            data=arr.astype(float),
            shape=arr.shape,
            dim_name=f"{name}_dim",
            coords=self._as_str_coords(coords),
            standardize=bool(standardize),
            mean=None if mean is None else mean.astype(float),
            scale=None if std is None else std.astype(float),
            qr=bool(qr),
            qr_R=None if R is None else R.astype(float),
        )

    def add_grm(
        self,
        name: str,
        grm_matrix: pd.DataFrame | np.ndarray,
        *,
        n_components: int = 300,
        top_eigen2fixed: float | None = None,
        subtract_fixed: str | None = None,
        jitter: float = 1e-6,
        store_full: bool = False,
        align: Literal["strict", "intersection"] = "strict",
    ) -> None:
        """
        Adds a GRM node using low-rank X = U*sqrt(S) with K ≈ X X^T.

        - align='strict': require GRM to contain all obs_ids (recommended).
        - top_eigen2fixed: move top fraction of eigen-variance into an exo node '<name>_fixed'
          (dense fixed-effect PCs).
        - subtract_fixed: project out an existing exo node from the random-effect X.
        - store_full: store K_full (dense) for exact/collapsed compilers (memory heavy).
        """
        if isinstance(grm_matrix, pd.DataFrame):
            if align == "strict":
                if not (set(self.df_main.index).issubset(set(grm_matrix.index)) and set(self.df_main.index).issubset(set(grm_matrix.columns))):
                    raise ValueError("GRM does not contain all obs_ids; use align='intersection' or pre-align.")
                K = grm_matrix.loc[self.df_main.index, self.df_main.index].to_numpy(dtype=float)
            else:
                idx = self.df_main.index.intersection(grm_matrix.index).intersection(grm_matrix.columns)
                self.df_main = self.df_main.loc[idx]
                self.obs_ids = [str(x) for x in self.df_main.index.tolist()]
                K = grm_matrix.loc[idx, idx].to_numpy(dtype=float)
        else:
            K = np.asarray(grm_matrix, dtype=float)

        n = K.shape[0]
        U, S = self.decompose_grm(K, r=int(n_components), jitter=float(jitter))
        X = U * np.sqrt(S)[None, :]

        fixed_node = None
        if top_eigen2fixed is not None:
            frac = float(top_eigen2fixed)
            if not (0.0 < frac < 1.0):
                raise ValueError("top_eigen2fixed must be a float in (0,1).")
            total = float(np.sum(S)) if float(np.sum(S)) > 0 else 1.0
            cum = np.cumsum(S) / total
            cutoff = int(np.searchsorted(cum, frac) + 1)
            cutoff = max(1, min(cutoff, X.shape[1] - 1))
            X_fixed = X[:, :cutoff]
            fixed_node = f"{name}_fixed"
            self.G.add_node(
                fixed_node,
                type="exo",
                data=X_fixed.astype(float),
                shape=X_fixed.shape,
                dim_name=f"{fixed_node}_dim",
                coords=self._as_str_coords([f"Pc{i}" for i in range(1, cutoff + 1)]),
                standardize=False,
                mean=None,
                scale=None,
                qr=False,
                qr_R=None,
            )
            X = X[:, cutoff:]

        if subtract_fixed is not None:
            if subtract_fixed not in self.G.nodes:
                raise ValueError(f"subtract_fixed='{subtract_fixed}' not found.")
            fix = self.G.nodes[subtract_fixed]
            if fix["type"] != "exo":
                raise ValueError("subtract_fixed must be an exo node.")
            C = np.asarray(fix["data"], dtype=float)
            beta, *_ = np.linalg.lstsq(C, X, rcond=None)
            X = X - C @ beta

        kdiag = np.sum(X**2, axis=1)
        K_full = None
        if bool(store_full):
            K_full = 0.5 * (K + K.T)

        self.G.add_node(
            name,
            type="grm",
            data=X.astype(float),
            shape=X.shape,
            dim_name=f"{name}_pc",
            coords=self._as_str_coords([f"Pc{i}" for i in range(1, X.shape[1] + 1)]),
            fixed_node=fixed_node,
            K_full=K_full,
            kdiag_mean=float(np.mean(kdiag)),
        )

    def add_latent(
        self,
        name: str,
        *,
        n_dim: int = 1,
        prior_sigma: float = 1.0,
        track_values: bool = True,
        variance_budget: bool = False,
        budget_w_eps: float = 1e-6,
        budget_concentration: float = 20.0,
        is_factor: bool = False,
    ) -> None:
        n_dim = int(n_dim)
        self.G.add_node(
            name,
            type="latent",
            data=None,
            shape=(len(self.obs_ids), n_dim),
            dim_name=f"{name}_dim",
            coords=self._as_str_coords([f"{name}{i}" for i in range(1, n_dim + 1)]),
            prior_sigma=float(prior_sigma),
            track_values=bool(track_values),
            variance_budget=bool(variance_budget),
            budget_w_eps=float(budget_w_eps),
            budget_concentration=float(budget_concentration),
            is_factor=bool(is_factor),
        )

    def add_adjacency(
        self,
        name: str,
        *,
        W: "pd.DataFrame | np.ndarray | Any | None" = None,
        edges: "pd.DataFrame | list[tuple[Any, Any]] | None" = None,
        src_col: str = "iid1",
        dst_col: str = "iid2",
        weight_col: str | None = None,
        binary: bool = True,
        threshold: float = 0.0,
        symmetrize: bool = True,
        drop_diagonal: bool = True,
        align: Literal["strict", "intersection"] = "strict",
    ) -> None:
        """
        Add an adjacency node for BYM2 edges.
    
        Provide either:
          - W: adjacency matrix (DataFrame with index/cols, ndarray, or scipy sparse), OR
          - edges: edge list (DataFrame or list of (src, dst)).
    
        BYM2/ICAR expects symmetric (undirected) adjacency; we symmetrize by default.
        If binary=True, any nonzero becomes 1 (optionally after threshold).
        """
        import scipy.sparse as sps
        obs = self.df_main.index
        n = len(obs)
    
        # ---- build sparse adjacency ----
        if W is None:
            if edges is None:
                raise ValueError("Provide W or edges to add_adjacency().")
    
            if isinstance(edges, pd.DataFrame):
                if (src_col not in edges.columns) or (dst_col not in edges.columns):
                    raise ValueError(f"edges must contain columns '{src_col}' and '{dst_col}'.")
                src = edges[src_col].astype(str).to_numpy()
                dst = edges[dst_col].astype(str).to_numpy()
                if (weight_col is not None) and (weight_col in edges.columns) and (not binary):
                    val = edges[weight_col].astype(float).to_numpy()
                else:
                    val = np.ones(len(edges), dtype=float)
            else:
                pairs = [(str(a), str(b)) for a, b in edges]
                src = np.asarray([a for a, _ in pairs], dtype=object)
                dst = np.asarray([b for _, b in pairs], dtype=object)
                val = np.ones(len(pairs), dtype=float)
    
            id2i = {str(x): i for i, x in enumerate(obs)}
            rr, cc, vv = [], [], []
            for a, b, w in zip(src, dst, val, strict=False):
                ia = id2i.get(str(a), None)
                ib = id2i.get(str(b), None)
                if ia is None or ib is None:
                    continue
                rr.append(ia)
                cc.append(ib)
                vv.append(float(w))
    
            Wsp = sps.csr_matrix((vv, (rr, cc)), shape=(n, n))
    
        else:
            if isinstance(W, pd.DataFrame):
                if align == "strict":
                    if not (set(obs).issubset(set(W.index)) and set(obs).issubset(set(W.columns))):
                        raise ValueError("Adjacency DF missing obs_ids; pre-align or use align='intersection'.")
                    A = W.loc[obs, obs].to_numpy()
                else:
                    idx = obs.intersection(W.index).intersection(W.columns)
                    self.df_main = self.df_main.loc[idx]
                    self.obs_ids = [str(x) for x in self.df_main.index.tolist()]
                    obs = self.df_main.index
                    n = len(obs)
                    A = W.loc[obs, obs].to_numpy()
                Wsp = sps.csr_matrix(A)
            else:
                Wsp = sps.csr_matrix(W) if not sps.issparse(W) else W.tocsr()
                if Wsp.shape != (n, n):
                    raise ValueError(f"W must have shape {(n, n)} matching df_main; got {Wsp.shape}.")
    
        # ---- sanitize ----
        if binary:
            if float(threshold) > 0:
                Wsp = (Wsp > float(threshold)).astype(float).tocsr()
            else:
                if Wsp.nnz > 0:
                    Wsp.data[:] = 1.0
    
        if symmetrize:
            Wsp = (Wsp + Wsp.T)
            if binary:
                Wsp = (Wsp > 0).astype(float).tocsr()
            else:
                Wsp = (0.5 * Wsp).tocsr()
    
        if drop_diagonal:
            Wsp.setdiag(0.0)
            Wsp.eliminate_zeros()
    
        # basic stats for QC
        deg = np.asarray(Wsp.sum(axis=1)).ravel()
        mean_deg = float(deg.mean()) if len(deg) else 0.0
        # ---- dummy data so pm.Data exists (compiler needs model_vars[parent]) ----
        dummy = np.zeros((n, 1), dtype=float)
    
        self.G.add_node(
            name,
            type="adj",
            data=dummy,
            shape=dummy.shape,
            dim_name=f"{name}_dim",
            coords=self._as_str_coords([f"{name}1"]),
            standardize=False,
            mean=None,
            scale=None,
            qr=False,
            qr_R=None,
            W=Wsp.toarray(),
            nnz=int(Wsp.nnz),
            mean_degree=mean_deg,
        )

    def add_trait(
        self,
        name: str,
        *,
        data_matrix: pd.DataFrame | np.ndarray | None = None,
        data_cols: list[str] | None = None,
        likelihood: Literal["normal", "poisson", "bernoulli"] = "normal",
        link: Literal["identity", "log", "logit"] = "identity",
        residual: Literal["diag", "lkj", "fa", "factor"] = "diag",
        # residual hyperparams
        eta_E: float = 4.0,
        var_E: float = 1.0,
        n_factors_E: int | None = None,
        # missing handling (NEW)
        missing: Literal["auto", "mask", "pattern", "impute", "row"] = "auto",
        pattern_max_groups: int = 2048,
        pattern_max_t: int = 25,
        # backwards compat
        impute_missing: bool | None = None,
        # tracking
        track_values: bool = True,
        track_values_per_path: bool = False,
        variance_budget: bool = False,
        budget_w_eps: float = 1e-6,
        budget_concentration: float = 20.0,
    ) -> None:
        if data_matrix is None and data_cols is None:
            raise ValueError("Provide data_matrix or data_cols.")
        if data_matrix is None:
            data = self.df_main[list(data_cols)]
        else:
            data = data_matrix
        data = data.reindex(self.df_main.index) if isinstance(data, (pd.DataFrame, pd.Series)) else data
        if isinstance(data, pd.Series):
            arr = data.to_numpy(dtype=float).reshape(-1, 1)
            coords = [data.name if data.name else f"{name}1"]
        elif isinstance(data, pd.DataFrame):
            arr = data.to_numpy(dtype=float)
            coords = list(map(str, data.columns))
        else:
            arr = self._ensure_2d(np.asarray(data, dtype=float))
            coords = [f"{name}{i}" for i in range(1, arr.shape[1] + 1)]

        likelihood = str(likelihood).lower()
        residual = str(residual).lower()
        link = str(link).lower()

        # Backwards compat: if impute_missing provided, map to missing mode
        if impute_missing is not None:
            missing = "impute" if bool(impute_missing) else "auto"

        self.G.add_node(
            name,
            type="endo",
            data=arr,
            shape=arr.shape,
            dim_name=f"{name}_dim",
            coords=self._as_str_coords(coords),
            likelihood=likelihood,
            link=link,
            residual=residual,
            eta_E=float(eta_E),
            sd_E=float(np.sqrt(max(float(var_E), 1e-12))),
            n_factors_E=None if n_factors_E is None else int(n_factors_E),
            missing=str(missing).lower(),
            pattern_max_groups=int(pattern_max_groups),
            pattern_max_t=int(pattern_max_t),
            track_values=bool(track_values),
            track_values_per_path=bool(track_values_per_path),
            variance_budget=bool(variance_budget),
            budget_w_eps=float(budget_w_eps),
            budget_concentration=float(budget_concentration),
        )

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        kind: Literal["auto", "dense", "grm", "factor", 'bym2'] = "auto",
        prior_variance: float = 1.0,
        eta: float = 2.0,
        grm_trait_cov: Literal["lkj", "diag", "scalar"] = "lkj",
        z_prior: Literal["iid", "ard_pc", "rhs"] = "iid",
        rhs_global_scale: float = 0.3,
        factor_identifiable: bool = True,
        track_variance: bool = False,
        adjust_variance_by_shape: bool = True,
    ) -> None:
        if source not in self.G.nodes or target not in self.G.nodes:
            raise ValueError(f"Missing nodes for edge {source}->{target}")
        src = self.G.nodes[source]
        tgt = self.G.nodes[target]
        if kind == "auto":
            if src["type"] == "grm":
                kind = "grm"
            elif src["type"] == "latent" and bool(src.get("is_factor", False)) and tgt["type"] == "endo":
                kind = "factor"
            else:
                kind = "dense"

        kind = str(kind).lower()
        prior_variance = float(prior_variance)
        if adjust_variance_by_shape and kind != "grm":
            prior_variance = prior_variance / float(max(1, src["shape"][1]))
        prior_sigma = float(np.sqrt(max(prior_variance, 1e-12)))

        self.G.add_edge(
            source,
            target,
            kind=kind,
            prior_sigma=prior_sigma,
            eta=float(eta),
            grm_trait_cov=str(grm_trait_cov).lower(),
            z_prior=str(z_prior).lower(),
            rhs_global_scale=float(rhs_global_scale),
            factor_identifiable=bool(factor_identifiable),
            track_variance=bool(track_variance),
            adjust_variance_by_shape=bool(adjust_variance_by_shape),
        )

    # -----------------------------
    # graph QC / plotting
    # -----------------------------
    def graph_qc(self) -> None:
        print("--- Causal Graph QC ---")
        print(f"Nodes: {self.G.number_of_nodes()} | Edges: {self.G.number_of_edges()}")
        try:
            cycles = list(nx.simple_cycles(self.G))
            print("✅ DAG: acyclic" if not cycles else f"⚠️ cycles detected: {cycles}")
        except Exception:
            print("cycle check skipped")
        isolates = list(nx.isolates(self.G))
        if isolates:
            print(f"⚠️ isolates: {isolates}")

        nodes = pd.DataFrame.from_dict(dict(self.G.nodes(data=True)), orient="index")
        nodes.index.name = "node"
        nodes["n_parents"] = [self.G.in_degree(n) for n in nodes.index]
        nodes["n_children"] = [self.G.out_degree(n) for n in nodes.index]
        if "shape" in nodes.columns:
            nodes["shape_str"] = nodes["shape"].apply(lambda s: "x".join(map(str, s)) if isinstance(s, (tuple, list)) else str(s))
        def misspct(a):
            x = a.get("data", None)
            if x is None:
                return np.nan
            x = np.asarray(x)
            return float(np.isnan(x).mean()) * 100.0
        nodes["missing_pct"] = [misspct(dict(self.G.nodes[n])) for n in nodes.index]
        cols = [c for c in ["type","shape_str","n_parents","n_children","missing_pct","likelihood","residual","missing"] if c in nodes.columns]
        print(nodes.reset_index()[["node"]+cols].to_markdown(index=False))

        edges = nx.to_pandas_edgelist(self.G)
        if len(edges):
            print(edges.sort_values(["source","target"]).to_markdown(index=False))

    def plot_graph(self, *, show_edge_kinds: bool = True, show_edge_param_labels: bool = True):
        """
        Interactive DAG view (HoloViews/Bokeh).
    
        - Edge width scales with the *unscaled* prior sigma (undoes variance-by-shape normalization for display).
        - Edge hover shows kind, n_params, and prior variance/sigma.
        - Optional edge midpoint labels with (n_params, variance).
        """
        P = nx.DiGraph()
        # -----------------
        # nodes
        # -----------------
        for n, attr in self.G.nodes(data=True):
            ntype = str(attr.get("type", "unknown"))
            shape = attr.get("shape", None)
            shape_str = "x".join(map(str, shape)) if shape is not None else "?"
            likelihood = str(attr.get("likelihood", "N/A"))
            residual = str(attr.get("residual", "N/A"))
            missing = str(attr.get("missing", "N/A"))
            P.add_node(  n, type=ntype,  shape_str=shape_str, likelihood=likelihood, residual=residual, missing=missing, )
    
        def _edge_param_count(u: str, v: str, ed: dict[str, Any]) -> int:
            kind = str(ed.get("kind", "dense")).lower()
            su = self.G.nodes[u].get("shape", (0, 0))
            sv = self.G.nodes[v].get("shape", (0, 0))
            p = int(su[1]) if (su is not None and len(su) > 1) else 0
            q = int(sv[1]) if (sv is not None and len(sv) > 1) else 0
    
            if kind == "dense": return int(p * q)

            if kind == "bym2":
                n_obs = int(self.G.nodes[v].get("shape", (len(self.obs_ids), 0))[0])
                d = q
                if d <= 0: return 0
                # ICAR_raw + IID_raw per dim => ~2*n_obs*d plus rho + sigma_bym
                n = int(2 * n_obs * d + 2)
                tc = str(ed.get("grm_trait_cov", "diag")).lower()
                if tc == "diag": n += int(d)
                elif tc == "scalar": n += 1
                else: n += int(d * (d + 1) // 2)
                return int(n)
    
            if kind == "factor":
                k = p
                t = q
                if k <= 0 or t <= 0:  return 0
                if bool(ed.get("factor_identifiable", True)): return int(max(0, t * k - (k * (k - 1)) // 2))
                return int(t * k)
    
            if kind == "grm":
                r,d = p, q
                if r <= 0 or d <= 0:  return 0
                n = int(r * d)
                # shrinkage prior extras
                z_prior = str(ed.get("z_prior", "iid")).lower()
                if z_prior == "ard_pc": n += int(r)  # tau_pc per PC
                elif z_prior == "rhs":  n += int(r) + 2  # lam per PC + global tau + c2
                tc = str(ed.get("grm_trait_cov", "lkj")).lower()
                if tc == "diag":n += int(d)
                elif tc == "scalar":   n += 1
                else: n += int(d * (d + 1) // 2)
                return int(n)
            return 0
    
        # -----------------
        # edges (with prior scaling + parameter counts)
        # -----------------
        for u, v, attr in self.G.edges(data=True):
            kind = str(attr.get("kind", "dense")).lower()
    
            # stored prior_sigma may be normalized by source dim (variance budget);
            # for visualization undo that normalization so thickness reflects user prior_variance.
            raw_sigma = float(attr.get("prior_sigma", 1.0))
            if bool(attr.get("adjust_variance_by_shape", False)) and kind != "grm":
                su = self.G.nodes[u].get("shape", (1, 1))
                raw_sigma *= float(np.sqrt(max(int(su[1]), 1)))
    
            variance = float(raw_sigma**2)
            confidence = float(min(4.0, max(raw_sigma, 0.1)))  # clamped width driver
            n_params = int(_edge_param_count(u, v, attr))
    
            P.add_edge(u,v, kind=kind, n_params=n_params, prior_sigma=float(raw_sigma),
                       variance=variance, confidence=confidence)
    
        # left/middle/right layout by in/out degree
        for n in P.nodes():
            out_degree = P.out_degree(n)
            in_degree = P.in_degree(n)
            subset = 0 if in_degree == 0 else (2 if out_degree == 0 else 1)
            P.nodes[n]["subset"] = int(subset)
    
        pos = nx.multipartite_layout(P, subset_key="subset", scale=2.0)
    
        color_map = {"exo": "lightsteelblue", "grm": "salmon", "latent": "palegoldenrod", "endo": "darkseagreen", 'adj': 'orange'}
        marker_map = {"exo": "circle", "grm": "hex", "latent": "square", "endo": "square",  'adj': 'hex'}
        edge_color_map = {"dense": "steelblue", "grm": "firebrick", "factor": "purple", 'bym2': 'orange'}
    
        for n in P.nodes():
            t = P.nodes[n]["type"]
            P.nodes[n]["color"] = str(color_map.get(t, "white"))
            P.nodes[n]["marker"] = str(marker_map.get(t, "circle"))
    
        edge_tooltips = [
            ("Edge", "@start -> @end"),
            ("Kind", "@kind"),
            ("Params", "@n_params"),
            ("Prior sigma", "@prior_sigma"),
            ("Prior var", "@variance") ]
        node_tooltips = [
            ("Node", "@index"),
            ("Type", "@type"),
            ("Shape", "@shape_str"),
            ("Likelihood", "@likelihood"),
            ("Residual", "@residual"),
            ("Missing", "@missing")]
    
        edge_plot = hvnx.draw_networkx_edges(
            P, pos,edge_color="kind" if show_edge_kinds else "black",
            alpha=0.6,arrow_style="-|>", arrow_size=5, edge_width=4 + hv.dim("confidence").norm() * 15,
            cmap=edge_color_map ).opts(tools=["hover"], hover_tooltips=edge_tooltips)
    
        node_plot = hvnx.draw_networkx_nodes(
            P,  pos, node_color="type", node_marker="marker",
            alpha=1.0, node_size=5200, cmap=color_map,
        ).opts(tools=["hover"], hover_tooltips=node_tooltips)
    
        label_df = pd.DataFrame(  {"x": [float(pos[n][0]) for n in P.nodes()],
                "y": [float(pos[n][1]) for n in P.nodes()],
                "text": [f"{n}\n{P.nodes[n]['shape_str']}" for n in P.nodes()]})
        labels = hv.Labels(label_df, kdims=["x", "y"], vdims=["text"]).opts(
                            text_color="black", text_font_size="8pt", yoffset=0.0,
                            text_align="center",  text_baseline="middle" )
    
        edge_param_labels = None
        if show_edge_param_labels and P.number_of_edges() > 0:
            rows = []
            for u, v, ed in P.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                rows.append(
                    {"x": float((x0 + x1) / 2.0),"y": float((y0 + y1) / 2.0),
                    "text": f"n_params: {int(ed['n_params'])}\nvariance: {round(float(ed['variance']), 3)}" } )
            edge_param_labels = hv.Labels(pd.DataFrame(rows), kdims=["x", "y"], vdims=["text"]).opts(
                text_font_size="9pt", text_color="black",
                text_align="left", text_baseline="middle", )
    
        graph = edge_plot * node_plot * labels
        if edge_param_labels is not None: graph = graph * edge_param_labels
    
        return graph.opts( title="GraphView", width=900, height=650, xaxis=None,
                           yaxis=None, tools=["tap", "box_select", "hover"] )

    # -----------------------------
    # build_model: engines & missing resolution
    # -----------------------------
    def _simple_grm_to_traits_spec(self, traits_node: str | None = None) -> dict[str, Any] | None:
        endos = [n for n, a in self.G.nodes(data=True) if a.get("type") == "endo"]
        if traits_node is None:
            if len(endos) != 1:
                return None
            traits_node = endos[0]
        if traits_node not in self.G.nodes or self.G.nodes[traits_node].get("type") != "endo":
            return None
        # no latents
        if any(a.get("type") == "latent" for _, a in self.G.nodes(data=True)):
            return None
        # traits is leaf
        if len(list(self.G.successors(traits_node))) != 0:
            return None
        parents = list(self.G.predecessors(traits_node))
        grm_parents = [p for p in parents if self.G.nodes[p].get("type") == "grm"]
        if len(grm_parents) != 1:
            return None
        g = grm_parents[0]
        e = self.G.edges[g, traits_node]
        if str(e.get("kind", "dense")).lower() != "grm":
            return None
        for p in parents:
            if p == g:
                continue
            if self.G.nodes[p].get("type") != "exo":
                return None
            if str(self.G.edges[p, traits_node].get("kind", "dense")).lower() != "dense":
                return None
        return {"traits": traits_node, "grm": g, "parents": parents}

    @staticmethod
    def _pattern_groups(Y: np.ndarray) -> tuple[int, dict[bytes, np.ndarray]]:
        # returns (#patterns, {pattern_bytes: row_idx})
        miss = np.isnan(Y)
        packed = _pack_pattern_mask(miss)
        # unique rows
        uniq, inv = np.unique(packed, axis=0, return_inverse=True)
        groups: dict[bytes, list[int]] = {}
        for i, g in enumerate(inv):
            key = uniq[g].tobytes()
            groups.setdefault(key, []).append(i)
        out = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        return len(out), out

    def _resolve_missing_mode(self, node: str, Y: np.ndarray) -> str:
        a = self.G.nodes[node]
        residual = str(a.get("residual","diag")).lower()
        mode = str(a.get("missing","auto")).lower()
        if not np.isnan(Y).any():
            return "none"
        if mode != "auto":
            return mode
        # auto
        if residual in ("diag","factor"):
            return "mask"
        if residual in ("lkj","fa"):
            t = Y.shape[1]
            if t <= int(a.get("pattern_max_t", 25)):
                npatt, _ = self._pattern_groups(Y)
                if npatt <= int(a.get("pattern_max_groups", 2048)):
                    return "pattern"
            raise ValueError(
                f"Node '{node}' has residual='{residual}' with elementwise missingness. "
                "Auto mode refuses because pattern MVN would explode. "
                "Use residual='factor' (recommended) or residual='diag', or set missing='impute' for small problems."
            )
        return "mask"

    # -----------------------------
    # Core: generic build_model
    # -----------------------------
    def build_model(
        self,
        *,
        engine: Literal["auto", "generic", "marginal_evd"] = "auto",
        sd_prior_sigma: float = 0.5,
        jitter: float = 1e-6,
        poisson_clip: float = 20.0,
        pattern_jitter: float = 1e-6,
    ) -> pm.Model:
        """
        engine='auto':
          - if graph matches simple GRM->traits and traits missing in {'none','row'} and traits residual in {'lkj','fa','diag','factor'},
            then use build_model_grm_traits_marginal_evd (collapsed) when possible.
          - otherwise uses generic compiler with missing auto-resolution.

        engine='generic' forces generic compiler.
        engine='marginal_evd' forces collapsed compiler (requires eligibility).
        """
        if engine not in ("auto","generic","marginal_evd"):
            raise ValueError("engine must be 'auto'|'generic'|'marginal_evd'")

        # auto fast path
        if engine in ("auto","marginal_evd"):
            spec = self._simple_grm_to_traits_spec()
            if spec is not None:
                tnode = spec["traits"]
                a = self.G.nodes[tnode]
                Y = np.asarray(a["data"], dtype=float)
                miss_mode = str(a.get("missing","auto")).lower()
                if miss_mode == "auto":
                    miss_mode = self._resolve_missing_mode(tnode, Y)
                # collapsed only if no elementwise missing (none) OR user explicitly wants row-complete
                if miss_mode in ("none","row") and str(a.get("likelihood","normal")).lower() == "normal":
                    if engine == "marginal_evd" or (engine == "auto"):
                        model = self.build_model_grm_traits_marginal_evd(
                            target=tnode,
                            row_complete=(miss_mode=="row"),
                            jitter=float(jitter),
                            sd_prior_sigma=float(sd_prior_sigma),
                        )
                        self._last_build = _ModelBuildInfo(engine="marginal_evd", obs_ids=self.obs_ids, trait_nodes=[tnode])
                        return model
            if engine == "marginal_evd":
                raise ValueError("engine='marginal_evd' requested but graph is not eligible.")

        # generic compiler
        order = list(nx.topological_sort(self.G))
        coords: dict[str, list[str]] = {"obs_id": self._as_str_coords(self.obs_ids)}
        for n, a in self.G.nodes(data=True):
            coords[a["dim_name"]] = self._as_str_coords(a["coords"])

        def _sd_dist(scale: float, shape=None):
            scale = max(float(scale), 1e-8)
            return pm.LogNormal.dist(mu=np.log(scale), sigma=float(sd_prior_sigma), shape=shape)

        with pm.Model(coords=coords) as model:
            model_vars: dict[str, pt.TensorVariable] = {}
            # pm.Data for exo/grm
            for node in order:
                a = self.G.nodes[node]
                if a["type"] in ("exo","grm", "adj"):
                    model_vars[node] = pm.Data(node, a["data"], dims=("obs_id", a["dim_name"]))

            # compile latents/endos
            for node in order:
                a = self.G.nodes[node]
                if a["type"] in ("exo","grm", "adj"): continue

                n_dim = int(a["shape"][1])
                dim_name = a["dim_name"]
                parents = list(self.G.predecessors(node))

                # variance budget weights (optional; applied as sqrt(w) to each parent's contribution)
                w_par = None
                if bool(a.get("variance_budget", False)) and len(parents) > 1:
                    conc = float(a.get("budget_concentration", 20.0))
                    eps = float(a.get("budget_w_eps", 1e-6))
                    w_raw = pm.Dirichlet(f"{node}:w_parent_raw", a=np.ones(len(parents)) * conc)
                    w_par = pm.Deterministic(f"{node}:w_parent", w_raw * (1.0 - eps * len(parents)) + eps)
                # linear predictor excluding residual factors
                mu = pt.zeros((len(self.obs_ids), n_dim))

                # keep track of parameter-implied genetic variances for this node
                V_G_param_total = None

                for i, parent in enumerate(parents):
                    pa = self.G.nodes[parent]
                    e = self.G.edges[parent, node]
                    kind = str(e.get("kind","dense")).lower()
                    Xp = model_vars[parent]
                    scale_w = pt.sqrt(w_par[i]) if (w_par is not None) else 1.0

                    if kind == "dense":
                        beta = pm.Normal(
                            f"beta:{parent}->{node}",
                            mu=0.0,
                            sigma=float(e.get("prior_sigma", 1.0)),
                            dims=(pa["dim_name"], dim_name),
                        )
                        contrib = pt.dot(Xp, beta) * scale_w

                    elif kind == "bym2":
                        W = pa.get("W", None)
                        if W is None:
                            raise ValueError(f"BYM2 edge needs parent '{parent}' node to have W adjacency.")
                        trait_cov = str(e.get("grm_trait_cov", "diag")).lower()
                        sdG = float(e.get("prior_sigma", 1.0))
                    
                        # build L (same as your GRM logic)
                        if trait_cov == "diag":
                            sd_vec = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(dim_name,),
                            )
                            L = pt.diag(sd_vec)
                        elif trait_cov == "scalar":
                            sd_s = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                            )
                            L = sd_s * pt.eye(n_dim)
                        else:
                            chol_packed = pm.LKJCholeskyCov(
                                f"cholG:{parent}->{node}",
                                n=n_dim,
                                eta=float(e.get("eta", 2.0)),
                                sd_dist=_sd_dist(sdG, shape=n_dim),
                                compute_corr=False,
                            )
                            L = pm.expand_packed_triangular(n_dim, chol_packed, lower=True)
                    
                        w_scale = w_par[i] if (w_par is not None) else 1.0
                        L_eff = L * pt.sqrt(w_scale)
                    
                        # BYM2 mixing
                        rho = pm.Beta(f"rho:{parent}->{node}", alpha=1.0, beta=1.0)
                        sigma_bym = pm.HalfNormal(f"sigma_bym:{parent}->{node}", sigma=1.0)
                    
                        cols = []
                        for j in range(n_dim):
                            u = pm.ICAR(
                                f"ICAR_raw:{parent}->{node}:{j}",
                                W=W,
                                sigma=1.0,
                                zero_sum_stdev=float(e.get("icar_zero_sum_stdev", 0.001)),
                                dims=("obs_id",),
                            )
                            v = pm.Normal(f"IID_raw:{parent}->{node}:{j}", mu=0.0, sigma=1.0, dims=("obs_id",))
                            phi = sigma_bym * (pt.sqrt(rho) * u + pt.sqrt(1.0 - rho) * v)
                            cols.append(phi)
                    
                        Phi = pt.stack(cols, axis=1)           # (N, n_dim)
                        contrib = pt.dot(Phi, L_eff.T)         # (N, n_dim)

                    elif kind == "factor":
                        # latent source must exist
                        k = int(pa["shape"][1])
                        p = int(n_dim)
                        if p < k:
                            raise ValueError(f"factor edge requires target dim >= source dim: {parent}->{node}")
                        # identifiable loadings
                        # build mask for first k rows lower-triangular
                        mask = np.ones((p, k), dtype=float)
                        for r in range(min(p, k)):
                            if r + 1 < k:
                                mask[r, (r+1):] = 0.0
                        mask_t = pt.constant(mask)
                        Lambda_raw = pm.Normal(
                            f"Lambda_raw:{parent}->{node}",
                            mu=0.0,
                            sigma=float(e.get("prior_sigma", 1.0)),
                            dims=(dim_name, pa["dim_name"]),  # (traits_dim, factors_dim)
                        )
                        Lambda = Lambda_raw * mask_t
                        if bool(e.get("factor_identifiable", True)):
                            diag = pm.LogNormal(
                                f"Lambda_diag:{parent}->{node}",
                                mu=np.log(max(float(e.get("prior_sigma", 1.0)), 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(pa["dim_name"],),
                            )
                            idx = np.arange(k)
                            Lambda = pt.set_subtensor(Lambda[idx, idx], diag)
                        pm.Deterministic(f"Lambda:{parent}->{node}", Lambda, dims=(dim_name, pa["dim_name"]))
                        contrib = pt.dot(Xp, Lambda.T) * scale_w

                    elif kind == "grm":
                        # PC-space effects
                        r_dim = int(pa["shape"][1])
                        z_prior = str(e.get("z_prior", "iid")).lower()
                        Z_raw = pm.Normal(
                            f"Z_raw:{parent}->{node}",
                            mu=0.0,
                            sigma=1.0,
                            dims=(pa["dim_name"], dim_name),)
                        if z_prior == "iid": Z = Z_raw
                        elif z_prior == "ard_pc":
                            tau = pm.LogNormal(
                                f"tau_pc:{parent}->{node}",
                                mu=np.log(max(float(e.get("prior_sigma", 1.0)), 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(pa["dim_name"],), )
                            Z = Z_raw * tau[:, None]
                        elif z_prior == "rhs":
                            # regularized horseshoe (per PC)
                            # global shrinkage tau, local lambdas
                            tau = pm.HalfCauchy(f"rhs_tau:{parent}->{node}", beta=float(e.get("rhs_global_scale", 0.3)))
                            lam = pm.HalfCauchy(f"rhs_lam:{parent}->{node}", beta=1.0, dims=(pa["dim_name"],))
                            c2 = pm.InverseGamma(f"rhs_c2:{parent}->{node}", alpha=1.0, beta=1.0)
                            lam_tilde = pt.sqrt(c2) * lam / pt.sqrt(c2 + (tau**2) * (lam**2))
                            Z = Z_raw * (tau * lam_tilde)[:, None]
                        else:
                            raise ValueError("z_prior must be iid|ard_pc|rhs")

                        trait_cov = str(e.get("grm_trait_cov","lkj")).lower()
                        sdG = float(e.get("prior_sigma", 1.0))

                        if trait_cov == "lkj":
                            chol_packed = pm.LKJCholeskyCov(
                                f"cholG:{parent}->{node}",
                                n=n_dim,
                                eta=float(e.get("eta", 2.0)),
                                sd_dist=_sd_dist(sdG, shape=n_dim),
                                compute_corr=False,
                            )
                            L = pm.expand_packed_triangular(n_dim, chol_packed, lower=True)
                        elif trait_cov == "diag":
                            sd_vec = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                                dims=(dim_name,),
                            )
                            L = pt.diag(sd_vec)
                        elif trait_cov == "scalar":
                            sd_s = pm.LogNormal(
                                f"sdG:{parent}->{node}",
                                mu=np.log(max(sdG, 1e-8)),
                                sigma=float(sd_prior_sigma),
                            )
                            L = sd_s * pt.eye(n_dim)
                        else:
                            raise ValueError("grm_trait_cov must be lkj|diag|scalar")

                        # apply variance budget scaling consistently to reported SigmaG
                        model.add_coords({dim_name + "_t": coords[dim_name]})
                        w_scale = w_par[i] if (w_par is not None) else 1.0
                        L_eff = L * pt.sqrt(w_scale)
                        pm.Deterministic(f"L:{parent}->{node}", L_eff, dims=(dim_name, dim_name+'_t'))
                        SigmaG = pt.dot(L_eff, L_eff.T)
                        pm.Deterministic(f"SigmaG:{parent}->{node}", SigmaG, dims=(dim_name, dim_name+'_t'))
                        RG = _corr_from_cov(SigmaG)
                        pm.Deterministic(f"RG:{parent}->{node}", RG, dims=(dim_name, dim_name+'_t'))

                        # parameter-implied per-trait genetic variance adjusted by mean diag(K)
                        kdiag_mean = float(pa.get("kdiag_mean", 1.0))
                        Vg_param = pt.diag(SigmaG) * float(kdiag_mean)
                        pm.Deterministic(f"V_G_param:{parent}->{node}", Vg_param, dims=(dim_name,))
                        V_G_param_total = Vg_param if (V_G_param_total is None) else (V_G_param_total + Vg_param)

                        contrib = pt.dot(pt.dot(Xp, Z), L_eff.T)  # already includes w_scale via L_eff

                    else:
                        raise ValueError(f"Unknown edge kind '{kind}' for {parent}->{node}")

                    if bool(a.get("track_values_per_path", False)):
                        pm.Deterministic(f"prediction:{parent}->{node}", contrib, dims=("obs_id", dim_name))
                    if bool(e.get("track_variance", False)):
                        pm.Deterministic(f"V_path_realized:{parent}->{node}", pt.var(contrib, axis=0), dims=(dim_name,))
                    mu = mu + contrib

                # store predictor mean
                if bool(a.get("track_values", False)):
                    pm.Deterministic(f"prediction:{node}", mu, dims=("obs_id", dim_name))
                    pm.Deterministic(f"V_pred:{node}", pt.var(mu, axis=0), dims=(dim_name,))

                # latent node
                if a["type"] == "latent":
                    sigma_lat = float(a.get("prior_sigma", 1.0))
                    offset = pm.Normal(f"{node}:offset", mu=0.0, sigma=sigma_lat, dims=("obs_id", dim_name))
                    lat = pm.Deterministic(node, mu + offset, dims=("obs_id", dim_name))
                    model_vars[node] = lat
                    continue

                # endogenous node likelihood
                if a["type"] != "endo":
                    raise ValueError(f"Unknown node type {a['type']}")

                lik = str(a.get("likelihood","normal")).lower()
                residual = str(a.get("residual","diag")).lower()
                Y = np.asarray(a["data"], dtype=float)

                # missing resolution
                miss_mode = self._resolve_missing_mode(node, Y)

                # ---- NORMAL ----
                if lik == "normal":
                    sdE = float(a.get("sd_E", 1.0))

                    # residual='factor': generative low-rank residual covariance
                    Efac = None
                    if residual == "factor":
                        k = int(a.get("n_factors_E") or min(3, n_dim))
                        k = max(1, min(k, n_dim))
                        # F: per-individual factor scores
                        model.add_coords({f"{node}_Efac": [f"F{i}" for i in range(1, k+1)]})
                        # add coords for factor dim
                        F = pm.Normal(f"{node}:Efac_scores", mu=0.0, sigma=1.0, dims=("obs_id", f"{node}_Efac"))
                        # Loadings W: identifiable lower-triangular (first k rows) + positive diagonal
                        W_raw = pm.Normal(f"{node}:Efac_loadings_raw", mu=0.0, sigma=1.0, shape=(n_dim, k))
                        # zero upper triangle in first k rows
                        W = W_raw
                        for r in range(k):
                            for c in range(r+1, k):
                                W = pt.set_subtensor(W[r, c], 0.0)
                        # positive diagonal
                        diag = pm.LogNormal(f"{node}:Efac_diag", mu=np.log(max(sdE, 1e-8)), sigma=float(sd_prior_sigma), shape=(k,))
                        idx = np.arange(k)
                        W = pt.set_subtensor(W[idx, idx], diag)
                        pm.Deterministic(f"{node}:Efac_loadings", W)
                        Efac = pt.dot(F, W.T)  # (N x n_dim)
                        pm.Deterministic(f"{node}:Efac", Efac, dims=("obs_id", dim_name))

                    # residual scale
                    sigma = pm.LogNormal(
                        f"{node}:sigma",
                        mu=np.log(max(sdE, 1e-8)),
                        sigma=float(sd_prior_sigma),
                        dims=(dim_name,) if n_dim > 1 else None,
                    )
                    if n_dim == 1:
                        sigma = sigma if isinstance(sigma, pt.TensorVariable) else sigma
                        mu1 = (mu[:, 0] + (Efac[:, 0] if Efac is not None else 0.0))
                        y1 = Y[:, 0]
                        if miss_mode in ("mask","pattern"):
                            obs = ~np.isnan(y1)
                            idx = np.where(obs)[0].astype(np.int64)
                            pm.Normal(f"{node}:obs", mu=mu1[idx], sigma=sigma, observed=y1[idx])
                            model_vars[node] = mu1
                        elif miss_mode == "impute":
                            pm.Normal(node, mu=mu1, sigma=sigma, observed=np.ma.masked_invalid(y1), dims=("obs_id",))
                            model_vars[node] = mu1
                        elif miss_mode == "none":
                            pm.Normal(node, mu=mu1, sigma=sigma, observed=y1, dims=("obs_id",))
                            model_vars[node] = mu1
                        else:
                            raise ValueError(f"Unsupported missing mode '{miss_mode}' for univariate normal.")
                    else:
                        mean_total = mu + (Efac if Efac is not None else 0.0)
                        # We cannot reference W outside; rebuild diag if needed:
                        if Efac is None:
                            pm.Deterministic(f"V_E_param:{node}", sigma**2, dims=(dim_name,))
                        else:
                            # loadings W stored as Deterministic, read it back
                            W_det = model[f"{node}:Efac_loadings"]
                            pm.Deterministic(f"V_E_param:{node}", pt.sum(W_det**2, axis=1) + sigma**2, dims=(dim_name,))

                        if V_G_param_total is not None:
                            pm.Deterministic(f"V_G_param_total:{node}", V_G_param_total, dims=(dim_name,))
                            pm.Deterministic(f"h2_param:{node}", V_G_param_total / (V_G_param_total + model[f"V_E_param:{node}"]), dims=(dim_name,))
                        pm.Deterministic(f"V_total_param:{node}", (V_G_param_total if V_G_param_total is not None else 0.0) + model[f"V_E_param:{node}"], dims=(dim_name,))

                        # likelihood
                        if miss_mode == "none":
                            pm.Normal(node, mu=mean_total, sigma=sigma, observed=Y, dims=("obs_id", dim_name))
                            model_vars[node] = mean_total
                        elif miss_mode == "impute":
                            pm.Normal(node, mu=mean_total, sigma=sigma, observed=np.ma.masked_invalid(Y), dims=("obs_id", dim_name))
                            model_vars[node] = mean_total
                        elif miss_mode == "mask":
                            obs = ~np.isnan(Y)
                            rows, cols = np.where(obs)
                            idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                            y_obs = Y[rows, cols].astype(float)
                            mu_flat = mean_total.reshape((-1,))
                            idx_t = pt.constant(idx_flat)
                            mu_obs = mu_flat[idx_t]
                            sigma_obs = sigma[pt.constant(cols.astype(np.int64))]
                            pm.Normal(f"{node}:obs", mu=mu_obs, sigma=sigma_obs, observed=y_obs, shape=y_obs.shape[0])
                            model_vars[node] = mean_total
                        elif miss_mode == "pattern":
                            # for diag residual we still prefer mask (pattern is wasted)
                            obs = ~np.isnan(Y)
                            rows, cols = np.where(obs)
                            idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                            y_obs = Y[rows, cols].astype(float)
                            mu_flat = mean_total.reshape((-1,))
                            mu_obs = mu_flat[pt.constant(idx_flat)]
                            sigma_obs = sigma[pt.constant(cols.astype(np.int64))]
                            pm.Normal(f"{node}:obs", mu=mu_obs, sigma=sigma_obs, observed=y_obs, shape=y_obs.shape[0])
                            model_vars[node] = mean_total
                        else:
                            raise ValueError(f"Unsupported missing mode '{miss_mode}' for normal/diag|factor.")
                    # Explained variance (realized)
                    if bool(a.get("track_values", False)):
                        V_pred = model[f"V_pred:{node}"]
                        V_total_real = V_pred + model[f"V_E_param:{node}"]
                        pm.Deterministic(f"V_total_real:{node}", V_total_real, dims=(dim_name,))
                        for parent in parents:
                            key = f"V_path_realized:{parent}->{node}"
                            if key in model.named_vars:
                                pm.Deterministic(f"EV_path:{parent}->{node}", model[key] / V_total_real, dims=(dim_name,))
                    continue

                # ---- POISSON ----
                if lik == "poisson":
                    rate = pt.exp(pt.clip(mu, -float(poisson_clip), float(poisson_clip))) if str(a.get("link","log")).lower() == "log" else pt.maximum(mu, 1e-12)
                    y = np.asarray(a["data"], dtype=float)
                    if np.isnan(y).any() and miss_mode in ("mask","pattern","row"):
                        obs = ~np.isnan(y)
                        rows, cols = np.where(obs)
                        y_obs = y[rows, cols].astype(int)
                        idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                        rate_flat = rate.reshape((-1,))
                        pm.Poisson(f"{node}:obs", mu=rate_flat[pt.constant(idx_flat)], observed=y_obs, shape=y_obs.shape[0])
                        model_vars[node] = rate
                    else:
                        pm.Poisson(node, mu=rate, observed=y.astype(int), dims=("obs_id", dim_name))
                        model_vars[node] = rate
                    continue

                # ---- BERNOULLI ----
                if lik == "bernoulli":
                    if str(a.get("link","logit")).lower() == "logit":
                        p = pm.math.sigmoid(mu)
                    else:
                        p = pt.clip(mu, 0.0, 1.0)
                    y = np.asarray(a["data"], dtype=float)
                    if np.isnan(y).any() and miss_mode in ("mask","pattern","row"):
                        obs = ~np.isnan(y)
                        rows, cols = np.where(obs)
                        y_obs = y[rows, cols].astype(int)
                        idx_flat = rows.astype(np.int64) * np.int64(n_dim) + cols.astype(np.int64)
                        p_flat = p.reshape((-1,))
                        pm.Bernoulli(f"{node}:obs", p=p_flat[pt.constant(idx_flat)], observed=y_obs, shape=y_obs.shape[0])
                        model_vars[node] = p
                    else:
                        pm.Bernoulli(node, p=p, observed=y.astype(int), dims=("obs_id", dim_name))
                        model_vars[node] = p
                    continue

                raise ValueError(f"Unsupported likelihood '{lik}' for node '{node}'")

        trait_nodes = [n for n,a in self.G.nodes(data=True) if a.get("type")=="endo"]
        self._last_build = _ModelBuildInfo(engine="generic", obs_ids=self.obs_ids, trait_nodes=trait_nodes)
        return model

    # -----------------------------
    # Collapsed marginal-eVD compiler (GCTA/MPH correctness for the simple case)
    # -----------------------------
    def build_model_grm_traits_marginal_evd(
        self,
        *,
        target: str = "traits",
        trait_cov_G: Literal["lkj","diag","scalar","fa"] = "lkj",
        residual: Literal["diag","lkj","fa","scalar"] = "diag",
        n_factors_G: int | None = None,
        n_factors_E: int | None = None,
        eta_G: float = 2.0,
        eta_E: float = 4.0,
        sd_G: float = 1.0,
        sd_E: float = 1.0,
        sd_prior_sigma: float = 0.5,
        jitter: float = 1e-6,
        row_complete: bool = False,
    ) -> pm.Model:
        """
        Collapsed likelihood for simple GRM->traits LMM using eigen-rotation:
          K = U diag(d) U^T
          y*_i ~ MVN(mu*_i, d_i * Sigma_G + Sigma_E)

        This integrates out individual random effects and matches the intended GCTA/MPH LMM math.
        Supports row_complete=True to drop any rows with missing values (approx).
        """
        if target not in self.G:
            raise ValueError(f"target '{target}' not found.")
        targ = self.G.nodes[target]
        if targ.get("type") != "endo" or str(targ.get("likelihood","normal")).lower() != "normal":
            raise ValueError("marginal_evd requires a Normal endo target.")
        if len(list(self.G.successors(target))) != 0:
            raise ValueError("marginal_evd requires target to be terminal (leaf).")
        Y_full = np.asarray(targ["data"], dtype=float)
        if row_complete:
            keep = ~np.isnan(Y_full).any(axis=1)
            if keep.sum() < 5:
                raise ValueError("Too few complete rows after row_complete filtering.")
        else:
            if np.isnan(Y_full).any():
                raise ValueError("marginal_evd requires no missing values unless row_complete=True.")
            keep = np.ones(Y_full.shape[0], dtype=bool)

        parents = list(self.G.predecessors(target))
        grm_parents = [p for p in parents if self.G.nodes[p].get("type") == "grm"]
        exo_parents = [p for p in parents if self.G.nodes[p].get("type") == "exo"]
        if len(grm_parents) != 1:
            raise ValueError("marginal_evd fast path supports exactly one GRM parent.")
        g = grm_parents[0]
        Kg = self.G.nodes[g].get("K_full")
        if Kg is None:
            raise ValueError(f"GRM node '{g}' must be added with store_full=True.")
        K = np.asarray(Kg, dtype=float)
        K = K[np.ix_(keep, keep)]
        Y = Y_full[keep]
        obs_ids = [oid for oid, k in zip(self.obs_ids, keep) if k]

        # EVD
        K = 0.5*(K+K.T)
        d, U = np.linalg.eigh(K)
        order = np.argsort(d)[::-1]
        d = np.clip(d[order], 0.0, np.inf)
        U = U[:, order]
        U_T = U.T.astype(float)
        d_vec = d.astype(float)

        n, t = Y.shape
        dim_name = targ["dim_name"]

        coords = {"obs_id": self._as_str_coords(obs_ids), dim_name: self._as_str_coords(targ["coords"])}

        def _sd_dist(scale: float, shape=None):
            scale = max(float(scale), 1e-8)
            return pm.LogNormal.dist(mu=np.log(scale), sigma=float(sd_prior_sigma), shape=shape)

        # exo data subset
        X_exo = {}
        for p in exo_parents:
            a = self.G.nodes[p]
            X_exo[p] = np.asarray(a["data"], dtype=float)[keep]
            coords[a["dim_name"]] = self._as_str_coords(a["coords"])

        with pm.Model(coords=coords) as model:
            Xp = {p: pm.Data(p, X_exo[p], dims=("obs_id", self.G.nodes[p]["dim_name"])) for p in exo_parents}
            mu = pt.zeros((n, t))
            for p in exo_parents:
                e = self.G.edges[p, target]
                beta = pm.Normal(
                    f"beta:{p}->{target}",
                    mu=0.0,
                    sigma=float(e.get("prior_sigma", 1.0)),
                    dims=(self.G.nodes[p]["dim_name"], dim_name),
                )
                mu = mu + pt.dot(Xp[p], beta)
            pm.Deterministic(f"prediction:{target}", mu, dims=("obs_id", dim_name))

            # Sigma_G
            trait_cov_G = str(trait_cov_G).lower()
            if trait_cov_G == "lkj":
                chol_packed = pm.LKJCholeskyCov(
                    f"cholG:{g}->{target}",
                    n=t,
                    eta=float(eta_G),
                    sd_dist=_sd_dist(sd_G, shape=t),
                    compute_corr=False,
                )
                L_G = pm.expand_packed_triangular(t, chol_packed, lower=True)
            elif trait_cov_G == "diag":
                sdv = pm.LogNormal(f"sdG:{g}->{target}", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma), shape=t)
                L_G = pt.diag(sdv)
            elif trait_cov_G == "scalar":
                s = pm.LogNormal(f"sdG:{g}->{target}", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma))
                L_G = s * pt.eye(t)
            elif trait_cov_G == "fa":
                q = int(n_factors_G or min(3, t))
                # simple identifiable FA chol
                W_raw = pm.Normal("GFA_W_raw", 0.0, 1.0, shape=(t, q))
                W = W_raw
                for r in range(min(t, q)):
                    for c in range(r+1, q):
                        W = pt.set_subtensor(W[r, c], 0.0)
                diag = pm.LogNormal("GFA_diag", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma), shape=(q,))
                idx = np.arange(q)
                W = pt.set_subtensor(W[idx, idx], diag)
                psi = pm.LogNormal("GFA_psi", mu=np.log(max(sd_G, 1e-8)), sigma=float(sd_prior_sigma), shape=(t,))
                SigmaG = pt.dot(W, W.T) + pt.diag(psi) + float(jitter)*pt.eye(t)
                L_G = pt.linalg.cholesky(SigmaG)
            else:
                raise ValueError("trait_cov_G must be lkj|diag|scalar|fa")

            L_G_eff = L_G
            pm.Deterministic(f"L:{g}->{target}", L_G_eff, dims=(dim_name, dim_name))
            SigmaG = pt.dot(L_G_eff, L_G_eff.T)
            pm.Deterministic(f"SigmaG:{g}->{target}", SigmaG, dims=(dim_name, dim_name))
            pm.Deterministic(f"RG:{g}->{target}", _corr_from_cov(SigmaG), dims=(dim_name, dim_name))

            kdiag_mean = float(self.G.nodes[g].get("kdiag_mean", 1.0))
            pm.Deterministic(f"V_G_param:{g}->{target}", pt.diag(SigmaG)*kdiag_mean, dims=(dim_name,))

            # Sigma_E
            residual = str(residual).lower()
            if residual == "diag":
                sdv = pm.LogNormal(f"{target}:sdE", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma), shape=t)
                SigmaE = pt.diag(sdv**2)
            elif residual == "scalar":
                s = pm.LogNormal(f"{target}:sdE", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma))
                SigmaE = (s**2)*pt.eye(t)
            elif residual == "lkj":
                chol_packed = pm.LKJCholeskyCov(
                    f"{target}:cholE",
                    n=t,
                    eta=float(eta_E),
                    sd_dist=_sd_dist(sd_E, shape=t),
                    compute_corr=False,
                )
                L_E = pm.expand_packed_triangular(t, chol_packed, lower=True)
                SigmaE = pt.dot(L_E, L_E.T)
            elif residual == "fa":
                q = int(n_factors_E or min(3, t))
                W_raw = pm.Normal("EFA_W_raw", 0.0, 1.0, shape=(t, q))
                W = W_raw
                for r in range(min(t, q)):
                    for c in range(r+1, q):
                        W = pt.set_subtensor(W[r, c], 0.0)
                diag = pm.LogNormal("EFA_diag", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma), shape=(q,))
                idx = np.arange(q)
                W = pt.set_subtensor(W[idx, idx], diag)
                psi = pm.LogNormal("EFA_psi", mu=np.log(max(sd_E, 1e-8)), sigma=float(sd_prior_sigma), shape=(t,))
                SigmaE = pt.dot(W, W.T) + pt.diag(psi) + float(jitter)*pt.eye(t)
            else:
                raise ValueError("residual must be diag|scalar|lkj|fa")

            pm.Deterministic(f"SigmaE:{target}", SigmaE, dims=(dim_name, dim_name))
            pm.Deterministic(f"V_E_param:{target}", pt.diag(SigmaE), dims=(dim_name,))
            pm.Deterministic(f"h2_param:{target}", pt.diag(SigmaG)*kdiag_mean/(pt.diag(SigmaG)*kdiag_mean + pt.diag(SigmaE)), dims=(dim_name,))

            # rotated residuals
            Uc = pt.constant(U_T)
            Yc = pt.constant(Y)
            resid_mat = Yc - mu
            resid_rot = pt.dot(Uc, resid_mat)  # (n x t)

            covs = pt.constant(d_vec)[:, None, None] * SigmaG[None, :, :] + SigmaE[None, :, :] + float(jitter) * pt.eye(t)[None, :, :]
            Ls = pt.linalg.cholesky(covs)
            z = pt.linalg.solve_triangular(Ls, resid_rot[:, :, None], lower=True)[:, :, 0]
            quad = pt.sum(z**2, axis=1)
            logdet = 2.0 * pt.sum(pt.log(pt.diagonal(Ls, axis1=1, axis2=2)), axis=1)
            logp = -0.5 * (t*np.log(2*np.pi) + logdet + quad)
            pm.Potential(f"{target}:marginal_logp", pt.sum(logp))
            return model

    # -----------------------------
    # Posterior helper utilities
    # -----------------------------
    def genetic_correlation(self, idata, parent: str, target: str, *, summary: Literal["mean","median","none"]="mean", as_dataframe: bool = True):
        key = f"RG:{parent}->{target}"
        R = idata.posterior[key]
        if summary == "mean":
            R = R.mean(dim=("chain","draw"))
        elif summary == "median":
            R = R.median(dim=("chain","draw"))
        if as_dataframe:
            cols = self.G.nodes[target]["coords"]
            return R.to_pandas().set_axis(cols).set_axis(cols, axis=1)
        return R

    def latent_draws(self, idata, node: str, *, n_draws: int | None = None):
        X = idata.posterior[node]
        if n_draws is not None:
            X = X.isel(draw=slice(0, int(n_draws)))
        return X

    def latent_summary(self, idata, node: str, *, summary: Literal["mean","median"]="mean") -> pd.DataFrame:
        X = idata.posterior[node]
        if summary == "mean":
            arr = X.mean(dim=("chain","draw")).to_pandas()
        else:
            arr = X.median(dim=("chain","draw")).to_pandas()
        arr.index = self.obs_ids[:arr.shape[0]]
        return arr

    def blup_draws(self, idata, parent: str, target: str, *, n_draws: int = 100, obs_idx: np.ndarray | None = None) -> np.ndarray:
        """
        Posterior draws of genetic values g = X @ Z @ L^T for a GRM edge.
        Returns array (n_draws, n_obs, n_traits) or subset if obs_idx provided.
        """
        pa = self.G.nodes[parent]
        ta = self.G.nodes[target]
        X = np.asarray(pa["data"], dtype=float)
        if obs_idx is not None:
            X = X[np.asarray(obs_idx, dtype=int)]
        Z = idata.posterior[f"Z_raw:{parent}->{target}"]
        # resolve effective Z according to prior; simplest: use posterior of deterministic? (we didn't store)
        # We stored only Z_raw. If z_prior != iid, scale in post-hoc using stored scales if present.
        z_prior = str(self.G.edges[parent, target].get("z_prior","iid")).lower()
        if z_prior == "iid":
            Z_eff = Z
        elif z_prior == "ard_pc":
            tau = idata.posterior[f"tau_pc:{parent}->{target}"]
            Z_eff = Z * tau[..., :, None]
        elif z_prior == "rhs":
            tau = idata.posterior[f"rhs_tau:{parent}->{target}"]
            lam = idata.posterior[f"rhs_lam:{parent}->{target}"]
            c2  = idata.posterior[f"rhs_c2:{parent}->{target}"]
            lam_tilde = np.sqrt(c2) * lam / np.sqrt(c2 + (tau**2) * (lam**2))
            Z_eff = Z * (tau * lam_tilde)[..., :, None]
        else:
            Z_eff = Z
        L = idata.posterior[f"L:{parent}->{target}"]
        # select draws
        Z_eff = Z_eff.stack(sample=("chain","draw")).isel(sample=slice(0,int(n_draws))).values
        L = L.stack(sample=("chain","draw")).isel(sample=slice(0,int(n_draws))).values
        # compute g: (n_draws, n_obs, n_traits)
        g = np.einsum("nr,drt,dtp->dnp", X, Z_eff, np.transpose(L, (0,2,1)), optimize=True)
        return g

    def posterior_impute(
        self,
        idata,
        node: str,
        *,
        n_draws: int = 200,
        include_residual: bool = True,
        random_seed: int = 0,
    ) -> np.ndarray:
        """
        Sample imputations Y~p(Y|theta) for a node, returning array (n_draws, n_obs, n_dim).
        Works even when missing='mask' because we sample from posterior deterministics + residual params.
        """
        a = self.G.nodes[node]
        Y = np.asarray(a["data"], dtype=float)
        n, t = Y.shape
        rng = np.random.default_rng(int(random_seed))

        mu = idata.posterior[f"prediction:{node}"].stack(sample=("chain","draw")).isel(sample=slice(0, int(n_draws))).values
        # If the likelihood mean included residual factor effects, we stored node: Efac deterministics
        mean_total = mu
        if include_residual and f"{node}:Efac" in idata.posterior:
            efac = idata.posterior[f"{node}:Efac"].stack(sample=("chain","draw")).isel(sample=slice(0, int(n_draws))).values
            mean_total = mean_total + efac

        sigma = idata.posterior[f"{node}:sigma"].stack(sample=("chain","draw")).isel(sample=slice(0, int(n_draws))).values
        if sigma.ndim == 1:
            sigma = sigma[:, None]  # (draw,1)

        out = mean_total.copy()
        if include_residual:
            eps = rng.normal(size=out.shape) * sigma[:, None, :]
            out = out + eps
        # for observed entries, overwrite with observed values if desired? (we keep posterior predictive draw)
        return out

    def explained_variance(self, idata, node: str, *, summary: Literal["mean","median"]="mean") -> pd.DataFrame:
        """
        Return per-trait realized variance diagnostics if present:
          V_pred:<node>, V_E_param:<node>, V_total_real:<node>, h2_param:<node>
        and EV_path:* if available.
        """
        a = self.G.nodes[node]
        cols = a["coords"]
        post = idata.posterior

        def _summ(x):
            if summary == "mean":
                return x.mean(dim=("chain","draw")).to_pandas()
            return x.median(dim=("chain","draw")).to_pandas()

        rows = {}
        for key in [f"V_pred:{node}", f"V_E_param:{node}", f"V_total_real:{node}", f"h2_param:{node}", f"V_total_param:{node}"]:
            if key in post:
                rows[key] = _summ(post[key]).values
        # EV per path
        for p in self.G.predecessors(node):
            key = f"EV_path:{p}->{node}"
            if key in post:
                rows[key] = _summ(post[key]).values
        df = pd.DataFrame(rows, index=cols)
        return df

    def clone(self) -> "CausalGraph":
        cg = CausalGraph(self.df_main)
        cg.obs_ids = list(self.obs_ids)
        cg.G = copy.deepcopy(self.G)
        cg._last_build = copy.deepcopy(self._last_build)
        return cg

    def qc_tabs(
        self,
        fit: Any,
        *,
        hdi_prob: float = 0.95,
        max_text_animals: int = 100,
        max_text_cells: int = 4000,
        top_k_parent: int = 60,
        width: int = 950,
        height: int = 680,
    ):
        """
        Panel Tabs QC view:
          - Graph tab (edge width ~ explained variance)
          - Per-edge tabs: EV heatmap (+ corr heatmap if LKJ/corr present)
          - Per latent/traits node tabs: mean + CI heatmap over animals x dims
    
        Supports:
          - posterior InferenceData (fit.posterior exists)
          - MAP dict (single point; CI collapses to the point)
        """
        import panel as pn
        import holoviews as hv
        import hvplot.pandas  # noqa: F401
        import xarray as xr
    
        hv.extension("bokeh")
    
        def _is_posterior(obj: Any) -> bool:
            return hasattr(obj, "posterior")
    
        def _stack_samples(da: xr.DataArray) -> xr.DataArray:
            if ("chain" in da.dims) and ("draw" in da.dims):
                return da.stack(sample=("chain", "draw"))
            if "sample" in da.dims:
                return da
            # already point estimate
            return da.expand_dims(sample=[0])
    
        def _mean_ci(da: xr.DataArray, prob: float) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
            da = _stack_samples(da)
            q = (1.0 - float(prob)) / 2.0
            mean = da.mean("sample")
            lo = da.quantile(q, dim="sample").drop_vars("quantile", errors="ignore")
            hi = da.quantile(1.0 - q, dim="sample").drop_vars("quantile", errors="ignore")
            return mean, lo, hi
    
        def _heatmap_with_text(
            mean_df: pd.DataFrame,
            lo_df: pd.DataFrame | None,
            hi_df: pd.DataFrame | None,
            *,
            title: str,
            x_name: str,
            y_name: str,
            add_text: bool,
        ):
            long = mean_df.stack().reset_index()
            long.columns = [y_name, x_name, "mean"]
            if (lo_df is not None) and (hi_df is not None):
                lo_long = lo_df.stack().reset_index(drop=True)
                hi_long = hi_df.stack().reset_index(drop=True)
                long["lo"] = lo_long
                long["hi"] = hi_long
                if add_text:
                    long["text"] = long.apply(
                        lambda r: f"{r['mean']:.3g}\n[{r['lo']:.3g},{r['hi']:.3g}]",
                        axis=1,
                    )
            hm = long.hvplot.heatmap(
                x=x_name,
                y=y_name,
                C="mean",
                title=title,
                frame_width=int(width),
                frame_height=520,
                xlabel="",
                ylabel="",
            )
            if add_text and ("text" in long.columns):
                labels = hv.Labels(long, kdims=[x_name, y_name], vdims=["text"]).opts(
                    text_font_size="7pt",
                    text_align="center",
                    text_baseline="middle",
                )
                return hm * labels
            return hm
    
        def chol2cov(L, return_correlation: bool = False, eps: float = 1e-12):
            # expects xarray DataArray with square last-two dims
            variance = (L**2).sum(dim=L.dims[-1])
            stds = np.sqrt(variance)
            L_t = L.rename({L.dims[-2]: "dim_0", L.dims[-1]: "dim_t"})
            cov = xr.dot(L_t, L_t.rename({"dim_0": "dim_1"}), dims="dim_t")
            if not return_correlation:
                return cov
            s0 = stds.rename({stds.dims[-1]: "dim_0"})
            s1 = stds.rename({stds.dims[-1]: "dim_1"})
            denom = s0 * s1
            return cov / (denom + eps)
    
        def chol2covMAP(map_trace: dict[str, Any], name: str, columns, return_correlation: bool = False, eps: float = 1e-12):
            L = np.asarray(map_trace[name])
            cov = L.dot(L.T)
            if not return_correlation:
                return pd.DataFrame(cov, columns=columns, index=columns)
            sig = np.sqrt(np.diag(cov))
            denom = np.outer(sig, sig)
            return pd.DataFrame(cov / (denom + eps), columns=columns, index=columns)
    
        # ---- pick posterior vs MAP ----
        is_post = _is_posterior(fit)
        post = fit.posterior if is_post else None
        map_trace = fit if (not is_post) else None
    
        # ---- explained variance per edge (diag approximation) ----
        edge_ev_scalar: dict[tuple[str, str], float] = {}
        edge_tabs = []
    
        # Build a plotting graph P with EV-based "confidence"
        P = nx.DiGraph()
        for n, attr in self.G.nodes(data=True):
            ntype = str(attr.get("type", "unknown"))
            shape = attr.get("shape", None)
            shape_str = "x".join(map(str, shape)) if shape is not None else "?"
            likelihood = str(attr.get("likelihood", "N/A"))
            P.add_node(n, type=ntype, shape_str=shape_str, likelihood=likelihood)
    
        def _edge_ev_matrix(u: str, v: str, ed: dict[str, Any]):
            kind = str(ed.get("kind", "dense")).lower()
    
            # ------- dense: beta -------
            if kind == "dense":
                bname = f"beta:{u}->{v}"
                X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                if X.ndim != 2:
                    return None, None, None
                varX = X.var(axis=0, ddof=0)
    
                if is_post and (bname in post):
                    beta = post[bname]  # (chain,draw,p,q)
                    # diag approx EV[p,q] = varX[p] * beta[p,q]^2
                    varX_da = xr.DataArray(varX, dims=(beta.dims[-2],), coords={beta.dims[-2]: beta.coords[beta.dims[-2]]})
                    ev = (beta**2) * varX_da[..., None]
                    mean, lo, hi = _mean_ci(ev, hdi_prob)
                    return mean, lo, hi
    
                if (not is_post) and (bname in map_trace):
                    beta = np.asarray(map_trace[bname])
                    ev = (beta**2) * varX[:, None]
                    pcoords = self.G.nodes[u].get("coords", [f"p{i}" for i in range(ev.shape[0])])
                    qcoords = self.G.nodes[v].get("coords", [f"q{i}" for i in range(ev.shape[1])])
                    mean = xr.DataArray(ev, dims=("p", "q"), coords={"p": pcoords, "q": qcoords})
                    lo = mean.copy()
                    hi = mean.copy()
                    return mean, lo, hi
    
                return None, None, None
    
            # ------- factor: Lambda + latent variance -------
            if kind == "factor":
                lname = f"Lambda:{u}->{v}"
                if not is_post:
                    # MAP: just use Lambda^2 * Var(latent) using point estimate (latent may not be stored)
                    if (lname in map_trace) and (u in map_trace):
                        Lam = np.asarray(map_trace[lname])  # (traits, k) by compiler
                        Lat = np.asarray(map_trace[u])      # (obs, k)
                        varLat = Lat.var(axis=0, ddof=0)     # (k,)
                        ev = (Lam**2) * varLat[None, :]
                        # return as (k, traits) for "beta x trait"
                        ev = ev.T
                        kcoords = self.G.nodes[u].get("coords", [f"k{i}" for i in range(ev.shape[0])])
                        tcoords = self.G.nodes[v].get("coords", [f"t{i}" for i in range(ev.shape[1])])
                        mean = xr.DataArray(ev, dims=("k", "t"), coords={"k": kcoords, "t": tcoords})
                        return mean, mean.copy(), mean.copy()
                    return None, None, None
    
                if (lname in post) and (u in post):
                    Lam = post[lname]     # (chain,draw,traits,k)
                    Lat = post[u]         # (chain,draw,obs_id,k)
                    varLat = Lat.var(dim="obs_id")  # (chain,draw,k)
                    # EV[trait,k] = Lam^2 * Var(latent_k)
                    ev_tk = (Lam**2) * varLat[..., None, :]  # broadcast to (chain,draw,traits,k)
                    # to "beta x trait": (k, traits)
                    ev = ev_tk.transpose(*ev_tk.dims[:-2], ev_tk.dims[-1], ev_tk.dims[-2])
                    mean, lo, hi = _mean_ci(ev, hdi_prob)
                    return mean, lo, hi
    
                return None, None, None
    
            # ------- grm: Z (+ shrink) -------
            if kind == "grm":
                zname = f"Z_raw:{u}->{v}"
                z_prior = str(ed.get("z_prior", "iid")).lower()
    
                X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                if X.ndim != 2:
                    return None, None, None
                varX = X.var(axis=0, ddof=0)  # (pc,)
    
                if is_post and (zname in post):
                    Zraw = post[zname]  # (chain,draw,pc,dim)
    
                    Z = Zraw
                    if z_prior == "ard_pc":
                        tname = f"tau_pc:{u}->{v}"
                        if tname in post:
                            tau = post[tname]  # (chain,draw,pc)
                            Z = Zraw * tau[..., :, None]
                    elif z_prior == "rhs":
                        # match compiler: lam_tilde = sqrt(c2)*lam / sqrt(c2 + tau^2*lam^2)
                        tname = f"rhs_tau:{u}->{v}"
                        lname = f"rhs_lam:{u}->{v}"
                        cname = f"rhs_c2:{u}->{v}"
                        if (tname in post) and (lname in post) and (cname in post):
                            tau = post[tname]      # scalar (chain,draw)
                            lam = post[lname]      # (chain,draw,pc)
                            c2 = post[cname]       # scalar (chain,draw)
                            lam_tilde = (np.sqrt(1.0) * pt.sqrt(c2) * lam) / pt.sqrt(c2 + (tau**2) * (lam**2))
                            # xarray-safe: avoid pytensor; do it numerically with xr
                            lam_tilde = (c2**0.5) * lam / (c2 + (tau**2) * (lam**2))**0.5
                            Z = Zraw * (tau * lam_tilde)[..., :, None]
    
                    # EV[pc,dim] = varX[pc] * Z[pc,dim]^2
                    varX_da = xr.DataArray(varX, dims=(Z.dims[-2],), coords={Z.dims[-2]: Z.coords[Z.dims[-2]]})
                    ev = (Z**2) * varX_da[..., None]  # (chain,draw,pc,dim)
    
                    # optionally restrict parent dim for display
                    pc_dim = ev.dims[-2]
                    dim_dim = ev.dims[-1]
                    n_pc = int(ev.sizes[pc_dim])
                    if n_pc > int(top_k_parent):
                        score = _stack_samples(ev).mean("sample").sum(dim_dim)  # (pc,)
                        top_idx = np.argsort(score.values)[::-1][: int(top_k_parent)]
                        ev = ev.isel({pc_dim: top_idx})
    
                    mean, lo, hi = _mean_ci(ev, hdi_prob)
                    return mean, lo, hi
    
                if (not is_post) and (zname in map_trace):
                    Z = np.asarray(map_trace[zname])
                    if z_prior == "ard_pc":
                        tname = f"tau_pc:{u}->{v}"
                        if tname in map_trace:
                            tau = np.asarray(map_trace[tname])
                            Z = Z * tau[:, None]
                    ev = (Z**2) * varX[:, None]
                    # restrict
                    if ev.shape[0] > int(top_k_parent):
                        score = ev.mean(axis=1)
                        top_idx = np.argsort(score)[::-1][: int(top_k_parent)]
                        ev = ev[top_idx, :]
    
                    pcoords = self.G.nodes[u].get("coords", [f"Pc{i}" for i in range(ev.shape[0])])
                    dcoords = self.G.nodes[v].get("coords", [f"d{i}" for i in range(ev.shape[1])])
                    mean = xr.DataArray(ev, dims=("pc", "d"), coords={"pc": pcoords[: ev.shape[0]], "d": dcoords})
                    return mean, mean.copy(), mean.copy()
    
                return None, None, None
    
            return None, None, None
    
        def _edge_corr_matrix(u: str, v: str):
            # prefer RG deterministic if present; else try L -> corr via chol2cov
            rg = f"RG:{u}->{v}"
            l = f"L:{u}->{v}"
    
            if is_post:
                if rg in post:
                    mean, lo, hi = _mean_ci(post[rg], hdi_prob)
                    return mean, lo, hi
                if l in post:
                    corr = chol2cov(post[l], return_correlation=True)
                    mean, lo, hi = _mean_ci(corr, hdi_prob)
                    return mean, lo, hi
                return None, None, None
    
            # MAP
            if (not is_post) and (rg in map_trace):
                C = np.asarray(map_trace[rg])
                cols = self.G.nodes[v].get("coords", [f"d{i}" for i in range(C.shape[0])])
                mean = xr.DataArray(C, dims=("d0", "d1"), coords={"d0": cols, "d1": cols})
                return mean, mean.copy(), mean.copy()
            if (not is_post) and (l in map_trace):
                cols = self.G.nodes[v].get("coords", [f"d{i}" for i in range(np.asarray(map_trace[l]).shape[0])])
                Cdf = chol2covMAP(map_trace, l, columns=cols, return_correlation=True)
                mean = xr.DataArray(Cdf.values, dims=("d0", "d1"), coords={"d0": cols, "d1": cols})
                return mean, mean.copy(), mean.copy()
    
            return None, None, None
    
        # Compute edge EV scalars and build edge tabs
        for u, v, ed in self.G.edges(data=True):
            mean_ev, lo_ev, hi_ev = _edge_ev_matrix(u, v, ed)
            if mean_ev is None:
                edge_ev_scalar[(u, v)] = 0.0
                continue
    
            # scalar EV for graph thickness: sum over matrix
            ev_scalar = float(np.asarray(mean_ev).sum())
            edge_ev_scalar[(u, v)] = ev_scalar
    
            # edge tab content
            # Format EV matrix as DataFrames
            # The EV matrix is "beta_dim x trait_dim" for factor/grm display; for dense it is parent_dim x child_dim.
            ev_mean_df = mean_ev.to_pandas()
            ev_lo_df = lo_ev.to_pandas() if lo_ev is not None else None
            ev_hi_df = hi_ev.to_pandas() if hi_ev is not None else None
    
            add_text = (ev_mean_df.size <= int(max_text_cells))
            ev_plot = _heatmap_with_text(
                ev_mean_df,
                ev_lo_df,
                ev_hi_df,
                title=f"Explained variance: {u} -> {v}",
                x_name="child_dim",
                y_name="beta_dim",
                add_text=add_text,
            )
    
            col = pn.Column(ev_plot)
    
            # Add correlation heatmap if present
            corr_mean, corr_lo, corr_hi = _edge_corr_matrix(u, v)
            if corr_mean is not None:
                cm = corr_mean.to_pandas()
                cl = corr_lo.to_pandas() if corr_lo is not None else None
                ch = corr_hi.to_pandas() if corr_hi is not None else None
                corr_plot = _heatmap_with_text(
                    cm,
                    cl,
                    ch,
                    title=f"Correlation (LKJ/corr): {u} -> {v}",
                    x_name="dim_1",
                    y_name="dim_0",
                    add_text=(cm.size <= int(max_text_cells)),
                )
                col.append(corr_plot)
    
            edge_tabs.append((f"edge: {u}->{v}", col))
    
        # ---- Graph plot scaled by EV ----
        # Build plotting graph with EV-scaled edge "confidence"
        for u, v, ed in self.G.edges(data=True):
            kind = str(ed.get("kind", "dense")).lower()
            ev = float(edge_ev_scalar.get((u, v), 0.0))
            P.add_edge(u, v, kind=kind, ev=ev)
    
        # subsets for layout
        for n in P.nodes():
            out_degree = P.out_degree(n)
            in_degree = P.in_degree(n)
            subset = 0 if in_degree == 0 else (2 if out_degree == 0 else 1)
            P.nodes[n]["subset"] = int(subset)
    
        pos = nx.multipartite_layout(P, subset_key="subset", scale=2.0)
    
        color_map = {"exo": "lightsteelblue", "grm": "salmon", "latent": "palegoldenrod", "endo": "darkseagreen"}
        marker_map = {"exo": "circle", "grm": "hex", "latent": "square", "endo": "square"}
        edge_color_map = {"dense": "steelblue", "grm": "firebrick", "factor": "purple"}
    
        for n in P.nodes():
            t = P.nodes[n]["type"]
            P.nodes[n]["color"] = str(color_map.get(t, "white"))
            P.nodes[n]["marker"] = str(marker_map.get(t, "circle"))
    
        # normalize EV to a bounded "confidence"
        ev_vals = np.array([P.edges[e].get("ev", 0.0) for e in P.edges()], dtype=float)
        ev_max = float(ev_vals.max()) if len(ev_vals) else 1.0
        for u, v in P.edges():
            ev = float(P.edges[u, v].get("ev", 0.0))
            frac = 0.0 if ev_max <= 0 else (ev / ev_max)
            P.edges[u, v]["confidence"] = float(0.1 + 3.9 * np.sqrt(max(frac, 0.0)))
    
        edge_tooltips = [("Edge", "@start -> @end"), ("Kind", "@kind"), ("EV", "@ev")]
        node_tooltips = [("Node", "@index"), ("Type", "@type"), ("Shape", "@shape_str"), ("Likelihood", "@likelihood")]
    
        edge_plot = hvnx.draw_networkx_edges(
            P,
            pos,
            edge_color="kind",
            alpha=0.65,
            arrow_style="-|>",
            arrow_size=5,
            edge_width=4 + hv.dim("confidence").norm() * 15,
            cmap=edge_color_map,
        ).opts(tools=["hover"], hover_tooltips=edge_tooltips)
    
        node_plot = hvnx.draw_networkx_nodes(
            P,
            pos,
            node_color="type",
            node_marker="marker",
            alpha=1.0,
            node_size=5200,
            cmap=color_map,
        ).opts(tools=["hover"], hover_tooltips=node_tooltips)
    
        label_df = pd.DataFrame(
            {"x": [float(pos[n][0]) for n in P.nodes()],
             "y": [float(pos[n][1]) for n in P.nodes()],
             "text": [f"{n}\n{P.nodes[n]['shape_str']}" for n in P.nodes()]}
        )
        labels = hv.Labels(label_df, kdims=["x", "y"], vdims=["text"]).opts(
            text_color="black",
            text_font_size="8pt",
            text_align="center",
            text_baseline="middle",
        )
    
        graph = (edge_plot * node_plot * labels).opts(
            title="Graph (edge width ~ explained variance)",
            width=int(width),
            height=int(height),
            xaxis=None,
            yaxis=None,
            tools=["tap", "box_select", "hover"],
        )
    
        # ---- Node tabs (latent + endo) ----
        node_tabs = []
        for n, a in self.G.nodes(data=True):
            if str(a.get("type")) not in {"latent", "endo"}:
                continue
    
            if not is_post:
                # MAP: try to show point estimate if present
                if n in map_trace:
                    A = np.asarray(map_trace[n])
                    cols = a.get("coords", [f"d{i}" for i in range(A.shape[1])])
                    idx = self.G.nodes.get("obs_id", None)
                    mean_df = pd.DataFrame(A, columns=cols)
                    lo_df = mean_df.copy()
                    hi_df = mean_df.copy()
                    add_text = (mean_df.shape[0] <= int(max_text_animals)) and (mean_df.size <= int(max_text_cells))
                    plot = _heatmap_with_text(
                        mean_df,
                        lo_df,
                        hi_df,
                        title=f"{n} (MAP point)",
                        x_name="dim",
                        y_name="animal",
                        add_text=add_text,
                    )
                    node_tabs.append((f"node: {n}", pn.Column(plot)))
                continue
    
            # posterior case
            if a["type"] == "endo" and (f"prediction:{n}" in post):
                mu = post[f"prediction:{n}"]
                # if residual factor exists, add it
                efac_name = f"{n}:Efac"
                if efac_name in post:
                    mu = mu + post[efac_name]
                target = mu
            else:
                if n not in post:
                    continue
                target = post[n]
    
            mean, lo, hi = _mean_ci(target, hdi_prob)
            mdf = mean.to_pandas()
            ldf = lo.to_pandas()
            hdf = hi.to_pandas()
    
            add_text = (mdf.shape[0] <= int(max_text_animals)) and (mdf.size <= int(max_text_cells))
            plot = _heatmap_with_text(
                mdf,
                ldf,
                hdf,
                title=f"{n}: mean and CI",
                x_name="dim",
                y_name="animal",
                add_text=add_text,
            )
            node_tabs.append((f"node: {n}", pn.Column(plot)))
    
        tabs = pn.Tabs(("graph", graph), *edge_tabs, *node_tabs)
        return tabs
    
    
    def trace_tabs(
        self,
        fit: Any,
        *,
        hdi_prob: float = 0.95,
        top_k_parent: int = 60,
        width: int = 950,
        frame_width: int = 520,
        frame_height: int = 220,
    ):
        """
        Same structure as qc_tabs(), but edge/node tabs show trace-style plots
        (KDE + draw lines) similar to plottrace() / Gcorr_trace().
    
        Only meaningful for posterior InferenceData. For MAP dict, it returns
        a minimal Tabs with a message + graph.
        """
        import panel as pn
        import holoviews as hv
        import hvplot.pandas  # noqa: F401
        import xarray as xr
    
        hv.extension("bokeh")
    
        def _is_posterior(obj: Any) -> bool:
            return hasattr(obj, "posterior")
    
        if not _is_posterior(fit):
            return pn.Tabs(("trace", pn.pane.Markdown("MAP fit has no posterior draws to plot traces.")))
    
        post = fit.posterior
    
        def _stack_samples(da: xr.DataArray) -> xr.DataArray:
            if ("chain" in da.dims) and ("draw" in da.dims):
                return da.stack(sample=("chain", "draw"))
            return da
    
        def _kde_plus_line(df: pd.DataFrame, *, title: str, xlim=None, ylim=None):
            # df: rows=draws, cols=series labels
            long = df.reset_index(names="draw").melt(id_vars="draw", var_name="series", value_name="value")
            kde = long.hvplot.kde(
                by="series",
                y="value",
                title=title,
                alpha=0.25,
                frame_width=int(frame_width),
                frame_height=int(frame_height),
                cut=0,
                xlabel="",
                xlim=xlim,
            ).opts(show_legend=False)
            line = long.hvplot.line(
                by="series",
                x="draw",
                y="value",
                alpha=0.55,
                frame_width=int(frame_width),
                frame_height=int(frame_height),
                xlabel="",
                ylim=ylim,
            ).opts(show_legend=False)
            return (kde + line).opts(toolbar="right").opts(shared_axes=False)
    
        def _corr_trace_plot(corr: xr.DataArray, *, title: str):
            # corr dims: (chain,draw,dim0,dim1) or stacked sample
            C = _stack_samples(corr)
            # take upper triangle pairs
            d0, d1 = C.dims[-2], C.dims[-1]
            labels0 = list(map(str, C.coords[d0].values))
            labels1 = list(map(str, C.coords[d1].values))
            n = len(labels0)
            tri = np.triu_indices(n, k=1)
    
            # build DataFrame: sample x pair
            mat = C.values  # (sample,n,n)
            pairs = []
            cols = []
            for i, j in zip(*tri):
                cols.append(f"{labels0[i]}_{labels1[j]}")
                pairs.append(mat[:, i, j])
            if not cols:
                return pn.pane.Markdown("Correlation matrix too small to plot pairs.")
    
            df = pd.DataFrame(np.vstack(pairs).T, columns=cols)
            return _kde_plus_line(df, title=title, xlim=(-1, 1), ylim=(-1, 1))
    
        # We reuse qc_tabs' graph quickly (no EV width here; user can call qc_tabs for that)
        graph = self.plot_graph(show_edge_kinds=True)
    
        tabs = [("graph", graph)]
    
        # ---- edge tabs ----
        for u, v, ed in self.G.edges(data=True):
            kind = str(ed.get("kind", "dense")).lower()
    
            col = pn.Column()
    
            if kind == "dense":
                bname = f"beta:{u}->{v}"
                if bname in post:
                    B = _stack_samples(post[bname])  # (sample,p,q)
                    # summarize per child dim: sum_p VarX[p] * beta^2
                    X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                    varX = X.var(axis=0, ddof=0)
                    # (sample,q)
                    ev_by_child = (B**2 * xr.DataArray(varX, dims=(B.dims[-2],))).sum(dim=B.dims[-2])
                    df = ev_by_child.to_pandas()
                    col.append(_kde_plus_line(df, title=f"EV by child dim: {u}->{v}"))
    
            elif kind == "factor":
                lname = f"Lambda:{u}->{v}"
                if (lname in post) and (u in post):
                    Lam = _stack_samples(post[lname])  # (sample,traits,k)
                    Lat = _stack_samples(post[u])      # (sample,obs,k)
                    varLat = Lat.var(dim=Lat.dims[-2]) # (sample,k)
                    ev_by_trait = ((Lam**2) * varLat[..., None, :]).sum(dim=Lam.dims[-1])  # (sample,traits)
                    df = ev_by_trait.to_pandas()
                    col.append(_kde_plus_line(df, title=f"EV by trait: {u}->{v}"))
    
            elif kind == "grm":
                zname = f"Z_raw:{u}->{v}"
                z_prior = str(ed.get("z_prior", "iid")).lower()
                if zname in post:
                    Zraw = _stack_samples(post[zname])  # (sample,pc,dim)
                    Z = Zraw
                    if z_prior == "ard_pc":
                        tname = f"tau_pc:{u}->{v}"
                        if tname in post:
                            tau = _stack_samples(post[tname])  # (sample,pc)
                            Z = Zraw * tau[..., :, None]
    
                    X = np.asarray(self.G.nodes[u].get("data"), dtype=float)
                    varX = X.var(axis=0, ddof=0)
    
                    # restrict PCs for plots
                    pc_dim = Z.dims[-2]
                    dim_dim = Z.dims[-1]
                    n_pc = int(Z.sizes[pc_dim])
                    if n_pc > int(top_k_parent):
                        score = (Z**2).mean("sample").sum(dim_dim)
                        top_idx = np.argsort(score.values)[::-1][: int(top_k_parent)]
                        Z = Z.isel({pc_dim: top_idx})
                        varX = varX[top_idx]
    
                    ev_by_dim = ((Z**2) * xr.DataArray(varX, dims=(Z.dims[-2],))[:, None]).sum(dim=Z.dims[-2])  # (sample,dim)
                    df = ev_by_dim.to_pandas()
                    col.append(_kde_plus_line(df, title=f"EV by latent dim (top PCs): {u}->{v}"))
    
            # correlation traces if present
            rg = f"RG:{u}->{v}"
            if rg in post:
                col.append(_corr_trace_plot(post[rg], title=f"Corr traces: {u}->{v}"))
    
            if len(col) == 0:
                col.append(pn.pane.Markdown(f"No trace plots available for edge {u}->{v}."))
    
            tabs.append((f"edge: {u}->{v}", col))
    
        # ---- node tabs ----
        for n, a in self.G.nodes(data=True):
            if str(a.get("type")) not in {"latent", "endo"}:
                continue
            # prefer V_pred if available (already per-dim)
            vname = f"V_pred:{n}"
            if vname in post:
                V = _stack_samples(post[vname])  # (sample,dim)
                tabs.append((f"node: {n}", pn.Column(_kde_plus_line(V.to_pandas(), title=f"V_pred traces: {n}"))))
                continue
    
            # fallback: var over animals of node value (or mu+Efac for endo)
            if a["type"] == "endo" and (f"prediction:{n}" in post):
                mu = post[f"prediction:{n}"]
                efac = f"{n}:Efac"
                if efac in post:
                    mu = mu + post[efac]
                Z = _stack_samples(mu)  # (sample,obs,dim)
            else:
                if n not in post:
                    continue
                Z = _stack_samples(post[n])
    
            # var over obs -> (sample,dim)
            var_over_obs = Z.var(dim=Z.dims[-2])
            tabs.append((f"node: {n}", pn.Column(_kde_plus_line(var_over_obs.to_pandas(), title=f"Var over animals: {n}"))))
    
        return pn.Tabs(*tabs)

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


# -----------------------------
# Example snippets (copy/paste)
# -----------------------------
EXAMPLE_MGREML = r"""
mgreml = genomic_sem_patched5.CausalGraph(df_pheno)
mgreml.add_grm('FullGRM', fgrm, top_eigen2fixed=None, n_components=2000, store_full=True)
mgreml.add_trait('traits', data_cols=df_pheno.columns,
                 track_values=True, track_values_per_path=True,
                 residual='diag', missing='auto')
mgreml.add_edge('FullGRM', 'traits', kind='grm',
                grm_trait_cov='lkj', eta=1.2,
                track_variance=True, prior_variance=0.55,
                adjust_variance_by_shape=False)

mgreml.graph_qc()
mgreml.plot_graph()

# auto chooses generic because missing is elementwise; if you set missing='row' it will choose marginal_evd
model = mgreml.build_model(engine='auto')
"""

EXAMPLE_LATENT_FACTOR = r"""
cg = genomic_sem_patched5.CausalGraph(df_pheno)
cg.add_grm("FullGRM", fgrm, top_eigen2fixed=0.2, n_components=400)
cg.add_grm("TopSNPs", topgrm, subtract_fixed="FullGRM_fixed", n_components=200)
cg.add_latent("OUD_latent", n_dim=6, is_factor=True, prior_sigma=0.2, track_values=True)

# For high missingness, prefer missing='auto' and residual='factor' or 'diag' to avoid pattern explosions
cg.add_trait("traits", data_cols=df_pheno.columns,
             track_values=True, track_values_per_path=False,
             residual="factor", n_factors_E=4,
             missing="auto")

cg.add_edge("FullGRM", "OUD_latent", kind="grm", grm_trait_cov="diag", prior_variance=0.55)
cg.add_edge("TopSNPs", "OUD_latent", kind="grm", grm_trait_cov="diag", prior_variance=0.55)
cg.add_edge("FullGRM_fixed", "traits", kind="dense", prior_variance=0.3)
cg.add_edge("OUD_latent", "traits", kind="factor", prior_variance=0.8)

cg.graph_qc()
cg.plot_graph()

model = cg.build_model(engine="generic")
"""