import os, math, numpy as np, pandas as pd
from dask import delayed, compute
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (r2_score, accuracy_score,
                             explained_variance_score, max_error,
                             mean_absolute_error, balanced_accuracy_score,
                             f1_score)

class DaskOuterLGBMImputer:
    """Fast LightGBM imputer: Dask *process* pool outside, joblib threads inside."""

    # ─────── constants ───────
    _KF = KFold(n_splits=5, shuffle=True, random_state=42)
    _NUM_SCORERS = dict(explained_variance=explained_variance_score,
                        max_error=max_error,
                        neg_mean_absolute_error=lambda y,p:-mean_absolute_error(y,p),
                        r2=r2_score)
    _CAT_SCORERS = dict(accuracy=accuracy_score,
                        balanced_accuracy=balanced_accuracy_score,
                        f1_weighted=lambda y,p:f1_score(y,p,average="weighted"))

    # ─────── constructor ───────
    def __init__(self, *,
                 n_iter=1, atol=1e-4,
                 window=100, qc=False,
                 max_rows=None, block_size=8,
                 early_stopping_rounds=50, feature_fraction=0.6,
                 n_estimators=400,
                 n_workers=None,          # Dask processes
                 device_type="cpu",
                 **lgbm_kw):

        self.n_iter, self.atol   = n_iter, atol
        self.window, self.qc     = window, qc
        self.max_rows            = max_rows
        self.block_size          = block_size
        self.es_rounds           = early_stopping_rounds
        self.n_workers           = n_workers or os.cpu_count()

        self._reg = LGBMRegressor(device_type=device_type,
                                  n_estimators=n_estimators,
                                  n_jobs=1,
                                  feature_fraction=feature_fraction,
                                  early_stopping_rounds=self.es_rounds,
                                  verbosity=-1, **lgbm_kw)
        self._clf = LGBMClassifier(device_type=device_type,
                                   n_estimators=n_estimators,
                                   n_jobs=1,
                                   class_weight="balanced",
                                   feature_fraction=feature_fraction,
                                   early_stopping_rounds=self.es_rounds,
                                   verbosity=-1, **lgbm_kw)

        self.models_      : dict[str, object] = {}
        self.model_table  : pd.DataFrame      = pd.DataFrame()

    # ─────── public API ───────
    def fit_transform(self, X: pd.DataFrame, columns_subset=None):
        if columns_subset is not None:
            bad = set(columns_subset) - set(X.columns)
            if bad: raise KeyError(f"{bad} not in DataFrame")
            tgt_mask = X.columns.isin(columns_subset)
        else:
            tgt_mask = np.ones(len(X.columns), bool)

        out, prev = X.copy(), None
        for it in range(self.n_iter):
            if it: print(f"[Imputer] pass {it+1}/{self.n_iter}")
            out = self._single_pass(out, tgt_mask)
            if prev is not None and np.allclose(prev.values, out.values,
                                                atol=self.atol, equal_nan=True):
                print(f"[Imputer] converged at pass {it+1}")
                break
            prev = out.copy()
        return out

    def fit(self, X, columns_subset=None):
        self.fit_transform(X, columns_subset); return self

    def transform(self, X):
        X2 = X.copy()
        for feat,row in self.model_table.iterrows():
            miss = X2[feat].isna()
            if miss.any():
                X2.loc[miss, feat] = self.models_[feat].predict(X2.loc[miss, row["columns"]])
        return X2

    # ─────── helpers ───────
    def _wins(self, p):
        return [np.r_[max(0,i-self.window):i, i+1:min(p,i+self.window+1)]
                for i in range(p)]

    @staticmethod
    def _tr_val(n):
        if n < 10: return slice(None), slice(0,0)
        c = math.ceil(0.8*n); return slice(0,c), slice(c,n)

    # ─────── single Dask-process pass ───────
    def _single_pass(self, Xdf, tgt_mask):
        X     = Xdf.to_numpy(np.float32, copy=True)
        miss  = np.isnan(X); p = X.shape[1]
        wins  = self._wins(p)
        isnum = Xdf.dtypes.apply(lambda dt: dt.kind in "fi").to_numpy()
        names = Xdf.columns.to_numpy()

        targets = np.flatnonzero(tgt_mask)
        num  = [i for i in targets if isnum[i]]
        cat  = [i for i in targets if not isnum[i]]

        blocks = [num[i:i+self.block_size] for i in range(0,len(num),self.block_size)]
        blocks += [[j] for j in cat]

        delayed_tasks = [delayed(self._proc_block)(blk,X,miss,wins,isnum,names)
                         for blk in blocks]

        results = compute(*delayed_tasks, scheduler="processes",
                          num_workers=self.n_workers)

        recs = []
        for res in results:
            if res is None: continue
            for tup in res:
                name, model, col_list, tr_sc, qc, idx_missing, preds = tup
                if idx_missing.size:
                    X[idx_missing, names==name] = preds
                self.models_[name] = model
                row = dict(feature=name, isnumeric=isnum[names==name][0],
                           model=model, num=int(np.where(names==name)[0][0]),
                           columns=col_list, train_score=tr_sc)
                if qc: row.update(qc)
                recs.append(row)

        self.model_table = (pd.DataFrame(recs)
                              .set_index("feature")
                              .reindex(names[targets]))
        return pd.DataFrame(X, index=Xdf.index, columns=Xdf.columns)

    # ─────── worker: fit block (process) ───────
    def _proc_block(self, idx_block, X, miss, wins, isnum, names):
        cols = np.unique(np.concatenate([wins[j] for j in idx_block]))
        rows = np.flatnonzero(~miss[:, idx_block].any(axis=1))
        if rows.size == 0: return None
        if self.max_rows and rows.size > self.max_rows:
            rows = np.random.choice(rows, self.max_rows, replace=False)

        Xtr = X[rows][:, cols]

        if len(idx_block) == 1:                      # single target
            j  = idx_block[0]
            y  = X[rows, j]
            tr, val = self._tr_val(len(rows))
            base = clone(self._reg if isnum[j] else self._clf)
            fit_kw = {"eval_set":[(Xtr[val], y[val])]} if val.stop else {}
            base.fit(Xtr[tr], y[tr], **fit_kw)
            tr_sc = (r2_score if isnum[j] else accuracy_score)(y, base.predict(Xtr))
            qc = self._cheap_cv(base, Xtr, y, isnum[j]) if self.qc else None
            idx_miss = np.flatnonzero(miss[:,j])
            preds = base.predict(X[idx_miss][:, cols]) if idx_miss.size else np.array([])
            return [(names[j], base, list(names[cols]), tr_sc, qc, idx_miss, preds)]

        # numeric multi-output block
        Ytr = X[rows][:, idx_block]
        est = MultiOutputRegressor(clone(self._reg), n_jobs=min(4,len(idx_block)))
        est.fit(Xtr, Ytr)                          # no eval_set => avoids 2-D label error
        preds_tr = est.predict(Xtr)

        out = []
        for k,j in enumerate(idx_block):
            tr_sc = r2_score(Ytr[:,k], preds_tr[:,k])
            qc = self._cheap_cv(est.estimators_[k], Xtr, Ytr[:,k], True) if self.qc else None
            idx_miss = np.flatnonzero(miss[:,j])
            full_pred = est.predict(X[:, cols])[:,k] if idx_miss.size else np.array([])
            out.append((names[j], est.estimators_[k],
                        list(names[cols]), tr_sc, qc,
                        idx_miss, full_pred))
        return out

    # ─────── cheap no-refit CV ───────
    def _cheap_cv(self, est, Xtr, y, is_num):
        scorers = self._NUM_SCORERS if is_num else self._CAT_SCORERS
        agg = {k:0. for k in scorers}
        for tr, te in self._KF.split(Xtr):
            pr = est.predict(Xtr[te])
            for nm, fn in scorers.items():
                agg[nm] += fn(y[te], pr)
        return {k:v/self._KF.get_n_splits() for k,v in agg.items()}

