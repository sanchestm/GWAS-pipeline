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

import re
from copy import deepcopy
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, cross_validate, train_test_split
from tqdm.auto import tqdm

try:  import optuna
except ImportError: optuna = None


class LGBMImputer_optuna:
    def __init__(
        self,
        window=100,
        qc: bool = True,
        device="cpu",
        classifier=None,
        regressor=None,
        silent=False,
        max_iter=1,
        iter_tol=1e-4,
        iter_patience=1,
        use_optuna=False,
        optuna_trials=25,
        optuna_timeout=None,
        tune_n_targets=3,
        tune_sample_frac=0.5,
        early_stopping_rounds=50,
        validation_fraction=0.15,
        random_state=42,
    ):
        self.map_score = {
            True: ["explained_variance", "neg_mean_absolute_error", "r2"],
            False: ["accuracy", "balanced_accuracy", "f1_weighted"],
        }
        self.primary_score = {True: "r2", False: "balanced_accuracy"}
        self.kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        self.model_table = pd.DataFrame()
        self.window = window
        self.device = device
        self.qc = qc
        self.silent = silent
        self.max_iter = max(1, int(max_iter))
        self.iter_tol = float(iter_tol)
        self.iter_patience = int(iter_patience)
        self.use_optuna = use_optuna
        self.optuna_trials = int(optuna_trials)
        self.optuna_timeout = optuna_timeout
        self.tune_n_targets = int(tune_n_targets)
        self.tune_sample_frac = float(tune_sample_frac)
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = float(validation_fraction)
        self.random_state = random_state

        self.classifier = classifier if classifier is not None else LGBMClassifier(
            device=self.device,
            feature_fraction=0.5,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.regressor = regressor if regressor is not None else LGBMRegressor(
            device=self.device,
            feature_fraction=1.0,
            verbosity=-1,
            n_jobs=-1,
            random_state=self.random_state,
        )

        self.needs_rename = False
        self._rename_map = {}
        self._observed_mask = None
        self.iteration_history_ = pd.DataFrame()
        self.best_iteration_ = 1
        self.tuned_params_ = {True: {}, False: {}}

    def _set_rename_map(self, columns):
        safe_cols = pd.Index([re.sub(r"\W", "_", str(c)) for c in columns])
        if safe_cols.duplicated().any():
            dupes = safe_cols[safe_cols.duplicated()].tolist()
            raise ValueError(
                "Column sanitization creates duplicate names. Rename these first: "
                + ", ".join(map(str, dupes))
            )
        self.needs_rename = not columns.equals(safe_cols)
        self._rename_map = dict(zip(columns, safe_cols))
        if self.needs_rename and not self.silent:
            print(r"[warning] \W characters in column names will be replaced with _ when fitting the model")

    def _rename_X(self, X):
        if not self.needs_rename:
            return X
        return X.rename(columns=self._rename_map)

    def _prepare_model_table(self, X, columns_subset=None):
        all_columns = pd.Index(X.columns)
        self.model_table = pd.DataFrame(index=all_columns)
        self.model_table.index.name = "features"
        self.model_table["isnumeric"] = X.dtypes.map(pd.api.types.is_numeric_dtype).astype(bool)
        self.model_table["model"] = self.model_table["isnumeric"].map(
            lambda is_num: clone(self.regressor) if is_num else clone(self.classifier)
        )
        self.model_table["num"] = np.arange(len(self.model_table))
        cols_list = list(all_columns)
        self.model_table["columns"] = [
            cols_list[max(0, i - self.window):i] + cols_list[i + 1:i + 1 + self.window]
            for i in range(len(cols_list))
        ]
        self.model_table["primary_metric"] = self.model_table["isnumeric"].map(self.primary_score)
        self.model_table["constant_pred"] = None

        if columns_subset is not None:
            columns_subset = all_columns[all_columns.isin(columns_subset)]
            if not len(columns_subset):
                if not self.silent:
                    print(r"[warning] subset of columns not present in dataset, using the whole set of columns")
            else:
                self.model_table = self.model_table.loc[columns_subset]

    def _subsample(self, X, y):
        if self.tune_sample_frac >= 1.0 or len(y) <= self.kf.n_splits:
            return X, y
        n = max(self.kf.n_splits, int(np.ceil(len(y) * self.tune_sample_frac)))
        if n >= len(y):
            return X, y
        rng = np.random.RandomState(self.random_state)
        take = np.sort(rng.choice(len(y), size=n, replace=False))
        return X.iloc[take], y.iloc[take]

    def _suggest_lgbm_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def _tune_templates(self, X):
        if not self.use_optuna:
            return
        if optuna is None:
            raise ImportError("optuna is required when use_optuna=True")

        for is_numeric, base_estimator in ((True, self.regressor), (False, self.classifier)):
            target_names = self.model_table.index[self.model_table["isnumeric"].eq(is_numeric)]
            if len(target_names) == 0:
                continue

            observed_counts = X[target_names].notna().sum(axis=0).sort_values(ascending=False)
            target_names = observed_counts.head(self.tune_n_targets).index.tolist()
            primary_metric = self.primary_score[is_numeric]

            def objective(trial):
                params = self._suggest_lgbm_params(trial)
                trial_scores = []
                for yname in target_names:
                    cols = self.model_table.at[yname, "columns"]
                    observed = self._observed_mask[yname]
                    X_fit = X.loc[observed, cols]
                    y_fit = X.loc[observed, yname]

                    if len(y_fit) < self.kf.n_splits:
                        continue
                    if (not is_numeric) and (pd.Series(y_fit).nunique() < 2):
                        continue

                    X_fit, y_fit = self._subsample(X_fit, y_fit)
                    X_fit = self._rename_X(X_fit)
                    est = clone(base_estimator).set_params(**params)

                    cv_res = cross_validate(
                        est,
                        X=X_fit,
                        y=y_fit,
                        scoring=primary_metric,
                        cv=self.kf,
                        return_train_score=False,
                    )
                    trial_scores.append(np.mean(cv_res["test_score"]))

                if not trial_scores:
                    return -np.inf
                return float(np.mean(trial_scores))

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )
            study.optimize(
                objective,
                n_trials=self.optuna_trials,
                timeout=self.optuna_timeout,
                show_progress_bar=not self.silent,
            )
            self.tuned_params_[is_numeric] = study.best_params

            for yname in self.model_table.index[self.model_table["isnumeric"].eq(is_numeric)]:
                self.model_table.at[yname, "model"] = clone(base_estimator).set_params(**study.best_params)

    def _fit_estimator(self, model, X_fit, y_fit, is_numeric):
        if (
            self.early_stopping_rounds is None
            or self.validation_fraction <= 0
            or len(y_fit) < max(20, self.kf.n_splits * 2)
        ):
            return model.fit(X_fit, y_fit)

        y_series = pd.Series(y_fit)
        stratify = None
        if (not is_numeric) and (y_series.nunique() > 1):
            min_count = y_series.value_counts().min()
            if min_count >= 2:
                stratify = y_series

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_fit,
                y_fit,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
            if (not is_numeric) and (pd.Series(y_val).nunique() < 2):
                return model.fit(X_fit, y_fit)

            return model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(self.early_stopping_rounds, verbose=False),
                    log_evaluation(0),
                ],
            )
        except Exception:
            return model.fit(X_fit, y_fit)

    def _compute(self, X, ret, columns_subset=None):
        X_work = X.copy()

        empty_cols = X_work.isna().all(axis=0)
        if empty_cols.any():
            if not self.silent:
                print("[warning] dropping fully empty columns: " + "|".join(X_work.columns[empty_cols]))
            X_work = X_work.loc[:, ~empty_cols]

        self._set_rename_map(X_work.columns)
        self._observed_mask = X_work.notna()
        self._prepare_model_table(X_work, columns_subset=columns_subset)
        self._tune_templates(X_work)

        best_state = None
        best_iter_score = -np.inf
        bad_rounds = 0
        history = []

        for iteration in range(1, self.max_iter + 1):
            iter_scores = []

            for yname in tqdm(self.model_table.index, total=len(self.model_table), disable=self.silent):
                row = self.model_table.loc[yname]
                cols = row["columns"]
                is_numeric = bool(row["isnumeric"])
                observed = self._observed_mask[yname]

                X_fit = X_work.loc[observed, cols]
                y_fit = X_work.loc[observed, yname]
                X_pred = X_work.loc[~observed, cols]

                X_fit_model = self._rename_X(X_fit)
                X_pred_model = self._rename_X(X_pred)

                if len(y_fit) == 0:
                    continue

                if pd.Series(y_fit).nunique(dropna=False) < 2:
                    constant_value = y_fit.iloc[0]
                    self.model_table.at[yname, "constant_pred"] = constant_value
                    self.model_table.at[yname, "model"] = None
                    if len(X_pred_model):
                        X_work.loc[X_pred.index, yname] = constant_value
                    iter_scores.append(1.0)
                    self.model_table.loc[yname, ["n"]] = [len(y_fit)]
                    continue

                self.model_table.at[yname, "constant_pred"] = None
                model = clone(self.model_table.at[yname, "model"])
                model = self._fit_estimator(model, X_fit_model, y_fit, is_numeric)
                self.model_table.at[yname, "model"] = model

                if len(X_pred_model):
                    X_work.loc[X_pred.index, yname] = model.predict(X_pred_model)

                if self.qc:
                    scorers = self.map_score[is_numeric]
                    cv_res = cross_validate(
                        clone(self.model_table.at[yname, "model"]),
                        X=X_fit_model,
                        y=y_fit,
                        scoring=scorers,
                        cv=self.kf,
                        return_train_score=False,
                    )
                    row_qc = {k: float(np.mean(v)) for k, v in cv_res.items()}
                    row_qc["n"] = int(len(y_fit))
                    for k, v in row_qc.items():
                        self.model_table.at[yname, k] = v
                    iter_scores.append(row_qc.get(f'test_{row["primary_metric"]}', np.nan))
                else:
                    primary = row["primary_metric"]
                    score = get_scorer(primary)(model, X_fit_model, y_fit)
                    self.model_table.at[yname, primary] = score
                    self.model_table.at[yname, "n"] = int(len(y_fit))
                    iter_scores.append(score)

            remaining_na = int(X_work.loc[:, self.model_table.index].isna().sum().sum())
            iter_score = float(np.nanmean(iter_scores)) if iter_scores else np.nan
            history.append({"iteration": iteration, "score": iter_score, "remaining_na": remaining_na})

            if self.qc and np.isfinite(iter_score):
                if iter_score > best_iter_score + self.iter_tol:
                    best_iter_score = iter_score
                    self.best_iteration_ = iteration
                    best_state = (X_work.copy(), deepcopy(self.model_table))
                    bad_rounds = 0
                elif iter_score < best_iter_score - self.iter_tol:
                    bad_rounds += 1
                    if bad_rounds > self.iter_patience:
                        if not self.silent:
                            print(f"[info] stopping at iteration {iteration}; best iteration was {self.best_iteration_}")
                        break

            if remaining_na == 0:
                break

        self.iteration_history_ = pd.DataFrame(history)

        if self.qc and best_state is not None:
            X_work, self.model_table = best_state

        if ret == "qc":
            return self.model_table
        if ret == "imputed":
            return X_work
        return self

    def fit(self, X, columns_subset=None):
        return self._compute(X, ret="self", columns_subset=columns_subset)

    def fit_transform(self, X, columns_subset=None):
        return self._compute(X, ret="imputed", columns_subset=columns_subset)

    def transform(self, X):
        X_new = X.copy()
        X_model = self._rename_X(X_new) if self.needs_rename else X_new

        for yname, row in tqdm(self.model_table.iterrows(), total=len(self.model_table), disable=self.silent):
            if yname not in X_new.columns:
                continue

            to_impute = X_new[yname].isna()
            if not to_impute.any():
                continue

            cols = row["columns"]
            constant_pred = row["constant_pred"]

            if constant_pred is not None:
                X_new.loc[to_impute, yname] = constant_pred
                continue

            model = row["model"]
            if model is None:
                continue

            X_pred = X_model.loc[to_impute, cols]
            X_new.loc[to_impute, yname] = model.predict(X_pred)

        return X_new



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


