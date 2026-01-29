# osp_2stage_2.py
# Two-step rolling estimation with cross-fitted Step-1 regressor,
# enriched Excel reporting + additive bias options.
# Minimal defensive patch: ensure Step-2 never receives NaN y inputs,
# coerce y to numeric, and emit short diagnostics if a problematic window occurs.

import argparse
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
warnings.filterwarnings("ignore", message=".*Objective did not converge.*")

# ----------------------------- helpers -----------------------------

def normalize_name(s: str) -> str:
    return (
        s.strip()
         .lower()
         .replace("(", " ")
         .replace(")", " ")
         .replace("-", " ")
         .replace("/", " ")
         .replace("%", " ")
         .replace(".", " ")
         .replace(",", " ")
    )

def find_col(df: pd.DataFrame, target_header: str) -> str:
    want = normalize_name(target_header)
    norm_map = {c: normalize_name(str(c)) for c in df.columns}
    for c, nc in norm_map.items():
        if nc == want:
            return c
    for c, nc in norm_map.items():
        if want in nc:
            return c
    raise KeyError(f"Could not find a column close to '{target_header}'. "
                   f"Available: {list(df.columns)}")

def metrics_from_arrays(y_true, y_pred):
    if y_true.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def mape_safe(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    ape = np.abs((y_pred - y_true) / denom)
    return float(np.nanmean(ape))

def smape(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    sm = 2.0 * np.abs(y_pred - y_true) / np.where(denom < 1e-12, np.nan, denom)
    return float(np.nanmean(sm))

def sign_hit(a: np.ndarray, b: np.ndarray):
    sa = np.sign(np.nan_to_num(a, nan=np.nan))
    sb = np.sign(np.nan_to_num(b, nan=np.nan))
    hits = (sa == sb) & ~np.isnan(sa) & ~np.isnan(sb)
    if hits.size == 0:
        return np.nan, np.array([])
    return float(np.mean(hits)), hits

def build_feature_matrix(df: pd.DataFrame, drop_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    keep_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[keep_cols].to_numpy(dtype=float)
    names = list(keep_cols)
    return X, names

def drop_all_nan_columns(X_tr: np.ndarray, X_te: np.ndarray, names: list[str]):
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)
    if X_te.ndim == 1:
        X_te = X_te.reshape(-1, 1)
    all_nan = np.isnan(X_tr).all(axis=0)
    keep = ~all_nan
    if keep.sum() == 0:
        return X_tr, X_te, names
    X_tr = X_tr[:, keep]
    X_te = X_te[:, keep]
    names = [nm for nm, k in zip(names, keep) if k]
    return X_tr, X_te, names

def fit_sklearn(model_name: str, alpha: float, impute_strategy: str):
    if impute_strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    else:
        imputer = SimpleImputer(strategy="mean")

    name = model_name.upper()
    if name == "RIDGE":
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
        steps = [("impute", imputer),
                 ("scale", StandardScaler(with_mean=True, with_std=True)),
                 ("model", model)]
        scaled = True
    elif name == "LASSO":
        model = Lasso(alpha=float(alpha), fit_intercept=True, random_state=0, max_iter=5000)
        steps = [("impute", imputer),
                 ("scale", StandardScaler(with_mean=True, with_std=True)),
                 ("model", model)]
        scaled = True
    elif name == "OLS":
        model = LinearRegression(fit_intercept=True)
        steps = [("impute", imputer),
                 ("model", model)]
        scaled = False
    else:
        raise ValueError("model must be one of: OLS, RIDGE, LASSO")

    return Pipeline(steps), scaled

def step1_oof_preds(
    df: pd.DataFrame,
    tr_idx: np.ndarray,
    drop_cols_step1: list[str],
    y1: np.ndarray,
    model1: str, alpha1: float,
    impute: str,
    win_inner: int
) -> np.ndarray:
    oof = np.full(tr_idx.size, np.nan, dtype=float)
    for j, r in enumerate(tr_idx):
        if win_inner == 0:
            inner_start = tr_idx[0]
        else:
            inner_start = max(tr_idx[0], r - win_inner)
        inner_idx = np.array([i for i in tr_idx if inner_start <= i < r], dtype=int)
        if inner_idx.size < 4:
            continue

        df_in = df.iloc[inner_idx]
        X_in, names = build_feature_matrix(df_in, drop_cols_step1)
        df_r = df.iloc[[r]]
        X_r, _ = build_feature_matrix(df_r, drop_cols_step1)
        if X_r.shape[1] != X_in.shape[1]:
            X_r = df_r[names].to_numpy(dtype=float)

        X_in, X_r, _ = drop_all_nan_columns(X_in, X_r, names)
        if X_in.shape[1] == 0:
            continue

        pipe1, _ = fit_sklearn(model1, alpha1, impute)
        # safe: y1 inner subset forced numeric at call site
        pipe1.fit(X_in, y1[inner_idx])
        oof[j] = float(pipe1.predict(X_r)[0])
    return oof

# ------------------------- core two-step rolling -------------------------

def rolling_two_step_oos(
    df: pd.DataFrame,
    date_idx: pd.Index,
    y1_col: str,
    y2_col: str,
    model1: str, alpha1: float,
    model2: str, alpha2: float,
    win: int,
    oos_start: int,
    holdout: int,
    impute: str,
    drop_cols_step1: list[str],
    drop_cols_step2: list[str],
    bias_mode: str = "none",
    bias_cap: float | None = None,
    bias_k: int = 6,
):
    n = len(df)

    # Coerce y to numeric (force non-numeric -> NaN) to make NaN provenance explicit
    y1 = pd.to_numeric(df[y1_col], errors="coerce").to_numpy(dtype=float)
    y2 = pd.to_numeric(df[y2_col], errors="coerce").to_numpy(dtype=float)

    oos_rows = []
    step1_detail = []
    step2_detail = []
    step1_train_ins_rows = []
    step1_train_oof_rows  = []
    step2_train_ins_rows = []

    for t in range(oos_start, n):
        t0 = 0 if win == 0 else max(0, t - win)
        tr_idx = np.arange(t0, t)
        if tr_idx.size < 4:
            oos_rows.append({
                "Date": date_idx[t],
                "TrainStart": pd.NaT, "TrainEnd": pd.NaT,
                "y1_true": y1[t], "y1_pred": np.nan,
                "y2_true": y2[t], "y2_pred": np.nan,
                "y2_pred_bias": np.nan, "bias_y2": np.nan, "bias_mode_used": bias_mode,
                "Step1_R2_train": np.nan, "Step2_R2_train": np.nan
            })
            continue

        train_start = date_idx[t0]
        train_end   = date_idx[t-1]

        # ----------- STEP 1 -----------
        df_tr1 = df.iloc[tr_idx]
        X1_tr, names1 = build_feature_matrix(df_tr1, drop_cols_step1)
        df_te1 = df.iloc[[t]]
        X1_te, _ = build_feature_matrix(df_te1, drop_cols_step1)
        if X1_te.shape[1] != X1_tr.shape[1]:
            X1_te = df_te1[names1].to_numpy(dtype=float)
        X1_tr, X1_te, names1 = drop_all_nan_columns(X1_tr, X1_te, names1)

        if X1_tr.shape[1] == 0:
            y1_pred_t = np.nan
            step1_r2  = np.nan
            step1_vars = []
            step1_betas = np.array([])
            step1_inter = np.nan
            step1_scaled = (model1.upper() != "OLS")
            y1_hat_tr_ins = np.full(tr_idx.size, np.nan)
        else:
            pipe1, step1_scaled = fit_sklearn(model1, alpha1, impute)
            # safe: y1 already numeric
            # but ensure alignment: use mask where X1_tr rows are finite
            try:
                pipe1.fit(X1_tr, y1[tr_idx])
                y1_hat_tr_ins = pipe1.predict(X1_tr)
            except Exception as e:
                # If fitting step1 fails due to NaN in y1 or X mismatch, produce NaN outputs
                print(f"[WARN] Step1 fit failed at t={t} date={date_idx[t]}: {e}", file=sys.stderr)
                y1_pred_t = np.nan
                step1_r2  = np.nan
                step1_vars = []
                step1_betas = np.array([])
                step1_inter = np.nan
                step1_scaled = (model1.upper() != "OLS")
                y1_hat_tr_ins = np.full(tr_idx.size, np.nan)
            else:
                step1_r2 = float(r2_score(y1[tr_idx], y1_hat_tr_ins)) if len(np.unique(y1[tr_idx]))>1 else np.nan
                y1_pred_t = float(pipe1.predict(X1_te)[0])
                m1 = pipe1.named_steps["model"]
                step1_betas = getattr(m1, "coef_", np.array([]))
                if step1_betas.ndim == 2:
                    step1_betas = step1_betas.ravel()
                step1_inter = float(getattr(m1, "intercept_", np.nan))
                step1_vars = names1

        if len(step1_vars) == 0:
            step1_detail.append({
                "Date": date_idx[t], "TrainStart": train_start, "TrainEnd": train_end,
                "VarName": "(none)", "Coef": np.nan, "Intercept": step1_inter,
                "Train_R2": step1_r2, "Scaled": step1_scaled,
                "Model": model1, "Alpha": alpha1, "NumTrain": int(tr_idx.size)
            })
        else:
            for var, beta in zip(step1_vars, step1_betas):
                step1_detail.append({
                    "Date": date_idx[t], "TrainStart": train_start, "TrainEnd": train_end,
                    "VarName": var, "Coef": float(beta), "Intercept": step1_inter,
                    "Train_R2": step1_r2, "Scaled": step1_scaled,
                    "Model": model1, "Alpha": alpha1, "NumTrain": int(tr_idx.size)
                })

        for idx_obs, yhat_ins in zip(tr_idx, y1_hat_tr_ins):
            step1_train_ins_rows.append({
                "OOS_Date": date_idx[t], "TrainStart": train_start, "TrainEnd": train_end,
                "ObsDate": date_idx[idx_obs], "y1_true": y1[idx_obs], "y1_hat_inSample": yhat_ins
            })

        # ----------- Step-1 OOF for Step-2 training -----------
        y1_hat_oof = step1_oof_preds(
            df=df, tr_idx=tr_idx, drop_cols_step1=drop_cols_step1, y1=y1,
            model1=model1, alpha1=alpha1, impute=impute, win_inner=win
        )
        for idx_obs, yhat_oof in zip(tr_idx, y1_hat_oof):
            step1_train_oof_rows.append({
                "OOS_Date": date_idx[t], "TrainStart": train_start, "TrainEnd": train_end,
                "ObsDate": date_idx[idx_obs], "y1_true": y1[idx_obs], "y1_hat_OOF": yhat_oof
            })

        # ----------- STEP 2 -----------
        df_tr2 = df.iloc[tr_idx].copy()
        df_tr2["__step1_fitted__"] = y1_hat_oof
        drop2_final = list(drop_cols_step2) + [y1_col]
        X2_tr, names2 = build_feature_matrix(df_tr2, drop2_final)

        df_te2 = df.iloc[[t]].copy()
        df_te2["__step1_fitted__"] = y1_pred_t
        X2_te, _ = build_feature_matrix(df_te2, drop2_final)
        if X2_te.shape[1] != X2_tr.shape[1]:
            X2_te = df_te2[names2].to_numpy(dtype=float)
        X2_tr, X2_te, names2 = drop_all_nan_columns(X2_tr, X2_te, names2)

        if X2_tr.shape[1] == 0:
            y2_pred_t = np.nan
            step2_r2  = np.nan
            step2_vars = []
            step2_betas = np.array([])
            step2_inter = np.nan
            step2_scaled = (model2.upper() != "OLS")
            y2_hat_tr_ins = np.full(tr_idx.size, np.nan)
        else:
            # ---------- Defensive patch ----------
            pipe2, step2_scaled = fit_sklearn(model2, alpha2, impute)

            # numeric y vector for training slice
            y2_tr = y2[tr_idx]
            y2_tr_float = np.asarray(y2_tr, dtype=float)

            # create boolean mask: rows where all X columns are finite AND y is finite
            valid_X_rows = np.isfinite(X2_tr).all(axis=1)
            valid_y_rows = np.isfinite(y2_tr_float)
            valid_mask = valid_X_rows & valid_y_rows

            # Diagnostics if something odd
            if valid_mask.sum() < 4:
                # print short diagnostic to stderr to help debug missingness (doesn't stop run)
                print("=== Step-2 skip diagnostic ===", file=sys.stderr)
                print(f"OOS date={date_idx[t]}  t={t}  tr_idx_start={tr_idx[0]} tr_idx_end={tr_idx[-1]}", file=sys.stderr)
                print(f"X2_tr shape={X2_tr.shape}  valid_X_rows={int(valid_X_rows.sum())}/{len(valid_X_rows)}", file=sys.stderr)
                print(f"y2_tr finite={int(valid_y_rows.sum())}/{len(valid_y_rows)}  valid_mask_sum={int(valid_mask.sum())}", file=sys.stderr)
                print("Rows with invalid X but valid y (indices):", file=sys.stderr)
                bad = [int(tr_idx[i]) for i in range(len(tr_idx)) if (not valid_X_rows[i]) and valid_y_rows[i]]
                print(bad, file=sys.stderr)
                print("Rows with invalid y but valid X (indices):", file=sys.stderr)
                bad2 = [int(tr_idx[i]) for i in range(len(tr_idx)) if valid_X_rows[i] and (not valid_y_rows[i])]
                print(bad2, file=sys.stderr)
                # fallback: skip fit for this OOS
                y2_pred_t = np.nan
                step2_r2  = np.nan
                step2_vars = []
                step2_betas = np.array([])
                step2_inter = np.nan
                step2_scaled = (model2.upper() != "OLS")
                y2_hat_tr_ins = np.full(tr_idx.size, np.nan)
            else:
                # fit only on rows where both X and y are valid
                pipe2.fit(X2_tr[valid_mask], y2_tr_float[valid_mask])
                # produce in-sample predictions aligned to tr_idx (fill NaN where invalid)
                y2_hat_tr_ins = np.full(tr_idx.size, np.nan)
                y2_hat_tr_ins[valid_mask] = pipe2.predict(X2_tr[valid_mask])

                step2_r2 = float(r2_score(y2_tr_float[valid_mask], y2_hat_tr_ins[valid_mask])) if len(np.unique(y2_tr_float[valid_mask]))>1 else np.nan
                # predict for t (te) using X2_te (pipeline handles imputation/scaling)
                y2_pred_t = float(pipe2.predict(X2_te)[0])
                m2 = pipe2.named_steps["model"]
                step2_betas = getattr(m2, "coef_", np.array([]))
                if step2_betas.ndim == 2:
                    step2_betas = step2_betas.ravel()
                step2_inter = float(getattr(m2, "intercept_", np.nan))
                step2_vars = names2

        # ----------- Additive bias (level shift) -----------
        bias_val = np.nan
        mode_used = bias_mode.lower()
        y2_pred_t_bias = np.nan

        if np.isfinite(y2_pred_t):
            if mode_used == "mean":
                tr_resid = y2[tr_idx] - y2_hat_tr_ins
                if np.isfinite(tr_resid).any():
                    bias_val = float(np.nanmean(tr_resid))
            elif mode_used == "recent":
                tr_resid = y2[tr_idx] - y2_hat_tr_ins
                if np.isfinite(tr_resid).any():
                    k = max(1, int(bias_k))
                    bias_val = float(np.nanmean(tr_resid[-k:]))
            elif mode_used == "anchor":
                # use last OOS error (previous t) if available
                if len(oos_rows) > 0:
                    prev = oos_rows[-1]
                    if np.isfinite(prev.get("y2_true", np.nan)) and np.isfinite(prev.get("y2_pred", np.nan)):
                        bias_val = float(prev["y2_true"] - prev["y2_pred"])
                # fallback: recent if prev not available
                if not np.isfinite(bias_val):
                    tr_resid = y2[tr_idx] - y2_hat_tr_ins
                    if np.isfinite(tr_resid).any():
                        k = max(1, int(bias_k))
                        bias_val = float(np.nanmean(tr_resid[-k:]))

            if np.isfinite(bias_val):
                if (bias_cap is not None) and np.isfinite(bias_cap):
                    bias_val = float(np.clip(bias_val, -abs(bias_cap), abs(bias_cap)))
                y2_pred_t_bias = float(y2_pred_t + bias_val)

        # ---- details logging for step-2
        if len(step2_vars) == 0:
            step2_detail.append({
                "Date": date_idx[t], "TrainStart": train_start, "TrainEnd": train_end,
                "VarName": "(none)", "Coef": np.nan, "Intercept": step2_inter,
                "Train_R2": step2_r2, "Scaled": step2_scaled,
                "Model": model2, "Alpha": alpha2, "NumTrain": int(tr_idx.size)
            })
        else:
            for var, beta in zip(step2_vars, step2_betas):
                step2_detail.append({
                    "Date": date_idx[t], "TrainStart": train_start, "TrainEnd": train_end,
                    "VarName": var, "Coef": float(beta), "Intercept": step2_inter,
                    "Train_R2": step2_r2, "Scaled": step2_scaled,
                    "Model": model2, "Alpha": alpha2, "NumTrain": int(tr_idx.size)
                })

        for idx_obs, yhat_ins in zip(tr_idx, y2_hat_tr_ins):
            step2_train_ins_rows.append({
                "OOS_Date": date_idx[t], "TrainStart": train_start, "TrainEnd": train_end,
                "ObsDate": date_idx[idx_obs], "y2_true": y2[idx_obs], "y2_hat_inSample": yhat_ins
            })

        oos_rows.append({
            "Date": date_idx[t],
            "TrainStart": train_start, "TrainEnd": train_end,
            "y1_true": y1[t], "y1_pred": y1_pred_t,
            "y2_true": y2[t], "y2_pred": y2_pred_t,
            "y2_pred_bias": y2_pred_t_bias, "bias_y2": bias_val, "bias_mode_used": mode_used,
            "Step1_R2_train": step1_r2, "Step2_R2_train": step2_r2
        })

    # Assemble OOS
    oos_df = pd.DataFrame(oos_rows)

    # ---- Add hits, deltas, errors to OOS ----
    lvl_hit_rate_y1, lvl_hits_y1 = sign_hit(oos_df["y1_true"].to_numpy(), oos_df["y1_pred"].to_numpy())
    lvl_hit_rate_y2, lvl_hits_y2 = sign_hit(oos_df["y2_true"].to_numpy(), oos_df["y2_pred"].to_numpy())
    oos_df["y1_sign_hit"] = lvl_hits_y1
    oos_df["y2_sign_hit"] = lvl_hits_y2

    oos_df["y2_true_delta"] = oos_df["y2_true"].astype(float).diff()
    oos_df["y2_pred_delta"] = oos_df["y2_pred"].astype(float).diff()

    if "y2_pred_bias" in oos_df.columns:
        oos_df["y2_pred_bias_delta"] = oos_df["y2_pred_bias"].astype(float).diff()

    d_hit_rate_y2, d_hits_y2 = sign_hit(oos_df["y2_true_delta"].to_numpy(), oos_df["y2_pred_delta"].to_numpy())
    oos_df["y2_delta_sign_hit"] = d_hits_y2

    oos_df["abs_err_y2"] = np.abs(oos_df["y2_pred"] - oos_df["y2_true"])
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(np.abs(oos_df["y2_true"]) < 1e-12, np.nan, np.abs(oos_df["y2_true"]))
        oos_df["ape_y2"] = np.abs((oos_df["y2_pred"] - oos_df["y2_true"]) / denom)

    # ------------------- OOS metric calc -------------------
    y2_true_all = oos_df["y2_true"].to_numpy(float)
    y2_pred_all = oos_df["y2_pred"].to_numpy(float)
    mask = np.isfinite(y2_true_all) & np.isfinite(y2_pred_all)

    oos_m = metrics_from_arrays(y2_true_all[mask], y2_pred_all[mask])
    if np.sum(mask) > 1 and np.std(y2_true_all[mask]) > 0 and np.std(y2_pred_all[mask]) > 0:
        corr = float(np.corrcoef(y2_true_all[mask], y2_pred_all[mask])[0,1])
    else:
        corr = np.nan
    mape_val = mape_safe(y2_true_all[mask], y2_pred_all[mask])
    smape_val = smape(y2_true_all[mask], y2_pred_all[mask])

    # OOS metrics (biased)
    y2_pred_bias_all = oos_df.get("y2_pred_bias", pd.Series(index=oos_df.index, dtype=float)).to_numpy(float)
    mask_b = np.isfinite(y2_true_all) & np.isfinite(y2_pred_bias_all)

    if np.any(mask_b):
        oos_m_b = metrics_from_arrays(y2_true_all[mask_b], y2_pred_bias_all[mask_b])
        if np.sum(mask_b) > 1 and np.std(y2_true_all[mask_b]) > 0 and np.std(y2_pred_bias_all[mask_b]) > 0:
            corr_b = float(np.corrcoef(y2_true_all[mask_b], y2_pred_bias_all[mask_b])[0,1])
        else:
            corr_b = np.nan
        mape_b = mape_safe(y2_true_all[mask_b], y2_pred_bias_all[mask_b])
        smape_b = smape(y2_true_all[mask_b], y2_pred_bias_all[mask_b])
        lvl_hit_rate_y2_b, _ = sign_hit(y2_true_all[mask_b], y2_pred_bias_all[mask_b])
    else:
        oos_m_b = {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
        corr_b = np.nan
        mape_b = np.nan
        smape_b = np.nan
        lvl_hit_rate_y2_b = np.nan

    summary = {
        "STEP1_R2_mean": float(np.nanmean(oos_df["Step1_R2_train"])) if len(oos_df)>0 else np.nan,
        "STEP2_R2_mean": float(np.nanmean(oos_df["Step2_R2_train"])) if len(oos_df)>0 else np.nan,
        "OOS_RMSE": oos_m["RMSE"],
        "OOS_MAE": oos_m["MAE"],
        "OOS_R2":  oos_m["R2"],
        "OOS_N":   int(np.sum(mask)),
        "OOS_CORR": corr,
        "OOS_MAPE": mape_val,
        "OOS_SMAPE": smape_val,
        "OOS_SignHit_y2": lvl_hit_rate_y2,
        "OOS_SignHitDelta_y2": d_hit_rate_y2,
        "OOS_RMSE_bias": oos_m_b["RMSE"],
        "OOS_MAE_bias": oos_m_b["MAE"],
        "OOS_R2_bias":  oos_m_b["R2"],
        "OOS_CORR_bias": corr_b,
        "OOS_MAPE_bias": mape_b,
        "OOS_SMAPE_bias": smape_b,
        "OOS_SignHit_y2_bias": lvl_hit_rate_y2_b,
    }

    step1_detail = pd.DataFrame(step1_detail)
    step2_detail = pd.DataFrame(step2_detail)
    step1_train_ins = pd.DataFrame(step1_train_ins_rows)
    step1_train_oof = pd.DataFrame(step1_train_oof_rows)
    step2_train_ins = pd.DataFrame(step2_train_ins_rows)

    return oos_df, step1_detail, step2_detail, step1_train_ins, step1_train_oof, step2_train_ins, summary

# ------------------------------- main -------------------------------------

def main():
    p = argparse.ArgumentParser(description="Two-step OSP forecast with cross-fitted Step-1 and rich Excel reporting")
    p.add_argument("--excel", required=True, help="Path to Excel")
    p.add_argument("--sheet", default="Inputs")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--y1", default="ArabMedDiff (monthly change)")
    p.add_argument("--y2", default="Actual_Spread")
    p.add_argument("--model1", default="RIDGE", choices=["OLS","RIDGE","LASSO"])
    p.add_argument("--model2", default="RIDGE", choices=["OLS","RIDGE","LASSO"])
    p.add_argument("--alpha1", type=float, default=1.0)
    p.add_argument("--alpha2", type=float, default=1.0)
    p.add_argument("--win", type=int, default=36)
    p.add_argument("--oos-start", type=int, default=36)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--impute", default="mean", choices=["mean","zero"])
    p.add_argument("--drop1", default="", help="Comma-separated columns to drop in Step-1")
    p.add_argument("--drop2", default="", help="Comma-separated columns to drop in Step-2")
    p.add_argument("--out", default="OSP_2stage_output.xlsx")
    p.add_argument("--tune-wins", default="", help="Comma-separated windows to try; writes *_tuning.xlsx and exits.")
    # bias options
    p.add_argument("--bias", default="none", choices=["none","mean","recent","anchor"],
                   help="Additive bias for Step-2 OOS prediction: mean, recent (last K), or anchor (last OOS error).")
    p.add_argument("--bias-k", type=int, default=6, help="K for recent-bias (default 6).")
    p.add_argument("--bias-cap", type=float, default=None, help="Optional cap for |bias| per window.")

    args = p.parse_args()

    df_raw = pd.read_excel(args.excel, sheet_name=args.sheet)
    if args.date_col in df_raw.columns:
        df_raw.index = pd.to_datetime(df_raw[args.date_col])
    else:
        df_raw.index = pd.RangeIndex(start=0, stop=len(df_raw), step=1)

    y1_col = find_col(df_raw, args.y1)
    y2_col = find_col(df_raw, args.y2)

    drop1 = [c.strip() for c in args.drop1.split(",") if c.strip()]
    drop2 = [c.strip() for c in args.drop2.split(",") if c.strip()]

    tune_list = [int(s) for s in args.tune_wins.split(",") if s.strip().isdigit()]
    if tune_list:
        rows = []
        for w in tune_list:
            (oos_df, step1_detail, step2_detail,
             step1_train_ins, step1_train_oof, step2_train_ins,
             summary) = rolling_two_step_oos(
                df=df_raw, date_idx=df_raw.index,
                y1_col=y1_col, y2_col=y2_col,
                model1=args.model1, alpha1=args.alpha1,
                model2=args.model2, alpha2=args.alpha2,
                win=w, oos_start=args.oos_start, holdout=args.holdout,
                impute=args.impute,
                drop_cols_step1=drop1 + [y1_col, y2_col],
                drop_cols_step2=drop2 + [y2_col],
                bias_mode=args.bias, bias_cap=args.bias_cap, bias_k=args.bias_k,
            )
            rows.append({
                "WIN": w,
                "OOS_RMSE": summary["OOS_RMSE"],
                "OOS_MAE": summary["OOS_MAE"],
                "OOS_R2": summary["OOS_R2"],
                "OOS_CORR": summary["OOS_CORR"],
                "OOS_MAPE": summary["OOS_MAPE"],
                "OOS_SMAPE": summary["OOS_SMAPE"],
                "OOS_SignHit_y2": summary["OOS_SignHit_y2"],
                "OOS_SignHitDelta_y2": summary["OOS_SignHitDelta_y2"],
                "STEP1_R2_mean": summary["STEP1_R2_mean"],
                "STEP2_R2_mean": summary["STEP2_R2_mean"],
                "OOS_RMSE_bias": summary["OOS_RMSE_bias"],
                "OOS_R2_bias": summary["OOS_R2_bias"],
                "OOS_CORR_bias": summary["OOS_CORR_bias"],
            })

        tuning_df = pd.DataFrame(rows).sort_values(["OOS_RMSE", "OOS_R2"], ascending=[True, False])
        out_tune = args.out.replace(".xlsx", f"_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        with pd.ExcelWriter(out_tune, engine="xlsxwriter") as xw:
            tuning_df.to_excel(xw, sheet_name="Tuning", index=False)
        best = tuning_df.iloc[0].to_dict() if len(tuning_df) else {}
        print("Saved tuning comparison to:", out_tune)
        if best:
            print(f"Best WIN by OOS_RMSE: {int(best['WIN'])}  "
                  f"(RMSE={best['OOS_RMSE']:.4f}, R2={best['OOS_R2']:.3f}, CORR={best['OOS_CORR']:.3f})")
        return

    (oos_df, step1_detail, step2_detail,
     step1_train_ins, step1_train_oof, step2_train_ins,
     summary) = rolling_two_step_oos(
        df=df_raw, date_idx=df_raw.index,
        y1_col=y1_col, y2_col=y2_col,
        model1=args.model1, alpha1=args.alpha1,
        model2=args.model2, alpha2=args.alpha2,
        win=args.win, oos_start=args.oos_start, holdout=args.holdout,
        impute=args.impute,
        drop_cols_step1=drop1 + [y1_col, y2_col],
        drop_cols_step2=drop2 + [y2_col],
        bias_mode=args.bias, bias_cap=args.bias_cap, bias_k=args.bias_k,
    )

    settings = pd.DataFrame({
        "Param": ["Excel", "Sheet", "DateCol",
                  "Y1 (Step1 target)", "Y2 (levels target)",
                  "Model1", "Alpha1", "Model2", "Alpha2",
                  "Win", "OOS_start", "Holdout", "Impute",
                  "Drop1", "Drop2", "BiasMode", "BiasK", "BiasCap", "RunTimestamp"],
        "Value": [args.excel, args.sheet, args.date_col,
                  y1_col, y2_col,
                  args.model1, args.alpha1, args.model2, args.alpha2,
                  args.win, args.oos_start, args.holdout, args.impute,
                  ", ".join(drop1), ", ".join(drop2),
                  args.bias, args.bias_k, args.bias_cap,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

    summary_tbl = pd.DataFrame([{
        "MODEL1": args.model1, "ALPHA1": args.alpha1,
        "MODEL2": args.model2, "ALPHA2": args.alpha2,
        "WIN": args.win, "OOS_R2": summary["OOS_R2"],
        "STEP1_R2_mean": summary["STEP1_R2_mean"],
        "STEP2_R2_mean": summary["STEP2_R2_mean"],
        "OOS_RMSE": summary["OOS_RMSE"],
        "OOS_MAE": summary["OOS_MAE"],
        "OOS_MAPE": summary["OOS_MAPE"],
        "OOS_SMAPE": summary["OOS_SMAPE"],
        "OOS_CORR": summary["OOS_CORR"],
        "OOS_SignHit_y2": summary["OOS_SignHit_y2"],
        "OOS_SignHitDelta_y2": summary["OOS_SignHitDelta_y2"],
        "OOS_N": summary["OOS_N"],
        "OOS_R2_bias": summary["OOS_R2_bias"],
        "OOS_RMSE_bias": summary["OOS_RMSE_bias"],
        "OOS_MAE_bias": summary["OOS_MAE_bias"],
        "OOS_CORR_bias": summary["OOS_CORR_bias"],
        "OOS_MAPE_bias": summary["OOS_MAPE_bias"],
        "OOS_SMAPE_bias": summary["OOS_SMAPE_bias"],
        "OOS_SignHit_y2_bias": summary["OOS_SignHit_y2_bias"],
    }])

    with pd.ExcelWriter(args.out, engine="xlsxwriter") as xw:
        settings.to_excel(xw, sheet_name="Settings", index=False)
        summary_tbl.to_excel(xw, sheet_name="Summary", index=False)
        oos_df.to_excel(xw, sheet_name="OOS_Predictions", index=False)
        step1_detail.to_excel(xw, sheet_name="Step1_Detail", index=False)
        step2_detail.to_excel(xw, sheet_name="Step2_Detail", index=False)
        step1_train_ins.to_excel(xw, sheet_name="Step1_TrainInSample", index=False)
        step1_train_oof.to_excel(xw, sheet_name="Step1_TrainOOF", index=False)
        step2_train_ins.to_excel(xw, sheet_name="Step2_TrainInSample", index=False)

        wb  = xw.book
        ws_oos = xxw = xw.sheets["OOS_Predictions"]

        ws_ch = wb.add_worksheet("Charts")
        nrows = len(oos_df)
        cols = {name: i for i, name in enumerate(oos_df.columns)}  # zero-based
        title_fmt = wb.add_format({"bold": True, "font_size": 12})

        # Chart 1: Step1
        ch1 = wb.add_chart({"type": "line"})
        ch1.set_x_axis({"name": "Date"}); ch1.set_y_axis({"name": "Step1 Target & Prediction"})
        ch1.set_title({"name": "Step1: y1_true vs y1_pred"})
        ch1.add_series({
            "name":       ["OOS_Predictions", 0, cols["y1_true"]],
            "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
            "values":     ["OOS_Predictions", 1, cols["y1_true"], nrows, cols["y1_true"]],
        })
        ch1.add_series({
            "name":       ["OOS_Predictions", 0, cols["y1_pred"]],
            "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
            "values":     ["OOS_Predictions", 1, cols["y1_pred"], nrows, cols["y1_pred"]],
        })
        ws_ch.write("A1", "Chart 1: Step1 y1_true vs y1_pred", title_fmt)
        ws_ch.insert_chart("A2", ch1, {"x_scale": 1.3, "y_scale": 1.1})

        # Chart 2: Step2 (add biased series if present)
        ch2 = wb.add_chart({"type": "line"})
        ch2.set_x_axis({"name": "Date"}); ch2.set_y_axis({"name": "Step2 Target & Prediction"})
        ch2.set_title({"name": "Step2: y2_true vs y2_pred"})
        ch2.add_series({
            "name":       ["OOS_Predictions", 0, cols["y2_true"]],
            "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
            "values":     ["OOS_Predictions", 1, cols["y2_true"], nrows, cols["y2_true"]],
        })
        ch2.add_series({
            "name":       ["OOS_Predictions", 0, cols["y2_pred"]],
            "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
            "values":     ["OOS_Predictions", 1, cols["y2_pred"], nrows, cols["y2_pred"]],
        })
        if "y2_pred_bias" in cols:
            ch2.add_series({
                "name":       ["OOS_Predictions", 0, cols["y2_pred_bias"]],
                "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
                "values":     ["OOS_Predictions", 1, cols["y2_pred_bias"], nrows, cols["y2_pred_bias"]],
            })
        ws_ch.write("A20", "Chart 2: Step2 y2_true vs y2_pred (and y2_pred_bias)", title_fmt)
        ws_ch.insert_chart("A21", ch2, {"x_scale": 1.3, "y_scale": 1.1})

        # Chart 3: Δy2
        ch3 = wb.add_chart({"type": "line"})
        ch3.set_x_axis({"name": "Date"}); ch3.set_y_axis({"name": "Change in Spread"})
        ch3.set_title({"name": "Step2: Δy2 (true vs pred)"})
        ch3.add_series({
            "name":       ["OOS_Predictions", 0, cols["y2_true_delta"]],
            "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
            "values":     ["OOS_Predictions", 1, cols["y2_true_delta"], nrows, cols["y2_true_delta"]],
        })
        ch3.add_series({
            "name":       ["OOS_Predictions", 0, cols["y2_pred_delta"]],
            "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
            "values":     ["OOS_Predictions", 1, cols["y2_pred_delta"], nrows, cols["y2_pred_delta"]],
        })
        if "y2_pred_bias_delta" in cols:
            ch3.add_series({
                "name":       ["OOS_Predictions", 0, cols["y2_pred_bias_delta"]],
                "categories": ["OOS_Predictions", 1, cols["Date"], nrows, cols["Date"]],
                "values":     ["OOS_Predictions", 1, cols["y2_pred_bias_delta"], nrows, cols["y2_pred_bias_delta"]],
            })
        ws_ch.write("A38", "Chart 3: Δy2_true vs Δy2_pred (and Δy2_pred_bias)", title_fmt)
        ws_ch.insert_chart("A39", ch3, {"x_scale": 1.3, "y_scale": 1.1})

        ws_ch.write("J2", "Notes:", title_fmt)
        ws_ch.write("J3", "• Sign hits treat +0/-0 as 0; both zero counts as hit.")
        ws_ch.write("J4", "• MAPE ignores months where |y2_true|≈0; sMAPE is symmetric.")
        ws_ch.write("J5", "• Charts reflect only OOS rows starting at --oos-start.")

    print("=== Two-stage run complete ===")
    print(f"Excel: {args.out}")
    for k in ["OOS_RMSE","OOS_R2","OOS_CORR","OOS_RMSE_bias","OOS_R2_bias","OOS_CORR_bias"]:
        print(f"{k}: {summary.get(k)}")

if __name__ == "__main__":
    main()
