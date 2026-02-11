import os
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components


# ----------------------------
# Global chart theme helpers
# ----------------------------
def apply_tv_dark(fig, height=360, title=None):
    """
    Apply a TradingView-like dark look to plotly figures.
    """
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=title if title is not None else fig.layout.title.text,
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(color="#e6edf3"),
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.15)",
        tickfont=dict(color="#c9d1d9"),
        title_font=dict(color="#c9d1d9"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.15)",
        tickfont=dict(color="#c9d1d9"),
        title_font=dict(color="#c9d1d9"),
    )
    return fig


def month_label_from_latest_date(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return ""
    return dt.strftime("%b %Y")


# ----------------------------
# Dark HTML table rendering (robust in Streamlit)
# ----------------------------
def render_dark_table(df: pd.DataFrame, height_px: int = 360, index: bool = False):
    """
    Render a truly dark table in Streamlit using HTML + CSS.
    Uses components.html() to reliably render HTML tables across Streamlit versions.
    """
    if df is None or len(df) == 0:
        st.info("No rows to display.")
        return

    df_disp = df.copy()

    # Convert datetimes nicely
    for c in df_disp.columns:
        try:
            if np.issubdtype(df_disp[c].dtype, np.datetime64):
                df_disp[c] = df_disp[c].dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    table_html = df_disp.to_html(index=index, escape=True)

    full_html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        html, body {{
            background: #0b0f14;
            margin: 0;
            padding: 0;
        }}
        .dark-table-container {{
            max-height: {height_px}px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            background: #0b0f14;
        }}
        .dark-table-container table {{
            width: 100%;
            border-collapse: collapse;
            background: #0b0f14;
            color: #e6edf3;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            font-size: 13px;
        }}
        .dark-table-container thead th {{
            position: sticky;
            top: 0;
            background: #0b0f14;
            color: #e6edf3;
            text-align: left;
            padding: 10px 10px;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            z-index: 2;
            white-space: nowrap;
        }}
        .dark-table-container tbody td {{
            padding: 8px 10px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            vertical-align: top;
            white-space: nowrap;
        }}
        .dark-table-container tbody tr:hover td {{
            background: rgba(110,179,255,0.08);
        }}
      </style>
    </head>
    <body>
      <div class="dark-table-container">
        {table_html}
      </div>
    </body>
    </html>
    """

    components.html(full_html, height=height_px + 40, scrolling=False)


# ----------------------------
# Load your ridge engine module
# ----------------------------
def load_engine(py_path: str):
    spec = importlib.util.spec_from_file_location("osp_engine", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# ----------------------------
# Excel loaders / helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_workbook(excel_path: str):
    xl = pd.ExcelFile(excel_path)
    sheets = {name: pd.read_excel(excel_path, sheet_name=name) for name in xl.sheet_names}
    return sheets


def parse_modelling_architecture(df: pd.DataFrame) -> pd.DataFrame:
    """
    The 'Modelling Architecture' sheet is a formatted block.
    We extract the block that starts at the row where a cell equals 'Dimension'.
    """
    df2 = df.copy()
    df2.columns = [str(c) for c in df2.columns]

    hit = None
    for i in range(len(df2)):
        row = df2.iloc[i].astype(str).str.strip().tolist()
        if any(cell == "Dimension" for cell in row):
            hit = i
            break
    if hit is None:
        return pd.DataFrame()

    header = df2.iloc[hit].tolist()
    out = df2.iloc[hit + 1 :].copy()
    out.columns = header
    out = out.dropna(how="all")

    keep_cols = [c for c in out.columns if str(c).strip() not in ["nan", "NaN", "None", ""]]
    out = out[keep_cols]
    return out


def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")


def normalize_cols(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


# ----------------------------
# Arab light deterministic view
# ----------------------------
def arab_light_view(df_al: pd.DataFrame):
    df_al = normalize_cols(df_al)

    d = df_al.copy()
    d["__dt__"] = safe_to_datetime(d.get("Description", pd.Series([None] * len(d))))
    daily = d[d["__dt__"].notna()].copy()

    cash_col_candidates = [c for c in daily.columns if "cash premium" in c.lower()]
    cash_col = cash_col_candidates[0] if cash_col_candidates else None

    monthly = d[d.get("TimeStamp", pd.Series([None] * len(d))).notna()].copy()
    if "TimeStamp" in monthly.columns:
        monthly["TimeStamp"] = pd.to_datetime(monthly["TimeStamp"], errors="coerce")

    forecast_month = ""
    if "TimeStamp" in monthly.columns and monthly["TimeStamp"].notna().any():
        last_ts = monthly["TimeStamp"].dropna().max()
        forecast_month = month_label_from_latest_date(last_ts)

    latest_proj = np.nan
    monthly_pred_col = None
    for candidate in [
        "Predicted change in OSP (before announcement)",
        "Predicted change in OSP",
        "Predicted ΔOSP",
    ]:
        if candidate in monthly.columns:
            monthly_pred_col = candidate
            break

    if monthly_pred_col is not None and len(monthly) > 0:
        m = monthly.copy()
        m[monthly_pred_col] = pd.to_numeric(m[monthly_pred_col], errors="coerce")
        m = m.dropna(subset=["TimeStamp", monthly_pred_col]).sort_values("TimeStamp")
        if len(m) > 0:
            latest_proj = float(m[monthly_pred_col].iloc[-1])

    c1, c2 = st.columns([1.2, 1.0], gap="large")

    with c1:
        st.subheader("signal: Dubai cash premium / backwardation")
        if cash_col and len(daily) > 0:
            daily = daily.sort_values("__dt__")
            fig = px.line(
                daily,
                x="__dt__",
                y=cash_col,
                title="Dubai cash premium (Mo01 - Mo03)",
            )
            fig = apply_tv_dark(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not find daily cash premium series in the Arablight OSP sheet.")

    with c2:
        st.subheader("latest prediction")
        label = f"predicted ΔOSP (t+1) — {forecast_month}" if forecast_month else "predicted ΔOSP (t+1)"
        st.metric(label, value=f"{latest_proj:,.3f}" if np.isfinite(latest_proj) else "NA")

        monthly2 = None
        if (
            len(monthly) > 0
            and "Predicted change in OSP (before announcement)" in monthly.columns
            and "Change in OSP" in monthly.columns
        ):
            monthly2 = monthly.copy()
            monthly2["Change in OSP"] = pd.to_numeric(monthly2["Change in OSP"], errors="coerce")
            monthly2["Predicted change in OSP (before announcement)"] = pd.to_numeric(
                monthly2["Predicted change in OSP (before announcement)"], errors="coerce"
            )
            monthly2 = monthly2.dropna(
                subset=["TimeStamp", "Predicted change in OSP (before announcement)", "Change in OSP"]
            ).sort_values("TimeStamp")

            if len(monthly2) > 0:
                last = monthly2.iloc[-1]
                a = float(last["Change in OSP"])
                p = float(last["Predicted change in OSP (before announcement)"])
                st.metric("last actual ΔOSP", f"{a:,.3f}" if np.isfinite(a) else "NA")
                st.metric("last predicted ΔOSP", f"{p:,.3f}" if np.isfinite(p) else "NA")

        # correlation table
        st.markdown("#### correlations (forecast vs realized ΔOSP)")
        corr_rows = []
        if monthly2 is not None and len(monthly2) >= 3:
            base = monthly2.copy().sort_values("TimeStamp")
            base["y_true"] = base["Change in OSP"]
            base["y_pred"] = base["Predicted change in OSP (before announcement)"]

            def _corr_for(df_):
                x = df_["y_true"].to_numpy(float)
                y = df_["y_pred"].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 3:
                    return np.nan, int(m.sum())
                return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())

            c_all, n_all = _corr_for(base)
            corr_rows.append({"window": "entire history", "corr": c_all, "n": n_all})

            tail24 = base.tail(24)
            c_24, n_24 = _corr_for(tail24)
            corr_rows.append({"window": "last 24 months", "corr": c_24, "n": n_24})

            tail12 = base.tail(12)
            c_12, n_12 = _corr_for(tail12)
            corr_rows.append({"window": "last 12 months", "corr": c_12, "n": n_12})

            corr_df = pd.DataFrame(corr_rows)
            corr_df["corr"] = corr_df["corr"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "NA")
            render_dark_table(corr_df, height_px=170, index=False)
        else:
            st.info("Need monthly realized + forecast series to compute correlations.")

    st.subheader("forecast vs realized: Arab light ΔOSP")
    if (
        len(monthly) > 0
        and "Change in OSP" in monthly.columns
        and "Predicted change in OSP (before announcement)" in monthly.columns
        and "TimeStamp" in monthly.columns
    ):
        comp = monthly.copy()
        comp["Change in OSP"] = pd.to_numeric(comp["Change in OSP"], errors="coerce")
        comp["Predicted change in OSP (before announcement)"] = pd.to_numeric(
            comp["Predicted change in OSP (before announcement)"], errors="coerce"
        )
        comp = comp.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")

        comp_long = comp[["TimeStamp", "Change in OSP", "Predicted change in OSP (before announcement)"]].copy()
        comp_long = comp_long.rename(
            columns={
                "Change in OSP": "realized ΔOSP",
                "Predicted change in OSP (before announcement)": "forecasted ΔOSP",
            }
        )
        comp_long = comp_long.melt(id_vars="TimeStamp", var_name="series", value_name="value")
        comp_long = comp_long.dropna(subset=["value"])

        fig_cmp = px.line(comp_long, x="TimeStamp", y="value", color="series", title="Forecasted vs realized ΔOSP")
        fig_cmp = apply_tv_dark(fig_cmp, height=360)
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Monthly comparison table not found (need TimeStamp + realized + forecast columns).")

    st.subheader("performance: rolling correlation (actual vs predicted ΔOSP)")
    if (
        len(monthly) >= 12
        and "Change in OSP" in monthly.columns
        and "Predicted change in OSP (before announcement)" in monthly.columns
        and "TimeStamp" in monthly.columns
    ):
        monthly3 = monthly.copy().sort_values("TimeStamp")
        y = pd.to_numeric(monthly3["Change in OSP"], errors="coerce")
        yhat = pd.to_numeric(monthly3["Predicted change in OSP (before announcement)"], errors="coerce")
        roll = (
            pd.concat([monthly3["TimeStamp"], y, yhat], axis=1)
            .dropna()
            .assign(roll_corr=lambda z: z[y.name].rolling(12).corr(z[yhat.name]))
        )
        fig2 = px.line(roll, x="TimeStamp", y="roll_corr", title="Rolling 12m correlation")
        fig2 = apply_tv_dark(fig2, height=320)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough monthly history in Arablight OSP sheet to compute rolling correlation.")


# ----------------------------
# Ridge model view (Arabmed / Arabxlt / Arabhvy)
# ----------------------------
def ridge_view(engine, df_in: pd.DataFrame, sheet_name: str):
    df_in = normalize_cols(df_in).copy()

    if "Date" not in df_in.columns:
        st.error(f"Sheet '{sheet_name}' must contain a 'Date' column.")
        return

    df_in["Date"] = pd.to_datetime(df_in["Date"], errors="coerce")
    df_in = df_in.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if df_in.shape[1] < 2:
        st.error(f"Sheet '{sheet_name}' must have at least 2 columns: Date + y.")
        return
    y_col = df_in.columns[1]

    st.caption(f"Using target (y): **{y_col}**")

    st.sidebar.markdown("### ridge parameters")
    alpha1 = st.sidebar.number_input("alpha1 (step-1 ridge)", value=1.0, min_value=0.0, step=0.1)
    alpha2 = st.sidebar.number_input("alpha2 (step-2 ridge)", value=1.0, min_value=0.0, step=0.1)
    win = st.sidebar.number_input("rolling window (win)", value=12, min_value=0, step=1)
    oos_start = st.sidebar.number_input("oos-start", value=60, min_value=0, step=1)

    oos_horizon = st.sidebar.number_input(
        "OOS horizon to display/score (months, 0 = all)",
        value=12,
        min_value=0,
        step=1,
    )

    impute = st.sidebar.selectbox("impute", ["mean", "zero"], index=0)
    bias = st.sidebar.selectbox("bias mode", ["none", "mean", "recent", "anchor"], index=3)
    bias_k = st.sidebar.number_input("bias_k", value=6, min_value=1, step=1)
    bias_cap = st.sidebar.number_input("bias_cap (0 = no cap)", value=0.5, min_value=0.0, step=0.1)

    run = st.sidebar.button("run ridge model", use_container_width=True)

    if not run:
        st.info("Choose parameters on the left and click **run ridge model**.")
        return

    df_model = df_in.copy().set_index("Date")
    for c in df_model.columns:
        if c == y_col:
            continue
        df_model[c] = pd.to_numeric(df_model[c], errors="coerce")

    oos_df, step1_detail, step2_detail, *_rest, summary = engine.rolling_two_step_oos(
        df=df_model,
        date_idx=df_model.index,
        y1_col=y_col,
        y2_col=y_col,
        model1="RIDGE",
        alpha1=float(alpha1),
        model2="RIDGE",
        alpha2=float(alpha2),
        win=int(win),
        oos_start=int(oos_start),
        holdout=0,
        impute=impute,
        drop_cols_step1=[y_col],
        drop_cols_step2=[],
        bias_mode=bias,
        bias_k=int(bias_k),
        bias_cap=None if bias_cap == 0 else float(bias_cap),
    )

    if len(oos_df) == 0:
        st.error("No OOS rows were produced. Check oos-start / win.")
        return

    oos_df = oos_df.copy()
    oos_df["Date"] = pd.to_datetime(oos_df["Date"], errors="coerce")
    oos_df = oos_df.dropna(subset=["Date"]).sort_values("Date")

    if int(oos_horizon) > 0 and len(oos_df) > int(oos_horizon):
        oos_df = oos_df.iloc[-int(oos_horizon) :].copy()

    last = oos_df.iloc[-1]
    forecast_month = month_label_from_latest_date(pd.to_datetime(last["Date"], errors="coerce"))

    y_pred_col = "y2_pred"
    y_true_col = "y2_true"
    dy_pred_col = "y2_pred_delta"
    dy_true_col = "y2_true_delta"

    y_pred = float(last[y_pred_col]) if np.isfinite(last[y_pred_col]) else np.nan
    y_true = float(last[y_true_col]) if np.isfinite(last[y_true_col]) else np.nan
    d_pred = float(last[dy_pred_col]) if np.isfinite(last[dy_pred_col]) else np.nan
    d_true = float(last[dy_true_col]) if np.isfinite(last[dy_true_col]) else np.nan

    c1, c2, c3, c4 = st.columns(4, gap="large")
    c1.metric("predicted level (t)", f"{y_pred:,.3f}" if np.isfinite(y_pred) else "NA")
    label = f"predicted Δ (t) — {forecast_month}" if forecast_month else "predicted Δ (t)"
    c2.metric(label, f"{d_pred:,.3f}" if np.isfinite(d_pred) else "NA")
    c3.metric("actual level (t)", f"{y_true:,.3f}" if np.isfinite(y_true) else "NA")
    c4.metric("actual Δ (t)", f"{d_true:,.3f}" if np.isfinite(d_true) else "NA")

    y_true_arr = pd.to_numeric(oos_df[y_true_col], errors="coerce").to_numpy(float)
    y_pred_arr = pd.to_numeric(oos_df[y_pred_col], errors="coerce").to_numpy(float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)

    disp_corr = float(np.corrcoef(y_true_arr[mask], y_pred_arr[mask])[0, 1]) if mask.sum() > 2 else np.nan
    disp_rmse = float(np.sqrt(np.mean((y_pred_arr[mask] - y_true_arr[mask]) ** 2))) if mask.sum() > 0 else np.nan
    disp_mae = float(np.mean(np.abs(y_pred_arr[mask] - y_true_arr[mask]))) if mask.sum() > 0 else np.nan
    disp_hit = float(np.mean(np.sign(y_true_arr[mask]) == np.sign(y_pred_arr[mask]))) if mask.sum() > 0 else np.nan

    dy_true_arr = pd.to_numeric(oos_df[dy_true_col], errors="coerce").to_numpy(float)
    dy_pred_arr = pd.to_numeric(oos_df[dy_pred_col], errors="coerce").to_numpy(float)
    maskd = np.isfinite(dy_true_arr) & np.isfinite(dy_pred_arr)

    disp_corr_d = float(np.corrcoef(dy_true_arr[maskd], dy_pred_arr[maskd])[0, 1]) if maskd.sum() > 2 else np.nan
    disp_hit_d = float(np.mean(np.sign(dy_true_arr[maskd]) == np.sign(dy_pred_arr[maskd]))) if maskd.sum() > 0 else np.nan

    st.subheader("OOS statistics (for displayed horizon)")
    stats = {
        "OOS window shown (months)": int(oos_horizon) if int(oos_horizon) > 0 else "all",
        "OOS corr (level)": disp_corr,
        "OOS corr (Δ)": disp_corr_d,
        "OOS RMSE (level)": disp_rmse,
        "OOS MAE (level)": disp_mae,
        "OOS sign hit (level)": disp_hit,
        "OOS sign hit (Δ)": disp_hit_d,
    }
    stats_df = pd.DataFrame([stats])
    render_dark_table(stats_df, height_px=140, index=False)

    st.subheader("Δ series: actual vs predicted")
    plot_df = oos_df[["Date", dy_true_col, dy_pred_col]].copy()
    plot_df = plot_df.rename(columns={dy_true_col: "y_true_delta", dy_pred_col: "y_pred_delta"})
    plot_df = plot_df.melt(id_vars="Date", var_name="series", value_name="value")
    fig = px.line(plot_df, x="Date", y="value", color="series", title="Δy (true vs predicted)")
    fig = apply_tv_dark(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("rolling correlation (12m): Δ true vs Δ pred")
    wide = oos_df[["Date", dy_true_col, dy_pred_col]].dropna()
    wide = wide.rename(columns={dy_true_col: "y_true_delta", dy_pred_col: "y_pred_delta"})
    if len(wide) >= 12:
        wide = wide.sort_values("Date")
        wide["roll_corr"] = wide["y_true_delta"].rolling(12).corr(wide["y_pred_delta"])
        fig2 = px.line(wide, x="Date", y="roll_corr", title="Rolling 12m correlation on Δ")
        fig2 = apply_tv_dark(fig2, height=300)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough OOS points (in the selected horizon) to compute 12m rolling correlation.")

    st.subheader("explainability: step-2 coefficients (latest window)")
    latest_date = pd.to_datetime(last["Date"], errors="coerce")

    coefs = step2_detail.copy()
    coefs["Date"] = pd.to_datetime(coefs["Date"], errors="coerce")
    coefs = coefs[coefs["Date"] == latest_date].copy()

    if len(coefs) == 0:
        st.info("No coefficient table available for the latest date.")
        return

    coefs = coefs[coefs["VarName"] != "(none)"]
    coefs = coefs.sort_values("Coef", key=lambda s: np.abs(s), ascending=False)

    show_cols = [c for c in ["VarName", "Coef", "Intercept", "Train_R2", "NumTrain"] if c in coefs.columns]
    render_dark_table(coefs[show_cols], height_px=360, index=False)

    top = coefs.head(12)
    if len(top) > 0 and "Coef" in top.columns and "VarName" in top.columns:
        fig3 = px.bar(top, x="Coef", y="VarName", orientation="h", title="Top coefficients (by |coef|)")
        fig3 = apply_tv_dark(fig3, height=420)
        st.plotly_chart(fig3, use_container_width=True)


# ----------------------------
# Streamlit app
# ----------------------------
st.set_page_config(page_title="OSP modeller", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f14; }
    h1, h2, h3, h4, p, div, label { color: #e6edf3 !important; }

    section[data-testid="stSidebar"] {
        background-color: #0b0f14 !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #e6edf3 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox div,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stTextInput input {
        color: #6fb3ff !important;
    }

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stNumberInput {
        background-color: rgba(255,255,255,0.02) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("OSP modeller")

default_excel = "OSP_modelling_architecture.xlsx"
default_engine = "osp_2stage_2.py"

with st.sidebar:
    st.markdown("## data & engine")
    excel_path = st.text_input("Excel path", value=default_excel)
    engine_path = st.text_input("Python engine path", value=default_engine)

if not os.path.exists(excel_path):
    st.error(f"Excel file not found: {excel_path}")
    st.stop()

if not os.path.exists(engine_path):
    st.error(f"Engine file not found: {engine_path}")
    st.stop()

engine = load_engine(engine_path)
sheets = load_workbook(excel_path)

tab_run, tab_info = st.tabs(["run models", "info / readme"])

with tab_info:
    st.subheader("modelling architecture (from Excel)")

    arch_df = None
    for key in ["Modelling Architecture", "modelling architecture", "MOdelling architecture", "Modelling architecture"]:
        if key in sheets:
            arch_df = sheets[key]
            break

    if arch_df is None:
        st.info("Sheet 'Modelling Architecture' not found.")
        st.write("Available sheets:", list(sheets.keys()))
    else:
        parsed = parse_modelling_architecture(arch_df)
        if len(parsed) == 0:
            st.info("Could not parse the summary block automatically; showing top rows of the sheet.")
            render_dark_table(arch_df.head(60), height_px=420, index=False)
        else:
            render_dark_table(parsed, height_px=420, index=False)

with tab_run:
    st.sidebar.markdown("---")
    model = st.sidebar.selectbox(
        "model",
        [
            "Arab light (deterministic)",
            "Arab med-light (ridge)",
            "Arab xlt-light (ridge)",
            "Arab hvy-light (ridge)",
        ],
        index=0,
    )

    mtime = datetime.fromtimestamp(os.path.getmtime(excel_path)).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Excel source: **{excel_path}** (last modified: {mtime})")

    if model == "Arab light (deterministic)":
        df_al = sheets.get("Arablight OSP")
        if df_al is None:
            st.error("Sheet 'Arablight OSP' not found.")
        else:
            arab_light_view(df_al)

    elif model == "Arab med-light (ridge)":
        df_am = sheets.get("Arabmedlight")
        if df_am is None:
            st.error("Sheet 'Arabmedlight' not found.")
        else:
            ridge_view(engine, df_am, "Arabmedlight")

    elif model == "Arab xlt-light (ridge)":
        df_ax = sheets.get("Arabxltlight")
        if df_ax is None:
            st.error("Sheet 'Arabxltlight' not found.")
        else:
            ridge_view(engine, df_ax, "Arabxltlight")

    else:
        df_ah = sheets.get("Arabhvylight")
        if df_ah is None:
            st.error("Sheet 'Arabhvylight' not found.")
        else:
            ridge_view(engine, df_ah, "Arabhvylight")
