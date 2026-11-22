"""
forecast_utils.py
Shared logic for the Dengue-associated pharmaceutical demand forecasting project.
The notebook and the Streamlit / FastAPI apps import this SAME module, so the
features used at training time and at inference time can never drift apart.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

N_LAGS = 8


# ----------------------------------------------------------------------------
# Model wrapper (must live in an importable module so joblib can unpickle it)
# ----------------------------------------------------------------------------
class AverageEnsemble:
    """Simple mean of several fitted sklearn-style regressors."""

    def __init__(self, models):
        self.models = list(models)

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


# ----------------------------------------------------------------------------
# Calendar / feature helpers
# ----------------------------------------------------------------------------
def get_season(month):
    if month in (6, 7, 8, 9):
        return "Southwest_Monsoon"
    if month in (10, 11):
        return "Northeast_Monsoon"
    if month in (12, 1, 2):
        return "Winter"
    return "Summer"


def calendar_features(date, origin):
    date = pd.Timestamp(date)
    month = date.month
    week = int(date.isocalendar().week)
    return {
        "Month_Sin": math.sin(2 * math.pi * month / 12),
        "Month_Cos": math.cos(2 * math.pi * month / 12),
        "Week_Sin": math.sin(2 * math.pi * week / 52.1775),
        "Week_Cos": math.cos(2 * math.pi * week / 52.1775),
        "Week_Index": (date - pd.Timestamp(origin)).days / 7,
        "Season": get_season(month),
    }


def make_feature_row(history, date, origin, static, feature_columns):
    """One feature row from the last 8 observed/forecast weekly values."""
    h = np.asarray(history, dtype=float)
    last = h[-N_LAGS:]
    row = {f"Units_lag{i}": last[-i] for i in range(1, N_LAGS + 1)}
    row["Units_roll4_mean"] = float(last[-4:].mean())
    row["Units_roll8_mean"] = float(last.mean())
    row.update(calendar_features(date, origin))
    row.update(static)
    return {c: row.get(c, np.nan) for c in feature_columns}


def recursive_forecast(model, hist_df, feature_columns, horizon, origin=None):
    """
    Multi-week forecast for every medicine.
    hist_df needs: Medicine_ID, Week_Start, Units_Sold and any static columns the
    model uses (Category, Dosage_Form, Unit_Price_INR). Each predicted week is fed
    back as the lag input of the next week (recursive strategy).
    Returns: Medicine_ID, Forecast_Week, Horizon, Forecast
    """
    hist_df = hist_df.copy()
    hist_df["Medicine_ID"] = hist_df["Medicine_ID"].astype(str)
    hist_df["Week_Start"] = pd.to_datetime(hist_df["Week_Start"])
    hist_df = hist_df.sort_values(["Medicine_ID", "Week_Start"])
    origin = pd.Timestamp(origin) if origin is not None else hist_df["Week_Start"].min()
    last_week = hist_df["Week_Start"].max()
    static_cols = [c for c in ("Category", "Dosage_Form", "Unit_Price_INR") if c in feature_columns]

    state = {}
    for med, g in hist_df.groupby("Medicine_ID"):
        if len(g) < N_LAGS:
            continue
        static = {"Medicine_ID": str(med)}
        for c in static_cols:
            static[c] = g[c].iloc[-1]
        state[med] = {"hist": list(g["Units_Sold"].astype(float).to_numpy()), "static": static}

    out = []
    for h in range(1, horizon + 1):
        date = last_week + pd.Timedelta(days=7 * h)
        meds = list(state)
        X = pd.DataFrame(
            [make_feature_row(state[m]["hist"], date, origin, state[m]["static"], feature_columns) for m in meds]
        )[feature_columns]
        pred = np.clip(model.predict(X), 0, None)
        for m, p in zip(meds, pred):
            state[m]["hist"].append(float(p))
            out.append({"Medicine_ID": m, "Forecast_Week": date, "Horizon": h, "Forecast": float(p)})
    return pd.DataFrame(out)


def add_intervals(fc, resid_q, level_label="80%"):
    """
    Approximate prediction intervals from out-of-fold residual quantiles.
    resid_q: index Medicine_ID (plus optional '__pooled__'), columns q_lo, q_hi
    (quantiles of actual - predicted for 1-step forecasts). Width grows with sqrt(horizon).
    """
    fc = fc.copy()
    pooled = resid_q.loc["__pooled__"] if "__pooled__" in resid_q.index else None

    def bounds(r):
        q = resid_q.loc[r["Medicine_ID"]] if r["Medicine_ID"] in resid_q.index else pooled
        if q is None:
            return pd.Series({"Lower": np.nan, "Upper": np.nan})
        s = math.sqrt(r["Horizon"])
        return pd.Series({"Lower": max(0.0, r["Forecast"] + q["q_lo"] * s), "Upper": r["Forecast"] + q["q_hi"] * s})

    fc[["Lower", "Upper"]] = fc.apply(bounds, axis=1)
    fc.attrs["interval"] = level_label
    return fc


# ----------------------------------------------------------------------------
# Inventory policy
# ----------------------------------------------------------------------------
def inventory_plan(fc, stock, sigma, service_level=0.90, review_weeks=1, demand_multiplier=1.0):
    """
    fc     : output of recursive_forecast (Medicine_ID, Horizon, Forecast)
    stock  : Medicine_ID, On_Hand, On_Order, Lead_Time_Weeks, MOQ [, Medicine_Name, Unit_Cost]
    sigma  : Series indexed by Medicine_ID - one-week forecast error std (RMSE)
    Policy : periodic review, order-up-to level.
      Lead-time demand LTD = sum of forecast over ceil(lead) weeks
      Safety stock        = z * sigma * sqrt(lead + review)
      Reorder point       = LTD + safety stock
      Order-up-to level S = forecast over (lead + review) weeks + safety stock
      Order qty           = S - inventory position, rounded up to MOQ
    """
    z = float(norm.ppf(service_level))
    pooled_sigma = float(sigma.median()) if len(sigma) else 0.0
    rows = []
    for _, s in stock.iterrows():
        med = str(s["Medicine_ID"])
        f = fc[fc["Medicine_ID"] == med].sort_values("Horizon")["Forecast"].to_numpy() * demand_multiplier
        if len(f) == 0:
            continue
        lead = max(1, int(math.ceil(float(s["Lead_Time_Weeks"]))))
        cover = lead + int(review_weeks)
        if cover > len(f):  # pad with the last forecast value
            f = np.concatenate([f, np.repeat(f[-1], cover - len(f))])
        ltd = float(f[:lead].sum())
        sd = float(sigma.get(med, pooled_sigma)) * demand_multiplier
        ss = z * sd * math.sqrt(cover)
        rop = ltd + ss
        order_up_to = float(f[:cover].sum()) + ss
        on_hand = float(s["On_Hand"])
        position = on_hand + float(s["On_Order"])
        moq = max(1, int(s["MOQ"]))
        raw = max(0.0, order_up_to - position) if position <= rop else 0.0
        qty = int(math.ceil(raw / moq) * moq) if raw > 0 else 0
        avg4 = float(f[:4].mean()) if len(f) else np.nan
        if position < ltd:
            status = "Critical - stock-out likely before delivery"
        elif position <= rop:
            status = "Reorder now"
        else:
            status = "OK"
        rows.append({
            "Medicine_ID": med,
            "Medicine_Name": s.get("Medicine_Name", med),
            "On_Hand": on_hand, "On_Order": float(s["On_Order"]),
            "Inventory_Position": position,
            "Lead_Time_Weeks": lead,
            "Lead_Time_Demand": round(ltd, 1),
            "Safety_Stock": round(ss, 1),
            "Reorder_Point": round(rop, 1),
            "Order_Up_To": round(order_up_to, 1),
            "MOQ": moq,
            "Recommended_Order": qty,
            "Weeks_Of_Cover": round(on_hand / avg4, 1) if avg4 and avg4 > 0 else np.nan,
            "Status": status,
            "Est_Order_Cost": round(qty * float(s["Unit_Cost"]), 2) if "Unit_Cost" in stock.columns and pd.notna(s.get("Unit_Cost")) else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Workbook loader used by the apps (same rules as the notebook's cleaning step)
# ----------------------------------------------------------------------------
def load_history_from_workbook(file_or_path, sheet="Master_Weekly"):
    df = pd.read_excel(file_or_path, sheet_name=sheet)
    df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
    df["Units_Sold"] = pd.to_numeric(df["Units_Sold"], errors="coerce")
    df = df.dropna(subset=["Week_Start", "Units_Sold"])
    df = df[df["Units_Sold"] >= 0]
    if "Is_Partial_Week" in df.columns:
        df = df[pd.to_numeric(df["Is_Partial_Week"], errors="coerce").fillna(0).astype(int) == 0]
    df["Medicine_ID"] = df["Medicine_ID"].astype(str)
    return df.sort_values(["Medicine_ID", "Week_Start"]).reset_index(drop=True)
