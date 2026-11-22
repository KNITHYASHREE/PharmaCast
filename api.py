from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

import forecast_utils as fu

ART = Path("artifacts")
bundle = joblib.load(ART / "final_model.joblib")
history = pd.read_csv(ART / "history.csv", parse_dates=["Week_Start"])
history["Medicine_ID"] = history["Medicine_ID"].astype(str)
resid_q = pd.read_csv(ART / "residual_quantiles.csv", index_col=0)

app = FastAPI(title="Pharma Demand Forecast API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["model_name"], "last_week": str(history["Week_Start"].max().date())}


@app.get("/forecast")
def forecast(horizon: int = 4, medicine_id: str | None = None):
    if not 1 <= horizon <= 12:
        raise HTTPException(400, "horizon must be between 1 and 12")
    fc = fu.recursive_forecast(bundle["model"], history, bundle["feature_columns"], horizon, bundle["origin"])
    fc = fu.add_intervals(fc, resid_q)
    if medicine_id:
        fc = fc[fc["Medicine_ID"] == medicine_id]
        if fc.empty:
            raise HTTPException(404, f"Unknown medicine_id {medicine_id}")
    fc["Forecast_Week"] = fc["Forecast_Week"].dt.strftime("%Y-%m-%d")
    return fc.round(2).to_dict(orient="records")
