
# nisin_ecoli_fit_aw.py
import pandas as pd
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

CSV_PATH = Path("C:/Users/kaoru/iGEM/mnt/data/nisin_ecoli_enriched.csv")

@dataclass
class FitResult:
    params: Dict[str, float]
    success: bool
    message: str

def R_weibull(t, Cug, pH, aw, Tgrow, PretT, PretMin, WarmT, WarmMin,
              b0, bC, bPH2, bAW, bTg, bPrT, bPrM, bWT, bWM, p_shape):
    eps = 1e-9
    lnC = math.log(max(Cug, eps))
    pHterm = ((pH-6.0) if not np.isnan(pH) else 0.0)**2
    awterm = ((aw-0.98) if not np.isnan(aw) else 0.0)
    TgrowTerm = (Tgrow-37.0) if not np.isnan(Tgrow) else 0.0
    PretTterm = (PretT-25.0) if not np.isnan(PretT) else 0.0
    PretMterm = PretMin if not np.isnan(PretMin) else 0.0
    WarmTterm = (WarmT-25.0) if not np.isnan(WarmT) else 0.0
    WarmMterm = WarmMin if not np.isnan(WarmMin) else 0.0
    delta = math.exp(b0 + bC*lnC + bPH2*pHterm + bAW*awterm + bTg*TgrowTerm +
                     bPrT*PretTterm + bPrM*PretMterm + bWT*WarmTterm + bWM*WarmMterm)
    return (max(t, eps) / max(delta, eps)) ** max(p_shape, eps)

def fit_timekill(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    d = df[(df["response_type"]=="log_reduction") & df["time_h"].notna()].copy()
    d = d.dropna(subset=["nisin_ug_per_ml","pH","time_h"])
    if d.empty:
        return FitResult({}, False, "No suitable time-kill rows with time_h found.")
    t = d["time_h"].astype(float).values
    y = d["response_value"].astype(float).values
    Cug = d["nisin_ug_per_ml"].astype(float).values
    pH = d["pH"].astype(float).values
    aw = d["aw"].astype(float).values if "aw" in d.columns else np.full_like(t, np.nan)
    Tgrow = d["growth_temp_C"].astype(float).values if "growth_temp_C" in d.columns else np.full_like(t, np.nan)
    PretT = d["pretreat_main_temp_C"].astype(float).values if "pretreat_main_temp_C" in d.columns else np.full_like(t, np.nan)
    PretMin = d["pretreat_time_min"].astype(float).values if "pretreat_time_min" in d.columns else np.full_like(t, np.nan)
    WarmT = d["warm_temp_C"].astype(float).values if "warm_temp_C" in d.columns else np.full_like(t, np.nan)
    WarmMin = d["warm_time_min"].astype(float).values if "warm_time_min" in d.columns else np.full_like(t, np.nan)

    def model_func(idx, b0,bC,bPH2,bAW,bTg,bPrT,bPrM,bWT,bWM,p_shape):
        out = np.zeros_like(idx, dtype=float)
        for i, k in enumerate(idx.astype(int)):
            out[i] = R_weibull(t[i], Cug[i], pH[i], aw[i], Tgrow[i], PretT[i], PretMin[i], WarmT[i], WarmMin[i],
                               b0,bC,bPH2,bAW,bTg,bPrT,bPrM,bWT,bWM,p_shape)
        return out

    idx = np.arange(len(t), dtype=int)
    p0 = [0.0, -0.5, 0.0, -2.0, -0.02, -0.02, -0.01, -0.02, -0.01, 1.0]
    bounds = ([-10,-5,-5,-5,-1,-1,-1,-1,-1, 0.1],
              [ 10, 5, 5, 5, 1, 1, 1, 1, 1, 5.0])
    popt, pcov = curve_fit(model_func, idx, y, p0=p0, bounds=bounds, maxfev=30000)
    keys = ["b0","bC","bPH2","bAW","bTg","bPrT","bPrM","bWT","bWM","p_shape"]
    params = {k: float(v) for k, v in zip(keys, popt)}
    yhat = model_func(idx, *popt)
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    params["rmse"] = rmse

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(y, yhat)
    plt.xlabel("Observed log-reduction")
    plt.ylabel("Fitted log-reduction")
    plt.title("Observed vs. Fitted (Weibull with aw, split temps)")
    plt.savefig("C:/Users/kaoru/iGEM/mnt/data/fit_aw_obsvsfit.png", dpi=160, bbox_inches="tight")
    plt.close()

    C_grid = np.logspace(-1, 4, 60)
    pH_grid = np.linspace(4.5, 7.5, 60)
    Tgrow = 37.0; PretT = 25.0; PretMin=0.0; WarmT=25.0; WarmMin=0.0; aw0=0.98
    Z = np.zeros((len(pH_grid), len(C_grid)))
    for i, pHval in enumerate(pH_grid):
        for j, Cval in enumerate(C_grid):
            Z[i,j] = R_weibull(24.0, Cval, pHval, aw0, Tgrow, PretT, PretMin, WarmT, WarmMin, *popt)
    plt.figure()
    CS = plt.contour(np.log10(C_grid), pH_grid, Z, levels=[1,2,3,4,5])
    plt.clabel(CS, inline=True, fontsize=8)
    plt.xlabel("log10 Nisin (ug/mL)")
    plt.ylabel("pH")
    plt.title("Log-reduction at 24 h (aw=0.98, 37°C)")
    plt.savefig("C:/Users/kaoru/iGEM/mnt/data/contour_aw_pH_Cug.png", dpi=160, bbox_inches="tight")
    plt.close()

    return FitResult(params, True, f"Fit OK. RMSE={rmse:.3f}. Plots saved.")

if __name__ == "__main__":
    res = fit_timekill()
    print(res.message)
    if res.success:
        for k,v in res.params.items():
            print(f"{k} = {v:.4g}")
