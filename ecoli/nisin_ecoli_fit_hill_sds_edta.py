
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

@dataclass
class FitResult:
    params: Dict[str, float]
    success: bool
    message: str

def fit_timekill(
    csv_path: str = "./mnt/data/nisin_ecoli_enriched.csv",
    policy: str = "clip",
    use_hill: bool = True,
    estimate_pH_aw: bool = True,
    make_plots: bool = True,
    contour_out: str = "./mnt/data/contour_v2_nocof.png",
    contour_aw: float = 0.98,
    contour_Tgrow: float = 37.0,
    contour_hours: float = 24.0,
    use_cofactors: bool = False,   # NEW
    drop_cofactors: bool = True    # NEW
) -> FitResult:
    df = pd.read_csv(csv_path)

    # Basic columns
    for col in ["time_h","response_value","nisin_ug_per_ml","pH","aw",
                "growth_temp_C","pretreat_main_temp_C","pretreat_time_min",
                "warm_temp_C","warm_time_min","sd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cofactor flags
    def has_word(val: str, word: str) -> bool:
        return isinstance(val, str) and (word.upper() in val.upper())

    df["has_SDS"]  = df.apply(lambda r: 1 if (has_word(r.get("medium",""),"SDS") or has_word(r.get("cofactor_type",""),"SDS")) else 0, axis=1)
    df["has_EDTA"] = df.apply(lambda r: 1 if (has_word(r.get("medium",""),"EDTA") or has_word(r.get("cofactor_type",""),"EDTA")) else 0, axis=1)

    if drop_cofactors:
        df = df[(df["has_SDS"]==0) & (df["has_EDTA"]==0)].copy()

    # Select usable rows
    d = df[(df["response_type"] == "log_reduction") & df["time_h"].notna()].copy()
    d = d.dropna(subset=["nisin_ug_per_ml","pH","time_h"])

    if not use_cofactors:
        params["bSDS"] = 0.0
        params["bEDTA"] = 0.0
    
    # Kill-only policy
    if policy == "exclude":
        d = d[d["response_value"] >= 0].copy()
    elif policy == "clip":
        d.loc[:, "response_value"] = d["response_value"].clip(lower=0.0)

    if d.empty:
        return FitResult({}, False, "No usable rows after filtering.")

    # Arrays
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

    if use_cofactors:
        has_SDS = d["has_SDS"].astype(float).values if "has_SDS" in d.columns else np.zeros_like(t)
        has_EDTA = d["has_EDTA"].astype(float).values if "has_EDTA" in d.columns else np.zeros_like(t)
    else:
        has_SDS = np.zeros_like(t, dtype=float)
        has_EDTA = np.zeros_like(t, dtype=float)

    # Weights
    sigma = d["sd"].fillna(1.0).replace(0,1.0).astype(float).values if "sd" in d.columns else np.ones_like(y, dtype=float)

    # Model
    def hill(C, EC50, h):
        eps = 1e-9
        C = max(C, eps); EC50 = max(EC50, eps)
        return (C**h) / (EC50**h + C**h)

    def R_weibull_hill(ti, Ci, pHi, awi, Tg, PrT, PrM, WT, WM,
                       b0, bHill, EC50, h, bPH2, bAW, bTg, bPrT, bPrM, bWT, bWM,
                       bSDS, bEDTA, p_shape, pH_opt, aw_ref, has_SDS_i, has_EDTA_i):
        eps = 1e-9
        H = hill(Ci, EC50, h)
        pHterm = (pHi - pH_opt)**2 if not np.isnan(pHi) else 0.0
        awterm = (awi - aw_ref) if not np.isnan(awi) else 0.0
        TgrowTerm = (Tg - 37.0) if not np.isnan(Tg) else 0.0
        PretTterm = (PrT - 25.0) if not np.isnan(PrT) else 0.0
        PretMterm = (PrM) if not np.isnan(PrM) else 0.0
        WarmTterm = (WT - 25.0) if not np.isnan(WT) else 0.0
        WarmMterm = (WM) if not np.isnan(WM) else 0.0
        delta = math.exp(b0 + bHill*H + bPH2*pHterm + bAW*awterm + bTg*TgrowTerm +
                         bPrT*PretTterm + bPrM*PretMterm + bWT*WarmTterm + bWM*WarmMterm +
                         bSDS*has_SDS_i + bEDTA*has_EDTA_i)
        return (max(ti, eps) / max(delta, eps)) ** max(p_shape, eps)

    def model_func(idx, *theta):
        out = np.zeros_like(idx, dtype=float)
        for i, _ in enumerate(idx.astype(int)):
            (b0,bHill,EC50,h,bPH2,bAW,bTg,bPrT,bPrM,bWT,bWM,bSDS,bEDTA,p_shape,pH_opt,aw_ref) = theta
            out[i] = R_weibull_hill(t[i], Cug[i], pH[i], aw[i], Tgrow[i], PretT[i], PretMin[i], WarmT[i], WarmMin[i],
                                    b0,bHill,EC50,h,bPH2,bAW,bTg,bPrT,bPrM,bWT,bWM,bSDS,bEDTA,p_shape,pH_opt,aw_ref,
                                    has_SDS[i], has_EDTA[i])
        return out

    # Params & bounds (Hill)
    p0 =  [0.0,   -3.0,  10.0, 1.0,  0.0,  -1.0, -0.02, -0.02, -0.01, -0.02, -0.01,  0.0,  0.0, 1.0, 6.0, 0.98]
    lb =  [-10,   -10.0,  0.01, 0.2, -5.0, -5.0, -1.0,  -1.0,  -1.0,  -1.0,  -1.0,   0.0,  0.0, 0.1, 4.0, 0.94]
    ub =  [ 10,     0.0, 1e6,   5.0,  5.0,  5.0,  1.0,   1.0,   1.0,   1.0,   1.0,   0.0,  0.0, 5.0, 8.0, 1.00]
    if use_cofactors:
        lb[11], ub[11] = -5.0, 5.0
        lb[12], ub[12] = -5.0, 5.0
    if not estimate_pH_aw:
        lb[-2] = ub[-2] = 6.0
        lb[-1] = ub[-1] = 0.98
    bounds = (lb, ub)

    popt, pcov = curve_fit(model_func, np.arange(len(t)), y, p0=p0, bounds=bounds, sigma=sigma, maxfev=120000)
    yhat = model_func(np.arange(len(t)), *popt)
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    keys = ["b0","bHill","EC50","h","bPH2","bAW","bTg","bPrT","bPrM","bWT","bWM","bSDS","bEDTA","p_shape","pH_opt","aw_ref"]
    params = {k: float(v) for k, v in zip(keys, popt)}
    params["rmse"] = rmse

    if make_plots:
        plt.figure()
        plt.scatter(y, yhat)
        plt.xlabel("Observed log-reduction")
        plt.ylabel("Fitted log-reduction")
        plt.title("Observed vs. Fitted (Weibull v2, no cofactors)")
        plt.savefig("./mnt/data/fit_v2_nocof_obsvsfit.png", dpi=160, bbox_inches="tight")
        plt.close()

        C_grid = np.logspace(-1, 4, 60)
        pH_grid = np.linspace(4.5, 7.5, 60)
        Z = np.zeros((len(pH_grid), len(C_grid)))
        PretT = 25.0; PretMin = 0.0; WarmT = 25.0; WarmMin = 0.0
        for i, pHval in enumerate(pH_grid):
            for j, Cval in enumerate(C_grid):
                (b0,bHill,EC50,h,bPH2,bAW,bTg,bPrT,bPrM,bWT,bWM,bSDS,bEDTA,p_shape,pH_opt,aw_ref) = popt
                z = R_weibull_hill(contour_hours, Cval, pHval, contour_aw, contour_Tgrow, PretT, PretMin, WarmT, WarmMin,
                                   b0,bHill,EC50,h,bPH2,bAW,bTg,bPrT,bPrM,bWT,bWM,0.0,0.0,p_shape,pH_opt,aw_ref,0.0,0.0)
                Z[i, j] = z
        plt.figure()
        CS = plt.contour(np.log10(C_grid), pH_grid, Z, levels=[1,2,3,4,5])
        plt.clabel(CS, inline=True, fontsize=8)
        plt.xlabel("log10 Nisin (ug/mL)")
        plt.ylabel("pH")
        plt.title(f"Log-reduction at {int(contour_hours)} h (aw={contour_aw}, {int(contour_Tgrow)}°C, no cofactors)")
        plt.savefig(contour_out, dpi=160, bbox_inches="tight")
        plt.close()

    with open("./mnt/data/parameters_v2_nocof.txt", "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k} = {v}\n")

    return FitResult(params, True, f"Fit OK (no cofactors). RMSE={rmse:.3f} (policy={policy}).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./mnt/data/nisin_ecoli_enriched.csv")
    ap.add_argument("--policy", default="clip", choices=["exclude","clip","keep"])
    ap.add_argument("--contour-out", default="./mnt/data/contour_v2_nocof.png")
    ap.add_argument("--contour-aw", type=float, default=0.98)
    ap.add_argument("--contour-Tgrow", type=float, default=37.0)
    ap.add_argument("--contour-hours", type=float, default=24.0)
    ap.add_argument("--use-cofactors", dest="use_cofactors", action="store_true")
    ap.add_argument("--no-cofactors", dest="use_cofactors", action="store_false")
    ap.set_defaults(use_cofactors=False)
    ap.add_argument("--drop-cofactors", dest="drop_cofactors", action="store_true")
    ap.add_argument("--keep-cofactors", dest="drop_cofactors", action="store_false")
    ap.set_defaults(drop_cofactors=True)
    args = ap.parse_args()

    res = fit_timekill(csv_path=args.csv, policy=args.policy,
                       contour_out=args.contour_out, contour_aw=args.contour_aw,
                       contour_Tgrow=args.contour_Tgrow, contour_hours=args.contour_hours,
                       use_cofactors=args.use_cofactors, drop_cofactors=args.drop_cofactors)
    print(res.message)
    if res.success:
        for k, v in res.params.items():
            print(f"{k} = {v:.6g}")
