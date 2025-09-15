
# nisin_listeria_fit_simple.py
# Weibull(time) with Hill(C) and quadratic pH, linear T effect on ln(delta)
# delta(C, pH, T) = exp( b0 + bHill*H(C) + bPH2*(pH - pH_opt)^2 + bT*(T - Tref) )

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def hill(C, EC50, h):
    eps = 1e-12
    C = max(C, eps)
    EC50 = max(EC50, eps)
    return (C**h) / (EC50**h + C**h)

def model_values(idx, t, y, C, pH, T, params, Tref):
    # params = [b0, bHill, EC50, h, bPH2, bT, p_shape, pH_opt]
    b0, bHill, EC50, h, bPH2, bT, p_shape, pH_opt = params
    out = np.zeros_like(idx, dtype=float)
    eps = 1e-12
    for i, k in enumerate(idx.astype(int)):
        H = hill(C[k], EC50, h)
        ln_delta = b0 + bHill*H + bPH2*((pH[k] - pH_opt)**2) + bT*(T[k] - Tref)
        delta = math.exp(ln_delta)
        out[i] = (max(t[k], eps) / max(delta, eps)) ** max(p_shape, eps)
    return out

def fit(csv_path:str,
        Tref:float=7.0,
        policy:str="clip",
        make_plots:bool=True,
        out_dir:str="./mnt/data/plots_listeria_simple"):
    # Load
    df = pd.read_csv(csv_path)
    # Required columns
    required = ["response_type", "time_h", "response_value", "nisin_ug_per_ml", "pH", "growth_temp_C"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Filter to time-kill
    d = df[(df["response_type"] == "log_reduction") & df["time_h"].notna()].copy()
    d = d.dropna(subset=["nisin_ug_per_ml","pH","growth_temp_C","time_h","response_value"])
    if d.empty:
        raise ValueError("No usable rows after filtering.")
    # Types
    for c in ["time_h","response_value","nisin_ug_per_ml","pH","growth_temp_C","sd"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # Handle y < 0
    if policy == "clip":
        d.loc[:, "response_value"] = d["response_value"].clip(lower=0.0)
    elif policy == "exclude":
        d = d[d["response_value"] >= 0.0].copy()

    # Arrays
    t = d["time_h"].astype(float).values
    y = d["response_value"].astype(float).values
    C = d["nisin_ug_per_ml"].astype(float).values
    pH = d["pH"].astype(float).values
    T = d["growth_temp_C"].astype(float).values

    # Weights
    if "sd" in d.columns and d["sd"].notna().any():
        sigma = d["sd"].fillna(1.0).replace(0, 1.0).astype(float).values
    else:
        sigma = np.ones_like(y, dtype=float)

    # Parameter vector: [b0, bHill, EC50, h, bPH2, bT, p_shape, pH_opt]
    p0 = np.array([ 0.0,  -3.0,  10.0, 1.0,  0.0,  -0.05, 1.2, 5.8], dtype=float)
    lb = np.array([-10.0, -10.0,  0.01, 0.2, -5.0,  -1.0,  0.1, 4.0], dtype=float)
    ub = np.array([ 10.0,   0.0, 1e6,  5.0,  5.0,   1.0,  5.0, 8.0], dtype=float)

    idx = np.arange(len(t), dtype=int)

    def wrapped(idx, b0, bHill, EC50, h, bPH2, bT, p_shape, pH_opt):
        params = [b0, bHill, EC50, h, bPH2, bT, p_shape, pH_opt]
        return model_values(idx, t, y, C, pH, T, params, Tref)

    popt, pcov = curve_fit(
        wrapped, idx, y, p0=p0, bounds=(lb, ub), sigma=sigma, maxfev=150000
    )
    yhat = wrapped(idx, *popt)
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))

    params = {
        "b0": float(popt[0]),
        "bHill": float(popt[1]),
        "EC50": float(popt[2]),
        "h": float(popt[3]),
        "bPH2": float(popt[4]),
        "bT": float(popt[5]),
        "p_shape": float(popt[6]),
        "pH_opt": float(popt[7]),
        "Tref": float(Tref),
        "rmse": rmse
    }

    # Outputs
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Observed vs Fitted
    if make_plots:
        plt.figure()
        plt.scatter(y, yhat)
        plt.xlabel("Observed log-reduction (L. monocytogenes)")
        plt.ylabel("Fitted log-reduction")
        plt.title("Observed vs Fitted (Weibull × Hill)")
        xy_min = min(np.min(y), np.min(yhat))
        xy_max = max(np.max(y), np.max(yhat))
        plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1)
        plt.savefig(out_dir / "fit_observed_vs_fitted.png", dpi=160, bbox_inches="tight")
        plt.close()

        # Contour: log-reduction after fixed time across C and pH for a given T
        hours = 24.0
        C_grid = np.logspace(-1, 4, 80)  # ug/mL
        pH_grid = np.linspace(4.0, 7.5, 80)
        Z = np.zeros((len(pH_grid), len(C_grid)))
        for i, pHval in enumerate(pH_grid):
            for j, Cval in enumerate(C_grid):
                H = hill(Cval, params["EC50"], params["h"])
                ln_delta = params["b0"] + params["bHill"]*H + params["bPH2"]*(pHval - params["pH_opt"])**2 + params["bT"]*(Tref - Tref)
                delta = math.exp(ln_delta)
                Z[i, j] = (hours / max(delta, 1e-12)) ** max(params["p_shape"], 1e-12)

        plt.figure()
        CS = plt.contour(np.log10(C_grid), pH_grid, Z, levels=[0.5,1,2,3,4,5])
        plt.clabel(CS, inline=True, fontsize=8)
        plt.xlabel("log10 Nisin (ug/mL)")
        plt.ylabel("pH")
        plt.title(f"Predicted log-reduction at {int(hours)} h (T={Tref}°C)")
        plt.savefig(out_dir / "contour_logred_24h.png", dpi=160, bbox_inches="tight")
        plt.close()

    # Save params
    param_path = out_dir / "fit_params_listeria_simple.txt"
    with open(param_path, "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k} = {v}\n")

    return params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="./mnt/data/nisin_listeria_enriched.csv")
    ap.add_argument("--Tref", type=float, default=7.0, help="Reference temperature (°C) used in ln(delta)")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--policy", type=str, default="clip", choices=["clip","exclude"])
    args = ap.parse_args()

    params = fit(csv_path=args.csv, Tref=args.Tref, make_plots=(not args.no_plots), policy=args.policy)
    print("Fit complete. Parameters:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"{k} = {v:.6g}")
        else:
            print(f"{k} = {v}")

if __name__ == "__main__":
    main()
