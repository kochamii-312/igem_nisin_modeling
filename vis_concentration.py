# Plot the relationship between nisin concentration and antibacterial effect
# using the latest estimated parameters (no-cofactor fit).
# Condition: 24 h, pH = pH_opt (estimated), aw = aw_ref (estimated), Tgrow = 37 C,
# no pretreatment/warm steps.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load parameters (estimated values)
params = {}
for line in Path("/mnt/data/parameters_v2_nocof.txt").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        k = k.strip()
        try:
            params[k] = float(v.strip())
        except:
            pass

b0      = params.get("b0", 0.0)
bHill   = params.get("bHill", -3.0)
EC50    = params.get("EC50", 10.0)
h       = params.get("h", 1.0)
bPH2    = params.get("bPH2", 0.0)
bAW     = params.get("bAW", 0.0)
bTg     = params.get("bTg", 0.0)
bPrT    = params.get("bPrT", 0.0)
bPrM    = params.get("bPrM", 0.0)
bWT     = params.get("bWT", 0.0)
bWM     = params.get("bWM", 0.0)
p_shape = params.get("p_shape", 1.0)
pH_opt  = params.get("pH_opt", 6.0)
aw_ref  = params.get("aw_ref", 0.98)

def H_hill(C):
    eps = 1e-12
    C = np.maximum(C, eps)
    return (C**h) / (EC50**h + C**h)

def delta(C, pH, aw, Tgrow, PretT=25.0, PretMin=0.0, WarmT=25.0, WarmMin=0.0):
    exp_in = (
        b0 + bHill * H_hill(C)
        + bPH2 * (pH - pH_opt) ** 2
        + bAW * (aw - aw_ref)
        + bTg * (Tgrow - 37.0)
        + bPrT * (PretT - 25.0) + bPrM * (PretMin)
        + bWT * (WarmT - 25.0) + bWM * (WarmMin)
    )
    return np.exp(exp_in)

def log_reduction(t_h, C, pH, aw, Tgrow):
    eps = 1e-12
    d = delta(C, pH, aw, Tgrow)
    return (np.maximum(t_h, eps) / np.maximum(d, eps)) ** np.maximum(p_shape, eps)

# Plot settings
t_hours = 24.0
pH_use = pH_opt   # use estimated optimum to isolate concentration effect
aw_use = aw_ref
T_use  = 37.0     # baseline growth/storage temperature

# X axis: log10 Nisin from 0.01 to 10^4 ug/mL
x_logC = np.linspace(-2, 4, 400)
C = 10 ** x_logC

# Compute response
R = log_reduction(t_hours, C, pH_use, aw_use, T_use)

# Make figure
plt.figure()
plt.plot(x_logC, R, label="24 h @ pH=pH_opt, aw=aw_ref, 37 C")

# Annotate EC50 and typical "switch" band (0.25x–4x EC50)
x_ec50 = np.log10(EC50) if EC50 > 0 else None
if x_ec50 is not None and np.isfinite(x_ec50):
    plt.axvline(x_ec50, linestyle="--")
    plt.text(x_ec50, max(R)*0.7, "EC50", rotation=90, va="center", ha="right")
    # shade practical transition band roughly 0.25x–4x EC50
    x_lo = np.log10(EC50*0.25) if EC50*0.25 > 0 else x_logC.min()
    x_hi = np.log10(EC50*4.0) if EC50*4.0 > 0 else x_logC.max()
    plt.axvspan(x_lo, x_hi, alpha=0.1)

plt.xlabel("log10 Nisin (ug/mL)")
plt.ylabel("Log10 reduction (24 h)")
plt.title("Nisin concentration vs antibacterial effect\n(estimated parameters)")
plt.grid(True, which="both")
plt.legend()

outpath = "./mnt/data/concentration_vs_logreduction_24h.png"
plt.savefig(outpath, dpi=160, bbox_inches="tight")
plt.close()

outpath
