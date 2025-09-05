# Create two graphs:
# 1) Antibacterial effect vs pH (acidic range), multiple concentrations; T=25 C, aw=aw_ref
# 2) Antibacterial effect vs Temperature (0–40 C), multiple pH lines (acidic side), at C=2 ug/mL; aw=aw_ref
#
# Uses estimated parameters from /mnt/data/parameters_v2_nocof.txt

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load parameters
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

def delta(C, pH, aw, Tgrow, PretT=25.0, PretMin=0.0, WarmT=25.0, WarmMin=0.0, use_bTg=True):
    exp_in = (
        b0 + bHill * H_hill(C)
        + bPH2 * (pH - pH_opt) ** 2
        + bAW * (aw - aw_ref)
        + (bTg * (Tgrow - 37.0) if use_bTg else 0.0)
        + bPrT * (PretT - 25.0) + bPrM * (PretMin)
        + bWT * (WarmT - 25.0) + bWM * (WarmMin)
    )
    return np.exp(exp_in)

def log_reduction(t_h, C, pH, aw, Tgrow, use_bTg=True):
    eps = 1e-12
    d = delta(C, pH, aw, Tgrow, use_bTg=use_bTg)
    return (np.maximum(t_h, eps) / np.maximum(d, eps)) ** np.maximum(p_shape, eps)

# Common settings
t_hours = 24.0
aw_use = aw_ref

# 1) pH (acidic side) vs log reduction, multiple concentrations, T=25 C
pH_vals = np.linspace(4.0, 6.5, 251)  # acidic side focus
concs = [0.5, 1.0, 2.0, 5.0, 10.0]    # ug/mL
T_use = 25.0

plt.figure()
for C in concs:
    R = log_reduction(t_hours, C, pH_vals, aw_use, T_use, use_bTg=True)
    plt.plot(pH_vals, R, label=f"C={C:g} ug/mL")
plt.axvline(pH_opt, linestyle="--")
plt.text(pH_opt, 0.05, "pH_opt", rotation=90, va="bottom", ha="right")
plt.xlabel("pH")
plt.ylabel("Log10 reduction (24 h)")
plt.title("Antibacterial effect vs pH (acidic range)\n(T=25 C, aw=aw_ref, no pretreatment)")
plt.legend()
plt.grid(True, which="both")
out1 = "/mnt/data/pH_acidic_vs_logreduction_25C.png"
plt.savefig(out1, dpi=160, bbox_inches="tight")
plt.close()

# 2) Temperature (0–40 C) vs log reduction, multiple acidic pH lines, at C = 2 ug/mL
T_vals = np.linspace(0.0, 40.0, 401)
C_use = 2.0  # ~above EC50 to show slope but not fully saturated
pH_lines = [4.5, 5.0, 5.5, 6.0]

plt.figure()
for pHv in pH_lines:
    R = log_reduction(t_hours, C_use, pHv, aw_use, T_vals, use_bTg=True)
    plt.plot(T_vals, R, label=f"pH={pHv:g}")
plt.xlabel("Temperature (C)")
plt.ylabel("Log10 reduction (24 h)")
plt.title("Antibacterial effect vs Temperature (0-40 C)\n(C=2 ug/mL, aw=aw_ref, no pretreatment)")
plt.legend()
plt.grid(True, which="both")
out2 = "/mnt/data/temperature_vs_logreduction_multiph_acidic.png"
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close()

out1, out2
