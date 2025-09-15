# plot_timekill_by_concentrationpy
# Predict time-kill curves (log reduction vs time) per Nisin concentration
# using parameters fitted by nisin_listeria_fit_simple.py

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def load_params(path: str) -> dict:
    params = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = [s.strip() for s in line.split('=', 1)]
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    # sanity
    require = ['b0','bHill','EC50','h','bPH2','bT','p_shape','pH_opt','Tref']
    for r in require:
        if r not in params:
            raise ValueError(f"Missing parameter '{r}' in {path}")
    return params

def hill(C, EC50, h):
    eps = 1e-12
    C = np.maximum(C, eps)
    EC50 = max(EC50, eps)
    return (C**h) / (EC50**h + C**h)

def log_reduction(t, C, pH, T, par):
    """
    y(t) = ( t / delta(C,pH,T) )^p_shape
    delta = exp( b0 + bHill*H(C) + bPH2*(pH - pH_opt)^2 + bT*(T - Tref) )
    """
    H = hill(C, par['EC50'], par['h'])
    ln_delta = par['b0'] + par['bHill']*H + par['bPH2']*(pH - par['pH_opt'])**2 + par['bT']*(T - par['Tref'])
    delta = np.exp(ln_delta)
    eps = 1e-12
    return (np.maximum(t, eps) / np.maximum(delta, eps)) ** max(par['p_shape'], eps)

def main():
    ag = argparse.ArgumentParser()
    ag.add_argument('--params', default='./mnt/data/plots_listeria_simple/fit_params_listeria_simple.txt')
    ag.add_argument('--pH', type=float, default=None, help='pH used for prediction (default: fitted pH_opt)')
    ag.add_argument('--T', type=float, default=None, help='Temperature (°C) used for prediction (default: Tref)')
    ag.add_argument('--concs', type=str, default='0,0.025,2,5,10,1000,2000',
                    help='Comma-separated list of Nisin concentrations (mg/L == ug/mL)')
    ag.add_argument('--tmax_h', type=float, default=200.0)
    ag.add_argument('--points', type=int, default=200)
    ag.add_argument('--out', type=str, default='./mnt/data/plots_listeria_simple/fig2_pred_timekill_concentration.png')
    args = ag.parse_args()

    par = load_params(args.params)
    pH = par['pH_opt'] if args.pH is None else args.pH
    T = par['Tref'] if args.T is None else args.T

    concs = [float(x) for x in args.concs.split(',') if x.strip()]
    t = np.linspace(0.0, args.tmax_h, args.points)

    plt.figure(figsize=(8,6))
    for C in concs:
        y = log_reduction(t, C, pH, T, par)
        plt.plot(t, y, label=f"{C:g} mg/L")

    plt.xlabel('Time (h)')
    plt.ylabel('Predicted log N0/Nt')
    plt.title(f'Predicted time-kill curves (pH={pH:.2f}, T={T:.1f}°C)')
    plt.legend(title='Nisin (mg/L)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()