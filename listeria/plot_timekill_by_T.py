# plot_timekill_by_T_from_params.py
import argparse
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
    for r in ['b0','bHill','EC50','h','bPH2','bT','p_shape','pH_opt','Tref']:
        if r not in params:
            raise ValueError(f"Missing parameter '{r}' in {path}")
    return params

def hill(C, EC50, h):
    eps = 1e-12
    C = np.maximum(C, eps)
    EC50 = max(EC50, eps)
    return (C**h) / (EC50**h + C**h)

def log_reduction(t, C, pH, T, par):
    H = hill(C, par['EC50'], par['h'])
    ln_delta = par['b0'] + par['bHill']*H + par['bPH2']*(pH - par['pH_opt'])**2 + par['bT']*(T - par['Tref'])
    delta = np.exp(ln_delta)
    eps = 1e-12
    return (np.maximum(t, eps) / np.maximum(delta, eps)) ** max(par['p_shape'], eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='./mnt/data/plots_listeria_simple/fit_params_listeria_simple.txt')
    ap.add_argument('--C', type=float, default=1000.0)
    ap.add_argument('--pH', type=float, default=None)
    ap.add_argument('--T_list', type=str, default='3,5,7,15,25')
    ap.add_argument('--include_Tref', action='store_true')
    ap.add_argument('--tmax_h', type=float, default=100.0)
    ap.add_argument('--points', type=int, default=200)
    ap.add_argument('--out', type=str, default='./mnt/data/plots_listeria_simple/fig4_pred_timekill_by_T.png')
    args = ap.parse_args()

    par = load_params(args.params)
    pH = par['pH_opt'] if args.pH is None else args.pH

    T_vals = [float(x) for x in args.T_list.split(',') if x.strip()]
    if args.include_Tref and (par['Tref'] not in T_vals):
        T_vals.append(par['Tref'])

    t = np.linspace(0.0, args.tmax_h, args.points)

    plt.figure(figsize=(9,6))
    for Tv in sorted([v for v in T_vals if abs(v - par['Tref']) > 1e-6]):
        y = log_reduction(t, args.C, pH, Tv, par)
        plt.plot(t, y, linestyle='--', linewidth=1.2, label=f'{Tv:g} °C')

    y_ref = log_reduction(t, args.C, pH, par['Tref'], par)
    plt.plot(t, y_ref, linewidth=3.0, label=f"Tref = {par['Tref']:.1f} °C")

    plt.xlabel('Time (h)')
    plt.ylabel('Predicted log N0/Nt')
    plt.title(f'Predicted time‑kill curves by temperature (C={args.C:g} mg/L, pH={pH:.2f})')
    plt.legend(title='Temperature')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
