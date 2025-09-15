# plot_timekill_by_P H_from_params.py
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
    ag = argparse.ArgumentParser()
    ag.add_argument('--params', default='./mnt/data/plots_listeria_simple/fit_params_listeria_simple.txt')
    ag.add_argument('--C', type=float, default=1000.0)
    ag.add_argument('--T', type=float, default=None)
    ag.add_argument('--pH_list', type=str, default='4.5,5.0,5.5,6.0,6.5,7.0')
    ag.add_argument('--include_pHopt', action='store_true')
    ag.add_argument('--tmax_h', type=float, default=200.0)
    ag.add_argument('--points', type=int, default=200)
    ag.add_argument('--out', type=str, default='./mnt/data/plots_listeria_simple/pred_timekill_by_pH.png')
    args = ag.parse_args()

    par = load_params(args.params)
    T = par['Tref'] if args.T is None else args.T

    pH_vals = [float(x) for x in args.pH_list.split(',') if x.strip()]
    if args.include_pHopt and (par['pH_opt'] not in pH_vals):
        pH_vals.append(par['pH_opt'])

    t = np.linspace(0.0, args.tmax_h, args.points)

    plt.figure(figsize=(9,6))
    for ph in sorted([v for v in pH_vals if abs(v - par['pH_opt']) > 1e-6]):
        y = log_reduction(t, args.C, ph, T, par)
        plt.plot(t, y, linestyle='--', linewidth=1.2, label=f'pH {ph:g}')

    y_opt = log_reduction(t, args.C, par['pH_opt'], T, par)
    plt.plot(t, y_opt, linewidth=3.0, label=f"pH_opt = {par['pH_opt']:.2f}")

    plt.xlabel('Time (h)')
    plt.ylabel('Predicted log N0/Nt')
    plt.title(f'Predicted time-kill curves by pH (C={args.C:g} mg/L, T={T:.1f}°C)')
    plt.legend(title='pH')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
