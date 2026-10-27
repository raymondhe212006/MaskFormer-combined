"""Scan histogram binning / core window of the Gaussian fit to E_pred and the resulting resolution-curve chi2.
Usage: python binning_scan.py <eval_dir> <particle> <Emax_for_curve_fit>"""
import glob, os, re, sys, json
import numpy as np, awkward as ak
from scipy.optimize import curve_fit
MAX_DROP = 0.10
def gauss(x, a, mu, s): return a * np.exp(-0.5 * ((x - mu) / s) ** 2)
def peak_seed(e):
    lo, hi = np.quantile(e, [0.01, 0.99]); h, edges = np.histogram(e[(e > lo) & (e < hi)], bins=100)
    c = 0.5 * (edges[1:] + edges[:-1]); i = int(np.argmax(h)); above = np.where(h >= 0.5 * h[i])[0]
    return c[i], max((c[above[-1]] - c[above[0]] + (edges[1] - edges[0])) / 2.355, 2 * (edges[1] - edges[0]))
def fit(e, nbins, nsig):
    mu, s = peak_seed(e)
    while True:
        for _ in range(6):
            sel = (e > mu - nsig * s) & (e < mu + nsig * s); w = e[sel]
            h, edges = np.histogram(w, bins=nbins); c = 0.5 * (edges[1:] + edges[:-1])
            (a, mu, s), cov = curve_fit(gauss, c, h, p0=[h.max(), mu, s], sigma=np.sqrt(np.maximum(h, 1)), absolute_sigma=True, maxfev=20000); s = abs(s)
        sel = (e > mu - nsig * s) & (e < mu + nsig * s)
        if 1 - sel.mean() <= MAX_DROP or nsig >= 4: return mu, s, float(np.sqrt(cov[2, 2])), float(1 - sel.mean()), nsig
        nsig += 0.25
evaldir, particle, emax = sys.argv[1], sys.argv[2], float(sys.argv[3])
sets = sorted([d for d in os.listdir(evaldir) if re.fullmatch(particle + r'_E\d+', d)], key=lambda d: int(d.rsplit('_E', 1)[1]))
data = {}
for d in sets:
    f = glob.glob(os.path.join(evaldir, d, '*__test_*.parquet'))
    if not f: continue
    p = ak.from_parquet(f[0], columns=['ticlSimTracksters.*'])['ticlSimTracksters']
    pred = ak.to_numpy(ak.flatten(p['ticlSimTracksters_pred_e'])).astype(float); data[float(d.rsplit('_E', 1)[1])] = pred[np.isfinite(pred)]
def model(E, a2, c2): return np.sqrt(a2 / E + c2)
print(f"{'nbins':>5}{'nsig':>5} | " + "".join(f"{int(E):>7}" for E in data) + f" | {'a%':>5}{'c%':>5}{'chi2/ndf':>10}{'maxdrop%':>9}")
print("sigma/E in %% per energy (peak-seeded core fit), then the sqrt(a2/E+c2) fit over E <= %.0f GeV" % emax)
for nbins in (15, 20, 30, 40, 60):
    for nsig in (1.5, 2.0, 2.5, 3.0):
        E, y, ey, drops = [], [], [], []
        for En, e in data.items():
            mu, s, se, drop, ns = fit(e, nbins, nsig); E.append(En); y.append(s / En); ey.append(se / En); drops.append(drop)
        E, y, ey = map(np.array, (E, y, ey)); m = E <= emax
        (a2, c2), _ = curve_fit(model, E[m], y[m], p0=[0.02, 5e-4], sigma=ey[m], absolute_sigma=True, maxfev=20000)
        chi2 = np.sum(((y[m] - model(E[m], a2, c2)) / ey[m]) ** 2); ndf = m.sum() - 2
        print(f"{nbins:>5}{nsig:>5.1f} | " + "".join(f"{100*v:>7.2f}" for v in y) + f" | {100*np.sqrt(a2):>5.1f}{100*np.sqrt(c2):>5.1f}{chi2:>7.1f}/{ndf:<2d}{100*max(drops):>9.1f}")
