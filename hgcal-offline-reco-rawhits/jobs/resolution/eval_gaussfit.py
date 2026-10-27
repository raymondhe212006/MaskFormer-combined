"""Gaussian fit of the predicted-energy histogram per mono-energetic eval set.

Two fits of E_pred (matched queries of the true particles) per set:
  quant : seed mu/sigma from median and quantile width, window median +- 2.5 sigma, 3 passes (first version)
  peak  : seed mu from the histogram mode and sigma from the FWHM of the peak, window mu +- 2 sigma, up to
          6 passes; events outside the final window are "dropped" (fraction recorded, must stay <= MAX_DROP:
          if it does not, the window is widened in 0.25 sigma steps until it does).
Both use a Poisson-weighted chi2 fit of a Gaussian to a 40-bin histogram of the window.
Writes <eval_dir>/gaussfit_summary.json and per-set diagnostic histograms in <eval_dir>/resolution_plots/hists/.
Usage: python eval_gaussfit.py <eval_dir> [set ...]
"""
import glob, os, re, sys, json
import numpy as np, awkward as ak
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_DROP = 0.10
def gauss(x, a, mu, s): return a * np.exp(-0.5 * ((x - mu) / s) ** 2)

def fit_window(e, mu, s, nsig, passes, nbins=40):
    for _ in range(passes):
        sel = (e > mu - nsig * s) & (e < mu + nsig * s); w = e[sel]
        h, edges = np.histogram(w, bins=nbins); c = 0.5 * (edges[1:] + edges[:-1])
        (a, mu, s), cov = curve_fit(gauss, c, h, p0=[h.max(), mu, s], sigma=np.sqrt(np.maximum(h, 1)),
                                    absolute_sigma=True, maxfev=20000)
        s = abs(s)
    sel = (e > mu - nsig * s) & (e < mu + nsig * s)
    return dict(mu=float(mu), sigma=float(s), mu_err=float(np.sqrt(cov[1, 1])), sigma_err=float(np.sqrt(cov[2, 2])),
                n_core=int(sel.sum()), drop=float(1 - sel.mean()), lo=float(mu - nsig * s), hi=float(mu + nsig * s), nsig=nsig)

NANFIT = dict(mu=float('nan'), sigma=float('nan'), mu_err=float('nan'), sigma_err=float('nan'), n_core=0, drop=float('nan'), lo=0.0, hi=0.0, nsig=0.0)
def safe(fn):
    def w(*a, **k):
        try: return fn(*a, **k)
        except Exception as ex: print(f"  fit failed ({fn.__name__}): {ex}"); return dict(NANFIT)
    return w
@safe
def fit_quant(e):
    mu, s = np.median(e), 0.5 * (np.quantile(e, 0.84) - np.quantile(e, 0.16))
    return fit_window(e, mu, s, 2.5, 3)

@safe
def fit_peak(e):
    lo, hi = np.quantile(e, [0.01, 0.99]); h, edges = np.histogram(e[(e > lo) & (e < hi)], bins=100)
    c = 0.5 * (edges[1:] + edges[:-1]); i = int(np.argmax(h)); mu = c[i]
    above = np.where(h >= 0.5 * h[i])[0]; fwhm = c[above[-1]] - c[above[0]] + (edges[1] - edges[0])
    s = max(fwhm / 2.355, 2 * (edges[1] - edges[0]))
    nsig = 2.0
    while True:
        r = fit_window(e, mu, s, nsig, 6)
        if r["drop"] <= MAX_DROP or nsig >= 4.0: return r
        nsig += 0.25

def sigma68(x): return 0.5 * (np.quantile(x, 0.84) - np.quantile(x, 0.16))

evaldir = sys.argv[1]; wanted = sys.argv[2:]
hdir = os.path.join(evaldir, "resolution_plots", "hists"); os.makedirs(hdir, exist_ok=True)
sets = sorted([d for d in os.listdir(evaldir) if re.fullmatch(r'(photons|pions|K0L)_E\d+', d)],
              key=lambda d: ({'photons': 0, 'pions': 1, 'K0L': 2}[d.rsplit('_E', 1)[0]], int(d.rsplit('_E', 1)[1])))
out = {}
print(f"{'set':<13}{'E_nom':>6}{'N':>6}{'s68(r)%':>8} | {'quant: mu':>10}{'sig':>7}{'sig/E%':>7}{'drop%':>6} | {'peak: mu':>9}{'sig':>7}{'sig/E%':>7}{'drop%':>6}{'nsig':>5} | {'sumTS: med':>12}{'s68%':>6}{'gaus%':>7}{'zero%':>6}")
for d in sets:
    if wanted and d not in wanted: continue
    f = glob.glob(os.path.join(evaldir, d, '*__test_*.parquet'))
    if not f: continue
    enom = float(d.rsplit('_E', 1)[1])
    tab = ak.from_parquet(f[0], columns=['ticlSimTracksters.*', 'ticlTrackstersCLUE3DHigh.*'])
    p = tab['ticlSimTracksters']; tk = tab['ticlTrackstersCLUE3DHigh']
    # sumTS: sum of the raw energies of ALL CLUE3D tracksters in the event (single-particle gun, so this is the
    # non-ML 'add up the tracksters' baseline; the matched query's mask covers ~all of them anyway)
    tke = [x for x in tk.fields if x.endswith('raw_energy')][0]
    masksum = ak.to_numpy(ak.sum(tk[tke], axis=1)).astype(float)
    pred = ak.to_numpy(ak.flatten(p['ticlSimTracksters_pred_e'])).astype(float)
    cands = [x for x in p.fields if x.endswith('regressed_energy')] or [x for x in p.fields if x.endswith('raw_energy')]
    tname = cands[0]; true = ak.to_numpy(ak.flatten(p[tname])).astype(float)
    ok = np.isfinite(pred) & np.isfinite(true) & (true > 0); pred, true, masksum = pred[ok], true[ok], masksum[ok]
    q, k = fit_quant(pred), fit_peak(pred)
    msok = masksum > 0
    ms = fit_peak(masksum[msok]) if msok.sum() >= 50 else dict(mu=float('nan'), sigma=float('nan'), mu_err=float('nan'), sigma_err=float('nan'), n_core=0, drop=float('nan'), lo=0, hi=0, nsig=0)
    ms['n'] = int(msok.sum()); ms['frac_zero'] = float(1 - msok.mean())
    ms['sigma68_ratio'] = float(sigma68(masksum[msok] / true[msok])); ms['median_ratio'] = float(np.median(masksum[msok] / true[msok]))
    ms['sigma_over_Enom'] = ms['sigma'] / enom
    rec = dict(E_nom=enom, n=int(len(pred)), truth=tname, mean_true=float(true.mean()), sigma68_ratio=float(sigma68(pred / true)),
               quant=q, peak=k, masksum=ms)
    for name, r in (("quant", q), ("peak", k)):
        r["sigma_over_Enom"] = r["sigma"] / enom; r["sigma_over_mu"] = r["sigma"] / r["mu"]
    # backwards-compatible top-level = peak fit
    rec.update({kk: k[kk] for kk in ("mu", "sigma", "mu_err", "sigma_err", "n_core", "sigma_over_Enom", "sigma_over_mu")})
    out[d] = rec
    print(f"{d:<13}{enom:>6.0f}{len(pred):>6}{100*rec['sigma68_ratio']:>8.2f} | {q['mu']:>10.2f}{q['sigma']:>7.2f}{100*q['sigma_over_Enom']:>7.2f}{100*q['drop']:>6.1f} | "
          f"{k['mu']:>9.2f}{k['sigma']:>7.2f}{100*k['sigma_over_Enom']:>7.2f}{100*k['drop']:>6.1f}{k['nsig']:>5.2f} | {ms['median_ratio']:>12.3f}{100*ms['sigma68_ratio']:>6.2f}{100*ms['sigma_over_Enom']:>7.2f}{100*ms['frac_zero']:>6.1f}")
    # diagnostic histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo, hi = np.quantile(pred, [0.002, 0.998]); h, edges, _ = ax.hist(pred, bins=100, range=(lo, hi), histtype="step", color="k", lw=1.2, label=f"E_pred (N={len(pred)})")
    x = np.linspace(lo, hi, 400); bw = edges[1] - edges[0]
    for name, r, col in (("quant", q, "#1f3fbf"), ("peak", k, "#e4002b")):
        sel = (pred > r["lo"]) & (pred < r["hi"]); amp = sel.sum() * bw / (r["sigma"] * np.sqrt(2 * np.pi))
        ax.plot(x, gauss(x, amp, r["mu"], r["sigma"]), color=col, lw=1.5, label=f"{name}: mu={r['mu']:.2f} s={r['sigma']:.2f} ({100*r['sigma_over_Enom']:.1f}%) drop={100*r['drop']:.1f}%")
        ax.axvspan(r["lo"], r["hi"], color=col, alpha=0.06)
    ax.axvline(enom, color="gray", ls="--", lw=1); ax.set_xlabel("E_pred [GeV]"); ax.set_ylabel("entries"); ax.set_title(d)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(hdir, f"epred_{d}.png"), dpi=120); plt.close(fig)
json.dump(out, open(os.path.join(evaldir, 'gaussfit_summary.json'), 'w'), indent=1)
