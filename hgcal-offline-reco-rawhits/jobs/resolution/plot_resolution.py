"""Energy-resolution plot sigma(E)/E vs E from gaussfit_summary.json, fitted with sqrt(a^2/E + c^2).

Points: sigma of the Gaussian fit to the E_pred histogram divided by the nominal energy, error bars
from the Gaussian-fit sigma uncertainty. Usage: python plot_resolution.py <eval_dir> <particle> [label]
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
try:
    import mplhep as hep; plt.style.use(hep.style.CMS); HAS_HEP = True
except Exception:
    HAS_HEP = False

evaldir, particle = sys.argv[1], sys.argv[2]
variant = os.environ.get("RES_FIT", "peak")
EMAX = float(os.environ.get("RES_FIT_EMAX", "inf"))  # fit only points with E <= EMAX; others drawn without the curve
HIDE = os.environ.get("RES_HIDE_ABOVE", "0") == "1"  # 1: do not draw the points above EMAX at all
label = sys.argv[3] if len(sys.argv) > 3 else {"photons": "Photons γ", "pions": "Pions π±", "K0L": "K0L"}.get(particle, particle)
gf = json.load(open(os.path.join(evaldir, "gaussfit_summary.json")))
pts = sorted([(v["E_nom"], v[variant]["sigma"] / v["E_nom"], v[variant]["sigma_err"] / v["E_nom"]) for k, v in gf.items() if k.startswith(particle + "_E")])
E, y, ey = (np.array(x, dtype=float) for x in zip(*pts))

def model(E, a2, c2): return np.sqrt(a2 / E + c2)
infit = E <= EMAX
(a2, c2), cov = curve_fit(model, E[infit], y[infit], p0=[0.04, 1e-4], sigma=ey[infit], absolute_sigma=True, maxfev=20000)
a2e, c2e = np.sqrt(np.diag(cov))
chi2 = float(np.sum(((y[infit] - model(E[infit], a2, c2)) / ey[infit]) ** 2)); ndf = int(infit.sum()) - 2

fig, ax = plt.subplots(figsize=(10, 8))
Ef = np.logspace(np.log10(E.min()), np.log10(E[infit].max()), 200)
ax.plot(Ef, model(Ef, a2, c2), color="#e4002b", lw=2, zorder=2)
ax.errorbar(E[infit], y[infit], yerr=ey[infit], fmt="x", color="#1f3fbf", ms=9, mew=1.8, capsize=0, zorder=3)
if (~infit).any() and not HIDE: ax.errorbar(E[~infit], y[~infit], yerr=ey[~infit], fmt="x", color="#1f3fbf", ms=9, mew=1.8, capsize=0, zorder=3, alpha=0.55)
ax.set_xscale("log"); ax.set_xlabel("Energy [GeV]"); ax.set_ylabel("σ(E)/E")
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
Eshown = E[infit] if HIDE else E
ax.set_xlim(Eshown.min() / 1.3, Eshown.max() * 1.3)
ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500) if Eshown.min() / 1.3 <= t <= Eshown.max() * 1.3]
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_minor_formatter(NullFormatter())
ymax = (y + ey)[infit].max() if HIDE else (y + ey).max()
ax.set_ylim(0, max(0.17, 1.35 * ymax)); ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
if HAS_HEP: hep.cms.label(label="Preliminary", data=False, rlabel="Phase 2 (PU 0)", ax=ax, fontsize=22)
def sci(v, e):
    ex = int(np.floor(np.log10(abs(v)))); return f"({v/10**ex:.3f} ± {e/10**ex:.3f}) × 10$^{{{ex}}}$"
box = dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=1)
ax.text(0.04, 0.96, f"χ² / ndf = {chi2:.2f} / {ndf}\na² = {sci(a2, a2e)}\nc² = {sci(c2, c2e)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=18, bbox=box)
ax.text(0.04, 0.74, r"Fit: $\sqrt{\frac{a^2}{E} + c^2}$" + (f"  (E ≤ {EMAX:.0f} GeV)" if np.isfinite(EMAX) else ""), transform=ax.transAxes, ha="left", va="top", fontsize=20, bbox=box)
ax.text(0.52, 0.96, f"{label}, MaskFormer\nStochastic term a : {100*np.sqrt(a2):.0f}%\nConstant term c : {100*np.sqrt(c2):.1f}%",
        transform=ax.transAxes, ha="left", va="top", fontsize=18, bbox=box)
suffix = (f"_fitE{EMAX:.0f}" if np.isfinite(EMAX) else "") + ("_only" if HIDE else "")
outdir = os.path.join(evaldir, "resolution_plots"); os.makedirs(outdir, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(outdir, f"energy_resolution_{particle}_{variant}{suffix}.{ext}"), dpi=150, bbox_inches="tight")
res = dict(particle=particle, variant=variant, fit_emax=EMAX, drop=[gf[f"{particle}_E{int(e)}"][variant]["drop"] for e in E], E=E.tolist(), sigma_over_E=y.tolist(), err=ey.tolist(), a2=a2, a2_err=a2e, c2=c2, c2_err=c2e,
           a_pct=100*np.sqrt(a2), c_pct=100*np.sqrt(c2), chi2=chi2, ndf=ndf, mplhep=HAS_HEP)
json.dump(res, open(os.path.join(outdir, f"energy_resolution_{particle}_{variant}{suffix}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
