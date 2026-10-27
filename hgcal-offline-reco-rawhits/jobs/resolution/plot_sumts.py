"""Two-panel plot per particle: top sigma68 of E/E_true vs E for the trackster sum (sumTS) and the regression
(E_pred); bottom median(sumTS/E_true) and median(E_pred/E_true). Usage: python plot_sumts.py <eval_dir> <particle>"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
try:
    import mplhep as hep; plt.style.use(hep.style.CMS); HAS_HEP = True
except Exception:
    HAS_HEP = False
evaldir, particle = sys.argv[1], sys.argv[2]
label = {"photons": "Photons γ", "pions": "Pions π±", "K0L": "K0L"}.get(particle, particle)
gf = json.load(open(os.path.join(evaldir, "gaussfit_summary.json")))
rows = sorted([v for k, v in gf.items() if k.startswith(particle + "_E")], key=lambda v: v["E_nom"])
E = np.array([v["E_nom"] for v in rows]); n = np.array([v["n"] for v in rows])
ts_s68 = np.array([v["masksum"]["sigma68_ratio"] for v in rows]); ts_med = np.array([v["masksum"]["median_ratio"] for v in rows])
pr_s68 = np.array([v["sigma68_ratio"] for v in rows]); pr_med = np.array([v["peak"]["mu"] / v["mean_true"] for v in rows])
# statistical error of sigma68 ~ sigma68 * sqrt(1/(2n)) * 1.2 (robust-width approximation); of the median ~ 1.25 sigma68/sqrt(n)
e_ts = ts_s68 * 1.2 / np.sqrt(2 * n); e_pr = pr_s68 * 1.2 / np.sqrt(2 * n)
em_ts = 1.25 * ts_s68 / np.sqrt(n); em_pr = 1.25 * pr_s68 / np.sqrt(n)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.2], hspace=0.08))
a1.errorbar(E, ts_s68, yerr=e_ts, fmt="o", color="#e4002b", ms=7, label="sum of CLUE3D tracksters (sumTS)")
a1.errorbar(E, pr_s68, yerr=e_pr, fmt="x", color="#1f3fbf", ms=9, mew=1.8, label="MaskFormer regression (E_pred)")
a1.set_ylabel("σ68(E / E_true)"); a1.set_ylim(0, 1.25 * max(ts_s68.max(), pr_s68.max())); a1.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
a1.legend(loc="upper right", fontsize=16, frameon=True); a1.text(0.04, 0.95, f"{label}", transform=a1.transAxes, ha="left", va="top", fontsize=20)
a2.errorbar(E, ts_med, yerr=em_ts, fmt="o", color="#e4002b", ms=7)
a2.errorbar(E, pr_med, yerr=em_pr, fmt="x", color="#1f3fbf", ms=9, mew=1.8)
a2.axhline(1.0, color="gray", ls="--", lw=1); a2.set_ylabel("median(E / E_true)"); a2.set_xlabel("Energy [GeV]")
lo = min(ts_med.min(), pr_med.min()); a2.set_ylim(min(0.7, lo - 0.05), 1.1); a2.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
a2.set_xscale("log"); a2.set_xlim(E.min() / 1.3, E.max() * 1.3)
ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500) if E.min() / 1.3 <= t <= E.max() * 1.3]
a2.xaxis.set_major_locator(FixedLocator(ticks)); a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
a2.xaxis.set_minor_locator(FixedLocator([])); a2.xaxis.set_minor_formatter(NullFormatter())
if HAS_HEP: hep.cms.label(label="Preliminary", data=False, rlabel="Phase 2 (PU 0)", ax=a1, fontsize=22)
outdir = os.path.join(evaldir, "resolution_plots"); os.makedirs(outdir, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(outdir, f"sumTS_vs_regression_{particle}.{ext}"), dpi=150, bbox_inches="tight")
print(f"{'E':>5}{'sumTS s68%':>12}{'sumTS med':>11}{'Epred s68%':>12}{'Epred med':>11}")
for i in range(len(E)): print(f"{E[i]:>5.0f}{100*ts_s68[i]:>12.2f}{ts_med[i]:>11.3f}{100*pr_s68[i]:>12.2f}{pr_med[i]:>11.3f}")
