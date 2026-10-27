"""E_pred vs sum of CLUE3D trackster raw energies per event: saturation check at high energy.
Usage: python epred_vs_sumts.py <eval_dir> <particle> <E> [<E> ...]"""
import sys, os, glob, json
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
evaldir, particle = sys.argv[1], sys.argv[2]; Es = [int(e) for e in sys.argv[3:]]
fig, axes = plt.subplots(1, len(Es), figsize=(5.2 * len(Es), 5), squeeze=False); axes = axes[0]
summary = {}
for ax, E in zip(axes, Es):
    f = glob.glob(os.path.join(evaldir, f"{particle}_E{E}", "*__test_*.parquet"))[0]
    t = ak.from_parquet(f, columns=['ticlSimTracksters.*', 'ticlTrackstersCLUE3DHigh.*'])
    p = t['ticlSimTracksters']; tk = t['ticlTrackstersCLUE3DHigh']
    pred = ak.to_numpy(ak.flatten(p['ticlSimTracksters_pred_e'])).astype(float)
    true = ak.to_numpy(ak.flatten(p['ticlSimTracksters_regressed_energy'])).astype(float)
    sumts = ak.to_numpy(ak.sum(tk['ticlTrackstersCLUE3DHigh_raw_energy'], axis=1)).astype(float)
    ok = np.isfinite(pred) & (sumts > 0); pred, true, sumts = pred[ok], true[ok], sumts[ok]
    # profile: median E_pred in bins of sumTS
    edges = np.quantile(sumts, np.linspace(0, 1, 13)); c = 0.5 * (edges[1:] + edges[:-1]); med = np.array([np.median(pred[(sumts >= a) & (sumts < b)]) for a, b in zip(edges[:-1], edges[1:])])
    slope_lo = np.polyfit(sumts[sumts < np.median(sumts)], pred[sumts < np.median(sumts)], 1)[0]
    slope_hi = np.polyfit(sumts[sumts >= np.median(sumts)], pred[sumts >= np.median(sumts)], 1)[0]
    r = np.corrcoef(sumts, pred)[0, 1]
    ax.scatter(sumts, pred, s=6, alpha=0.35, color="#1f3fbf", label="events")
    ax.plot(c, med, "o-", color="#e4002b", ms=5, label="median E_pred in sumTS bins")
    lo, hi = min(sumts.min(), pred.min()) * 0.95, max(sumts.max(), pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="E_pred = sumTS"); ax.axhline(E, color="gray", ls=":", lw=1, label=f"E_true = {E}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("sum of CLUE3D trackster raw energy [GeV]"); ax.set_ylabel("E_pred [GeV]")
    ax.set_title(f"{particle} {E} GeV  (r={r:.2f}, slope lo/hi={slope_lo:.2f}/{slope_hi:.2f})", fontsize=11); ax.legend(fontsize=8, loc="upper left")
    summary[E] = dict(n=int(len(pred)), corr=float(r), slope_below_median=float(slope_lo), slope_above_median=float(slope_hi),
                      sumts_median=float(np.median(sumts)), sumts_q10_q90=[float(x) for x in np.quantile(sumts, [0.1, 0.9])],
                      pred_median=float(np.median(pred)), ratio_pred_over_sumts_q10_50_90=[float(x) for x in np.quantile(pred / sumts, [0.1, 0.5, 0.9])],
                      profile_sumts=c.tolist(), profile_pred_median=med.tolist())
fig.tight_layout(); out = os.path.join(evaldir, "resolution_plots", f"epred_vs_sumts_{particle}.png"); fig.savefig(out, dpi=130)
print(json.dumps(summary, indent=1))
