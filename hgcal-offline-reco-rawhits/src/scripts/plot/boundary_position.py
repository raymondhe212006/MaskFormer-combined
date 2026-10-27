#!/usr/bin/env python
"""Performance of the entrance-position regression (the boundary targets).

The boundary targets are the point where the true particle crossed the HGCAL front face
(``ticlSimTracksters_boundaryX/Y``, the seed SimCluster's impact point; see EXTRA_TARGETS in
``src/dataset/dataset_maskformers.py``).  Writes, under ``<outdir>/boundary_position/``:

  xy_pred_vs_true.png           predicted vs true x and y (2D), with the identity line
  residual_xy_2d.png            (x_pred - x_true, y_pred - y_true) with the d68 circle
  distance.png                  transverse distance d = |(x,y)_pred - (x,y)_true|: the model
                                against the two clustering-only baselines (below)
  distance_logx.png             the same on a log axis (tails)
  distance_vs_energy.png        median and 68% quantile of d in bins of the true energy
  distance_vs_eta.png           ... and of the true eta
  distance_by_class.png         d per true particle class
  residual_vs_true_<x|y>.png    median +- sigma68 of the x (y) residual in bins of the true
                                x (y): edge effects of the front face
  entry_point_map.png           true and predicted entry points on the front face

Baselines (what one gets without a position regression): the energy-weighted barycenter of
the nodes assigned to the particle, projected onto the TRUE boundary z along the straight
line from the origin, using (a) the nodes the model assigned (``pred_match_idx``) and (b) the
truth-assigned nodes (the ceiling of that recipe with perfect clustering).  Particles without
nodes have no baseline and are left out of the baseline histograms.

The 1D residuals of x and y alone, by class and vs kinematics, are in ``resolution/`` and
``resolution_by_class/`` like every other regressed feature; the response plots skip x and y
(signed, centred on zero).  Nothing is drawn when the model did not regress boundary_x and
boundary_y or when the file has no boundary truth.

Standalone:  python boundary_position.py --input <pred.parquet> --outdir <dir>
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import (  # noqa: E402
    CMS_blue, CMS_gray, CMS_orange, CMS_red, EvalData, PHYSICS_CLASSES, class_color, class_name,
    profile, robust_sigma, savefig, setup_style, stat_label,
)
from plot_maskformer_eval import E_EDGES, ETA_EDGES, energy_axis_label  # noqa: E402

plt = setup_style()

#: histogram range of the transverse distance [cm]
D_MAX = 40.0
#: range of the 2D residual and of the 1D residual profiles [cm]
RES_MAX = 30.0
#: the HGCAL front face spans |x|, |y| < ~250 cm
XY_EDGES = np.linspace(-250.0, 250.0, 21)
XY_MAP_EDGES = np.linspace(-260.0, 260.0, 105)


def masks(data):
    """``(tag, legend name, mask, colour)`` of the two query selections used everywhere."""
    return [("all", "all matched queries", np.ones(len(data.p_e), bool), CMS_blue),
            ("valid", "pred_valid only", data.p_valid, CMS_red)]


def _dstats(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(median=None, d68=None, d95=None, mean=None, n=0)
    return dict(median=float(np.median(d)), d68=float(np.percentile(d, 68.0)),
                d95=float(np.percentile(d, 95.0)), mean=float(d.mean()), n=int(len(d)))


def _dlabel(d, unit=" cm"):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return "empty"
    return (r"$\mu_{1/2}$=%.3g%s  $d_{68}$=%.3g%s  N=%d"
            % (np.median(d), unit, np.percentile(d, 68.0), unit, len(d)))


def distance(data, mask):
    """Transverse distance between the predicted and the true entry point per particle, NaN
    outside ``mask`` and where either is not finite."""
    dx = data.p_pred_boundary_x - data.p_boundary_x
    dy = data.p_pred_boundary_y - data.p_boundary_y
    d = np.hypot(dx, dy)
    return np.where(mask & np.isfinite(d), d, np.nan)


def node_barycenter_baseline(data, assignment):
    """Entry point estimated from the nodes alone: the energy-weighted barycenter of the nodes
    whose ``assignment`` (``data.n_pred`` or ``data.n_true``, global particle index, -1 for
    none) points to the particle, projected onto the particle's true boundary z along the
    line from the origin.  Returns ``(x, y)`` per particle, NaN without nodes."""
    n = len(data.p_e)
    sel = assignment >= 0
    idx = assignment[sel]
    w = data.n_energy[sel]
    sw = np.bincount(idx, weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        bx = np.bincount(idx, weights=w * data.n_x[sel], minlength=n) / sw
        by = np.bincount(idx, weights=w * data.n_y[sel], minlength=n) / sw
        bz = np.bincount(idx, weights=w * data.n_z[sel], minlength=n) / sw
        zb = data.p_boundary_z if data.p_boundary_z is not None else np.sign(bz) * 321.05
        scale = np.where((sw > 0) & (bz != 0), zb / bz, np.nan)
    return bx * scale, by * scale


def _quantile_profile(x, d, edges, qs=(50.0, 68.0), min_n=3):
    x, d = np.asarray(x, float), np.asarray(d, float)
    good = np.isfinite(x) & np.isfinite(d)
    x, d = x[good], d[good]
    which = np.digitize(x, edges) - 1
    ctr = 0.5 * (edges[:-1] + edges[1:])
    out = {q: np.full(len(ctr), np.nan) for q in qs}
    n = np.zeros(len(ctr), dtype=int)
    for i in range(len(ctr)):
        sel = which == i
        n[i] = sel.sum()
        if n[i] < min_n:
            continue
        for q in qs:
            out[q][i] = np.percentile(d[sel], q)
    return ctr, out, n


def _none_where_nan(a):
    return np.where(np.isfinite(a), a, None).tolist()


# --------------------------------------------------------------------------- #

def plot_boundary_position(data, outdir, metrics):
    if not data.has_boundary_pred:
        print("[boundary] boundary_x / boundary_y were not regressed: entrance-position plots skipped")
        return
    if not data.has_boundary_truth:
        print("[boundary] the file carries no boundary truth (%s): entrance-position plots skipped"
              % data.particle_collection)
        return
    sub = os.path.join(outdir, "boundary_position")
    m = metrics.setdefault("boundary_position", {})
    sel = masks(data)
    label = data.particle_label
    xt, yt = data.p_boundary_x, data.p_boundary_y
    xp, yp = data.p_pred_boundary_x, data.p_pred_boundary_y
    dx, dy = xp - xt, yp - yt
    fin = np.isfinite(dx) & np.isfinite(dy)

    # ---- 1. predicted vs true, per coordinate ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (name, t, p) in zip(axes, [("x", xt, xp), ("y", yt, yp)]):
        good = np.isfinite(t) & np.isfinite(p)
        ax.hist2d(t[good], p[good], bins=[XY_MAP_EDGES, XY_MAP_EDGES], cmap="viridis", cmin=1)
        ax.plot(XY_MAP_EDGES[[0, -1]], XY_MAP_EDGES[[0, -1]], "k:", lw=1.2, label="pred = true")
        ax.set_xlabel(r"true %s $%s^{\mathrm{b}}$ [cm]" % (label, name))
        ax.set_ylabel(r"predicted $%s^{\mathrm{b}}$ [cm]" % name)
        ax.set_title("entry point %s: predicted vs true" % name, fontsize=18)
        ax.legend(fontsize=12, loc="upper left")
    savefig(fig, sub, "xy_pred_vs_true")

    # ---- 2. 2D residual ----
    d_all = distance(data, sel[0][2])
    d68 = float(np.nanpercentile(d_all, 68.0)) if np.isfinite(d_all).any() else np.nan
    fig, ax = plt.subplots(figsize=(9, 8))
    edges = np.linspace(-RES_MAX, RES_MAX, 121)
    h = ax.hist2d(dx[fin], dy[fin], bins=[edges, edges], cmap="viridis", cmin=1)
    fig.colorbar(h[3], ax=ax, fraction=0.046, label="%ss / bin" % label)
    if np.isfinite(d68):
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(d68 * np.cos(t), d68 * np.sin(t), color=CMS_red, lw=2, label=r"$d_{68}$ = %.2f cm" % d68)
    ax.axhline(0, color="k", ls=":", lw=1); ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_xlabel(r"$x^{\mathrm{b}}_{\mathrm{pred}} - x^{\mathrm{b}}_{\mathrm{true}}$ [cm]")
    ax.set_ylabel(r"$y^{\mathrm{b}}_{\mathrm{pred}} - y^{\mathrm{b}}_{\mathrm{true}}$ [cm]")
    ax.set_title("Entry-point residual (all matched queries)", fontsize=19)
    ax.set_aspect("equal")
    ax.legend(fontsize=13, loc="upper right")
    ax.grid(False)
    savefig(fig, sub, "residual_xy_2d")
    m["residual"] = {}
    for key, res in (("x", dx), ("y", dy)):
        r = res[fin]
        m["residual"][key] = dict(median=float(np.median(r)) if len(r) else None,
                                  sigma68=float(robust_sigma(r)) if len(r) else None,
                                  mean=float(r.mean()) if len(r) else None, n=int(len(r)))

    # ---- 3. transverse distance: model vs the clustering-only baselines ----
    base = {}
    for tag, name, assignment, color in [
            ("baseline_pred_nodes", "node barycenter of the predicted mask\n(projected to the boundary z)", data.n_pred, CMS_orange),
            ("baseline_true_nodes", "node barycenter of the TRUE nodes\n(projected; perfect clustering)", data.n_true, CMS_gray)]:
        bx, by = node_barycenter_baseline(data, assignment)
        base[tag] = (name, np.hypot(bx - xt, by - yt), color)
    series = [("model_%s" % tag, "model, %s" % name, distance(data, mask), color, "-") for tag, name, mask, color in sel]
    series += [(tag, name, d, color, "--") for tag, (name, d, color) in base.items()]
    m["distance"] = {}
    for logx in (False, True):
        fig, ax = plt.subplots()
        bins = np.logspace(-2, np.log10(300.0), 81) if logx else np.linspace(0.0, D_MAX, 101)
        for tag, name, d, color, ls in series:
            v = d[np.isfinite(d)]
            if logx:
                v = v[v > 0]
            ax.hist(v, bins=bins, histtype="step", lw=2, color=color, ls=ls,
                    label="%s\n%s" % (name, _dlabel(v)))
            if not logx:
                m["distance"][tag] = dict(_dstats(d), overflow_frac=float(np.mean(v > D_MAX)) if len(v) else 0.0)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(r"$d = |(x,y)^{\mathrm{b}}_{\mathrm{pred}} - (x,y)^{\mathrm{b}}_{\mathrm{true}}|$ [cm]")
        ax.set_ylabel("%ss / bin" % label)
        ax.set_title("Entry-point resolution: transverse distance", fontsize=19)
        ax.legend(fontsize=10)
        savefig(fig, sub, "distance_logx" if logx else "distance")

    # ---- 4. vs energy / eta ----
    d_model = {tag: distance(data, mask) for tag, _, mask, _ in sel}
    for xkey, xarr, edges, xlabel, logx in [
            ("energy", data.p_e, E_EDGES, energy_axis_label(data), True),
            ("eta", data.p_eta, ETA_EDGES, r"true %s $\eta$" % label, False)]:
        fig, ax = plt.subplots()
        curves = [("model_%s" % tag, name, d_model[tag], color, "o-") for tag, name, _, color in sel]
        curves += [(tag, name, d, color, "s--") for tag, (name, d, color) in base.items()]
        for tag, name, d, color, fmt in curves:
            ctr, q, n = _quantile_profile(xarr, d, edges)
            ok = np.isfinite(q[50.0])
            ax.plot(ctr[ok], q[50.0][ok], fmt, color=color, lw=2, ms=7, label="%s: median" % name.split("\n")[0])
            ax.plot(ctr[ok], q[68.0][ok], fmt, color=color, lw=1.2, ms=4, alpha=0.6,
                    label=r"%s: $d_{68}$" % name.split("\n")[0])
            if xkey == "energy":
                m.setdefault("distance_vs_energy", {})[tag] = dict(
                    centre=ctr.tolist(), median=_none_where_nan(q[50.0]), d68=_none_where_nan(q[68.0]), n=n.tolist())
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$d$ [cm]  (median, $d_{68}$)")
        ax.set_ylim(bottom=0)
        ax.set_title("Entry-point distance vs true %s" % xkey, fontsize=19)
        ax.legend(fontsize=9, ncol=2)
        savefig(fig, sub, "distance_vs_%s" % xkey)

    # ---- 5. by class ----
    fig, ax = plt.subplots()
    bins = np.linspace(0.0, D_MAX, 81)
    m["by_class"] = {}
    for c in PHYSICS_CLASSES:
        mask = sel[0][2] & (data.p_class == c)
        if mask.sum() < 10:
            continue
        d = distance(data, mask)
        d = d[np.isfinite(d)]
        ax.hist(d, bins=bins, histtype="step", lw=2, density=True, color=class_color(c),
                label="%s: med %.2f, $d_{68}$ %.2f cm (N=%d)" % (class_name(c), np.median(d), np.percentile(d, 68.0), len(d)))
        m["by_class"][class_name(c)] = _dstats(d)
    ax.set_xlabel(r"$d$ [cm]")
    ax.set_ylabel("normalised %ss / bin" % label)
    ax.set_title("Entry-point distance by true class (all queries)", fontsize=19)
    ax.legend(fontsize=12)
    savefig(fig, sub, "distance_by_class")

    # ---- 6. residual vs the true coordinate (edge effects) ----
    for key, t, res in (("x", xt, dx), ("y", yt, dy)):
        fig, ax = plt.subplots()
        for tag, name, mask, color in sel:
            r = np.where(mask, res, np.nan)
            ctr, med, sig, n = profile(t, r, XY_EDGES)
            ok = np.isfinite(med)
            ax.errorbar(ctr[ok], med[ok], yerr=np.where(np.isfinite(sig[ok]), sig[ok], 0.0),
                        fmt="o-", color=color, lw=2, ms=7, capsize=3, label=name)
        ax.axhline(0.0, color="k", ls=":", lw=1.2)
        ax.set_xlabel(r"true %s $%s^{\mathrm{b}}$ [cm]" % (label, key))
        ax.set_ylabel(r"$%s^{\mathrm{b}}_{\mathrm{pred}} - %s^{\mathrm{b}}_{\mathrm{true}}$ [cm]  (median $\pm\,\sigma_{68}$)" % (key, key))
        ax.set_title("Entry-point %s residual vs true %s" % (key, key), fontsize=19)
        ax.legend(fontsize=13)
        savefig(fig, sub, "residual_vs_true_%s" % key)

    # ---- 7. where on the front face ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (name, x, y) in zip(axes, [("true", xt, yt), ("predicted (all matched queries)", xp, yp)]):
        good = np.isfinite(x) & np.isfinite(y)
        ax.hist2d(x[good], y[good], bins=[XY_MAP_EDGES, XY_MAP_EDGES], cmap="viridis", cmin=1)
        ax.set_xlabel(r"$x^{\mathrm{b}}$ [cm]"); ax.set_ylabel(r"$y^{\mathrm{b}}$ [cm]")
        ax.set_title("%s entry points" % name, fontsize=17)
        ax.set_aspect("equal")
        ax.grid(False)
    savefig(fig, sub, "entry_point_map")


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="prediction parquet written by HGCPredictionWriter")
    ap.add_argument("--outdir", default="plots/maskformer_eval")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args(argv)
    data = EvalData(args.input, max_events=args.max_events)
    metrics = dict(dataset=data.summary())
    plot_boundary_position(data, args.outdir, metrics)
    os.makedirs(os.path.join(args.outdir, "boundary_position"), exist_ok=True)
    with open(os.path.join(args.outdir, "boundary_position", "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[boundary] done -> %s" % os.path.join(args.outdir, "boundary_position"))


if __name__ == "__main__":
    main()
