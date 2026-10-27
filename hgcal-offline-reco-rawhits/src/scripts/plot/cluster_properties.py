#!/usr/bin/env python
"""Distributions and responses of the regressed cluster properties.

Writes, under ``<outdir>/cluster_properties/``:

  dist_<feat>.png                 true vs predicted distribution of e, eta, cosphi, sinphi, r
                                  (plus dist_e_logy) and of the boundary targets when regressed
  response/response_<feat>.png    1D response feat_pred / feat_true (not for the signed
                                  entry-point coordinates boundary_x / boundary_y)
  response/by_<x>/response_<feat>_vs_<x>.png
                                  median response (error bars: sigma68) in bins of the TRUE x,
                                  for every (feat, x) pair: energy response vs eta, eta response
                                  vs energy, ...

The predictions are the Hungarian-matched query of every true particle (see ``EvalData``), so
every true particle enters once; "pred_valid only" keeps the queries whose predicted class is
not the null class, as in the other modules.  The response is the plain ratio pred/true:
entries whose |true| is below ``RATIO_MIN_TRUE`` (a sin/cos phi crossing zero) are dropped
and counted; the ``resolution/`` residuals stay the robust reference for those.

Standalone:  python cluster_properties.py --input <pred.parquet> --outdir <dir>
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import (  # noqa: E402
    CMS_blue, CMS_gray, CMS_red, EvalData, profile, robust_sigma, savefig, setup_style,
    stat_label,
)
from plot_maskformer_eval import E_EDGES, ETA_EDGES, active_variables  # noqa: E402

plt = setup_style()

#: histogram range of pred/true per feature
RATIO_RANGE = {"e": (0.0, 3.0), "eta": (0.8, 1.2), "r": (0.8, 1.2),
               "sinphi": (-1.0, 3.0), "cosphi": (-1.0, 3.0),
               "boundary_z": (0.8, 1.2), "boundary_eta": (0.8, 1.2),
               "boundary_sinphi": (-1.0, 3.0), "boundary_cosphi": (-1.0, 3.0)}
#: entries with |true| below this are dropped from the ratio (and counted)
RATIO_MIN_TRUE = {"e": 1e-3, "eta": 0.1, "r": 1.0, "sinphi": 0.1, "cosphi": 0.1,
                  "boundary_z": 1.0, "boundary_eta": 0.1, "boundary_sinphi": 0.1, "boundary_cosphi": 0.1}
#: signed transverse coordinates of the entry point, centred on zero: pred/true is
#: meaningless for them, so they get the distributions only (their residuals are in
#: resolution/ and the dedicated boundary_position/ plots)
NO_RATIO = {"boundary_x", "boundary_y"}
#: the particle r is the 3D radius of the shower position, ~320-350 cm at the HGCAL front face
R_EDGES = np.array([320, 325, 330, 335, 340, 345, 350, 360, 375, 400, 450, 520, 600], float)
PHI_EDGES = np.linspace(-1.0, 1.0, 11)
#: transverse entry point: the HGCAL front face spans |x|, |y| < ~250 cm
XY_EDGES = np.linspace(-250.0, 250.0, 11)
BIN_EDGES = {"e": E_EDGES, "eta": ETA_EDGES, "cosphi": PHI_EDGES, "sinphi": PHI_EDGES, "r": R_EDGES,
             "boundary_x": XY_EDGES, "boundary_y": XY_EDGES, "boundary_eta": ETA_EDGES,
             "boundary_sinphi": PHI_EDGES, "boundary_cosphi": PHI_EDGES}
LOG_X = {"e"}


def ratio_variables(data):
    """The regressed variables for which the response pred/true is a meaningful quantity."""
    return [v for v in active_variables(data) if v["key"] not in NO_RATIO]


def masks(data):
    """``(tag, legend name, mask, colour)`` of the two query selections used everywhere."""
    return [("all", "all matched queries", np.ones(len(data.p_e), bool), CMS_blue),
            ("valid", "pred_valid only", data.p_valid, CMS_red)]


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(median=None, sigma68=None, mean=None, n=0)
    return dict(median=float(np.median(x)), sigma68=float(robust_sigma(x)),
                mean=float(x.mean()), n=int(len(x)))


def _sym(var):
    return var["label"].strip("$")


def ratio_label(var):
    return r"$%s_{\mathrm{pred}} \, / \, %s_{\mathrm{true}}$" % (_sym(var), _sym(var))


def feature_label(data, var, true=True):
    unit = (" [%s]" % var["unit"].strip()) if var["unit"].strip() else ""
    return "%s%s %s%s" % ("true " if true else "", data.particle_label, var["label"], unit)


def response(data, var, mask):
    """pred/true per particle; NaN outside ``mask``, for non-finite values and where
    ``|true| < RATIO_MIN_TRUE``."""
    t = getattr(data, var["truth"])
    p = getattr(data, var["pred"])
    eps = RATIO_MIN_TRUE.get(var["key"], 0.0)
    ok = mask & np.isfinite(t) & np.isfinite(p) & (np.abs(t) >= eps) & (t != 0)
    out = np.full(len(t), np.nan)
    out[ok] = p[ok] / t[ok]
    return out


def _bins(values, log, n=60):
    v = np.concatenate([np.asarray(x, float) for x in values])
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if len(v) == 0:
        return np.linspace(0, 1, n + 1)
    lo, hi = np.percentile(v, [0.1, 99.9])
    if log:
        lo = max(lo, 1e-3)
        hi = max(hi, lo * 10)
        return np.logspace(np.log10(lo), np.log10(hi * 1.1), n + 1)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return np.linspace(lo - pad, hi + pad, n + 1)


def _quantile_edges(x, n=10):
    x = x[np.isfinite(x)]
    if len(x) < 2 * n:
        return np.linspace(x.min(), x.max(), n + 1) if len(x) else np.linspace(0, 1, n + 1)
    return np.unique(np.percentile(x, np.linspace(1, 99, n + 1)))


# --------------------------------------------------------------------------- #
# 1. distributions
# --------------------------------------------------------------------------- #

def plot_distributions(data, outdir, metrics):
    sub = os.path.join(outdir, "cluster_properties")
    m = metrics.setdefault("cluster_properties", {}).setdefault("distributions", {})
    for var in active_variables(data):
        key = var["key"]
        t = getattr(data, var["truth"])
        p = getattr(data, var["pred"])
        log = key in LOG_X
        bins = _bins([t, p], log)
        sel = [("true %s" % data.particle_label, t, np.ones(len(t), bool), CMS_gray, "-"),
               ("predicted, all matched queries", p, np.ones(len(p), bool), CMS_blue, "-"),
               ("predicted, pred_valid only", p, data.p_valid, CMS_red, "--")]
        for logy in ((False, True) if log else (False,)):
            fig, ax = plt.subplots()
            for name, arr, mask, color, ls in sel:
                v = arr[mask]
                v = v[np.isfinite(v)]
                ax.hist(v, bins=bins, histtype="step", lw=2, color=color, ls=ls,
                        label="%s\n%s" % (name, stat_label(v, var["unit"])))
            if log:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(feature_label(data, var, true=False))
            ax.set_ylabel("%ss / bin" % data.particle_label)
            ax.set_title("%s: true vs predicted" % var["label"], fontsize=20)
            ax.legend(fontsize=11)
            savefig(fig, sub, "dist_%s%s" % (key, "_logy" if logy else ""))
        m[key] = dict(true=_stats(t), pred_all=_stats(p), pred_valid=_stats(p[data.p_valid]))


# --------------------------------------------------------------------------- #
# 2. 1D response
# --------------------------------------------------------------------------- #

def plot_response(data, outdir, metrics):
    sub = os.path.join(outdir, "cluster_properties", "response")
    m = metrics.setdefault("cluster_properties", {}).setdefault("response", {})
    for var in ratio_variables(data):
        key = var["key"]
        rng = RATIO_RANGE.get(key, (0.0, 2.0))
        bins = np.linspace(rng[0], rng[1], 101)
        eps = RATIO_MIN_TRUE.get(key, 0.0)
        fig, ax = plt.subplots()
        for tag, name, mask, color in masks(data):
            r = response(data, var, mask)
            fin = np.isfinite(r)
            v = r[fin]
            n_dropped = int(mask.sum() - fin.sum())
            overflow = float(np.mean((v < rng[0]) | (v > rng[1]))) if len(v) else 0.0
            ax.hist(v, bins=bins, histtype="step", lw=2, color=color,
                    label="%s\n%s" % (name, stat_label(v)))
            m.setdefault(key, {})[tag] = dict(_stats(v), n_dropped=n_dropped, overflow_frac=overflow)
        if eps > 0:
            ax.plot([], [], " ", label=r"$|%s_{\mathrm{true}}| < %g$ dropped" % (_sym(var), eps))
        ax.axvline(1.0, color="k", ls=":", lw=1.2)
        ax.set_xlabel(ratio_label(var))
        ax.set_ylabel("%ss / bin" % data.particle_label)
        ax.set_title("%s response" % var["label"], fontsize=20)
        ax.legend(fontsize=12)
        savefig(fig, sub, "response_%s" % key)


# --------------------------------------------------------------------------- #
# 3. response in bins of every feature
# --------------------------------------------------------------------------- #

def plot_response_vs_features(data, outdir, metrics):
    base = os.path.join(outdir, "cluster_properties", "response")
    m = metrics.setdefault("cluster_properties", {}).setdefault("response_vs", {})
    # binned in every regressed variable (the entry point included: edge effects), the
    # response of the ratio-able ones
    variables = ratio_variables(data)
    for xvar in active_variables(data):
        xkey = xvar["key"]
        x = getattr(data, xvar["truth"])
        edges = BIN_EDGES.get(xkey)
        if edges is None:
            edges = _quantile_edges(x)
        sub = os.path.join(base, "by_%s" % xkey)
        for yvar in variables:
            ykey = yvar["key"]
            fig, ax = plt.subplots()
            for tag, name, mask, color in masks(data):
                r = response(data, yvar, mask)
                ctr, med, sig, n = profile(x, r, edges)
                ok = np.isfinite(med)
                ax.errorbar(ctr[ok], med[ok], yerr=np.where(np.isfinite(sig[ok]), sig[ok], 0.0),
                            fmt="o-", color=color, lw=2, ms=7, capsize=3, label=name)
                if tag == "all":
                    m.setdefault(ykey, {})[xkey] = dict(
                        centre=ctr.tolist(),
                        median=np.where(np.isfinite(med), med, None).tolist(),
                        sigma68=np.where(np.isfinite(sig), sig, None).tolist(),
                        n=n.tolist())
            ax.axhline(1.0, color="k", ls=":", lw=1.2)
            if xkey in LOG_X:
                ax.set_xscale("log")
            ax.set_xlabel(feature_label(data, xvar, true=True))
            ax.set_ylabel(ratio_label(yvar) + r"  (median $\pm\,\sigma_{68}$)")
            ax.set_title("%s response vs true %s" % (yvar["label"], xvar["label"]), fontsize=20)
            ax.legend(fontsize=13)
            savefig(fig, sub, "response_%s_vs_%s" % (ykey, xkey))


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
    plot_distributions(data, args.outdir, metrics)
    plot_response(data, args.outdir, metrics)
    plot_response_vs_features(data, args.outdir, metrics)
    with open(os.path.join(args.outdir, "cluster_properties", "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[cluster] done -> %s" % os.path.join(args.outdir, "cluster_properties"))


if __name__ == "__main__":
    main()
