#!/usr/bin/env python
"""Distributions of the node (topocluster / trackster) input features, raw and transformed.

Writes, under ``<outdir>/topocluster_features/``:

  raw/node_<feat>.png                    the raw feature as read from the file (e, eta, cosphi,
                                         sinphi, r, time) for every node
  transform/node_<feat>_transformed.png  the same after the scale-dict transform the dataset
                                         applies before feeding the network (needs the node
                                         scale dict: ``data.scale_dict_path`` of the training config)

The transform mirrors ``HGCDataset.load_event``: ``e``, ``eta`` and ``r`` are scaled under
their own key, ``time`` under ``t``; ``cosphi``/``sinphi``/``phi`` are passed untransformed
(``hgcal_eval_common.NODE_SCALE_KEY``).  The time is read on demand (it is not part of
``EvalData``) and its ``-99`` "no time" sentinel is counted and left out of the histograms;
the dataset itself scales it as any other value.

Standalone:  python topocluster_features.py --input <pred.parquet> --outdir <dir> [--scale-dict <yaml>]
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import (  # noqa: E402
    CMS_blue, CMS_gray, EvalData, FeatureScalerNP, NODE_FEATURE_INFO, NODE_SCALE_KEY,
    VarTransformNP, read_schema, robust_sigma, savefig, setup_style, stat_label,
)
from dataset.collections import TOPO_FIELD_MAP  # noqa: E402  (importable after hgcal_eval_common)

plt = setup_style()

DEFAULT_NODE_FEATURES = ["e", "eta", "cosphi", "sinphi", "r"]
TIME_SENTINEL = -99.0


def read_time(data):
    """The raw node time, or ``None`` (with a message) when the collection has no time branch."""
    topo = data.topocluster_collection
    branch = TOPO_FIELD_MAP.get(topo, {}).get("time")
    if branch is None or branch not in read_schema(data.path).get(topo, set()):
        print("[topo] %s carries no time branch in %s: 'time' skipped"
              % (topo, os.path.basename(data.path)))
        return None
    arr = ak.from_parquet(data.path, columns=["%s.%s" % (topo, branch)])[:data.n_events]
    t = np.asarray(ak.to_numpy(ak.flatten(arr[topo][branch]))).astype(np.float64)
    if len(t) != len(data.n_energy):
        raise ValueError("%s has %d entries for %d nodes" % (branch, len(t), len(data.n_energy)))
    return t


def node_feature_arrays(data, features):
    """``{feature: raw values per node}`` for the requested input features (unknown ones and
    a missing time are skipped with a message)."""
    out = {}
    for f in features:
        if f in out:
            continue
        if f == "time":
            t = read_time(data)
            if t is not None:
                out[f] = t
            continue
        try:
            out[f] = data.node_feature(f)
        except KeyError:
            print("[topo] unknown node feature %r: skipped" % f)
    return out


def _sentinel(name, x):
    return (x == TIME_SENTINEL) if name == "time" else np.zeros(len(x), bool)


def _bins(x, log, n=80):
    x = x[np.isfinite(x)]
    if log:
        x = x[x > 0]
    if len(x) == 0:
        return np.linspace(0, 1, n + 1)
    lo, hi = np.percentile(x, [0.1, 99.9])
    if log:
        lo = max(lo, 1e-4)
        hi = max(hi, lo * 10)
        return np.logspace(np.log10(lo), np.log10(hi * 1.1), n + 1)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return np.linspace(lo - pad, hi + pad, n + 1)


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=None, std=None, median=None, sigma68=None, n=0)
    return dict(mean=float(x.mean()), std=float(x.std()), median=float(np.median(x)),
                sigma68=float(robust_sigma(x)), n=int(len(x)))


def _info(f):
    return NODE_FEATURE_INFO.get(f, dict(label=f, unit="", log=False))


def _unit(info):
    return (" " + info["unit"]) if info["unit"] else ""


# --------------------------------------------------------------------------- #

def plot_raw(data, outdir, features, metrics):
    sub = os.path.join(outdir, "topocluster_features", "raw")
    m = metrics.setdefault("topocluster_features", {}).setdefault("raw", {})
    arrays = node_feature_arrays(data, features or DEFAULT_NODE_FEATURES)
    for f, x in arrays.items():
        info = _info(f)
        sent = _sentinel(f, x)
        good = np.isfinite(x) & ~sent
        v = x[good]
        n_nonfinite = int((~np.isfinite(x)).sum())
        fig, ax = plt.subplots()
        ax.hist(v, bins=_bins(v, info["log"]), histtype="step", lw=2, color=CMS_blue,
                label="%s %s\n%s" % (data.node_label, info["label"], stat_label(v, _unit(info))))
        if sent.any():
            ax.plot([], [], " ", label="%d nodes with t = %g (no time), not shown" % (sent.sum(), TIME_SENTINEL))
        if n_nonfinite:
            ax.plot([], [], " ", label="%d non-finite, not shown" % n_nonfinite)
        if info["log"]:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("%s %s%s" % (data.node_label, info["label"],
                                   (" [%s]" % info["unit"]) if info["unit"] else ""))
        ax.set_ylabel("%ss / bin" % data.node_label)
        ax.set_title("%s input: raw %s" % (data.node_label, info["label"]), fontsize=20)
        ax.legend(fontsize=12)
        savefig(fig, sub, "node_%s" % f)
        m[f] = dict(_stats(v), n_nonfinite=n_nonfinite, n_sentinel=int(sent.sum()))
    return arrays


def plot_transformed(data, outdir, features, scaler, metrics):
    """``scaler``: a ``FeatureScalerNP`` (or path to the node scale dict)."""
    if not isinstance(scaler, FeatureScalerNP):
        scaler = FeatureScalerNP(scaler)
    sub = os.path.join(outdir, "topocluster_features", "transform")
    m = metrics.setdefault("topocluster_features", {}).setdefault("transform", {})
    arrays = node_feature_arrays(data, features or DEFAULT_NODE_FEATURES)
    for f, x in arrays.items():
        info = _info(f)
        key = NODE_SCALE_KEY.get(f)
        tr = scaler[key] if key is not None else VarTransformNP(f, {"type": "std"})
        sent = _sentinel(f, x)
        y = tr.transform(x)
        good = np.isfinite(y) & ~sent
        v = y[good]
        n_nonfinite = int(((~np.isfinite(y)) & ~sent).sum())
        if tr.type == "min_max":
            lo, hi, what = 0.0, 1.0, "range"
        elif tr.type == "min_max_sym":
            lo, hi, what = -1.0, 1.0, "range"
        else:
            lo, hi, what = -1.0, 1.0, r"$\pm 1\sigma$"
        frac_out = float(np.mean((v < lo) | (v > hi))) if len(v) else 0.0

        fig, ax = plt.subplots()
        how = ("dict key %r: %s%s" % (key, tr.type, (" + " + tr.fn) if tr.fn in tr.FNS else "")
               if key is not None else "not scaled by the dataset (identity)")
        ax.hist(v, bins=_bins(v, False), histtype="step", lw=2, color=CMS_blue,
                label="%s\n%s" % (how, stat_label(v)))
        for g in (lo, hi):
            ax.axvline(g, color=CMS_gray, ls=":", lw=1.5)
        ax.plot([], [], " ", label="outside %s [%g, %g]: %.3f" % (what, lo, hi, frac_out))
        if sent.any():
            ax.plot([], [], " ", label="%d nodes with t = %g (no time), not shown" % (sent.sum(), TIME_SENTINEL))
        if n_nonfinite:
            ax.plot([], [], " ", label="%d non-finite after the transform, not shown" % n_nonfinite)
        ax.set_xlabel("scaled %s %s = %s" % (data.node_label, info["label"], tr.describe()))
        ax.set_ylabel("%ss / bin" % data.node_label)
        ax.set_title("%s input: transformed %s" % (data.node_label, info["label"]), fontsize=20)
        ax.legend(fontsize=12)
        savefig(fig, sub, "node_%s_transformed" % f)
        m[f] = dict(_stats(v), key=key, type=tr.type, fn=tr.fn if tr.fn in tr.FNS else None,
                    shift=tr.shift, scale=tr.scale, frac_outside=frac_out,
                    n_nonfinite=n_nonfinite, n_sentinel=int(sent.sum()))
    return arrays


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="prediction parquet written by HGCPredictionWriter")
    ap.add_argument("--outdir", default="plots/maskformer_eval")
    ap.add_argument("--scale-dict", default=None, help="node scale dict (data.scale_dict_path)")
    ap.add_argument("--node-features", nargs="+", default=None,
                    help="node input features (data.inputs.topocluster), default: %s" % " ".join(DEFAULT_NODE_FEATURES))
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args(argv)
    data = EvalData(args.input, max_events=args.max_events)
    metrics = dict(dataset=data.summary())
    plot_raw(data, args.outdir, args.node_features, metrics)
    if args.scale_dict:
        plot_transformed(data, args.outdir, args.node_features, FeatureScalerNP(args.scale_dict), metrics)
    with open(os.path.join(args.outdir, "topocluster_features", "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[topo] done -> %s" % os.path.join(args.outdir, "topocluster_features"))


if __name__ == "__main__":
    main()
