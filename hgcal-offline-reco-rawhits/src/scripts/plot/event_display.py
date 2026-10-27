#!/usr/bin/env python
"""Event displays: true vs predicted cluster positions, with the node cloud.

Writes, under ``<outdir>/event_display/``, for the first ``n_display`` events of the file:

  event_<i>_3d_clusters.png     3D: true clusters (blue rings) and predicted clusters (red
                                rings; dotted when the query is not pred_valid); the ring area
                                is proportional to the cluster energy
  event_<i>_3d_with_nodes.png   the same with the nodes (topoclusters / tracksters / hits) as
                                gray dots sized by their energy
  event_<i>_projections.png     2D projections y vs x, x vs z, y vs z and rho vs z
                                (rho = transverse radius) with nodes, predicted and true clusters

The true position is the particle position of the file (SimTrackster barycenter, SimCluster
impact point); the predicted one is rebuilt from the regressed (r, eta, phi), r being the 3D
radius (``EvalData.pred_xyz``).  The marker scale is shared by the true and the predicted
clusters of one event, so their sizes can be compared.

Standalone:  python event_display.py --input <pred.parquet> --outdir <dir> [--n-display N]
"""

import argparse
import os
import sys

import numpy as np
import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import CMS_blue, CMS_gray, CMS_red, EvalData, savefig, setup_style  # noqa: E402

plt = setup_style()

#: RecHit events have O(1e5) nodes: more than this is randomly subsampled for the display
NODE_CAP = 200_000
CLUSTER_SIZE = (15.0, 600.0)
NODE_SIZE = (3.0, 80.0)


def marker_sizes(e, e_ref, s_min, s_max):
    """Marker area (points^2) linear in the energy: ``e_ref`` gets ``s_max``."""
    e = np.asarray(e, float)
    e = np.where(np.isfinite(e) & (e > 0), e, 0.0)
    if not np.isfinite(e_ref) or e_ref <= 0:
        return np.full(len(e), s_min)
    return s_min + (s_max - s_min) * np.clip(e / e_ref, 0.0, 1.0)


def event_arrays(data, i, xyz=None, rng=None):
    """``(nodes, true, pred)`` dicts of x, y, z, e (+ ``valid`` for pred) of event ``i``;
    ``pred`` is None when no position was regressed.  ``xyz`` = ``data.pred_xyz()``."""
    ps, ns = data.event_slices(i)
    nodes = dict(x=data.n_x[ns], y=data.n_y[ns], z=data.n_z[ns], e=data.n_energy[ns], capped=False)
    if len(nodes["e"]) > NODE_CAP:
        rng = rng or np.random.default_rng(0)
        keep = np.sort(rng.choice(len(nodes["e"]), NODE_CAP, replace=False))
        nodes = dict({k: nodes[k][keep] for k in ("x", "y", "z", "e")}, capped=True)
    true = dict(x=data.p_x[ps], y=data.p_y[ps], z=data.p_z[ps], e=data.p_e[ps])
    pred = None
    if xyz is not None:
        x, y, z = xyz
        e = data.p_pred_e[ps] if data.p_pred_e is not None else np.zeros(ps.stop - ps.start)
        pred = dict(x=x[ps], y=y[ps], z=z[ps], e=e, valid=data.p_valid[ps])
    return nodes, true, pred


def _limits(arrays, pad=0.05):
    lims = {}
    for k in ("x", "y", "z"):
        v = np.concatenate([np.asarray(a[k], float) for a in arrays if a is not None]) if arrays else np.array([])
        v = v[np.isfinite(v)]
        if len(v) == 0:
            lims[k] = (-1.0, 1.0)
            continue
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo if hi > lo else 1.0
        lims[k] = (lo - pad * span, hi + pad * span)
    return lims


def _title(data, i, true, pred):
    n_valid = int(pred["valid"].sum()) if pred is not None else 0
    e_pred = float(np.nansum(pred["e"])) if pred is not None else float("nan")
    return ("event %d: %d true %ss, %d/%d pred_valid queries, "
            r"$\sum E_{\mathrm{true}}$ = %.1f GeV, $\sum E_{\mathrm{pred}}$ = %.1f GeV"
            % (i, len(true["e"]), data.particle_label, n_valid, len(true["e"]),
               float(np.nansum(true["e"])), e_pred))


def _handles(data, with_nodes, pred):
    h = [Line2D([], [], marker="o", mfc="none", mec=CMS_blue, mew=2, ls="", ms=12,
                label="true %s (ring area ~ E)" % data.particle_label)]
    if pred is not None:
        h.append(Line2D([], [], marker="o", mfc="none", mec=CMS_red, mew=2, ls="", ms=12,
                        label="predicted cluster, pred_valid"))
        if not pred["valid"].all():
            h.append(Line2D([], [], marker="o", mfc="none", mec=CMS_red, mew=1, ls="", ms=12,
                            label="predicted cluster, not pred_valid (dotted ring)"))
    if with_nodes:
        h.append(Line2D([], [], marker="o", color=CMS_gray, ls="", ms=8, alpha=0.6,
                        label="%s (dot area ~ E)" % data.node_label))
    return h


def _rings(ax, pts, s, color, valid=None, three_d=False):
    """Hollow circles; ``pts`` = (a, b[, c]) in axis order."""
    kw = dict(facecolors="none", edgecolors=color, linewidths=1.8)
    if three_d:
        kw["depthshade"] = False   # depth shading fades the edges of hollow markers
    if valid is None:
        valid = np.ones(len(s), bool)
    if valid.any():
        ax.scatter(*[p[valid] for p in pts], s=s[valid], linestyles="-", **kw)
    if (~valid).any():
        ax.scatter(*[p[~valid] for p in pts], s=s[~valid], linestyles=":", **kw)


def plot_event_3d(data, i, outdir, with_nodes, xyz=None):
    nodes, true, pred = event_arrays(data, i, xyz)
    e_ref = max(float(np.nanmax(true["e"])) if len(true["e"]) else 0.0,
                float(np.nanmax(pred["e"])) if pred is not None and len(pred["e"]) else 0.0)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection="3d")
    ax.grid(False)   # the dotted 2D grid of the CMS style turns the 3D panes into noise
    # z along the horizontal axis: the shower develops along it
    if with_nodes and len(nodes["e"]):
        ax.scatter(nodes["z"], nodes["x"], nodes["y"],
                   s=marker_sizes(nodes["e"], float(np.nanmax(nodes["e"])), *NODE_SIZE),
                   c=CMS_gray, alpha=0.5, linewidths=0, depthshade=False)
    _rings(ax, (true["z"], true["x"], true["y"]), marker_sizes(true["e"], e_ref, *CLUSTER_SIZE),
           CMS_blue, three_d=True)
    if pred is not None:
        _rings(ax, (pred["z"], pred["x"], pred["y"]), marker_sizes(pred["e"], e_ref, *CLUSTER_SIZE),
               CMS_red, valid=pred["valid"], three_d=True)
    lims = _limits([nodes if with_nodes else None, true, pred])
    ax.set_xlim(*lims["z"])
    ax.set_ylim(*lims["x"])
    ax.set_zlim(*lims["y"])
    ext = np.array([lims["z"][1] - lims["z"][0], lims["x"][1] - lims["x"][0], lims["y"][1] - lims["y"][0]])
    ext = np.maximum(ext, 0.1 * ext.max())   # equal scale, but no axis collapses to a line
    # zoom < 1 leaves room for the axis labels, which the tight bbox of 3D axes clips otherwise
    ax.set_box_aspect(tuple(ext), zoom=0.85)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))   # a short axis must not pile up its ticks
    ax.set_xlabel("z [cm]", labelpad=14, fontsize=14)
    ax.set_ylabel("x [cm]", labelpad=14, fontsize=14)
    ax.set_zlabel("y [cm]", labelpad=14, fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(handles=_handles(data, with_nodes, pred), fontsize=11, loc="upper left")
    title = _title(data, i, true, pred)
    if with_nodes and nodes["capped"]:
        title += "\n(%d of the nodes shown)" % NODE_CAP
    ax.set_title(title, fontsize=13)
    savefig(fig, outdir, "event_%d_3d_%s" % (i, "with_nodes" if with_nodes else "clusters"))


def plot_event_projections(data, i, outdir, xyz=None):
    nodes, true, pred = event_arrays(data, i, xyz)
    for d in (nodes, true, pred):
        if d is not None:
            d["rho"] = np.hypot(d["x"], d["y"])
    e_ref = max(float(np.nanmax(true["e"])) if len(true["e"]) else 0.0,
                float(np.nanmax(pred["e"])) if pred is not None and len(pred["e"]) else 0.0)
    s_true = marker_sizes(true["e"], e_ref, *CLUSTER_SIZE)
    s_pred = marker_sizes(pred["e"], e_ref, *CLUSTER_SIZE) if pred is not None else None
    s_node = marker_sizes(nodes["e"], float(np.nanmax(nodes["e"])) if len(nodes["e"]) else 0.0, *NODE_SIZE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    panels = [("x", "y", axes[0, 0]), ("z", "x", axes[0, 1]), ("z", "y", axes[1, 0]), ("z", "rho", axes[1, 1])]
    labels = {"x": "x [cm]", "y": "y [cm]", "z": "z [cm]", "rho": r"$\rho = \sqrt{x^2 + y^2}$ [cm]"}
    for a, b, ax in panels:
        if len(nodes["e"]):
            ax.scatter(nodes[a], nodes[b], s=s_node, c=CMS_gray, alpha=0.5, linewidths=0)
        _rings(ax, (true[a], true[b]), s_true, CMS_blue)
        if pred is not None:
            _rings(ax, (pred[a], pred[b]), s_pred, CMS_red, valid=pred["valid"])
        ax.set_xlabel(labels[a])
        ax.set_ylabel(labels[b])
        if (a, b) == ("x", "y"):
            ax.set_aspect("equal", adjustable="datalim")
    fig.legend(handles=_handles(data, True, pred), fontsize=12, loc="lower center",
               ncol=2 if pred is None else 4, bbox_to_anchor=(0.5, -0.01))
    title = _title(data, i, true, pred)
    if nodes["capped"]:
        title += "  (%d of the nodes shown)" % NODE_CAP
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    savefig(fig, outdir, "event_%d_projections" % i)


def plot_event_displays(data, outdir, n_display=5):
    """Draw the three displays of the first ``n_display`` non-empty events; returns their number."""
    sub = os.path.join(outdir, "event_display")
    xyz = data.pred_xyz()
    if xyz is None:
        print("[display] r, eta and phi were not all regressed: the predicted clusters are not drawn")
    shown = 0
    for i in range(data.n_events):
        if shown >= n_display:
            break
        ps, ns = data.event_slices(i)
        if ps.stop == ps.start and ns.stop == ns.start:
            continue
        plot_event_3d(data, i, sub, with_nodes=False, xyz=xyz)
        plot_event_3d(data, i, sub, with_nodes=True, xyz=xyz)
        plot_event_projections(data, i, sub, xyz=xyz)
        shown += 1
    print("[display] %d event(s) drawn under %s" % (shown, sub))
    return shown


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="prediction parquet written by HGCPredictionWriter")
    ap.add_argument("--outdir", default="plots/maskformer_eval")
    ap.add_argument("--n-display", type=int, default=5, help="number of events to draw")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args(argv)
    data = EvalData(args.input, max_events=args.max_events)
    plot_event_displays(data, args.outdir, args.n_display)


if __name__ == "__main__":
    main()
