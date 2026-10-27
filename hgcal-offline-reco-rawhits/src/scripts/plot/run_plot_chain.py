#!/usr/bin/env python
"""Standard plotting chain for one MaskFormer prediction file.

Loads the file once (``hgcal_eval_common.EvalData``) and runs, section by section,

  plot_maskformer_eval   -> <outdir>/{resolution,resolution_by_class,efficiency,diagnostics}/
                            <outdir>/metrics.json, <outdir>/summary.txt
  cluster_properties     -> <outdir>/cluster_properties/{,response/,response/by_<feat>/}
  boundary_position      -> <outdir>/boundary_position/ (when the entry point was regressed)
  topocluster_features   -> <outdir>/topocluster_features/{raw,transform}/ (transform needs --scale-dict)
  event_reconstruction   -> <outdir>/event_reconstruction/
  event_display          -> <outdir>/event_display/ (the first --n-display events)
  energy_investigation   -> <outdir>/energy/*.png, energy_metrics.json, energy_summary.txt

A failing section is reported (and recorded in ``<outdir>/plot_chain_failures.txt``) without
stopping the others.  This is what ``src/scripts/evaluate/evaluate_maskformer.py`` runs after
the test loop; it only needs awkward, numpy, pyarrow and matplotlib (mplhep optional).

Usage
-----
    python src/scripts/plot/run_plot_chain.py \\
        --input <ckpt_dir>/<ckpt>__test.parquet --outdir <ckpt_dir>/<ckpt>__test_plots \\
        [--scale-dict configs/<node_var_transform>.yaml --node-features e eta cosphi sinphi r]

The (topocluster, particle) collections are auto-detected from the file unless given; the
node input features and their scale dict are those of the training config (``data.inputs.topocluster``,
``data.scale_dict_path``), which evaluate_maskformer.py passes on.

Exit code: 0 all sections done, 1 some section failed, 2 the file could not be loaded.
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import EvalData, FeatureScalerNP  # noqa: E402
import plot_maskformer_eval as ev  # noqa: E402
import energy_investigation as en  # noqa: E402
import cluster_properties as cp  # noqa: E402
import boundary_position as bp  # noqa: E402
import topocluster_features as tf  # noqa: E402
import event_reconstruction as er  # noqa: E402
import event_display as ed  # noqa: E402


def _run(name, fn, failures):
    print("[chain] %s ..." % name, flush=True)
    try:
        fn()
    except Exception:  # noqa: BLE001 - one broken plot must not hide the others
        failures[name] = traceback.format_exc()
        print("[chain] %s FAILED:\n%s" % (name, failures[name]), flush=True)


def _dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def run_chain(input, outdir, topocluster_collection=None, particle_collection=None,
              association_collection=None, min_link_fraction=0.0, max_events=None,
              skip_energy=False, scale_dict=None, node_features=None, n_display=5,
              skip_display=False):
    """Run the whole chain; returns ``dict(metrics, energy_metrics, failures, outdir)``.

    ``scale_dict`` (the node scale dict of the training config) enables the transformed
    node-feature plots, ``node_features`` is the model's ``data.inputs.topocluster`` list
    (default: e eta cosphi sinphi r), ``n_display`` the number of event displays.
    Raises if the prediction file cannot be loaded; plot failures are collected.
    """
    print("[chain] loading %s" % input, flush=True)
    data = EvalData(input, max_events=max_events,
                    topocluster_collection=topocluster_collection,
                    particle_collection=particle_collection,
                    association_collection=association_collection,
                    min_link_fraction=min_link_fraction)
    print("[chain] %s" % data.summary(), flush=True)

    # the node multiplicity per shower is ~10x larger for RecHits than for
    # LayerClusters, so the binning has to follow the file
    nmax = max(data.p_pred_n_nodes.max(initial=1), data.p_n_true_nodes.max(initial=1))
    ev.set_node_edges(nmax)
    en.set_node_edges(nmax)

    os.makedirs(outdir, exist_ok=True)
    failures = {}

    node_features = list(node_features) if node_features else list(tf.DEFAULT_NODE_FEATURES)

    # ---- resolution / efficiency / diagnostics ----
    metrics = dict(dataset=data.summary(),
                   inputs=dict(node_features=node_features, scale_dict=scale_dict, n_display=n_display))
    sections = [
        ("inclusive resolutions", lambda: ev.plot_inclusive_resolutions(data, outdir, metrics)),
        ("resolutions by class", lambda: ev.plot_resolutions_by_class(data, outdir, metrics)),
        ("energy log-ratio resolution", lambda: ev.plot_energy_logratio(data, outdir, metrics)),
        ("resolution vs kinematics", lambda: ev.plot_resolution_vs_kinematics(data, outdir, metrics)),
        ("clustering efficiency", lambda: ev.plot_efficiency(data, outdir, metrics)),
        ("resolution vs pred_n_nodes", lambda: ev.plot_resolution_vs_n_nodes(data, outdir, metrics)),
        ("diagnostics", lambda: ev.plot_diagnostics(data, outdir, metrics)),
        # ---- cluster properties, node inputs, event level, displays ----
        ("cluster properties: distributions", lambda: cp.plot_distributions(data, outdir, metrics)),
        ("cluster properties: response", lambda: cp.plot_response(data, outdir, metrics)),
        ("cluster properties: response vs features", lambda: cp.plot_response_vs_features(data, outdir, metrics)),
        ("boundary position", lambda: bp.plot_boundary_position(data, outdir, metrics)),
        ("topocluster features: raw", lambda: tf.plot_raw(data, outdir, node_features, metrics)),
    ]
    if scale_dict is not None:
        sections.append(("topocluster features: transformed",
                         lambda: tf.plot_transformed(data, outdir, node_features, FeatureScalerNP(scale_dict), metrics)))
    else:
        print("[chain] no scale dict given: the transformed node features are not plotted", flush=True)
    sections.append(("event reconstruction", lambda: er.plot_event_reconstruction(data, outdir, metrics)))
    if skip_display or n_display <= 0:
        print("[chain] event displays skipped", flush=True)
    else:
        sections.append(("event display", lambda: ed.plot_event_displays(data, outdir, n_display)))
    for name, fn in sections:
        _run(name, fn, failures)
    _dump(metrics, os.path.join(outdir, "metrics.json"))
    _run("summary", lambda: ev.write_summary(metrics, os.path.join(outdir, "summary.txt")), failures)

    # ---- energy deep-dive ----
    energy_metrics = None
    if not skip_energy:
        edir = os.path.join(outdir, "energy")
        os.makedirs(edir, exist_ok=True)
        masks = en.get_masks(data)
        energy_metrics = dict(dataset=data.summary())
        if data.p_pred_e is None:
            print("[chain] no predicted energy in the file: skipping the energy investigation")
        else:
            for name, fn in [
                ("energy: 2D correlation", lambda: en.plot_2d(data, edir, masks)),
                ("energy: response profiles", lambda: en.plot_response(data, edir, masks, energy_metrics)),
                ("energy: log-log fit", lambda: en.loglog_fit(data, edir, masks, energy_metrics)),
                ("energy: log-ratio", lambda: en.log_ratio_distribution(data, edir, masks, energy_metrics)),
                ("energy: budget", lambda: en.energy_budget(data, edir, energy_metrics)),
                ("energy: mask closure", lambda: en.truth_sanity(data, edir, energy_metrics)),
                ("energy: vs pred_n_nodes", lambda: en.response_vs_n_nodes(data, edir, masks, energy_metrics)),
                ("energy: alternative targets", lambda: en.alternative_targets(data, edir, energy_metrics)),
            ]:
                _run(name, fn, failures)
        _dump(energy_metrics, os.path.join(edir, "energy_metrics.json"))
        _run("energy: summary", lambda: en.report(energy_metrics, os.path.join(edir, "energy_summary.txt")), failures)

    failures_path = os.path.join(outdir, "plot_chain_failures.txt")
    if failures:
        with open(failures_path, "w") as fh:
            for name, tb in failures.items():
                fh.write("### %s\n%s\n" % (name, tb))
    elif os.path.exists(failures_path):
        os.remove(failures_path)

    n_png = sum(len([f for f in files if f.endswith(".png")]) for _, _, files in os.walk(outdir))
    print("[chain] %d plots written under %s; %d section(s) failed%s"
          % (n_png, outdir, len(failures), (": " + ", ".join(failures)) if failures else ""), flush=True)
    return dict(metrics=metrics, energy_metrics=energy_metrics, failures=failures, outdir=outdir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="prediction parquet written by HGCPredictionWriter")
    ap.add_argument("--outdir", required=True, help="directory for the plots and metrics")
    ap.add_argument("--topocluster-collection", default=None,
                    help="node collection carrying <coll>_pred_match_idx (default: auto-detect)")
    ap.add_argument("--particle-collection", default=None,
                    help="particle collection carrying <coll>_pred_valid (default: auto-detect)")
    ap.add_argument("--association-collection", default=None,
                    help="TICL association table used as truth for the trackster pair "
                         "(default: the pair's default table)")
    ap.add_argument("--min-link-fraction", type=float, default=0.0,
                    help="drop association links below this energy share (trackster pair)")
    ap.add_argument("--max-events", type=int, default=None, help="use only the first N events")
    ap.add_argument("--skip-energy", action="store_true", help="skip the energy investigation")
    ap.add_argument("--scale-dict", default=None,
                    help="node scale dict of the training config (data.scale_dict_path): enables "
                         "topocluster_features/transform/")
    ap.add_argument("--node-features", nargs="+", default=None,
                    help="node input features of the model (data.inputs.topocluster); default: %s"
                         % " ".join(tf.DEFAULT_NODE_FEATURES))
    ap.add_argument("--n-display", type=int, default=5,
                    help="number of events drawn in event_display/ (0 disables)")
    ap.add_argument("--skip-display", action="store_true", help="skip the event displays")
    args = ap.parse_args(argv)

    try:
        result = run_chain(args.input, args.outdir,
                           topocluster_collection=args.topocluster_collection,
                           particle_collection=args.particle_collection,
                           association_collection=args.association_collection,
                           min_link_fraction=args.min_link_fraction,
                           max_events=args.max_events, skip_energy=args.skip_energy,
                           scale_dict=args.scale_dict, node_features=args.node_features,
                           n_display=args.n_display, skip_display=args.skip_display)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("[chain] could not load %s" % args.input)
        return 2
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
