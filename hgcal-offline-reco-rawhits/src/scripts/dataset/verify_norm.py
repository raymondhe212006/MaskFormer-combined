#!/usr/bin/env python3
"""End-to-end check of the input/target normalization through HGCDataset.

For one training config, builds the dataset on a parquet file exactly as training would,
iterates events, and reports the scaled range of every node input feature and every
regression target. It then applies pass/fail checks for the specific failure modes fixed
in docs/NORMALIZATION_FIX.md:

  1. the node energy feature has real dynamic range (it used to collapse to a constant
     because the fn=sqrt stats were measured on the raw variable);
  2. the node `r` input is scaled (it used to enter raw, ~320-570 cm);
  3. the regression targets are scaled with the PARTICLE dict, not the node dict (same
     key names, spreads that differ by 15-340x);
  4. the config hands the regression task the particle dict (`scale_dict_path:
     *particle_scale_dict_path`), so targets, proxies and predictions share one space;
  5. everything is finite.

Usage:
  python src/scripts/dataset/verify_norm.py --config configs/base_layercluster_simcluster.yaml \
      --parquet /path/to/file.parquet --num-events 200

Exit code 0 when every check passes, 1 otherwise. Only the `data:` block of the config is
used, so the training file paths in it do not need to exist.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from dataset.dataset_maskformers import HGCDataset  # noqa: E402


def fmt(t: torch.Tensor) -> str:
    t = t.float()
    return f"min={t.min():8.3f} max={t.max():8.3f} mean={t.mean():8.3f} std={t.std():7.3f} finite={bool(torch.isfinite(t).all())}"


def regression_task_scale_dicts(model_cfg) -> list[str]:
    """scale_dict_path of every task that also declares `fields` (the regression tasks)."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "scale_dict_path" in o and "fields" in o:
                found.append(o["scale_dict_path"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(model_cfg)
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Training config (its data: block selects collections, inputs, targets and scale dicts)")
    p.add_argument("--parquet", required=True, help="Parquet file to read events from")
    p.add_argument("--num-events", type=int, default=100)
    p.add_argument("--quiet", action="store_true", help="Suppress the dataset's own loading output")
    args = p.parse_args(argv)

    cfg_all = yaml.safe_load(open(args.config))
    cfg = cfg_all["data"]
    node_dict = REPO_ROOT / cfg["scale_dict_path"]
    particle_dict = REPO_ROOT / cfg["particle_scale_dict_path"]
    print(f"config      : {args.config}")
    print(f"collections : {cfg['topocluster_collection']} -> {cfg['particle_collection']}")
    print(f"node dict   : {node_dict}")
    print(f"particle dict: {particle_dict}")

    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    print("\n--- config consistency ---")
    check("node and particle dicts are different files", node_dict.resolve() != particle_dict.resolve())
    check("both dict files exist", node_dict.is_file() and particle_dict.is_file(), f"{node_dict.name}: {node_dict.is_file()}, {particle_dict.name}: {particle_dict.is_file()}")
    task_dicts = regression_task_scale_dicts(cfg_all.get("model", {}))
    check(
        "regression task is given the PARTICLE dict",
        bool(task_dicts) and all(d == cfg["particle_scale_dict_path"] for d in task_dicts),
        f"task scale_dict_path={task_dicts}",
    )
    if not (node_dict.is_file() and particle_dict.is_file()):
        print("\nCannot build the dataset without both dicts; see docs/NORMALIZATION_FIX.md for how to produce them.")
        return 1
    node_e = yaml.safe_load(open(node_dict))["e"]
    part_e = yaml.safe_load(open(particle_dict))["e"]
    check("node/particle `e` entries were fitted separately", node_e != part_e, f"node e={node_e}  particle e={part_e}")

    print("\n--- dataset ---")
    sink = io.StringIO() if args.quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        ds = HGCDataset(
            filepath=args.parquet,
            inputs=cfg["inputs"],
            targets=cfg["targets"],
            scale_dict_path=str(node_dict),
            particle_scale_dict_path=str(particle_dict),
            topocluster_collection=cfg["topocluster_collection"],
            particle_collection=cfg["particle_collection"],
            num_events=args.num_events,
            num_objects=cfg["num_objects"],
            max_nodes=cfg.get("max_nodes", 50000),
            incidence_cutval=float(cfg["incidence_cutval"]),
            normalize_incidence=cfg.get("normalize_incidence", True),
            association_collection=cfg.get("association_collection"),
            min_link_fraction=float(cfg.get("min_link_fraction", 0.0)),
        )
    fields = cfg["inputs"]["topocluster"]
    targets = cfg["targets"]["particle"]
    node_cols = {f: [] for f in fields}
    part_cols = {f: [] for f in targets}
    for i in range(len(ds)):
        inp, lab = ds[i]
        valid = inp["node_valid"]
        for j, f in enumerate(fields):
            node_cols[f].append(inp["node_features"][valid, j])
        pvalid = lab["particle_valid"]
        for f in targets:
            part_cols[f].append(lab[f"particle_{f}"][pvalid])
    node_cols = {f: torch.cat(v) for f, v in node_cols.items()}
    part_cols = {f: torch.cat(v) for f, v in part_cols.items()}
    n_nodes = len(node_cols[fields[0]])
    n_particles = len(part_cols[targets[0]])
    print(f"events={len(ds)}  valid nodes={n_nodes:,}  valid particles={n_particles:,}")

    print("\n--- node input features (scaled, valid nodes) ---")
    for f, t in node_cols.items():
        print(f"  node {f:7s} {fmt(t)}")
    print("--- regression targets (scaled, valid particles) ---")
    for f, t in part_cols.items():
        print(f"  part {f:7s} {fmt(t)}")

    print("\n--- checks ---")
    check("all node features finite", all(bool(torch.isfinite(t).all()) for t in node_cols.values()))
    check("all targets finite", all(bool(torch.isfinite(t).all()) for t in part_cols.values()))
    if "e" in node_cols:
        s = node_cols["e"].std().item()
        check("node e has dynamic range (std > 0.1)", s > 0.1, f"std={s:.3f}; the collapsed feature had std 0.001-0.02")
    if "r" in node_cols:
        m = node_cols["r"].abs().max().item()
        check("node r is scaled (|r| <= 1.5)", m <= 1.5, f"|r|max={m:.3f}; unscaled R is 320-570 cm")
    if "e" in part_cols:
        t = part_cols["e"]
        # particle e is type std on log1p(e): unit-ish spread, not squeezed into a corner
        check("particle e has unit-scale spread (0.3 < std < 3)", 0.3 < t.std().item() < 3, f"std={t.std().item():.3f}, range [{t.min():.2f}, {t.max():.2f}]")
    if "r" in part_cols:
        t = part_cols["r"]
        check("particle r bulk inside [-1.5, 1.5] (q1/q99)", bool(t.quantile(0.01) > -1.5 and t.quantile(0.99) < 1.5), f"q1={t.quantile(0.01):.3f} q99={t.quantile(0.99):.3f}")
    if "eta" in part_cols:
        t = part_cols["eta"]
        check("particle eta bulk inside [-1.5, 1.5] (q1/q99)", bool(t.quantile(0.01) > -1.5 and t.quantile(0.99) < 1.5), f"q1={t.quantile(0.01):.3f} q99={t.quantile(0.99):.3f}")
    # entrance-position targets (boundary_x, boundary_y, ...): min_max_sym entries of the
    # particle dict, so the bulk must land inside [-1.5, 1.5] like eta and r
    for f in [f for f in part_cols if f.startswith("boundary_") and not f.endswith(("sinphi", "cosphi"))]:
        t = part_cols[f]
        check(f"particle {f} bulk inside [-1.5, 1.5] (q1/q99)", bool(t.quantile(0.01) > -1.5 and t.quantile(0.99) < 1.5), f"q1={t.quantile(0.01):.3f} q99={t.quantile(0.99):.3f}")
    for f in [f for f in part_cols if f.endswith(("sinphi", "cosphi"))]:
        t = part_cols[f]
        check(f"particle {f} inside [-1, 1]", bool(t.abs().max() <= 1.0 + 1e-6), f"|max|={t.abs().max():.3f}")
    for f in [f for f in targets if f.startswith("boundary_")]:
        check(f"'{f}' has its own entry in the particle dict", f in yaml.safe_load(open(particle_dict)), "missing entries fall back to the identity transform")
    check("dataset exposes particle_scaler (used by the prediction writer)", hasattr(ds, "particle_scaler"))

    print(f"\nALL CHECKS PASSED: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
