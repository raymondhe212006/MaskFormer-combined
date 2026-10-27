#!/usr/bin/env python
"""Event-level reconstruction: the sum of all predicted clusters of an event.

A photon converting in front of HGCAL is reconstructed as two (e+, e-) clusters, and what
matters for such an object is their sum, not the single clusters.  Per event this module sums
the predicted energies of all matched queries and takes the energy-weighted average of their
eta, cos(phi), sin(phi) and r, and compares them with the same quantities built from the true
particles.  Writes, under ``<outdir>/event_reconstruction/``:

  total_energy_true_vs_pred.png                 per-event sum E: true vs predicted
  total_energy_response.png                     sum E_pred / sum E_true
  total_energy_response_vs_true_energy.png      the same, in bins of sum E_true
  total_energy_pred_vs_true_2d.png
  weighted_<q>_true_vs_pred.png                 energy-weighted <q>, q = eta, cosphi, sinphi, r
                                                (+ boundary_x, boundary_y when regressed)
  weighted_<q>_residual.png                     <q>_pred - <q>_true
  weighted_<q>_residual_vs_true_energy.png
  n_fake_particles.png                          (when the branch exists, see below)

Two selections, as everywhere: "all matched queries" and "pred_valid only" (the true
reference is always the full set of true particles, so lost particles show as a deficit).

Limitation: the prediction file holds only the queries matched to a true particle.  Valid
but unmatched queries ("fakes") are counted in the top-level ``n_fake_particles`` branch but
their kinematics are not stored, so they cannot enter the sums; their count is plotted.
``energy/event_energy_budget`` already shows the per-event energy sum; this module adds the
position features and the response profiles.

Standalone:  python event_reconstruction.py --input <pred.parquet> --outdir <dir>
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import (  # noqa: E402
    CMS_blue, CMS_gray, CMS_red, EvalData, profile, read_schema, robust_sigma, savefig,
    setup_style, stat_label,
)

plt = setup_style()

#: energy-weighted quantities: key, label, unit, truth / prediction attributes, residual range
QUANTITIES = [
    dict(key="eta", label=r"$\eta$", unit="", truth="p_eta", pred="p_pred_eta", rng=(-0.3, 0.3)),
    dict(key="cosphi", label=r"$\cos\phi$", unit="", truth="p_cosphi", pred="p_pred_cosphi", rng=(-0.5, 0.5)),
    dict(key="sinphi", label=r"$\sin\phi$", unit="", truth="p_sinphi", pred="p_pred_sinphi", rng=(-0.5, 0.5)),
    dict(key="r", label=r"$r$", unit=" cm", truth="p_r", pred="p_pred_r", rng=(-40.0, 40.0)),
    # entry point into HGCAL (boundary targets), when regressed and when the file has the truth
    dict(key="boundary_x", label=r"$x^{\mathrm{b}}$", unit=" cm", truth="p_boundary_x", pred="p_pred_boundary_x", rng=(-40.0, 40.0)),
    dict(key="boundary_y", label=r"$y^{\mathrm{b}}$", unit=" cm", truth="p_boundary_y", pred="p_pred_boundary_y", rng=(-40.0, 40.0)),
]
#: per-event energy sums reach a few TeV on the tau samples
EVENT_E_EDGES = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0])


def masks(data):
    return [("all", "all matched queries", np.ones(len(data.p_e), bool), CMS_blue),
            ("valid", "pred_valid only", data.p_valid, CMS_red)]


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(median=None, sigma68=None, mean=None, n=0)
    return dict(median=float(np.median(x)), sigma68=float(robust_sigma(x)),
                mean=float(x.mean()), n=int(len(x)))


def _weighted_mean(ev, x, w, nev):
    good = np.isfinite(x) & (w > 0)
    num = np.bincount(ev[good], weights=(x * w)[good], minlength=nev)
    den = np.bincount(ev[good], weights=w[good], minlength=nev)
    return np.divide(num, den, out=np.full(nev, np.nan), where=den > 0)


def event_sums(data, mask):
    """Per event: ``dict(true=..., pred=...)`` with ``e`` (energy sum), ``n`` (number of
    particles / queries) and the energy-weighted means of the QUANTITIES (NaN when the
    event has no energy under the selection)."""
    nev = data.n_events
    ev = data.p_event
    w_t = np.where(np.isfinite(data.p_e) & (data.p_e > 0), data.p_e, 0.0)
    true = dict(e=np.bincount(ev, weights=w_t, minlength=nev), n=np.bincount(ev, minlength=nev))
    pe = data.p_pred_e
    # the log1p inverse can give tiny negative energies: no weight
    w_p = np.where(mask & np.isfinite(pe) & (pe > 0), pe, 0.0)
    pred = dict(e=np.bincount(ev, weights=w_p, minlength=nev), n=np.bincount(ev[mask], minlength=nev))
    for q in QUANTITIES:
        t = getattr(data, q["truth"], None)
        if t is None:
            continue
        true[q["key"]] = _weighted_mean(ev, t, w_t, nev)
        p = getattr(data, q["pred"], None)
        if p is not None:
            pred[q["key"]] = _weighted_mean(ev, p, w_p, nev)
    return dict(true=true, pred=pred)


def read_n_fakes(data):
    """The per-event number of valid but unmatched queries, or ``None`` if not stored."""
    if "n_fake_particles" not in read_schema(data.path):
        return None
    arr = ak.from_parquet(data.path, columns=["n_fake_particles"])[:data.n_events]
    return np.asarray(ak.to_numpy(arr["n_fake_particles"])).astype(np.int64)


def _log_bins(values, n=60):
    v = np.concatenate([np.asarray(x, float) for x in values])
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return np.logspace(-1, 3, n + 1)
    lo, hi = np.percentile(v, [0.1, 99.9])
    lo = max(lo, 1e-2)
    hi = max(hi * 1.2, lo * 10)
    return np.logspace(np.log10(lo), np.log10(hi), n + 1)


def _lin_bins(values, n=60):
    v = np.concatenate([np.asarray(x, float) for x in values])
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.linspace(0, 1, n + 1)
    lo, hi = np.percentile(v, [0.1, 99.9])
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return np.linspace(lo - pad, hi + pad, n + 1)


def _energy_axis(data):
    return r"$\sum E_{\mathrm{true}}$ per event (%s %s) [GeV]" % (data.particle_label, data.energy_label)


# --------------------------------------------------------------------------- #

def plot_event_reconstruction(data, outdir, metrics):
    sub = os.path.join(outdir, "event_reconstruction")
    if data.p_pred_e is None:
        print("[event] no predicted energy in the file: event reconstruction skipped")
        return
    m = metrics.setdefault("event_reconstruction", {})
    sel = masks(data)
    sums = {tag: event_sums(data, mask) for tag, _, mask, _ in sel}
    true = sums["all"]["true"]
    nev = data.n_events
    has_true = true["n"] > 0
    m["n_events"] = int(nev)
    m["n_events_no_true_particle"] = int((~has_true).sum())
    m["n_events_no_pred"] = {tag: int((has_true & (sums[tag]["pred"]["e"] <= 0)).sum()) for tag, *_ in sel}

    # ---- total energy ----
    bins = _log_bins([true["e"]] + [sums[tag]["pred"]["e"] for tag, *_ in sel])
    fig, ax = plt.subplots()
    v = true["e"][has_true]
    ax.hist(v, bins=bins, histtype="step", lw=2, color=CMS_gray,
            label=r"true: $\sum E$ of all %ss" % data.particle_label + "\n" + stat_label(v, " GeV"))
    for tag, name, _, color in sel:
        v = sums[tag]["pred"]["e"]
        v = v[has_true & (v > 0)]
        ax.hist(v, bins=bins, histtype="step", lw=2, color=color,
                label=r"predicted: $\sum E$, %s" % name + "\n" + stat_label(v, " GeV"))
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sum E$ per event [GeV]")
    ax.set_ylabel("events / bin")
    ax.set_title("Event energy: sum of all clusters", fontsize=20)
    ax.legend(fontsize=11)
    savefig(fig, sub, "total_energy_true_vs_pred")

    ratio = {}
    fig, ax = plt.subplots()
    rbins = np.linspace(0.0, 3.0, 91)
    for tag, name, _, color in sel:
        ep = sums[tag]["pred"]["e"]
        good = has_true & (true["e"] > 0) & (ep > 0)
        ratio[tag] = np.where(good, ep / np.where(true["e"] > 0, true["e"], 1.0), np.nan)
        v = ratio[tag][good]
        ax.hist(v, bins=rbins, histtype="step", lw=2, color=color,
                label="%s\n%s" % (name, stat_label(v)))
        m.setdefault("total_energy_response", {})[tag] = dict(
            _stats(v), n_events_no_pred=m["n_events_no_pred"][tag],
            overflow_frac=float(np.mean(v > rbins[-1])) if len(v) else 0.0)
    ax.axvline(1.0, color="k", ls=":", lw=1.2)
    ax.set_xlabel(r"$\sum E_{\mathrm{pred}} \, / \, \sum E_{\mathrm{true}}$ per event")
    ax.set_ylabel("events / bin")
    ax.set_title("Event energy response", fontsize=20)
    ax.legend(fontsize=12)
    savefig(fig, sub, "total_energy_response")

    fig, ax = plt.subplots()
    for tag, name, _, color in sel:
        ctr, med, sig, n = profile(true["e"], ratio[tag], EVENT_E_EDGES)
        ok = np.isfinite(med)
        ax.errorbar(ctr[ok], med[ok], yerr=np.where(np.isfinite(sig[ok]), sig[ok], 0.0),
                    fmt="o-", color=color, lw=2, ms=7, capsize=3, label=name)
        if tag == "all":
            m["total_energy_response_vs_true_energy"] = dict(
                centre=ctr.tolist(), median=np.where(ok, med, None).tolist(),
                sigma68=np.where(np.isfinite(sig), sig, None).tolist(), n=n.tolist())
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(_energy_axis(data))
    ax.set_ylabel(r"$\sum E_{\mathrm{pred}} / \sum E_{\mathrm{true}}$  (median $\pm\,\sigma_{68}$)")
    ax.set_title("Event energy response vs true event energy", fontsize=20)
    ax.legend(fontsize=13)
    savefig(fig, sub, "total_energy_response_vs_true_energy")

    fig, ax = plt.subplots()
    ep = sums["all"]["pred"]["e"]
    good = has_true & (true["e"] > 0) & (ep > 0)
    lb = _log_bins([true["e"][good], ep[good]], n=50)
    ax.hist2d(true["e"][good], ep[good], bins=[lb, lb], cmap="viridis", cmin=1)
    ctr, med, sig, n = profile(true["e"], np.where(good, ep, np.nan), lb)
    ok = np.isfinite(med)
    ax.plot(ctr[ok], med[ok], "o-", color=CMS_red, lw=2, ms=5, label="median")
    ax.plot(lb, lb, "k:", lw=1.2, label="pred = true")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(_energy_axis(data))
    ax.set_ylabel(r"$\sum E_{\mathrm{pred}}$ per event, all matched queries [GeV]")
    ax.set_title("Event energy: predicted vs true", fontsize=20)
    ax.legend(fontsize=13, loc="upper left")
    savefig(fig, sub, "total_energy_pred_vs_true_2d")

    # ---- energy-weighted positions ----
    for q in QUANTITIES:
        key = q["key"]
        if key not in sums["all"]["pred"]:
            continue
        t = true[key]
        bins = _lin_bins([t] + [sums[tag]["pred"][key] for tag, *_ in sel])
        fig, ax = plt.subplots()
        v = t[np.isfinite(t)]
        ax.hist(v, bins=bins, histtype="step", lw=2, color=CMS_gray,
                label="true (E-weighted over all %ss)\n%s" % (data.particle_label, stat_label(v, q["unit"])))
        for tag, name, _, color in sel:
            v = sums[tag]["pred"][key]
            v = v[np.isfinite(v)]
            ax.hist(v, bins=bins, histtype="step", lw=2, color=color,
                    label="predicted (E-weighted), %s\n%s" % (name, stat_label(v, q["unit"])))
        ax.set_xlabel(r"energy-weighted %s per event%s" % (q["label"], (" [%s]" % q["unit"].strip()) if q["unit"].strip() else ""))
        ax.set_ylabel("events / bin")
        ax.set_title("Event %s: energy-weighted average" % q["label"], fontsize=20)
        ax.legend(fontsize=11)
        savefig(fig, sub, "weighted_%s_true_vs_pred" % key)

        res = {}
        fig, ax = plt.subplots()
        rb = np.linspace(q["rng"][0], q["rng"][1], 101)
        for tag, name, _, color in sel:
            res[tag] = sums[tag]["pred"][key] - t
            v = res[tag][np.isfinite(res[tag])]
            ax.hist(v, bins=rb, histtype="step", lw=2, color=color,
                    label="%s\n%s" % (name, stat_label(v, q["unit"])))
            m.setdefault("weighted_residual", {}).setdefault(key, {})[tag] = dict(
                _stats(v), overflow_frac=float(np.mean(np.abs(v) > q["rng"][1])) if len(v) else 0.0)
        ax.axvline(0.0, color="k", ls=":", lw=1.2)
        sym = q["label"].strip("$")
        ax.set_xlabel(r"$\langle %s \rangle_{E,\mathrm{pred}} - \langle %s \rangle_{E,\mathrm{true}}$%s"
                      % (sym, sym, (" [%s]" % q["unit"].strip()) if q["unit"].strip() else ""))
        ax.set_ylabel("events / bin")
        ax.set_title("Event %s residual" % q["label"], fontsize=20)
        ax.legend(fontsize=12)
        savefig(fig, sub, "weighted_%s_residual" % key)

        fig, ax = plt.subplots()
        for tag, name, _, color in sel:
            ctr, med, sig, n = profile(true["e"], res[tag], EVENT_E_EDGES)
            ok = np.isfinite(med)
            ax.errorbar(ctr[ok], med[ok], yerr=np.where(np.isfinite(sig[ok]), sig[ok], 0.0),
                        fmt="o-", color=color, lw=2, ms=7, capsize=3, label=name)
        ax.axhline(0.0, color="k", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(_energy_axis(data))
        ax.set_ylabel(r"$\Delta\langle %s \rangle_E$  (median $\pm\,\sigma_{68}$)%s"
                      % (sym, (" [%s]" % q["unit"].strip()) if q["unit"].strip() else ""))
        ax.set_title("Event %s residual vs true event energy" % q["label"], fontsize=20)
        ax.legend(fontsize=13)
        savefig(fig, sub, "weighted_%s_residual_vs_true_energy" % key)

    # ---- fakes: the part of the event the sums cannot see ----
    n_fake = read_n_fakes(data)
    if n_fake is not None:
        fig, ax = plt.subplots()
        hi = int(n_fake.max()) if len(n_fake) else 0
        ax.hist(n_fake, bins=np.arange(-0.5, hi + 1.5), histtype="step", lw=2, color=CMS_blue,
                label="valid but unmatched queries\nmean %.3f, events with >= 1: %.3f"
                      % (n_fake.mean(), np.mean(n_fake > 0)))
        ax.set_xlabel("fake queries per event (not in the sums: kinematics not stored)")
        ax.set_ylabel("events / bin")
        ax.set_title("Fake queries per event", fontsize=20)
        ax.legend(fontsize=12)
        savefig(fig, sub, "n_fake_particles")
        m["n_fake_particles"] = dict(mean=float(n_fake.mean()), max=int(hi),
                                     frac_events_with_fakes=float(np.mean(n_fake > 0)))


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
    plot_event_reconstruction(data, args.outdir, metrics)
    with open(os.path.join(args.outdir, "event_reconstruction", "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[event] done -> %s" % os.path.join(args.outdir, "event_reconstruction"))


if __name__ == "__main__":
    main()
