"""Why do low-energy events have no CLUE3D trackster? Per set: events with/without a CLUE3D trackster,
what the layer clusters / rechits / other trackster collections look like in each class, and the smallest
CLUE3D trackster that survives (raw energy, number of layer clusters). Usage: python x.py <parquet dir> <set> ..."""
import sys, glob, os
import numpy as np, awkward as ak
base = sys.argv[1]
for s in sys.argv[2:]:
    part, E = s.rsplit('_E', 1)
    files = sorted(glob.glob(os.path.join(base, part, f'{part}_E{E}_batch*.parquet')))
    t = ak.concatenate([ak.from_parquet(f, columns=['ticlTrackstersCLUE3DHigh.*', 'ticlTracksterLinks.*', 'ticlTrackstersRecovery.*',
                                                    'LayerCluster.*', 'RecHitHGC.RecHitHGC_energy', 'ticlSimTracksters.*']) for f in files])
    ts = t['ticlTrackstersCLUE3DHigh']; lk = t['ticlTracksterLinks']; rc = t['ticlTrackstersRecovery']; lc = t['LayerCluster']; st = t['ticlSimTracksters']
    nts = ak.num(ts['ticlTrackstersCLUE3DHigh_raw_energy']); nlk = ak.num(lk['ticlTracksterLinks_raw_energy']); nrc = ak.num(rc['ticlTrackstersRecovery_raw_energy'])
    nlc = ak.num(lc['LayerCluster_energy']); elc = ak.sum(lc['LayerCluster_energy'], axis=1); erh = ak.sum(t['RecHitHGC']['RecHitHGC_energy'], axis=1)
    nlc3 = ak.sum(lc['LayerCluster_nHits'] >= 3, axis=1)
    est = ak.firsts(st['ticlSimTracksters_raw_energy']); esr = ak.firsts(st['ticlSimTracksters_regressed_energy'])
    has = nts > 0; no = ~has
    def med(x): x = ak.to_numpy(x); return np.nanmedian(x) if len(x) else float('nan')
    print(f"\n=== {s}: {len(nts)} events, {int(ak.sum(has))} with CLUE3D trackster, {int(ak.sum(no))} without")
    print(f"{'':>28}{'with TS':>10}{'no TS':>10}")
    for name, arr in (("median n LayerClusters", nlc), ("median n LC (nHits>=3)", nlc3), ("median sum LC energy [GeV]", elc),
                      ("median sum RecHit energy", erh), ("median SimTS raw_energy", est), ("median SimTS regressed_e", esr),
                      ("median n Links tracksters", nlk), ("median n Recovery tracksters", nrc)):
        print(f"{name:>28}{med(arr[has]):>10.2f}{med(arr[no]):>10.2f}")
    print(f"{'events with 0 LayerClusters':>28}{int(ak.sum((nlc == 0) & has)):>10}{int(ak.sum((nlc == 0) & no)):>10}")
    print(f"{'events with Links TS':>28}{int(ak.sum((nlk > 0) & has)):>10}{int(ak.sum((nlk > 0) & no)):>10}")
    print(f"{'events with Recovery TS':>28}{int(ak.sum((nrc > 0) & has)):>10}{int(ak.sum((nrc > 0) & no)):>10}")
    e = ak.to_numpy(ak.flatten(ts['ticlTrackstersCLUE3DHigh_raw_energy'])); nv = ak.to_numpy(ak.flatten(ts['ticlTrackstersCLUE3DHigh_nticlTrackstersCLUE3DHighvertices']))
    if len(e):
        print(f"  CLUE3D tracksters: min raw_energy {e.min():.3f} GeV, 1% quantile {np.quantile(e, 0.01):.3f}; "
              f"min n vertices (LCs) {nv.min()}, counts of nvertices<=5: {dict(zip(*np.unique(nv[nv <= 5], return_counts=True)))}")
    # among no-TS events: distribution of max LC count on a single... layer not available; show nLC quantiles
    q = ak.to_numpy(nlc[no]); print(f"  no-TS events: n LayerClusters quantiles 10/50/90% = {np.quantile(q, [0.1, 0.5, 0.9]) if len(q) else '-'}")
