import sys, glob, os, numpy as np, awkward as ak
base = sys.argv[1]
print(f"{'set':<13}{'events':>7} | {'SimTS/evt mean':>14}{'n=0':>6}{'n=1':>6}{'n=2':>6}{'n>=3':>6}{'max':>5} | {'FromCPs/evt':>12}{'max':>5} | {'E(2nd)/E(1st) med':>18}")
for part in ("photons", "pions", "K0L"):
    for E in (2, 3, 5, 10, 25, 50, 75, 100, 200, 300):
        fs = sorted(glob.glob(os.path.join(base, part, f"{part}_E{E}_batch*.parquet")))
        t = ak.concatenate([ak.from_parquet(f, columns=['ticlSimTracksters.ticlSimTracksters_raw_energy', 'ticlSimTrackstersFromCPs.ticlSimTrackstersFromCPs_raw_energy']) for f in fs])
        e = t['ticlSimTracksters']['ticlSimTracksters_raw_energy']; n = ak.to_numpy(ak.num(e))
        ncp = ak.to_numpy(ak.num(t['ticlSimTrackstersFromCPs']['ticlSimTrackstersFromCPs_raw_energy']))
        multi = e[n >= 2]; ratio = ak.to_numpy(ak.sort(multi, ascending=False)[:, 1] / ak.sort(multi, ascending=False)[:, 0]) if len(multi) else np.array([])
        print(f"{part+'_E'+str(E):<13}{len(n):>7} | {n.mean():>14.3f}{(n==0).sum():>6}{(n==1).sum():>6}{(n==2).sum():>6}{(n>=3).sum():>6}{n.max():>5} | {ncp.mean():>12.3f}{ncp.max():>5} | {np.median(ratio) if len(ratio) else float('nan'):>18.3f}")
