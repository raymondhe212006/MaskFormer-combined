"""Merged results table over the <particle>_E<E>/ sets of an eval directory (default: cwd).
Usage: python eval_table.py [eval_dir]   (run eval_gaussfit.py first for the Gaussian / sumTS columns)"""
import json, glob, os, re, sys
if len(sys.argv) > 1: os.chdir(sys.argv[1])
def key(d):
    p, e = d.rsplit('_E', 1); return ({'photons':0,'pions':1,'K0L':2}.get(p, 9), int(e))
rows = []
gf = json.load(open('gaussfit_summary.json')) if os.path.isfile('gaussfit_summary.json') else {}
for d in sorted([d for d in glob.glob('*_E*') if re.fullmatch(r'(photons|pions|K0L)_E\d+', d)], key=key):
    m = os.path.join(d, 'plots', 'metrics.json'); l = glob.glob(os.path.join(d, '*_losses.json'))
    if not os.path.isfile(m) or not l: continue
    M = json.load(open(m)); L = json.load(open(l[0]))['final']
    loss = sum(v for t in L.values() for v in t.values())
    ri = M['resolution_inclusive']; ds = M['dataset']; cl = M['classification']['pred_class_counts']
    rows.append((d, ds['n_events'], ds['n_nodes'], ds['valid_fraction'], cl.get('electromagnetic',0), cl.get('hadronic',0),
                 ri['e_all']['median'], ri['e_all']['sigma68'], 100*ri['e_all']['sigma68'], 100*gf[d]['sigma_over_Enom'] if d in gf else float('nan'), 100*gf[d]['masksum']['sigma68_ratio'] if d in gf and 'masksum' in gf[d] else float('nan'), 100*gf[d]['masksum']['sigma_over_Enom'] if d in gf and 'masksum' in gf[d] else float('nan'), ri['eta_all']['sigma68'], ri['r_all']['sigma68'], loss))
print(f"{'set':<13}{'events':>7}{'nodes':>7}{'valid':>7}{'pEM':>5}{'pHad':>5}{'E med':>9}{'E s68':>8}{'E res%':>8}{'gaus%':>7}{'ts68%':>7}{'tsG%':>6}{'eta s68':>9}{'r s68':>8}{'loss':>7}")
for r in rows:
    print(f"{r[0]:<13}{r[1]:>7}{r[2]:>7}{r[3]:>7.3f}{r[4]:>5}{r[5]:>5}{r[6]:>+9.4f}{r[7]:>8.4f}{r[8]:>8.2f}{r[9]:>7.2f}{r[10]:>7.2f}{r[11]:>6.2f}{r[12]:>9.4f}{r[13]:>8.3f}{r[14]:>7.2f}")
