if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, math, json
a1 = 2.338107410459767
cs = 2 * a1 ** 1.5 / 3 ** 1.5

def step_stats(edges, vals):
    edges = np.asarray(edges, float)
    vals = np.asarray(vals, float)
    mass = vals * np.diff(edges)
    Z = mass.sum()
    vals = vals / Z
    mass = mass / Z

    def pval(t):
        i = np.searchsorted(edges, t, side='right') - 1
        return np.where((i >= 0) & (i < len(vals)), vals[np.clip(i, 0, len(vals) - 1)], 0.0)
    pts = np.unique(np.concatenate([edges, edges / 2]))
    mid = 0.5 * (pts[:-1] + pts[1:])
    w = np.diff(pts)
    H2 = float(np.sum(w * (np.sqrt(pval(mid)) - np.sqrt(2 * pval(2 * mid))) ** 2))

    def F(t):
        return np.where(t > 0, (t * np.log(np.maximum(t, 1e-300)) - t) / math.log(2), 0.0)
    Es = float(np.sum(vals * (F(edges[1:]) - F(edges[:-1]))))
    S = float(-np.sum(mass[mass > 0] * np.log(vals[mass > 0])))
    return (H2, Es, S)
out = []
for theta in [100.0, 1000.0, 10000.0, 100000.0]:
    n = int(12 * theta ** (1 / 3) + 60)
    D = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    w, V = np.linalg.eigh(theta * D + np.diag(np.arange(n, dtype=float)))
    c = np.abs(V[:, 0])
    c /= np.linalg.norm(c)
    vals = c ** 2 / 2.0 ** np.arange(n)
    order = np.argsort(-vals)
    vals_s = vals[order]
    mult = 2.0 ** np.arange(n)[order]
    keep = vals_s > 1e-300
    vals_s = vals_s[keep]
    mult = mult[keep]
    edges = np.concatenate([[0], np.cumsum(mult)])
    H2, Es, S = step_stats(edges, vals_s)
    H = math.sqrt(H2)
    out.append({'theta': theta, 'levels': n, 'H2': H2, 'E_log2T': Es, 'bound': cs / H - 2 * math.log2(1 / H) - 3, 'S': S, 'ln2*E': math.log(2) * Es, 'E*H/c*': Es * H / cs})
    print(out[-1], flush=True)
if any(row['E_log2T']+1e-9<row['bound'] or row['S']+1e-9<row['ln2*E'] for row in out): raise RuntimeError('Airy diagnostic inequality failed')
json.dump(out, open('airy_check_results.json', 'w'), indent=1)
