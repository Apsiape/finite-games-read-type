if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import math, json, itertools, hashlib, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import airy, ai_zeros
from scipy.integrate import quad
OUT = Path(__file__).resolve().parent
START = time.perf_counter()
R = {}
a = math.log(2)
I = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
Y = np.array([[0.0, -1j], [1j, 0.0]])
square = [[np.kron(X, I), np.kron(I, X), np.kron(X, X)], [np.kron(I, Z), np.kron(Z, I), np.kron(Z, Z)], [np.kron(X, Z), np.kron(Z, X), np.kron(Y, Y).real]]
triples = list(itertools.product((1, -1), repeat=3))
rows = [s for s in triples if math.prod(s) == 1]
cols = [[s for s in triples if math.prod(s) == v] for v in (1, 1, -1)]

def joint(ops, s):
    p = np.eye(4)
    for o, v in zip(ops, s):
        p = p @ (np.eye(4) + v * o) / 2
    return p

def lift(p=None, f=0):
    z = np.zeros((5, 5))
    z[0, 0] = f
    if p is not None:
        z[1:, 1:] = p
    return z
A = {}
B = {}
for i in range(3):
    A['R', i] = {s: lift(joint(square[i], s)) for s in rows}
    A['R', i]['f'] = lift(f=1)
    B['C', i] = {s: lift(joint([square[k][i] for k in range(3)], s)) for s in cols[i]}
    B['C', i]['f'] = lift(f=1)
special = [(1, 1, 1), (1, -1, -1)]
cv = [np.array([1, 0, 1, 0]) / math.sqrt(2), np.array([0, 1, 0, 1]) / math.sqrt(2)]
rv = [np.array([1, 1, 1, 1]) / 2, np.array([1, -1, 1, -1]) / 2]
for k in range(2):
    A['T', k] = {s: lift(joint([square[i][0] for i in range(3)], s)) for s in cols[0] if s != special[k]}
    B['U', k] = {s: lift(joint(square[0], s)) for s in rows if s != special[k]}
    for v in (1, -1):
        u = np.r_[1, v * cv[k]] / math.sqrt(2)
        A['T', k][v] = np.outer(u, u)
        u = np.r_[1, v * rv[k]] / math.sqrt(2)
        B['U', k][v] = np.outer(u, u)

def accepts(q, s, r, t):
    x, i = q
    y, j = r
    if x == 'R' and y == 'C':
        return s == t if s == 'f' or t == 'f' else s[j] == t[i]
    if x == 'T' and y == 'C' and (j == 0):
        return t == 'f' or t == special[i] if isinstance(s, int) else s == t
    if x == 'R' and y == 'U' and (i == 0):
        return s == 'f' or s == special[j] if isinstance(t, int) else s == t
    if x == 'T' and y == 'U' and isinstance(s, int) and isinstance(t, int):
        return s * t == (-1) ** (i * j)
    return True

def block_loss(aa, bb):
    d = np.array([aa, bb, bb, bb, bb])
    outer = np.outer(d, d)
    return sum((float(np.sum(outer * p * q)) for x, ps in A.items() for y, qs in B.items() for s, p in ps.items() for t, q in qs.items() if not accepts(x, s, y, t)))
poly = [block_loss(1, 0), block_loss(0, 1), block_loss(1, 1) - block_loss(1, 0) - block_loss(0, 1)]
R['block_polynomial_aa2_bb2_aabb'] = poly
assert np.allclose(poly, [2, 1, -2 * math.sqrt(2)], atol=2e-14)
R['pvm_completeness_error'] = max((float(np.max(abs(sum(p.values()) - np.eye(5)))) for d in (A, B) for p in d.values()))

def full_loss(c):
    c = np.array(c, float)
    c /= np.linalg.norm(c)
    L = len(c) - 1
    words = [(l, b) for l in range(L + 1) for b in range(2 ** l)]
    ix = {(atom, l, b): k for k, (atom, (l, b)) in enumerate(itertools.product(range(9), words))}
    n = len(ix)
    d = np.zeros(n)
    for (atom, l, b), k in ix.items():
        d[k] = c[l] / math.sqrt(9 * 2 ** l)
    total = 0.0
    link = 0.0
    max_completeness = 0.0
    flagA = {('R', i): 'f' for i in range(3)} | {('T', i): 1 for i in range(2)}
    flagB = {('C', i): 'f' for i in range(3)} | {('U', i): 1 for i in range(2)}
    assert sum((not accepts(q, s, r, t) for q, s in flagA.items() for r, t in flagB.items())) == 1
    completions = []
    for k, s0 in enumerate(rows):
        da = {('R', i): s0 for i in range(3)}
        db = {('C', j): (s0[j], s0[j], s0[j] if j < 2 else -s0[j]) for j in range(3)}
        for t in range(2):
            col = db['C', 0]
            row = da['R', 0]
            da['T', t] = 1 if col == special[t] else col
            db['U', t] = 1 if row == special[t] else row
        losses = sum((not accepts(q, s, r, t) for q, s in da.items() for r, t in db.items()))
        if losses != 1:
            found = None
            for sa in itertools.product(rows, repeat=3):
                if sa[1] != s0:
                    continue
                for tb in itertools.product(*cols):
                    if sum((sa[i][j] != tb[j][i] for i in range(3) for j in range(3))) != 1:
                        continue
                    da0 = {('R', i): sa[i] for i in range(3)}
                    db0 = {('C', j): tb[j] for j in range(3)}
                    for signs in itertools.product((1, -1), repeat=4):
                        da1 = da0 | {('T', t): signs[t] if tb[0] == special[t] else tb[0] for t in range(2)}
                        db1 = db0 | {('U', t): signs[2 + t] if sa[0] == special[t] else sa[0] for t in range(2)}
                        if sum((not accepts(q, s, r, t) for q, s in da1.items() for r, t in db1.items())) == 1:
                            found = (da1, db1)
                            break
                    if found:
                        break
                if found:
                    break
            assert found
            da, db = found
        completions.append((da, db))
    for j in range(9):
        used = set()
        for l in range(L):
            for b in range(2 ** l):
                for bit in range(2):
                    inds = [ix[j, l + 1, bit * 2 ** l + b]] + [ix[(j + 1 + 2 * k + bit) % 9, l, b] for k in range(4)]
                    assert not used & set(inds)
                    used.update(inds)
                    total += block_loss(d[inds[0]], d[inds[1]])
                    assert all((((j + 1 + 2 * k + bit) % 9 in ((j + 1) % 9, (j + 2) % 9)) == (k == 0) for k in range(4)))
        defects = set(range(n)) - used
        assert len(defects) == 1 + 8 * 2 ** L
        for k in defects:
            atom, l, b = next((key for key, v in ix.items() if v == k))
            if atom == j:
                assert l == 0
                da, db = (flagA, flagB)
            else:
                assert l == L
                corner = (atom - j - 1) % 9 // 2
                da, db = completions[corner]
            total += d[k] ** 2 * sum((not accepts(q, s, r, t) for q, s in da.items() for r, t in db.items()))
            assert (da['R', 0] == 'f') == (atom == j)
            assert (da['R', 1] == (1, 1, 1)) == (atom in ((j + 1) % 9, (j + 2) % 9))
    formula = (2 * sum(np.diff(c) ** 2) + c[0] ** 2 + 8 * c[-1] ** 2) / 243
    return {'L': L, 'dimension': n, 'enumerated_loss': total / 243, 'formula': formula, 'error': abs(total / 243 - formula), 'link_loss': link}
R['full_game_checks'] = [full_loss(c) for c in ([1, 1], [1, 0.2, 2], [0.3, 1, 0.4, 0.1])]
assert max((x['error'] for x in R['full_game_checks'])) < 2e-14
root = -ai_zeros(1)[0][0]
kap = a * 2 * root ** 1.5 / math.sqrt(27) * math.sqrt(2 / 243)

def profile(c):
    c = c / np.linalg.norm(c)
    w = c * c
    lv = np.arange(len(c))
    eps = (2 * np.dot(np.diff(c), np.diff(c)) + w[0] + 8 * w[-1]) / 243
    hw = -np.dot(w[w > 0], np.log(w[w > 0]))
    ml = np.dot(lv, w)
    return (eps, math.log(9) + hw + a * ml, a * (len(c) - ml) + math.log1p(-2.0 ** (-len(c))) - hw, ml)
airyrows = []
for mu in (1000.0, 1000000.0, 1000000000.0):
    L = int(18 * (2 * mu) ** (1 / 3) + 40)
    for wall in ('bottom', 'top'):
        x = np.arange(L + 1, dtype=float)
        if wall == 'top':
            x = L - x
        diag = x + 4 * mu
        diag[0] -= mu
        diag[-1] += 6 * mu
        eig, u = eigh_tridiagonal(diag, np.full(L, -2 * mu), select='i', select_range=(0, 0))
        eps, S, D, ml = profile(abs(u[:, 0]))
        distance = ml if wall == 'bottom' else L + 1 - ml
        airyrows.append({'mu': mu, 'L': L, 'wall': wall, 'loss': eps, 'S_sqrt_loss': S * math.sqrt(eps), 'D_sqrt_loss': D * math.sqrt(eps), 'distance_term': a * distance * math.sqrt(eps)})
R['airy'] = airyrows
norm = quad(lambda x: airy(x - root)[0] ** 2, 0, 50, epsabs=1e-12)[0]
V = quad(lambda x: x * airy(x - root)[0] ** 2, 0, 50, epsabs=1e-12)[0] / norm
T = quad(lambda x: airy(x - root)[1] ** 2, 0, 50, epsabs=1e-12)[0] / norm
R['constants'] = {'airy_zero': root, 'kappa': kap, 'sine': math.pi * a / math.sqrt(486), 'gaussian': a * a / 486, 'wall_coefficient': a / math.sqrt(486), 'airy_T': T, 'airy_V': V, 'airy_V_sqrt_T': V * math.sqrt(T)}
R['gaussian'] = []
for s in (2, 8, 32, 128):
    lv = np.arange(-int(12 * s), int(12 * s) + 1, dtype=float)
    w = np.exp(-lv * lv / (2 * s * s))
    w /= sum(w)
    c = np.sqrt(w)
    E = sum(np.diff(np.r_[0, c, 0]) ** 2)
    eps = 2 * E / 243
    logh = np.log(w) - a * lv - math.log(9)
    mean = np.dot(w, logh)
    R['gaussian'].append({'s': s, 'eps': eps, 'eps_var_logh': eps * np.dot(w, (logh - mean) ** 2), 'eps_var_level_ln2': eps * a * a * np.dot(w, lv * lv), 'analytic_E': 2 * (1 - math.exp(-1 / (8 * s * s)))})

def sorted_H2(levels, weights):
    levels = np.array(levels, float)
    weights = np.array(weights, float)
    weights /= sum(weights)
    length = 9 * np.exp2(levels)
    dens = weights / length
    order = np.argsort(-dens)
    length = length[order]
    dens = dens[order]
    edge = np.r_[0, np.cumsum(length)]
    pts = np.unique(np.r_[edge, edge / 2])

    def val(t):
        k = np.searchsorted(edge, t, side='right') - 1
        return math.sqrt(dens[k]) if 0 <= k < len(dens) else 0.0
    return (sum(((v - u) * (val((u + v) / 2) - math.sqrt(2) * val(u + v)) ** 2 for u, v in zip(pts[:-1], pts[1:]))), order.tolist())
R['rearrangement_counterexamples'] = []
for lv, w in (([-1, 0, 1], [0.01, 0.01, 0.98]), ([-2, -1, 0, 1, 2], [0.02, 0.08, 0.8, 0.08, 0.02])):
    c = np.sqrt(w)
    E = sum(np.diff(np.r_[0, c, 0]) ** 2)
    H, order = sorted_H2(lv, w)
    R['rearrangement_counterexamples'].append({'levels': lv, 'weights': w, 'ladder_E': E, 'sorted_H2': H, 'density_order': order, 'difference': H - E})
R['sorted_contraction_checks'] = []
rng = np.random.default_rng(429)
for _ in range(50):
    lv = np.arange(-5, 6)
    w = rng.random(len(lv))
    w /= sum(w)
    E = sum(np.diff(np.r_[0, np.sqrt(w), 0]) ** 2)
    hs, _ = sorted_H2(lv, w)
    assert hs <= E + 2e-14
    R['sorted_contraction_checks'].append({'E': E, 'H2': hs})
lv = np.arange(-100, 101)
w = np.where(lv < 0, 1.5 ** lv, 0.8 ** lv)
w /= sum(w)
R['monotone_bilateral_approximation'] = {'E': float(sum(np.diff(np.r_[0, np.sqrt(w), 0]) ** 2)), 'H2': sorted_H2(lv, w)[0]}
H = math.sqrt(2 - math.sqrt(2))
AA = math.log2(1 / H) + 2
R['lemma_Lii_pure_counterexample'] = {'S': 0, 'H2': H * H, 'H': H, 'A': AA, 'rejected_candidate_value': a * (1 - H * (AA + 1)) ** 2 / (2 * H) - 1}
assert R['lemma_Lii_pure_counterexample']['rejected_candidate_value'] > 0
R['variance_identity_counterexamples'] = []
for L in (10, 100, 1000):
    lv = np.array([0, L])
    w = np.array([0.25, 0.75])
    z = np.log(w) - a * lv
    vl = np.dot(w, (lv - np.dot(w, lv)) ** 2)
    vz = np.dot(w, (z - np.dot(w, z)) ** 2)
    R['variance_identity_counterexamples'].append({'L': L, 'Var_logh_minus_ln2sq_Var_level': vz - a * a * vl})
src = OUT / 'price_checks.py'
R['supplied_copy_sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
assert src.read_bytes() == (OUT / 'price_checks.py').read_bytes()
R['elapsed_seconds'] = time.perf_counter() - START
try:
    import ctypes
    from ctypes import wintypes

    class COUNTERS(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    ct = COUNTERS()
    ct.cb = ctypes.sizeof(ct)
    proc = ctypes.windll.kernel32.GetCurrentProcess
    proc.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(COUNTERS), wintypes.DWORD]
    query.restype = wintypes.BOOL
    assert query(proc(), ctypes.byref(ct), ct.cb)
    R['peak_working_set_bytes'] = ct.PeakWorkingSetSize
except Exception as exc:
    R['memory_measurement_error'] = str(exc)
(OUT / 'independent_results.json').write_text(json.dumps(R, indent=2), encoding='utf-8')
print(json.dumps(R, indent=2), flush=True)
