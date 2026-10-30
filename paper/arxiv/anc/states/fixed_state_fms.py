if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import json, itertools, math
import numpy as np
from math import comb
from scipy.optimize import linprog
OUT = os.path.dirname(os.path.abspath(__file__))
GAME_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base-game.json')
ROWS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
ETA = (1, 1, -1)
COLS = [[t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == ETA[j]] for j in range(3)]
C0, C1T = ((1, 1, 1), (1, -1, -1))
CX = [C0, C1T]
RY = [C0, C1T]

def A_answers(q):
    if q[0] == 'R':
        return [list(t) for t in ROWS] + ['f']
    x = int(q[1])
    return [list(t) for t in COLS[0] if t != CX[x]] + [1, -1]

def B_answers(r):
    if r[0] == 'C':
        j = int(r[1]) - 1
        return [list(t) for t in COLS[j]] + ['f']
    y = int(r[1])
    return [list(t) for t in ROWS if t != RY[y]] + [1, -1]

def wins(q, r, a, b):
    if q[0] == 'R' and r[0] == 'C':
        i, j = (int(q[1]) - 1, int(r[1]) - 1)
        if a == 'f' or b == 'f':
            return a == 'f' and b == 'f'
        return a[j] == b[i]
    if q[0] == 'T' and r == 'C1':
        x = int(q[1])
        if isinstance(a, int):
            return b == 'f' or tuple(b) == CX[x]
        return b != 'f' and list(a) == list(b)
    if q == 'R1' and r[0] == 'U':
        y = int(r[1])
        if isinstance(b, int):
            return a == 'f' or tuple(a) == RY[y]
        return a != 'f' and list(a) == list(b)
    if q[0] == 'T' and r[0] == 'U':
        x, y = (int(q[1]), int(r[1]))
        if isinstance(a, int) and isinstance(b, int):
            return a * b == (-1) ** (x * y)
        return True
    return True
AQ = ['R1', 'R2', 'R3', 'T0', 'T1']
BQ = ['C1', 'C2', 'C3', 'U0', 'U1']

def K1():
    d = json.load(open(GAME_JSON))
    ok = True
    n = 0
    for p in d['supported_pairs']:
        if not (p['alice'].startswith('1:') and p['bob'].startswith('1:')):
            continue
        q, r = (p['alice'][2:], p['bob'][2:])
        A, B = (A_answers(q), B_answers(r))
        ok &= A == d['alice']['1:' + q] and B == d['bob']['1:' + r]
        mine = sorted([[ia, ib] for ia, a in enumerate(A) for ib, b in enumerate(B) if not wins(q, r, a, b)])
        ok &= mine == sorted(p['forbidden'])
        n += 1
    return (ok, n)
I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
SY = np.kron(np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([[0.0, -1.0], [1.0, 0.0]])) * -1.0
SQ = [[np.kron(X, I2), np.kron(I2, X), np.kron(X, X)], [np.kron(I2, Z), np.kron(Z, I2), np.kron(Z, Z)], [np.kron(X, Z), np.kron(Z, X), SY]]

def joint(ops, s):
    P = np.eye(4)
    for O, v in zip(ops, s):
        P = P @ (np.eye(4) + v * O) / 2
    return P

def emb(P4=None, flag=0.0):
    M = np.zeros((5, 5))
    M[0, 0] = flag
    if P4 is not None:
        M[1:, 1:] = P4
    return M
cvec = [np.array([1.0, 0.0, 1.0, 0.0]) / math.sqrt(2), np.array([0.0, 1.0, 0.0, 1.0]) / math.sqrt(2)]
rvec = [np.array([1.0, 1.0, 1.0, 1.0]) / 2, np.array([1.0, -1.0, 1.0, -1.0]) / 2]

def signp(v4, s):
    v = np.concatenate([[1.0], s * v4]) / math.sqrt(2)
    return np.outer(v, v)

def A_pvm(q):
    out = []
    for a in A_answers(q):
        if q[0] == 'R':
            i = int(q[1]) - 1
            out.append(emb(None, 1.0) if a == 'f' else emb(joint(SQ[i], a)))
        else:
            x = int(q[1])
            out.append(signp(cvec[x], a) if isinstance(a, int) else emb(joint([SQ[0][0], SQ[1][0], SQ[2][0]], a)))
    return out

def B_pvm(r):
    out = []
    for b in B_answers(r):
        if r[0] == 'C':
            j = int(r[1]) - 1
            out.append(emb(None, 1.0) if b == 'f' else emb(joint([SQ[0][j], SQ[1][j], SQ[2][j]], b)))
        else:
            y = int(r[1])
            out.append(signp(rvec[y], b) if isinstance(b, int) else emb(joint(SQ[0], b)))
    return out
APV = {q: A_pvm(q) for q in AQ}
BPV = {r: B_pvm(r) for r in BQ}
FORB = [(q, r, ia, ib) for q in AQ for r in BQ for ia, a in enumerate(A_answers(q)) for ib, b in enumerate(B_answers(r)) if not wins(q, r, a, b)]

def forbidden_mass(c):
    C = np.diag(c)
    tot = 0.0
    for q, r, ia, ib in FORB:
        tot += float(np.trace(C @ APV[q][ia] @ C @ BPV[r][ib].T))
    return tot

def K2():
    perr = 0.0
    for q in AQ:
        P = APV[q]
        perr = max(perr, np.abs(sum(P) - np.eye(5)).max(), max((np.abs(p @ p - p).max() for p in P)))
    for r in BQ:
        P = BPV[r]
        perr = max(perr, np.abs(sum(P) - np.eye(5)).max(), max((np.abs(p @ p - p).max() for p in P)))
    c = np.array([1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]) / 3
    return (forbidden_mass(c), perr)

def K3():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(20):
        a, b = rng.random(2)
        worst = max(worst, abs(forbidden_mass(np.array([a, b, b, b, b])) - (math.sqrt(2) * a - b) ** 2))
    return worst

def block_types(D):
    return list(itertools.combinations_with_replacement(D, 4))

def block_unit_loss(lam, offs):
    return forbidden_mass(np.array([1.0] + [lam ** (d / 2) for d in offs]))

def powers_upper_bound(lam, n, D, cache):
    levels = n + 1
    logv = [k * math.log(lam) - n * math.log(1 + lam) for k in range(levels)]
    Mlev = np.array([math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + logv[k]) for k in range(levels)])
    types = block_types(D)
    rows_i = []
    A_cols = []
    cost = []
    onemass = []
    for i in range(levels):
        for t in types:
            lv = [i + d for d in t]
            if min(lv) < 0 or max(lv) >= levels:
                continue
            col = np.zeros(levels)
            col[i] += 1.0
            for d, l in zip(t, lv):
                col[l] += lam ** d
            bm = 1.0 + sum((lam ** d for d in t))
            A_cols.append(col)
            cost.append(cache[t] - bm)
            onemass.append(math.exp(logv[i]) * bm)
    A = np.array(A_cols).T
    cost = np.array(cost)
    res = linprog(cost, A_ub=A, b_ub=Mlev, bounds=(0, None), method='highs')
    y = np.clip(res.x, 0, None)
    for _ in range(200):
        use = A @ y
        bad = np.where(use > Mlev)[0]
        if len(bad) == 0:
            break
        for l in bad:
            f = Mlev[l] / use[l] * (1 - 1e-12)
            cols_l = A[l] > 0
            y[cols_l] *= f
    assert (A @ y <= Mlev).all()
    loss_lp = (float(cost @ y) + 1.0) / 25.0
    support = y > 1e-15
    rounding = float(np.sum(np.array(onemass)[support])) / 25.0
    H = n * (math.log(1 + lam) - lam * math.log(lam) / (1 + lam))
    return (loss_lp, loss_lp + rounding, H)
if __name__ == '__main__':
    out = {}
    ok, n1 = K1()
    out['K1_predicate_matches_enumerated'] = ok
    out['K1_pairs'] = n1
    print('K1 predicate vs enumerated game.json copy one:', ok, n1, flush=True)
    fm, perr = K2()
    out['K2_ideal_forbidden_mass'] = fm
    out['K2_pvm_err'] = perr
    print('K2 ideal forbidden mass:', fm, 'PVM err:', perr, flush=True)
    k3 = K3()
    out['K3_identity_err'] = k3
    print('K3 (sqrt2 a - b)^2 identity max err:', k3, flush=True)
    D = [-3, -2, -1, 0, 1, 2, 3]
    res = {}
    for lam, name in [(0.5, '1/2'), (2 ** (-0.5), '2^-1/2'), (1 / 3, '1/3'), (0.25, '1/4')]:
        cache = {t: block_unit_loss(lam, t) for t in block_types(D)}
        best = min(cache.items(), key=lambda kv: kv[1] / (1 + sum((lam ** d for d in kv[0]))))
        rows = []
        for n in [4, 8, 16, 32, 64, 128, 256]:
            lp, li, H = powers_upper_bound(lam, n, D, cache)
            rows.append(dict(n=n, upper_bound_LP=lp, upper_bound_integer=li, entropy_nats=H))
            print(f'lam={name:7s} n={n:4d}  u_n(LP)={lp:.6e}  u_n(int)={li:.6e}  S={H:.3f} nats', flush=True)
        res[name] = dict(best_block_offsets=list(best[0]), best_block_loss_per_unit_mass=best[1] / (1 + sum((lam ** d for d in best[0]))) / 25.0, rows=rows)
        print(f"   best block type {best[0]}: per-unit-mass game loss {res[name]['best_block_loss_per_unit_mass']:.4e}", flush=True)
    out['K4'] = res
    if not (ok and n1==25 and max(abs(fm),perr,k3)<1e-11): raise RuntimeError('Fixed-state finite checks failed')
    for item in res.values():
        for row in item['rows']:
            if not (0<=row['upper_bound_LP']<=row['upper_bound_integer']+1e-12 and row['upper_bound_integer']<=1): raise RuntimeError('State approximation diagnostic failed')
    json.dump(out, open(os.path.join(OUT, 'fixed_state_fms_results.json'), 'w'), indent=1)
