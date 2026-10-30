if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[v] = '1'
import json, math, itertools, sys
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
res = {}
rng = np.random.default_rng(20261029)
Rset = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
eta = (1, 1, -1)
Cset = [[t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == eta[j]] for j in range(3)]
c = [(1, 1, 1), (1, -1, -1)]
r = [(1, 1, 1), (1, -1, -1)]

def alice_answers(q):
    if q[0] == 'R':
        return list(Rset) + ['f']
    x = int(q[1])
    return ['+', '-'] + [t for t in Cset[0] if t != c[x]]

def bob_answers(q):
    if q[0] == 'C':
        j = int(q[1]) - 1
        return list(Cset[j]) + ['f']
    y = int(q[1])
    return ['+', '-'] + [s for s in Rset if s != r[y]]
AQ = ['R1', 'R2', 'R3', 'T0', 'T1']
BQ = ['C1', 'C2', 'C3', 'U0', 'U1']

def fms_win(qa, qb, a, b):
    if qa[0] == 'R' and qb[0] == 'C':
        i, j = (int(qa[1]) - 1, int(qb[1]) - 1)
        if a == 'f' and b == 'f':
            return True
        if a == 'f' or b == 'f':
            return False
        return a[j] == b[i]
    if qa[0] == 'T' and qb == 'C1':
        x = int(qa[1])
        if a in ('+', '-'):
            return b == 'f' or b == c[x]
        return a == b
    if qa == 'R1' and qb[0] == 'U':
        y = int(qb[1])
        if b in ('+', '-'):
            return a == 'f' or a == r[y]
        return a == b
    if qa[0] == 'T' and qb[0] == 'U':
        x, y = (int(qa[1]), int(qb[1]))
        if a in ('+', '-') and b in ('+', '-'):
            return (1 if a == '+' else -1) * (1 if b == '+' else -1) == (-1) ** (x * y)
        return True
    return True

def giii_pairs():
    pairs = []
    for jcopy in range(9):
        for qa in AQ:
            for qb in BQ:
                fb = []
                for ia, a in enumerate(alice_answers(qa)):
                    for ib, b in enumerate(bob_answers(qb)):
                        if not fms_win(qa, qb, a, b):
                            fb.append((ia, ib))
                pairs.append((f'{jcopy}:{qa}', f'{jcopy}:{qb}', fb))
        fb = []
        for ia, a in enumerate(alice_answers('R1')):
            for b in range(9):
                if (a == 'f') != (b == jcopy):
                    fb.append((ia, b))
        pairs.append((f'{jcopy}:R1', 'Lab', fb))
        fb = []
        for ia, a in enumerate(alice_answers('R2')):
            for b in range(9):
                if (a == (1, 1, 1)) != (b in ((jcopy + 1) % 9, (jcopy + 2) % 9)):
                    fb.append((ia, b))
        pairs.append((f'{jcopy}:R2', 'Lab', fb))
    return pairs
PAIRS = giii_pairs()
try:
    g = json.load(open(os.path.join(OUT, 'giii-game.json')))

    def lab(x):
        return tuple(x) if isinstance(x, list) else x
    mism = 0
    mine = {(pa, pb): set(((alice_answers(pa.split(':')[1])[ia], bob_answers(pb.split(':')[1])[ib] if pb != 'Lab' else ib) for ia, ib in fb)) for pa, pb, fb in PAIRS}
    theirs = {}
    for P in g['supported_pairs']:
        sg = lambda x: ('+' if x == 1 else '-') if not isinstance(x, list) and x in (1, -1) else lab(x)
        A = [sg(x) for x in g['alice'][P['alice']]]
        B = [lab(x) for x in g['bob'][P['bob']]] if P['bob'] == 'Lab' else [sg(x) for x in g['bob'][P['bob']]]
        theirs[P['alice'], P['bob']] = set(((A[i], B[j]) for i, j in P['forbidden']))
    for k in mine:
        if mine[k] != theirs.get(k):
            mism += 1
    res['C1'] = {'pairs_mine': len(mine), 'pairs_json': len(theirs), 'mismatched_pairs': mism}
except Exception as ex:
    res['C1'] = {'error': repr(ex)}
print('C1', res['C1'], flush=True)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
Y = np.array([[0, -1j], [1j, 0]])
M = [[np.kron(X, I2), np.kron(I2, X), np.kron(X, X)], [np.kron(I2, Z), np.kron(Z, I2), np.kron(Z, Z)], [np.kron(X, Z), np.kron(Z, X), np.kron(Y, Y)]]

def emb(P4):
    P = np.zeros((5, 5), dtype=complex)
    P[1:, 1:] = P4
    return P
Pf = np.zeros((5, 5), dtype=complex)
Pf[0, 0] = 1

def joint(ops, signs):
    P = np.eye(4, dtype=complex)
    for O, s in zip(ops, signs):
        P = P @ (np.eye(4) + s * O) / 2
    return P

def vecproj(v):
    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())
cvec = [np.array([1, 0, 1, 0]) / math.sqrt(2), np.array([0, 1, 0, 1]) / math.sqrt(2)]
rvec = [np.array([1, 1, 1, 1]) / 2, np.array([1, -1, 1, -1]) / 2]
for x in range(2):
    assert np.linalg.norm(joint([M[0][0], M[1][0], M[2][0]], c[x]) @ cvec[x] - cvec[x]) < 1e-12
for y in range(2):
    assert np.linalg.norm(joint(M[0], r[y]) @ rvec[y] - rvec[y]) < 1e-12

def ideal_alice(q):
    out = []
    if q[0] == 'R':
        i = int(q[1]) - 1
        for a in alice_answers(q):
            out.append(Pf if a == 'f' else emb(joint(M[i], a)))
    else:
        x = int(q[1])
        fv = np.zeros(5)
        fv[0] = 1
        cv = np.zeros(5, dtype=complex)
        cv[1:] = cvec[x]
        for a in alice_answers(q):
            if a == '+':
                out.append(vecproj(fv + cv))
            elif a == '-':
                out.append(vecproj(fv - cv))
            else:
                out.append(emb(joint([M[0][0], M[1][0], M[2][0]], a)))
    return out

def ideal_bob(q):
    out = []
    if q[0] == 'C':
        j = int(q[1]) - 1
        col = [M[0][j], M[1][j], M[2][j]]
        for b in bob_answers(q):
            out.append(Pf if b == 'f' else emb(joint(col, b)))
    else:
        y = int(q[1])
        fv = np.zeros(5)
        fv[0] = 1
        rv = np.zeros(5, dtype=complex)
        rv[1:] = rvec[y]
        for b in bob_answers(q):
            if b == '+':
                out.append(vecproj(fv + rv))
            elif b == '-':
                out.append(vecproj(fv - rv))
            else:
                out.append(emb(joint(M[0], b)))
    return out
IA = {q: ideal_alice(q) for q in AQ}
IB = {q: ideal_bob(q) for q in BQ}
for d in (IA, IB):
    for q, L in d.items():
        S = sum(L)
        assert np.linalg.norm(S - np.eye(5)) < 1e-12, q
        for P in L:
            assert np.linalg.norm(P @ P - P) < 1e-12

def fms_forbidden_mass(K, A=IA, B=IB):
    tot = 0.0
    for qa in AQ:
        for qb in BQ:
            for ia, a in enumerate(alice_answers(qa)):
                for ib, b in enumerate(bob_answers(qb)):
                    if not fms_win(qa, qb, a, b):
                        V = A[qa][ia] @ K @ B[qb][ib].T
                        tot += float(np.real(np.vdot(V, V)))
    return tot
K0 = np.diag([1 / 3] + [math.sqrt(2) / 3] * 4).astype(complex)
res['C2_ideal_forbidden_mass'] = fms_forbidden_mass(K0)
errs = []
for _ in range(6):
    a, b = rng.random(2)
    K = np.diag([a] + [b] * 4).astype(complex)
    errs.append(abs(fms_forbidden_mass(K) - (math.sqrt(2) * a - b) ** 2))
res['C2_identity_max_err'] = max(errs)
print('C2', res['C2_ideal_forbidden_mass'], res['C2_identity_max_err'], flush=True)

def H2_seq(Lam):
    L = np.sort(np.asarray(Lam, float))[::-1]
    n = len(L)
    doubled = np.repeat(L, 2)
    Lp = np.concatenate([L, np.zeros(n)])
    return 0.5 * float(np.sum((doubled - math.sqrt(2) * Lp) ** 2))

def H2_integral(eigs):
    e = np.sort(np.asarray(eigs, float))[::-1]
    e = e[e > 0]
    n = len(e)
    pts = np.unique(np.concatenate([np.arange(0, n + 1), np.arange(0, n + 1) / 2.0]))
    tot = 0.0
    for t0, t1 in zip(pts[:-1], pts[1:]):
        tm = 0.5 * (t0 + t1)
        p1 = e[int(math.floor(tm))] if tm < n else 0.0
        p2 = e[int(math.floor(2 * tm))] if 2 * tm < n else 0.0
        tot += (t1 - t0) * (math.sqrt(p1) - math.sqrt(2 * p2)) ** 2
    return tot

def perm_cuntz_strategy(Lam):
    Lam = np.asarray(Lam, float)
    n = len(Lam)
    N = 2 * n
    dim = 9 * N
    idx = lambda a, p: a * N + (p - 1)
    K = np.zeros((dim, dim), dtype=complex)
    for a in range(9):
        for p in range(1, n + 1):
            K[idx(a, p), idx(a, p)] = Lam[p - 1] / 3
    A, B = ({}, {})
    for j in range(9):
        pairsj = [((j + 1) % 9, (j + 2) % 9), ((j + 3) % 9, (j + 4) % 9), ((j + 5) % 9, (j + 6) % 9), ((j + 7) % 9, (j + 8) % 9)]
        blocks = []
        covered = set()
        for m in range(1, N + 1):
            par = (m + 1) // 2
            basis = [idx(j, m)] + [idx(pairsj[z][(m - 1) % 2], par) for z in range(4)]
            blocks.append(basis)
            covered.update(basis)
        leftover = [k for k in range(dim) if k not in covered]
        for qa in AQ:
            ops = []
            for ia, P5 in enumerate(IA[qa]):
                P = np.zeros((dim, dim), dtype=complex)
                for bas in blocks:
                    P[np.ix_(bas, bas)] = P5
                if ia == 0:
                    for k in leftover:
                        P[k, k] = 1
                ops.append(P)
            A[f'{j}:{qa}'] = ops
        for qb in BQ:
            ops = []
            for ib, P5 in enumerate(IB[qb]):
                P = np.zeros((dim, dim), dtype=complex)
                for bas in blocks:
                    P[np.ix_(bas, bas)] = P5
                if ib == 0:
                    for k in leftover:
                        P[k, k] = 1
                ops.append(P)
            B[f'{j}:{qb}'] = ops
    lab = []
    for k in range(9):
        P = np.zeros((dim, dim), dtype=complex)
        for p in range(1, N + 1):
            P[idx(k, p), idx(k, p)] = 1
        lab.append(P)
    B['Lab'] = lab
    return (A, B, K)

def giii_loss(A, B, K):
    tot = 0.0
    cacheG = {}
    Kd = K.conj().T
    for pa, pb, fb in PAIRS:
        if pa not in cacheG:
            cacheG[pa] = [Kd @ Ea @ K for Ea in A[pa]]
        G = cacheG[pa]
        Bl = B[pb]
        s = 0.0
        for ia, ib in fb:
            s += float(np.real(np.sum(G[ia] * Bl[ib])))
        tot += s
    return tot / 243.0

def check_strategy_valid(A, B, dim):
    e = 0.0
    for d in (A, B):
        for q, L in d.items():
            e = max(e, np.linalg.norm(sum(L) - np.eye(dim)))
            for P in L:
                e = max(e, np.linalg.norm(P @ P - P), np.linalg.norm(P - P.conj().T))
    return e
c3 = []
tests = [np.ones(4) / 2, np.array([2.0 ** (-k / 2) for k in range(6)]), rng.random(5), rng.random(7)]
for Lam in tests:
    Lam = np.sort(Lam)[::-1]
    Lam = Lam / np.linalg.norm(Lam)
    A, B, K = perm_cuntz_strategy(Lam)
    val = check_strategy_valid(A, B, K.shape[0])
    loss = giii_loss(A, B, K)
    h2 = H2_seq(Lam)
    c3.append({'n': len(Lam), 'pvm_err': val, 'loss': loss, '2H2/243': 2 * h2 / 243, 'abs_diff': abs(loss - 2 * h2 / 243)})
    print('C3', c3[-1], flush=True)
Lam = rng.random(6)
Lam /= np.linalg.norm(Lam)
A, B, K = perm_cuntz_strategy(Lam)
loss = giii_loss(A, B, K)
n = len(Lam)
Lp = np.concatenate([Lam, np.zeros(n)])
unsorted = 0.5 * sum(((Lp[(m + 1) // 2 - 1] - math.sqrt(2) * Lp[m - 1]) ** 2 for m in range(1, 2 * n + 1)))
c3.append({'unsorted_case_loss': loss, '2*unsortedH2/243': 2 * unsorted / 243, 'sortedH2_le_unsorted': H2_seq(Lam) <= unsorted + 1e-15})
print('C3u', c3[-1], flush=True)
res['C3'] = c3
c4 = []
for _ in range(5):
    Lam = np.sort(rng.random(rng.integers(3, 12)))[::-1]
    Lam /= np.linalg.norm(Lam)
    a = H2_seq(Lam)
    b = H2_integral(Lam ** 2)
    amp = np.repeat(Lam / 3, 9)
    c4.append({'seq': a, 'integral': b, 'amp9_integral': H2_integral(amp ** 2), 'max_diff': max(abs(a - b), abs(a - H2_integral(amp ** 2)))})
res['C4'] = c4
print('C4', max((x['max_diff'] for x in c4)), flush=True)
c5 = []
for _ in range(3):
    n = 5
    Lam = np.sort(rng.random(n))[::-1]
    Lam /= np.linalg.norm(Lam)
    target = np.repeat(Lam, 2) / math.sqrt(2)
    best = 0.5 * 2 * H2_seq(Lam) / 2 * 2
    direct = float(np.sum((target - np.concatenate([Lam, np.zeros(n)])) ** 2))
    worst_gap = 1000000000.0
    for _ in range(200):
        Ua = np.linalg.qr(rng.normal(size=(2 * n, n)) + 1j * rng.normal(size=(2 * n, n)))[0]
        Ub = np.linalg.qr(rng.normal(size=(2 * n, n)) + 1j * rng.normal(size=(2 * n, n)))[0]
        Kpsi = np.diag(Lam).astype(complex)
        lhs = Ua @ Kpsi @ Ub.T
        tgt = np.kron(np.diag(Lam), np.eye(2)) / math.sqrt(2)
        worst_gap = min(worst_gap, np.linalg.norm(lhs - tgt) ** 2 - direct)
    c5.append({'H2': H2_seq(Lam), 'min_dist2_aligned': direct, 'random_isometries_min_excess': worst_gap})
res['C5'] = c5
print('C5', c5, flush=True)

def unitary_near(dim, t):
    G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Hm = (G + G.conj().T) / 2
    w, V = np.linalg.eigh(Hm)
    return V * np.exp(1j * t * w) @ V.conj().T
c6 = []
Lam0 = np.array([2.0 ** (-k / 2) * math.sin(math.pi * (k + 1) / 6) for k in range(5)])
Lam0 = np.sort(np.abs(Lam0))[::-1]
Lam0 /= np.linalg.norm(Lam0)
A0, B0, K0g = perm_cuntz_strategy(Lam0)
dim = K0g.shape[0]
base_loss = giii_loss(A0, B0, K0g)
for t_state, t_meas in [(0.02, 0), (0.05, 0), (0.1, 0), (0.2, 0), (0, 0.02), (0, 0.05), (0.05, 0.02), (0.1, 0.05)]:
    ratios = []
    for trial in range(3):
        K = K0g.copy()
        if t_state > 0:
            G = rng.normal(size=K.shape) + 1j * rng.normal(size=K.shape)
            K = K + t_state * G / np.linalg.norm(G)
            K = K / np.linalg.norm(K)
        A, B = (A0, B0)
        if t_meas > 0:
            A = {}
            B = {}
            for q, L in A0.items():
                U = unitary_near(dim, t_meas)
                A[q] = [U @ P @ U.conj().T for P in L]
            for q, L in B0.items():
                U = unitary_near(dim, t_meas)
                B[q] = [U @ P @ U.conj().T for P in L]
        loss = giii_loss(A, B, K)
        sv = np.linalg.svd(K, compute_uv=False)
        ratios.append(H2_seq(sv) / loss)
    c6.append({'t_state': t_state, 't_meas': t_meas, 'max_ratio_H2_over_loss': max(ratios), 'min_ratio': min(ratios)})
    print('C6', c6[-1], flush=True)
res['C6'] = {'base_ratio': H2_seq(Lam0) / base_loss, 'perturbed': c6}
a1 = 2.338107410459767
c_star = 2 * a1 ** 1.5 / 3 ** 1.5
c7 = []
for theta in [10, 100, 1000, 10000.0, 100000.0]:
    nsite = int(12 * theta ** (1 / 3) + 60)
    Dm = 2 * np.eye(nsite) - np.eye(nsite, k=1) - np.eye(nsite, k=-1)
    Hm = theta * Dm + np.diag(np.arange(nsite, dtype=float))
    lam0 = float(np.linalg.eigvalsh(Hm)[0])
    bound = (a1 * theta ** (1 / 3) - 1.5) / (1 + a1 * theta ** (-2 / 3) / 6)
    c7.append({'theta': theta, 'lambda0': lam0, 'a1_theta^(1/3)': a1 * theta ** (1 / 3), 'lower_bound': bound, 'holds': lam0 >= bound})
for x in [0.01, 0.001, 0.0001]:
    best = -1000000000.0
    for theta in np.geomspace(1, 100000000.0, 200):
        nsite = int(12 * theta ** (1 / 3) + 60)
        if nsite > 3000:
            break
        Dm = 2 * np.eye(nsite) - np.eye(nsite, k=1) - np.eye(nsite, k=-1)
        lam0 = float(np.linalg.eigvalsh(theta * Dm + np.diag(np.arange(nsite, dtype=float)))[0])
        best = max(best, lam0 - theta * x)
    c7.append({'x': x, 'legendre_value': best, 'c_star/sqrt(x)': c_star / math.sqrt(x), 'ratio': best * math.sqrt(x) / c_star})
res['C7'] = c7
print('C7', c7, flush=True)
c8 = []
for nsite in [3, 10, 50]:
    Dm = 2 * np.eye(nsite) - np.eye(nsite, k=1) - np.eye(nsite, k=-1)
    c8.append({'n': nsite, 'min_energy': float(np.linalg.eigvalsh(Dm)[0]), 'formula': 2 * (1 - math.cos(math.pi / (nsite + 1)))})
res['C8'] = c8
print('C8', c8, flush=True)

def profile_stats(widths, heights):
    edges = np.concatenate([[0], np.cumsum(widths)])
    mass = heights * widths
    mass = mass / mass.sum()
    heights = heights / (heights * widths).sum()
    EK = float(np.sum(mass * -np.log(heights)))
    EK2 = float(np.sum(mass * np.log(heights) ** 2))
    varK = EK2 - EK ** 2
    Es, Es2 = (0.0, 0.0)
    for k in range(len(widths)):
        a, b = (edges[k], edges[k + 1])
        tt = np.linspace(a, b, 4001)[1:] if a == 0 else np.linspace(a, b, 4001)
        s = np.log2(np.maximum(tt, 1e-300))
        Es += mass[k] * float(np.mean(s))
        Es2 += mass[k] * float(np.mean(s ** 2))
    Vs = Es2 - Es ** 2
    w0 = 0.5 * math.log(max(Vs, 1e-300)) + 1.5 * math.log(3) + math.log(math.log(2))
    EW2 = w0 ** 2 + 3 * w0 + 4.5 if w0 > 0 else 4.5
    bound = max(0.0, math.log(2) * math.sqrt(max(Vs, 0)) - math.sqrt(EW2)) ** 2
    return (varK, Vs, bound)
c9 = []
for trial in range(40):
    nb = rng.integers(2, 40)
    widths = 2.0 ** rng.uniform(-3, 12, size=nb)
    heights = np.sort(2.0 ** rng.uniform(-30, 0, size=nb))[::-1]
    varK, Vs, bound = profile_stats(widths, heights)
    c9.append((varK, Vs, bound, varK >= bound - 1e-09))
varK, Vs, bound = profile_stats(np.array([1.0, 2.0 ** 20]), np.array([1.0, 2.0 ** (-21)]))
res['C9'] = {'random_profiles_all_hold': all((x[3] for x in c9)), 'n': len(c9), 'worst_slack': min((x[0] - x[2] for x in c9)), 'two_level_example': {'varK': varK, 'var_log2T': Vs, 'bound': bound}}
print('C9', res['C9'], flush=True)
if res['C1'].get('mismatched_pairs',-1)!=0 or res['C1'].get('pairs_mine')!=243: raise RuntimeError('Predicate agreement failed')
if max(abs(res['C2_ideal_forbidden_mass']),res['C2_identity_max_err'])>1e-11: raise RuntimeError('Bridge identity failed')
if any(row.get('pvm_err',0)>1e-11 or row.get('abs_diff',0)>1e-11 for row in res['C3']): raise RuntimeError('Converse check failed')
if any(row['max_diff']>1e-11 for row in res['C4']): raise RuntimeError('Integral identity failed')
if any(not row.get('holds',True) for row in res['C7']): raise RuntimeError('Airy diagnostic failed')
if not res['C9']['random_profiles_all_hold']: raise RuntimeError('Profile inequality failed')
if any(abs(row['min_energy']-row['formula'])>1e-11 for row in res['C8']): raise RuntimeError('Endpoint spectrum failed')
json.dump(res, open(os.path.join(OUT, 'checks_r_results.json'), 'w'), indent=1, default=str)
print('done', flush=True)
