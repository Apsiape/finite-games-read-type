if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[v] = '1'
import json, math, itertools
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
res = {}
LN2 = math.log(2.0)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
Y = np.array([[0, -1j], [1j, 0]])
kron = np.kron
sq = [[kron(X, I2), kron(I2, X), kron(X, X)], [kron(I2, Z), kron(Z, I2), kron(Z, Z)], [kron(X, Z), kron(Z, X), kron(Y, Y)]]
eta = [1, 1, -1]
R = [s for s in itertools.product([1, -1], repeat=3) if s[0] * s[1] * s[2] == 1]
C = [[t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == eta[j]] for j in range(3)]
c0 = c_0 = (1, 1, 1)
c1 = (1, -1, -1)
cx = [c0, c1]
ry = [c0, c1]

def joint_proj(ops, signs):
    P = np.eye(4, dtype=complex)
    for O, s in zip(ops, signs):
        P = P @ (np.eye(4) + s * O) / 2
    return P

def emb5(P4=None, flag=0.0):
    M = np.zeros((5, 5), dtype=complex)
    M[0, 0] = flag
    if P4 is not None:
        M[1:, 1:] = P4
    return M
A = {}
for i in range(3):
    A['R', i] = {s: emb5(joint_proj(sq[i], s)) for s in R}
    A['R', i]['f'] = emb5(None, 1.0)
B = {}
for j in range(3):
    col = [sq[0][j], sq[1][j], sq[2][j]]
    B['C', j] = {t: emb5(joint_proj(col, t)) for t in C[j]}
    B['C', j]['f'] = emb5(None, 1.0)

def vec5(fl, v4):
    w = np.zeros(5, dtype=complex)
    w[0] = fl
    w[1:] = v4
    return w
col1 = [sq[0][0], sq[1][0], sq[2][0]]
row1 = sq[0]
for x in range(2):
    P = joint_proj(col1, cx[x])
    w, U = np.linalg.eigh(P)
    v = U[:, -1]
    v = v * np.exp(-1j * np.angle(v[np.argmax(abs(v))]))
    v = np.real_if_close(v)
    d = {}
    for a in (1, -1):
        u = vec5(1.0, a * v) / math.sqrt(2)
        d[a] = np.outer(u, u.conj())
    for t in C[0]:
        if t != cx[x]:
            d[t] = emb5(joint_proj(col1, t))
    A['T', x] = d
for y in range(2):
    P = joint_proj(row1, ry[y])
    w, U = np.linalg.eigh(P)
    v = U[:, -1]
    v = v * np.exp(-1j * np.angle(v[np.argmax(abs(v))]))
    v = np.real_if_close(v)
    d = {}
    for b in (1, -1):
        u = vec5(1.0, b * v) / math.sqrt(2)
        d[b] = np.outer(u, u.conj())
    for s in R:
        if s != ry[y]:
            d[s] = emb5(joint_proj(row1, s))
    B['U', y] = d

def sign_vec(Pq):
    w, U = np.linalg.eigh(Pq)
    return U[:, -1]
vc = [sign_vec(joint_proj(col1, cx[x])) for x in range(2)]
vr = [sign_vec(joint_proj(row1, ry[y])) for y in range(2)]
ph_r0 = np.vdot(vc[0], vr[0])
vr[0] = vr[0] * abs(ph_r0) / ph_r0
ph_r1 = np.vdot(vc[0], vr[1])
vr[1] = vr[1] * abs(ph_r1) / ph_r1
ph_c1 = np.vdot(vc[1], vr[0])
vc[1] = vc[1] * ph_c1 / abs(ph_c1)
overl = [[np.vdot(vc[x], vr[y]) for y in range(2)] for x in range(2)]
res['C1_overlaps'] = [[complex(o).real for o in row] for row in overl]
for x in range(2):
    for a in (1, -1):
        u = vec5(1.0, a * vc[x]) / math.sqrt(2)
        A['T', x][a] = np.outer(u, u.conj())
for y in range(2):
    for b in (1, -1):
        u = vec5(1.0, b * vr[y]) / math.sqrt(2)
        B['U', y][b] = np.outer(u, u.conj())

def wins(qa, a, qb, b):
    ta, ia = qa
    tb, jb = qb
    if ta == 'R' and tb == 'C':
        if a == 'f' or b == 'f':
            return a == 'f' and b == 'f'
        return a[jb] == b[ia]
    if ta == 'T' and tb == 'C' and (jb == 0):
        if a in (1, -1):
            return b == 'f' or b == cx[ia]
        return a == b
    if ta == 'R' and tb == 'U' and (ia == 0):
        if b in (1, -1):
            return a == 'f' or a == ry[jb]
        return a == b
    if ta == 'T' and tb == 'U':
        if a in (1, -1) and b in (1, -1):
            return a * b == (-1) ** (ia * jb)
        return True
    return True

def forbidden_mass(a_coef, b_coef):
    psi = np.zeros(25, dtype=complex)
    psi[0] = a_coef
    for z in range(4):
        psi[(z + 1) * 5 + (z + 1)] = b_coef
    tot = 0.0
    for qa in A:
        for qb in B:
            for a, Ea in A[qa].items():
                for b, Fb in B[qb].items():
                    if not wins(qa, a, qb, b):
                        tot += float(np.real(np.vdot(psi, np.kron(Ea, Fb.conj()) @ psi)))
    return tot
comp = max((np.max(abs(sum(A[q].values()) - np.eye(5))) for q in A))
comp = max(comp, max((np.max(abs(sum(B[q].values()) - np.eye(5))) for q in B)))
rng = np.random.default_rng(7)
c1errs = []
for _ in range(6):
    a_, b_ = rng.uniform(0.1, 1.0, size=2)
    fm = forbidden_mass(a_, b_)
    c1errs.append(abs(fm - (math.sqrt(2) * a_ - b_) ** 2))
res['C1_completeness_err'] = float(comp)
res['C1_identity_max_err'] = float(max(c1errs))
res['C1_ideal_forbidden'] = forbidden_mass(1 / 3, math.sqrt(2) / 3)
print('C1 overlaps', res['C1_overlaps'], 'completeness', comp, 'identity err', max(c1errs), flush=True)

def template(c):
    c = np.asarray(c, float)
    c = c / np.linalg.norm(c)
    L = len(c) - 1
    w = c ** 2
    loss = (2 * np.sum(np.diff(c) ** 2) + c[0] ** 2 + 8 * c[-1] ** 2) / 243
    Hw = -np.sum(w[w > 0] * np.log(w[w > 0]))
    lv = np.arange(L + 1)
    S_I = math.log(9) + Hw + LN2 * np.sum(lv * w)
    dim = 9 * (2 ** (L + 1) - 1) if L < 1000 else None
    lnd = math.log(9) + (L + 1) * LN2 + math.log1p(-2.0 ** (-(L + 1)))
    D_II1 = lnd - S_I
    lnh = np.log(np.where(w > 0, w, 1)) - lv * LN2 - math.log(9)
    m1 = np.sum(w * lnh)
    var_lnh = np.sum(w * (lnh - m1) ** 2)
    m2 = np.sum(w * lv)
    var_l = np.sum(w * (lv - m2) ** 2)
    return dict(L=L, loss=float(loss), S_I=float(S_I), D_II1=float(D_II1), var_lnh=float(var_lnh), var_level_ln2sq=float(var_l * LN2 ** 2), mean_level=float(m2), H_w=float(Hw))
chk = []
for L in range(1, 6):
    t = template(np.ones(L + 1))
    chk.append(dict(L=L, loss=t['loss'], loss_formula=1 / (27 * (L + 1)), S=t['S_I'], S_formula=math.log(9) + math.log(L + 1) + L / 2 * LN2))
res['C2_uniform_vs_prior'] = chk
chk = []
for L in (2, 3, 5, 50):
    lv = np.arange(L + 1)
    c = np.sqrt(2 / L) * np.sin(np.pi * lv / L)
    t = template(c)
    chk.append(dict(L=L, loss=t['loss'], loss_formula=4 * (1 - math.cos(math.pi / L)) / 243, S=t['S_I']))
res['C2_sine_vs_verifier'] = chk
print('C2 uniform', res['C2_uniform_vs_prior'][-1], flush=True)
print('C2 sine', res['C2_sine_vs_verifier'], flush=True)

def airy_profile(L, mu, wall='bottom'):
    n = L + 1
    x = np.arange(n, dtype=float) if wall == 'bottom' else L - np.arange(n, dtype=float)
    Q = np.zeros((n, n))
    for l in range(L):
        Q[l, l] += 2
        Q[l + 1, l + 1] += 2
        Q[l, l + 1] -= 2
        Q[l + 1, l] -= 2
    Q[0, 0] += 1
    Q[L, L] += 8
    Mtx = np.diag(x) + mu * Q
    w, U = np.linalg.eigh(Mtx)
    c = np.abs(U[:, 0])
    return c

def gauss_profile(L, s):
    lv = np.arange(L + 1)
    m = L / 2
    return np.exp(-(lv - m) ** 2 / (4 * s * s))
a1 = 2.338107410459767
kappaA = 2 * a1 ** 1.5 / 3 ** 1.5
const_I = LN2 * kappaA * math.sqrt(2 / 243)
const_sine = math.pi * LN2 / math.sqrt(486)
const_var = LN2 ** 2 / 486
res['C3_constants'] = dict(a1=a1, kappaA=kappaA, airy_entropy_const=const_I, sine_entropy_const=const_sine, prior_uniform_const_times_eps=LN2 / 54, gaussian_var_const=const_var)
print('C3 constants', res['C3_constants'], flush=True)
rows = []
for mu in [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
    L = int(12 * mu ** (1 / 3) + 40)
    c = airy_profile(L, mu, 'bottom')
    t = template(c)
    rows.append(dict(mu=mu, L=L, eps=t['loss'], S_I=t['S_I'], S_sqrt_eps=t['S_I'] * math.sqrt(t['loss']), leading=LN2 * t['mean_level'] * math.sqrt(t['loss']), S_minus_logs=(t['S_I'] - math.log(9) - t['H_w']) * math.sqrt(t['loss'])))
res['C3_airy_typeI'] = rows
for r in rows:
    print('airy I', {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
rows = []
for mu in [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
    L = int(12 * mu ** (1 / 3) + 40)
    c = airy_profile(L, mu, 'top')
    t = template(c)
    rows.append(dict(mu=mu, L=L, eps=t['loss'], D_II1=t['D_II1'], D_sqrt_eps=t['D_II1'] * math.sqrt(t['loss']), leading=LN2 * (L + 1 - t['mean_level']) * math.sqrt(t['loss'])))
res['C3_airy_II1'] = rows
for r in rows:
    print('airy II1', {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
rows = []
for s in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
    L = int(16 * s + 20)
    c = gauss_profile(L, s)
    t = template(c)
    rows.append(dict(s=s, L=L, eps=t['loss'], var_level_eps=t['var_level_ln2sq'] * t['loss'], var_lnh_eps=t['var_lnh'] * t['loss']))
res['C3_gauss_IIinf'] = rows
for r in rows:
    print('gauss', {k: round(v, 8) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
rows = []
for L in [10, 30, 100, 300, 1000, 3000]:
    lv = np.arange(L + 1)
    c = np.sqrt(2 / L) * np.sin(np.pi * lv / L)
    t = template(c)
    rows.append(dict(L=L, eps=t['loss'], S_sqrt_eps=t['S_I'] * math.sqrt(t['loss'])))
res['C3_sine'] = rows
print('sine', rows[-1], flush=True)

def rearr_hellinger(c, levels_trace):
    edges = np.concatenate([[0.0], np.cumsum([T for T, _ in levels_trace])])
    vals = np.array([math.sqrt(hv) for _, hv in levels_trace])
    tot = 0.0
    pts = set(edges.tolist()) | set((edges / 2).tolist())
    pts = np.array(sorted((p for p in pts if p <= edges[-1])))

    def lam(t):
        k = np.searchsorted(edges, t, side='right') - 1
        return vals[k] if 0 <= k < len(vals) else 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (a + b)
        tot += (lam(mid) - math.sqrt(2) * lam(2 * mid)) ** 2 * (b - a)
    return tot
rows = []
for L, s in [(60, 4.0), (120, 8.0), (240, 16.0)]:
    c = gauss_profile(L, s)
    c = c / np.linalg.norm(c)
    w = c ** 2
    lev = [(9 * 2.0 ** l, w[l] / (9 * 2.0 ** l)) for l in range(L + 1)]
    lev.sort(key=lambda p: -p[1])
    H2 = rearr_hellinger(c, lev)
    t = template(c)
    edges = np.concatenate([[0.0], np.cumsum([T for T, _ in lev])])
    m1 = m2 = 0.0
    for (T, hv), e0 in zip(lev, edges[:-1]):
        a_, b_ = (e0, e0 + T)
        if a_ == 0:
            a_ = 1e-300

        def F1(u):
            return u * (math.log(u) - 1) / LN2

        def F2(u):
            return u * ((math.log(u) / LN2) ** 2 - 2 * math.log(u) / LN2 ** 2 + 2 / LN2 ** 2)
        m1 += hv * (F1(b_) - (F1(a_) if a_ > 1e-299 else 0.0))
        m2 += hv * (F2(b_) - (F2(a_) if a_ > 1e-299 else 0.0))
    varT = m2 - m1 ** 2
    rows.append(dict(L=L, s=s, eps=t['loss'], H2=H2, H2_over_eps=H2 / t['loss'], var_log2T=varT, HCR_bound=(1 - H2) / (4 * H2)))
res['C4_rearrangement'] = rows
for r in rows:
    print('C4', {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
print('Abstract spectral checks complete; no physical energy or cosmological bound is inferred.', flush=True)
if max(res['C1_completeness_err'],res['C1_identity_max_err'],abs(res['C1_ideal_forbidden']))>1e-11: raise RuntimeError('Price bridge identities failed')
for row in res['C2_uniform_vs_prior']+res['C2_sine_vs_verifier']:
    if abs(row['loss']-row['loss_formula'])>1e-11: raise RuntimeError('Endpoint formula failed')
if any(row['var_log2T']+1e-7<row['HCR_bound'] for row in res['C4_rearrangement']): raise RuntimeError('Variance comparison failed')
with open(os.path.join(OUT, 'price_checks_results.json'), 'w') as fh:
    json.dump(res, fh, indent=1, default=float)
print('done', flush=True)
