if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import json, itertools, math
import numpy as np
GAME_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base-game.json')
OUT = os.path.dirname(os.path.abspath(__file__))
R = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
ETA = (1, 1, -1)

def colset(j):
    return [t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == ETA[j]]
C = [colset(0), colset(1), colset(2)]
c_tr = [(1, 1, 1), (1, -1, -1)]
r_tr = [(1, 1, 1), (1, -1, -1)]

def answers_alice(q):
    if q in ('R1', 'R2', 'R3'):
        return [list(t) for t in R] + ['f']
    x = int(q[1])
    return [list(t) for t in C[0] if t != c_tr[x]] + [1, -1]

def answers_bob(r):
    if r in ('C1', 'C2', 'C3'):
        j = int(r[1]) - 1
        return [list(t) for t in C[j]] + ['f']
    y = int(r[1])
    return [list(t) for t in R if t != r_tr[y]] + [1, -1]
AQ = ['R1', 'R2', 'R3', 'T0', 'T1']
BQ = ['C1', 'C2', 'C3', 'U0', 'U1']

def fms_win(q, r, a, b):
    if q[0] == 'R' and r[0] == 'C':
        i, j = (int(q[1]) - 1, int(r[1]) - 1)
        if a == 'f' and b == 'f':
            return True
        if (a == 'f') != (b == 'f'):
            return False
        return a[j] == b[i]
    if q[0] == 'T' and r == 'C1':
        x = int(q[1])
        if isinstance(a, int):
            return b == 'f' or tuple(b) == c_tr[x]
        return b != 'f' and tuple(a) == tuple(b)
    if q == 'R1' and r[0] == 'U':
        y = int(r[1])
        if isinstance(b, int):
            return a == 'f' or tuple(a) == r_tr[y]
        return a != 'f' and tuple(a) == tuple(b)
    if q[0] == 'T' and r[0] == 'U':
        x, y = (int(q[1]), int(r[1]))
        if isinstance(a, int) and isinstance(b, int):
            return a * b == (-1) ** (x * y)
        return True
    return True

def check_C1():
    d = json.load(open(GAME_JSON))
    ok = True
    n = 0
    for p in d['supported_pairs']:
        if not (p['alice'].startswith('1:') and p['bob'].startswith('1:')):
            continue
        q, r = (p['alice'][2:], p['bob'][2:])
        A, B = (answers_alice(q), answers_bob(r))
        assert A == d['alice']['1:' + q] and B == d['bob']['1:' + r], (q, r)
        mine = sorted([[ia, ib] for ia, a in enumerate(A) for ib, b in enumerate(B) if not fms_win(q, r, a, b)])
        ok &= mine == sorted(p['forbidden'])
        n += 1
    return (ok, n)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
Y = np.array([[0, -1], [1, 0]])
YY = np.kron(Y, Y) * -1
SQ = [[np.kron(X, I2), np.kron(I2, X), np.kron(X, X)], [np.kron(I2, Z), np.kron(Z, I2), np.kron(Z, Z)], [np.kron(X, Z), np.kron(Z, X), YY]]

def joint_proj(ops, s):
    P = np.eye(4)
    for O, v in zip(ops, s):
        P = P @ (np.eye(4) + v * O) / 2
    return P

def embed5(P4=None, flag=0.0):
    M = np.zeros((5, 5))
    M[0, 0] = flag
    if P4 is not None:
        M[1:, 1:] = P4
    return M
c_vec = [np.array([1, 0, 1, 0]) / math.sqrt(2), np.array([0, 1, 0, 1]) / math.sqrt(2)]
r_vec = [np.array([1, 1, 1, 1]) / 2, np.array([1, -1, 1, -1]) / 2]

def sign_proj(v4, s):
    v = np.concatenate([[1.0], s * v4]) / math.sqrt(2)
    return np.outer(v, v)

def alice_pvm(q):
    out = []
    for a in answers_alice(q):
        if q[0] == 'R':
            i = int(q[1]) - 1
            out.append(embed5(None, 1.0) if a == 'f' else embed5(joint_proj(SQ[i], a)))
        else:
            x = int(q[1])
            if isinstance(a, int):
                out.append(sign_proj(c_vec[x], a))
            else:
                out.append(embed5(joint_proj([SQ[0][0], SQ[1][0], SQ[2][0]], a)))
    return out

def bob_pvm(r):
    out = []
    for b in answers_bob(r):
        if r[0] == 'C':
            j = int(r[1]) - 1
            out.append(embed5(None, 1.0) if b == 'f' else embed5(joint_proj([SQ[0][j], SQ[1][j], SQ[2][j]], b)))
        else:
            y = int(r[1])
            if isinstance(b, int):
                out.append(sign_proj(r_vec[y], b))
            else:
                out.append(embed5(joint_proj(SQ[0], b)))
    return out

def check_C2():
    Cm = np.diag([1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]) / 3
    worst = 0.0
    proj_err = 0.0
    for q in AQ:
        P = alice_pvm(q)
        proj_err = max(proj_err, np.abs(sum(P) - np.eye(5)).max(), max((np.abs(p @ p - p).max() for p in P)))
    for r in BQ:
        Q = bob_pvm(r)
        proj_err = max(proj_err, np.abs(sum(Q) - np.eye(5)).max(), max((np.abs(p @ p - p).max() for p in Q)))
    for q in AQ:
        for r in BQ:
            for ia, a in enumerate(answers_alice(q)):
                for ib, b in enumerate(answers_bob(r)):
                    if not fms_win(q, r, a, b):
                        p = np.trace(Cm @ alice_pvm(q)[ia] @ Cm @ bob_pvm(r)[ib].T)
                        worst = max(worst, abs(p))
    corner = alice_pvm('R2')[0]
    return (worst, proj_err, float(np.abs(corner - embed5(np.diag([1, 0, 0, 0]))).max()))
NA = 9

def corner_pairs(j):
    return [((j + 1) % 9, (j + 2) % 9), ((j + 3) % 9, (j + 4) % 9), ((j + 5) % 9, (j + 6) % 9), ((j + 7) % 9, (j + 8) % 9)]

def giii_pairs():
    pairs = []
    for j in range(NA):
        for q in AQ:
            for r in BQ:
                pairs.append(('copy', j, q, r))
    for j in range(NA):
        pairs.append(('link1', j, 'R1', 'Lab'))
        pairs.append(('link2', j, 'R2', 'Lab'))
    return pairs

def giii_win(pair, a, b):
    kind, j, q, r = pair
    if kind == 'copy':
        return fms_win(q, r, a, b)
    if kind == 'link1':
        return (a == 'f') == (b == j)
    if kind == 'link2':
        return (a == [1, 1, 1]) == (b in corner_pairs(j)[0])
    raise ValueError

def check_C5():
    ok = True
    for j in range(NA):
        used = [j] + [x for pr in corner_pairs(j) for x in pr]
        ok &= sorted(used) == list(range(9))
    cover = [0] * 9
    for j in range(NA):
        for x in corner_pairs(j)[0]:
            cover[x] += 1
    ok &= cover == [2] * 9
    A = np.zeros((9, 9))
    for j in range(9):
        for x in corner_pairs(j)[0]:
            A[x, j] = 1
    ev = max(abs(np.linalg.eigvals(A)))
    Ak = np.eye(9)
    prim = None
    for k in range(1, 40):
        Ak = Ak @ A
        if (Ak > 0).all():
            prim = k
            break
    return (ok, cover, float(ev), prim)

def words(L):
    W = [()]
    for l in range(1, L + 1):
        W += list(itertools.product([1, 2], repeat=l))
    return W

def classical_strategy(kind, R2):
    if kind == 'flag':
        al = {'R1': 'f', 'R2': 'f', 'R3': 'f', 'T0': 1, 'T1': 1}
        bo = {'C1': 'f', 'C2': 'f', 'C3': 'f', 'U0': 1, 'U1': 1}
        return (al, bo)
    if R2 == (1, 1, 1):
        rows = [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
        cols = [(1, 1, 1), (1, 1, 1), (1, 1, -1)]
    else:
        rows = [(1, 1, 1), (1, -1, -1), (1, -1, -1)]
        cols = [(1, 1, 1), (1, -1, -1), (1, -1, 1)]
    al = {'R1': list(rows[0]), 'R2': list(rows[1]), 'R3': list(rows[2])}
    bo = {'C1': list(cols[0]), 'C2': list(cols[1]), 'C3': list(cols[2])}
    al['T0'] = 1
    al['T1'] = list(cols[0])
    bo['U0'] = 1
    bo['U1'] = list(rows[0])
    return (al, bo)

def check_C4(L):
    Wd = words(L)
    idx = {w: k for k, w in enumerate(Wd)}
    N = len(Wd)
    lev = np.array([len(w) for w in Wd])
    rho = np.array([2.0 ** (-l) for l in lev]) / (L + 1)
    t = []
    for i in (1, 2):
        T = np.zeros((N, N))
        for w in Wd:
            if len(w) < L:
                T[idx[(i,) + w], idx[w]] = 1.0
        t.append(T)
    P0 = np.zeros((N, N))
    P0[idx[()], idx[()]] = 1.0
    PL = np.diag((lev == L).astype(float))
    eig_err = max((np.abs(np.diag(rho) @ ti.T - 2 * ti.T @ np.diag(rho)).max() for ti in t))
    D = NA * N
    h = np.kron(np.eye(NA) / NA, np.diag(rho))
    K = np.diag(np.sqrt(np.diag(h)))

    def e(a, b):
        m = np.zeros((NA, NA))
        m[a, b] = 1.0
        return m
    I_N = np.eye(N)
    total_loss = 0.0
    comp_err = 0.0
    block_err = 0.0
    lab = [np.kron(e(a, a), I_N) for a in range(NA)]
    comp_err = max(comp_err, np.abs(sum(lab) - np.eye(D)).max())

    def prob(A, B):
        return float(np.trace(K @ A @ K @ B.T))
    npairs = len(giii_pairs())
    w = 1.0 / npairs
    per_kind = {'copy': 0.0, 'link1': 0.0, 'link2': 0.0}
    for j in range(NA):
        alice_eff = {}
        bob_eff = {}
        cps = corner_pairs(j)
        Wk = [np.kron(e(a, j), t[0].T) + np.kron(e(b, j), t[1].T) for a, b in cps]
        f_t = np.kron(e(j, j), I_N - P0)
        Emat = [[None] * 5 for _ in range(5)]
        Emat[0][0] = f_t
        for k in range(4):
            Emat[k + 1][0] = Wk[k]
            Emat[0][k + 1] = Wk[k].T
            for l in range(4):
                Emat[k + 1][l + 1] = Wk[k] @ Wk[l].T
        for k in range(4):
            block_err = max(block_err, np.abs(Wk[k].T @ Wk[k] - f_t).max(), np.abs(h @ Wk[k] - 2 * Wk[k] @ h).max())
        blockproj = sum((Emat[a][a] for a in range(5)))
        defect = np.eye(D) - blockproj
        pieces = [('flag', None, np.kron(e(j, j), P0))]
        for a in range(NA):
            if a == j:
                continue
            R2 = (1, 1, 1) if a in cps[0] else (1, -1, -1)
            pieces.append(('magic', R2, np.kron(e(a, a), PL)))
        block_err = max(block_err, np.abs(sum((p[2] for p in pieces)) - defect).max())
        for q in AQ:
            P = alice_pvm(q)
            effs = []
            for ia, ans in enumerate(answers_alice(q)):
                Mx = sum((P[ia][a, b] * Emat[a][b] for a in range(5) for b in range(5)))
                for kind, R2, Pi in pieces:
                    if classical_strategy(kind, R2)[0][q] == ans:
                        Mx = Mx + Pi
                effs.append(Mx)
            comp_err = max(comp_err, np.abs(sum(effs) - np.eye(D)).max())
            alice_eff[j, q] = effs
        for r in BQ:
            Q = bob_pvm(r)
            effs = []
            for ib, ans in enumerate(answers_bob(r)):
                Mx = sum((Q[ib][a, b] * Emat[a][b] for a in range(5) for b in range(5)))
                for kind, R2, Pi in pieces:
                    if classical_strategy(kind, R2)[1][r] == ans:
                        Mx = Mx + Pi
                effs.append(Mx)
            comp_err = max(comp_err, np.abs(sum(effs) - np.eye(D)).max())
            bob_eff[j, r] = effs
        for pr in [p for p in giii_pairs() if p[1] == j]:
            kind, _, q, r = pr
            A_ans = answers_alice(q)
            Aeff = alice_eff[j, q]
            if kind == 'copy':
                B_ans = answers_bob(r)
                Beff = bob_eff[j, r]
            else:
                B_ans = list(range(NA))
                Beff = lab
            lp = 0.0
            for ia, a in enumerate(A_ans):
                for ib, b in enumerate(B_ans):
                    if not giii_win(pr, a, b):
                        lp += prob(Aeff[ia], Beff[ib])
            per_kind[kind] += w * lp
            total_loss += w * lp
        del alice_eff, bob_eff, Emat, Wk
    ent = -float(np.sum(np.diag(h) * np.log(np.diag(h))))
    pred_loss = 1.0 / (27 * (L + 1))
    pred_ent = math.log(9) + math.log(L + 1) + L * math.log(2) / 2
    return dict(L=L, dim=D, loss=total_loss, predicted_loss=pred_loss, loss_by_kind=per_kind, entropy_nats=ent, predicted_entropy=pred_ent, eig_err=eig_err, block_err=block_err, completeness_err=comp_err)

def check_C6():
    L = 6
    A = np.zeros((L + 2, L + 1))
    for k in range(L):
        A[k, k] = 1
        A[k, k + 1] = -1
    A[L, 0] = 1
    A[L + 1, L] = 1
    rank = np.linalg.matrix_rank(A)
    return (int(rank), L + 1)
if __name__ == '__main__':
    res = {}
    ok1, n1 = check_C1()
    res['C1_fms_predicate_matches_enumerated'] = ok1
    res['C1_pairs'] = n1
    print('C1 FMS predicate vs game.json copy one:', ok1, 'pairs:', n1, flush=True)
    worst, perr, cerr = check_C2()
    res['C2_ideal_max_forbidden_prob'] = worst
    res['C2_pvm_err'] = perr
    res['C2_R2plus_is_00_corner_err'] = cerr
    print('C2 ideal FMS max forbidden prob:', worst, ' PVM err:', perr, ' R2(+++)=|00> corner err:', cerr, flush=True)
    prs = giii_pairs()
    res['C3_pairs'] = len(prs)
    print('C3 G_III supported pairs:', len(prs), flush=True)
    ok5, cover, ev, prim = check_C5()
    res['C5'] = dict(ok=ok5, cover=cover, perron=ev, primitive_power=prim)
    print('C5 atom design ok:', ok5, 'cover:', cover, 'Perron:', ev, 'primitive at power:', prim, flush=True)
    res['C4'] = []
    for L in range(1, 5):
        r = check_C4(L)
        res['C4'].append(r)
        print('C4', r, flush=True)
    rk, nv = check_C6()
    res['C6_rank'] = rk
    res['C6_unknowns'] = nv
    print('C6 halving-invariance system rank', rk, 'of', nv, 'unknowns (full rank => only zero solution)', flush=True)
    if not (ok1 and n1==25 and len(prs)==243 and ok5 and rk==nv): raise RuntimeError('Finite predicate or graph check failed')
    if max(abs(worst),perr,cerr)>1e-11: raise RuntimeError('Ideal strategy check failed')
    for row in res['C4']:
        if max(abs(row['loss']-row['predicted_loss']),abs(row['entropy_nats']-row['predicted_entropy']),row['eig_err'],row['block_err'],row['completeness_err'])>1e-10: raise RuntimeError('Finite-shadow identity failed')
    json.dump(res, open(os.path.join(OUT, 'checks_results.json'), 'w'), indent=1)
