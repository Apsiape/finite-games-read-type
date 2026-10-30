if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
import itertools as it
import json, math, hashlib, time
from pathlib import Path
import numpy as np
import sympy as s
from scipy import sparse as z
import psutil
OUT = Path(__file__).resolve().parent
INPUT = OUT
SQ2 = s.sqrt(2)
I = s.eye(2)
X = s.Matrix([[0, 1], [1, 0]])
Z = s.diag(1, -1)
Y = s.Matrix([[0, -s.I], [s.I, 0]])
K = s.kronecker_product
O = [[K(X, I), K(I, X), K(X, X)], [K(I, Z), K(Z, I), K(Z, Z)], [K(X, Z), K(Z, X), K(Y, Y)]]
rows = [v for v in it.product((1, -1), repeat=3) if math.prod(v) == 1]
cols = [[v for v in it.product((1, -1), repeat=3) if math.prod(v) == t] for t in (1, 1, -1)]
AQ = ['R1', 'R2', 'R3', 'T0', 'T1']
BQ = ['C1', 'C2', 'C3', 'U0', 'U1']
dist = [(1, 1, 1), (1, -1, -1)]

def answers(q):
    if q[0] == 'R':
        return rows + ['f']
    if q[0] == 'C':
        return cols[int(q[1]) - 1] + ['f']
    base = cols[0] if q[0] == 'T' else rows
    return [v for v in base if v != dist[int(q[1])]] + [1, -1]

def win(q, r, a, b):
    if q[0] == 'R' and r[0] == 'C':
        return a == b == 'f' if 'f' in (a, b) else a[int(r[1]) - 1] == b[int(q[1]) - 1]
    if q[0] == 'T' and r == 'C1':
        return b in ('f', dist[int(q[1])]) if isinstance(a, int) else a == b
    if q == 'R1' and r[0] == 'U':
        return a in ('f', dist[int(r[1])]) if isinstance(b, int) else a == b
    if q[0] == 'T' and r[0] == 'U' and isinstance(a, int) and isinstance(b, int):
        return a * b == (-1) ** (int(q[1]) * int(r[1]))
    return True

def jp(ops, ans):
    p = s.eye(4)
    for o, a in zip(ops, ans):
        p = p * (s.eye(4) + a * o) / 2
    return p
cv = [s.Matrix([1, 0, 1, 0]) / SQ2, s.Matrix([0, 1, 0, 1]) / SQ2]
rv = [s.Matrix([1, 1, 1, 1]) / 2, s.Matrix([1, -1, 1, -1]) / 2]

def ideal(q):
    result = []
    for a in answers(q):
        p = s.zeros(5)
        if a == 'f':
            p[0, 0] = 1
        elif isinstance(a, int):
            v = s.Matrix([1] + list(a * (cv if q[0] == 'T' else rv)[int(q[1])])) / SQ2
            p = v * v.T
        else:
            ops = O[int(q[1]) - 1] if q[0] == 'R' else [O[i][int(q[1]) - 1] for i in range(3)] if q[0] == 'C' else [O[i][0] for i in range(3)] if q[0] == 'T' else O[0]
            p[1:, 1:] = jp(ops, a)
        result.append(p)
    assert sum(result, s.zeros(5)) == s.eye(5)
    assert all((p * p == p and p.T == p for p in result))
    return result
IDEAL = {q: ideal(q) for q in AQ + BQ}

def pairs(j):
    return [((j + 2 * k + 1) % 9, (j + 2 * k + 2) % 9) for k in range(4)]

def classical(j, atom):
    if atom == j:
        return (dict(zip(AQ, ['f', 'f', 'f', 1, 1])), dict(zip(BQ, ['f', 'f', 'f', 1, 1])))
    yes = atom in pairs(j)[0]
    ra = [(1, 1, 1)] * 3 if yes else [(1, 1, 1), (1, -1, -1), (1, -1, -1)]
    cb = [(1, 1, 1), (1, 1, 1), (1, 1, -1)] if yes else [(1, 1, 1), (1, -1, -1), (1, -1, 1)]
    return (dict(zip(AQ, ra + [1, cb[0]])), dict(zip(BQ, cb + [1, ra[0]])))

def poly(j, p):
    d = {}

    def add(a, b, l, c):
        key = (a, b, l)
        d[key] = s.expand(d.get(key, 0) + c)
    add(j, j, 0, p[0, 0])
    for k, (a, b) in enumerate(pairs(j), 1):
        for n, atom in enumerate((a, b), 1):
            add(atom, j, -n, p[k, 0])
            add(j, atom, n, p[0, k])
        for h, (c, e) in enumerate(pairs(j), 1):
            add(a, c, 0, p[k, h])
            add(b, e, 0, p[k, h])
    return {key: v for key, v in d.items() if v != 0}

def kms_prob(a, b):
    v = s.Integer(0)
    for (i, j, l), c in a.items():
        d = b.get((j, i, -l), 0)
        if d != 0:
            v += c * d * (1 if l == 0 else 1 / SQ2)
    return s.expand(v / 9)

def finite(L, c):
    words = [w for n in range(L + 1) for w in it.product((0, 1), repeat=n)]
    idx = {w: i for i, w in enumerate(words)}
    N = len(words)
    D = 9 * N
    lev = np.array([len(w) for w in words])
    rho = np.power(2.0, -lev) * np.asarray(c)[lev] ** 2
    h = np.tile(rho / 9, 9)
    root = z.diags(np.sqrt(h), format='csr')

    def atom(a, b):
        return z.csr_matrix(([1.0], ([a], [b])), shape=(9, 9))
    shifts = []
    for digit in (0, 1):
        ij = [(idx[(digit,) + w], idx[w]) for w in words if len(w) < L]
        shifts.append(z.csr_matrix((np.ones(len(ij)), tuple(zip(*ij))), shape=(N, N)))
    eye = z.eye(N, format='csr')
    pe = z.diags([float(n == 0) for n in lev])
    pl = z.diags([float(n == L) for n in lev])
    lab = [z.kron(atom(a, a), eye, format='csr') for a in range(9)]
    all_eff = {}
    for j in range(9):
        w = [z.kron(atom(a, j), shifts[0].T, format='csr') + z.kron(atom(b, j), shifts[1].T, format='csr') for a, b in pairs(j)]
        e = [[None] * 5 for _ in range(5)]
        e[0][0] = z.kron(atom(j, j), eye - pe, format='csr')
        for k in range(4):
            e[k + 1][0] = w[k]
            e[0][k + 1] = w[k].T
            for n in range(4):
                e[k + 1][n + 1] = w[k] @ w[n].T
        defects = [z.kron(atom(a, a), pe if a == j else pl, format='csr') for a in range(9)]
        for q in AQ + BQ:
            vals = []
            for ans, p in zip(answers(q), IDEAL[q]):
                mat = z.csr_matrix((D, D))
                for a in range(5):
                    for b in range(5):
                        if p[a, b] != 0:
                            mat += float(p[a, b]) * e[a][b]
                for a in range(9):
                    ca, cb = classical(j, a)
                    if (ca if q in AQ else cb)[q] == ans:
                        mat += defects[a]
                vals.append(mat)
            assert abs((sum(vals) - z.eye(D)).data).max(initial=0) < 1e-12
            assert max((abs((v @ v - v).data).max(initial=0) for v in vals)) < 1e-12
            all_eff[j, q] = vals

    def probability(a, b):
        t = a @ root @ b
        return float(t.multiply(t.conjugate()).sum())
    return (all_eff, lab, probability, h, D)

def main():
    start = time.monotonic()
    res = {}
    enumeratedd = json.loads((INPUT / 'base-game.json').read_text())
    matches = 0
    for pair in enumeratedd['supported_pairs']:
        if not (pair['alice'].startswith('1:') and pair['bob'].startswith('1:')):
            continue
        q, r = (pair['alice'][2:], pair['bob'][2:])
        bad = [[i, j] for i, a in enumerate(answers(q)) for j, b in enumerate(answers(r)) if not win(q, r, a, b)]
        assert bad == sorted(pair['forbidden'])
        matches += 1
    assert matches == 25
    a, b = s.symbols('a b', real=True)
    d = s.diag(a, b, b, b, b)
    bad_sum = 0
    for q in AQ:
        for r in BQ:
            for ia, aa in enumerate(answers(q)):
                for ib, bb in enumerate(answers(r)):
                    if not win(q, r, aa, bb):
                        bad_sum += s.trace(d * IDEAL[q][ia] * d * IDEAL[r][ib])
    assert s.expand(bad_sum - (SQ2 * a - b) ** 2) == 0
    res['exact_unweighted_fms_loss'] = str(s.expand(bad_sum))
    res['forced_flag_to_corner_density_ratio'] = 'b^2/a^2=2'
    ap = {(j, q): [poly(j, p) for p in IDEAL[q]] for j in range(9) for q in AQ}
    bp = {(j, q): [poly(j, p) for p in IDEAL[q]] for j in range(9) for q in BQ}
    labs = [{(a, a, 0): s.Integer(1)} for a in range(9)]
    game = {'alice': {}, 'bob': {'Lab': list(range(9))}, 'supported_pairs': []}
    exact = []
    full_count = 0
    minprob = 1.0
    full_numeric = []
    for (j, q), av in ap.items():
        game['alice'][f'{j}:{q}'] = answers(q)
        numeric_rows = [[] for _ in av]
        for (k, r), bv in list(bp.items()) + [(('Lab', 'Lab'), labs)]:
            probs = [[kms_prob(p, t) for t in bv] for p in av]
            assert all((v.is_nonnegative is True for row in probs for v in row))
            for i, row in enumerate(probs):
                numeric_rows[i].extend((float(v) for v in row))
            assert s.expand(sum(map(sum, probs)) - 1) == 0
            minprob = min(minprob, min((float(v) for row in probs for v in row)))
            full_count += 1
            support = j == k or (k == 'Lab' and q in ('R1', 'R2'))
            if not support:
                continue
            an, bn = (f'{j}:{q}', f'{k}:{r}' if k != 'Lab' else 'Lab')
            bans = answers(r) if k != 'Lab' else list(range(9))

            def accepted(aa, bb):
                if k != 'Lab':
                    return win(q, r, aa, bb)
                return (aa == 'f') == (bb == j) if q == 'R1' else (aa == (1, 1, 1)) == (bb in pairs(j)[0])
            forbidden = [[ia, ib] for ia, aa in enumerate(answers(q)) for ib, bb in enumerate(bans) if not accepted(aa, bb)]
            assert all((probs[x][y] == 0 for x, y in forbidden)), (an, bn, [(x, y, probs[x][y]) for x, y in forbidden if probs[x][y] != 0])
            game['supported_pairs'].append({'alice': an, 'bob': bn, 'weight': '1/243', 'forbidden': forbidden})
            exact.append({'alice': an, 'bob': bn, 'probabilities': [[str(v) for v in row] for row in probs]})
        full_numeric.extend(numeric_rows)
    for j in range(9):
        for r in BQ:
            game['bob'][f'{j}:{r}'] = answers(r)
    assert len(game['supported_pairs']) == 243 and full_count == 2070 and (minprob >= 0)
    (OUT / 'giii-game.json').write_text(json.dumps(game, indent=1))
    (OUT / 'exact-supported-correlations.json').write_text(json.dumps(exact, indent=1))
    res.update(predicate_matches_pairs=matches, full_kms_pairs_checked=full_count, supported_pairs=243, full_kms_min_probability=minprob)
    print('Exact polynomial KMS correlations: all 2070 pairs normalized/nonnegative; all 243 supported pairs perfect.', flush=True)
    res['finite'] = []
    full_limit = np.array(full_numeric)
    scaled_previous = None
    for L, kind in [(1, 'uniform'), (3, 'uniform'), (5, 'uniform'), (2, 'sine'), (3, 'sine'), (5, 'sine')]:
        if kind == 'uniform':
            c = np.ones(L + 1) / math.sqrt(L + 1)
        else:
            c = math.sqrt(2 / L) * np.sin(np.pi * np.arange(L + 1) / L)
            c[0] = c[-1] = 0.0
        eff, lab, prob, h, D = finite(L, c)
        full_error = None
        affine_error = None
        if kind == 'uniform':

            def flatten(mats, weighted):
                rs = []
                cs = []
                vs = []
                for row, mat in enumerate(mats):
                    co = mat.tocoo()
                    rs.extend([row] * len(co.data))
                    cs.extend(co.row * D + co.col)
                    vv = co.data * np.sqrt(h[co.row] * h[co.col]) if weighted else co.data
                    vs.extend(vv)
                return z.csr_matrix((vs, (rs, cs)), shape=(len(mats), D * D))
            all_a = [v for j in range(9) for q in AQ for v in eff[j, q]]
            all_b = [v for j in range(9) for r in BQ for v in eff[j, r]] + lab
            full = (flatten(all_a, True) @ flatten(all_b, False).T).toarray()
            delta = full - full_limit
            full_error = float(np.abs(delta).max())
            scaled = (L + 1) * delta
            if scaled_previous is not None:
                affine_error = float(np.abs(scaled - scaled_previous).max())
                assert affine_error < 1e-12
            scaled_previous = scaled
        per = []
        mixerr = 0.0
        normalerr = 0.0
        for q in AQ:
            for r in BQ + (['Lab'] if q in ('R1', 'R2') else []):
                av = eff[0, q]
                bv = eff[0, r] if r != 'Lab' else lab
                bs = answers(r) if r != 'Lab' else list(range(9))
                loss = 0.0
                total = 0.0
                for ia, aa in enumerate(answers(q)):
                    for ib, bb in enumerate(bs):
                        p = prob(av[ia], bv[ib])
                        total += p
                        good = win(q, r, aa, bb) if r != 'Lab' else (aa == 'f') == (bb == 0) if q == 'R1' else (aa == (1, 1, 1)) == (bb in pairs(0)[0])
                        if not good:
                            loss += p
                        if kind == 'uniform':
                            pi = float(kms_prob(ap[0, q][ia], bp[0, r][ib] if r != 'Lab' else labs[ib]))
                            cl = sum((int(classical(0, a)[0][q] == aa and (classical(0, a)[1][r] if r != 'Lab' else a) == bb) for a in range(9))) / 9
                            mixerr = max(mixerr, abs(p - (L * pi + cl) / (L + 1)))
                normalerr = max(normalerr, abs(total - 1))
                per.append((q, r, loss))
        loss = sum((v for q, r, v in per)) / 27
        predicted = (2 * np.sum(np.diff(c) ** 2) + c[0] ** 2 + 8 * c[-1] ** 2) / 243
        ent = -float(sum((t * math.log(t) for t in h if t > 0)))
        level_entropy = -float(sum((t * t * math.log(t * t) for t in c if t > 0)))
        ep = math.log(9) + level_entropy + float(np.dot(np.arange(L + 1), c * c)) * math.log(2)
        assert abs(loss - predicted) < 1e-12 and abs(ent - ep) < 1e-12 and (normalerr < 1e-12) and (mixerr < 1e-12)
        assert all((v < 1e-12 for q, r, v in per if r == 'Lab'))
        if kind == 'sine':
            assert abs(predicted - 4 * (1 - math.cos(math.pi / L)) / 243) < 1e-12
        row = dict(L=L, kind=kind, dimension=D, loss=loss, predicted_loss=float(predicted), entropy_nats=ent, entropy_error=abs(ent - ep), normalization_error=normalerr, uniform_mixture_error=mixerr, links_lossless=True, full_52650_event_max_limit_error=full_error, full_affine_scaling_error=affine_error)
        res['finite'].append(row)
        print(row, flush=True)
    res['sine_scalar_large_L'] = []
    for L in (10, 100, 1000, 10000):
        c = math.sqrt(2 / L) * np.sin(np.pi * np.arange(1, L) / L)
        H = -float(np.sum(c * c * np.log(c * c)))
        eps = 8 * math.sin(math.pi / (2 * L)) ** 2 / 243
        ent = math.log(9) + H + L * math.log(2) / 2
        res['sine_scalar_large_L'].append(dict(L=L, loss=eps, entropy_nats=ent, loss_times_entropy_squared=eps * ent ** 2, loss_times_entropy=eps * ent))
    res['peak_working_set_bytes'] = getattr(psutil.Process().memory_info(), 'peak_wset', None)
    res['elapsed_seconds'] = time.monotonic() - start
    res['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [INPUT / 'base-game.json']}
    (OUT / 'independent-results.json').write_text(json.dumps(res, indent=2))
    print('PASS; peak working set', res['peak_working_set_bytes'], 'elapsed', res['elapsed_seconds'], flush=True)
if __name__ == '__main__':
    main()
