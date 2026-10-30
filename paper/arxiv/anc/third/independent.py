if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from itertools import product
import json, time, hashlib, ctypes
import sympy as s
START = time.perf_counter()
ROOT = Path(__file__).resolve().parent
OUT = ROOT
Z = s.zeros(4)
I = s.eye(2)
X = s.Matrix([[0, 1], [1, 0]])
Y = s.Matrix([[0, -s.I], [s.I, 0]])
Z2 = s.diag(1, -1)
K = s.kronecker_product
O = [[K(X, I), K(I, X), K(X, X)], [K(I, Z2), K(Z2, I), K(Z2, Z2)], [K(X, Z2), K(Z2, X), K(Y, Y)]]
for i in range(3):
    assert O[i][0] * O[i][1] * O[i][2] == s.eye(4)
    for j in range(3):
        assert O[i][j] ** 2 == s.eye(4)
for j in range(3):
    assert O[0][j] * O[1][j] * O[2][j] == (1 if j < 2 else -1) * s.eye(4)
for row in O + [[O[i][j] for i in range(3)] for j in range(3)]:
    for a, b in product(row, repeat=2):
        assert a * b == b * a
a, b, c, d = (O[0][0], O[0][1], O[1][0], O[1][1])
assert a * d + d * a == s.zeros(4) and b * c + c * b == s.zeros(4)
assert all((u * v == v * u for u, v in ((a, b), (a, c), (d, b), (d, c))))

def proj(v):
    return v * v.T / (v.T * v)[0]
base = {}
for letter in ('R', 'C'):
    for k in range(3):
        obs = O[k] if letter == 'R' else [O[i][k] for i in range(3)]
        eta = -1 if letter == 'C' and k == 2 else 1
        effects = {}
        for signs in product((1, -1), repeat=3):
            if signs[0] * signs[1] * signs[2] != eta:
                continue
            system = s.Matrix.vstack(*[ob - sign * s.eye(4) for ob, sign in zip(obs, signs)])
            kernel = system.nullspace()
            assert len(kernel) == 1
            effects[''.join(('+' if z == 1 else '-' for z in signs))] = proj(kernel[0])
        base[letter + str(k + 1)] = effects
vectors = [s.Matrix(v) for v in ((1, 1, -1, 0), (1, -1, 0, 1), (0, 1, 1, 1), (1, 0, 1, -1))]
V = s.Matrix.hstack(*vectors)
assert V.T * V == 3 * s.eye(4)
base['H'] = {str(j): proj(v) for j, v in enumerate(vectors)}
coords = [next((lab for lab, P in base['R2'].items() if P[k, k] == 1)) for k in range(4)]
assert coords == ['+++', '-+-', '+--', '--+']
flag = s.diag(1, 0, 0, 0, 0)

def lift(P):
    return s.diag(s.zeros(1), P)
A = {q: {'f': flag, **{lab: lift(P) for lab, P in es.items()}} for q, es in base.items()}
B = {q: dict(es) for q, es in A.items()}
for x in range(2):
    es = {}
    for sign, label in ((1, '+'), (-1, '-')):
        v = s.zeros(5, 1)
        v[0] = 1
        v[x + 1] = sign
        es[label] = proj(v)
    es.update({'k' + str(j): lift(base['R2'][coords[j]]) for j in range(4) if j != x})
    A['T' + str(x)] = es
for y in range(2):
    es = {}
    for sign, label in ((1, '+'), (-1, '-')):
        es[label] = proj(s.Matrix([1, *[sign * z / s.sqrt(3) for z in vectors[y]]]))
    es.update({'h' + str(j): lift(base['H'][str(j)]) for j in range(4) if j != y})
    B['U' + str(y)] = es
for family in (A, B):
    for es in family.values():
        assert sum(es.values(), s.zeros(5)) == s.eye(5)
        for la, P in es.items():
            assert s.simplify(P * P - P) == s.zeros(5) and P.T == P
            for lb, Q in es.items():
                if la != lb:
                    assert s.simplify(P * Q) == s.zeros(5)

def wins(q, r, la, lb):
    if q in base and r in base:
        if la == 'f' or lb == 'f':
            return la == lb
        return s.trace(base[q][la] * base[r][lb]) > 0
    if q in ('T0', 'T1') and r == 'R2':
        return lb in ('f', coords[int(q[-1])]) if la in ('+', '-') else lb == coords[int(la[-1])]
    if q == 'H' and r in ('U0', 'U1'):
        return la in ('f', r[-1]) if lb in ('+', '-') else la == lb[-1]
    if q in ('T0', 'T1') and r in ('U0', 'U1') and (la in ('+', '-')) and (lb in ('+', '-')):
        return (la == lb) == (q != 'T1' or r != 'U1')
    return True
given = json.loads((ROOT / 'g3-game.json').read_text())
given_pairs = {(p['alice'], p['bob']): p for p in given['pairs']}
assert given['alice_questions'] == list(A) and given['bob_questions'] == list(B)
assert given['alice_answers'] == {q: list(es) for q, es in A.items()}
assert given['bob_answers'] == {q: list(es) for q, es in B.items()}
assert len(given['pairs']) == len(given_pairs) == 81
D = s.diag(1, *[s.sqrt(3)] * 4)
all_probabilities = []
cells = []
loss = 0
forbidden = 0
for q, es in A.items():
    for r, fs in B.items():
        total = 0
        rejected = []
        probabilities = []
        for la, P in es.items():
            for lb, Q in fs.items():
                p = s.expand(s.trace(P * D * Q.T * D) / 13)
                assert p.is_nonnegative is True, (q, r, la, lb, p)
                total += p
                probabilities.append({'alice': la, 'bob': lb, 'probability': str(p), 'wins': bool(wins(q, r, la, lb))})
                if not wins(q, r, la, lb):
                    rejected.append([la, lb])
                    forbidden += 1
                    loss += p
                    assert p == 0, (q, r, la, lb, p)
        assert s.expand(total) == 1
        assert {tuple(v) for v in rejected} == {tuple(v) for v in given_pairs[q, r]['forbidden']}
        assert given_pairs[q, r]['weight'] == '1/81'
        cells.append({'alice': q, 'bob': r, 'forbidden': len(rejected), 'exact_total': '1', 'exact_loss': '0'})
        all_probabilities.append({'alice': q, 'bob': r, 'events': probabilities})
assert loss == 0 and forbidden == 732
rigidity = []
for j, v in enumerate(vectors):
    orth = [(q, label, P) for q, es in base.items() if q != 'H' for label, P in es.items() if P * v == s.zeros(4, 1)]
    S = sum((P for q, label, P in orth), s.zeros(4))
    assert S.rank() == 3 and S * v == s.zeros(4, 1)
    assert proj(S.nullspace()[0]) == base['H'][str(j)]
    rigidity.append({'ray': j, 'orthogonal_old_answers': [(q, label) for q, label, P in orth], 'sum_rank_exact': S.rank(), 'null_projector_equals_conference': True})
for i in range(4):
    for j in range(i + 1, 4):
        for sign in (1, -1):
            v = s.zeros(4, 1)
            v[i] = 1
            v[j] = sign
            assert any((P == proj(v) for q in ('C1', 'C2', 'C3') for P in base[q].values()))
for x, y in product(range(2), repeat=2):
    assert vectors[y][x] == (-1) ** (x * y)
variables = [('A', q) for q in A] + [('B', r) for r in B]
domains = {('A', q): set(es) for q, es in A.items()} | {('B', q): set(es) for q, es in B.items()}
allowed = {(q, r): {(la, lb) for la in A[q] for lb in B[r] if wins(q, r, la, lb)} for q in A for r in B}
nodes = 0

def solve(dom):
    global nodes
    nodes += 1
    changed = True
    while changed:
        changed = False
        for q, r in allowed:
            da, db = (dom['A', q], dom['B', r])
            na = {a for a in da if any(((a, b) in allowed[q, r] for b in db))}
            nb = {b for b in db if any(((a, b) in allowed[q, r] for a in da))}
            if not na or not nb:
                return False
            if na != da:
                dom['A', q] = na
                changed = True
            if nb != db:
                dom['B', r] = nb
                changed = True
    if all((len(d) == 1 for d in dom.values())):
        return True
    var = min((v for v in dom if len(dom[v]) > 1), key=lambda v: len(dom[v]))
    return any((solve(dom | {var: {a}}) for a in sorted(dom[var])))
classical = solve(domains)
assert not classical
from ctypes import wintypes

class PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
pmc = PMC()
pmc.cb = ctypes.sizeof(pmc)
ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
peak = pmc.PeakWorkingSetSize / 1024 ** 2
assert peak < 1024
summary = {'status': 'PASS', 'pairs': 81, 'events': 2025, 'forbidden': forbidden, 'exact_loss': '0', 'all_PVMs_exact': True, 'input_predicate_identical_on_all_events': True, 'rigidity': rigidity, 'classical_perfect_strategy': classical, 'classical_search_nodes': nodes, 'peak_working_set_MB': peak, 'seconds': time.perf_counter() - START, 'input_game_sha256': hashlib.sha256((ROOT / 'g3-game.json').read_bytes()).hexdigest()}
(OUT / 'independent-results.json').write_text(json.dumps(summary, indent=2))
(OUT / 'independent-all-probabilities.json').write_text(json.dumps(all_probabilities, indent=2))
(OUT / 'independent-81-cells.json').write_text(json.dumps(cells, indent=2))
print(json.dumps(summary, indent=2), flush=True)
