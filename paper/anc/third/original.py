if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import math
import time
import numpy as np
OUT = Path(__file__).resolve().parent
START = time.perf_counter()
Z0 = (F(0), F(0))
ONE = (F(1), F(0))

def add(x, y):
    return (x[0] + y[0], x[1] + y[1])

def mul(x, y):
    return (x[0] * y[0] + 3 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])

def field_sign(x):
    a, b = x
    if not a:
        return (b > 0) - (b < 0)
    if not b or (a > 0) == (b > 0):
        return (a > 0) - (a < 0)
    if a * a == 3 * b * b:
        return 0
    return (a > 0) - (a < 0) if a * a > 3 * b * b else (b > 0) - (b < 0)

def number(x):
    return float(x[0]) + math.sqrt(3) * float(x[1])

def zero_matrix(n=5):
    return [[Z0 for j in range(n)] for i in range(n)]

def rational_effect(p):
    a = zero_matrix()
    for i in range(4):
        for j in range(4):
            a[i + 1][j + 1] = (F(p[i][j]), F(0))
    return a

def numeric(a):
    return np.array([[number(x) for x in row] for row in a])

def exact_probability(a, b):
    ans = Z0
    for i in range(5):
        for j in range(5):
            weight = ONE if i == j == 0 else (F(0), F(1)) if i == 0 or j == 0 else (F(3), F(0))
            ans = add(ans, mul(mul(a[i][j], b[i][j]), weight))
    return (ans[0] / 13, ans[1] / 13)
I2 = np.eye(2, dtype=int)
X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.diag([1, -1])
YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
square = [[np.kron(X, I2), np.kron(I2, X), np.kron(X, X)], [np.kron(I2, Z), np.kron(Z, I2), np.kron(Z, Z)], [np.kron(X, Z), np.kron(Z, X), YY]]
I4 = np.eye(4, dtype=int)
for row in square:
    assert np.array_equal(row[0] @ row[1] @ row[2], I4)
for j in range(3):
    assert np.array_equal(square[0][j] @ square[1][j] @ square[2][j], (1 if j < 2 else -1) * I4)

def triple_label(s):
    return ''.join(('+' if x == 1 else '-' for x in s))
base = {}
base_labels = {}
for kind in ['R', 'C']:
    for k in range(3):
        q = kind + str(k + 1)
        obs = square[k] if kind == 'R' else [square[i][k] for i in range(3)]
        eta = -1 if kind == 'C' and k == 2 else 1
        base[q] = {}
        for s in product([1, -1], repeat=3):
            if math.prod(s) != eta:
                continue
            numerator = (I4 + s[0] * obs[0]) @ (I4 + s[1] * obs[1])
            p = [[F(int(numerator[i, j]), 4) for j in range(4)] for i in range(4)]
            base[q][triple_label(s)] = p
        base_labels[q] = list(base[q])
V = np.array([[1, 1, -1, 0], [1, -1, 0, 1], [0, 1, 1, 1], [1, 0, 1, -1]], dtype=int)
assert np.array_equal(V @ V.T, 3 * I4)
base['H'] = {str(k): [[F(int(V[k, i] * V[k, j]), 3) for j in range(4)] for i in range(4)] for k in range(4)}
base_labels['H'] = list(base['H'])
BASE_Q = list(base)

def rational_trace_product(p, q):
    return sum((p[i][j] * q[j][i] for i in range(4) for j in range(4)))

def coordinate_label(k):
    return next((a for a, p in base['R2'].items() if p[k][k] == 1))
coordinate_labels = [coordinate_label(k) for k in range(4)]
flag = zero_matrix()
flag[0][0] = ONE
A = {q: {'f': flag, **{a: rational_effect(p) for a, p in ps.items()}} for q, ps in base.items()}
B = {q: dict(es) for q, es in A.items()}
for x in range(2):
    q = 'T' + str(x)
    A[q] = {}
    for sign, name in [(1, '+'), (-1, '-')]:
        p = zero_matrix()
        p[0][0] = (F(1, 2), F(0))
        p[x + 1][x + 1] = (F(1, 2), F(0))
        p[0][x + 1] = p[x + 1][0] = (F(sign, 2), F(0))
        A[q][name] = p
    for k in range(4):
        if k != x:
            A[q]['k' + str(k)] = rational_effect(base['R2'][coordinate_labels[k]])
for y in range(2):
    q = 'U' + str(y)
    B[q] = {}
    for sign, name in [(1, '+'), (-1, '-')]:
        p = zero_matrix()
        p[0][0] = (F(1, 2), F(0))
        for i in range(4):
            p[0][i + 1] = p[i + 1][0] = (F(0), F(sign * int(V[y, i]), 6))
            for j in range(4):
                p[i + 1][j + 1] = (F(int(V[y, i] * V[y, j]), 6), F(0))
        B[q][name] = p
    for k in range(4):
        if k != y:
            B[q]['h' + str(k)] = rational_effect(base['H'][str(k)])

def accept(q, r, a, b):
    if q in base and r in base:
        if a == 'f' or b == 'f':
            return a == b
        return rational_trace_product(base[q][a], base[r][b]) != 0
    if q.startswith('T') and r == 'R2':
        x = int(q[1])
        return b in ['f', coordinate_labels[x]] if a in ['+', '-'] else b == coordinate_labels[int(a[1])]
    if q == 'H' and r.startswith('U'):
        y = int(r[1])
        return a in ['f', str(y)] if b in ['+', '-'] else a == b[1]
    if q.startswith('T') and r.startswith('U') and (a in ['+', '-']) and (b in ['+', '-']):
        return (1 if a == b else -1) == (-1) ** (int(q[1]) * int(r[1]))
    return True
rigidity = []
for k in range(4):
    old = [(q, a, p) for q in BASE_Q if q != 'H' for a, p in base[q].items() if rational_trace_product(base['H'][str(k)], p) == 0]
    total = sum((np.array(p, dtype=float) for q, a, p in old), np.zeros((4, 4)))
    assert np.linalg.matrix_rank(total, tol=1e-10) == 3
    assert np.linalg.norm(total @ V[k]) < 1e-12
    rigidity.append({'ray': k, 'orthogonal_old_answers': [(q, a) for q, a, p in old], 'span_rank': 3})
pvm_error = 0.0
for measurements in [A, B]:
    for q, es in measurements.items():
        mats = [numeric(a) for a in es.values()]
        pvm_error = max(pvm_error, float(np.linalg.norm(sum(mats) - np.eye(5))))
        for i, a in enumerate(mats):
            pvm_error = max(pvm_error, float(np.linalg.norm(a @ a - a)))
            for j, b in enumerate(mats):
                if i != j:
                    pvm_error = max(pvm_error, float(np.linalg.norm(a @ b)))

def embed_o3(p):
    c = np.zeros((7, 13, 13))
    c[0, 0, 0] = p[0, 0]
    for i in range(3):
        for k in range(4):
            ak = 1 + 3 * k + i
            c[1 + i, 0, ak] = p[0, 1 + k]
            c[4 + i, ak, 0] = p[1 + k, 0]
            for ell in range(4):
                al = 1 + 3 * ell + i
                c[0, ak, al] = p[1 + k, 1 + ell]
    return c
EA = {q: {a: embed_o3(numeric(p)) for a, p in es.items()} for q, es in A.items()}
EB = {q: {a: embed_o3(numeric(p)) for a, p in es.items()} for q, es in B.items()}

def o3_probability(a, b):
    ans = np.sum(a[0] * b[0].T)
    for i in range(3):
        ans += (np.sum(a[1 + i] * b[4 + i].T) + np.sum(a[4 + i] * b[1 + i].T)) / math.sqrt(3)
    return float(ans / 13)
events = 0
forbidden = 0
loss = Z0
prob_min = 1.0
o3_error = 0.0
game = {'name': 'Conference-basis flagged magic square G3', 'alice_questions': list(A), 'bob_questions': list(B), 'alice_answers': {q: list(es) for q, es in A.items()}, 'bob_answers': {q: list(es) for q, es in B.items()}, 'distribution': 'uniform on all 81 question pairs', 'pairs': []}
for q, es in A.items():
    for r, fs in B.items():
        total = Z0
        forbidden_pairs = []
        for a, p in es.items():
            for b, s in fs.items():
                events += 1
                prob = exact_probability(p, s)
                assert field_sign(prob) >= 0, (q, r, a, b, prob)
                total = add(total, prob)
                prob_min = min(prob_min, number(prob))
                o3_error = max(o3_error, abs(number(prob) - o3_probability(EA[q][a], EB[r][b])))
                if not accept(q, r, a, b):
                    forbidden += 1
                    forbidden_pairs.append([a, b])
                    assert prob == Z0, (q, r, a, b, prob)
                    loss = add(loss, prob)
        assert total == ONE, (q, r, total)
        game['pairs'].append({'alice': q, 'bob': r, 'weight': '1/81', 'forbidden': forbidden_pairs})
assert loss == Z0
assert pvm_error < 1e-12
assert o3_error < 1e-12
fms_A = {q: {'f': numeric(flag), **{a: numeric(rational_effect(p)) for a, p in base[q].items()}} for q in ['R1', 'R2', 'R3']}
fms_B = {q: {'f': numeric(flag), **{a: numeric(rational_effect(p)) for a, p in base[q].items()}} for q in ['C1', 'C2', 'C3']}
c = [np.array([1, 0, 1, 0]) / math.sqrt(2), np.array([0, 1, 0, 1]) / math.sqrt(2)]
rays = [np.array([1, 1, 1, 1]) / 2, np.array([1, -1, 1, -1]) / 2]
selected = ['+++', '+--']
for x in range(2):
    es = {}
    for sign, name in [(1, '+'), (-1, '-')]:
        v = np.r_[1, sign * c[x]] / math.sqrt(2)
        es[name] = np.outer(v, v)
    es.update({a: numeric(rational_effect(p)) for a, p in base['C1'].items() if a != selected[x]})
    fms_A['T' + str(x)] = es
for y in range(2):
    es = {}
    for sign, name in [(1, '+'), (-1, '-')]:
        v = np.r_[1, sign * rays[y]] / math.sqrt(2)
        es[name] = np.outer(v, v)
    es.update({a: numeric(rational_effect(p)) for a, p in base['R1'].items() if a != selected[y]})
    fms_B['U' + str(y)] = es

def fms_accept(q, r, a, b):
    if q.startswith('R') and r.startswith('C'):
        return a == b if a == 'f' or b == 'f' else a[int(r[1]) - 1] == b[int(q[1]) - 1]
    if q.startswith('T') and r == 'C1':
        return b in ['f', selected[int(q[1])]] if len(a) == 1 else a == b
    if q == 'R1' and r.startswith('U'):
        return a in ['f', selected[int(r[1])]] if len(b) == 1 else a == b
    if q.startswith('T') and r.startswith('U') and (len(a) == len(b) == 1):
        return (1 if a == b else -1) == (-1) ** (int(q[1]) * int(r[1]))
    return True
d2 = np.diag([1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]) / 3
fms_loss = 0.0
for q, es in fms_A.items():
    for r, fs in fms_B.items():
        for a, p in es.items():
            for b, s in fs.items():
                if not fms_accept(q, r, a, b):
                    fms_loss += float(np.trace(p @ d2 @ s @ d2)) / 25
assert abs(fms_loss) < 1e-12
peak_mb = None
try:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    if not ok:
        raise OSError('GetProcessMemoryInfo failed')
    peak_mb = pmc.PeakWorkingSetSize / (1024 * 1024)
    assert 0 < peak_mb < 1024
except (AttributeError, OSError):
    pass
results = {'status': 'PASS', 'questions': [len(A), len(B)], 'answers_per_question': 5, 'question_pairs': 81, 'events_checked': events, 'forbidden_events': forbidden, 'exact_loss_Qsqrt3': ['0', '0'], 'normalization': 'all 81 pairs exactly 1', 'nonnegativity': 'all 2025 probabilities exact nonnegative', 'pvm_error_float': pvm_error, 'o3_moment_max_error_float': o3_error, 'original_fms_loss_float': fms_loss, 'coordinate_answer_labels': coordinate_labels, 'ray_rigidity': rigidity, 'peak_working_set_MB': peak_mb, 'seconds': time.perf_counter() - START, 'scope': 'Exact finite probabilities plus independent O_3 KMS moments; not a numerical proof of modular extraction.'}
(OUT / 'g3-game.json').write_text(json.dumps(game, indent=2), encoding='utf-8')
(OUT / 'check-results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
(OUT / 'check-output.txt').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2), flush=True)
