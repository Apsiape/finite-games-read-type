if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import math, json, itertools
from pathlib import Path
from fractions import Fraction
import numpy as np
import sympy as s
from scipy.optimize import linprog
root = Path(__file__).resolve().parent
r2 = s.sqrt(2)
I = s.eye(2)
X = s.Matrix([[0, 1], [1, 0]])
Z = s.diag(1, -1)
Y = s.Matrix([[0, -s.I], [s.I, 0]])
k = s.kronecker_product
square = [[k(X, I), k(I, X), k(X, X)], [k(I, Z), k(Z, I), k(Z, Z)], [k(X, Z), k(Z, X), k(Y, Y)]]
rows = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
cols = [[t for t in itertools.product([1, -1], repeat=3) if math.prod(t) == eta] for eta in [1, 1, -1]]
cvec = [s.Matrix([1, 0, 1, 0]) / r2, s.Matrix([0, 1, 0, 1]) / r2]
rvec = [s.Matrix([1, 1, 1, 1]) / 2, s.Matrix([1, -1, 1, -1]) / 2]

def joint(ops, signs):
    out = s.eye(4)
    for O, v in zip(ops, signs):
        out = out * (s.eye(4) + v * O) / 2
    return out

def projector(q, a, alice):
    if q[0] in 'RC':
        if a == 'f':
            return s.diag(1, 0, 0, 0, 0)
        j = int(q[1]) - 1
        ops = square[j] if alice else [square[i][j] for i in range(3)]
        return s.diag(0, joint(ops, a))
    j = int(q[1])
    if isinstance(a, int):
        v = s.Matrix([1] + list(a * (cvec if alice else rvec)[j])) / r2
        return v * v.T
    ops = [square[i][0] for i in range(3)] if alice else square[0]
    return s.diag(0, joint(ops, a))
game = json.loads((root / 'game.json').read_text())
H = s.zeros(5)
for pair in game['supported_pairs']:
    if not (pair['alice'].startswith('1:') and pair['bob'].startswith('1:')):
        continue
    aq, bq = (pair['alice'], pair['bob'])
    ap = [projector(aq[2:], a, True) for a in game['alice'][aq]]
    bp = [projector(bq[2:], a, False) for a in game['bob'][bq]]
    for i, j in pair['forbidden']:
        H += ap[i].multiply_elementwise(bp[j])
H = s.simplify(H)
print('Exact summed-forbidden quadratic matrix:', H, flush=True)
a, b = s.symbols('a b', real=True)
v = s.Matrix([a, b, b, b, b])
assert s.simplify((v.T * H * v)[0] - (r2 * a - b) ** 2) == 0
print('Exact block identity verified.', flush=True)
n = 16
lam = Fraction(1, 4)
q = [lam ** i / (1 + lam) ** n for i in range(n + 1)]
capacity = [math.comb(n, i) for i in range(n + 1)]
types = list(itertools.combinations_with_replacement(range(-3, 4), 4))
columns = []
energies = {}
metadata = []
cost = []
for t in types:
    v = s.Matrix([1] + [s.Rational(1, 2) ** d for d in t])
    energies[t] = s.expand((v.T * H * v)[0])
for i in range(n + 1):
    for t in types:
        levels = [i] + [i + d for d in t]
        if min(levels) < 0 or max(levels) > n:
            continue
        counts = {j: levels.count(j) for j in set(levels)}
        col = np.zeros(n + 1)
        for j, c in counts.items():
            col[j] = float(c * q[j] / q[i])
        columns.append(col)
        metadata.append((i, t, counts))
        cost.append(float(energies[t]) - sum(col))
A = np.array(columns).T
res = linprog(cost, A_ub=A, b_ub=[float(c * qj) for c, qj in zip(capacity, q)], bounds=(0, None), method='highs', options={'threads': 1})
assert res.success
remaining = capacity.copy()
cert = []
loss = s.Rational(1)
for y, (i, t, counts) in zip(res.x, metadata):
    if y <= 0:
        continue
    proposed = max(0, int(Fraction.from_float(float(y)) / q[i]))
    actual = min([proposed] + [remaining[j] // c for j, c in counts.items()])
    if not actual:
        continue
    for j, c in counts.items():
        remaining[j] -= actual * c
    qi = s.Rational(q[i].numerator, q[i].denominator)
    mass = sum((s.Rational(q[j].numerator, q[j].denominator) * c for j, c in counts.items()))
    loss += actual * (qi * energies[t] - mass)
    cert.append({'flag_level': i, 'magic_offsets': t, 'blocks': actual})
assert all((x >= 0 for x in remaining))
loss = s.expand(loss / 25)
answer = {'n': n, 'lambda': '1/4', 'exact_loss': str(loss), 'loss_decimal': str(s.N(loss, 35)), 'blocks': cert, 'unused_counts': remaining, 'covered_vectors': 2 ** n - sum(remaining), 'all_capacities_exactly_feasible': True, 'quadratic_matrix': str(H)}
(root / 'independent_integer_certificate.json').write_text(json.dumps(answer, indent=2), encoding='utf8')
print('Integer-feasible independent entry:', {k: v for k, v in answer.items() if k not in ['blocks', 'quadratic_matrix', 'unused_counts']}, flush=True)
for name, L in [('1/3', s.Rational(1, 3)), ('1/4', s.Rational(1, 4))]:
    floor = s.simplify((s.sqrt(1 / L) - r2) ** 2 / (25 * (1 + 4 / L)))
    print('Closed-form ' + name + ':', floor, '=', s.N(floor, 35), flush=True)
A9 = s.zeros(9)
for j in range(9):
    for d in [1, 2]:
        A9[j, (j + d) % 9] = 1
assert all((x > 0 for x in A9 ** 8))
print('A^8 minimum entry:', min(A9 ** 8), 'row sums 2; cycles 5 and 9 give period 1.', flush=True)
original = (root / 'expected-output.txt').read_text().splitlines()
rerun = (root / 'fixed_state_fms_output.txt').read_text().splitlines()
print('Supplied output equals rerun:', original == rerun, flush=True)
