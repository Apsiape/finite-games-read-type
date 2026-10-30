if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import math, json, ctypes
from pathlib import Path
import numpy as np
rng = np.random.default_rng(89134)
out = {}

def sch(x):
    return np.linalg.svd(x, compute_uv=False)

def dist_lists(a, b):
    n = max(len(a), len(b))
    return np.linalg.norm(np.pad(a, (0, n - len(a))) - np.pad(b, (0, n - len(b))))

def H2(lam):
    a = np.sort(np.asarray(lam))[::-1]
    return float(np.linalg.norm(np.repeat(a, 2) / math.sqrt(2) - np.pad(a, (0, len(a)))) ** 2)

def unitary(n, t):
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = (h + h.conj().T) / 2
    val, vec = np.linalg.eigh(h)
    return vec * np.exp(1j * t * val) @ vec.conj().T
seq = []
for mult in [1, 2]:
    d = 9 * mult
    qs = []
    for k in range(9):
        q = np.zeros((d, d))
        q[k * mult:(k + 1) * mult, k * mult:(k + 1) * mult] = np.eye(mult)
        qs.append(q)
    for t in [0.0, 1e-05, 0.01, 0.1, 0.7]:
        for trial in range(4):
            lam = np.exp(rng.uniform(-5, 0, d))
            lam /= np.linalg.norm(lam)
            x = np.diag(lam).astype(complex)
            noise = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            x += 0.12 * noise / np.linalg.norm(noise)
            x /= np.linalg.norm(x)
            fs = []
            for q in qs:
                u = unitary(d, t)
                fs.append(u @ q @ u.conj().T)
            ds = np.array([np.linalg.norm(f @ x - x @ q) for f, q in zip(fs, qs)])
            nl = float(ds @ ds)
            ts = []
            prefix = np.eye(d, dtype=complex)
            for f in fs:
                ts.append(f @ prefix)
                prefix = (np.eye(d) - f) @ prefix
            ts.append(prefix)
            isoerr = np.linalg.norm(sum((a.conj().T @ a for a in ts)) - np.eye(d))
            subsets = [list(range(9)), [0, 2, 5], [], list(range(1, 9))]
            maxratio = 0.0
            maxvec = 0.0
            for subset in subsets:
                if not subset:
                    continue
                b = sum((qs[k] for k in subset))
                joined = np.sort(np.concatenate([sch(x @ qs[k]) for k in subset]))[::-1]
                err = dist_lists(sch(x @ b), joined)
                verror = 0.0
                for k, a in enumerate(ts):
                    for l in subset:
                        block = a @ x @ qs[l]
                        if k == l:
                            block = block - x @ qs[l]
                        verror += float(np.linalg.norm(block) ** 2)
                if nl > 1e-24:
                    maxratio = max(maxratio, err / math.sqrt(54 * nl))
                    maxvec = max(maxvec, verror / (54 * nl))
                assert verror <= 54 * nl + 1e-20
                assert err <= math.sqrt(54 * nl) + 1e-10
            seq.append({'d': d, 't': t, 'N_L': nl, 'isometry_error': float(isoerr), 'spectral_bound_ratio': float(maxratio), 'vector_squared_bound_ratio': float(maxvec)})
out['sequential_noncommuting_flags'] = {'trials': len(seq), 'max_isometry_error': max((s['isometry_error'] for s in seq)), 'max_vector_squared_bound_ratio': max((s['vector_squared_bound_ratio'] for s in seq)), 'max_spectral_bound_ratio': max((s['spectral_bound_ratio'] for s in seq))}
errs = []
for d in [2, 4]:
    n = 3
    gs = []
    for i in range(n):
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        gs.append(a.conj().T @ a)
    vals, vec = np.linalg.eigh(sum(gs))
    invs = vec / np.sqrt(vals) @ vec.conj().T
    es = [invs @ g @ invs for g in gs]
    w = np.zeros((n * d, n * d), complex)
    for i, e in enumerate(es):
        vals, vec = np.linalg.eigh(e)
        root = vec * np.sqrt(np.maximum(vals, 0)) @ vec.conj().T
        w[i * d:(i + 1) * d, :d] = root
    ii = np.eye(n * d)
    u = np.block([[w, ii - w @ w.conj().T], [ii - w.conj().T @ w, w.conj().T]])
    p = []
    for i in range(n):
        q = np.zeros((2 * n * d, 2 * n * d))
        q[i * d:(i + 1) * d, i * d:(i + 1) * d] = np.eye(d)
        if i == 0:
            q[n * d:, n * d:] = ii
        p.append(u.conj().T @ q @ u)
    errs.append({'d': d, 'unitary_error': float(np.linalg.norm(u.conj().T @ u - np.eye(2 * n * d))), 'compression_error': float(max((np.linalg.norm(p[i][:d, :d] - es[i]) for i in range(n)))), 'pvm_error': float(max((np.linalg.norm(z @ z - z) for z in p)))})
out['naimark'] = errs
c = 0.5 * (1 + 2 * math.sqrt(2) + 3 * math.sqrt(54)) + 2 * math.sqrt(108)
out['constants'] = {'exact_link_coefficient': c, 'link_coefficient_squared': c * c, 'claimed_allowance': 1138, 'C_F_status': 'Universal finite constant; no validated numerical upper bound', 'C_R_formula': '243/8 * (C_F**2 + 4552)', 'small_H_sufficient_loss_formula': '0.05**2 / C_R'}
assert c * c < 1138

def step_h2(widths, heights):
    order = np.argsort(-heights)
    widths = np.asarray(widths, float)[order]
    heights = np.asarray(heights, float)[order]
    heights /= float(widths @ heights)
    edges = np.r_[0, np.cumsum(widths)]
    pts = np.unique(np.r_[edges, edges / 2])
    mid = (pts[1:] + pts[:-1]) / 2

    def p(t):
        i = np.searchsorted(edges, t, side='right') - 1
        return np.where((i >= 0) & (i < len(heights)), heights[np.clip(i, 0, len(heights) - 1)], 0)
    return float(np.diff(pts) @ (np.sqrt(p(mid)) - np.sqrt(2 * p(2 * mid))) ** 2)
family = []
for levels in [10, 30, 100, 300]:
    k = np.arange(levels)
    a = np.sqrt(2 / (levels + 1)) * np.sin(np.pi * (k + 1) / (levels + 1))
    h = step_h2(2.0 ** k, a * a / 2.0 ** k)
    family.append({'levels': levels, 'sorted_H2': h, 'classical_environment_vector_error': 2 - math.sqrt(2), 'ratio_error_to_H2': (2 - math.sqrt(2)) / h})
out['same_marginal_mixed_or_center_counterexample'] = family
a1 = 2.338107410459767
cs = 2 * a1 ** 1.5 / 3 ** 1.5
slacks = []
for h in np.geomspace(1e-07, 0.05, 2000):
    wall = math.ceil(2 * math.log2(1 / h))
    delta = 2 / (math.sqrt(2) - 1) ** 2 * h * h * 2 ** (-wall)
    mass = 1 - delta
    energy = h * h + math.sqrt(2) * delta
    theta = (a1 * mass / (3 * energy)) ** 1.5
    ground = (a1 * theta ** (1 / 3) - 1.5) / (1 + a1 * theta ** (-2 / 3) / 6)
    exact_bound = mass * ground - theta * energy - wall - delta / math.log(2)
    asserted = cs / h - 2 * math.log2(1 / h) - 3.5
    slacks.append(exact_bound - asserted)
out['explicit_airy_cutoff'] = {'samples': len(slacks), 'minimum_slack': float(min(slacks))}
assert min(slacks) > 0
if os.name == 'nt':

    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    counts = PMC()
    counts.cb = ctypes.sizeof(counts)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counts), counts.cb)
    out['resource'] = {'query_succeeded': bool(ok), 'peak_working_set_bytes': counts.PeakWorkingSetSize, 'thread_environment': {k: os.environ[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}}
    assert not ok or counts.PeakWorkingSetSize < 1000000000
dest = Path(__file__).with_suffix('.json')
dest.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2), flush=True)
