"""numpy-only receipts for docs/STATE_STABILITY.md (no torch, no model; < 1 min on a laptop CPU).

    python -I scripts/state_stability_numeric_check.py

RWKV-7 transition (key-first, controlsafe/adapters.py:138):  A = diag(w) - diag(a) kk kk^T,
    w in [exp(-e^-0.5), 1) = [0.5452, 1),  a in (0, 1),  ||kk|| = 1.
GDN transition (controlsafe/adapters.py:141):                A = alpha (I - beta k k^T).
Checks (each prints PASS/FAIL):
  R1  explicit 2x2 RWKV-7 transition with ||A||_2 > 1 (common 2-norm bound refuted)
  R2  the T2(b) per-step certificate U >= ||A||_2 on random draws, and its tightness for scalar a
  R3  spectrum of A is real and inside [w_min - sum a kk^2, w_max] (similarity to a symmetric matrix)
  R4  an admissible PERIOD-2 switching pair with rho(A2 A1) > 1 (joint spectral radius of the box > 1)
  G1  ||I - beta k k^T||_2 = max(1, |1 - beta ||k||^2|), incl. bf16-inflated ||k||^2 with beta near 2
  P1  bf16 rounding of alpha near 1, and ||k||^2 of a bf16-rounded unit vector
"""
from __future__ import annotations

import struct

import numpy as np

WMIN = float(np.exp(-np.exp(-0.5)))  # 0.54524...
rng = np.random.default_rng(20261010)
ok_all = True


def check(name, cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {msg}")


def A_rwkv(w, a, kk):
    kk = kk / np.linalg.norm(kk)
    return np.diag(w) - (a * kk)[:, None] * kk[None, :]


def U_cert(w, a, kk):
    """T2(b): ||A||_2 <= max(w_max, abar - w_min) + sqrt(sum kk^2 (a - abar)^2),  abar = sum a kk^2."""
    kk = kk / np.linalg.norm(kk)
    abar = float((a * kk * kk).sum())
    return max(w.max(), abar - w.min()) + float(np.sqrt((kk * kk * (a - abar) ** 2).sum()))


n2 = lambda M: float(np.linalg.norm(M, 2))

# R1
A = A_rwkv(np.array([0.99, 0.99]), np.array([0.99, 0.01]), np.array([1.0, 1.0]))
check("R1", n2(A) > 1, f"w=(.99,.99) a=(.99,.01) kk=(1,1)/sqrt2: A={A.round(4).tolist()} ||A||_2={n2(A):.4f} "
      f"rho(A)={max(abs(np.linalg.eigvals(A))):.4f}")

# R2
worst_gap, scalar_gap = np.inf, 0.0
for _ in range(20000):
    N = int(rng.integers(2, 65))
    w = rng.uniform(WMIN, 1, N); a = rng.uniform(0, 1, N); kk = rng.normal(size=N)
    worst_gap = min(worst_gap, U_cert(w, a, kk) - n2(A_rwkv(w, a, kk)))
    a_s = np.full(N, rng.uniform(0, 1))
    scalar_gap = max(scalar_gap, U_cert(w, a_s, kk) - n2(A_rwkv(w, a_s, kk)))
check("R2", worst_gap >= -1e-12, f"min(U - ||A||_2) over 20k draws = {worst_gap:.3e} (>= 0: certificate sound); "
      f"scalar-a max(U - ||A||_2) = {scalar_gap:.3e} (U = max(w_max, a - w_min) there, < 1)")

# R3
bad = 0
for _ in range(5000):
    N = int(rng.integers(2, 65))
    w = rng.uniform(WMIN, 1, N); a = rng.uniform(1e-3, 1, N); kk = rng.normal(size=N); kk /= np.linalg.norm(kk)
    ev = np.linalg.eigvals(A_rwkv(w, a, kk))
    lo, hi = w.min() - (a * kk * kk).sum(), w.max()
    bad += int(abs(ev.imag).max() > 1e-9 or ev.real.min() < lo - 1e-9 or ev.real.max() > hi + 1e-9)
check("R3", bad == 0, f"{bad}/5000 draws violate 'real spectrum in [w_min - sum a kk^2, w_max]' (so rho(A) < 1 always)")

# sup ||A||_2 over the box (numerical lower bound; embeds into any head size)
best = 0.0
for _ in range(3000):
    w = rng.uniform(WMIN, 1, 2); a = rng.uniform(0, 1, 2); kk = rng.normal(size=2)
    best = max(best, n2(A_rwkv(w, a, kk)))
th = np.linspace(0, np.pi, 4001)
grid = max(n2(A_rwkv(np.ones(2), np.array([1.0, 0.0]), np.array([np.cos(t), np.sin(t)]))) for t in th)
print(f"INFO sup||A||_2 (n=2, random) >= {best:.4f};  closure w=1, a=(1,0), kk on the circle: {grid:.5f} "
      f"(2/sqrt3 = {2 / np.sqrt(3):.5f})")

# R4: period-2 switching, explicit pair (w held at 0.99 so it is not a boundary artefact)
def pair(t1, t2, w=0.99, lo=0.01, hi=0.99):
    A1 = A_rwkv(np.full(2, w), np.array([hi, lo]), np.array([np.cos(t1), np.sin(t1)]))
    A2 = A_rwkv(np.full(2, w), np.array([lo, hi]), np.array([np.cos(t2), np.sin(t2)]))
    return A1, A2

best = (0, None)
for t1 in np.linspace(0, np.pi, 181):
    for t2 in np.linspace(0, np.pi, 181):
        A1, A2 = pair(t1, t2)
        r = max(abs(np.linalg.eigvals(A2 @ A1))) ** 0.5
        if r > best[0]:
            best = (r, (t1, t2))
r, (t1, t2) = best
A1, A2 = pair(t1, t2)
check("R4", r > 1, f"w=0.99, a1=(.99,.01), kk1=({np.cos(t1):.4f},{np.sin(t1):.4f}); a2=(.01,.99), "
      f"kk2=({np.cos(t2):.4f},{np.sin(t2):.4f}): rho(A2 A1)^(1/2) = {r:.4f} > 1, so ||(A2 A1)^m|| grows like {r:.4f}^(2m); "
      f"||A1||={n2(A1):.4f} ||A2||={n2(A2):.4f} rho(A1)={max(abs(np.linalg.eigvals(A1))):.4f}")
M = np.linalg.matrix_power(A2 @ A1, 100)
print(f"INFO ||(A2 A1)^100||_2 = {n2(M):.3e} (200 steps, every step has rho(A_t) <= 0.99)")

# G1
for beta, kn2 in ((0.9, 1.0), (0.99, 1.0016), (1.9, 1.0), (1.99, 1.0016), (2.0, 1.0016)):
    k = rng.normal(size=128); k = k / np.linalg.norm(k) * np.sqrt(kn2)
    M = np.eye(128) - beta * np.outer(k, k)
    f = max(1.0, abs(1 - beta * kn2))
    check("G1", abs(n2(M) - f) < 1e-9, f"beta={beta} ||k||^2={kn2}: ||I - beta k k^T||_2 = {n2(M):.6f} (formula {f:.6f})")


# P1
def bf16(x):
    b = struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


print("INFO bf16(alpha):", {al: bf16(al) for al in (0.996, 0.998, 0.9985, 0.999)})
mx = max(sum(bf16(x) ** 2 for x in v / np.linalg.norm(v)) for v in rng.normal(size=(2000, 128)))
print(f"INFO max ||bf16(k/||k||)||^2 over 2000 draws (dim 128) = {mx:.6f}")
print("ALL PASS" if ok_all else "SOME CHECK FAILED")
