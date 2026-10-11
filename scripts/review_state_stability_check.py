"""Adversarial numpy receipts for REVIEW_STATE_STABILITY.md (reviewer of docs/STATE_STABILITY.md @62df6219).

    python -I scripts/review_state_stability_check.py

numpy only, CPU, < 2 min. Each block prints PASS/FAIL/INFO. "FAIL" here means "the reviewed claim broke".
  V1  T2(b) certificate U >= ||A||_2 under adversarial (hill-climbing) search, n = 2..8, incl. a, w at box edges
  V2  is c = abar the best choice in T2(b)? (min over c of the bound vs the abar bound)
  V3  sup ||A||_2 over the RWKV-7 box: is it 2/sqrt3 (takeaway) or only >= 2/sqrt3 (body)?
  V4  JSR > 1 construction: smallest common decay w that still gives rho(A2 A1)^(1/2) > 1, and with a in [.05,.95]
  V5  T2(c) scalar-a heads under adversarial switching: max over random products of ||prod||^(1/m) stays < 1
  V6  GDN: ||alpha (I - beta k k^T)|| with k from bf16 l2norm (fla stores k in input dtype) and beta = sigmoid
  V7  float rounding of the decays: alpha / w = exp(.) rounding to exactly 1.0 in fp32 and bf16
  V8  per-layer contraction vs whole-network influence: a 2-layer toy recurrent stack with a residual stream
"""
from __future__ import annotations

import struct

import numpy as np

rng = np.random.default_rng(20261010 + 7)
WLO = float(np.exp(-np.exp(-0.5)))
n2 = lambda M: float(np.linalg.norm(M, 2))
ok = True


def rep(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {tag}: {msg}")


def A_rwkv(w, a, kk):
    kk = kk / np.linalg.norm(kk)
    return np.diag(w) - (a * kk)[:, None] * kk[None, :]


def U(w, a, kk):
    kk = kk / np.linalg.norm(kk)
    ab = float((a * kk * kk).sum())
    return max(w.max(), ab - w.min()) + float(np.sqrt((kk * kk * (a - ab) ** 2).sum()))


def clip(w, a):
    return np.clip(w, WLO, 1.0), np.clip(a, 0.0, 1.0)


# V1: hill-climb to minimise U - ||A||
worst = np.inf
for trial in range(300):
    N = int(rng.integers(2, 9))
    w, a = clip(rng.uniform(WLO, 1, N), rng.choice([0.0, 1.0], N) * 0.5 + rng.uniform(0, .5, N))
    kk = rng.normal(size=N)
    gap = U(w, a, kk) - n2(A_rwkv(w, a, kk))
    step = 0.2
    for it in range(250):
        w2, a2 = clip(w + step * rng.normal(size=N), a + step * rng.normal(size=N))
        k2 = kk + step * rng.normal(size=N)
        g2 = U(w2, a2, k2) - n2(A_rwkv(w2, a2, k2))
        if g2 < gap:
            w, a, kk, gap = w2, a2, k2, g2
        else:
            step *= 0.995
    worst = min(worst, gap)
rep("V1", worst >= -1e-12, f"adversarial min(U - ||A||_2) over 300 hill-climbs x 250 steps = {worst:.3e} (>= 0: certificate holds; "
    f"it is a Weyl + rank-one triangle bound, so it should be unbreakable)")

# V2: optimal c
improve, frac = 0.0, 0
for _ in range(3000):
    N = int(rng.integers(2, 33))
    w = rng.uniform(WLO, 1, N); a = rng.uniform(0, 1, N); kk = rng.normal(size=N); kk /= np.linalg.norm(kk)
    ab = float((a * kk * kk).sum())
    cs = np.linspace(0, 1.5, 601)
    b = [max(w.max(), c - w.min()) + np.linalg.norm((a - c) * kk) for c in cs]
    d = U(w, a, kk) - min(b)
    improve = max(improve, d)
    frac += int(d > 1e-9)
print(f"INFO V2: a grid over c improves on c = abar in {frac}/3000 draws (max {improve:.2e}). Reason: on the RWKV-7 box "
      f"abar - w_min <= 1 - {WLO:.4f} = {1 - WLO:.4f} < {WLO:.4f} <= w_max, so the max() branch is always w_max, "
      f"U_t = w_max + sigma_a, and abar (the minimiser of the rank-one term) is optimal")
skew = -np.inf
for _ in range(20000):
    N = int(rng.integers(2, 65))
    w = rng.uniform(WLO, 1, N); a = rng.uniform(0, 1, N); kk = rng.normal(size=N); kk /= np.linalg.norm(kk)
    ab = float((a * kk * kk).sum())
    skew = max(skew, ab - w.min() - w.max())
rep("V2", skew < 0, f"max(abar - w_min - w_max) over 20k draws = {skew:.4f} < 0, so max(w_max, abar - w_min) == w_max always")

# V3: sup over the box, n = 2..4, hill-climb maximising ||A||
best = 0.0
for trial in range(200):
    N = int(rng.integers(2, 5))
    w, a = clip(rng.uniform(WLO, 1, N), rng.uniform(0, 1, N)); kk = rng.normal(size=N)
    val = n2(A_rwkv(w, a, kk)); step = 0.3
    for it in range(300):
        w2, a2 = clip(w + step * rng.normal(size=N), a + step * rng.normal(size=N)); k2 = kk + step * rng.normal(size=N)
        v2 = n2(A_rwkv(w2, a2, k2))
        if v2 > val:
            w, a, kk, val = w2, a2, k2, v2
        else:
            step *= 0.99
    best = max(best, val)
rep("V3", best <= 2 / np.sqrt(3) + 1e-9, f"hill-climbed sup ||A||_2 over closure of the box (n<=4) = {best:.6f}; 2/sqrt3 = {2 / np.sqrt(3):.6f} "
    f"(analytic: with w=1, sigma_max^2 = (2 - p + sqrt(p(4-3p)))/2 maximised at p = a^T kk^2 = 1/3 gives 4/3)")


# V4: JSR construction vs decay level
def best_pair(w, lo, hi, n=91):
    best = 0.0
    for t1 in np.linspace(0, np.pi, n):
        for t2 in np.linspace(0, np.pi, n):
            A1 = A_rwkv(np.full(2, w), np.array([hi, lo]), np.array([np.cos(t1), np.sin(t1)]))
            A2 = A_rwkv(np.full(2, w), np.array([lo, hi]), np.array([np.cos(t2), np.sin(t2)]))
            best = max(best, max(abs(np.linalg.eigvals(A2 @ A1))) ** 0.5)
    return best


rows = []
for w in (0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, WLO):
    rows.append((w, best_pair(w, 0.01, 0.99), best_pair(w, 0.05, 0.95)))
print("INFO V4: period-2 growth rate rho(A2A1)^(1/2) by common decay w  [a in (.01,.99)] [a in (.05,.95)]:")
for w, r1, r2 in rows:
    print(f"      w={w:.4f}  {r1:.4f}  {r2:.4f}")
wcrit = min([w for w, r1, _ in rows if r1 > 1], default=None)
print(f"INFO V4: smallest tested common decay with growth (a in .01/.99): {wcrit}")

# V5: scalar-a heads: random switching products
worst_rate = 0.0
for _ in range(400):
    N = 4; m = 40; P = np.eye(N)
    for t in range(m):
        w = rng.uniform(WLO, 1 - 1e-3, N); a = np.full(N, rng.uniform(0, 1)); kk = rng.normal(size=N)
        P = A_rwkv(w, a, kk) @ P
    worst_rate = max(worst_rate, n2(P) ** (1 / m))
rep("V5", worst_rate < 1, f"scalar-a: worst ||A_m...A_1||^(1/m) over 400 random 40-step products = {worst_rate:.4f} < 1")


# V6 / V7: precision
def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    b = x.view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return b.view(np.float32)


mx, worst6 = 0.0, 0.0
for _ in range(3000):
    k = rng.normal(size=128).astype(np.float32) * rng.choice([1e-2, 1.0, 30.0])
    kb = bf16(k)
    kn = bf16(kb / np.sqrt((kb.astype(np.float64) ** 2).sum() + 1e-6))  # fla l2norm_fwd: fp32 math, store bf16
    s = float((kn.astype(np.float64) ** 2).sum())
    mx = max(mx, s)
    beta = 1 / (1 + np.exp(-rng.normal() * 8))
    worst6 = max(worst6, n2(np.eye(128) - beta * np.outer(kn, kn)))
rep("V6", worst6 <= 1 + 1e-12, f"bf16-stored l2norm k: max ||k||^2 = {mx:.6f} (>1, so the doc's '||k|| < 1' is exact-arithmetic only), "
    f"but with beta=sigmoid<1, max ||I - beta k k^T|| = {worst6:.9f} -> ||A_t|| = alpha_t still holds")
f32_one = np.float32(np.exp(np.float32(-3e-8)))
print(f"INFO V7: fp32 exp(-3e-8) = {f32_one!r} (== 1.0: {f32_one == np.float32(1)}); bf16 exp(-0.0019) = {bf16(np.exp(-0.0019))[()]!r}, "
      f"bf16 exp(-0.002) = {bf16(np.exp(-0.002))[()]!r}")
print("      RWKV-7 rwkv7_recurrent computes decay = torch.exp(w_log) in the ACTIVATION dtype (modeling_rwkv7.py:144):")
print("      in a bf16 model any channel with w_log > -0.00195 (sigmoid(.) < 0.0032) gets w = 1.0 exactly -> T2(a) 'rho < 1' fails in floats.")

# V8: per-layer contraction is not network-level influence decay when a non-decaying path exists
# toy: layer 1 recurrent contraction rho; layer 2 = attention-like running mean (no decay) over layer-1 outputs.
rho, T, t1 = 0.9, 400, 10
x = np.zeros(T); x[:t1] = 1.0  # span = first t1 tokens, clean afterwards (both runs agree after t1)
def run(xin):
    s, out1 = 0.0, []
    for t in range(T):
        s = rho * s + xin[t]; out1.append(s)
    out1 = np.array(out1)
    att = np.cumsum(out1) / np.arange(1, T + 1)  # uniform causal attention
    return out1, att
o1a, ata = run(x); o1b, atb = run(np.zeros(T))
print(f"INFO V8: toy stack, span of {t1} tokens: layer-1 |dy| at tau=300: {abs(o1a - o1b)[t1 + 300]:.2e} (decays rho^tau); "
      f"after a uniform-attention layer |dy|: {abs(ata - atb)[t1 + 300]:.2e} (~1/t, no geometric decay)")
print("ALL PASS" if ok else "SOME CHECK FAILED (see FAIL lines)")
