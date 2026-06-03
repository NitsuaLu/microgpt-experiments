"""
Pure attention complexity micro-benchmark.
No training, no model — just measure attention compute cost vs sequence length.
Demonstrates O(n) vs O(n²) growth.
"""
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})

torch.manual_seed(42)

d = 64                          # head dimension (like a real model)
seq_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
n_warmup = 2                    # warmup runs before timing

times_nocache = []              # total time for full sequence, no cache
times_cache = []                # total time for full sequence, with cache

for T in seq_lengths:
    # random projection matrices (simulate W_k, W_v, W_q of one head)
    W_k = torch.randn(d, d)
    W_v = torch.randn(d, d)
    W_q = torch.randn(d, d)

    # pre-create all input vectors for the sequence
    X = torch.randn(T, d)

    # ── Warmup ────────────────────────────────────────────
    for _ in range(n_warmup):
        for t in range(T):
            k = X[:t+1] @ W_k
            v = X[:t+1] @ W_v
            q = X[t] @ W_q
            s = q @ k.T / (d ** 0.5)
            w = torch.softmax(s, dim=-1)
            _ = w @ v

    # ── NO KV Cache ───────────────────────────────────────
    # Each step t: recompute K,V for ALL positions 0..t, then attention.
    # Step cost ∝ (t+1)×d² (projection) + (t+1)×d (attention) ≈ O(t)
    # Total cost = Σ(t from 0 to T-1) O(t) = O(T²)
    t0 = time.perf_counter()
    for t in range(T):
        k = X[:t+1] @ W_k              # (t+1, d) ← recomputed every time
        v = X[:t+1] @ W_v              # (t+1, d)
        q = X[t] @ W_q                 # (d,)
        s = q @ k.T / (d ** 0.5)       # (d,) @ (d, t+1) = (t+1,)
        w = torch.softmax(s, dim=-1)
        _ = w @ v                      # (t+1,) @ (t+1, d) = (d,)
    t1 = time.perf_counter()
    times_nocache.append((t1 - t0) * 1000)  # total ms

    # ── WITH KV Cache ─────────────────────────────────────
    # Each step t: compute K,V for only the NEW token (cost = d²),
    # lookup past K,V from cache, then attention over growing history.
    # Step cost ∝ d² (new projection) + (t+1)×d (attention) ≈ dominated by d²
    # Total cost = T × O(d²) = O(T)
    t0 = time.perf_counter()
    # Pre-allocate cache buffers — write directly, no copy overhead
    K_cache = torch.zeros(T, d)
    V_cache = torch.zeros(T, d)
    for t in range(T):
        x = X[t:t+1]                   # (1, d)
        K_cache[t] = x @ W_k           # write to pre-allocated slot
        V_cache[t] = x @ W_v
        q = x @ W_q                    # (1, d)
        k = K_cache[:t+1]              # (t+1, d) — view, no copy
        v = V_cache[:t+1]
        s = q @ k.T / (d ** 0.5)       # (1, d) @ (d, t+1) = (1, t+1)
        w = torch.softmax(s, dim=-1)
        _ = w @ v                      # (1, t+1) @ (t+1, d) = (1, d)
    t1 = time.perf_counter()
    times_cache.append((t1 - t0) * 1000)

# ── Plot ──────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

RED = '#E74C3C'
BLUE = '#2980B9'
GRAY1 = '#95A5A6'
GRAY2 = '#BDC3C7'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Plot 1: linear axes ───────────────────────────────────
ax = axes[0]
ax.fill_between(seq_lengths, times_cache, times_nocache,
                alpha=0.08, color=RED)
ax.plot(seq_lengths, times_nocache, 'o-', color=RED, linewidth=2.2,
        markersize=6, markeredgewidth=0, label='No KV Cache')
ax.plot(seq_lengths, times_cache, 's-', color=BLUE, linewidth=2.2,
        markersize=6, markeredgewidth=0, label='KV Cache')
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Total Time (ms)')
ax.set_title('Linear Scale')
ax.legend(frameon=True, fancybox=True, framealpha=0.9)

# clean up spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# ── Plot 2: log-log axes ──────────────────────────────────
ax = axes[1]
ax.fill_between(seq_lengths, times_cache, times_nocache,
                alpha=0.08, color=RED)
ax.loglog(seq_lengths, times_nocache, 'o-', color=RED, linewidth=2.2,
          markersize=6, markeredgewidth=0, label='No KV Cache  O(n²)')
ax.loglog(seq_lengths, times_cache, 's-', color=BLUE, linewidth=2.2,
          markersize=6, markeredgewidth=0, label='KV Cache  O(n)')

# reference slope lines
mid = len(seq_lengths) // 2
ref_start = seq_lengths[1]
ref_end = seq_lengths[-1]
# O(n) reference through mid point of cache curve
o1_y0 = times_cache[mid] * (ref_start / seq_lengths[mid])
o1_y1 = times_cache[mid] * (ref_end / seq_lengths[mid])
ax.loglog([ref_start, ref_end], [o1_y0, o1_y1],
          '--', color=GRAY1, linewidth=1.5, alpha=0.8, label='O(n) reference')
# O(n²) reference through mid point of no-cache curve
o2_y0 = times_nocache[mid] * (ref_start / seq_lengths[mid])**2
o2_y1 = times_nocache[mid] * (ref_end / seq_lengths[mid])**2
ax.loglog([ref_start, ref_end], [o2_y0, o2_y1],
          ':', color=GRAY1, linewidth=1.5, alpha=0.8, label='O(n²) reference')

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Total Time (ms)')
ax.set_title('Log-Log Scale')
ax.legend(frameon=True, fancybox=True, framealpha=0.9)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# ── Global title ──────────────────────────────────────────
fig.suptitle('KV Cache: O(n) vs O(n²)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('attention_complexity.png', bbox_inches='tight', dpi=200)
print("chart saved to attention_complexity.png")

# ── Print summary ─────────────────────────────────────────
print(f"\n{'seq_len':>10}  {'no cache (ms)':>15}  {'cache (ms)':>12}  {'ratio':>8}")
print("-" * 50)
for i, T in enumerate(seq_lengths):
    print(f"{T:>10}  {times_nocache[i]:>15.2f}  {times_cache[i]:>12.2f}  {times_nocache[i]/times_cache[i]:>7.1f}x")
