"""
KV Cache benchmark + visualization.
Trains the model once, then times inference with and without KV Cache,
plots step-index vs per-step time to compare O(n) vs O(n²).
"""
import os
import time
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

random.seed(42)
torch.manual_seed(42)

# ── Dataset ──────────────────────────────────────────────
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ── Tokenizer ────────────────────────────────────────────
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ── Parameters ───────────────────────────────────────────
n_layer = 3
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.02: torch.normal(0.0, std, (nout, nin), requires_grad=True)

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# ── Model ────────────────────────────────────────────────
def linear(x, w):
    return x @ w.T

def softmax(logits):
    max_val = logits.max(dim=-1, keepdim=True)[0]
    exps = (logits - max_val).exp()
    total = exps.sum(dim=-1, keepdim=True)
    return exps / total

def rmsnorm(x):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(ms + 1e-5)
    return x * scale

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = tok_emb + pos_emb
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = torch.tensor([])
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = torch.stack([ki[hs:hs+head_dim] for ki in keys[li]])
            v_h = torch.stack([vi[hs:hs+head_dim] for vi in values[li]])
            attn_weight = softmax((q_h @ k_h.T) / (head_dim ** 0.5))
            head_out = attn_weight @ v_h
            x_attn = torch.cat([x_attn, head_out], dim=0)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual
    logits = linear(x, state_dict['lm_head'])
    return logits

# ── Training ─────────────────────────────────────────────
num_steps = 1000
accum_steps = 4
optimizer = torch.optim.Adam(state_dict.values(), lr=1e-3)
ema_loss = None
doc_idx = 0
print("training...")
for step in range(num_steps):
    optimizer.zero_grad()
    accum_loss = 0
    for micro_step in range(accum_steps):
        doc = docs[doc_idx % len(docs)]
        doc_idx += 1
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)
        (loss / accum_steps).backward()
        accum_loss += loss.data.item()
    optimizer.step()
    avg_loss = accum_loss / accum_steps
    if ema_loss is None:
        ema_loss = avg_loss
    else:
        ema_loss = 0.9 * ema_loss + 0.1 * avg_loss
    if (step + 1) % 200 == 0 or step == 0:
        print(f"  step {step+1:4d} / {num_steps:4d} | loss {avg_loss:.4f} | ema {ema_loss:.4f}")

# ── Benchmark ─────────────────────────────────────────────
temperature = 0.5
torch.manual_seed(123)  # fixed seed for reproducible sampling

print("\nbenchmarking KV Cache version...")
times_kv = []
keys = [[] for _ in range(n_layer)]
values = [[] for _ in range(n_layer)]
token_id = BOS
tokens_kv = [BOS]

for pos_id in range(block_size):
    t0 = time.perf_counter()
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits / temperature)
    token_id = torch.multinomial(probs, 1).item()
    t1 = time.perf_counter()
    times_kv.append((t1 - t0) * 1000)  # ms
    if token_id == BOS:
        break
    tokens_kv.append(token_id)

print(f"  KV Cache: {len(times_kv)} tokens, total {sum(times_kv):.2f}ms")

torch.manual_seed(123)  # same seed → same tokens

print("benchmarking NO KV Cache version...")
times_nokv = []
past_tokens = [BOS]
tokens_nokv = [BOS]

for pos_id in range(block_size):
    t0 = time.perf_counter()
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    for i in range(len(past_tokens)):
        logits = gpt(past_tokens[i], i, keys, values)
    probs = softmax(logits / temperature)
    token_id = torch.multinomial(probs, 1).item()
    t1 = time.perf_counter()
    times_nokv.append((t1 - t0) * 1000)  # ms
    if token_id == BOS:
        break
    tokens_nokv.append(token_id)
    past_tokens.append(token_id)

print(f"  NO KV Cache: {len(times_nokv)} tokens, total {sum(times_nokv):.2f}ms")

# ── Correctness check ─────────────────────────────────────
assert tokens_kv == tokens_nokv, \
    f"TOKEN MISMATCH!\n  KV: {tokens_kv}\n  NoKV: {tokens_nokv}"
name = ''.join(uchars[t] for t in tokens_kv[1:] if t != BOS)
print(f"  tokens match! generated: \"{name}\"")

# ── Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: per-step time
ax = axes[0]
steps = list(range(1, len(times_kv) + 1))
ax.plot(steps, times_kv, 'o-', color='#2196F3', linewidth=2, markersize=5, label='KV Cache')
ax.plot(steps, times_nokv, 's-', color='#F44336', linewidth=2, markersize=5, label='No KV Cache')
ax.set_xlabel('Generation Step', fontsize=12)
ax.set_ylabel('Time per step (ms)', fontsize=12)
ax.set_title('Per-Step Inference Time', fontsize=14)
ax.legend(fontsize=11)

# Plot 2: cumulative time
ax = axes[1]
cum_kv = [sum(times_kv[:i]) for i in range(1, len(times_kv) + 1)]
cum_nokv = [sum(times_nokv[:i]) for i in range(1, len(times_nokv) + 1)]
ax.plot(steps, cum_kv, 'o-', color='#2196F3', linewidth=2, markersize=5, label='KV Cache')
ax.plot(steps, cum_nokv, 's-', color='#F44336', linewidth=2, markersize=5, label='No KV Cache')
ax.set_xlabel('Generation Step', fontsize=12)
ax.set_ylabel('Cumulative time (ms)', fontsize=12)
ax.set_title('Cumulative Inference Time', fontsize=14)
ax.legend(fontsize=11)

fig.suptitle(f'KV Cache vs No Cache — {n_layer} layers, block_size={block_size}', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('KV_Cache_benchmark.png', bbox_inches='tight', dpi=150)
print("\nchart saved to KV_Cache_benchmark.png")

# ── Summary stats ─────────────────────────────────────────
ratio = sum(times_nokv) / sum(times_kv)
print(f"\n--- summary ---")
print(f"KV Cache:       {len(times_kv)} steps, {sum(times_kv):.2f}ms total, {sum(times_kv)/len(times_kv):.3f}ms/step avg")
print(f"No KV Cache:    {len(times_nokv)} steps, {sum(times_nokv):.2f}ms total, {sum(times_nokv)/len(times_nokv):.3f}ms/step avg")
print(f"speedup:        {ratio:.1f}x faster with KV Cache")
print(f"theoretical:    O(n) vs O(n²) — KV Cache per-step time stays flat,")
print(f"                No Cache grows linearly with sequence length")
