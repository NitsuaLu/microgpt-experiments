"""
microgpt — KV Cache version.
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
Everything else is just efficiency.
@karpathy
"""
import os
import random
import time
import torch


random.seed(42)
torch.manual_seed(42)

# Dataset
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]

random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
print(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# Parameters
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

# Model
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
        # 1) Multi-head Attention block
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
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits


# Training
num_steps = 1000
accum_steps = 4
optimizer = torch.optim.Adam(state_dict.values(), lr=1e-3)

ema_loss = None
doc_idx = 0

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

    if (step + 1) % 100 == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {avg_loss:.4f} | ema {ema_loss:.4f}")
    else:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {avg_loss:.4f} | ema {ema_loss:.4f}", end='\r')

# Inference with KV Cache
temperature = 0.5
print("\n--- inference (WITH KV Cache) ---")
generated = []
all_times = []

for sample_idx in range(20):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    token_id = BOS
    sample = []
    sample_times = []

    for pos_id in range(block_size):
        t0 = time.perf_counter()
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits / temperature)
        token_id = torch.multinomial(probs, 1).item()
        t1 = time.perf_counter()
        sample_times.append(t1 - t0)

        if token_id == BOS:
            break
        sample.append(uchars[token_id])

    name = ''.join(sample)
    generated.append(name)
    all_times.append(sample_times)

    times_str = ', '.join([f'{t*1000:.2f}ms' for t in sample_times])
    print(f"sample {sample_idx+1:2d}: {name:15s} | steps: {len(sample_times):2d} | {times_str}")

# Overfit check
docs_set = set(docs)
overlap = [n for n in generated if n in docs_set]
print(f"\n--- overfit check ---")
print(f"directly copied from training set: {overlap}, {len(overlap)}/20")

# Timing summary
total_time = sum(sum(t) for t in all_times)
total_calls = sum(len(t) for t in all_times)
print(f"\n--- timing summary (KV Cache) ---")
print(f"total gpt() calls: {total_calls}")
print(f"total inference time: {total_time:.3f}s")
print(f"avg time per call: {total_time/total_calls*1000:.3f}ms")
if all_times:
    print(f"first 3 steps avg: {sum(all_times[0][:3])/min(3,len(all_times[0]))*1000:.3f}ms per step")
    last_times = [t[-1] for t in all_times if len(t) >= 3]
    if last_times:
        print(f"last step avg (len>=3): {sum(last_times)/len(last_times)*1000:.3f}ms")
