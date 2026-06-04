"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
@karpathy
"""
import os       # os.path.exists
import random   # random.seed, random.choices, random.gauss, random.shuffle
import torch


random.seed(42) # Let there be order among chaos
torch.manual_seed(42)

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]


random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token

print(uchars)

vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head

#create random matrix (nout * nin), in which each is tensor
matrix = lambda nout, nin, std=0.08: torch.normal(0.0, std, (nout, nin), requires_grad=True)

state_dict = {
    'wte': matrix(vocab_size, n_embd), # Word Token Embedding. from token id to embedding vector
    # input x = token embedding + position embedding
    'wpe': matrix(block_size, n_embd), # Word Position Embedding, to describe the position
    'lm_head': matrix(vocab_size, n_embd) #Language Model Head, to turn hidden state to vocabulary logits(the probability of each token)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd) #Q = XW_q
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd) #K = XW_k
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd) #V = XW_v
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd) # A = softmax(QK^T / sqrt(d_k) ) V and Y = AW_o

    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) #respectively out_dim and input_dim
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)


# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return x @ w.T

def softmax(logits): #safe softmax
    max_val = logits.max(dim=-1, keepdim=True)[0]
    exps = (logits - max_val).exp()
    total = exps.sum(dim=-1, keepdim=True)
    return exps / total

def rmsnorm(x):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(ms + 1e-5)
    return x * scale

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = tok_emb + pos_emb # joint token and position embedding

    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

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

            attn_weight = softmax((q_h @k_h.T) / (head_dim ** 0.5))
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
num_steps = 1000 # number of training steps
optimizer = torch.optim.Adam(state_dict.values(), lr=1e-3)

for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    optimizer.zero_grad()
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()
    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits / temperature)
        token_id = torch.multinomial(probs, 1).item()
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
