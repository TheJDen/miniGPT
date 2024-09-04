import torch
import torch.nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5_000
eval_interval = 500
learning_rate = 3e-4
device = "cpu"#'mps' if torch.backends.mps.is_available() else 'cpu'
# mps has bug for large models right now
eval_iters = 200
num_embedding = 128
num_layers = 6
num_heads = 4
dropout = 0.2

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

s_to_i = {ch: i for i, ch in enumerate(chars)}
i_to_s = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: list(map(s_to_i.get, s))
decode = lambda l: "".join(map(i_to_s.get, l))

data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BatchedSelfAttention(torch.nn.Module):
    def __init__(self, num_embedding, num_heads):
        assert num_embedding % num_heads == 0
        super().__init__()
        self.num_embedding = num_embedding
        self.num_heads = num_heads
        self.attention = torch.nn.Linear(num_embedding, 3 * num_embedding, bias=False)
        self.proj = torch.nn.Linear(num_embedding, num_embedding)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size, device=device).view(1, 1, block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.attention(x).split(self.num_embedding, dim=2)
        k = k.view(B * T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B * T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B * T, self.num_heads, C // self.num_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        y = att @ v
        out = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))

class FeedForward(torch.nn.Module):
    def __init__(self, num_embedding):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_embedding, 4 * num_embedding),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * num_embedding, num_embedding),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, num_embedding, num_heads):
        super().__init__()
        head_size = num_embedding // num_heads
        self.sa = BatchedSelfAttention(num_embedding, num_heads)
        self.ffwd = FeedForward(num_embedding)
        self.ln1 = torch.nn.LayerNorm(num_embedding)
        self.ln2 = torch.nn.LayerNorm(num_embedding)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, num_embedding)
        self.position_embedding_table = torch.nn.Embedding(block_size, num_embedding)
        self.blocks = torch.nn.Sequential(*(Block(num_embedding, num_heads=num_heads) for _ in range(num_layers)))
        self.ln_f = torch.nn.LayerNorm(num_embedding)
        self.lm_head = torch.nn.Linear(num_embedding, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        x = position_embeddings + token_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = LanguageModel()
model = m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

import os
model_path = "weights.pt"
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    for i in range(max_iters):
        if i % eval_interval == 0:
            losses = estimate_loss()
            print(f'step {i}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    losses = estimate_loss()
    print(f'step {max_iters - 1}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
except KeyboardInterrupt:
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

with open(model_path, 'wb') as f:
    torch.save(model.state_dict(), f)

