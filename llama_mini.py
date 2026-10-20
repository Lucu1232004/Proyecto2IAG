# llama_mini.py — GPT tipo LLaMA mínimo (PyTorch), char y word
# Arquitectura: RMSNorm + RoPE + SwiGLU, pre-norm, sin bias, embeddings atados.
# Referencias: nanoGPT (loop de entrenamiento), rasbt/LLMs-from-scratch (GPT -> Llama),
# litgpt (config TinyStories), potat0qwq/llm-from-scratch (RMSNorm+RoPE+SwiGLU),
# Zhang19 (RMSNorm), Su21 (RoPE), Shazeer20 (SwiGLU), Touvron23 (LLaMA).
# Corre en CPU y en GPU T4 (Kaggle/Colab). Solo depende de torch.
#
# Cambios frente a la versión anterior:
#   - evaluate() restaura el modo train y usa ventanas fijas (val loss estable).
#   - batchify() usa un generador de torch sembrado (seed reproducible).
#   - get_corpus() ya no descarga Shakespeare en silencio.
#   - WordTok se construye solo con train y calcula OOV en val/test.
#   - BPC se reporta por carácter además de por token.
#   - Historial de entrenamiento, checkpoints, early stopping.
#   - AMP, gradient clipping, warmup + coseno, weight decay separado.
#   - Muestreo con top-k, top-p y modo greedy.

import contextlib
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "set_seed", "rope_cos_sin", "apply_rope", "RMSNorm", "SwiGLU", "LlamaBlock",
    "LlamaMini", "CharTok", "WordTok", "get_corpus", "make_splits", "batchify",
    "eval_batches", "evaluate", "train", "sample", "save_checkpoint", "load_checkpoint",
]


def set_seed(seed=42):
    """Fija random, torch y CUDA para que el run sea reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Bloques LLaMA ----------

def rope_cos_sin(head_dim, seq_len, base=10000.0, device="cpu", dtype=torch.float32):
    """Cosenos y senos de RoPE con la convención de LLaMA (mitades, no pares)."""
    assert head_dim % 2 == 0, "head_dim debe ser par para RoPE"
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                 # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, H, T, hd). cos, sin: (T, hd)."""
    t = q.shape[2]
    c = cos[:t].to(device=q.device, dtype=q.dtype)[None, None, :, :]
    s = sin[:t].to(device=q.device, dtype=q.dtype)[None, None, :, :]
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


class RMSNorm(nn.Module):
    """RMSNorm de Zhang19. En fp16/bf16 calcula en fp32, como LLaMA."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dt = x.dtype
        if dt in (torch.float16, torch.bfloat16):
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.g
        return x.to(dt)


class SwiGLU(nn.Module):
    """SwiGLU de Shazeer20: down(silu(gate(x)) * up(x))."""

    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LlamaBlock(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0, max_seq=256, rope_base=10000.0):
        super().__init__()
        assert d % n_heads == 0, "d debe ser divisible entre n_heads"
        self.n_heads = n_heads
        self.hd = d // n_heads
        assert self.hd % 2 == 0, "d/n_heads debe ser par para RoPE"
        self.rope_base = rope_base
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = SwiGLU(d, int(8 / 3 * d))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout
        cos, sin = rope_cos_sin(self.hd, max_seq, base=rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if T > self.cos.shape[0]:
            cos, sin = rope_cos_sin(self.hd, T, base=self.rope_base, device=x.device)
        else:
            cos, sin = self.cos, self.sin
        q, k = apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.permute(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


class LlamaMini(nn.Module):
    def __init__(self, vocab, d=256, layers=6, heads=8, seq=256, dropout=0.1,
                 rope_base=10000.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [LlamaBlock(d, heads, dropout, max_seq=seq, rope_base=rope_base) for _ in range(layers)]
        )
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.seq = seq
        self.config = {
            "vocab_size": int(vocab),
            "d": int(d),
            "layers": int(layers),
            "heads": int(heads),
            "seq": int(seq),
            "dropout": float(dropout),
            "rope_base": float(rope_base),
        }
        self.apply(self._init_weights)
        self.head.weight = self.tok_emb.weight  # weight tying

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            vocab=cfg["vocab_size"], d=cfg["d"], layers=cfg["layers"],
            heads=cfg["heads"], seq=cfg["seq"], dropout=cfg.get("dropout", 0.1),
            rope_base=cfg.get("rope_base", 10000.0),
        )

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        loss = acc = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            acc = (logits.argmax(-1) == targets).float().mean()
        return logits, loss, acc


# ---------- Tokenizadores ----------

class CharTok:
    """Tokenizador de caracteres. El id 0 queda reservado para <UNK>."""

    name = "char"

    def __init__(self, text, lower=True):
        self.lower = lower
        text = text.lower() if lower else text
        self.chars = sorted(set(text))
        self.itos = ["<UNK>"] + self.chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.unk_id = 0

    def encode(self, s):
        if self.lower:
            s = s.lower()
        return [self.stoi.get(c, self.unk_id) for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def oov_rate(self, text, max_words=None):
        if self.lower:
            text = text.lower()
        if not text:
            return 0.0
        return sum(c not in self.stoi for c in text) / len(text)

    def to_dict(self):
        return {"type": "char", "itos": list(self.itos), "lower": self.lower, "unk_id": self.unk_id}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.lower = d.get("lower", True)
        obj.itos = list(d["itos"])
        obj.stoi = {c: i for i, c in enumerate(obj.itos)}
        obj.unk_id = d.get("unk_id", 0)
        obj.chars = obj.itos[1:]
        return obj

    def __len__(self):
        return len(self.itos)


class WordTok:
    """Tokenizador de palabras. Id 0 = <UNK>, id 1 = <PAD>."""

    name = "word"

    def __init__(self, text, max_tokens=10000, min_freq=1, lower=True):
        from collections import Counter
        if max_tokens is not None and max_tokens < 3:
            raise ValueError("max_tokens debe ser >= 3 o None")
        self.lower = lower
        words = self._split(text)
        counts = Counter(words)
        limit = None if max_tokens is None else max_tokens - 2
        vocab = [w for w, c in counts.most_common(limit) if c >= min_freq]
        self.itos = ["<UNK>", "<PAD>"] + vocab
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk_id = 0

    def _split(self, s):
        return (s.lower() if self.lower else s).split()

    def encode(self, s):
        return [self.stoi.get(w, self.unk_id) for w in self._split(s)]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if i != 1)

    def oov_rate(self, text, max_words=None):
        words = self._split(text)
        if max_words is not None:
            words = words[:max_words]
        if not words:
            return 0.0
        return sum(w not in self.stoi for w in words) / len(words)

    def to_dict(self):
        return {"type": "word", "itos": list(self.itos), "lower": self.lower, "unk_id": self.unk_id}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.lower = d.get("lower", True)
        obj.itos = list(d["itos"])
        obj.stoi = {w: i for i, w in enumerate(obj.itos)}
        obj.unk_id = d.get("unk_id", 0)
        return obj

    def __len__(self):
        return len(self.itos)


def tokenizer_from_dict(d):
    if d.get("type") == "char":
        return CharTok.from_dict(d)
    if d.get("type") == "word":
        return WordTok.from_dict(d)
    raise ValueError(f"Tipo de tokenizador desconocido: {d.get('type')}")


# ---------- Corpus y splits ----------

def _resolve_path(path):
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return alt if os.path.exists(alt) else path


def get_corpus(path="corpus_nuevo.txt", max_chars=None):
    """Lee el corpus. Falla con un mensaje claro si no existe (sin descargas silenciosas)."""
    path = _resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el corpus en '{path}'. "
            "Descarga TinyStories o pasa una ruta válida."
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if max_chars is not None:
        text = text[:max_chars]
    if not text:
        raise ValueError(f"El corpus '{path}' está vacío.")
    return text


def _snap_to_space(text, idx):
    """Mueve idx hacia adelante hasta el siguiente espacio para no cortar palabras."""
    idx = max(0, min(idx, len(text)))
    while idx < len(text) and not text[idx].isspace():
        idx += 1
    return idx


def make_splits(text, train_chars=1_000_000, val_chars=60_000):
    """train [:1M], val [1M:1.06M], test resto (como corpus_stats.md)."""
    if len(text) < 100:
        raise ValueError("El corpus es demasiado pequeño.")
    if train_chars <= 0 or val_chars <= 0:
        raise ValueError("train_chars y val_chars deben ser positivos.")
    tr_end = _snap_to_space(text, train_chars)
    if tr_end >= len(text) - 1:
        raise ValueError("train_chars cubre todo el corpus; no queda val/test.")
    va_end = _snap_to_space(text, tr_end + val_chars)
    if va_end <= tr_end:
        raise ValueError("val_chars no deja texto para validación.")
    return text[:tr_end], text[tr_end:va_end], text[va_end:]


# ---------- Lotes ----------

def batchify(ids, seq, bs, device="cpu", seed=None):
    """Generador infinito de ventanas aleatorias. Reproducible si se pasa seed."""
    if len(ids) <= seq + 1:
        raise ValueError("El split de train es más corto que seq+1.")
    ids = ids.to(device)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    n = len(ids) - seq - 1
    arange = torch.arange(seq)
    while True:
        ix = torch.randint(0, n, (bs,), generator=gen)
        offsets = (ix[:, None] + arange[None, :]).to(device)
        x = ids[offsets]
        y = ids[offsets + 1]
        yield x, y


def eval_batches(ids, seq, bs, device="cpu"):
    """Ventanas fijas y no solapadas. Hace que la val loss sea comparable entre evals."""
    if len(ids) <= seq:
        raise ValueError("El split es más corto que seq.")
    ids = ids.to(device)
    n = (len(ids) - 1) // seq
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        xs, ys = [], []
        for i in range(start, stop):
            xs.append(ids[i * seq:(i + 1) * seq])
            ys.append(ids[i * seq + 1:(i + 1) * seq + 1])
        yield torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(model, ids, seq, device="cpu", bs=32, max_batches=None, chars_per_token=1.0):
    """Evalúa con ventanas fijas. Restaura el modo del modelo y devuelve un dict."""
    was_training = model.training
    model.eval()
    tot_l = tot_a = 0.0
    n = 0
    n_batches = 0
    try:
        for x, y in eval_batches(ids, seq, bs, device):
            _, loss, acc = model(x, y)
            nt = x.numel()
            tot_l += loss.item() * nt
            tot_a += acc.item() * nt
            n += nt
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break
    finally:
        model.train(was_training)
    if n == 0:
        raise ValueError("No se evaluó ningún token.")
    loss = tot_l / n
    ppl = math.exp(loss) if loss < 700 else float("inf")
    bpc_token = loss / math.log(2)
    cpt = max(1e-9, float(chars_per_token))
    return {
        "loss": loss,
        "ppl": ppl,
        "bpc": bpc_token / cpt,
        "bpc_token": bpc_token,
        "acc": tot_a / n,
        "n_tokens": n,
        "chars_per_token": cpt,
    }


# ---------- Entrenamiento ----------

def _lr_at(step, steps, warmup, lr, min_lr_ratio):
    warmup = max(0, min(warmup, steps))
    min_lr = lr * min_lr_ratio
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(text, level="char", d=256, layers=6, heads=8, seq=128, steps=3000, bs=64,
          lr=3e-4, min_lr_ratio=0.1, weight_decay=0.1, dropout=0.1, grad_clip=1.0,
          warmup=100, train_chars=1_000_000, val_chars=60_000, test_chars=500_000,
          word_vocab=10000, eval_every=250, eval_bs=32, test_batches=50,
          patience=5, min_delta=1e-4, seed=42, device=None, amp=True,
          checkpoint_path=None, tokenizer=None, verbose=True):
    """Entrena un LlamaMini char o word y devuelve (model, tokenizer, history).

    - Construye el tokenizador solo con train (sin fuga de val/test).
    - Val/test usan ventanas fijas; el test se limita a test_chars para no evaluar 18M.
    - Guarda el mejor checkpoint (por val loss) si checkpoint_path no es None.
    - Al final carga los pesos del mejor paso antes de medir test.
    """
    if level not in ("char", "word"):
        raise ValueError("level debe ser 'char' o 'word'.")
    if steps < 1 or eval_every < 1:
        raise ValueError("steps y eval_every deben ser >= 1.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)

    train_text, val_text, test_text = make_splits(text, train_chars, val_chars)

    if tokenizer is None:
        tokenizer = (CharTok(train_text) if level == "char"
                     else WordTok(train_text, max_tokens=word_vocab))
    elif tokenizer.name != level:
        raise ValueError(f"El tokenizador es '{tokenizer.name}' pero level='{level}'.")

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    test_eval_text = test_text[:test_chars] if test_chars is not None else test_text
    test_ids = (torch.tensor(tokenizer.encode(test_eval_text), dtype=torch.long)
                if test_eval_text else None)

    cpt_val = len(val_text) / max(1, len(val_ids))
    cpt_test = len(test_eval_text) / max(1, len(test_ids)) if test_ids is not None else cpt_val
    oov_val = tokenizer.oov_rate(val_text)
    oov_test = tokenizer.oov_rate(test_eval_text, max_words=200_000) if test_eval_text else 0.0

    model = LlamaMini(len(tokenizer), d, layers, heads, seq, dropout).to(device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-5,
    )

    amp_enabled = bool(amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def autocast_ctx():
        if amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    history = {
        "step": [], "train_loss": [], "val_loss": [], "ppl": [], "bpc": [],
        "bpc_token": [], "acc": [], "lr": [], "tokens": [], "tok_s": [], "elapsed": [],
        "level": level,
        "config": dict(model.config),
        "vocab_size": len(tokenizer),
        "params": model.num_params(),
        "params_no_emb": model.num_params(non_embedding=True),
        "split": {
            "train_chars": len(train_text), "val_chars": len(val_text),
            "test_chars": len(test_text), "test_eval_chars": len(test_eval_text),
            "train_tokens": len(train_ids), "val_tokens": len(val_ids),
            "test_tokens": len(test_ids) if test_ids is not None else 0,
        },
        "oov": {"val": oov_val, "test": oov_test},
        "chars_per_token": {"val": cpt_val, "test": cpt_test},
        "best_val": None, "best_step": None, "stopped_early": False,
        "seed": seed, "device": str(device), "amp": amp_enabled,
    }

    if verbose:
        print(f"level={level} device={device} amp={amp_enabled} vocab={len(tokenizer)} "
              f"params={model.num_params()/1e6:.2f}M "
              f"({model.num_params(non_embedding=True)/1e6:.2f}M sin embeddings)")
        print(f"split chars: train={len(train_text):,} val={len(val_text):,} "
              f"test={len(test_text):,} (eval={len(test_eval_text):,})")
        print(f"tokens: train={len(train_ids):,} val={len(val_ids):,} | "
              f"OOV val={oov_val:.3f} test={oov_test:.3f} | "
              f"chars/token val={cpt_val:.2f}")

    # Evaluación inicial (paso 0) para que las curvas tengan punto de partida.
    v0 = evaluate(model, val_ids, seq, device, bs=eval_bs, chars_per_token=cpt_val)
    history["step"].append(0)
    history["train_loss"].append(None)
    history["val_loss"].append(v0["loss"])
    history["ppl"].append(v0["ppl"])
    history["bpc"].append(v0["bpc"])
    history["bpc_token"].append(v0["bpc_token"])
    history["acc"].append(v0["acc"])
    history["lr"].append(0.0)
    history["tokens"].append(0)
    history["tok_s"].append(None)
    history["elapsed"].append(0.0)
    if verbose:
        print(f"step     0/{steps} | val {v0['loss']:.4f} | ppl {v0['ppl']:.1f} | "
              f"bpc {v0['bpc']:.3f} | acc {v0['acc']:.3f}")

    gen = batchify(train_ids, seq, bs, device, seed=seed + 1)
    best = float("inf")
    best_step = None
    best_state = None
    bad_evals = 0
    run_loss = 0.0
    run_n = 0
    train_time = 0.0
    t_start = time.perf_counter()

    for step in range(1, steps + 1):
        cur_lr = _lr_at(step, steps, warmup, lr, min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = cur_lr

        x, y = next(gen)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            _, loss, _ = model(x, y)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        train_time += time.perf_counter() - t0

        li = float(loss.item())
        if math.isfinite(li):
            run_loss += li * x.numel()
            run_n += x.numel()

        if step % eval_every != 0 and step != steps:
            continue

        val = evaluate(model, val_ids, seq, device, bs=eval_bs, chars_per_token=cpt_val)
        tr_loss = run_loss / run_n if run_n else float("nan")
        elapsed = time.perf_counter() - t_start
        tok_s = (step * bs * seq) / train_time if train_time > 0 else 0.0

        improved = val["loss"] < best - min_delta
        if improved:
            best = val["loss"]
            best_step = step
            bad_evals = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_val"] = best
            history["best_step"] = best_step
        else:
            bad_evals += 1

        history["step"].append(step)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val["loss"])
        history["ppl"].append(val["ppl"])
        history["bpc"].append(val["bpc"])
        history["bpc_token"].append(val["bpc_token"])
        history["acc"].append(val["acc"])
        history["lr"].append(cur_lr)
        history["tokens"].append(step * bs * seq)
        history["tok_s"].append(tok_s)
        history["elapsed"].append(elapsed)

        if improved and checkpoint_path:
            save_checkpoint(checkpoint_path, model, tokenizer, history,
                            extra={"step": step, "val_loss": best})

        if verbose:
            print(f"step {step:5d}/{steps} | train {tr_loss:.4f} | val {val['loss']:.4f} | "
                  f"ppl {val['ppl']:.1f} | bpc {val['bpc']:.3f} | acc {val['acc']:.3f} | "
                  f"lr {cur_lr:.2e} | {tok_s/1000:.1f}k tok/s | {elapsed/60:.1f} min")

        run_loss = 0.0
        run_n = 0

        if patience and bad_evals >= patience:
            history["stopped_early"] = True
            if verbose:
                print(f"Early stopping en step {step}: {bad_evals} evals sin mejorar.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if history["best_step"] is not None:
        best_val_metrics = evaluate(model, val_ids, seq, device, bs=eval_bs,
                                    chars_per_token=cpt_val)
        history["best"] = best_val_metrics
    else:
        best_val_metrics = None

    if test_ids is not None and len(test_ids) > seq + 1:
        test_metrics = evaluate(model, test_ids, seq, device, bs=eval_bs,
                                max_batches=test_batches, chars_per_token=cpt_test)
        history["test"] = test_metrics
    else:
        test_metrics = None

    if verbose:
        if best_val_metrics is not None:
            msg = (f"FINAL level={level} best_step={history['best_step']} "
                   f"val_loss={best_val_metrics['loss']:.4f} "
                   f"PPL={best_val_metrics['ppl']:.1f} "
                   f"BPC={best_val_metrics['bpc']:.3f} "
                   f"acc={best_val_metrics['acc']:.3f}")
        else:
            msg = f"FINAL level={level} sin mejoras registradas"
        if test_metrics:
            msg += (f" | test_loss={test_metrics['loss']:.4f} "
                    f"PPL={test_metrics['ppl']:.1f} BPC={test_metrics['bpc']:.3f} "
                    f"acc={test_metrics['acc']:.3f}")
        print(msg)

    return model, tokenizer, history


# ---------- Muestreo ----------

@torch.no_grad()
def sample(model, tok, prompt, n_new=200, temp=0.8, top_k=None, top_p=None,
           device=None, seed=None):
    """Genera texto. temp<=0 usa greedy. top_k y top_p se aplican antes de multinomial."""
    device = torch.device(device or next(model.parameters()).device)
    if not prompt:
        if tok.name == "char":
            prompt = "\n"
        else:
            raise ValueError("Para word-level pasa un prompt no vacío.")
    if tok.name == "word" and not prompt.strip():
        raise ValueError("Para word-level pasa un prompt con al menos una palabra.")

    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    if ids.shape[1] == 0:
        ids = torch.zeros((1, 1), dtype=torch.long, device=device)

    try:
        for _ in range(n_new):
            inp = ids[:, -model.seq:]
            logits, _, _ = model(inp)
            logits = logits[0, -1]

            if temp is None or temp <= 0:
                nxt = logits.argmax().view(1, 1)
            else:
                logits = logits / max(float(temp), 1e-6)
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), logits.numel())
                    cutoff = torch.topk(logits, k).values[-1]
                    logits = torch.where(logits < cutoff,
                                         torch.full_like(logits, float("-inf")), logits)
                probs = F.softmax(logits, dim=-1)
                if top_p is not None and 0 < top_p < 1:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    remove = cum - sorted_probs > top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                    probs = probs / probs.sum()
                nxt = torch.multinomial(probs, 1, generator=gen).view(1, 1)

            ids = torch.cat([ids, nxt], dim=1)
    finally:
        model.train(was_training)

    return tok.decode(ids[0].tolist())


# ---------- Checkpoints ----------

def save_checkpoint(path, model, tok, history=None, extra=None):
    """Guarda pesos, config, tokenizador e historial en un solo archivo."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dict(model.config),
        "tokenizer": tok.to_dict(),
        "history": history or {},
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, device="cpu"):
    """Carga un checkpoint y devuelve (model, tokenizer, payload)."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    model = LlamaMini.from_config(payload["config"])
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    tok = tokenizer_from_dict(payload["tokenizer"])
    return model, tok, payload


# ---------- Self-check ----------

def _self_test():
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    text = None
    for name in ("corpus_nuevo.txt", "shakespeare_baseline.txt"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                text = f.read(40000)
            break
    if not text:
        text = "the little girl saw a cat . the cat was happy . " * 800

    tr, va, te = make_splits(text, train_chars=12000, val_chars=3000)
    assert len(tr) >= 12000 and len(va) >= 3000 and len(te) > 0, "splits rotos"

    ctok = CharTok(tr)
    probe = "the cat"
    assert ctok.decode(ctok.encode(probe)) == probe, "roundtrip char roto"
    assert 0.0 <= ctok.oov_rate(va) <= 1.0

    wtok = WordTok(tr, max_tokens=300)
    assert len(wtok) <= 300, "vocab word excede max_tokens"
    assert len(wtok.encode("the cat")) == 2
    assert 0.0 <= wtok.oov_rate(va) <= 1.0

    tr_ids = torch.tensor(ctok.encode(tr), dtype=torch.long)
    va_ids = torch.tensor(ctok.encode(va), dtype=torch.long)
    xb, yb = next(batchify(tr_ids, 16, 4, "cpu", seed=1))
    assert xb.shape == (4, 16) and yb.shape == (4, 16), "batchify roto"
    assert torch.equal(xb[:, 1:], yb[:, :-1]), "shift de batchify roto"

    model, tok, hist = train(
        text, level="char", d=32, layers=1, heads=4, seq=16, steps=4, bs=4,
        lr=1e-3, warmup=1, dropout=0.0, train_chars=12000, val_chars=3000,
        eval_every=2, patience=None, device="cpu", amp=False, verbose=False,
    )
    assert len(hist["step"]) >= 3, "historial incompleto"
    assert math.isfinite(hist["val_loss"][-1]), "val loss no finita"
    assert model.training, "train no deja el modelo en modo train"

    e1 = evaluate(model, va_ids, 16, "cpu", chars_per_token=1.0)
    e2 = evaluate(model, va_ids, 16, "cpu", chars_per_token=1.0)
    assert abs(e1["loss"] - e2["loss"]) < 1e-6, "evaluate no es determinista"
    assert abs(e1["bpc"] - e1["bpc_token"]) < 1e-9, "BPC char mal normalizado"

    s = sample(model, tok, "the", n_new=12, temp=0.8, top_k=5, top_p=0.9,
               device="cpu", seed=1)
    assert len(s) > 3, "sample top-k/top-p roto"
    sg = sample(model, tok, "the", n_new=5, temp=0, device="cpu")
    assert len(sg) > 3, "sample greedy roto"

    tmp = os.path.join(tempfile.gettempdir(), "llama_mini_selftest.pt")
    try:
        save_checkpoint(tmp, model, tok, hist)
        m2, tok2, _ = load_checkpoint(tmp, device="cpu")
        model.eval()
        x = torch.randint(0, len(tok), (2, 8))
        with torch.no_grad():
            l1 = model(x)[0]
            l2 = m2(x)[0]
        assert torch.allclose(l1, l2, atol=1e-5), "checkpoint no reproduce logits"
        assert len(tok2) == len(tok)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    wm, wtok2, whist = train(
        text, level="word", d=32, layers=1, heads=4, seq=8, steps=2, bs=2,
        lr=1e-3, warmup=1, dropout=0.0, train_chars=12000, val_chars=3000,
        word_vocab=300, eval_every=1, patience=None, device="cpu", amp=False,
        verbose=False,
    )
    assert math.isfinite(whist["val_loss"][-1]), "word val loss no finita"
    assert whist["chars_per_token"]["val"] > 1.0, "chars/token word mal calculado"
    assert whist["bpc"] < whist["bpc_token"], "BPC word no normalizado por carácter"

    print("SELF-CHECK OK")


if __name__ == "__main__":
    _self_test()
