"""Reviewer #3: cache frozen lower-layer activations, train only the top layers.

Mechanism (version-robust -- reuses ModernBERT's own forward, no internal
mask/rotary reimplementation):
  1. ONE frozen no_grad pass over all records with output_hidden_states=True;
     cache the hidden state at each cut point c (input to layer c).
  2. Replace layers[0:c] with cheap pass-throughs (skips their compute) and a
     forward-pre-hook on layer c injects the cached activation for the batch.
  3. Only layers[c:] + final_norm + classifier are trainable (full FT of the
     top block). Backward touches only the top -> far less compute AND memory
     (should avoid the OOM-kill that killed full-encoder LoRA in Docker).

Compares cut points so we see the speed/accuracy curve:
  c=22 -> train nothing above (pure frozen-head baseline, ~expect ~55-58%)
  c=18 -> train top 4 layers
  c=14 -> train top 8 layers
max_length=128. Reports per-cut: train seconds, accuracy, macro-F1.
Reference: frozen head ~58%, roberta full-LoRA ~70%.
"""

import collections, io, json, time, urllib.request, zipfile
import torch, torch.nn as nn

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
MAXLEN = 128
CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 11
EPOCHS = 4
CUTS = [10, 6]   # 10=top12; 6=top16 (deeper sweep)


def _get(url, timeout=120):
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 5: raise
            time.sleep(10 * (a + 1))


class Passthrough(nn.Module):
    """Skips a frozen lower layer's compute. Carries attention_type because
    ModernBERT's forward indexes the rotary dict via encoder_layer.attention_type
    before calling the layer."""
    def __init__(self, attention_type):
        super().__init__()
        self.attention_type = attention_type
    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cls = nn.Linear(d, 6)
    def forward(self, pooled):
        return self.cls(pooled)


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(MODEL)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    print(f"threads {torch.get_num_threads()} | max_len {MAXLEN}", flush=True)

    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))
    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None
    buckets = collections.defaultdict(list)
    for r in data:
        lab = label(r); s = r.get("summary")
        if lab in CLASSES and isinstance(s, str) and len(s.strip()) > 40 and len(buckets[lab]) < CAP:
            buckets[lab].append(s.strip()[:1000])
    texts, labels = [], []
    for c in CLASSES:
        for s in buckets[c]:
            texts.append(s); labels.append(IDX[c])
    n = len(texts)
    y = torch.tensor(labels)

    # fixed-length tokenization once (uniform 128 so cached tensors stack)
    E = tok(texts, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    ids, mask = E["input_ids"], E["attention_mask"]

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    tr, te = perm[:int(n*.72)], perm[int(n*.72):]
    print(f"records {n} | train {len(tr)} test {len(te)} | cuts {CUTS}\n", flush=True)

    # ---- 1. cache frozen hidden states at each cut, in ONE no_grad pass ----
    base = AutoModel.from_pretrained(path, local_files_only=True).eval()
    for p in base.parameters(): p.requires_grad = False
    want = sorted(set(CUTS))
    cache = {c: torch.zeros(n, MAXLEN, base.config.hidden_size) for c in want}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, 16):
            out = base(input_ids=ids[i:i+16], attention_mask=mask[i:i+16],
                       output_hidden_states=True)
            hs = out.hidden_states  # len 23: hs[k] = input to layer k
            for c in want:
                cache[c][i:i+16] = hs[c]
    print(f"frozen caching pass (all cuts): {time.time()-t0:.1f}s\n", flush=True)
    del base

    def macro_f1(pred, yt):
        f1 = []
        for c in range(6):
            tp = ((pred==c)&(yt==c)).sum().item()
            fp = ((pred==c)&(yt!=c)).sum().item()
            fn = ((pred!=c)&(yt==c)).sum().item()
            p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
            f1.append(2*p*r/(p+r) if p+r else 0)
        return sum(f1)/6*100

    # ---- 2+3. per cut: passthrough lower, train top block on cached acts ----
    for c in CUTS:
        m = AutoModel.from_pretrained(path, local_files_only=True)
        for i in range(c):
            m.layers[i] = Passthrough(m.layers[i].attention_type)
        # freeze embeddings; top layers [c:] trainable
        for p in m.embeddings.parameters(): p.requires_grad = False
        head = Head(m.config.hidden_size)

        inject = {"buf": None}
        def pre_hook(module, args, kwargs):
            new = (inject["buf"],) + tuple(args[1:])
            return (new, kwargs)
        h_handle = (m.layers[c].register_forward_pre_hook(pre_hook, with_kwargs=True)
                    if c < len(m.layers) else None)

        params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
        n_train = sum(p.numel() for p in params)
        opt = torch.optim.Adam(params, lr=5e-5 if c < 22 else 1e-3)
        lf = nn.CrossEntropyLoss()

        def run_batch(bidx):
            cb = cache[c][bidx]
            if h_handle is not None:
                inject["buf"] = cb
                out = m(input_ids=ids[bidx], attention_mask=mask[bidx])
                last = out.last_hidden_state
            else:  # c==22: no top layers; cached IS the final pre-norm state
                last = m.final_norm(cb)
            mm = mask[bidx].unsqueeze(-1).float()
            pooled = (last*mm).sum(1) / mm.sum(1).clamp(min=1.0)
            return head(pooled)

        t0 = time.time()
        m.train(); head.train()
        for ep in range(EPOCHS):
            order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(SEED*9+ep)).tolist()
            for i in range(0, len(tr), 16):
                bidx = torch.tensor([tr[order[k]] for k in range(i, min(i+16, len(tr)))])
                opt.zero_grad()
                lf(run_batch(bidx), y[bidx]).backward()
                opt.step()
        secs = time.time() - t0

        m.eval(); head.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(te), 16):
                bidx = torch.tensor([te[k] for k in range(i, min(i+16, len(te)))])
                preds.append(run_batch(bidx).argmax(1))
        pred = torch.cat(preds); yt = y[torch.tensor(te)]
        acc = (pred == yt).float().mean().item()*100
        tag = f"frozen-head" if c == 22 else f"top{22-c}layers"
        print(f"cut c={c:>2} [{tag:>11}]  train {secs:6.1f}s  "
              f"acc {acc:5.1f}%  macroF1 {macro_f1(pred,yt):5.1f}%  "
              f"(trainable {n_train/1e6:.1f}M)", flush=True)
        if h_handle is not None: h_handle.remove()
        del m, head


main()
