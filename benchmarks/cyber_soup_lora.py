"""LoRA model soup on cyber (souping's fair shot).

Same protocol as the full-FT soup, but the trainable part is a LoRA adapter on
the top-4 layers instead of full weights. LoRA starts as a no-op (B=0) and stays
low-rank/near-base -> all seeds share a basin, the regime where weight-averaging
is supposed to help. Manual LoRA so we keep the frozen-lower caching speed.

Identical start across runs (seeded adapter init + seeded head), vary only the
data-shuffle order. Fair eval: 3 individuals + soup on the SAME fixed test set.
Question: does the LoRA soup beat/stabilize the best single seed?
"""

import collections, io, json, math, time, urllib.request, zipfile
import torch, torch.nn as nn

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
MAXLEN = 128
CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
CUT = 14            # top-8 (stack the depth win with the soup win)
SPLIT_SEED = 11
HEAD_SEED = 7
ADAPTER_SEED = 3
TRAIN_SEEDS = [11, 23, 37, 51]   # 4 seeds: stronger soup + less noise
EPOCHS = 6
LR = 1e-3
R, ALPHA = 16, 32


def _get(url, timeout=120):
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 5: raise
            time.sleep(10 * (a + 1))


class Passthrough(nn.Module):
    def __init__(self, attention_type):
        super().__init__(); self.attention_type = attention_type
    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class LoRALinear(nn.Module):
    """base frozen; low-rank A@B adapter trainable; starts as a no-op (B=0)."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.empty(base.in_features, r))
        self.B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scaling


class Head(nn.Module):
    def __init__(self, d):
        super().__init__(); self.cls = nn.Linear(d, 6)
    def forward(self, pooled):
        return self.cls(pooled)


def wrap_linears(module, r, alpha):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            wrap_linears(child, r, alpha)


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(MODEL)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)

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
    n = len(texts); y = torch.tensor(labels)

    E = tok(texts, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    ids, mask = E["input_ids"], E["attention_mask"]
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=g).tolist()
    tr, te = perm[:int(n*.72)], perm[int(n*.72):]
    yt = y[torch.tensor(te)]
    print(f"records {n} | train {len(tr)} test {len(te)} | top-{22-CUT} LoRA r={R} | ep {EPOCHS} lr {LR} | seeds {TRAIN_SEEDS}\n", flush=True)

    base = AutoModel.from_pretrained(path, local_files_only=True).eval()
    for p in base.parameters(): p.requires_grad = False
    D = base.config.hidden_size
    cache = torch.zeros(n, MAXLEN, D)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, 16):
            hs = base(input_ids=ids[i:i+16], attention_mask=mask[i:i+16], output_hidden_states=True).hidden_states
            cache[i:i+16] = hs[CUT]
    print(f"frozen caching pass: {time.time()-t0:.1f}s\n", flush=True)
    del base

    def macro_f1(pred, yy):
        f1 = []
        for c in range(6):
            tp=((pred==c)&(yy==c)).sum().item(); fp=((pred==c)&(yy!=c)).sum().item(); fn=((pred!=c)&(yy==c)).sum().item()
            p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
            f1.append(2*p*r/(p+r) if p+r else 0)
        return sum(f1)/6*100

    def build():
        m = AutoModel.from_pretrained(path, local_files_only=True)
        for i in range(CUT):
            m.layers[i] = Passthrough(m.layers[i].attention_type)
        for p in m.parameters(): p.requires_grad = False     # freeze everything
        torch.manual_seed(ADAPTER_SEED)                       # identical adapter init every run
        for i in range(CUT, len(m.layers)):
            wrap_linears(m.layers[i], R, ALPHA)               # add trainable LoRA on top-4
        torch.manual_seed(HEAD_SEED)                          # identical head init every run
        head = Head(D)
        inject = {"buf": None}
        def pre_hook(module, args, kwargs):
            return ((inject["buf"],) + tuple(args[1:]), kwargs)
        h = m.layers[CUT].register_forward_pre_hook(pre_hook, with_kwargs=True)
        return m, head, inject, h

    def forward_batch(m, head, inject, bidx):
        inject["buf"] = cache[bidx]
        last = m(input_ids=ids[bidx], attention_mask=mask[bidx]).last_hidden_state
        mm = mask[bidx].unsqueeze(-1).float()
        pooled = (last*mm).sum(1) / mm.sum(1).clamp(min=1.0)
        return head(pooled)

    def evaluate(m, head, inject):
        m.eval(); head.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(te), 16):
                bidx = torch.tensor([te[k] for k in range(i, min(i+16, len(te)))])
                preds.append(forward_batch(m, head, inject, bidx).argmax(1))
        pred = torch.cat(preds)
        return (pred==yt).float().mean().item()*100, macro_f1(pred, yt)

    probe_m, probe_head, _, probe_h = build()
    tnames = [nm for nm, p in probe_m.named_parameters() if p.requires_grad]
    n_adapt = sum(p.numel() for nm, p in probe_m.named_parameters() if p.requires_grad)
    probe_h.remove(); del probe_m, probe_head
    print(f"LoRA trainable params: {n_adapt/1e6:.2f}M\n", flush=True)

    states_m, states_h, indiv = [], [], []
    for seed in TRAIN_SEEDS:
        m, head, inject, h = build()
        params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
        opt = torch.optim.Adam(params, lr=LR); lf = nn.CrossEntropyLoss()
        t0 = time.time(); m.train(); head.train()
        for ep in range(EPOCHS):
            order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed*9+ep)).tolist()
            for i in range(0, len(tr), 16):
                bidx = torch.tensor([tr[order[k]] for k in range(i, min(i+16, len(tr)))])
                opt.zero_grad(); lf(forward_batch(m, head, inject, bidx), y[bidx]).backward(); opt.step()
        secs = time.time()-t0
        acc, f1 = evaluate(m, head, inject)
        indiv.append(acc)
        print(f"seed {seed:>2}  train {secs:6.1f}s  acc {acc:5.1f}%  macroF1 {f1:5.1f}%", flush=True)
        sd = m.state_dict()
        states_m.append({nm: sd[nm].detach().clone() for nm in tnames})
        states_h.append({nm: p.detach().clone() for nm, p in head.state_dict().items()})
        h.remove(); del m, head

    soup_m = {nm: torch.stack([s[nm] for s in states_m]).mean(0) for nm in tnames}
    soup_h = {nm: torch.stack([s[nm] for s in states_h]).mean(0) for nm in states_h[0]}
    m, head, inject, h = build()
    msd = m.state_dict()
    for nm in tnames: msd[nm].copy_(soup_m[nm])
    m.load_state_dict(msd)
    hsd = head.state_dict()
    for nm in soup_h: hsd[nm].copy_(soup_h[nm])
    head.load_state_dict(hsd)
    sacc, sf1 = evaluate(m, head, inject); h.remove()

    best = max(indiv); avg = sum(indiv)/len(indiv)
    print(f"\n=== LoRA SOUP ===", flush=True)
    print(f"individuals: {[round(a,1) for a in indiv]}  (best {best:.1f}%, mean {avg:.1f}%)", flush=True)
    print(f"SOUP (avg adapters): acc {sacc:.1f}%  macroF1 {sf1:.1f}%", flush=True)
    print(f"soup vs best-single: {sacc-best:+.1f}pp   soup vs mean-single: {sacc-avg:+.1f}pp", flush=True)


main()
