"""Context-passing on VERIS: pack all class definitions INTO the sequence with
the incident, feed frozen ModernBERT, read out an embedding, train a probe.
Tests whether frozen bidirectional attention 'contextualizes' the input against
the definitions enough to beat the plain summary-only baseline (~58%).

Arms (same records/splits/seeds, frozen ModernBERT, linear probe):
  plain      : summary only (reference ~58%)
  ctx_all    : definitions + summary, mean-pool WHOLE sequence
  ctx_cls    : definitions + summary, [CLS] token (as the pitch specifies)
  ctx_input  : definitions + summary, mean-pool over INPUT tokens only
               (strongest: read the input representation after it attends to defs)

Per-class recall reported so we can see the surface-vs-intent split: context
should help lexical classes (malware/social/physical) more than intent
(hacking/error/misuse) if it works like the NLI cousin did.
"""

import collections, io, json, time, urllib.request, zipfile
import torch

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [11, 23, 37]

DEFS = {
    "hacking":  "unauthorized access by exploiting or bypassing controls; brute force, SQL injection, stolen credentials, exploiting a vulnerability.",
    "malware":  "malicious software installed to compromise systems; virus, ransomware, trojan, backdoor, spyware, keylogger.",
    "error":    "unintentional mistake by an authorized person; misconfiguration, misdelivery, accidental disclosure, lost device, publishing error.",
    "misuse":   "an authorized insider abusing legitimate access or privileges for an unauthorized purpose; privilege abuse, data mishandling.",
    "physical": "physical action; theft or loss of a device or document, tampering, surveillance, break-in.",
    "social":   "manipulating a person; phishing, pretexting, bribery, spoofed emails requesting credentials.",
}
PREFIX = "CYBER_DEFINITIONS:\n" + "\n".join(f"- {c}: {DEFS[c]}" for c in CLASSES) + "\n\nUSER_INPUT_TO_CLASSIFY:\n"


def _get(url, timeout=120):
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 5: raise
            time.sleep(10 * (a + 1))


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(MODEL)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModel.from_pretrained(path, local_files_only=True).eval()

    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))
    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None
    buckets = collections.defaultdict(list)
    for r in data:
        lab = label(r); s = r.get("summary")
        if lab in CLASSES and isinstance(s, str) and len(s.strip()) > 40 and len(buckets[lab]) < CAP:
            buckets[lab].append(s.strip()[:800])
    summaries, labels = [], []
    for c in CLASSES:
        for s in buckets[c]:
            summaries.append(s); labels.append(IDX[c])
    n = len(summaries); y = torch.tensor(labels)
    print(f"records {n}\n", flush=True)

    prefix_len = len(tok(PREFIX, add_special_tokens=True)["input_ids"]) - 1  # drop trailing special

    def embed(texts, maxlen, pool, inputmask=False):
        out = []
        for i in range(0, len(texts), 8):
            e = tok(texts[i:i+8], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
            with torch.no_grad():
                h = mdl(**e).last_hidden_state
            am = e["attention_mask"].clone().float()
            if pool == "cls":
                out += h[:, 0].tolist()
            else:
                if inputmask:                         # zero out the definition-prefix positions
                    am[:, :prefix_len] = 0.0
                m = am.unsqueeze(-1)
                out += ((h*m).sum(1) / m.sum(1).clamp(min=1.0)).tolist()
        return torch.tensor(out)

    def probe(X):
        X = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp(min=1e-6)
        accs, last_pred, last_yt = [], None, None
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=g).tolist()
            cut = int(n*.72); tr, te = perm[:cut], perm[cut:]
            clf = torch.nn.Linear(X.shape[1], 6)
            opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.03)
            lf = torch.nn.CrossEntropyLoss()
            for _ in range(400):
                opt.zero_grad(); lf(clf(X[tr]), y[tr]).backward(); opt.step()
            with torch.no_grad():
                pred = clf(X[te]).argmax(1)
            yt = y[torch.tensor(te)]
            accs.append((pred == yt).float().mean().item()*100)
            last_pred, last_yt = pred, yt
        recall = {CLASSES[c]: round((((last_pred==c)&(last_yt==c)).sum().item()) /
                  max(1, (last_yt==c).sum().item()) * 100) for c in range(6)}
        return sum(accs)/len(accs), [round(a,1) for a in accs], recall

    full = [PREFIX + s for s in summaries]
    reps = {
        "plain":     embed(summaries, 256, "mean"),
        "ctx_all":   embed(full, 512, "mean"),
        "ctx_cls":   embed(full, 512, "cls"),
        "ctx_input": embed(full, 512, "mean", inputmask=True),
    }
    for name, X in reps.items():
        acc, seeds, recall = probe(X)
        print(f"{name:10} {round(acc,1)}%   seeds {seeds}", flush=True)
        print(f"           per-class recall: {recall}", flush=True)


main()
