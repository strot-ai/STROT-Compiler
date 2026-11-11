"""Compile-time fine-tune tier: the optimizer rung above frozen distillation.

:mod:`strot.distill` trains a logistic head on the encoder's frozen
embeddings, which caps at the frozen representation ceiling. For
representation-bound tasks - where the decision-critical distinction is not
linearly present in the pooled embedding - no head, retrieval, or reframing
crosses that ceiling; only changing weights does. This module is that change,
kept as cheap and CPU-survivable as possible.

The lower encoder layers are frozen and their activations are computed once
and cached; only the top ``N`` layers carry a trainable LoRA adapter. Training
backprops through the top block only, so it finishes on CPU in minutes and
sidesteps the memory blow-up of full-encoder fine-tuning. ``N`` is small by
design (top-4 is the measured knee); depth beyond that buys noise. Unlike the
pure-Python frozen head this needs torch + transformers (the ``[finetune]``
extra) and emits an adapter (MB of weights, saved alongside the artifact), not
inlinable float literals. The optimizer only reaches this rung when
``optimize=True`` and the frozen tier misses the target on the real
measurement fold.

Fold hygiene is identical to distillation: teacher/synthetic data may TEACH
(train fold); only held-out real records JUDGE (measure fold decides whether
this tier ships; calibrate fold feeds :mod:`strot.calibrate`). This module
returns the trained adapter plus the probability distributions it produces on
the measure and calibrate folds, so the existing temperature / conformal /
selective-cutoff machinery applies unchanged.

Device-agnostic: runs on GPU when ``torch.cuda.is_available()`` (compile in
seconds), else CPU (minutes) - the same code, selected at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

from .exceptions import RouterConfigurationError

#: Default fine-tune base: a standard, peft-clean encoder (not the zero-shot
#: Julia decision model, which is a self-contained predictor, not an encoder).
DEFAULT_FINETUNE_BASE = "answerdotai/ModernBERT-base"

#: Validated recipe defaults (see benchmarks/cyber_cached_top.py,
#: cyber_soup_lora.py). ``top_layers`` is the measured knee.
DEFAULT_TOP_LAYERS = 4
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_EPOCHS = 6
DEFAULT_LR = 1e-3
DEFAULT_MAX_LEN = 128

MIN_FINETUNE_RECORDS = 48  #: below this the folds cannot train a top-N adapter


#: Default ensemble width. The serving router already averages K adapters; the
#: measured +~18pp cov@90 win comes from training a small battery of diverse
#: members rather than a single adapter. Internal knob - never surfaced on the
#: public run()/decide() API; the compiler sets it when it escalates.
DEFAULT_ENSEMBLE_SIZE = 3


@dataclass
class FineTuneConfig:
    """Knobs for the fine-tune tier. Defaults are the validated recipe."""

    base_model: str = DEFAULT_FINETUNE_BASE
    top_layers: int = DEFAULT_TOP_LAYERS
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    max_length: int = DEFAULT_MAX_LEN
    seed: int = 11
    #: Number of diverse adapters to train and average (K). Each member varies
    #: the torch seed (seed, seed+1, ...); their measure/calibrate softmax rows
    #: are averaged for judging + calibration and the list of adapters is baked
    #: into the artifact (serving averages them). 1 == single-adapter behaviour.
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE


@dataclass
class FineTuneResult:
    """Everything the compiler needs to ship or reject this tier.

    ``measure_distributions`` / ``calibrate_distributions`` are plain
    ``List[List[float]]`` softmax rows aligned to ``measure``/``calibrate``
    fold order - ready for :mod:`strot.calibrate`. ``adapter`` is the trainable
    state (LoRA tensors + classifier head) as nested python floats, so it is
    serialisable without a torch dependency at load inspection time.
    """

    measure_accuracy: float
    measure_distributions: List[List[float]]
    calibrate_distributions: List[List[float]]
    adapter: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Outcome of an ensemble escalation: K members, ensemble-averaged judging.

    ``adapters`` is the list of per-member adapter dicts the artifact bakes
    (serving averages them); when ``len(adapters) == 1`` this is the plain
    single-adapter case. ``measure_distributions`` /
    ``calibrate_distributions`` are the member-averaged softmax rows, so the
    existing calibrate machinery applies to the ensemble unchanged.
    ``measure_accuracy`` is the argmax accuracy of those averaged rows.
    """

    measure_accuracy: float
    measure_distributions: List[List[float]]
    calibrate_distributions: List[List[float]]
    adapters: List[Dict[str, Any]]
    config: Dict[str, Any] = field(default_factory=dict)
    member_accuracies: List[float] = field(default_factory=list)


def _import_torch():
    try:
        import torch  # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env guard
        raise RouterConfigurationError(
            "The fine-tune tier needs the finetune extra: "
            "pip install strot-ai[finetune]"
        ) from exc


def finetune_adapter(
    texts: Sequence[str],
    labels: Sequence[int],
    n_classes: int,
    measure: Sequence[int],
    calibrate: Sequence[int],
    train: Sequence[int],
    config: Optional[FineTuneConfig] = None,
) -> FineTuneResult:
    """Train a cached-lower + top-N LoRA adapter and score the judge folds.

    ``texts``/``labels`` are the full teacher-labeled sample; ``train`` /
    ``measure`` / ``calibrate`` are disjoint index folds (see
    :func:`strot.distill.three_way_split`). Returns a :class:`FineTuneResult`.
    """
    _import_torch()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    cfg = config or FineTuneConfig()
    if len(texts) < MIN_FINETUNE_RECORDS:
        raise ValueError(
            f"need >= {MIN_FINETUNE_RECORDS} records to fine-tune; got {len(texts)}"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = snapshot_download(cfg.base_model)
    tok = AutoTokenizer.from_pretrained(path)
    enc = tok(
        list(texts), padding="max_length", truncation=True,
        max_length=cfg.max_length, return_tensors="pt",
    )
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    y = torch.tensor(list(labels), device=device)

    base = AutoModel.from_pretrained(path).to(device).eval()
    for p in base.parameters():
        p.requires_grad = False
    n_layers = len(base.layers)
    cut = max(0, n_layers - cfg.top_layers)
    dim = base.config.hidden_size

    # Cache the frozen lower-layer activations once; training only backprops
    # through the top block, so these never change.
    cache = torch.zeros(len(texts), cfg.max_length, dim, device=device)
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            out = base(input_ids=ids[i:i + 16], attention_mask=mask[i:i + 16],
                       output_hidden_states=True)
            cache[i:i + 16] = out.hidden_states[cut]
    del base

    class _Passthrough(nn.Module):
        def __init__(self, attention_type):
            super().__init__()
            self.attention_type = attention_type

        def forward(self, hidden_states, **kwargs):
            return (hidden_states,)

    class _LoRALinear(nn.Module):
        def __init__(self, inner, r, alpha):
            super().__init__()
            self.inner = inner
            for p in self.inner.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.empty(inner.in_features, r))
            self.B = nn.Parameter(torch.zeros(r, inner.out_features))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scaling = alpha / r

        def forward(self, x):
            return self.inner(x) + (x @ self.A @ self.B) * self.scaling

    def _wrap(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, _LoRALinear(child, cfg.lora_rank, cfg.lora_alpha))
            else:
                _wrap(child)

    torch.manual_seed(cfg.seed)
    model = AutoModel.from_pretrained(path).to(device)
    for p in model.parameters():
        p.requires_grad = False
    for i in range(cut):
        model.layers[i] = _Passthrough(model.layers[i].attention_type)
    for i in range(cut, n_layers):
        _wrap(model.layers[i])
    head = nn.Linear(dim, n_classes).to(device)
    inject: Dict[str, Any] = {"buf": None}

    def _hook(_module, args, kwargs):
        return ((inject["buf"],) + tuple(args[1:]), kwargs)

    handle = model.layers[cut].register_forward_pre_hook(_hook, with_kwargs=True)

    def _logits(idx):
        inject["buf"] = cache[idx]
        last = model(input_ids=ids[idx], attention_mask=mask[idx]).last_hidden_state
        m = mask[idx].unsqueeze(-1).float()
        pooled = (last * m).sum(1) / m.sum(1).clamp(min=1.0)
        return head(pooled)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.Adam(trainable, lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_idx = torch.tensor(list(train), device=device)
    model.train(); head.train()
    for ep in range(cfg.epochs):
        order = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(cfg.seed * 9 + ep))
        for i in range(0, len(train_idx), 16):
            batch = train_idx[order[i:i + 16]]
            opt.zero_grad()
            loss_fn(_logits(batch), y[batch]).backward()
            opt.step()

    model.eval(); head.eval()

    def _distributions(fold):
        rows: List[List[float]] = []
        with torch.no_grad():
            for i in range(0, len(fold), 16):
                batch = torch.tensor(list(fold[i:i + 16]), device=device)
                rows.extend(F.softmax(_logits(batch), dim=1).cpu().tolist())
        return rows

    measure_dist = _distributions(list(measure))
    calibrate_dist = _distributions(list(calibrate))
    measure_labels = [int(labels[i]) for i in measure]
    correct = sum(
        1 for row, lab in zip(measure_dist, measure_labels)
        if max(range(len(row)), key=row.__getitem__) == lab
    )
    measure_accuracy = round(correct / len(measure_labels), 4) if measure_labels else 0.0

    adapter = {
        name: p.detach().cpu().tolist()
        for name, p in model.named_parameters() if p.requires_grad
    }
    adapter["__head_weight__"] = head.weight.detach().cpu().tolist()
    adapter["__head_bias__"] = head.bias.detach().cpu().tolist()
    handle.remove()

    return FineTuneResult(
        measure_accuracy=measure_accuracy,
        measure_distributions=measure_dist,
        calibrate_distributions=calibrate_dist,
        adapter=adapter,
        config={
            "base_model": cfg.base_model, "top_layers": cfg.top_layers,
            "cut": cut, "n_layers": n_layers, "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha, "max_length": cfg.max_length,
            "device": device, "n_classes": n_classes,
        },
    )


def validate_adapter(
    texts: Sequence[str],
    labels: Sequence[int],
    n_classes: int,
    measure: Sequence[int],
    calibrate: Sequence[int],
    adapters: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    _loader: Any = None,
) -> EnsembleResult:
    """Load a pre-built adapter battery and score the judge folds; no training.

    The "find/reuse before you train" rung of the optimizer's escalation:
    instead of training a fresh adapter, load one (or an ensemble of several)
    provided in ``adapters`` via the router's :func:`strot.router.
    _load_finetuned_forward` loader, run it over the measure and calibrate
    folds, and ensemble-average the softmax rows
    (:func:`average_distributions`) - the serving behaviour. The measure-fold
    argmax accuracy decides whether the compiler adopts it (skipping training)
    or falls through to :func:`finetune_ensemble`.

    ``adapters`` is a list of adapter dicts (one = single, many = ensemble);
    ``config`` is the adapter's config dict (base_model, cut, n_layers,
    lora_rank, lora_alpha, max_length, n_classes). No gradient training
    happens. Torch-guarded like :func:`finetune_adapter`. ``_loader`` is an
    injection seam for tests: a callable with :func:`strot.router.
    _load_finetuned_forward`'s signature returning ``(tokenizer, forward)``;
    defaults to the real loader.

    Raises ``ValueError`` if ``config["n_classes"] != n_classes`` - the adapter
    predicts a different label space and cannot be adopted for this task.
    """
    if not adapters:
        raise ValueError("need >= 1 adapter to validate")
    if int(config.get("n_classes", -1)) != n_classes:
        raise ValueError(
            f"adapter is for {config.get('n_classes')} classes but this task "
            f"has {n_classes}; cannot adopt a different label space"
        )
    _import_torch()
    loader = _loader or _load_finetuned_forward_default()

    measure_idx = list(measure)
    calibrate_idx = list(calibrate)
    measure_texts = [texts[i] for i in measure_idx]
    calibrate_texts = [texts[i] for i in calibrate_idx]

    per_measure: List[List[List[float]]] = []
    per_calibrate: List[List[List[float]]] = []
    for adapter in adapters:
        _tok, forward = loader(adapter, config)
        per_measure.append(forward(measure_texts))
        per_calibrate.append(forward(calibrate_texts))

    measure_dist = average_distributions(per_measure)
    calibrate_dist = average_distributions(per_calibrate)
    measure_labels = [int(labels[i]) for i in measure_idx]
    measure_accuracy = _argmax_accuracy(measure_dist, measure_labels)

    return EnsembleResult(
        measure_accuracy=measure_accuracy,
        measure_distributions=measure_dist,
        calibrate_distributions=calibrate_dist,
        adapters=list(adapters),
        config=dict(config),
        member_accuracies=[],
    )


def _load_finetuned_forward_default() -> Any:
    """Resolve the real router loader lazily (avoids an import cycle at import
    time; the router imports nothing from this module)."""
    from .router import _load_finetuned_forward

    return _load_finetuned_forward


def average_distributions(per_member: Sequence[List[List[float]]]) -> List[List[float]]:
    """Row-wise mean of K members' softmax distributions (pure Python).

    ``per_member`` is a list of ``K`` fold scorings, each a
    ``List[List[float]]`` aligned to the same fold order. Returns the
    member-averaged distribution rows - the exact prediction ensemble the
    serving router computes, so judging and calibration see what production
    will. Empty folds (no rows) return ``[]``.
    """
    if not per_member:
        raise ValueError("need at least one member to average")
    n_rows = len(per_member[0])
    if any(len(rows) != n_rows for rows in per_member):
        raise ValueError("all members must score the same number of rows")
    if n_rows == 0:
        return []
    n = len(per_member)
    n_cls = len(per_member[0][0])
    return [
        [sum(member[r][c] for member in per_member) / n for c in range(n_cls)]
        for r in range(n_rows)
    ]


def _argmax_accuracy(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(
        1 for row, lab in zip(rows, labels)
        if max(range(len(row)), key=row.__getitem__) == lab
    )
    return round(correct / len(labels), 4)


def finetune_ensemble(
    texts: Sequence[str],
    labels: Sequence[int],
    n_classes: int,
    measure: Sequence[int],
    calibrate: Sequence[int],
    train: Sequence[int],
    config: Optional[FineTuneConfig] = None,
    _trainer: Any = None,
) -> EnsembleResult:
    """Train ``config.ensemble_size`` diverse adapters and ensemble-judge them.

    Each member reuses the full :func:`finetune_adapter` recipe but with a
    distinct torch seed (``seed``, ``seed + 1``, ...), giving diverse LoRA
    draws over the shared frozen base. The members' measure/calibrate softmax
    rows are averaged (:func:`average_distributions`) for judging and
    calibration - the serving router averages the same list of adapters at
    inference, so the compile-time judge sees production behaviour. The list of
    member adapters is returned in :attr:`EnsembleResult.adapters` for the
    artifact to bake.

    ``ensemble_size == 1`` degrades to the single-adapter case (one member, no
    averaging overhead). ``_trainer`` is an injection seam for tests: a callable
    with :func:`finetune_adapter`'s signature; defaults to the real trainer.
    """
    cfg = config or FineTuneConfig()
    if cfg.ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")
    trainer = _trainer or finetune_adapter

    members: List[FineTuneResult] = []
    for member_ix in range(cfg.ensemble_size):
        member_cfg = replace(cfg, seed=cfg.seed + member_ix)
        members.append(
            trainer(texts, labels, n_classes, measure, calibrate, train, member_cfg)
        )

    measure_dist = average_distributions([m.measure_distributions for m in members])
    calibrate_dist = average_distributions([m.calibrate_distributions for m in members])
    measure_labels = [int(labels[i]) for i in measure]
    measure_accuracy = _argmax_accuracy(measure_dist, measure_labels)

    base_config = dict(members[0].config)
    base_config["ensemble_size"] = cfg.ensemble_size
    return EnsembleResult(
        measure_accuracy=measure_accuracy,
        measure_distributions=measure_dist,
        calibrate_distributions=calibrate_dist,
        adapters=[m.adapter for m in members],
        config=base_config,
        member_accuracies=[m.measure_accuracy for m in members],
    )
