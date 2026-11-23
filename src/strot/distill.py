"""Compile-time distillation: fine-tune a tiny head on the frozen encoder.

The LLM teacher labels real sample records once; a logistic head is trained
on the decision model's own frozen embeddings (seconds, CPU) and baked into
the artifact as plain weight literals. Mechanically a fine-tune; relationally
a distillation - the LLM's judgment compressed into the local model.

Fold hygiene is non-negotiable and enforced by :func:`three_way_split`:
synthetic-or-teacher data may TEACH (train fold), but only held-out real
records may JUDGE (measurement fold for model search, calibration fold for
the conformal fit). The distilled candidate ships only if it beats the
zero-shot router on the measurement fold.

The trainer is pure Python (stdlib) so the core package stays dependency
free; embeddings come from a pluggable ``embed_fn`` (production: the local
checkpoint's encoder via torch/transformers; tests: any deterministic fake).
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .constants import NLU_MAX_LENGTH
from .exceptions import RouterConfigurationError

#: Maps texts to fixed-size float vectors.
EmbedFn = Callable[[List[str]], List[List[float]]]

MIN_DISTILL_RECORDS = 24  #: below this, folds are too small to train or judge


def make_encoder_embedder(checkpoint: str, batch_size: int = 16) -> EmbedFn:
    """Production embedder: mean-pooled last hidden state of the frozen encoder.

    Handles two checkpoint layouts:

    * **Standard HuggingFace encoder** (the default base, e.g. a ModernBERT) —
      a hub id or local dir that ``AutoModel.from_pretrained`` loads directly.
    * **Wrapped decision model** (a self-contained predictor whose encoder is
      stored under an ``encoder/`` config dir with ``"encoder."``-prefixed
      tensors in the root ``model.safetensors``) — the encoder is extracted and
      loaded on its own.
    """
    try:
        import os

        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RouterConfigurationError(
            "Distillation needs the local runtime deps: pip install strot-ai[local]"
        ) from exc

    if os.path.isdir(os.path.join(str(checkpoint), "encoder")):
        # Wrapped decision model: pull the encoder out of the composite weights.
        from safetensors.torch import load_file
        from transformers import AutoConfig

        tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}/tokenizer")
        config = AutoConfig.from_pretrained(f"{checkpoint}/encoder")
        model = AutoModel.from_config(config)
        state = load_file(f"{checkpoint}/model.safetensors")
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise RouterConfigurationError(
                "Checkpoint has an encoder/ dir but no 'encoder.'-prefixed "
                "tensors; cannot build the distillation embedder from it."
            )
        model.load_state_dict(encoder_state, strict=False)
    else:
        # Standard HF encoder (the base STROT compiles on).
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    model.eval()

    def embed(texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=NLU_MAX_LENGTH,
            )
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            vectors.extend(pooled.tolist())
        return vectors

    return embed


def three_way_split(indices: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic split: ~50% train, ~25% measure, ~25% calibrate.

    Indices are shuffled with a fixed seed first - a plain interleave can
    align with periodicity in ordered datasets (e.g. alternating classes)
    and produce single-class folds.
    """
    shuffled = list(indices)
    random.Random(29).shuffle(shuffled)
    train = [i for pos, i in enumerate(shuffled) if pos % 4 in (0, 1)]
    measure = [i for pos, i in enumerate(shuffled) if pos % 4 == 2]
    calibrate = [i for pos, i in enumerate(shuffled) if pos % 4 == 3]
    return train, measure, calibrate


def train_head(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_classes: int,
    epochs: int = 300,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
    seed: int = 13,
) -> Tuple[List[List[float]], List[float]]:
    """Multinomial logistic head via plain gradient descent (pure Python).

    Returns (weights[n_classes][dim], bias[n_classes]). Deterministic given
    the seed. Small-data regime by design: hundreds of examples, seconds.

    Fresh random init every call (seeded) - part of decide's recompile-don't-
    increment invariant (see ``StrotCompiler.compile_decide``). Never add an
    ``init_weights`` / warm-start parameter that continues from previously
    trained weights; that would reintroduce catastrophic-forgetting risk.
    """
    if not embeddings or len(embeddings) != len(labels):
        raise ValueError("need equal, non-empty embeddings and labels")
    dim = len(embeddings[0])
    rng = random.Random(seed)
    weights = [[rng.gauss(0.0, 0.01) for _ in range(dim)] for _ in range(n_classes)]
    bias = [0.0] * n_classes
    n = len(embeddings)
    for _ in range(epochs):
        grad_w = [[0.0] * dim for _ in range(n_classes)]
        grad_b = [0.0] * n_classes
        for vector, true_class in zip(embeddings, labels):
            logits = [
                sum(w * x for w, x in zip(weights[c], vector)) + bias[c]
                for c in range(n_classes)
            ]
            peak = max(logits)
            exps = [math.exp(z - peak) for z in logits]
            total = sum(exps)
            probs = [e / total for e in exps]
            for c in range(n_classes):
                delta = probs[c] - (1.0 if c == true_class else 0.0)
                grad_b[c] += delta
                row = grad_w[c]
                for d, x in enumerate(vector):
                    row[d] += delta * x
        for c in range(n_classes):
            bias[c] -= learning_rate * grad_b[c] / n
            row_w, row_g = weights[c], grad_w[c]
            for d in range(dim):
                row_w[d] -= learning_rate * (row_g[d] / n + l2 * row_w[d])
    return (
        [[round(w, 6) for w in row] for row in weights],
        [round(b, 6) for b in bias],
    )


def head_predict(
    vector: Sequence[float],
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
) -> List[float]:
    """Softmax probabilities of the trained head for one embedding."""
    logits = [
        sum(w * x for w, x in zip(row, vector)) + b
        for row, b in zip(weights, bias)
    ]
    peak = max(logits)
    exps = [math.exp(z - peak) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def head_accuracy(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
) -> float:
    correct = 0
    for vector, true_class in zip(embeddings, labels):
        probs = head_predict(vector, weights, bias)
        if max(range(len(probs)), key=probs.__getitem__) == true_class:
            correct += 1
    return round(correct / len(labels), 4) if labels else 0.0


# --- Drift / out-of-distribution guard (pure Python, baked as float literals) ---
# A decision model should refuse records unlike anything it was trained on rather
# than answer them confidently. These fit per-class centroids + a distance
# threshold on the training embeddings; the runtime refuses records farther than
# the threshold from every centroid. (Lives here, not in the router, so the
# decide path owns it independently of the routing stack.)


def class_centroids(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_classes: int,
) -> List[List[float]]:
    """Mean embedding per class. A class with no vectors gets a zero centroid."""
    if not vectors:
        raise ValueError("need at least one vector to form centroids")
    dim = len(vectors[0])
    sums = [[0.0] * dim for _ in range(n_classes)]
    counts = [0] * n_classes
    for vector, label in zip(vectors, labels):
        row = sums[label]
        for d, x in enumerate(vector):
            row[d] += x
        counts[label] += 1
    return [
        [round(s / counts[c], 6) for s in sums[c]] if counts[c] else [0.0] * dim
        for c in range(n_classes)
    ]


def nearest_centroid_distance(
    vector: Sequence[float], centroids: Sequence[Sequence[float]]
) -> float:
    """Euclidean distance from ``vector`` to its closest class centroid."""
    if not centroids:
        raise ValueError("need at least one centroid")
    return min(
        math.sqrt(sum((x - c) * (x - c) for x, c in zip(vector, cen)))
        for cen in centroids
    )


def ood_threshold(
    vectors: Sequence[Sequence[float]],
    centroids: Sequence[Sequence[float]],
    percentile: float = 0.99,
) -> float:
    """A high percentile of in-distribution nearest-centroid distances.

    Records farther than this at runtime are flagged out-of-distribution. The
    default is conservative (0.99) so the always-on decide gate rarely flags an
    in-distribution record while still catching records beyond the envelope.
    """
    if not vectors:
        raise ValueError("need vectors to fit an OOD threshold")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    distances = sorted(nearest_centroid_distance(v, centroids) for v in vectors)
    rank = min(len(distances) - 1, max(0, math.ceil(percentile * len(distances)) - 1))
    return round(distances[rank], 6)
