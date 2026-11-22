"""Unit tests for the pure-Python distillation trainer and compile integration."""

import math
import os
import random
from types import SimpleNamespace

import pytest

from strot.distill import (
    head_accuracy,
    head_predict,
    three_way_split,
    train_head,
    make_encoder_embedder,
)
def test_encoder_embedder_mean_pools(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")

    class Encoder:
        def load_state_dict(self, *args, **kwargs):
            pass

        def eval(self):
            pass

        def __call__(self, attention_mask):
            hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
            return SimpleNamespace(last_hidden_state=hidden.expand(attention_mask.shape[0], -1, -1))

    encoder = Encoder()
    tokenizer = lambda texts, **kwargs: {"attention_mask": torch.ones(len(texts), 2)}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda _: None)
    monkeypatch.setattr(transformers.AutoModel, "from_config", lambda _: encoder)
    monkeypatch.setattr(safetensors, "load_file", lambda _: {"encoder.x": torch.zeros(1)})
    monkeypatch.setenv("STROT_NLU_CHECKPOINT", str(tmp_path))
    assert make_encoder_embedder(str(tmp_path))(["one", "two"]) == [[2.0, 0.0]] * 2


class TestTrainer:
    def _separable(self, n=80, dim=8, seed=3):
        rng = random.Random(seed)
        embeddings, labels = [], []
        for _ in range(n):
            label = rng.randrange(2)
            center = 1.0 if label else -1.0
            embeddings.append([center + rng.gauss(0, 0.3) for _ in range(dim)])
            labels.append(label)
        return embeddings, labels

    def test_learns_separable_data(self):
        embeddings, labels = self._separable()
        weights, bias = train_head(embeddings, labels, n_classes=2)
        assert head_accuracy(embeddings, labels, weights, bias) >= 0.95

    def test_generalizes_to_fresh_points(self):
        train_e, train_y = self._separable(seed=3)
        test_e, test_y = self._separable(seed=99)
        weights, bias = train_head(train_e, train_y, n_classes=2)
        assert head_accuracy(test_e, test_y, weights, bias) >= 0.9

    def test_deterministic(self):
        embeddings, labels = self._separable()
        assert train_head(embeddings, labels, 2) == train_head(embeddings, labels, 2)

    def test_predict_is_distribution(self):
        embeddings, labels = self._separable(n=20)
        weights, bias = train_head(embeddings, labels, 2)
        probs = head_predict(embeddings[0], weights, bias)
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_three_way_split_disjoint_and_complete(self):
        train, measure, calibrate = three_way_split(range(100))
        assert not (set(train) & set(measure))
        assert not (set(train) & set(calibrate))
        assert not (set(measure) & set(calibrate))
        assert sorted(train + measure + calibrate) == list(range(100))
        assert len(train) == 50 and len(measure) == 25 and len(calibrate) == 25

    def test_guards(self):
        with pytest.raises(ValueError):
            train_head([], [], 2)
        with pytest.raises(ValueError):
            train_head([[1.0]], [0, 1], 2)
