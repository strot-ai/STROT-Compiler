"""Shared constants with no heavy dependencies.

Kept in a tiny leaf module so the ``decide`` path can import them without pulling
in the routing stack (``router.py``). The base encoder is a STROT decision, not a
caller choice - there is no public model-selection surface.
"""

#: The frozen base encoder every decision model compiles on (validated on
#: Banking77: a head on this reaches full-fine-tune SOTA). Not caller-selectable.
DEFAULT_NLU_MODEL = "Alibaba-NLP/gte-modernbert-base"

#: Default confidence floor below which a routed decision would abstain. decide
#: exposes its own opt-in ``confidence_threshold`` instead; this stays as the
#: baked default for compiled artifacts.
CONFIDENCE_THRESHOLD = 0.40
