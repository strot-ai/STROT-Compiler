"""Shared constants with no heavy dependencies.

Kept in a tiny leaf module so the ``decide`` path can import them without pulling
in the routing stack (``router.py``). The base encoder is a STROT decision, not a
caller choice - there is no public model-selection surface.
"""

#: The frozen base encoder every decision model compiles on (validated on
#: Banking77: a head on this reaches full-fine-tune SOTA). Not caller-selectable.
DEFAULT_NLU_MODEL = "Alibaba-NLP/gte-modernbert-base"

#: Default vision encoder for image-modality compiles (head tier only). Swappable
#: via compile(..., encoder=...); any SigLIP / ViT / DINOv2 checkpoint works.
DEFAULT_IMAGE_ENCODER = "google/siglip-base-patch16-224"

#: Token window every tier uses, so `head` and `deep` always see the SAME text
#: (ModernBERT supports far longer; this is the STROT default). Shared here so the
#: tiers can never diverge - `deep` seeing less text than `head` is not a fair
#: superset.
NLU_MAX_LENGTH = 256

#: Default confidence floor below which a routed decision would abstain. decide
#: exposes its own opt-in ``confidence_threshold`` instead; this stays as the
#: baked default for compiled artifacts.
CONFIDENCE_THRESHOLD = 0.40
