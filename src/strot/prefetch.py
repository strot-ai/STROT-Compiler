"""Build-time provisioning for the local encoder.

The frozen base encoder is downloaded from Hugging Face on first use.
:func:`prefetch_nlu` is the paved path for deployments: call it once at
image-build time to bake the encoder into the artifact so a compiled decision
model boots instantly and fully offline.

Dockerfile recipe::

    RUN pip install strot-ai[local] \\
     && python -c "import strot; strot.prefetch_nlu(local_dir='/opt/encoder')"
    ENV STROT_NLU_CHECKPOINT=/opt/encoder
"""

from __future__ import annotations

from typing import Optional

from .constants import DEFAULT_NLU_MODEL
from .exceptions import RouterConfigurationError


def prefetch_nlu(
    model: str = DEFAULT_NLU_MODEL,
    local_dir: Optional[str] = None,
) -> str:
    """Download the frozen base encoder checkpoint.

    Args:
        model: Hugging Face repo id of the base encoder (defaults to the STROT
            base; not normally a caller choice).
        local_dir: Target directory (defaults to the Hugging Face cache).

    Returns:
        The local checkpoint directory, suitable for ``STROT_NLU_CHECKPOINT``.
        The base is a standard Hugging Face encoder, so nothing is pip-installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RouterConfigurationError(
            "prefetch_nlu requires huggingface_hub: pip install strot-ai[local]"
        ) from exc
    return snapshot_download(model, local_dir=local_dir)
