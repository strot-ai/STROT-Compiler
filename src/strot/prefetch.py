"""Build-time provisioning for the local NLU runtime.

The Julia decision model's inference package (``supersonic-julia``) is not on
PyPI — it is distributed inside the model's Hugging Face repository, so it
cannot be a declared dependency of ``strot-ai``. :func:`prefetch_nlu` is the
paved path: call it once at image-build time to bake the checkpoint (and
optionally its package) into the deploy artifact so production boots
instantly and fully offline.

Dockerfile recipe::

    RUN pip install strot-ai[local] \\
     && python -c "import strot; strot.prefetch_nlu(local_dir='/opt/julia-1', install=True)"
    ENV STROT_NLU_CHECKPOINT=/opt/julia-1
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

from .constants import DEFAULT_NLU_MODEL
from .exceptions import RouterConfigurationError


def prefetch_nlu(
    model: str = DEFAULT_NLU_MODEL,
    local_dir: Optional[str] = None,
    install: bool = False,
) -> str:
    """Download the NLU checkpoint; optionally pip-install its inference package.

    Args:
        model: Hugging Face repo id of the decision model.
        local_dir: Target directory (defaults to the Hugging Face cache).
        install: If True, ``pip install`` the downloaded repository, which
            contains the model's own inference package (``supersonic-julia``
            for Julia checkpoints). Never ``pip install julia`` — that is an
            unrelated PyPI package.

    Returns:
        The local checkpoint directory, suitable for ``STROT_NLU_CHECKPOINT``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RouterConfigurationError(
            "prefetch_nlu requires huggingface_hub: pip install strot-ai[local]"
        ) from exc
    path = snapshot_download(model, local_dir=local_dir)
    if install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", path])
    return path
