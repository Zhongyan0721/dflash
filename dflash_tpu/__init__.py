"""DFlash TPU plugin for vLLM / tpu-inference.

Registers the DFlash draft model so it can be loaded through
the standard tpu-inference model registry.
"""

from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model

logger = init_logger(__name__)


def register():
    """Entry-point called by vLLM's plugin system."""
    from .dflash_jax import DFlashForCausalLM

    register_model("DFlashDraftModel", DFlashForCausalLM)
    logger.info("Registered DFlashDraftModel with tpu-inference model registry.")
