"""Setup script for the DFlash TPU plugin.

Install with:
    pip install -e .

This registers the dflash_tpu package as a vLLM plugin so that the
DFlashForCausalLM architecture is available when serving with vLLM on TPU.
"""

from setuptools import setup, find_packages

setup(
    name="dflash_tpu",
    version="0.1.0",
    description="DFlash block-diffusion draft model plugin for vLLM on TPU",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # vllm and tpu_inference are expected to be pre-installed in the env
    ],
    entry_points={
        "vllm.general_plugins": [
            "dflash_tpu = dflash_tpu:register",
        ],
    },
)
