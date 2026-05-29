"""Loader for Flash Attention symbols used across the GO-1 model code.

Building ``flash-attn`` from source is slow. To avoid that, we first try to load
pre-built Flash Attention 2 binaries served through the Hugging Face ``kernels``
library. When pre-built binaries are not available for the current platform
(unsupported GPU/arch, no network, ...), we transparently fall back to a source
build of the ``flash-attn`` package, so the dependency stays optional.

See https://github.com/OpenDriveLab/AgiBot-World/issues/158 for context.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Kernel served by the `kernels` library that mirrors the `flash-attn` v2 API.
_FLASH_ATTN_KERNEL = "kernels-community/flash-attn2"
_FLASH_ATTN_KERNEL_VERSION = 1


@lru_cache(maxsize=None)
def load_flash_attn():
    """Return a dict of Flash Attention symbols, or an empty dict if unavailable.

    The returned dict exposes the same callables the model code relies on:
    ``flash_attn_func``, ``flash_attn_varlen_func``,
    ``flash_attn_varlen_qkvpacked_func``, ``pad_input``, ``unpad_input`` and
    ``index_first_axis``.

    Resolution order:
      1. Pre-built FA2 binaries via the ``kernels`` library (no compilation).
      2. A source build of the ``flash-attn`` package (the original behaviour).
    """
    symbols = _load_from_kernels()
    if symbols is not None:
        return symbols

    symbols = _load_from_source()
    if symbols is not None:
        return symbols

    return {}


def _load_from_kernels():
    try:
        from kernels import get_kernel

        module = get_kernel(_FLASH_ATTN_KERNEL, version=_FLASH_ATTN_KERNEL_VERSION)
        interface = module.flash_attention_interface
        bert_padding = module.bert_padding
        logger.info("Loaded pre-built Flash Attention 2 binaries via `kernels` (%s).", _FLASH_ATTN_KERNEL)
        return {
            "flash_attn_func": module.flash_attn_func,
            "flash_attn_varlen_func": module.flash_attn_varlen_func,
            "flash_attn_varlen_qkvpacked_func": module.flash_attn_varlen_qkvpacked_func,
            "pad_input": bert_padding.pad_input,
            "unpad_input": bert_padding.unpad_input,
            "index_first_axis": bert_padding.index_first_axis,
        }
    except Exception as exc:  # noqa: BLE001 - any failure should trigger the source fallback
        logger.info(
            "Pre-built Flash Attention via `kernels` unavailable (%s); falling back to a source build.", exc
        )
        return None


def _load_from_source():
    try:
        from flash_attn import (
            flash_attn_func,
            flash_attn_varlen_func,
            flash_attn_varlen_qkvpacked_func,
        )
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

        return {
            "flash_attn_func": flash_attn_func,
            "flash_attn_varlen_func": flash_attn_varlen_func,
            "flash_attn_varlen_qkvpacked_func": flash_attn_varlen_qkvpacked_func,
            "pad_input": pad_input,
            "unpad_input": unpad_input,
            "index_first_axis": index_first_axis,
        }
    except ImportError:
        return None
