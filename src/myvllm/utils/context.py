from dataclasses import dataclass 
import torch 


@dataclass
class Context:
    is_prefill: bool = False
    """Whether the current forward pass is the prefill phase."""

    cu_seqlens_q: torch.Tensor | None = None
    """Cumulative sequence lengths for query, used by flash attention."""

    cu_seqlens_k: torch.Tensor | None = None
    """Cumulative sequence lengths for key, used by flash attention."""

    max_seqlen_q: int = 0
    """Maximum sequence length among queries in the batch."""

    max_seqlen_k: int = 0
    """Maximum sequence length among keys in the batch."""

    slot_mapping: torch.Tensor | None = None
    """Maps each token position to its KV cache slot index."""

    context_lens: torch.Tensor | None = None
    """Length of the context (prompt + generated tokens) for each sequence."""

    block_tables: torch.Tensor | None = None
    """Block tables for PagedAttention, mapping logical blocks to physical blocks."""

_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
