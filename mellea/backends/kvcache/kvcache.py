from collections import defaultdict
import torch

from transformers import (
    DynamicCache,
)

SplitCache = defaultdict[int, defaultdict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]]

def split(cache: DynamicCache) -> SplitCache:
    """
    Split a DynamicCache into token-by-token entries.
    """
    results : SplitCache = defaultdict(lambda: defaultdict(lambda : dict()))

    for layer_idx in range(len(cache)): # type:ignore
        # Each layer is a tensor (batch, num_heads, seq_len, head_dim)
        k : torch.Tensor = cache[layer_idx][0] # type:ignore
        v : torch.Tensor = cache[layer_idx][1] # type:ignore

        for batch_idx in range(k.size(0)):
            for token_idx in range(k.size(2)):
                results[batch_idx][token_idx][layer_idx] = \
                    (k[batch_idx, :, token_idx, :],
                     v[batch_idx, :, token_idx, :])

    return results

