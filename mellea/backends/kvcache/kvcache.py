from collections import defaultdict
import torch

from transformers import (
    DynamicCache,
)

from mellea.backends.kvcache.trie import RadixTrie

TokenCache = dict[int, tuple[torch.Tensor, torch.Tensor]]   # layer index -> (key, value) tensors
SplitCache = defaultdict[int, defaultdict[int, TokenCache]] # batch index, token index -> TokenCache

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


def unsplit(cache: SplitCache) -> DynamicCache:
    """
    Merge a nested dict (produced by `split`) back into a DynamicCache.
    """

    result = DynamicCache()

    layers = dict()
    batch_size = len(cache)
    for batch_idx, batch in cache.items():
        seq_len = len(batch)
        for token_idx, layers_per_token in batch.items():
            for layer_idx, (k, v) in layers_per_token.items():
                k_num_heads, k_head_dim = k.shape
                v_num_heads, v_head_dim = v.shape
                if layer_idx not in layers:
                    layers[layer_idx] = (
                        torch.zeros(batch_size, k_num_heads, seq_len, k_head_dim, dtype=k.dtype, device=k.device),
                        torch.zeros(batch_size, v_num_heads, seq_len, v_head_dim, dtype=v.dtype, device=v.device),
                    )
                layers[layer_idx][0][batch_idx, :, token_idx, :] = k
                layers[layer_idx][1][batch_idx, :, token_idx, :] = v

    for layer_idx in range(len(layers)):
        result.update(layers[layer_idx][0], layers[layer_idx][1], layer_idx) # type:ignore

    return result


def to_trie(batch_tokens: torch.Tensor, cache: SplitCache | DynamicCache) -> RadixTrie[int,TokenCache]:

    if isinstance(cache, DynamicCache):
        cache : SplitCache = split(cache)

    assert batch_tokens.ndim == 2

    trie = RadixTrie[int,TokenCache]()
    layers = dict()
    batch_size = len(cache)
    for batch_idx, batch in cache.items():
        seq_len = len(batch)
        # batch is a defaultdict; converting it into a list
        contents : list[TokenCache] = [ batch[token_idx] for token_idx in range(seq_len) ]
        keys     : list[int]        = batch_tokens[batch_idx].tolist()
        trie[keys] = contents

    return trie


if __name__ == "__main__":

    import torch
    torch.set_printoptions(edgeitems=1, threshold=5)

    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto")
    inputs = tokenizer(["I like music, especially the country music from",
                        "I like music, especially the progressive rock from"], return_tensors="pt").to(model.device)

    past_key_values = DynamicCache()
    out = model.generate(**inputs, do_sample=False, max_new_tokens=20, past_key_values=past_key_values)
    print(tokenizer.batch_decode(out,skip_special_tokens=True))

    split_cache = split(past_key_values)

    past_key_values2 = unsplit(split_cache)

    for (k1, v1), (k2, v2) in zip(past_key_values, past_key_values2):
        assert torch.allclose(k1, k2)
        assert torch.allclose(v1, v2)


