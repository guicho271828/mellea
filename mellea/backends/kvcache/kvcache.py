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

