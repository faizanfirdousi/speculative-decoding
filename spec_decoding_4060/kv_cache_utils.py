from transformers.cache_utils import DynamicCache

def crop_past_key_values(past_key_values: DynamicCache, max_length: int) -> DynamicCache:
    """
    Truncates a Hugging Face DynamicCache in-place to `max_length` tokens.
    Essential for rolling back rejected draft tokens without recomputing the prompt.
    """
    if past_key_values is None:
        return None
    
    # transformers >= 4.40 supports native .crop() on DynamicCache
    if hasattr(past_key_values, "crop"):
        curr_len = past_key_values.get_seq_length()
        if curr_len > max_length:
            try:
                past_key_values.crop(-(curr_len - max_length))
            except Exception:
                past_key_values.crop(max_length)
        return past_key_values
    
    # Fallback manual slicing for older DynamicCache versions
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :max_length, :]
        past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :max_length, :]
    if hasattr(past_key_values, "_seen_tokens"):
        past_key_values._seen_tokens = max_length
    return past_key_values
