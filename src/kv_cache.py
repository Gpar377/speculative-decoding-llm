import torch

class KVCacheManager:
    def __init__(self):
        # Cache is stored as a list of tuples containing (key_states, value_states) per transformer layer
        self.past_key_values = None

    def update(self, new_past_key_values):
        """
        Updates the manager with the newly generated past_key_values.
        """
        self.past_key_values = new_past_key_values

    def truncate(self, length: int):
        """
        Truncates the KV cache along the sequence length dimension (dim = 2).
        This is invoked when speculative tokens proposed by the draft model are rejected.
        """
        if self.past_key_values is None:
            return

        truncated_past = []
        for layer_past in self.past_key_values:
            # HuggingFace standard past_key_values structure:
            # layer_past[0] shape: (batch_size, num_heads, seq_len, head_dim)
            # layer_past[1] shape: (batch_size, num_heads, seq_len, head_dim)
            key_states, value_states = layer_past[0], layer_past[1]
            
            # Truncate sequence length dimension (dimension index 2)
            truncated_k = key_states[:, :, :length, :]
            truncated_v = value_states[:, :, :length, :]
            
            truncated_past.append((truncated_k, truncated_v))
            
        self.past_key_values = tuple(truncated_past)

    def get_length(self) -> int:
        """
        Returns the current sequence length in the cache.
        """
        if self.past_key_values is None:
            return 0
        # Check sequence length of the key states in the first layer
        return self.past_key_values[0][0].shape[2]

    def clear(self):
        self.past_key_values = None
