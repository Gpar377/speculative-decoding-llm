import torch
import torch.nn.functional as F
from .kv_cache import KVCacheManager

class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, tokenizer):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        
        # Configure evaluation modes
        self.target_model.eval()
        self.draft_model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128, lookahead_k: int = 4, temperature: float = 1.0) -> str:
        """
        Executes speculative decoding generation.
        lookahead_k: Number of draft tokens proposed per loop iteration.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.target_model.device)
        
        # Initialize KV Cache Managers for both models
        target_cache = KVCacheManager()
        draft_cache = KVCacheManager()

        # Run initial forward pass on prompt to fill initial caches
        target_outputs = self.target_model(input_ids, past_key_values=None, use_cache=True)
        target_cache.update(target_outputs.past_key_values)
        
        draft_outputs = self.draft_model(input_ids, past_key_values=None, use_cache=True)
        draft_cache.update(draft_outputs.past_key_values)

        # Sample initial token from target output distribution
        next_token = torch.argmax(target_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        generated_count = 1
        
        while generated_count < max_new_tokens:
            prefix_len = input_ids.shape[1]
            
            # ----------------- Step 1: Draft Model Lookahead Generation -----------------
            draft_input = next_token
            draft_proposed_tokens = []
            draft_probs = []

            for _ in range(lookahead_k):
                outputs = self.draft_model(draft_input, past_key_values=draft_cache.past_key_values, use_cache=True)
                draft_cache.update(outputs.past_key_values)
                
                # Calculate probability distribution
                logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                sampled_token = torch.multinomial(probs, num_samples=1)
                draft_proposed_tokens.append(sampled_token)
                draft_probs.append(probs.gather(dim=-1, index=sampled_token))
                
                draft_input = sampled_token

            draft_proposed_tokens = torch.cat(draft_proposed_tokens, dim=-1) # Shape: (1, lookahead_k)

            # ----------------- Step 2: Target Model Parallel Validation -----------------
            # Target processes all proposed tokens in a single parallel step
            target_input = torch.cat([next_token, draft_proposed_tokens[:, :-1]], dim=-1)
            target_outputs = self.target_model(target_input, past_key_values=target_cache.past_key_values, use_cache=True)
            target_cache.update(target_outputs.past_key_values)

            target_logits = target_outputs.logits / max(temperature, 1e-5)
            target_probs = F.softmax(target_logits, dim=-1) # Shape: (1, lookahead_k, vocab_size)

            # ----------------- Step 3: Acceptance Check (Speculative Sampling) -----------------
            accepted_count = 0
            
            for idx in range(lookahead_k):
                p_x = target_probs[:, idx, draft_proposed_tokens[0, idx]] # Target probability
                q_x = draft_probs[idx] # Draft probability
                
                # Speculative rejection math:
                # Accept if target prob >= draft prob, else accept with probability ratio p_x / q_x
                r = torch.rand(1, device=self.target_model.device)
                if r < torch.minimum(torch.tensor(1.0, device=self.target_model.device), p_x / q_x):
                    accepted_count += 1
                    next_token = draft_proposed_tokens[:, idx:idx+1]
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                else:
                    # Token rejected! Adjust target distribution to select a corrected replacement token
                    rescale_probs = target_probs[:, idx, :] - draft_probs[idx]
                    rescale_probs = torch.clamp(rescale_probs, min=0.0)
                    rescale_probs = rescale_probs / rescale_probs.sum(dim=-1, keepdim=True)
                    
                    next_token = torch.multinomial(rescale_probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    break

            generated_count += accepted_count + 1

            # ----------------- Step 4: KV Cache Rollback & Sync -----------------
            # Determine correct sequence length to restore
            correct_len = prefix_len + accepted_count
            target_cache.truncate(correct_len)
            
            # Align draft cache with correct length
            draft_cache.truncate(correct_len)
            
            # Feed accepted prefix through draft model to align cache representation
            # This aligns the draft KV cache with the target accepted token trajectory
            _ = self.draft_model(input_ids[:, -1:], past_key_values=draft_cache.past_key_values, use_cache=True)
            draft_cache.update(_.past_key_values)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
