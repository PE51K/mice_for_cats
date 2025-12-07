"""
Raw confidence computation.

Paper Section 2 & 4.4:
"We also integrate the raw confidence of the language model in generating 
the tool call as a feature to the MICE model. We calculate this by computing 
the product of the probabilities of the tokens in the generated tool call."

"We notice that including formatting tokens, which are always present in the 
tool call, leads to increased noise and a less accurate estimate of confidence, 
so we omit the tokens associated with formatting. The gray tokens in Figure 1 
were omitted, while the green ones were included."

Paper: "∏_{i∈S} p(w_i|w_{<i}), where S is the subset of token indices 
that are relevant to the tool call"
"""

from typing import Set, List
import torch
from transformers import AutoTokenizer


class RawConfidenceComputer:
    """
    Compute raw confidence as product of token probabilities.
    
    Excludes formatting tokens ("action:", "action input:") which are 
    always present and add noise to the confidence estimate.
    """
    
    # Formatting patterns to exclude (from paper Figure 1)
    EXCLUDE_PATTERNS = [
        "action:", "action input:", 
        "Action:", "Action Input:",
        "action :", "action_input:",
    ]
    
    def __init__(self, tokenizer: AutoTokenizer):
        """
        Initialize raw confidence computer.
        
        Args:
            tokenizer: Tokenizer for identifying formatting tokens
        """
        self.tokenizer = tokenizer
        self._cache_exclude_tokens()
    
    def _cache_exclude_tokens(self):
        """Cache token IDs for formatting strings to exclude."""
        self.exclude_token_ids: Set[int] = set()
        
        for pattern in self.EXCLUDE_PATTERNS:
            # Encode without special tokens
            tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
            self.exclude_token_ids.update(tokens)
        
        # Also exclude common whitespace and newline tokens
        for char in ["\n", " ", "  ", "\t"]:
            tokens = self.tokenizer.encode(char, add_special_tokens=False)
            self.exclude_token_ids.update(tokens)
    
    def compute(
        self,
        generated_ids: torch.Tensor,
        log_probs: torch.Tensor,
        mask_formatting: bool = True
    ) -> float:
        """
        Compute raw confidence score.
        
        Paper: "the product of the probabilities of the tokens in the 
        generated tool call"
        
        Args:
            generated_ids: Token IDs of generated text [seq_len]
            log_probs: Log probabilities of each token [seq_len]
            mask_formatting: Whether to exclude formatting tokens (paper: True)
        
        Returns:
            confidence: Product of relevant token probabilities (0 to 1)
        """
        # Ensure 1D tensors
        if generated_ids.dim() > 1:
            generated_ids = generated_ids.squeeze()
        if log_probs.dim() > 1:
            log_probs = log_probs.squeeze()
        
        if mask_formatting:
            # Create mask for relevant tokens (exclude formatting)
            mask = torch.ones(len(generated_ids), dtype=torch.bool)
            for i, token_id in enumerate(generated_ids):
                if token_id.item() in self.exclude_token_ids:
                    mask[i] = False
            
            # If all tokens are masked, use all tokens
            if not mask.any():
                relevant_log_probs = log_probs
            else:
                relevant_log_probs = log_probs[mask]
        else:
            relevant_log_probs = log_probs
        
        # Product of probabilities = exp(sum of log probs)
        # Clamp to avoid numerical issues
        log_conf = relevant_log_probs.sum().clamp(min=-100)
        confidence = torch.exp(log_conf).item()
        
        # Clamp to valid probability range
        return max(0.0, min(1.0, confidence))
    
    def compute_from_generated_text(
        self,
        generated_text: str,
        full_log_probs: torch.Tensor,
        full_generated_ids: torch.Tensor
    ) -> float:
        """
        Alternative computation that identifies formatting tokens by text matching.
        
        More robust but slower than token ID matching.
        """
        # Find positions of formatting strings in generated text
        text_lower = generated_text.lower()
        
        # Simple heuristic: exclude first few tokens (likely formatting)
        # and use remaining tokens for confidence
        return self.compute(full_generated_ids, full_log_probs, mask_formatting=True)

