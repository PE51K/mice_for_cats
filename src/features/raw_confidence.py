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

Paper Section 4.4:
"S omits the tokens associated with formatting ('action:' and 'action input:',
which are generated for every tool call), and also omits tokens that are
generated after the arguments of the tool call."

Paper: "∏_{i∈S} p(w_i|w_{<i}), where S is the subset of token indices
that are relevant to the tool call"
"""

from typing import Optional, Set

import torch
from transformers import AutoTokenizer


class RawConfidenceComputer:
    """
    Compute raw confidence as product of token probabilities.

    Excludes:
    1. Formatting tokens ("action:", "action input:") which are always present
    2. Tokens after the tool call arguments (post-argument explanations)
    """

    # Formatting patterns to exclude (from paper Figure 1)
    EXCLUDE_PATTERNS = [
        "action:",
        "action input:",
        "Action:",
        "Action Input:",
        "action :",
        "action_input:",
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

    def _find_tool_call_end(self, generated_text: str) -> Optional[int]:
        """
        Find the character position where the tool call ends.

        Paper: "omits tokens that are generated after the arguments of the tool call"

        Returns character index of the end of the JSON action_input, or None if not found.
        """
        text_lower = generated_text.lower()

        # Find where action_input starts
        start_idx = -1
        if "action input:" in text_lower:
            start_idx = text_lower.find("action input:") + 13
        elif "action_input:" in text_lower:
            start_idx = text_lower.find("action_input:") + 13

        if start_idx == -1:
            return None

        rest = generated_text[start_idx:].strip()

        # Find the JSON object - look for balanced braces
        if not rest.startswith("{"):
            json_start = rest.find("{")
            if json_start == -1:
                return None
            rest = rest[json_start:]
            start_idx += json_start

        # Extract balanced JSON object
        brace_count = 0
        json_end = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(rest):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end > 0:
            # Return character position where tool call ends
            return start_idx + json_end

        return None

    def compute(
        self,
        generated_ids: torch.Tensor,
        log_probs: torch.Tensor,
        generated_text: Optional[str] = None,
        mask_formatting: bool = True,
        truncate_post_args: bool = True,
    ) -> float:
        """
        Compute raw confidence score.

        Paper: "the product of the probabilities of the tokens in the
        generated tool call"

        Args:
            generated_ids: Token IDs of generated text [seq_len]
            log_probs: Log probabilities of each token [seq_len]
            generated_text: Optional decoded text for finding tool call end
            mask_formatting: Whether to exclude formatting tokens (paper: True)
            truncate_post_args: Whether to exclude post-argument tokens (paper: True)

        Returns:
            confidence: Product of relevant token probabilities (0 to 1)
        """
        # Ensure 1D tensors
        if generated_ids.dim() > 1:
            generated_ids = generated_ids.squeeze()
        if log_probs.dim() > 1:
            log_probs = log_probs.squeeze()

        # Step 1: Truncate post-argument tokens if requested
        if truncate_post_args and generated_text is not None:
            end_pos = self._find_tool_call_end(generated_text)
            if end_pos is not None:
                # Decode prefix to find token boundary
                for token_idx in range(len(generated_ids)):
                    decoded = self.tokenizer.decode(
                        generated_ids[: token_idx + 1], skip_special_tokens=True
                    )
                    if len(decoded) >= end_pos:
                        # Truncate at this token
                        generated_ids = generated_ids[: token_idx + 1]
                        log_probs = log_probs[: token_idx + 1]
                        break

        # Step 2: Mask formatting tokens if requested
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
        full_generated_ids: torch.Tensor,
    ) -> float:
        """
        Alternative computation that identifies formatting tokens by text matching.

        More robust but slower than token ID matching.
        """
        return self.compute(
            full_generated_ids,
            full_log_probs,
            generated_text=generated_text,
            mask_formatting=True,
            truncate_post_args=True,
        )
