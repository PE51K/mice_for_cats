"""
Logit Lens decoding from intermediate layers.

Paper Section 2:
"MICE first decodes from each intermediate layer of the language model using
logit lens (nostalgebraist, 2020) and then computes similarity scores between
each layer's generation and the final output."

"We first decode the output string y at temperature 0. Then at each layer i < ℓ,
we obtain a preliminary output string y(i) of the same length by per-token
argmax decoding:
    y_t^(i) = argmax(h_t-1^(i) @ W_out)
where h_t-1^(i) is the model's layer-i encoding at the previous position,
whose product with the unembedding matrix W_out ∈ R^{d×|V|} is a vector of
logits ∈ R^{|V|}."
"""


import torch
from transformers import AutoTokenizer


class LogitLensDecoder:
    """
    Decode from intermediate layers using logit lens.

    This extracts the intermediate layer outputs and decodes them
    using the model's unembedding matrix (lm_head).

    Paper Figure 2 shows example generations from validation set across layers:
    "Generations from early layers (5, 15) are seemingly random, but later
    layers (25, 31) generate thematically relevant tokens."
    """

    def __init__(self, lm_head: torch.nn.Module, tokenizer: AutoTokenizer, num_layers: int):
        """
        Initialize logit lens decoder.

        Args:
            lm_head: Language model head (unembedding matrix W_out)
            tokenizer: Tokenizer for decoding
            num_layers: Number of transformer layers
        """
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.device = next(lm_head.parameters()).device

    @torch.no_grad()
    def decode_all_layers(
        self, hidden_states: list[torch.Tensor], input_len: int, generated_ids: torch.Tensor
    ) -> tuple[str, list[str], list[torch.Tensor]]:
        """
        Decode from all intermediate layers using logit lens.

        Paper: "This results in ℓ strings, where ℓ is the number of layers"

        Args:
            hidden_states: Hidden states from all layers (including embedding layer)
            input_len: Length of input sequence (to extract generation positions)
            generated_ids: Generated token IDs [1, gen_len]

        Returns:
            final_output: String decoded from final layer (y)
            layer_outputs: List of strings from each layer (y^(i))
            layer_logprobs: Log probabilities at each layer for generated tokens
        """
        gen_len = generated_ids.shape[1]
        generated_ids = generated_ids.to(self.device)

        layer_outputs = []
        layer_logprobs = []

        # Skip embedding layer (index 0), process transformer layers 1 to num_layers
        for layer_idx in range(1, len(hidden_states)):
            # Get hidden states for positions that generated output
            # Paper: "h_t-1^(i)" - use position t-1 to predict token t
            # For generation positions: input_len-1 to input_len+gen_len-2
            layer_hidden = hidden_states[layer_idx][:, input_len - 1 : input_len - 1 + gen_len, :]
            layer_hidden = layer_hidden.to(self.device)

            # Project through lm_head to get logits
            # Paper: "h_t-1^(i) @ W_out"
            logits = self.lm_head(layer_hidden)  # [1, gen_len, vocab_size]

            # Argmax decode to get layer's output string
            # Paper: "y_t^(i) = argmax(...)"
            layer_tokens = logits.argmax(dim=-1).squeeze(0)  # [gen_len]
            layer_text = self.tokenizer.decode(layer_tokens, skip_special_tokens=True)
            layer_outputs.append(layer_text)

            # Get log probabilities for the actual generated tokens
            log_probs = torch.log_softmax(logits, dim=-1)  # [1, gen_len, vocab_size]
            token_logprobs = (
                log_probs.squeeze(0).gather(1, generated_ids.squeeze(0).unsqueeze(1)).squeeze(1)
            )  # [gen_len]
            layer_logprobs.append(token_logprobs.cpu())

        # Final output is from the last layer
        final_output = self.tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)

        return final_output, layer_outputs, layer_logprobs

    def get_final_layer_logprobs(
        self, hidden_states: list[torch.Tensor], input_len: int, generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get log probabilities from the final layer only.

        Used for computing raw confidence.

        Returns:
            logprobs: Log probabilities for generated tokens [gen_len]
        """
        gen_len = generated_ids.shape[1]
        generated_ids = generated_ids.to(self.device)

        # Get final layer hidden states
        final_hidden = hidden_states[-1][:, input_len - 1 : input_len - 1 + gen_len, :]
        final_hidden = final_hidden.to(self.device)

        # Get logits and log probs
        logits = self.lm_head(final_hidden)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probs for generated tokens
        token_logprobs = (
            log_probs.squeeze(0).gather(1, generated_ids.squeeze(0).unsqueeze(1)).squeeze(1)
        )

        return token_logprobs.cpu()
