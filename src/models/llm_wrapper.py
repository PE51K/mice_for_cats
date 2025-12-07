"""
LLM wrapper with intermediate layer access for MICE.

Provides access to Llama models with proper HF authentication
and hooks for extracting hidden states from all layers.

Paper Section 4.2:
"We consider three LLMs: Llama3-8B-Instruct, Llama3.1-8B-Instruct,
and Llama3.2-3B-Instruct"
"""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.auth import setup_hf_auth

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "simulated-trial-and-error"
    / "STE"
    / "prompts"
    / "prompt_template.txt"
)


class LLMWrapper:
    """
    Wrapper for LLMs with intermediate layer access.

    Handles HF authentication for gated Llama models and provides
    methods for generation with hidden state extraction.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize LLM with HF authentication.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (auto-detected if None)
            torch_dtype: Model precision (bfloat16 for efficiency)
        """
        # Setup HF authentication first
        self.token = setup_hf_auth()

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name}...")

        # Load tokenizer with auth
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=self.token, trust_remote_code=True
        )

        # Set pad token if not set (common for Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with auth
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=self.token,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

        # Get model properties
        self.num_layers = self.model.config.num_hidden_layers
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size

        print(
            f"Model loaded: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}"
        )

    def format_prompt(self, query: str, api_description: str, demos: list[Any]) -> str:
        r"""
        Format input prompt with demonstrations.

        Exactly matches the STE prompt format from test_gpt.py lines 35-58.
        Uses the prompt template with API descriptions formatted as:
        "API_name: {name}\nDescription: {desc}"

        Args:
            query: User query
            api_description: API description for the tool
            demos: List of demonstration examples
        """
        # Load STE prompt template directly
        prompt_template = PROMPT_PATH.read_text(encoding="utf-8").strip()

        # Build API list - collect all unique APIs from demos + current example
        # STE groups by API, we need to build the full API list
        api_entries: dict[str, str] = {}

        # Add current example's API
        if demos and hasattr(demos[0], "api_name"):
            api_entries[demos[0].api_name] = (
                demos[0].api_description
                if hasattr(demos[0], "api_description")
                else api_description
            )

        # Add all demo APIs
        for ex in demos:
            if hasattr(ex, "api_name") and hasattr(ex, "api_description"):
                api_entries[ex.api_name] = ex.api_description

        # Format API descriptions exactly as STE does (lines 33-34 in test_gpt.py)
        api_descriptions = "\n\n".join(
            [f"API_name: {api_name}\nDescription: {desc}" for api_name, desc in api_entries.items()]
        )

        # Format API names list (line 38 in test_gpt.py)
        api_names = "\n".join(api_entries.keys())

        # Format the header using the template (line 38)
        prompt = prompt_template.format(api_descriptions=api_descriptions, api_names=api_names)

        # Add demonstrations if in ICL setting (lines 52-58 in test_gpt.py)
        if demos and len(demos) > 0:
            demo_blocks = []
            for demo in demos:
                # Format exactly as: "User Query: {}\nAction: {}\nAction Input: {}\n"
                demo_blocks.append(
                    f"User Query: {demo.query}\nAction: {demo.gold_action}\nAction Input: {demo.gold_action_input}\n"
                )

            prompt = (
                prompt
                + "\n\nBelow are some examples:\n\n"
                + "---\n".join(demo_blocks)
                + "Now it's your turn.\n\nUser Query: "
                + query
            )
        else:
            # Default setting (line 53)
            prompt = prompt + "\n\nUser Query: " + query

        return prompt

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        stop_sequences: list[str] | None = None,
    ) -> tuple[str, str, torch.Tensor, torch.Tensor]:
        """
        Generate text matching STE generation parameters.

        STE uses OpenAI chat API with messages format (my_llm.py lines 38-42):
        - System message: "You are a helpful assistant."
        - User message: prompt
        We replicate this with Llama's chat template.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate (STE default: 256)
            temperature: Sampling temperature (STE default: 1.0)
            stop_sequences: Stop at these sequences (STE default: ["Observation:"])

        Returns:
            generated_continuation: Raw continuation after prompt
            full_generated_text: Generated text for parsing (matches STE format)
            input_ids: Input token IDs
            generated_ids: Generated token IDs
        """
        # Wrap prompt in chat template to match STE's OpenAI chat format
        # STE: messages = [{"role": "system", ...}, {"role": "user", "content": prompt}]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)

        # Set stop sequences - default to STE's "Observation:"
        if stop_sequences is None:
            stop_sequences = ["Observation:"]

        # Generate with sampling to match STE (temp=1.0 in test_gpt.py line 60)
        do_sample = temperature > 0.0
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Extract generated portion (continuation after prompt)
        generated_ids = outputs[0, input_ids.shape[1] :]
        generated_continuation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Check for stop sequences and truncate if found
        for stop_seq in stop_sequences:
            if stop_seq in generated_continuation:
                generated_continuation = generated_continuation.split(stop_seq)[0]

        # STE format: The model generates the assistant response
        # For parsing, we use the raw continuation
        full_generated_text = generated_continuation

        return generated_continuation, full_generated_text, input_ids, generated_ids.unsqueeze(0)

    @torch.no_grad()
    def forward_with_hidden_states(
        self, input_ids: torch.Tensor, generated_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Forward pass returning hidden states from all layers.

        Paper Section 2: Uses hidden states from each layer for logit lens.

        Args:
            input_ids: Input token IDs [1, input_len]
            generated_ids: Generated token IDs [1, gen_len]

        Returns:
            hidden_states: List of hidden states from each layer
            logits: Final layer logits
        """
        # Combine input and generated for full forward pass
        full_ids = torch.cat([input_ids, generated_ids], dim=1)

        # Forward pass with hidden states
        outputs = self.model(
            full_ids.to(self.model.device), output_hidden_states=True, return_dict=True
        )

        # hidden_states is tuple of (num_layers + 1) tensors
        # Index 0 is embedding layer output, 1 to num_layers are transformer layers
        hidden_states = outputs.hidden_states
        logits = outputs.logits

        return list(hidden_states), logits

    def get_lm_head(self) -> torch.nn.Module:
        """Get the language model head (unembedding matrix) for logit lens."""
        return self.model.lm_head
