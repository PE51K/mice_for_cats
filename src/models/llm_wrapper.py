"""
LLM wrapper with intermediate layer access for MICE.

Provides access to Llama models with proper HF authentication
and hooks for extracting hidden states from all layers.

Paper Section 4.2:
"We consider three LLMs: Llama3-8B-Instruct, Llama3.1-8B-Instruct,
and Llama3.2-3B-Instruct"
"""

from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.auth import setup_hf_auth

DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "simulated-trial-and-error"
    / "STE"
    / "prompts"
    / "prompt_template.txt"
)

PROMPT_TEMPLATE = dedent(
    """
    Your task is to answer the user's query as best you can. You have access to the following tools which you can use via API call:

    {api_descriptions}

    The format you use the tools is by specifying 1) Action: the API function name you'd like to call 2) Action Input: the input parameters of the API call in a json string format. The result of the API call will be returned starting with "Observation:". Remember that you should only perform a SINGLE action at a time, do NOT return a list of multiple actions.

    Reminder:
    1) the only values that should follow "Action:" are:
    {api_names}

    2) use the following json string format for the API arguments:

    Action Input:
    {{
        "key_1": "value_1",
        ...
        "key_n": "value_n",
    }}

    Remember to ALWAYS use the following format:

    User Query: the input user query that you need to respond to
    Action: the API function name
    Action Input: the input parameters of the API call in json string format
    Observation: the return result of the API call
    ... (this Action/Action Input/Observation can repeat N times)
    Final Answer: the final answer to the original input question

    Begin! Remember that once you have enough information, please immediately use
    Final Answer:
    """
).strip()


class LLMWrapper:
    """
    Wrapper for LLMs with intermediate layer access.

    Handles HF authentication for gated Llama models and provides
    methods for generation with hidden state extraction.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
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

    def format_prompt(self, query: str, api_description: str, demos: List[Any]) -> str:
        """
        Format input prompt with demonstrations.

        Mirrors the STE prompt template used in the dataset for tool-calling,
        inserting API descriptions/names and providing in-context examples
        formatted with Action / Action Input / Final Answer blocks.

        Args:
            query: User query
            api_description: API description for the tool
            demos: List of demonstration examples
        """
        # Try to load STE prompt template from disk; fall back to bundled template.
        prompt_template = PROMPT_TEMPLATE
        if DEFAULT_PROMPT_PATH.exists():
            try:
                prompt_template = DEFAULT_PROMPT_PATH.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError:
                pass

        # Build API descriptors from demos + current example
        api_entries: Dict[str, str] = {}
        for ex in demos:
            api_entries[ex.api_name] = ex.api_description
        if demos:
            api_entries.setdefault(demos[0].api_name, api_description)
        else:
            api_entries.setdefault("API", api_description)
        api_desc_text = "\n".join(
            [f"{name}: {desc}" for name, desc in api_entries.items()]
        )
        api_names_text = ", ".join(api_entries.keys())

        header = prompt_template.format(
            api_descriptions=api_desc_text, api_names=api_names_text
        )

        # Format demonstrations using the Action / Action Input style from STE
        demo_blocks = []
        for i, demo in enumerate(demos, 1):
            demo_blocks.append(
                dedent(
                    f"""
                    Example {i}:
                    User Query: {demo.query}
                    Action: {demo.api_name}
                    Action Input: {demo.gold_action_input}
                    Final Answer:
                    {demo.gold_tool_call}
                    """
                ).strip()
            )

        demos_text = "\n\n".join(demo_blocks)

        # Current query block to elicit the tool call
        current_block = dedent(
            f"""
            User Query: {query}
            Action:
            """
        ).strip()

        prompt_parts = [header, demos_text, current_block]
        return "\n\n".join(part for part in prompt_parts if part)

    @torch.no_grad()
    def generate(
        self, prompt: str, max_new_tokens: int = 256
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate text with greedy decoding.

        Paper Section 4.3: "using greedy decoding to generate the tool calls y"

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            generated_text: Generated string
            input_ids: Input token IDs
            generated_ids: Generated token IDs
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)

        # Generate with greedy decoding (temperature=0 equivalent)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Extract generated portion
        generated_ids = outputs[0, input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, input_ids, generated_ids.unsqueeze(0)

    @torch.no_grad()
    def forward_with_hidden_states(
        self, input_ids: torch.Tensor, generated_ids: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
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
