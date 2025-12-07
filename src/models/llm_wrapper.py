"""
LLM wrapper with intermediate layer access for MICE.

Provides access to Llama models with proper HF authentication
and hooks for extracting hidden states from all layers.

Paper Section 4.2:
"We consider three LLMs: Llama3-8B-Instruct, Llama3.1-8B-Instruct, 
and Llama3.2-3B-Instruct"
"""

import os
from typing import Optional, Tuple, List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.auth import setup_hf_auth


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
        torch_dtype: torch.dtype = torch.bfloat16
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
            model_name,
            token=self.token,
            trust_remote_code=True
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
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # Get model properties
        self.num_layers = self.model.config.num_hidden_layers
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        
        print(f"Model loaded: {self.num_layers} layers, "
              f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}")
    
    def format_prompt(
        self,
        query: str,
        api_description: str,
        demos: List[Any]
    ) -> str:
        """
        Format input prompt with demonstrations.
        
        Follows Llama-3 chat format with system message and few-shot examples.
        
        Args:
            query: User query
            api_description: API description for the tool
            demos: List of demonstration examples
        """
        # System message explaining the task
        system_msg = (
            "You are a helpful assistant that generates tool calls. "
            "Given a user query and API description, respond with the appropriate "
            "action and action_input in the following format:\n"
            "action: <api_name>\n"
            "action_input: <json_arguments>"
        )
        
        # Format demonstrations
        demo_text = ""
        for i, demo in enumerate(demos, 1):
            demo_text += (
                f"\n\nExample {i}:\n"
                f"Query: {demo.query}\n"
                f"API: {demo.api_description}\n"
                f"Response:\n{demo.gold_tool_call}"
            )
        
        # Current query
        current_query = (
            f"\n\nNow respond to this query:\n"
            f"Query: {query}\n"
            f"API: {api_description}\n"
            f"Response:"
        )
        
        # Combine into full prompt
        full_prompt = system_msg + demo_text + current_query
        
        return full_prompt
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256
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
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract generated portion
        generated_ids = outputs[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, input_ids, generated_ids.unsqueeze(0)
    
    @torch.no_grad()
    def forward_with_hidden_states(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor
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
            full_ids.to(self.model.device),
            output_hidden_states=True,
            return_dict=True
        )
        
        # hidden_states is tuple of (num_layers + 1) tensors
        # Index 0 is embedding layer output, 1 to num_layers are transformer layers
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        
        return list(hidden_states), logits
    
    def get_lm_head(self) -> torch.nn.Module:
        """Get the language model head (unembedding matrix) for logit lens."""
        return self.model.lm_head

