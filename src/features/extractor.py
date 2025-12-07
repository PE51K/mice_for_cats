"""
Combined MICE feature extraction pipeline.

Paper Section 2:
"MICE is a simple learned probabilistic classifier whose features are
derived from model-internal activations."

Features:
1. BERTScore between final output and each intermediate layer's output (ell-1 features)
2. Raw confidence (product of token probabilities)
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ..data.dataset import STEExample
from ..data.demo_selector import DemoSelector
from ..models.llm_wrapper import LLMWrapper
from ..models.logit_lens import LogitLensDecoder
from .bertscore import BERTScoreComputer
from .raw_confidence import RawConfidenceComputer

STE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "simulated-trial-and-error" / "STE"
)


def _load_parse_response() -> Callable[..., Any]:
    """Load parse_response from the STE utils module via file path."""
    utils_path = STE_DIR / "utils.py"
    spec = importlib.util.spec_from_file_location("ste_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_response  # type: ignore[arg-type]


parse_response = _load_parse_response()


class MICEFeatureExtractor:
    """
    Extract MICE features for confidence estimation.

    Combines:
    1. Logit lens decoding from intermediate layers
    2. BERTScore computation between layer outputs and final output
    3. Raw confidence from token probabilities
    """

    def __init__(self, llm: LLMWrapper, bertscore_model: str = "microsoft/deberta-xlarge-mnli"):
        """
        Initialize feature extractor.

        Args:
            llm: LLM wrapper with hidden state access
            bertscore_model: Model for BERTScore computation
        """
        self.llm = llm

        # Initialize components
        self.logit_lens = LogitLensDecoder(
            lm_head=llm.get_lm_head(),
            tokenizer=llm.tokenizer,
            num_layers=llm.num_layers,
        )
        self.bertscore = BERTScoreComputer(model_name=bertscore_model)
        self.raw_conf = RawConfidenceComputer(tokenizer=llm.tokenizer)

    def extract_single(
        self,
        example: STEExample,
        demos: list[STEExample],
        max_new_tokens: int = 256,
        debug: bool = False,
    ) -> dict[str, Any]:
        """
        Extract features for a single example.

        Args:
            example: Test example to process
            demos: Demonstration examples for ICL
            max_new_tokens: Maximum tokens to generate
            debug: If True, print detailed debug info

        Returns:
            Dictionary containing:
            - generated_text: Model's generated output
            - bertscore_features: BERTScore for each layer [num_layers-1]
            - raw_confidence: Product of token probabilities
            - is_correct: Whether generation matches gold
            - layer_outputs: Decoded strings from each layer
        """
        # Format prompt with demonstrations
        prompt = self.llm.format_prompt(
            query=example.query, api_description=example.api_description, demos=demos
        )

        # Generate tool call - matching STE parameters (temp=1.0, stop="Observation:")
        (
            _generated_continuation,
            full_generated_text,
            input_ids,
            generated_ids,
        ) = self.llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            stop_sequences=["Observation:"],
        )

        # Get hidden states from all layers
        hidden_states, _ = self.llm.forward_with_hidden_states(
            input_ids=input_ids, generated_ids=generated_ids
        )

        # Decode from all layers using logit lens
        final_output, layer_outputs, _layer_logprobs = self.logit_lens.decode_all_layers(
            hidden_states=hidden_states,
            input_len=input_ids.shape[1],
            generated_ids=generated_ids,
        )

        # Compute BERTScore features
        bertscore_features = self.bertscore.compute_features(
            final_output=final_output, layer_outputs=layer_outputs
        )

        # Compute raw confidence
        # Paper Section 4.4: "S omits the tokens associated with formatting...
        # and also omits tokens that are generated after the arguments of the tool call"
        final_logprobs = self.logit_lens.get_final_layer_logprobs(
            hidden_states=hidden_states,
            input_len=input_ids.shape[1],
            generated_ids=generated_ids,
        )
        raw_confidence = self.raw_conf.compute(
            generated_ids=generated_ids.squeeze(),
            log_probs=final_logprobs,
            generated_text=full_generated_text,
            mask_formatting=True,
            truncate_post_args=True,
        )

        # Check correctness (exact match with gold)
        # Paper: "we label a generated tool call as correct if and only if
        # it exactly matches the one given by STE"
        is_correct = self._check_correctness(full_generated_text, example, demos, debug=debug)

        if debug:
            print(f"\n{'=' * 60}")
            print(f"Query: {example.query[:100]}...")
            print(f"API: {example.api_name}")
            print(f"\nGold tool call:\n{example.gold_tool_call}")
            print(f"\nGenerated text:\n{full_generated_text[:500]}...")
            print(f"\nRaw confidence: {raw_confidence:.6f}")
            print(f"Correct: {is_correct}")
            print(f"{'=' * 60}\n")

        return {
            "generated_text": full_generated_text,
            "bertscore_features": bertscore_features.numpy()
            if isinstance(bertscore_features, torch.Tensor)
            else np.array(bertscore_features),
            "raw_confidence": raw_confidence,
            "is_correct": int(is_correct),
            "layer_outputs": layer_outputs,
            "gold_tool_call": example.gold_tool_call,
        }

    def _check_correctness(
        self, generated: str, example: STEExample, demos: list[STEExample], debug: bool = False
    ) -> bool:
        """
        Check if generated output matches gold tool call.

        Exactly matches STE's evaluation approach from test_gpt.py line 61.
        Paper: "we label a generated tool call as correct if and only if
        it exactly matches the one given by STE"
        """
        # Build API_name_list and api_descriptions exactly as STE does
        # This matches the format used in test_gpt.py lines 29-34
        api_list = [example.api_name]

        # Build the formatted api_descriptions string as in test_gpt.py line 33
        api_descriptions_str = (
            f"API_name: {example.api_name}\nDescription: {example.api_description}"
        )

        # Call parse_response exactly as in test_gpt.py line 61
        # Note: Default parameters match test_gpt.py (no explicit params passed)
        parsed = parse_response(generated, api_list, api_descriptions_str)

        # Use .get() to safely access keys that may not exist when parsing fails
        gen_action = parsed.get("action", "")
        gen_input_str = parsed.get("action_input", "")

        if not parsed.get("parse_successful"):
            is_correct = False
        else:
            is_correct = (
                gen_action == example.gold_action.strip()
                and gen_input_str.strip() == example.gold_action_input.strip()
            )

        if debug:
            print(f"  Parse successful: {parsed.get('parse_successful')}")
            if not parsed.get("parse_successful"):
                print(f"  Parse error: {parsed.get('parse_error_msg', 'Unknown')}")
            print(f"  [{'CORRECT' if is_correct else 'INCORRECT'}]")
            print(f"  Gold action: '{example.gold_action.strip()}'")
            print(f"  Gen action:  '{gen_action}'")
            gold_preview = (
                example.gold_action_input[:100] + "..."
                if len(example.gold_action_input) > 100
                else example.gold_action_input
            )
            gen_preview = gen_input_str[:100] + "..." if len(gen_input_str) > 100 else gen_input_str
            print(f"  Gold input: '{gold_preview}'")
            print(f"  Gen input:  '{gen_preview}'")
            print(f"  Exact match: {is_correct}")

        return is_correct

    def extract_batch(
        self,
        examples: list[STEExample],
        demo_selector: DemoSelector,
        max_new_tokens: int = 256,
        desc: str = "Extracting features",
        debug_first_n: int = 0,
    ) -> dict[str, np.ndarray]:
        """
        Extract features for a batch of examples.

        Args:
            examples: List of examples to process
            demo_selector: DemoSelector for finding similar demonstrations
            max_new_tokens: Maximum tokens to generate
            desc: Progress bar description
            debug_first_n: Print debug info for first N examples (0 to disable)

        Returns:
            Dictionary containing arrays:
            - features: BERTScore features [n_examples, num_layers-1]
            - raw_confidences: Raw confidence scores [n_examples]
            - labels: Correctness labels [n_examples]
            - generated_texts: List of generated strings
        """
        all_features = []
        all_raw_confs = []
        all_labels = []
        all_generated = []

        for i, example in enumerate(tqdm(examples, desc=desc)):
            # Select demonstrations
            demos = demo_selector.select_demos(example.query)

            # Extract features (with debug for first N)
            debug = i < debug_first_n
            result = self.extract_single(
                example=example, demos=demos, max_new_tokens=max_new_tokens, debug=debug
            )

            all_features.append(result["bertscore_features"])
            all_raw_confs.append(result["raw_confidence"])
            all_labels.append(result["is_correct"])
            all_generated.append(result["generated_text"])

        # Print summary
        labels_arr = np.array(all_labels)
        print(
            f"\n  Batch summary: {labels_arr.sum()}/{len(labels_arr)} correct "
            f"({100 * labels_arr.mean():.1f}%)"
        )

        return {
            "features": np.array(all_features),
            "raw_confidences": np.array(all_raw_confs),
            "labels": np.array(all_labels),
            "generated_texts": all_generated,
        }
