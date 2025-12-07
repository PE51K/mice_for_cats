"""
Combined MICE feature extraction pipeline.

Paper Section 2:
"MICE is a simple learned probabilistic classifier whose features are
derived from model-internal activations."

Features:
1. BERTScore between final output and each intermediate layer's output (ell-1 features)
2. Raw confidence (product of token probabilities)
"""

import json
import re
import sys
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

STE_DIR = Path(__file__).parent.parent / "data" / "simulated-trial-and-error" / "STE"


def _load_parse_response() -> Callable[..., Any]:
    if str(STE_DIR) not in sys.path:
        sys.path.append(str(STE_DIR))
    from utils import parse_response  # type: ignore

    return parse_response


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

        # Generate tool call
        generated_text, input_ids, generated_ids = self.llm.generate(
            prompt=prompt, max_new_tokens=max_new_tokens
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
            generated_text=generated_text,
            mask_formatting=True,
            truncate_post_args=True,
        )

        # Check correctness (exact match with gold)
        # Paper: "we label a generated tool call as correct if and only if
        # it exactly matches the one given by STE"
        is_correct = self._check_correctness(generated_text, example, debug=debug)

        if debug:
            print(f"\n{'=' * 60}")
            print(f"Query: {example.query[:100]}...")
            print(f"API: {example.api_name}")
            print(f"\nGold tool call:\n{example.gold_tool_call}")
            print(f"\nGenerated (raw):\n{generated_text[:500]}...")
            print(f"\nRaw confidence: {raw_confidence:.6f}")
            print(f"Correct: {is_correct}")
            print(f"{'=' * 60}\n")

        return {
            "generated_text": generated_text,
            "bertscore_features": bertscore_features.numpy()
            if isinstance(bertscore_features, torch.Tensor)
            else np.array(bertscore_features),
            "raw_confidence": raw_confidence,
            "is_correct": int(is_correct),
            "layer_outputs": layer_outputs,
            "gold_tool_call": example.gold_tool_call,
        }

    def _check_correctness(self, generated: str, example: STEExample, debug: bool = False) -> bool:
        """
        Check if generated output matches gold tool call.

        Paper: "we label a generated tool call as correct if and only if
        it exactly matches the one given by STE"
        """
        parsed = parse_response(
            generated,
            API_name_list=[example.api_name],
            api_descriptions=example.api_description,
            proc_thought=False,
            proc_toolken=False,
            check_API_name=False,
            ground_API=False,
        )

        gen_action = parsed["action"] if parsed.get("parse_successful") else ""
        gen_input_str = parsed["action_input"] if parsed.get("parse_successful") else ""

        if not parsed.get("parse_successful"):
            is_correct = False
        else:
            # Strict string equality first
            is_correct = (
                gen_action == example.gold_action.strip()
                and gen_input_str.strip() == example.gold_action_input.strip()
            )
            # Fallback to JSON semantic equality for action_input
            if not is_correct:
                try:
                    gen_json = json.loads(gen_input_str)
                    gold_json = json.loads(example.gold_action_input)
                    is_correct = gen_action == example.gold_action.strip() and gen_json == gold_json
                except json.JSONDecodeError:
                    pass

        if debug:
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

    def _compare_json_semantic(self, gen_str: str, gold_str: str) -> bool:
        """
        Compare two JSON strings semantically (ignore formatting).

        Handles whitespace differences, key ordering, etc.
        """
        try:
            # Parse both as JSON
            gen_json = json.loads(gen_str)
            gold_json = json.loads(gold_str)

            # Compare parsed objects
            return gen_json == gold_json
        except json.JSONDecodeError:
            # If parsing fails, fall back to normalized string comparison
            gen_norm = self._normalize_json_string(gen_str)
            gold_norm = self._normalize_json_string(gold_str)
            return gen_norm == gold_norm

    def _normalize_json_string(self, s: str) -> str:
        """Normalize JSON string for comparison when parsing fails."""
        # Remove all whitespace
        s = re.sub(r"\s+", "", s)
        # Lowercase
        s = s.lower()
        return s

    def _normalize_tool_call(self, text: str) -> str:
        """Normalize tool call for comparison."""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Lowercase
        text = text.lower()
        # Remove common variations
        text = text.replace("action :", "action:")
        text = text.replace("action_input:", "action input:")
        return text.strip()

    # _extract_action and _extract_action_input were replaced by STE parse_response usage

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
