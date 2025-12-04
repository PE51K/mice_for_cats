"""Data loading and processing for STE dataset."""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from bert_score import score as bert_score
from tqdm import tqdm

from .feature_extraction import extract_layer_hidden_states
from .confidence import compute_raw_confidence_from_model, extract_tool_call_tokens


def load_ste_dataset(file_path: str) -> Dict:
    """
    Load STE dataset from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with API names as keys and lists of examples as values
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_tool_call(example: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Parse tool call from example.
    
    Args:
        example: Example dictionary with 'action' and 'action_input' fields
        
    Returns:
        Tuple of (is_correct, action, action_input_string)
    """
    # Check if tool call is correct
    # According to paper: correct if api_match == 1 and args_correct == 1
    is_correct = (
        example.get('api_match', 0) == 1 and
        example.get('args_correct', 0) == 1 and
        example.get('err', 1) == 0 and
        example.get('no_call', 1) == 0
    )
    
    action = example.get('action', None)
    action_input = example.get('action_input', None)
    
    # Format as string for comparison
    if action and action_input:
        tool_call_str = f"Action: {action}\nAction Input: {action_input}"
    else:
        tool_call_str = None
    
    return is_correct, action, tool_call_str


def extract_mice_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generated_text: str,
    input_prompt: str,
    bertscore_model: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[np.ndarray, float]:
    """
    Extract MICE features for a single example.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        generated_text: Generated tool call text
        input_prompt: Input prompt/query
        bertscore_model: BERTScore model name (default: microsoft/deberta-xlarge-mnli)
        device: Device to use for tensors
        
    Returns:
        Tuple of (bert_scores, raw_confidence)
    """
    if bertscore_model is None:
        bertscore_model = "microsoft/deberta-xlarge-mnli"
    
    # Tokenize and move to device
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
    generated_ids = tokenizer.encode(generated_text, return_tensors='pt', add_special_tokens=False).to(device)
    
    # Get number of layers
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        num_layers = model.config.n_layer
    else:
        raise ValueError("Could not determine number of layers")
    
    # Extract hidden states from all layers
    full_sequence = torch.cat([input_ids, generated_ids], dim=1).to(device)
    attention_mask = torch.ones_like(full_sequence).to(device)
    hidden_states_list = extract_layer_hidden_states(model, full_sequence, attention_mask)
    
    # Get unembedding matrix
    if hasattr(model, 'lm_head'):
        unembedding = model.lm_head.weight
    elif hasattr(model, 'embed_out'):
        unembedding = model.embed_out.weight
    else:
        unembedding = model.get_output_embeddings().weight
    
    # Decode from each layer
    layer_outputs = []
    input_len = input_ids.shape[1]
    
    for layer_idx, hidden_states in enumerate(hidden_states_list):
        # Decode generated tokens from this layer
        decoded_tokens = []
        for pos in range(input_len, full_sequence.shape[1]):
            hidden = hidden_states[0, pos, :]
            logits = torch.matmul(hidden, unembedding.t())
            token_id = torch.argmax(logits).item()
            decoded_tokens.append(token_id)
        
        # Decode to string
        decoded_text = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
        layer_outputs.append(decoded_text)
    
    # Compute BERTScore between each layer output and final output
    final_output = generated_text
    bert_scores = []
    
    # Batch all BERTScore computations together for efficiency
    if len(layer_outputs) > 0:
        try:
            # Prepare all layer outputs and references
            candidates = layer_outputs
            references = [final_output] * len(layer_outputs)
            
            # Compute BERTScore for all layers at once (much faster)
            P, R, F1 = bert_score(
                candidates,
                references,
                model_type=bertscore_model,
                verbose=False,
                batch_size=8,  # Process in batches
                device=device if device != "cpu" else None,
            )
            # Extract F1 scores
            if isinstance(F1, torch.Tensor):
                bert_scores = F1.cpu().numpy().tolist()
            else:
                bert_scores = [f.item() if hasattr(f, 'item') else float(f) for f in F1]
        except Exception as e:
            # If BERTScore fails, use default values
            print(f"Warning: BERTScore computation failed: {e}")
            bert_scores = [0.0] * len(layer_outputs)
    else:
        bert_scores = []
    
    # Compute raw confidence
    try:
        raw_confidence = compute_raw_confidence_from_model(
            model, tokenizer, input_ids, generated_text, generated_ids[0], device
        )
    except Exception as e:
        print(f"Warning: Raw confidence computation failed: {e}")
        raw_confidence = 0.0
    
    # Convert to numpy array
    bert_scores_array = np.array(bert_scores)
    
    return bert_scores_array, raw_confidence


def prepare_dataset(
    data_path: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_examples: Optional[int] = None,
    bertscore_model: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Prepare dataset with MICE features.
    
    Args:
        data_path: Path to STE dataset JSON file
        model: The language model
        tokenizer: The tokenizer
        max_examples: Maximum number of examples to process (None = all)
        bertscore_model: BERTScore model name
        device: Device to use for tensors
        
    Returns:
        Tuple of (features, labels, raw_confidences, examples_log)
        - features: [n_examples, n_layers + 1] (BERTScore features + raw confidence)
        - labels: [n_examples] (1 = correct, 0 = incorrect)
        - raw_confidences: [n_examples] (raw confidence scores)
        - examples_log: List of dicts with query, model_output, ground_truth, is_correct
    """
    # Load dataset
    data = load_ste_dataset(data_path)
    
    all_features = []
    all_labels = []
    all_raw_confidences = []
    examples_log = []
    
    # Process each API
    for api_name, examples in tqdm(data.items(), desc="Processing APIs"):
        for example in examples:
            if max_examples and len(all_labels) >= max_examples:
                break
            
            # Parse tool call
            is_correct, action, tool_call_str = parse_tool_call(example)
            
            if tool_call_str is None:
                continue
            
            # Get query
            query = example.get('query', '')
            if not query:
                continue
            
            # Get ground truth and model output
            # In STE dataset: api_name (key) is ground truth, example['action'] is model output
            ground_truth_api = api_name  # The key is the ground truth API name
            model_output_action = example.get('action', '')
            model_output_input = example.get('action_input', '')
            ground_truth = f"Action: {ground_truth_api}"  # Ground truth is just the API name
            
            # Format as prompt (simplified - in practice, you'd use the actual prompt format)
            prompt = query
            
            try:
                # Extract MICE features
                bert_scores, raw_confidence = extract_mice_features(
                    model, tokenizer, tool_call_str, prompt, bertscore_model, device
                )
                
                # Combine features: BERTScore features + raw confidence
                features = np.concatenate([bert_scores, [raw_confidence]])
                
                all_features.append(features)
                all_labels.append(1 if is_correct else 0)
                all_raw_confidences.append(raw_confidence)
                
                # Log example details
                examples_log.append({
                    'query': query,
                    'model_output': tool_call_str,
                    'model_action': model_output_action,
                    'model_action_input': model_output_input,
                    'ground_truth_api': ground_truth_api,
                    'ground_truth': ground_truth,
                    'is_correct': bool(is_correct),
                    'api_match': example.get('api_match', 0) == 1,
                    'args_correct': example.get('args_correct', 0) == 1,
                    'raw_confidence': float(raw_confidence),
                })
                
                # Progress indicator
                if len(all_labels) % 10 == 0:
                    print(f"  Processed {len(all_labels)} examples...")
                
            except Exception as e:
                print(f"Error processing example: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if max_examples and len(all_labels) >= max_examples:
            break
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    raw_confidences_array = np.array(all_raw_confidences)
    
    return features_array, labels_array, raw_confidences_array, examples_log


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    raw_confidences: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        features: Feature matrix
        labels: Labels
        raw_confidences: Raw confidence scores
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        random_seed: Random seed
        
    Returns:
        Tuple of (train, val, test) where each is (features, labels, raw_confidences)
    """
    np.random.seed(random_seed)
    n = len(labels)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train = (features[train_indices], labels[train_indices], raw_confidences[train_indices])
    val = (features[val_indices], labels[val_indices], raw_confidences[val_indices])
    test = (features[test_indices], labels[test_indices], raw_confidences[test_indices])
    
    return train, val, test

