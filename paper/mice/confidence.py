"""Raw confidence calculation from token probabilities."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import re


def extract_tool_call_tokens(
    generated_text: str,
    token_ids: List[int],
    tokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Extract token indices relevant to the tool call, omitting formatting tokens.
    
    According to the paper, we omit:
    - Tokens associated with formatting ("action:" and "action input:")
    - Tokens generated after the arguments of the tool call
    
    Args:
        generated_text: The generated text
        token_ids: List of token IDs
        tokenizer: The tokenizer
        
    Returns:
        Tuple of (relevant_token_indices, omitted_token_indices)
    """
    # Find the tool call part (between "Action:" and end of arguments)
    # This is a simplified version - in practice, you'd parse the structured output
    
    # Look for "Action:" and "Action Input:"
    action_match = re.search(r'Action:\s*(\w+)', generated_text, re.IGNORECASE)
    action_input_match = re.search(r'Action Input:\s*(\{.*?\})', generated_text, re.IGNORECASE | re.DOTALL)
    
    if not action_match or not action_input_match:
        # If we can't find the structure, include all tokens
        return list(range(len(token_ids))), []
    
    action_start = action_match.start()
    action_input_end = action_input_match.end()
    
    # Tokenize to find token boundaries
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # Find token indices corresponding to the tool call
    # This is approximate - in practice, you'd do proper alignment
    relevant_indices = []
    omitted_indices = []
    
    current_pos = 0
    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_len = len(token_text)
        
        # Check if this token is in the relevant range
        # (between action start and action input end, but not the formatting keywords)
        if action_start <= current_pos < action_input_end:
            # Check if it's a formatting token
            if 'action' in token_text.lower() and ':' in token_text:
                omitted_indices.append(i)
            elif 'input' in token_text.lower() and ':' in token_text:
                omitted_indices.append(i)
            else:
                relevant_indices.append(i)
        else:
            omitted_indices.append(i)
        
        current_pos += token_len
    
    return relevant_indices, omitted_indices


def compute_raw_confidence(
    logits: torch.Tensor,
    generated_token_ids: torch.Tensor,
    relevant_token_indices: Optional[List[int]] = None,
) -> float:
    """
    Compute raw confidence as the product of token probabilities.
    
    Args:
        logits: Logits for generated tokens [seq_len, vocab_size]
        generated_token_ids: Generated token IDs [seq_len]
        relevant_token_indices: Indices of tokens to include (None = all)
        
    Returns:
        Raw confidence score (product of probabilities)
    """
    if relevant_token_indices is None:
        relevant_token_indices = list(range(len(generated_token_ids)))
    
    if len(relevant_token_indices) == 0:
        return 0.0
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get probability of each generated token
    token_probs = []
    for idx in relevant_token_indices:
        if idx < len(generated_token_ids):
            token_id = generated_token_ids[idx].item()
            if idx < len(probs):
                prob = probs[idx, token_id].item()
                token_probs.append(prob)
    
    if len(token_probs) == 0:
        return 0.0
    
    # Product of probabilities
    confidence = 1.0
    for prob in token_probs:
        confidence *= prob
    
    return confidence


def compute_raw_confidence_from_model(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    generated_text: str,
    generated_token_ids: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> float:
    """
    Compute raw confidence from model outputs.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token IDs
        generated_text: Generated text string
        generated_token_ids: Generated token IDs (if None, will tokenize generated_text)
        
    Returns:
        Raw confidence score
    """
    if generated_token_ids is None:
        generated_token_ids = tokenizer.encode(generated_text, return_tensors='pt').to(device)
        if generated_token_ids.shape[0] > 1:
            generated_token_ids = generated_token_ids[0]
    else:
        generated_token_ids = generated_token_ids.to(device)
    
    # Ensure input_ids are on the correct device
    input_ids = input_ids.to(device)
    
    # Get logits for generated tokens
    full_sequence = torch.cat([input_ids, generated_token_ids.unsqueeze(0)], dim=1).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(full_sequence)
        logits = outputs.logits
        
        # Extract logits for generated tokens
        input_len = input_ids.shape[1]
        gen_logits = logits[0, input_len-1:-1, :]  # -1 because we predict next token
        
        # Extract relevant token indices
        relevant_indices, _ = extract_tool_call_tokens(
            generated_text,
            generated_token_ids.tolist(),
            tokenizer,
        )
        
        # Compute confidence
        confidence = compute_raw_confidence(
            gen_logits,
            generated_token_ids,
            relevant_indices,
        )
    
    return confidence

