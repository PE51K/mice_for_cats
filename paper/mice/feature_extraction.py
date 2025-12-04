"""Feature extraction using logit lens decoding from intermediate layers."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np


def decode_from_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    layer_idx: int,
    max_length: int = 512,
) -> str:
    """
    Decode from an intermediate layer using logit lens.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        layer_idx: Index of the layer to decode from (0-indexed)
        max_length: Maximum generation length
        
    Returns:
        Decoded string from the layer
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings
        embeddings = model.get_input_embeddings()(input_ids)
        
        # Forward pass up to the specified layer
        outputs = model.model.embed_tokens(input_ids)
        
        # Process through layers up to layer_idx
        for i in range(layer_idx + 1):
            if hasattr(model.model, 'layers'):
                layer = model.model.layers[i]
            elif hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'h'):
                layer = model.model.transformer.h[i]
            else:
                raise ValueError("Could not find model layers")
            
            # Apply layer with attention mask
            attention_mask = torch.ones_like(input_ids)
            if hasattr(layer, '__call__'):
                outputs = layer(outputs, attention_mask=attention_mask)[0]
            else:
                # Handle different model architectures
                outputs = layer(outputs)
        
        # Get the unembedding matrix (output projection)
        if hasattr(model, 'lm_head'):
            unembedding = model.lm_head.weight
        elif hasattr(model, 'embed_out'):
            unembedding = model.embed_out.weight
        else:
            # Try to find the output embedding
            unembedding = model.get_output_embeddings().weight
        
        # Decode token by token using argmax
        decoded_tokens = []
        current_hidden = outputs[0, -1, :]  # Last position
        
        for _ in range(max_length):
            # Compute logits
            logits = torch.matmul(current_hidden, unembedding.t())
            
            # Get next token
            next_token_id = torch.argmax(logits).item()
            decoded_tokens.append(next_token_id)
            
            # Stop at EOS token
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # For next iteration, we'd need to continue the forward pass
            # This is a simplified version - in practice, we decode the full sequence
            # from the layer's hidden states
            break
        
        # Decode the tokens
        decoded_text = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
        return decoded_text


def extract_logit_lens_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    num_layers: Optional[int] = None,
) -> List[str]:
    """
    Extract decoded strings from each intermediate layer using logit lens.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        generated_ids: Generated token IDs [batch_size, gen_len]
        num_layers: Number of layers to extract from (default: all layers)
        
    Returns:
        List of decoded strings, one per layer
    """
    if num_layers is None:
        if hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
        else:
            raise ValueError("Could not determine number of layers")
    
    layer_outputs = []
    
    # Combine input and generated tokens
    full_sequence = torch.cat([input_ids, generated_ids], dim=1)
    
    model.eval()
    with torch.no_grad():
        # Get embeddings
        embeddings = model.get_input_embeddings()(full_sequence)
        hidden_states = embeddings
        
        # Get unembedding matrix
        if hasattr(model, 'lm_head'):
            unembedding = model.lm_head.weight
        elif hasattr(model, 'embed_out'):
            unembedding = model.embed_out.weight
        else:
            unembedding = model.get_output_embeddings().weight
        
        # Process through each layer and decode
        layers = model.model.layers if hasattr(model.model, 'layers') else model.model.transformer.h
        
        for layer_idx in range(num_layers):
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                attention_mask = torch.ones_like(full_sequence)
                
                # Apply layer
                if hasattr(layer, '__call__'):
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
                else:
                    hidden_states = layer(hidden_states)
            
            # Decode from this layer's hidden states
            # For each position in the generated sequence
            decoded_tokens = []
            gen_start_idx = input_ids.shape[1]
            
            for pos in range(gen_start_idx, full_sequence.shape[1]):
                hidden = hidden_states[0, pos, :]
                logits = torch.matmul(hidden, unembedding.t())
                token_id = torch.argmax(logits).item()
                decoded_tokens.append(token_id)
            
            # Decode to string
            decoded_text = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
            layer_outputs.append(decoded_text)
    
    return layer_outputs


def extract_layer_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Extract hidden states from all intermediate layers.
    
    Args:
        model: The transformer model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        List of hidden states, one per layer [batch_size, seq_len, hidden_size]
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    model.eval()
    
    with torch.no_grad():
        # Use model's forward method with output_hidden_states=True
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract hidden states (excluding embeddings)
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
        
    return hidden_states

