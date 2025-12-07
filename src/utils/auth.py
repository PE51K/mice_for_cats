"""
Authentication utilities for Hugging Face model access.

Llama models are gated and require authentication.
Set HF_TOKEN environment variable or create a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login


def setup_hf_auth() -> str:
    """
    Setup Hugging Face authentication from environment.
    
    Looks for HF_TOKEN in:
    1. Environment variable (already set)
    2. .env file in project root
    
    Returns:
        token: The HF token if found
        
    Raises:
        ValueError: If no token is found
    """
    # Try to load from .env file in project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
    
    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        raise ValueError(
            "HF_TOKEN not found!\n\n"
            "Please set the HF_TOKEN environment variable or create a .env file:\n"
            "  1. Get your token from: https://huggingface.co/settings/tokens\n"
            "  2. Accept Llama model license at:\n"
            "     - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct\n"
            "     - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct\n"
            "  3. Set token:\n"
            "     export HF_TOKEN=hf_xxxxx\n"
            "     OR\n"
            "     cp env.template .env && edit .env"
        )
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=token)
    
    return token

