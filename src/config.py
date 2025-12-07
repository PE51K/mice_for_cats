"""
Configuration for MICE for CATs reproduction.

Hyperparameters are taken directly from the paper:
- Section 4.1: Dataset splits
- Section 4.3: Experimental settings (8-shot ICL)
- Section 4.4: Model configurations
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """
    Dataset configuration from paper Section 4.1.
    
    Paper: "The dataset consists of English-language queries that require 
    calling 50 distinct APIs."
    """
    # Paper: "demonstration set consisting of 4,520 examples taken from the STE training set"
    demo_set_size: int = 4520
    
    # Paper: "training set of 1500 examples (30 from each API)"
    train_set_size: int = 1500
    samples_per_api_train: int = 30
    
    # Paper: "validation set of 750 examples (15 from each API)"
    val_set_size: int = 750
    samples_per_api_val: int = 15
    
    # Paper: "test set of 750 examples"
    test_set_size: int = 750
    
    # Number of APIs in STE dataset
    num_apis: int = 50


@dataclass
class ModelConfig:
    """LLM configuration."""
    # Available models - using Llama-3.1-8B and Llama-3.2-3B as per requirements
    # Paper used Llama3-8B-Instruct (32 layers), Llama3.1-8B-Instruct (32 layers),
    # and Llama3.2-3B-Instruct (28 layers)
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # BERTScore model (paper Section 4.4)
    # Paper: "DeBERTa-xlarge-mnli to compute the BERTScore features 
    # as it is the strongest BERTScore base model"
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"


@dataclass
class ICLConfig:
    """
    In-context learning configuration from paper Section 4.3.
    
    Paper: "We run each LLM on our validation and test sets in an 8-shot 
    in-context learning setting, following Wang et al. (2024)"
    """
    # Paper: "8-shot in-context learning setting"
    num_shots: int = 8
    
    # Paper: "using SentenceBERT (Reimers and Gurevych, 2019)"
    # for similarity-based example selection
    sentence_transformer_model: str = "all-MiniLM-L6-v2"


@dataclass
class MICEConfig:
    """
    MICE model hyperparameters from paper Section 4.4.
    """
    # MICE Logistic Regression
    # Paper: "L2 regularization strength of 2"
    # Note: sklearn uses C = 1/lambda, so C = 0.5 for L2 strength of 2
    lr_l2_strength: float = 2.0
    lr_C: float = 0.5  # 1 / lr_l2_strength
    
    # MICE Random Forest
    # Paper: "1000 trees each with a maximum depth of 20 
    # and a maximum of 10 features to use at each split"
    rf_n_estimators: int = 1000
    rf_max_depth: int = 20
    rf_max_features: int = 10


@dataclass
class HREConfig:
    """
    Histogram Regression Estimator configuration from paper Section 4.4.
    
    Paper: "We use 25 bins: [0, 0.04), [0.04, 0.08), ..., [0.96, 1.0]"
    """
    num_bins: int = 25
    bin_width: float = 0.04


@dataclass
class ETCUConfig:
    """
    Expected Tool-Calling Utility configuration from paper Section 3.2.
    
    Paper Table 1 risk levels:
    - High Risk: fp = -9, τ = 0.9
    - Medium Risk: fp = -1, τ = 0.5  
    - Low Risk: fp = -1/9, τ = 0.1
    """
    thresholds: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # Paper: "we approximate AUC by evaluating expected tool-calling utility 
    # at each τ ∈ {0.001, 0.002, ..., 0.999}"
    auc_granularity: int = 999


@dataclass
class Config:
    """Main configuration combining all settings."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    icl: ICLConfig = field(default_factory=ICLConfig)
    mice: MICEConfig = field(default_factory=MICEConfig)
    hre: HREConfig = field(default_factory=HREConfig)
    etcu: ETCUConfig = field(default_factory=ETCUConfig)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Paths (relative to mice_for_cats directory)
    data_dir: Path = field(default_factory=lambda: Path("data/simulated-trial-and-error"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Generation settings
    max_new_tokens: int = 256

