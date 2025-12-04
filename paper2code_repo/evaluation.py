 ```python
## evaluation.py

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, expected_calibration_error
from torch.utils.data import DataLoader
from data_preprocessing import load_data, MiceDataset
from model import MiceModel
from config import config

def initialize_model(config):
    model = MiceModel()
    model.load_state_dict(torch.load(config['model']['checkpoint_path']))
    model.eval()
    return model

def evaluate_calibration(probs, labels, n_bins=15):
    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    return ece

def evaluate_utility(probs, labels, utility_threshold):
    utility = np.where(probs > utility_threshold, labels, np.where(probs < (1 - utility_threshold), 0, -1))
    return np.mean(utility)

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_data(config)
    model = initialize_model(config)
    model = model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Evaluate calibration
    ece = evaluate_calibration(all_probs, all_labels)
    print(f'Smooth Expected Calibration Error (smECE): {ece}')

    # Evaluate utility
    utility_thresholds = config['evaluation']['utility_thresholds']
    for risk_level, threshold in utility_thresholds.items():
        utility = evaluate_utility(all_probs, all_labels, threshold)
        print(f'Expected Tool-Calling Utility at {risk_level} risk level: {utility}')

    # Evaluate AUC
    auc = roc_auc_score(all_labels, all_probs)
    print(f'Validation AUC: {auc}')

if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
```

This code defines the evaluation process for the model, including the calculation of smooth expected calibration error (smECE), expected tool-calling utility, and the area under the receiver operating characteristic curve (AUC). It uses the configuration settings from `config.yaml` to determine the evaluation parameters and paths.