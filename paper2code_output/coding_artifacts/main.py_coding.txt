 ```python
## main.py

import yaml
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, expected_calibration_error
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class MiceDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'labels': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data(config):
    # Load the dataset
    df = pd.read_csv(config['data']['dataset_path'])  # Assuming CSV for simplicity

    # Split the dataset into features and labels
    X = df.drop(config['data']['label_column'], axis=1).values
    y = df[config['data']['label_column']].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data_split']['test_size'], random_state=config['data_split']['random_state'])

    # Create datasets and data loaders
    train_dataset = MiceDataset(X_train, y_train)
    test_dataset = MiceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'], shuffle=False)

    return train_loader, test_loader

class MiceModel(nn.Module):
    def __init__(self):
        super(MiceModel, self).__init__()
        self.bert = AutoModel.from_pretrained(config['model']['bert_model_name'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token output
        logits = self.classifier(pooled_output)
        probs = self.sigmoid(logits)
        return probs

def initialize_model(config):
    model = MiceModel()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.BCELoss()
    return model, optimizer, criterion

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                val_loss += criterion(outputs.squeeze(), labels.float())
        model.train()

        logging.info(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    return model

def evaluate_model(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

def evaluate_calibration(probs, labels, n_bins=15):
    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    return ece

def evaluate_utility(probs, labels, utility_threshold):
    utility = np.where(probs > utility_threshold, labels, np.where(probs < (1 - utility_threshold), 0, -1))
    return np.mean(utility)

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_data(config)
    model, optimizer, criterion = initialize_model(config)
    model = model.to(device)

    model = train_model(model, optimizer, criterion, train_loader, test_loader, config['training']['epochs'], device)

    probs, labels = evaluate_model(model, test_loader, device)
    roc_auc = roc_auc_score(labels, probs)
    logging.info(f'Validation AUC: {roc_auc}')

    # Evaluate calibration and utility
    ece = evaluate_calibration(probs, labels)
    logging.info(f'Smooth Expected Calibration Error (smECE): {ece}')

    utility_thresholds = config['evaluation']['utility_thresholds']
    for risk_level, threshold in utility_thresholds.items():
        utility = evaluate_utility(probs, labels, threshold)
        logging.info(f'Expected Tool-Calling Utility at {risk_level} risk level: {utility}')

if __name__ == "__main__":
    main(config)
```
