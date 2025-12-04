 ```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import load_data, MiceDataset
from model import MiceModel
from config import config

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

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

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

# Import necessary packages and functions
import numpy as np
from sklearn.metrics import roc_auc_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, val_loader = load_data(config)

# Initialize model
model, optimizer, criterion = initialize_model(config)
model = model.to(device)

# Train model
model = train_model(model, optimizer, criterion, train_loader, val_loader, config['training']['epochs'], device)

# Evaluate model
probs, labels = evaluate_model(model, val_loader, device)
roc_auc = roc_auc_score(labels, probs)
print(f'Validation AUC: {roc_auc}')
```