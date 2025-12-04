 ```python
## data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch

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
    df = pd.read_csv('path_to_dataset.csv')  # Assuming CSV for simplicity

    # Split the dataset into features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data_split']['test_size'], random_state=config['data_split']['random_state'])

    # Create datasets and data loaders
    train_dataset = MiceDataset(X_train, y_train)
    test_dataset = MiceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'], shuffle=False)

    return train_loader, test_loader
```

Reference to Configuration:
- `config['data_split']['test_size']`
- `config['training']['batch_size']`
- `config['evaluation']['batch_size']`
- `config['data_split']['random_state']`