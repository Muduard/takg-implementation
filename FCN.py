# Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


class MLP(nn.Module):
    def __init__(self, n_units, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, n_units)
        self.fc2 = nn.Linear(n_units, n_units)
        self.fc3 = nn.Linear(n_units, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Standard MNIST transform.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class FCN_OPT():

    def __init__(self, learning_rate, dropout_rate, batch_size, n_units, epochs, train_size, val_size, device):
        # Load MNIST train
        ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
        batch_size = int(batch_size)
        n_units = int(n_units)
        epochs = int(20 * epochs)
        train_size = int(train_size)
        assert train_size <= len(ds_train) - val_size
        if train_size == 0:
            train_size = len(ds_train) - val_size
        # Split train into train and validation
        I = np.random.permutation(len(ds_train))
        ds_val = Subset(ds_train, I[:val_size])
        ds_train = Subset(ds_train, I[val_size:val_size + train_size])
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.n_units = n_units
        self.train_size = train_size
        self.epochs = epochs
        self.device = device
        self.dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
        self.dl_val = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4)
        # Instantiate model and optimizer.
        self.model = MLP(self.n_units, self.dropout_rate).to(device)
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in tqdm(range(self.epochs), desc=f'Training'):
            self.train_epoch(epoch)

    # Function to train a model for a single epoch over the data loader
    def train_epoch(self, epoch='Unknown'):
        self.model.train()
        losses = []
        for (xs, ys) in self.dl_train:
            xs = xs.to(self.device)
            ys = ys.to(self.device)
            self.opt.zero_grad()
            logits = self.model(xs)
            loss = F.cross_entropy(logits, ys)
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        return np.mean(losses)

    # Function to evaluate model over all samples in the data loader
    def evaluate(self):
        self.model.eval()
        predictions = np.zeros(len(self.dl_val), dtype=int)
        gts = np.zeros(len(self.dl_val), dtype=int)
        i = 0
        for (xs, ys) in tqdm(self.dl_val, desc='Evaluating', leave=False):
            xs = xs.to(self.device)
            preds = torch.argmax(self.model(xs), dim=1)
            gts[i] = ys[0]
            predictions[i] = preds[0]
            i += 1
        val_error = torch.tensor(1 - (predictions == gts).sum() / len(gts)).unsqueeze(0)
        # Return validation error
        return val_error

'''
model = FCN_OPT(learning_rate=0.02,
                dropout_rate=0.2,
                batch_size=128,
                n_units=800,
                epochs=1,
                train_size=0,
                val_size=5000,
                device='cuda:0')
model.train()
print(model.evaluate())
'''