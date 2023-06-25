import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms



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

    def __init__(self, learning_rate, dropout_rate, batch_size, n_units, epochs, train_size, device):
        # Load MNIST train
        ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
        ds_val = MNIST(root='./data', train=False, download=True, transform=transform)
        batch_size = max(int(batch_size), 1)
        dropout_rate = max(min(dropout_rate, 0.999), 0.001)
        learning_rate = max(min(learning_rate, 0.999), 0.001)
        n_units = max(int(n_units), 10)
        epochs = max(int(epochs), 1)
        train_size = max(int(train_size), 100)
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
        for epoch in range(self.epochs):
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
        for (xs, ys) in self.dl_val:
            xs = xs.to(self.device)
            preds = torch.argmax(self.model(xs), dim=1)
            gts[i] = ys[0]
            predictions[i] = preds[0]
            i += 1
        val_error = torch.tensor(1 - (predictions == gts).sum() / len(gts)).unsqueeze(0)
        # Return validation error
        return val_error
