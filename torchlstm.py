import io
import io
import json
import requests
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import datasets, models, transforms

# Load Data 
df_train = pd.read_csv("/home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/train.csv")
df_train.dropna(axis = 0, inplace = True)
print(df_train.head())

training_data, validation_data = train_test_split(df_train, test_size=0.2, shuffle = False)

print(f"Training data size: {training_data.shape}",
      f"Validation data size: {validation_data.shape}")


# Configurando Hyperparametros

EPOCHS        = 1000
DROPOUT       = 0.2
DIRECTIONS    = 1
NUM_LAYERS    = 2
BATCH_SIZE    = 5
OUTPUT_SIZE   = 1
SEQ_LENGTH    = 60
NUM_FEATURES  = 6
HIDDEN_SIZE   = 100
LEARNING_RATE = 0.0001
STATE_DIM     = NUM_LAYERS * DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE
TARGET        = 'Target'
FEATURES      = ['Close','High', 'Low', 'Open', 'VWAP', 'Volume']

# Dataset

class CryptoDataset(Dataset):
    """Onchain dataset."""

    def __init__(self, csv_file, seq_length, features, target):
        """
        Args:
        """
        self.csv_file = csv_file
        self.target = target
        self.features = features
        self.seq_length = seq_length
        self.data_length = len(csv_file)

        self.metrics = self.create_xy_pairs()

    def create_xy_pairs(self):
        pairs = []
        for idx in range(self.data_length - self.seq_length):
            x = self.csv_file[idx:idx + self.seq_length][self.features].values
            y = self.csv_file[idx + self.seq_length:idx + self.seq_length + 1][self.target].values
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]


params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'drop_last': True, # Dispensando ultima batch incompleta
          'num_workers': 2}

params_test = {'batch_size': 1,
          'shuffle': False,
          'drop_last': False, # Dispensando ultima batch incompleta
          'num_workers': 2}

training_ds = CryptoDataset(training_data, SEQ_LENGTH, FEATURES, TARGET)
training_dl = DataLoader(training_ds, **params)

validation_ds = CryptoDataset(validation_data, SEQ_LENGTH, FEATURES, TARGET)
validation_dl = DataLoader(validation_ds, **params)

# Transferindo pro acelerador
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, directions=1):
    super(LSTM, self).__init__()

    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.directions = directions

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
    self.dropout = nn.Dropout(dropout_prob)
    self.linear = nn.Linear(hidden_size, output_size)

  def init_hidden_states(self, batch_size):
    state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
    return (torch.zeros(state_dim).to(device), torch.zeros(state_dim).to(device))

  def forward(self, x, states):
    x, (h, c) = self.lstm(x, states)
    out = self.linear(x)
    return out, (h, c)

model = LSTM(
    NUM_FEATURES,
    HIDDEN_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    DROPOUT
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.linear.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


def save_checkpoint(epoch, min_val_loss, model_state, opt_state):
  print(f"New minimum reached at epoch #{epoch + 1}, saving model state...")
  checkpoint = {
    'epoch': epoch + 1,
    'min_val_loss': min_val_loss,
    'model_state': model_state,
    'opt_state': opt_state,
  }
  torch.save(checkpoint, "./model_state.pt")


def load_checkpoint(path, model, optimizer):
    # load check point
    checkpoint = torch.load(path)
    min_val_loss = checkpoint["min_val_loss"]
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["opt_state"])
    return model, optimizer, checkpoint["epoch"], min_val_loss


def training(model, epochs, validate_every=2):

  training_losses = []
  validation_losses = []
  min_validation_loss = np.Inf

  # Chamando o train mode
  model.train()

  for epoch in tqdm(range(epochs)):

    # Initialize hidden and cell states with dimension:
    # (num_layers * num_directions, batch, hidden_size)
    states = model.init_hidden_states(BATCH_SIZE)
    running_training_loss = 0.0

    # Begin training
    for idx, (x_batch, y_batch) in enumerate(training_dl):
      # Convert to Tensors
      x_batch = x_batch.float().to(device)
      y_batch = y_batch.float().to(device)
      
      # Truncated Backpropagation
      states = [state.detach() for state in states]          

      optimizer.zero_grad()

      # Make prediction
      output, states = model(x_batch, states)

      # Calculate loss
      loss = criterion(output[:, -1, :], y_batch)
      loss.backward()
      running_training_loss += loss.item()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optimizer.step()
        
    # Average loss across timesteps
    training_losses.append(running_training_loss / len(training_dl))
        
    if epoch % validate_every == 0:

      # Set to eval mode
      model.eval()

      validation_states = model.init_hidden_states(BATCH_SIZE)
      running_validation_loss = 0.0

      for idx, (x_batch, y_batch) in enumerate(validation_dl):

        # Convert to Tensors
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().to(device)
      
        validation_states = [state.detach() for state in validation_states]
        output, validation_states = model(x_batch, validation_states)
        validation_loss = criterion(output[:, -1, :], y_batch)
        running_validation_loss += validation_loss.item()
        
    validation_losses.append(running_validation_loss / len(validation_dl))
    # Reset to training mode
    model.train()

    is_best = running_validation_loss / len(validation_dl) < min_validation_loss

    if is_best:
      min_validation_loss = running_validation_loss / len(validation_dl)
      save_checkpoint(epoch + 1, min_validation_loss, model.state_dict(), optimizer.state_dict())
        

  # Visualize loss
  epoch_count = range(1, len(training_losses) + 1)
  plt.plot(epoch_count, training_losses, 'r--')
  plt.legend(['Training Loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

  val_epoch_count = range(1, len(validation_losses) + 1)
  plt.plot(val_epoch_count, validation_losses, 'b--')
  plt.legend(['Validation loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

training(model, 100)


path = "./model_state.pt"
model, optimizer, start_epoch, valid_loss_min = load_checkpoint(path, model, optimizer)
print("model = ", model)
print("optimizer = ", optimizer)
print("start_epoch = ", start_epoch)
print("valid_loss_min = ", valid_loss_min)
print("valid_loss_min = {:.6f}".format(valid_loss_min))