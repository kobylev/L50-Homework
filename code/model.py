import torch
import torch.nn as nn

class LSTMFilter(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=1):
        super(LSTMFilter, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def prune_units(self, unit_indices):
        """Zeroes out linear weights to simulate component failure."""
        with torch.no_grad():
            self.fc.weight[:, unit_indices] = 0.0
            self.fc.bias[:] = 0.0
