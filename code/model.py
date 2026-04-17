import torch
import torch.nn as nn

class LSTMFilter(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=1):
        """
        Architecture:
        - Input: (Batch, Seq, Features) where Features = [Composite Signal, 4-D Control Vector]
        - LSTM: Extracts temporal dependencies.
        - FC: Maps hidden state at each timestep to clean amplitude prediction.
        """
        super(LSTMFilter, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden=None):
        # out shape: (Batch, Seq, Hidden)
        out, hidden = self.lstm(x, hidden)
        # Predict at every timestep (Sequence-to-Sequence)
        out = self.fc(out)
        return out, hidden
