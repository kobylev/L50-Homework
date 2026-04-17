import torch
import torch.nn as nn

class LSTMFilter(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=1):
        super(LSTMFilter, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden=None):
        # x is (batch_size, seq_len, input_dim)
        out, hidden = self.lstm(x, hidden)
        # out is (batch_size, seq_len, hidden_dim)
        out = self.fc(out) 
        # out is (batch_size, seq_len, 1)
        return out, hidden