import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# 1. Data Generation
def generate_data(fs=1000, duration=10):
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    freqs = [1, 3, 5, 7]
    clean_signals = []
    noisy_signals = []
    
    for f in freqs:
        # Clean signal
        clean = np.sin(2 * np.pi * f * t)
        clean_signals.append(clean)
        
        # Noise injection: random amplitude (0.8 to 1.2) and random phase (0 to 2pi)
        amp = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        noisy = amp * np.sin(2 * np.pi * f * t + phase)
        noisy_signals.append(noisy)
        
    # Composite signal: Sum and normalize by 4
    composite = np.sum(noisy_signals, axis=0) / 4.0
    
    return t, np.array(clean_signals), composite

# 2. Dataset Building
class SignalDataset(Dataset):
    def __init__(self, composite, clean_signals, window_size=10):
        self.window_size = window_size
        self.composite = composite
        self.clean_signals = clean_signals
        self.n_samples = len(composite) - window_size
        
        # Generate control vectors (randomly requested signal for each target point)
        # For training, we can cycle through control vectors or pick randomly.
        # To make it a full dataset, we'll create samples for each possible control vector at each time step
        # or just random ones. Let's do random to keep it simple but representative.
        self.controls = np.random.randint(0, 4, self.n_samples)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Input: sequence of 10 samples + 4D one-hot control vector
        # We'll concatenate the control vector to each sample in the sequence
        window = self.composite[idx : idx + self.window_size]
        control_idx = self.controls[idx]
        
        # Create one-hot control vector
        control_vec = np.zeros(4, dtype=np.float32)
        control_vec[control_idx] = 1.0
        
        # Construct input sequence (window_size, 1 + 4)
        # Each step in the sequence gets the same control vector
        x = np.zeros((self.window_size, 5), dtype=np.float32)
        x[:, 0] = window
        x[:, 1:] = control_vec
        
        # Target: clean value of the requested sine wave at the LAST time step of the window
        target = self.clean_signals[control_idx, idx + self.window_size - 1]
        
        return torch.tensor(x), torch.tensor(target, dtype=torch.float32).unsqueeze(0)

# 3. Model Architecture
class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        # We take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out, hidden

# 4. Training and Evaluation
def train_model(model, train_loader, val_loader, epochs=50, L=1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        hidden = None
        
        for i, (inputs, targets) in enumerate(train_loader):
            # Reset hidden state every L batches
            if L == 1 or i % L == 0:
                hidden = None
            else:
                # Detach hidden state to prevent exploding gradients/long backprop
                hidden = (hidden[0].detach(), hidden[1].detach())
            
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_losses.append(running_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs, _ = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
            
    return train_losses, val_losses

def main():
    # Parameters
    FS = 1000
    DURATION = 10
    WINDOW_SIZE = 10
    L = 1
    BATCH_SIZE = 64
    EPOCHS = 50
    
    # Generate data
    print("Generating data...")
    t, clean_signals, composite = generate_data(FS, DURATION)
    
    # Create dataset
    dataset = SignalDataset(composite, clean_signals, WINDOW_SIZE)
    
    # Split 80/20 (Sequential split for L > 1 to be meaningful)
    train_size = int(0.8 * len(dataset))
    # We create subsets based on sequential indices
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # DataLoaders
    # For L > 1 to work, we MUST NOT shuffle the data during training
    # so that the hidden state can be meaningfully carried between batches.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=(L==1))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = LSTMModel()
    
    # Train
    print(f"Starting training with L={L}...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, epochs=EPOCHS, L=L)
    
    # Visualization
    print("Visualizing results...")
    model.eval()
    
    # Pick a specific sine wave to predict for the whole duration for visualization
    # Let's pick the 3Hz wave (index 1)
    target_wave_idx = 1
    predictions = []
    actual_clean = []
    
    # Prepare a sequential test set for visualization
    with torch.no_grad():
        for i in range(0, 500): # Plot first 0.5 seconds
            window = composite[i : i + WINDOW_SIZE]
            control_vec = np.zeros(4, dtype=np.float32)
            control_vec[target_wave_idx] = 1.0
            
            x = np.zeros((WINDOW_SIZE, 5), dtype=np.float32)
            x[:, 0] = window
            x[:, 1:] = control_vec
            
            x_tensor = torch.tensor(x).unsqueeze(0)
            output, _ = model(x_tensor)
            predictions.append(output.item())
            actual_clean.append(clean_signals[target_wave_idx, i + WINDOW_SIZE - 1])
            
    plt.figure(figsize=(12, 6))
    plt.plot(t[WINDOW_SIZE-1:500+WINDOW_SIZE-1], actual_clean, label="Clean Signal (3Hz)", color='blue')
    plt.plot(t[WINDOW_SIZE-1:500+WINDOW_SIZE-1], predictions, label="Predicted Signal", color='red', linestyle='--')
    plt.title(f"Clean vs Predicted Signal (L={L})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_plot.png")
    plt.show()
    print("Plot saved as prediction_plot.png")

if __name__ == "__main__":
    main()
