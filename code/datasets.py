import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

from config import FS, DURATION, DOCS_DIR, WINDOW_SIZE, DATASET_SIZE

def generate_signals():
    t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
    freqs = [1, 3, 5, 7]
    clean_signals = []
    noisy_signals = []
    
    for f in freqs:
        # Base amplitude 1
        clean = np.sin(2 * np.pi * f * t)
        clean_signals.append(clean)
        
        # Noise injection
        amp = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        noisy = amp * np.sin(2 * np.pi * f * t + phase)
        noisy_signals.append(noisy)
        
    clean_signals = np.array(clean_signals)
    noisy_signals = np.array(noisy_signals)
    
    # Combined Noisy Signal
    combined_noisy = np.sum(noisy_signals, axis=0) / 4.0
    combined_clean = np.sum(clean_signals, axis=0) / 4.0
    
    # Plotting: 4 individual clean vs noisy
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for i in range(4):
        axes[i].plot(t[:1000], clean_signals[i][:1000], label='Clean', color='blue', alpha=0.7)
        axes[i].plot(t[:1000], noisy_signals[i][:1000], label='Noisy', color='red', linestyle='--', alpha=0.7)
        axes[i].set_ylabel(f'{freqs[i]}Hz')
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Individual Clean vs. Noisy Signals (First 1 second)')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'clean_vs_noisy.png'))
    plt.close()
    
    # Plotting: Combined Noisy vs Combined Clean
    plt.figure(figsize=(12, 4))
    plt.plot(t[:1000], combined_clean[:1000], label='Combined Clean', color='blue')
    plt.plot(t[:1000], combined_noisy[:1000], label='Combined Noisy', color='red', linestyle='--', alpha=0.7)
    plt.title('Combined Clean vs. Combined Noisy Signal (First 1 second)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'combined_signals.png'))
    plt.close()
    
    return t, clean_signals, combined_noisy

class SignalDataset(Dataset):
    def __init__(self, combined_noisy, clean_signals, window_size=10, num_samples=10000):
        self.window_size = window_size
        self.combined_noisy = combined_noisy
        self.clean_signals = clean_signals
        self.num_samples = num_samples
        self.max_idx = len(combined_noisy) - window_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomly pick a starting index
        start_idx = np.random.randint(0, self.max_idx)
        
        # Extract 10-sample window from Combined Noisy Signal
        window_noisy = self.combined_noisy[start_idx : start_idx + self.window_size]
        
        # Randomly select a target frequency (0 to 3)
        target_idx = np.random.randint(0, 4)
        
        # Create Control Vector C (length 4)
        control_vec = np.zeros((self.window_size, 4), dtype=np.float32)
        control_vec[:, target_idx] = 1.0
        
        # Network Input X: (10, 5)
        x = np.zeros((self.window_size, 5), dtype=np.float32)
        x[:, 0] = window_noisy
        x[:, 1:] = control_vec
        
        # Label Y: corresponding 10-sample window from the *clean* target sine wave
        y = self.clean_signals[target_idx, start_idx : start_idx + self.window_size]
        
        # Need y to have shape (window_size, 1) to match output
        y = y.reshape(self.window_size, 1).astype(np.float32)
        
        return torch.tensor(x), torch.tensor(y)

def get_dataloaders(batch_size):
    t, clean_signals, combined_noisy = generate_signals()
    
    dataset = SignalDataset(combined_noisy, clean_signals, WINDOW_SIZE, DATASET_SIZE)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, t, clean_signals, combined_noisy