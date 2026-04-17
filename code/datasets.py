import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from config import FS, DURATION, FREQS, WINDOW_SIZE, DOCS_DIR, SEED

def generate_signals(seed, fs=1000, duration=10):
    np.random.seed(seed) # Explicit seed per generation to guarantee train/test independence
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean_signals = []
    noisy_signals = []
    
    for f in FREQS:
        clean = np.sin(2 * np.pi * f * t)
        clean_signals.append(clean)
        
        # Independent noise per frequency: amplitude (0.8-1.2) and phase (0-2pi)
        amp = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        noisy = amp * np.sin(2 * np.pi * f * t + phase)
        noisy_signals.append(noisy)
        
    clean_signals = np.array(clean_signals)
    noisy_signals = np.array(noisy_signals)
    
    # Combined signal: Sum and normalize by 4
    combined_noisy = np.sum(noisy_signals, axis=0) / 4.0
    combined_clean = np.sum(clean_signals, axis=0) / 4.0
    
    return t, clean_signals, combined_noisy, combined_clean

class SignalDataset(Dataset):
    def __init__(self, combined_noisy, clean_signals, window_size=100, samples_per_freq=1000):
        self.window_size = window_size
        self.samples = []
        n_total = len(combined_noisy)
        
        for f_idx in range(len(FREQS)):
            # Randomly sample starting points for context windows
            indices = np.random.choice(n_total - window_size, samples_per_freq, replace=False)
            for start_idx in indices:
                x_win = combined_noisy[start_idx : start_idx + window_size]
                # Each timestep gets the same one-hot control vector
                c_vec = np.zeros((window_size, 4), dtype=np.float32)
                c_vec[:, f_idx] = 1.0
                # Shape: (window_size, 1 + 4)
                x = np.column_stack((x_win, c_vec))
                # Target is the FULL matching clean target-frequency window
                y = clean_signals[f_idx, start_idx : start_idx + window_size].reshape(-1, 1)
                self.samples.append((x.astype(np.float32), y.astype(np.float32)))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def get_dataloaders(batch_size=64, L=1):
    # Train set generation
    t_train, clean_train, noisy_train, c_clean_train = generate_signals(SEED, FS, DURATION)
    # Test set generation with a separate seed for independence
    t_test, clean_test, noisy_test, c_clean_test = generate_signals(SEED + 999, FS, DURATION)
    
    train_dataset = SignalDataset(noisy_train, clean_train, WINDOW_SIZE, samples_per_freq=2000)
    test_dataset = SignalDataset(noisy_test, clean_test, WINDOW_SIZE, samples_per_freq=500)
    
    # shuffle is required to be False for L > 1 to avoid hidden state leakage across non-contiguous windows
    shuffle_train = False if L > 1 else True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Foundational plots
    plt.figure(figsize=(12, 4))
    plt.plot(t_train[:1000], c_clean_train[:1000], label='Combined Clean', alpha=0.8)
    plt.plot(t_train[:1000], noisy_train[:1000], label='Combined Noisy (Inputs)', alpha=0.5, linestyle='--')
    plt.title("Combined Clean vs. Combined Noisy (Normalized by 4)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DOCS_DIR, 'combined_signals.png'))
    plt.close()
    
    return train_loader, test_loader

def get_test_loader_fn(batch_size=64):
    """Returns a function that generates a test loader for a specific seed."""
    def _get_test_loader(seed):
        _, clean_test, noisy_test, _ = generate_signals(seed, FS, DURATION)
        test_dataset = SignalDataset(noisy_test, clean_test, WINDOW_SIZE, samples_per_freq=500)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return _get_test_loader

def plot_noise_histogram():
    """Generates visual proof of noise independence between train and test sets."""
    _, clean_train, noisy_train, _ = generate_signals(SEED, FS, DURATION)
    _, clean_test, noisy_test, _ = generate_signals(SEED + 999, FS, DURATION)
    
    noise_train = (noisy_train - clean_train).flatten()
    noise_test = (noisy_test - clean_test).flatten()
    
    plt.figure(figsize=(8, 5))
    plt.hist(noise_train, bins=50, alpha=0.5, label=f'Train Noise (Seed {SEED})', density=True, color='blue')
    plt.hist(noise_test, bins=50, alpha=0.5, label=f'Test Noise (Seed {SEED + 999})', density=True, color='orange')
    plt.title('Train vs Test Noise Distribution (Independence Proof)')
    plt.xlabel('Noise Amplitude')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DOCS_DIR, 'noise_histogram.png'))
    plt.close()
    print("Saved noise_histogram.png")
