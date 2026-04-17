import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from config import FS, DURATION, FREQS, WINDOW_SIZE, DOCS_DIR, SEED

def generate_independent_signals(fs=1000, duration=10):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean_signals = []
    noisy_signals = []
    
    for f in FREQS:
        clean = np.sin(2 * np.pi * f * t)
        clean_signals.append(clean)
        
        # Independent noise injection
        amp = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        noisy = amp * np.sin(2 * np.pi * f * t + phase)
        noisy_signals.append(noisy)
        
    clean_signals = np.array(clean_signals)
    noisy_signals = np.array(noisy_signals)
    
    combined_noisy = np.sum(noisy_signals, axis=0) / len(FREQS)
    combined_clean = np.sum(clean_signals, axis=0) / len(FREQS)
    
    return t, clean_signals, combined_noisy, combined_clean

class SignalDataset(Dataset):
    def __init__(self, combined_noisy, clean_signals, window_size=100, samples_per_freq=1000):
        self.window_size = window_size
        self.samples = []
        n_total = len(combined_noisy)
        
        for f_idx in range(len(FREQS)):
            indices = np.random.choice(n_total - window_size, samples_per_freq, replace=False)
            for start_idx in indices:
                x_win = combined_noisy[start_idx : start_idx + window_size]
                c_vec = np.zeros((window_size, 4), dtype=np.float32)
                c_vec[:, f_idx] = 1.0
                x = np.column_stack((x_win, c_vec))
                y = clean_signals[f_idx, start_idx : start_idx + window_size].reshape(-1, 1)
                self.samples.append((x.astype(np.float32), y.astype(np.float32)))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def get_dataloaders(batch_size=64):
    np.random.seed(SEED)
    t_train, clean_train, noisy_train, combined_clean_train = generate_independent_signals(FS, DURATION)
    t_test, clean_test, noisy_test, combined_clean_test = generate_independent_signals(FS, DURATION)
    
    train_dataset = SignalDataset(noisy_train, clean_train, WINDOW_SIZE, samples_per_freq=2000)
    test_dataset = SignalDataset(noisy_test, clean_test, WINDOW_SIZE, samples_per_freq=500)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Pre-generate foundational plots
    plt.figure(figsize=(10, 6))
    for i in range(len(FREQS)):
        plt.subplot(len(FREQS), 1, i+1)
        plt.plot(t_train[:1000], clean_train[i][:1000], label='Clean')
        plt.title(f"{FREQS[i]}Hz Signal")
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'clean_signals.png'))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t_train[:1000], combined_clean_train[:1000], label='Combined Clean', alpha=0.8)
    plt.plot(t_train[:1000], noisy_train[:1000], label='Combined Noisy', alpha=0.5, linestyle='--')
    plt.title("Combined Clean vs. Combined Noisy (Sum/4)")
    plt.savefig(os.path.join(DOCS_DIR, 'combined_signals.png'))
    plt.close()
    
    return train_loader, test_loader, (t_test, clean_test, noisy_test, combined_clean_test)
