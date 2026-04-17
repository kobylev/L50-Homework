import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

from config import FS, DURATION, FREQS, SEED, BATCH_SIZE, EPOCHS, LR, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DEVICE, DOCS_DIR
from datasets import generate_signals, SignalDataset
from torch.utils.data import DataLoader
from model import LSTMFilter

def train_for_window(window_size):
    t_train, clean_train, noisy_train, _ = generate_signals(SEED, FS, DURATION)
    t_test, clean_test, noisy_test, _ = generate_signals(SEED + 999, FS, DURATION)
    
    train_dataset = SignalDataset(noisy_train, clean_train, window_size, samples_per_freq=1000)
    test_dataset = SignalDataset(noisy_test, clean_test, window_size, samples_per_freq=250)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LSTMFilter(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    val_losses = []
    for epoch in range(25): # Quick evaluation for ablation
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs, _ = model(inputs)
                v_loss += criterion(outputs, targets).item()
        val_losses.append(v_loss / len(test_loader))
        
    return val_losses

def main():
    print("Running Window Size Ablation (W=10, 50, 100)...")
    window_sizes = [10, 50, 100]
    final_mses = []
    
    for w in window_sizes:
        print(f"Training for window size {w}...")
        val_losses = train_for_window(w)
        # Average of the last 5 epochs to stabilize metric
        final_mse = np.mean(val_losses[-5:])
        final_mses.append(final_mse)
        print(f"Final Validation MSE for W={w}: {final_mse:.6f}")
    
    plt.figure(figsize=(8, 5))
    plt.bar([str(w) for w in window_sizes], final_mses, color=['red', 'orange', 'blue'])
    plt.title('Validation MSE vs Window Size')
    plt.xlabel('Temporal Window Size (samples)')
    plt.ylabel('Final Mean Squared Error (MSE)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, mse in enumerate(final_mses):
        plt.text(i, mse + 0.001, f"{mse:.4f}", ha='center')
        
    plt.savefig(os.path.join(DOCS_DIR, 'window_size_ablation.png'))
    plt.close()
    print("Saved docs/window_size_ablation.png")

if __name__ == "__main__":
    main()
