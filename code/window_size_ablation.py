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
    print("Running Window Size Ablation (W=10 vs W=100)...")
    val_loss_10 = train_for_window(10)
    val_loss_100 = train_for_window(100)
    
    plt.figure(figsize=(8, 5))
    plt.plot(val_loss_10, label='WINDOW_SIZE = 10', color='red', linestyle='--')
    plt.plot(val_loss_100, label='WINDOW_SIZE = 100', color='blue')
    plt.title('Validation Loss: Window Size 10 vs 100')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DOCS_DIR, 'window_size_comparison.png'))
    plt.close()
    print("Saved window_size_comparison.png")

if __name__ == "__main__":
    main()
