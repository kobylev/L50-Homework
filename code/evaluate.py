import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import FREQS, DOCS_DIR, DEVICE

def evaluate_all_frequencies(model, test_loader, L=1):
    """Generates visual reconstruction plots and quantitative MSE metrics."""
    model.eval()
    model.to(DEVICE)
    plotted_freqs = set()
    
    mse_per_freq = {f: [] for f in FREQS}
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, _ = model(inputs)
            
            # Unpack batch for per-frequency analysis
            for i in range(inputs.size(0)):
                f_idx = torch.argmax(inputs[i, 0, 1:]).item()
                f = FREQS[f_idx]
                
                # Quantitative Metric
                single_mse = torch.mean((outputs[i] - targets[i])**2).item()
                mse_per_freq[f].append(single_mse)
                
                # Visual Plot (one per frequency)
                if f not in plotted_freqs:
                    plt.figure(figsize=(10, 4))
                    # Extract single sequence for plotting
                    t_clean = targets[i, :, 0].cpu().numpy()
                    t_pred = outputs[i, :, 0].cpu().numpy()
                    
                    plt.plot(t_clean, label='Clean Ground Truth', color='blue', alpha=0.8)
                    plt.plot(t_pred, label='Model Prediction', color='red', linestyle='--', alpha=0.8)
                    plt.title(f"Target Frequency: {f}Hz Reconstruction (L={L})")
                    plt.xlabel("Timesteps (Context Window)")
                    plt.ylabel("Normalized Amplitude")
                    plt.legend(loc='upper right')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(DOCS_DIR, f'prediction_{f}Hz_L{L}.png'))
                    plt.close()
                    plotted_freqs.add(f)
                
    # Print quantitative summary to console for README population
    print(f"\nQuantitative Summary (L={L})")
    print("-" * 40)
    print(f"{'Frequency (Hz)':<15} | {'Test MSE':<10}")
    print("-" * 40)
    for f in FREQS:
        avg_mse = np.mean(mse_per_freq[f])
        print(f"{f:<15} | {avg_mse:.6f}")
    print("-" * 40)

def run_ablation_study(model, test_loader):
    """
    DEPRECATED: Ablation study removed as magnitude-based neuron analysis 
    lacks causal rigor in LSTM temporal dynamics. 
    Included as placeholder to maintain function signature compatibility.
    """
    print("Ablation study skipped: Insufficient causal evidence for component failure claims.")
    pass
