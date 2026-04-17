import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import FREQS, DOCS_DIR, DEVICE

def evaluate_all_frequencies(model, test_loader_fn, seeds=[42, 101, 202, 303, 404], L=1):
    """Generates visual reconstruction plots and aggregates quantitative MSE metrics across seeds."""
    model.eval()
    model.to(DEVICE)
    
    # Visual plots only on the first seed
    plotted_freqs = set()
    
    # Dictionary to store MSE lists per frequency across all seeds
    all_mse_per_freq = {f: [] for f in FREQS}
    
    print(f"\nRunning Statistical Evaluation across {len(seeds)} seeds (L={L})...")
    
    for seed in seeds:
        test_loader = test_loader_fn(seed)
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
                    
                    # Visual Plot (only on first seed)
                    if seed == seeds[0] and f not in plotted_freqs:
                        plt.figure(figsize=(10, 4))
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
        
        # Store average MSE for this seed
        for f in FREQS:
            all_mse_per_freq[f].append(np.mean(mse_per_freq[f]))
                
    # Print quantitative summary with Mean ± Std Dev
    print(f"\nQuantitative Summary (L={L}) - Statistical Aggregation")
    print("-" * 55)
    print(f"{'Frequency (Hz)':<15} | {'Test MSE (Mean ± Std Dev)':<25}")
    print("-" * 55)
    for f in FREQS:
        mean_mse = np.mean(all_mse_per_freq[f])
        std_mse = np.std(all_mse_per_freq[f])
        print(f"{f:<15} | {mean_mse:.6f} ± {std_mse:.6f}")
    print("-" * 55)

def run_ablation_study(model, test_loader):
    """
    Empirically identifies hidden units sensitive to 1Hz dynamics, prunes them, 
    and visualizes the degradation in signal extraction.
    """
    print("\nExecuting Ablation Study...")
    model.eval()
    model.to(DEVICE)
    
    # Collect activations for 1Hz vs others
    activations_1hz = []
    activations_other = []
    target_f_idx = 0 # 1Hz
    
    # For visualization
    x_1hz = None
    y_1hz_clean = None

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            # Forward through LSTM but not FC
            h_seq, _ = model.lstm(inputs)
            # Shape: (Batch, Seq, Hidden)
            
            for i in range(inputs.size(0)):
                f_idx = torch.argmax(inputs[i, 0, 1:]).item()
                # Average activation across the sequence for each hidden unit
                avg_act = torch.mean(torch.abs(h_seq[i]), dim=0).cpu().numpy()
                
                if f_idx == target_f_idx:
                    activations_1hz.append(avg_act)
                    if x_1hz is None:
                        x_1hz = inputs[i:i+1]
                        y_1hz_clean = targets[i, :, 0].cpu().numpy()
                else:
                    activations_other.append(avg_act)
            if len(activations_1hz) > 50 and len(activations_other) > 150:
                break

    # Calculate difference in mean activations
    mean_1hz = np.mean(activations_1hz, axis=0)
    mean_other = np.mean(activations_other, axis=0)
    diff = mean_1hz - mean_other
    
    # Top 20 units most specialized for 1Hz
    target_units = np.argsort(diff)[-20:]
    
    # Inference BEFORE pruning
    with torch.no_grad():
        out_before, _ = model(x_1hz)
        y_before = out_before[0, :, 0].cpu().numpy()
    
    # Inference AFTER pruning
    import copy
    pruned_model = copy.deepcopy(model)
    pruned_model.prune_units(target_units)
    with torch.no_grad():
        out_after, _ = pruned_model(x_1hz)
        y_after = out_after[0, :, 0].cpu().numpy()
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(y_1hz_clean, label='Clean 1Hz Ground Truth', color='blue', alpha=0.4)
    plt.plot(y_before, label='Prediction (Before Pruning)', color='red', linestyle='--')
    plt.plot(y_after, label='Prediction (After Targeted Pruning)', color='black', linewidth=2)
    plt.title("Ablation Study: Targeted Suppression of 1Hz Extraction")
    plt.xlabel("Timesteps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DOCS_DIR, 'ablation_plot.png'))
    plt.close()
    print("Ablation evidence saved to docs/ablation_plot.png")
