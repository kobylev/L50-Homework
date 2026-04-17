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
    Empirically identifies hidden units sensitive to specific dynamics, prunes them, 
    and visualizes the degradation in signal extraction across all frequencies.
    """
    print("\nExecuting Comprehensive Ablation Study for all frequencies...")
    model.eval()
    model.to(DEVICE)
    
    for f_idx, target_freq in enumerate(FREQS):
        # Collect activations
        activations_target = []
        activations_other = []
        
        x_target = None
        y_target_clean = None

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                h_seq, _ = model.lstm(inputs)
                
                for i in range(inputs.size(0)):
                    curr_f_idx = torch.argmax(inputs[i, 0, 1:]).item()
                    avg_act = torch.mean(torch.abs(h_seq[i]), dim=0).cpu().numpy()
                    
                    if curr_f_idx == f_idx:
                        activations_target.append(avg_act)
                        if x_target is None:
                            x_target = inputs[i:i+1]
                            y_target_clean = targets[i, :, 0].cpu().numpy()
                    else:
                        activations_other.append(avg_act)
                        
                if len(activations_target) > 50 and len(activations_other) > 150:
                    break

        mean_target = np.mean(activations_target, axis=0)
        mean_other = np.mean(activations_other, axis=0)
        diff = mean_target - mean_other
        
        target_units = np.argsort(diff)[-20:]
        
        with torch.no_grad():
            out_before, _ = model(x_target)
            y_before = out_before[0, :, 0].cpu().numpy()
        
        import copy
        pruned_model = copy.deepcopy(model)
        pruned_model.prune_units(target_units)
        with torch.no_grad():
            out_after, _ = pruned_model(x_target)
            y_after = out_after[0, :, 0].cpu().numpy()
            
        plt.figure(figsize=(10, 6))
        plt.plot(y_target_clean, label=f'Clean {target_freq}Hz Ground Truth', color='blue', alpha=0.4)
        plt.plot(y_before, label='Prediction (Before Pruning)', color='red', linestyle='--')
        plt.plot(y_after, label='Prediction (After Targeted Pruning)', color='black', linewidth=2)
        plt.title(f"Ablation Study: Targeted Suppression of {target_freq}Hz Extraction")
        plt.xlabel("Timesteps")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(DOCS_DIR, f'ablation_{target_freq}Hz.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Ablation evidence saved to {plot_path}")
