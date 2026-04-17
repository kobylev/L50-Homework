import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import FREQS, WINDOW_SIZE, DOCS_DIR, DEVICE

def evaluate_all_frequencies(model, test_loader, L=1):
    model.eval()
    model.to(DEVICE)
    
    mse_per_freq = {f: [] for f in FREQS}
    mae_per_freq = {f: [] for f in FREQS}
    
    # We want to save an example plot for each frequency
    plotted_freqs = set()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, _ = model(inputs)
            
            # Figure out which frequency is being requested from the Control Vector (one-hot)
            # Input shape: (Batch, Seq, 5). Control is bits 1-4.
            # Just look at the first sample in the batch's control vector at time step 0.
            ctrl = inputs[0, 0, 1:]
            f_idx = torch.argmax(ctrl).item()
            f = FREQS[f_idx]
            
            mse = torch.mean((outputs - targets)**2).item()
            mae = torch.mean(torch.abs(outputs - targets)).item()
            
            mse_per_freq[f].append(mse)
            mae_per_freq[f].append(mae)
            
            # Plot one example per frequency
            if f not in plotted_freqs:
                plt.figure(figsize=(10, 4))
                plt.plot(targets[0, :, 0].cpu(), label='Clean Ground Truth', color='blue', alpha=0.8)
                plt.plot(outputs[0, :, 0].cpu(), label='LSTM Prediction', color='red', linestyle='--', alpha=0.8)
                plt.title(f"Target Extraction: {f}Hz (L={L})")
                plt.legend()
                plt.savefig(os.path.join(DOCS_DIR, f'prediction_{f}Hz_L{L}.png'))
                plt.close()
                plotted_freqs.add(f)
                
    # Average results
    print(f"\nQuantitative Evaluation (L={L}):")
    print(f"{'Freq (Hz)':<10} | {'MSE':<10} | {'MAE':<10}")
    print("-" * 35)
    for f in FREQS:
        avg_mse = np.mean(mse_per_freq[f])
        avg_mae = np.mean(mae_per_freq[f])
        print(f"{f:<10} | {avg_mse:.6f} | {avg_mae:.6f}")

def run_ablation_study(model, test_loader):
    """
    Ablation Study:
    1. Identify hidden units that are highly sensitive to specific frequencies.
    2. Prune them and show frequency-specific failure.
    """
    print("\nStarting Ablation Study...")
    model.eval()
    model.to(DEVICE)
    
    # Collect all activations for analysis
    all_out = []
    all_ctrl = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            out, _ = model.lstm(inputs)
            all_out.append(out.cpu())
            all_ctrl.append(inputs[:, 0, 1:].cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())
            if i > 10: break # Use enough data for analysis
            
    all_out = torch.cat(all_out)
    all_ctrl = torch.cat(all_ctrl)
    all_targets = torch.cat(all_targets)
    all_inputs = torch.cat(all_inputs)
    
    # 1Hz is index 0, 7Hz is index 3
    f1_mask = (torch.argmax(all_ctrl, dim=1) == 0)
    f7_mask = (torch.argmax(all_ctrl, dim=1) == 3)
    
    if not any(f1_mask) or not any(f7_mask):
        print("Ablation study failed: Could not find both 1Hz and 7Hz in the test set.")
        return

    # Mean activation per hidden unit for 1Hz vs 7Hz
    act_f1 = all_out[f1_mask].mean(dim=(0, 1)) 
    act_f7 = all_out[f7_mask].mean(dim=(0, 1)) 
    
    # Neurons that are much more active for 1Hz than 7Hz
    diff = act_f1 - act_f7
    important_for_f1 = torch.topk(diff, k=30).indices.tolist()
    
    print(f"Pruning {len(important_for_f1)} units identified as sensitive to 1Hz...")
    import copy
    pruned_model = copy.deepcopy(model)
    pruned_model.prune_units(important_for_f1)
    
    # Show 1Hz failure but 7Hz survival
    with torch.no_grad():
        # Prediction for 1Hz
        x1 = all_inputs[f1_mask][0:1].to(DEVICE)
        y1_true = all_targets[f1_mask][0:1]
        y1_pred, _ = pruned_model(x1)
        
        # Prediction for 7Hz
        x7 = all_inputs[f7_mask][0:1].to(DEVICE)
        y7_true = all_targets[f7_mask][0:1]
        y7_pred, _ = pruned_model(x7)
        
    # Plotting Ablation Results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(y1_true[0, :, 0], label='Clean 1Hz', color='blue')
    plt.plot(y1_pred[0, :, 0].cpu(), label='Pruned Prediction (Failed)', color='red', linestyle='--')
    plt.title("Ablation: Pruning 1Hz-Sensitive Neurons (Effect on 1Hz)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(y7_true[0, :, 0], label='Clean 7Hz', color='blue')
    plt.plot(y7_pred[0, :, 0].cpu(), label='Pruned Prediction (Survived)', color='green', linestyle='--')
    plt.title("Ablation: Pruning 1Hz-Sensitive Neurons (Effect on 7Hz)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'ablation_study.png'))
    plt.close()
    print("Ablation plot saved as docs/ablation_study.png")
