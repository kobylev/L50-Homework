import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import FREQS, WINDOW_SIZE, DOCS_DIR, DEVICE

def evaluate_all_frequencies(model, test_loader, L=1):
    model.eval()
    model.to(DEVICE)
    plotted_freqs = set()
    
    mse_per_freq = {f: [] for f in FREQS}
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, _ = model(inputs)
            f_idx = torch.argmax(inputs[0, 0, 1:]).item()
            f = FREQS[f_idx]
            
            mse_per_freq[f].append(torch.mean((outputs - targets)**2).item())
            
            if f not in plotted_freqs:
                plt.figure(figsize=(10, 4))
                plt.plot(targets[0, :, 0].cpu(), label='Clean', alpha=0.8)
                plt.plot(outputs[0, :, 0].cpu(), label='Pred', alpha=0.8, linestyle='--')
                plt.title(f"Target: {f}Hz (L={L})")
                plt.legend()
                plt.savefig(os.path.join(DOCS_DIR, f'prediction_{f}Hz_L{L}.png'))
                plt.close()
                plotted_freqs.add(f)
                
    print(f"\nResults (L={L}):")
    for f in FREQS:
        print(f"{f}Hz: MSE {np.mean(mse_per_freq[f]):.6f}")

def run_ablation_study(model, test_loader):
    print("\nAblation Study...")
    model.eval()
    model.to(DEVICE)
    
    all_out, all_ctrl, all_targets, all_inputs = [], [], [], []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            out, _ = model.lstm(inputs)
            all_out.append(out.cpu())
            all_ctrl.append(inputs[:, 0, 1:].cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())
            if i > 10: break
            
    all_out = torch.cat(all_out)
    all_ctrl = torch.cat(all_ctrl)
    all_targets = torch.cat(all_targets)
    all_inputs = torch.cat(all_inputs)
    
    f1_mask = (torch.argmax(all_ctrl, dim=1) == 0)
    f7_mask = (torch.argmax(all_ctrl, dim=1) == 3)
    
    act_f1 = all_out[f1_mask].mean(dim=(0, 1)) 
    act_f7 = all_out[f7_mask].mean(dim=(0, 1)) 
    
    important_for_f1 = torch.topk(act_f1 - act_f7, k=30).indices.tolist()
    
    import copy
    pruned_model = copy.deepcopy(model)
    pruned_model.prune_units(important_for_f1)
    
    with torch.no_grad():
        x1, y1_true = all_inputs[f1_mask][0:1].to(DEVICE), all_targets[f1_mask][0:1]
        y1_pred, _ = pruned_model(x1)
        x7, y7_true = all_inputs[f7_mask][0:1].to(DEVICE), all_targets[f7_mask][0:1]
        y7_pred, _ = pruned_model(x7)
        
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(y1_true[0, :, 0], label='Clean 1Hz')
    plt.plot(y1_pred[0, :, 0].cpu(), label='Pruned', linestyle='--')
    plt.title("Ablation: Impact on 1Hz")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(y7_true[0, :, 0], label='Clean 7Hz')
    plt.plot(y7_pred[0, :, 0].cpu(), label='Survived', linestyle='--')
    plt.title("Ablation: Impact on 7Hz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'ablation_study.png'))
    plt.close()
