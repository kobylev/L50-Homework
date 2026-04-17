import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from config import DOCS_DIR, WINDOW_SIZE

def evaluate_and_plot(model, t, clean_signals, combined_noisy):
    model.eval()
    
    # We will pick a continuous segment to visualize the network's prediction.
    # We will request frequency 3Hz (idx 1).
    target_idx = 1
    freq = 3
    
    start_time_idx = 0
    end_time_idx = 1000 # 1 second
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        # Using sliding window over the continuous segment
        for i in range(start_time_idx, end_time_idx - WINDOW_SIZE + 1):
            window_noisy = combined_noisy[i : i + WINDOW_SIZE]
            control_vec = np.zeros((WINDOW_SIZE, 4), dtype=np.float32)
            control_vec[:, target_idx] = 1.0
            
            x = np.zeros((WINDOW_SIZE, 5), dtype=np.float32)
            x[:, 0] = window_noisy
            x[:, 1:] = control_vec
            
            x_tensor = torch.tensor(x).unsqueeze(0) # (1, 10, 5)
            
            output, _ = model(x_tensor)
            
            # We take the prediction for the FIRST time step in the window to form a continuous line
            # actually taking the last time step is more conventional for causal filters, 
            # let's take the last step of the window: i + WINDOW_SIZE - 1
            predictions.append(output[0, -1, 0].item())
            ground_truth.append(clean_signals[target_idx, i + WINDOW_SIZE - 1])
            
    time_axis = t[WINDOW_SIZE - 1 : end_time_idx]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, ground_truth, label=f'Ground Truth Clean ({freq}Hz)', color='blue')
    plt.plot(time_axis, predictions, label='LSTM Prediction', color='red', linestyle='--')
    # highlight error margin
    error = np.abs(np.array(ground_truth) - np.array(predictions))
    plt.fill_between(time_axis, ground_truth, predictions, color='red', alpha=0.2, label='Error Margin')
    plt.title(f'LSTM Target Extraction: Predicted vs. Ground Truth ({freq}Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'prediction.png'))
    plt.close()