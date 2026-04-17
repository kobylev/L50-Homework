import torch
import os
from config import BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, L_VALUES
from datasets import get_dataloaders, get_test_loader_fn, plot_noise_histogram
from model import LSTMFilter
from train import train_model
from evaluate import evaluate_all_frequencies, run_ablation_study

def main():
    """
    Main entry point for the LSTM Frequency Filter project.
    Orchestrates data generation, model training, and evaluation.
    """
    print("Project: LSTM-Based Conditional Frequency Filtering")
    print("-" * 50)
    
    # Generate noise independence proof
    print("\nGenerating Noise Independence Proof...")
    plot_noise_histogram()
    
    # Iterative Training and Evaluation across L-parameters
    for L in L_VALUES:
        print(f"\nPhase 1: Dataset Generation (L={L})")
        # For L > 1, shuffling is automatically disabled in get_dataloaders
        train_loader, test_loader = get_dataloaders(BATCH_SIZE, L=L)
        
        print(f"\nPhase 2: Training Pipeline (Hidden State Reset Interval L={L})")
        model = LSTMFilter(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
        model = train_model(model, train_loader, test_loader, L=L)
        
        print(f"\nPhase 3: Quantitative and Qualitative Evaluation (L={L})")
        test_loader_fn = get_test_loader_fn(BATCH_SIZE)
        evaluate_all_frequencies(model, test_loader_fn, L=L)
        
        if L == 1: # Run ablation on the baseline model
            print(f"\nPhase 4: Targeted Ablation Study")
            run_ablation_study(model, test_loader)
        
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE: Results, metrics, and plots saved to the 'docs/' directory.")
    print("=" * 50)

if __name__ == "__main__":
    main()
