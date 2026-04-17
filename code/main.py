import torch
import os
from config import BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, L_VALUES
from datasets import get_dataloaders
from model import LSTMFilter
from train import train_model
from evaluate import evaluate_all_frequencies

def main():
    """
    Main entry point for the LSTM Frequency Filter project.
    Orchestrates data generation, model training, and evaluation.
    """
    print("Project: LSTM-Based Conditional Frequency Filtering")
    print("-" * 50)
    
    # Initialize Data
    print("\nPhase 1: Dataset Generation")
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    print("Train/Test Dataloaders initialized with independent noise seeds.")
    
    # Iterative Training and Evaluation across L-parameters
    for L in L_VALUES:
        print(f"\nPhase 2: Training Pipeline (Hidden State Reset Interval L={L})")
        model = LSTMFilter(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
        model = train_model(model, train_loader, test_loader, L=L)
        
        print(f"\nPhase 3: Quantitative and Qualitative Evaluation (L={L})")
        evaluate_all_frequencies(model, test_loader, L=L)
        
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE: Results, metrics, and plots saved to the 'docs/' directory.")
    print("=" * 50)

if __name__ == "__main__":
    main()
