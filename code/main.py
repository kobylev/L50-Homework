import torch
from config import BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, L_VALUES
from datasets import get_dataloaders
from model import LSTMFilter
from train import train_model
from evaluate import evaluate_all_frequencies, run_ablation_study

def main():
    # 1. Prepare Data
    print("Preparing Datasets (Independent Train/Test Noise)...")
    train_loader, test_loader, test_meta = get_dataloaders(BATCH_SIZE)
    
    # 2. Train and Evaluate for different L values
    for L in L_VALUES:
        print(f"\n--- Training with L={L} ---")
        model = LSTMFilter(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
        model = train_model(model, train_loader, test_loader, L=L)
        
        print(f"\nEvaluating All Target Frequencies (L={L})...")
        evaluate_all_frequencies(model, test_loader, L=L)
        
        # Run Ablation Study on the last trained model (usually L=100 is more stable)
        if L == 100:
            run_ablation_study(model, test_loader)

    print("\nAll tasks complete. Images saved in 'docs/' folder.")

if __name__ == "__main__":
    main()
