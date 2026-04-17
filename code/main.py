import torch
from config import BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, L_VALUES
from datasets import get_dataloaders
from model import LSTMFilter
from train import train_model
from evaluate import evaluate_all_frequencies, run_ablation_study

def main():
    train_loader, test_loader, _ = get_dataloaders(BATCH_SIZE)
    for L in L_VALUES:
        model = train_model(LSTMFilter(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS), train_loader, test_loader, L=L)
        evaluate_all_frequencies(model, test_loader, L=L)
        if L == 100: run_ablation_study(model, test_loader)
    print("Done. Results in docs/.")

if __name__ == "__main__":
    main()
