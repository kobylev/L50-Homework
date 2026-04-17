from config import BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS
from datasets import get_dataloaders
from model import LSTMFilter
from train import train_model
from evaluate import evaluate_and_plot

def main():
    print("Initializing Data Loaders and Generating Signals...")
    train_loader, val_loader, t, clean_signals, combined_noisy = get_dataloaders(BATCH_SIZE)
    
    print("Initializing Model...")
    model = LSTMFilter(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    
    print("Starting Training...")
    model = train_model(model, train_loader, val_loader)
    
    print("Evaluating and Generating Plots...")
    evaluate_and_plot(model, t, clean_signals, combined_noisy)
    
    print("Process Complete. Check 'docs/' for outputs.")

if __name__ == "__main__":
    main()