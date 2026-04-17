import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from config import EPOCHS, LR, DOCS_DIR, DEVICE

def train_model(model, train_loader, val_loader, L=1):
    model.to(DEVICE)
    # Using standard MSE loss as per academic requirements
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        hidden = None
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Reset hidden state every L batches to test memory truncation
            if L == 1 or i % L == 0:
                hidden = None
            else:
                # Carry hidden state but detach from previous computational graph
                hidden = (hidden[0].detach(), hidden[1].detach())
            
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Validation phase
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs, _ = model(inputs)
                v_loss += criterion(outputs, targets).item()
        avg_val = v_loss / len(val_loader)
        val_losses.append(avg_val)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")
            
    # Save training trajectory
    plt.figure()
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Val MSE')
    plt.title(f'Learning Curve (L={L})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DOCS_DIR, f'loss_L{L}.png'))
    plt.close()
    
    return model
