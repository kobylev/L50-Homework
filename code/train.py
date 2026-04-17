import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from config import EPOCHS, LR, ALPHA_COSINE, DOCS_DIR, DEVICE

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        p = pred.squeeze(-1)
        t = target.squeeze(-1)
        cos_sim = F.cosine_similarity(p, t, dim=1).mean()
        cos_loss = 1.0 - cos_sim
        return (1 - self.alpha) * mse_loss + self.alpha * cos_loss

def train_model(model, train_loader, val_loader, L=1):
    model.to(DEVICE)
    criterion = HybridLoss(alpha=ALPHA_COSINE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, hidden = 0, None
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            if L == 1 or i % L == 0: hidden = None
            else: hidden = (hidden[0].detach(), hidden[1].detach())
            
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs, _ = model(inputs)
                v_loss += criterion(outputs, targets).item()
        avg_val = v_loss / len(val_loader)
        val_losses.append(avg_val)
        
        if (epoch + 1) % 20 == 0:
            print(f"L={L} Epoch {epoch+1} Train: {avg_train:.4f} Val: {avg_val:.4f}")
            
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'Loss (L={L})')
    plt.savefig(os.path.join(DOCS_DIR, f'loss_L{L}.png'))
    plt.close()
    return model
