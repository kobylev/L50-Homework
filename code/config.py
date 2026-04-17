import os
import torch

# Reproducibility
SEED = 42

# Signal Parameters
FS = 1000
DURATION = 10
FREQS = [1, 3, 5, 7]
WINDOW_SIZE = 100  # Expanded temporal receptive field

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 80
LR = 0.001
L_VALUES = [1, 100]
ALPHA_COSINE = 0.5  # Weight for phase/shape coherence

# Model Architecture
INPUT_DIM = 5
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Paths
BASE_DIR = "C:/Ai_Expert/L50-Homework"
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
