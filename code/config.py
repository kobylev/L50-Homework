import os
import torch

# Reproducibility
SEED = 42

# Signal Parameters
FS = 1000
DURATION = 10  # 10 seconds
FREQS = [1, 3, 5, 7]
WINDOW_SIZE = 100  # Increased from 10 for better temporal receptive field

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.001
L_VALUES = [1, 100]  # Testing stateless (L=1) vs longer memory (L=100)
ALPHA_COSINE = 0.5  # Weight for Cosine Similarity in Custom Loss

# Model Architecture
INPUT_DIM = 5  # 1 signal + 4 control
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
