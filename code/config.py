import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

# Hyperparameters
FS = 1000  # Sampling rate in Hz
DURATION = 10  # Duration in seconds
N_SAMPLES = int(FS * DURATION)
WINDOW_SIZE = 10
DATASET_SIZE = 10000

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
L = 1  # Reset limit

# Model architecture
INPUT_DIM = 5  # 1 signal + 4 one-hot control vector
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Ensure docs dir exists
os.makedirs(DOCS_DIR, exist_ok=True)