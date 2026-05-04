import torch

# Paths
TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"
TEST_DIR  = "data/test"

# Training Hyperparameters
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
VIT_LEARNING_RATE = 1e-4 # ViT typically needs a smaller learning rate than CNNs
NUM_WORKERS   = 0        # threads for data loading, concerned with mulithreading, used in data loading so if one 
                         # CPU thread is busy another could load data

# Model
IMG_SIZE      = 224      # resize all images to 224×224
NUM_CLASSES   = None     # will be set dynamically from data folder count
#CNN
CHANNELS = 32;
FEATURE_MAP_HEIGHT = 8;
FEATURE_MAP_WIDTH = 8; 
FEATURE_MAP_DEPTH = 8;
NUM_NEURONS = 128; #could be any number but preferably x2

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Work on CPU or GPU, for AI models, GPU is better if available

# Checkpointing
MODEL_SAVE_PATH = "models/best_model.pth"