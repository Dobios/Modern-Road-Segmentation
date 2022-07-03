from yacs.config import CfgNode as CN

MASSACHUSETTS_PATH = 'data/massachusetts'
DEEPGLOBE_PATH = 'data/deepglobe'

# Set default config parameters
_C = CN()

_C.LOGDIR = "logs"
_C.EXP_NAME = "UNetTest"

# Dataset
_C.DATASET = CN()
_C.DATASET.BATCHSIZE = 32
_C.DATASET.WORKERS = 6 
_C.DATASET.TRAIN_DS = 'massachusetts,deepglobe'
_C.DATASET.IMG_SIZE = 224
_C.DATASET.STRIDE = _C.DATASET.IMG_SIZE // 2
_C.DATASET.MASSACHUSETTS = CN()
_C.DATASET.MASSACHUSETTS.PATH = MASSACHUSETTS_PATH
_C.DATASET.MASSACHUSETTS.TRAIN_SPLITS = 'train,val'
_C.DATASET.MASSACHUSETTS.VAL_SPLITS = 'test'
_C.DATASET.DEEPGLOBE = CN()
_C.DATASET.DEEPGLOBE.PATH = DEEPGLOBE_PATH
_C.DATASET.AUG = CN()
_C.DATASET.AUG.RANDOM_BRIGHTNESSCONTRAST_PROB = 0.75
_C.DATASET.AUG.FLIP_PROB = 0.5
_C.DATASET.AUG.ROTATE_LIMIT = (-180,180)
_C.DATASET.AUG.ROTATE_PROB = 0.5
_C.DATASET.MIN_MASK_PERCENTAGE = 0.01

# Model
_C.MODEL = CN()
_C.MODEL.ARCH = 'UNet'

# Loss
_C.LOSS = CN()
_C.LOSS.NAME = "DiceLoss"
_C.LOSS.DICE = CN()
_C.LOSS.DICE.SMOOTH = 1.0

# Optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 0.00001
_C.OPTIMIZER.WD = 0.0

# Training
_C.TRAINING = CN()
_C.TRAINING.LOG_FREQ_IMAGES = 2000
_C.TRAINING.LOG_IMAGES = True
_C.TRAINING.EPOCHS = 100 
def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

