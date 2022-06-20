from yacs.config import CfgNode as CN

MASSACHUSSETS_PATH = 'data/massachussets'
DEEPGLOBE_PATH = 'data/deepglobe'

# Set default config parameters
_C = CN()

_C.DATASET = CN()
_C.DATASET.BATCHSIZE = 64
_C.DATASET.WORKERS = 8
_C.DATASET.TRAIN_DS = 'massachussets,deepglobe'



_C.TRAIN = CN()

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

