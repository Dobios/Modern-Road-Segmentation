import sys
sys.path.append('.')
from src.config import get_cfg_defaults
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('cfg', help="Path to config file.")

    args = parser.parse_args()

    # parse configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)