"""
Adopted from https://yandex-research.github.io/ddpm-segmentation/
"""
from importlib.resources import path
import sys
sys.path.append('.')
import torch
from torch import nn
from typing import List
from torch.utils.data import random_split, DataLoader
from collections import defaultdict
import cv2
import numpy as np
import random
import torch.utils.data as data_utils
import pandas as pd
from tqdm import tqdm
import os
import argparse
from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
import json
import wandb
from tqdm import tqdm
from math import ceil 
from src.utils import unfold_data, patchify, create_submission
from src.config import get_cfg_defaults
from src.datasets import BaseImageDataset, load_cil_metadata
from src.models import PixelClassifier, FeatureExtractorDDPM
from src.utils import collect_features
from src.metrics import accuracy, f1_score, jaccard

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_features(feature_extractor, noise, x, args):
    patches, _, _ = unfold_data(x, None, 144, 256)
    X = torch.zeros((patches.shape[0], *args['dim'][::-1]), dtype=torch.float)
    for i in range(patches.shape[0]):
        img = patches[i][None].to(device)
        raw_features = feature_extractor(img, noise=noise)
        X[i] = collect_features(args, raw_features).cpu() #[feats, 256, 256]
    # X.shape = [4, feats, 256, 256]
    X_combined = torch.zeros((args['dim'][::-1], 400, 400), dtype=torch.float)
    X_combined[:, :256, :256] = X[0]
    X_combined[:, :256, 256:] = X[1, :, :, 112:]
    X_combined[:, 256:, :256] = X[2, :, 112:, :]
    X_combined[:, 256:, 256:] = X[3, :, 112:, 112:]
    return X_combined

def predict(classifier, feature_extractor, noise, x, args):
    dims = args["dim"][-1]
    features = extract_features(feature_extractor, noise, x, args) # [feats, 400, 400]
    features = features.reshape(dims, -1).permute(1,0).to(device)
    y_hat = classifier(features).detach().cpu().reshape(1,400,400) 
    return y_hat

def test(args, options):
    test_ds = BaseImageDataset(options.DATASET, load_cil_metadata, is_train=False)
    classifier = PixelClassifier(args['dim'][-1])
    print(f"Loading pixel classifier from", args["model"] )
    #classifier.load_state_dict(torch.load(args["model"])["model_state_dict"])
    classifier.init_weights()
    classifier = classifier.to(device)
    classifier.eval()

    feature_extractor = FeatureExtractorDDPM(**args)
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=device)
    else:
        noise = None 
    
    predictions = []
    names = []
    for i, item in enumerate(tqdm(test_ds)):
        # We always use TTA with thres here
        x = item["image"]
        y_hat = torch.zeros((1,400,400)).cpu()
        for x_hat, flip in [(x, False), (x.flip(2), True)]:
            for k in range(0, 4):
                x_rot = x_hat.rot90(k, (1,2))
                with torch.no_grad():
                    y_hat_rot = predict(classifier, feature_extractor, noise,  x_rot, args)
                    y_hat_rot = y_hat_rot.detach().to('cpu').rot90(-k, (1,2))
                if flip:
                    y_hat_rot = y_hat_rot.flip(2)
                y_hat += y_hat_rot
            y_hat = (y_hat > 4).float()
            predictions.append(y_hat)
            names.append(item["image_path"])
    
    y_hat_all = torch.cat(predictions, dim=0)
    create_submission(y_hat_all, names, "ddmp_submission.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument('--exp', type=str)
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--seed', type=int,  default=0)

    # change default
    args = parser.parse_args()
    
    # Load the experiment config
    opts = vars(args)
    opts.update(json.load(open(args.exp, 'r')))
    opts['image_size'] = opts['dim'][0]
    print(opts)
    # Parse configs
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    
    test(opts, cfg)
    
