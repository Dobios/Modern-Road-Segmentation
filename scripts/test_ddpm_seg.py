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
from src.utils import unfold_data, patchify
from src.config import get_cfg_defaults
from src.datasets import CILFeaturesDataset, PixelDataset, BaseImageDataset, load_cil_metadata
from src.models import PixelClassifier, FeatureExtractorDDPM
from src.utils import collect_features
from src.losses import LossLogger
from src.metrics import accuracy, f1_score, jaccard

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def test(args, options):
    test_ds = BaseImageDataset(self.options, load_cil_metadata, is_train=False)
    classifier = PixelClassifier(args['dim'][-1])
    print(f"Loading pixel classifier from", args["model"] )
    classifier.load_state_dict(torch.load(args["model"]))
    classifier = classifier.to(device)
    classifier.eval()

    feature_extractor = FeatureExtractorDDPM(**args)

    iteration = 0
    break_count = 0
    best_loss = 10000000
    epochs = 30
    stop_sign = 0
    f1s = []
    accs = []
    window_size = 10
    for epoch in tqdm(range(epochs), total = epochs, position=0, desc="Epoch"):
        #for i in tqdm(train_loader, position=1, desc="Iteration"):
        dl = iter(train_loader)
        pb = tqdm(position=1, desc="Iteration")
        while(True):
            try:
                item = next(dl)
            except StopIteration:
                break
            pb.update()
            X_batch, y_batch = item["features"].to(device), item["y"].to(device)
            y_batch = y_batch.unsqueeze(1) # needs to be [bs, 1]

            optimizer.zero_grad()
            y_pred = classifier(X_batch)
            #print(y_batch.shape, y_pred.shape)
            loss, _ = criterion(y_pred, y_batch)
            dict = {}
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 1000 == 0:
                acc = accuracy(y_pred.detach().cpu(), y_batch.detach().cpu())
                f1 = f1_score(y_pred.detach().cpu().numpy(), y_batch.detach().cpu().numpy())
                accs.append(acc)
                f1s.append(f1)
                dict["epoch"] = epoch
                dict["iteration"] = iteration
                dict["acc"] = acc 
                dict["f1"] = f1      
                if len(f1s) >= window_size:
                    dict[f"m{window_size}_f1"] =  sum(f1s[-window_size:]) / window_size             
                    dict[f"m{window_size}_acc"] =  sum(accs[-window_size:]) / window_size             
                wandb.log(dict)
                #print(dict)
                #print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc, 'f1', f1)
            
        validate(val_ds, classifier, iteration, epoch)
        model_path = os.path.join(args['exp_dir'], 
                                    f'model_epoch{epoch}.pth')
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                    model_path)

def validate(val_ds, classifier, iteration, epoch):
    res = defaultdict(list)
    print("Validating...")
    vali = random.randint(0, len(val_ds))
    with torch.no_grad():
        for i, item in enumerate(tqdm(val_ds)):
            #[4, 3168, 256, 256]
            dims = item["features"].shape[1]
            features = item["features"]
            y = item["masks"]
            y_hat = None
            bs = item["features"].shape[0]
            for ip in range(item["features"].shape[0]):
                feats_patch = features[ip]
                feats_patch = feats_patch.reshape(dims, -1).permute(1,0).to(device)
                pred = classifier(feats_patch).detach().cpu().reshape(1,256,256) # [x, 1]
                if y_hat is None:
                    y_hat = pred
                else:
                    y_hat = torch.cat((y_hat, pred), dim=0)
            patched_y_hat = patchify(y_hat)
            patched_y = patchify(y)
            res['val/acc'].append(accuracy(y_hat, y))
            res['val/patched_acc'].append(accuracy(patched_y_hat, patched_y)), 
            res['val/f1'].append(f1_score(y_hat.numpy()>0.5, y.numpy()>0.5)) # binarize elements
            res['val/patched_f1'].append(f1_score(patched_y_hat.numpy(), patched_y.numpy()))
            res['val/jacc'].append(jaccard(y_hat, y))
            if i == vali:
                imgstack = item["patches"][0].permute(1, 2, 0)
                gtstack = y[0]
                predstack = y_hat[0]
                for i in range(1, 4):
                    imgstack = torch.cat((imgstack, item["patches"][i].permute(1, 2, 0)), axis=0)
                    gtstack = torch.cat((gtstack, y[i]), axis=0)
                    predstack = torch.cat((predstack, y_hat[i]), axis=0)
                gtstack = cv2.cvtColor(gtstack.numpy(), cv2.COLOR_GRAY2RGB)
                predstack = cv2.cvtColor(predstack.numpy(), cv2.COLOR_GRAY2RGB)
                imgstack = imgstack.numpy()

                # concat
                samples = np.concatenate((imgstack, gtstack, predstack), axis=1)
                # pred = np.concatenate((imgstack, predstack), axis=1)
                # # wandb
                wandb.log({
                    "samples": wandb.Image(samples, caption="predicted"),
                })
    for key in res.keys():
        res[key] = sum(res[key])/len(res[key])
    res["iteration"] = iteration
    res["epoch"] = epoch
    wandb.log(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--n_img', type=int)
    parser.add_argument('--model', '-m', type=str)

    #parser.add_argument('--cfg', type=str)
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Parse configs
    cfg = get_cfg_defaults()
    #cfg.merge_from_file(args.cfg)
    cfg.freeze()
    

    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
              
    train(opts, cfg)
    
