"""
from https://github.com/yandex-research/ddpm-segmentation
"""
from importlib.resources import path
import sys
sys.path.append('.')
import torch
from torch import nn
from typing import List
from src.config import get_cfg_defaults
from src.datasets import BaseImageDataset, load_cil_metadata
from src.utils import unfold_data
import pandas as pd
from tqdm import tqdm
from torch.utils.data import random_split

import os
import argparse
from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())

def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out

class FeatureExtractorDDPM(nn.Module):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, steps: List[int], blocks: List[int], model_path: str, input_activations: bool, **kwargs):
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []
        
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        from guided_diffusion.script_util import create_model_and_diffusion

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)
        

        self.model.load_state_dict(
            torch.load(model_path)
        )
        self.model.to(device)
        if kwargs['use_fp16']:
            self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            self.model(noisy_x, self.diffusion._scale_timesteps(t))

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations


def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(args['dim'][:-1])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"]
        )
        resized_activations.append(feats[0])
    
    return torch.cat(resized_activations, dim=0)

def main(args):
    cfg = get_cfg_defaults()
    #cfg.DATASET.CIL.PATH = "data/cil"
    feature_extractor = FeatureExtractorDDPM(**args)
    train_ds = BaseImageDataset(cfg.DATASET, load_cil_metadata, True)
    test_ds = BaseImageDataset(cfg.DATASET, load_cil_metadata, False)
    features_root = cfg.DATASET.CIL.PATH + "/features/"
    os.makedirs(features_root, exist_ok = True)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=device)
    else:
        noise = None 
    #X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    #y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)
    for i, item in enumerate(tqdm(train_ds)):
        patches, mask_patches, ufold_shape = unfold_data(item["image"], item["mask"], 144, 256)
        X = torch.zeros((patches.shape[0], *args['dim'][::-1]), dtype=torch.float)
        for i in range(patches.shape[0]):
            img = patches[i][None].to(device)
            raw_features = feature_extractor(img, noise=noise)
            X[i] = collect_features(args, raw_features).cpu() #[3168, 256, 256]
        
        fname = features_root + "/" + item["image_path"].split("/")[-1].split(".")[0]
        torch.save(X, fname+"_feats.pt")
        torch.save(patches, fname+"_patches.pt")
        torch.save(mask_patches, fname+"_masks.pt")

    # for row, (img, label) in enumerate(tqdm(dataset)):
    #     img = img[None].to(dev())
    #     features = feature_extractor(img, noise=noise)
    #     X[row] = collect_features(args, features).cpu()
        
    #     for target in range(args['number_class']):
    #         if target == args['ignore_label']: continue
    #         if 0 < (label == target).sum() < 20:
    #             print(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
    #             label[label == target] = args['ignore_label']
    #     y[row] = label
    
    # d = X.shape[1]
    # print(f'Total dimension {d}')
    # X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    # y = y.flatten()


if __name__ == '__main__':
    print(device)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    main(opts)