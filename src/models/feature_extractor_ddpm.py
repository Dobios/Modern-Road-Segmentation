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