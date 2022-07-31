# RoadSegmentation

Code for the road segmentation project.


## Installation

Install dependencies with:
```console
pip install -r requirements.txt
```

## Running training
NOTE: All of out scripts log to https://wandb.ai. To turn this off, set the environment variable `WANDB_MODE='offline'`.

### UPerNet and UNets

For training UPerNet and UNets training is done with the following command:
```console
python scripts/train.py --cfg PATH-TO-CONFIG-FILE -p
```
The `p` flag will automatically create a submission file. Be sure to modify the configuration file to your choosing. See the `configs/` folder for examples.

### Retraining best performing UPerNet

First download all datasets with
```console
sh scripts/download_datasets.sh PATH_TO_DATA_FOLDER
```
Modify the paths in the `configs/upernet_pt_ms.yaml` and `configs/upernet_base.yaml` to match your system.
To pretrain the model on the MSRoads dataset, run:
```console
python scripts/train.py --cfg configs/upernet_pt_ms.yaml
```
Check the logs of the script and extract the PATH of the best performing model (f1). Add this path to the `TRAINING.PRETRAINED_MODEL` section and run:
```console
python scripts/train.py --cfg configs/upernet_base.yaml -p
```
This will generate the submission file under `upernet_best_submission.csv`.

If any run crashes you can resume it by setting `TRAINING.RESUME_CKPT` to the latest checkpoint and rerun the command that failed. To test an arbitrary model trained by the config file `configs/upernet_base.yaml`, run:
```console
python scripts/test.py --cfg configs/upernet_base.yaml -m PATH_TO_CKPT
```

### Training DDPMs and pixel classifiers.

Training the DDPM itself requires a different repository. Read more [in this repository](https://github.com/jkminder/guided-diffusion) on how to train your own DDPM. You can download our pretrained checkpoint [here](https://drive.google.com/file/d/1nn8b97NS598tvdGB5WiAAcY8Nk1v2XNV/view?usp=sharing).

To train the pixel classifier, we require some of the code from the `guided-diffusion` repository. [Clone the repo](https://github.com/jkminder/guided-diffusion). Once cloned `cd` into it and run 
```console
pip install -e .
```
This installs the repo as a library for other repositories to use it. 

To train the pixel classifier, first point the `model_path` in the `configs/cil_dataset.json` file to your pretrained DDPM. Then run
```console
python scripts/train_ddpm_seg.py --exp configs/cil_dataset.json
```
This will train the classifier for 30 epochs. Alternativelly you can set `--n_img 5` to only train on 5 images or `--model PATH` to load a pretrained model. The trained models checkpoints will be stored under `pixel_classifier/pixel_classifiers/cil/50_150_200_250_4_5_6_7_8_12`. 

To produce a submission file run:
```console
python scripts/test_ddpm_seg.py --exp configs/cil_dataset.json --model PATH_TO_MODEL
```

## Creating prediction images

Use the scripts `scripts/create_predictions.py` and `scripts/create_ddpm_predictions.py` to create sample png images from the CIL validation set.