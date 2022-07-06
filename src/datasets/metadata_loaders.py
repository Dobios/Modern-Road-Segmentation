from curses import meta
import pandas as pd
import os

def load_massachusetts_metadata(options, is_train):
    metadata = pd.read_csv(os.path.join(options.MASSACHUSETTS.PATH, 'metadata.csv'))
    splits = options.MASSACHUSETTS.TRAIN_SPLITS if is_train else options.MASSACHUSETTS.VAL_SPLITS
    metadata = metadata[metadata["split"].isin(splits.split(","))]
    metadata = metadata[["tiff_image_path", "tif_label_path"]]
    metadata = metadata.reset_index(drop=True)
    metadata.columns = ["image_path", "mask_path"]
    metadata["image_path"] = metadata["image_path"].apply(lambda x: os.path.join(options.MASSACHUSETTS.PATH, x))
    metadata["mask_path"] = metadata["mask_path"].apply(lambda x: os.path.join(options.MASSACHUSETTS.PATH, x))
    return metadata

def load_cil_metadata(options, is_train):
    path = options.CIL.PATH
    if not path.endswith("/"):
        path += "/"
    if is_train:
        masks = sorted([f'{path}training/groundtruth/{fn}' for fn in os.listdir(f'{path}training/groundtruth/') if fn.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))
        images = sorted([f'{path}training/images/{fn}' for fn in os.listdir(f'{path}training/images/') if fn.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))
        return pd.DataFrame({'image_path': images, 'mask_path': masks})
    else:
        images = sorted([f'{path}test/images/{fn}' for fn in os.listdir(f'{path}test/images/') if fn.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))
        return pd.DataFrame({'image_path': images})