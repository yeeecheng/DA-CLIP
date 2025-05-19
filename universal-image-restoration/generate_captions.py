import os
import random
import sys

import numpy as np
import pandas as pd

import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm
import json


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

# DEGRADATION_TYPES = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']
# DEGRADATION_TYPES = ['blur0.5', 'blur1.0', 'blur1.5', 'blur2.0', 'blur2.5', 'blur3.0', 'blur3.5', 'blur4.0']
# DEGRADATION_TYPES = [f'jpeg{i}' for i in range(40, 81, 1) if i % 10 != 0]
# DEGRADATION_TYPES = ['noisy5', 'noisy10', 'noisy15', 'noisy20', 'noisy25', 'noisy30', 'noisy35', 'noisy40']
# DEGRADATION_TYPES = ['resize0.5', 'resize1.0', 'resize1.5', 'resize2.0', 'resize2.5', 'resize3.0', 'resize3.5', 'resize4.0']
# DEGRADATION_TYPES = ['jpeg10', 'jpeg20', 'jpeg30', 'jpeg40', 'jpeg50', 'jpeg60', 'jpeg70', 'jpeg80']

DEGRADATION_TYPES = ["random"]
print(DEGRADATION_TYPES)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    '''get image path list from image folder'''
    # print(path)
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def get_paired_paths(dataroot, deg_type):
    """
    Read LQ (Low Quality) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """
    GT_paths, LQ_paths = [], []
    paths1 = _get_paths_from_images(os.path.join(dataroot, deg_type, 'GT'))
    paths2 = _get_paths_from_images(os.path.join(dataroot, deg_type, 'LQ'))

    with open(os.path.join(dataroot, deg_type, 'degraded_prompts.json'), "r", encoding="utf-8") as f:
        degraded_prompts_data = json.load(f)
    # print(paths1, paths2)
    GT_paths.extend(paths1)  # GT list
    LQ_paths.extend(paths2)  # LR list

    print(f'GT length: {len(GT_paths)}, LQ length: {len(LQ_paths)}')
    return GT_paths, LQ_paths, degraded_prompts_data

def generate_captions(dataroot, ci, mode='train'):

    for deg_type in DEGRADATION_TYPES:
        print(deg_type)
        GT_paths, LQ_paths, degraded_prompts_data = get_paired_paths(dataroot, deg_type)

        future_df = {"filepath":[], "title":[]}

        for gt_image_path, lq_image_path in tqdm(zip(GT_paths, LQ_paths)):
            image = Image.open(gt_image_path).convert('RGB')
            caption = ci.generate_caption(image)

            filename = "./" + gt_image_path.split("/")[-1]
            dagradation = degraded_prompts_data[filename]

            title = f'{caption}| {dagradation}'

            future_df["filepath"].append(lq_image_path)
            future_df["title"].append(title)

        pd.DataFrame.from_dict(future_df).to_csv(
            os.path.join(dataroot, deg_type, f"daclip_{mode}.csv"), index=False, sep="\t"
        )


if __name__ == "__main__":
    dataroot = './datasets/DIV2K_HR/train'
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    # generate_captions(dataroot, ci, 'val')
    generate_captions(dataroot, ci, 'val')


