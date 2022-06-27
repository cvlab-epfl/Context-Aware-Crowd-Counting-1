import torch
import logging
from crowd.models.vgg import vgg19
from PIL import Image
from torchvision.utils import save_image
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F

DATA_DIR = '/home/toshiba_pc/Masaüstü/ME/crowd'    #modify it according to your data dir
CSV_FILE_PATH='/home/toshiba_pc/Masaüstü/ME/crowd' #modify it according to your data dir

def prepare_image_list():
    out_file_list = []
    sample_paths = open(CSV_FILE_PATH).readlines()
    sample_paths.sort()
    for sample_dir_path in sample_paths:
        file_list = []
        sample_dir_path = sample_dir_path.strip()
        for f in sorted(os.listdir(os.path.join(DATA_DIR, sample_dir_path))):
            if f.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                file_list.append(os.path.join(sample_dir_path, f))
        if len(file_list) < MIN_CLIP_LEN:
            print('ERROR !!!')
            print('len(file_list) < self.min_clip_len')
            exit()
        s_start = 0
        s_end = len(file_list) - 1
        inx_list = [round(i) for i in np.linspace(s_start, s_end, NUM_FRAMES)]
        file_list = [file_list[i] for i in inx_list]
        out_file_list.append(file_list)
    return out_file_list
