from pathlib import Path
from PIL import Image
import numpy as np
from itertools import compress
import csv
import json
from tqdm import tqdm

data_root = Path('../data')
ycb_path = data_root / 'YCB_Video_Dataset'


def get_metadata(img_list_filepath):
    metadata = []
    with open(img_list_filepath) as f:
        for line in tqdm(f.readlines()):
            local_path = ycb_path / 'data' / (line.rstrip() + '-color.png')
            p = local_path.relative_to(data_root)
            w, h = Image.open(local_path.as_posix()).size
            metadata.append('{{"fpath_img": "{}", "fpath_segm": "{}", "width": {}, "height": {} }}\n'.format(
                p,
                p.as_posix()[0:-10] + '-label.png',
                w, h))

    return metadata


def gather_filepaths():
    train_metadata = get_metadata(ycb_path / 'image_sets' / 'train.txt')
    val_metadata = get_metadata(ycb_path / 'image_sets' / 'train.txt')

    with (ycb_path / 'ycb_video_training.odgt').open(mode='w') as f:
        for datum in train_metadata:
            f.write(datum)

    with (ycb_path / 'ycb_video_validation.odgt').open(mode='w') as f:
        for datum in val_metadata:
            f.write(datum)


if __name__ == "__main__":
    gather_filepaths()
    # update_object_category_numbers()
