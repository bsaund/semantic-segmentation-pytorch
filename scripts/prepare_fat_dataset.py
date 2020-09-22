import pathlib
from pathlib import Path
from PIL import Image
import numpy as np
from itertools import compress
import csv
import json
from tqdm import tqdm

data_root = Path('../data')
fat_path = data_root / 'fat'


def split_train_val(all, val_fraction=0.1):
    train_ind = np.random.random(len(all_metadata)) > val_fraction
    return compress(all, train_ind), compress(all, 1 - train_ind)


def gather_filepaths():
    all_metadata = []
    for img_path in fat_path.glob('**/*.left.jpg'):
        p = img_path.relative_to(data_root)
        w, h = Image.open(img_path.as_posix()).size
        all_metadata.append('{{"fpath_img": "{}", "fpath_segm": "{}", "width": {}, "height": {} }}\n'.format(
            p.as_posix(),
            p.with_suffix('.seg.png'),
            w, h))
        # f.write('{{"fpath_img": "{}", "fpath_segm": "{}", "width": {}, "height": {} }}\n'.format(
        #     p.as_posix(),
        #     p.with_suffix('.seg.png'),
        #     w, h))
        # img_path.relative_to(data_root)

    train_metadata, val_metadata = split_train_val(all_metadata)

    with (data_root / 'fat_training.odgt').open(mode='w') as f:
        for datum in train_metadata:
            f.write(datum)

    with (data_root / 'fat_validation.odgt').open(mode='w') as f:
        for datum in val_metadata:
            f.write(datum)


def update_object_category_numbers():
    """
    Makes object category numbers dense and consistent across the dataset
    Object categories from the nvidia dataset are inconsistently labeled and also spaced out.
    This creates clearer visualizations in standard image utils, but adds needless complexity in training
    """
    update_single_object_category()
    update_mixed_object_category()


def update_single_object_category():
    with open(fat_path / "mixed" / "kitchen_0" / "_object_settings.json") as f:
        object_setting = json.loads(f.read())
    object_numbers = {d["class"]: int(d["segmentation_class_id"] / 12 + 1) for d in object_setting['exported_objects']}

    object_numbers['000_background'] = 1

    # Hardcode an inconsistency in the dataset
    object_numbers['006_mustard_bottle_16k'] = object_numbers['006_mustard_bottle_16K']
    object_numbers.pop('006_mustard_bottle_16K')

    with open(fat_path / "object_info.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["Idx", "Name"])
        writer.writeheader()
        # for data in object_numbers:
        #     writer.writerow(data)
        for id, name in sorted([(v, k) for k, v in object_numbers.items()]):
            f.write("{}, {}\n".format(id, name))

    for img_path in tqdm(fat_path.glob('single/**/*.left.seg.png')):
        class_id = object_numbers[img_path.parts[-3]]
        img = Image.open(img_path.as_posix())
        img2 = img.point(lambda i: class_id if i else 0)
        img2.save(img_path.as_posix())


def update_mixed_object_category():
    for img_path in tqdm(fat_path.glob('mixed/**/*.left.seg.png')):
        img = Image.open(img_path.as_posix())

        for val in set(img.getdata()):
            if val % 12:
                raise Exception("This image was already processed and has pixel with value {}".format(val))

        img2 = img.point(lambda i: i/12 + 1)
        # img2 = img.point(lambda i: i + 1)
        img2.save(img_path.as_posix())


if __name__ == "__main__":
    # gather_filepaths()
    update_object_category_numbers()
