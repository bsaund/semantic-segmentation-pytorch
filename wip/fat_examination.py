from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import numpy as np

data_root = Path('../data')

all_categories = set()

def gather_all_classes():

    with (data_root / 'fat_validation.odgt').open() as f:
        lines = f.readlines()
        for line in tqdm(lines):
            d = json.loads(line)
            seg = Image.open((data_root / d['fpath_segm']).as_posix())
            all_categories = all_categories.union(set(seg.getdata()))
            #
            # with open((data_root / d['fpath_img']).with_suffix('.json').as_posix()) as json_file:
            #     info = json.load(json_file)
            #
            # for object in info['objects']

    print("All segmentation categories")
    print(sorted(list(all_categories)))


def examine_ade20k():
    annotation_fp = data_root / "ADEChallengeData2016" / "annotations" / "validation" / "ADE_val_00000001.png"
    img_fp = data_root / "ADEChallengeData2016" / "images" / "validation" / "ADE_val_00000001.jpg"
    # with open(data_root / "ADE_ChallengeData2016" / "annotations" / "validation" / "ADE_val_0000001.png") as f:
    seg = Image.open(annotation_fp.as_posix())
    img = Image.open(img_fp.as_posix())
    print_counts(seg)


def print_counts(seg):
    w, h = seg.size
    pixs = w*h

    uniques, counts = np.unique(seg.getdata(), return_counts=True)
    for idx in np.argsort(counts)[::-1]:
        # name = names[uniques[idx] + 1]
        name = str(uniques[idx])
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))



def spotcheck_fat():
    # img_fp = data_root / "fat/single/003_cracker_box_16k/kitchen_0/000000.left.jpg"
    img_fp = data_root / "fat/mixed/kitchen_0/000000.left.jpg"
    seg_fp = img_fp.with_suffix('.seg.png')

    img = Image.open(img_fp)
    seg = Image.open(seg_fp)
    print_counts(seg)

    img.show()



if __name__ == "__main__":
    # gather_all_classes()
    # examine_ade20k()
    spotcheck_fat()