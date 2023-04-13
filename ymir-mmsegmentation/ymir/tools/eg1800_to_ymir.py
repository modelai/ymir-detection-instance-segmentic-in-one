"""
convert eg1800 dataset to ymir

# origin dataset
EG1800
├── eg1800_test.txt  [289 line 00001.png, 00002.png, 00003.png, ...]
├── eg1800_train.txt  [1447 line 00001.png, 00002.png, 00003.png, ...]
├── Images [1736 entries 00001.png, 00002.png, 00003.png, ...]
├── Labels [1887 entries 00001.png, 00002.png, 00003.png, ...]
└── README.md

# convert to ymir `local path import` format
ymir_eg1800
- train
  - images
    - xxx.jpg
    - xxx.jpg
  - gt
    - coco-annotations.json
- val
  - images
    - xxx.jpg
    - xxx.jpg
  - gt
    - coco-annotations.json

# cmd
python ymir/tools/eg1800_to_ymir.py --in-dir /xxx/EG1800 --out-dir /xxx/eg_ymir --split train
python ymir/tools/eg1800_to_ymir.py --in-dir /xxx/EG1800 --out-dir /xxx/eg_ymir --split val

empty mask for /xxx/EG1800/Images/02188.png /xxx/EG1800/Labels/02188.png 0
empty mask for /xxx/EG1800/Images/02299.png /xxx/EG1800/Labels/02299.png 1
"""

import argparse
import glob
import json
import os
import os.path as osp

import cv2
from tqdm import tqdm

from ymir.tools import pycococreatortools
from ymir.tools.result_to_coco import INFO, LICENSES


def get_args():
    parser = argparse.ArgumentParser(prog='convert EG1800 to ymir-coco segmentation format')
    parser.add_argument('--in-dir', help='input directory')
    parser.add_argument('--out-dir', help='output directory')
    parser.add_argument('--split', choices=['train', 'val'], help='split for dataset')
    parser.add_argument('--num', type=int, help='the sample number for output dataset', default=0)
    parser.add_argument('--ignore_background', action='store_true', default=False, help='ignore the background or not')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.ignore_background:
        class_names = ['foreground']
    else:
        class_names = ['background', 'foreground']

    if args.split == 'train':
        split_file = 'eg1800_train.txt'
    else:
        split_file = 'eg1800_test.txt'

    with open(os.path.join(args.in_dir, split_file), 'r') as fr:
        lines = fr.readlines()

    if args.num > 0:
        lines = lines[0:args.num]

    categories = []
    for idx, name in enumerate(class_names):
        categories.append(dict(id=idx, name=name, supercategory='none'))

    coco_output = {"info": INFO, "licenses": LICENSES, "categories": categories, "images": [], "annotations": []}

    image_id = 1
    annotation_id = 1
    rootdir_split = osp.join(args.out_dir, args.split)
    os.makedirs(osp.join(rootdir_split, 'images'), exist_ok=True)
    os.makedirs(osp.join(rootdir_split, 'gt'), exist_ok=True)

    for line in tqdm(lines):
        img_file = os.path.join(args.in_dir, 'Images', line.strip())
        ann_file = os.path.join(args.in_dir, 'Labels', line.strip())

        ann = cv2.imread(ann_file, cv2.IMREAD_GRAYSCALE)
        height, width = ann.shape[0:2]

        image_info = pycococreatortools.create_image_info(image_id=image_id,
                                                          file_name=osp.basename(img_file),
                                                          image_size=(width, height))

        coco_output["images"].append(image_info)  # type: ignore

        if args.ignore_background:
            class_values = [1]
        else:
            class_values = [0, 1]

        for class_id, class_value in enumerate(class_values):
            category_info = {'id': class_id, 'is_crowd': True}
            binary_mask = ann == class_value

            annotation_info = pycococreatortools.create_annotation_info(annotation_id,
                                                                        image_id,
                                                                        category_info,
                                                                        binary_mask,
                                                                        tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)  # type: ignore
                annotation_id = annotation_id + 1
            else:
                print(f'empty mask for {img_file} {ann_file} {class_value}')
        image_id += 1

        copy_img_file = osp.join(rootdir_split, 'images', line.strip())
        os.system(f'cp {img_file} {copy_img_file}')

    with open(osp.join(rootdir_split, 'gt', 'coco-annotations.json'), 'w') as fw:
        json.dump(coco_output, fw)

    # generate fake index file
    img_files = glob.glob(osp.join(rootdir_split, 'images', '*'))
    rel_ann_path = osp.join(args.split, 'gt', 'coco-annotations.json')
    with open(osp.join(args.out_dir, f'{args.split}-index.tsv'), 'w') as fw:
        for img_file in tqdm(img_files):
            rel_img_path = os.path.relpath(img_file, args.out_dir)
            fw.write(f'/in/{rel_img_path}\t/in/{rel_ann_path}\n')

    if args.split == 'val':
        # generate candidate index
        with open(osp.join(args.out_dir, 'candidate-index.tsv'), 'w') as fw:
            for img_file in tqdm(img_files):
                rel_img_path = os.path.relpath(img_file, args.out_dir)
                fw.write(f'/in/{rel_img_path}\n')
