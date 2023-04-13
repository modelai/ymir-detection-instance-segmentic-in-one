"""
convert cityscapes dataset to ymir dataset format

cityscapes dataset directory (convert it with tools/covn)
.
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    └── val
"""

import argparse
import glob
import os.path as osp
import random
import sys

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('convert cityscapes to ymir dataset format')
    parser.add_argument('--in_dir', help='the cityscapes input directory', required=True)
    parser.add_argument('--out_dir', help='the directory for output index txt format file', default=None)
    parser.add_argument('--sample_num', help='the sample number for test', default=0, type=int)

    return parser.parse_args()


def ann_to_image(root_dir, ann_path):
    split, sub_folder, basename = ann_path.split('/')[-3:]
    return osp.join(root_dir, 'leftImg8bit', split, sub_folder,
                    basename.replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png'))


def main():
    args = get_args()
    args.out_dir = args.out_dir or args.in_dir

    for split in ['train', 'val', 'test']:
        split_ann_files = glob.glob(osp.join(args.in_dir, 'gtFine', split, '*', '*_labelTrainIds.png'))

        random.shuffle(split_ann_files)

        if args.sample_num > 0:
            split_ann_files = split_ann_files[0:args.sample_num]
        split_img_files = [ann_to_image(args.in_dir, f) for f in split_ann_files]
        with open(osp.join(args.out_dir, f'ymir-{split}.txt'), 'w') as fp:
            for img, ann in tqdm(zip(split_img_files, split_ann_files)):
                if split in ['train', 'val']:
                    fp.write(f'{img}\t{ann}\n')
                else:
                    fp.write(f'{img}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
