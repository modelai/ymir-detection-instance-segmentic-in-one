import argparse

from mmcv.utils import Config
from tqdm import tqdm
from ymir_exc.util import get_merged_config

from mmseg.datasets import build_dataset
from ymir.ymir_util import modify_mmcv_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--without_background', action='store_true')

    args = parser.parse_args()
    ymir_cfg = get_merged_config()

    mmcv_cfg = Config.fromfile(ymir_cfg.param.config_file)
    modify_mmcv_config(ymir_cfg, mmcv_cfg)

    if args.without_background:
        class_names = ymir_cfg.param.class_names
    else:
        class_names = ['ymir_background'] + ymir_cfg.param.class_names

    num_classes = len(class_names)

    print('class_name: ', class_names)
    if args.split == 'train':
        split_dataset = build_dataset(mmcv_cfg.data.train)
    else:
        split_dataset = build_dataset(mmcv_cfg.data.val)

    for dataset in [split_dataset]:
        count = {cls: 0 for cls in class_names}
        for img_id, d in enumerate(tqdm(dataset)):
            class_ids = d['gt_semantic_seg'].data.unique()
            # print(d['img_metas'])
            # print(f'class_ids is {class_ids}')
            # print(f'image {img_id}: ' + '*' * 50)
            for idx in class_ids:
                if idx < 255:
                    assert idx <= num_classes, f'{idx} should <= {num_classes}'
                    # print(idx, class_names[idx - 1], np.count_nonzero(d['gt_semantic_seg'].data == idx))

                    count[class_names[idx]] += 1

            if img_id >= args.test_num:
                break
        print(count)
