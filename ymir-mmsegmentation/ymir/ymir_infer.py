import json
import logging
import os
import os.path as osp
import sys
from functools import partial
from typing import Dict, List

import mmcv
import numpy as np
import torch.distributed as dist
from easydict import EasyDict as edict
from mmcv.runner import init_dist, wrap_fp16_model
from tqdm import tqdm
from ymir_exc.result_writer import write_infer_result
from ymir_exc.util import (YmirStage, get_bool, get_merged_config,
                           get_weight_files, write_ymir_monitor_process)

from mmseg.apis import inference_segmentor, init_segmentor
from ymir.tools.result_to_coco import convert
from ymir.ymir_dist import run_dist
from ymir.ymir_util import get_best_weight_file

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def iter_fun(cfg, model, idx, image, N, monitor_gap):
    result = inference_segmentor(model, image)

    if idx % monitor_gap == 0 and RANK in [0, -1]:
        write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=idx / N, stage=YmirStage.TASK)

    # view mmseg.models.segmentors.base show_result()
    return dict(image=image, result=result[0])


def save_infer_result(cfg: edict, results: List[Dict]) -> int:
    """
    save the mask into out_dir, used for debug and visulization
    """
    if RANK not in [0, -1]:
        return 0

    out_dir = '/out/masks'
    class_num = len(cfg.param.class_names)

    # use dataset palette if not defined in hyper-parameter
    palette_str: str = cfg.param.get('palette', '')
    list_palette: List[int] = [int(x) for x in palette_str.split(',')]
    assert len(list_palette) == class_num * 3, f'length of palette {palette_str} should be {class_num * 3}'

    dict_palette = {}
    for i in range(class_num):
        dict_palette[i] = (list_palette[3 * i], list_palette[3 * i + 1], list_palette[3 * i + 2])

    logging.info(f'use palette {dict_palette}')

    os.makedirs(out_dir, exist_ok=True)
    for idx, d in enumerate(tqdm(results, desc='save infer result')):
        image = d['image']
        result_file = osp.join(out_dir, osp.basename(image))
        assert result_file != image, f'please avoid overwrite origin image {image}'

        seg = d['result']
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

        for id, color in dict_palette.items():
            color_seg[seg == id, :] = color

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        mmcv.imwrite(color_seg, result_file)
    return 0


def main() -> int:
    if LOCAL_RANK != -1:
        init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")
    ymir_cfg: edict = get_merged_config()
    config_files = get_weight_files(ymir_cfg, suffix=('.py'))
    if len(config_files) == 0:
        raise Exception('not found config file (xxx.py) in pretrained weight files')
    elif len(config_files) > 1:
        raise Exception(f'found multiple config files {config_files} in pretrained weight files')

    checkpoint_file = get_best_weight_file(ymir_cfg)
    if not checkpoint_file:
        raise Exception('not found pretrain weight file (*.pt or *.pth)')

    mmcv_config = mmcv.Config.fromfile(config_files[0])
    mmcv_config.model.train_cfg = None
    model = init_segmentor(config=mmcv_config,
                           checkpoint=checkpoint_file,
                           device=f'cuda:{RANK}' if RANK > 0 else 'cuda:0')

    if get_bool(ymir_cfg, 'fp16', False):
        wrap_fp16_model(model)

    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    results = run_dist(images, partial(iter_fun, ymir_cfg, model))

    if RANK in [0, -1]:
        # save_infer_result(ymir_cfg, results)
        coco_results = convert(ymir_cfg, results, mmcv_config.with_blank_area)
        write_infer_result(coco_results, algorithm='segmentation')
    return 0


if __name__ == '__main__':
    sys.exit(main())
