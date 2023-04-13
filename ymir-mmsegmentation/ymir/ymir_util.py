import glob
import json
import logging
import os
import os.path as osp
import warnings
from typing import Any, Iterable, List, Union

import cv2
import numpy as np
from easydict import EasyDict as edict
from mmcv.utils import Config, ConfigDict
from pycocotools import coco
from tqdm import tqdm
from ymir_exc.dataset_convert.ymir2mmseg import train_with_black_area_or_not
from ymir_exc.util import (get_bool, get_weight_files,
                           write_ymir_training_result)


def _find_any(str1: str, sub_strs: List[str]) -> bool:
    for s in sub_strs:
        if str1.find(s) > -1:
            return True
    return False


def get_best_weight_file(ymir_cfg: edict):
    """
    find the best weight file for ymir-executor
    1. find best_* in /in/models
    2. find epoch_* or iter_* in /in/models
    3. find xxx.pth or xxx.pt in /weights
    """
    weight_files = get_weight_files(ymir_cfg)

    # choose weight file by priority, best_xxx.pth > latest.pth > epoch_xxx.pth
    best_pth_files = [f for f in weight_files if osp.basename(f).startswith('best_')]
    if len(best_pth_files) > 0:
        return max(best_pth_files, key=os.path.getctime)

    epoch_pth_files = [f for f in weight_files if osp.basename(f).startswith(('epoch_', 'iter_'))]
    if len(epoch_pth_files) > 0:
        return max(epoch_pth_files, key=os.path.getctime)

    if ymir_cfg.ymir.run_training:
        model_name_splits = osp.basename(ymir_cfg.param.config_file).split('_')
        model_name = model_name_splits[0]
        weight_files = [
            f for f in glob.glob(f'/weights/**/{model_name}*', recursive=True) if f.endswith(('.pth', '.pt'))
        ]

        # eg: config_file = configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py
        # eg: best_model_file = fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth

        if len(weight_files) > 0:
            best_model_file = weight_files[0]
            for idx in range(2, len(model_name_splits)):
                prefix = '_'.join(model_name_splits[0:idx + 1])
                weight_files = [f for f in weight_files if osp.basename(f).startswith(prefix)]
                if len(weight_files) > 1:
                    best_model_file = weight_files[0]
                elif len(weight_files) == 1:
                    return weight_files[0]
                else:
                    return best_model_file

            return best_model_file

    return ""


def convert_annotation_dataset(ymir_cfg: edict, overwrite=False):
    """
    convert annotation images from coco.json to annotation image
    input directory:

    return new index files
    note: call before ddp, avoid multi-process problem
    """
    assert ymir_cfg.ymir.run_training, 'only training task need to convert dataset'

    ymir_ann_files = dict(train=ymir_cfg.ymir.input.training_index_file,
                          val=ymir_cfg.ymir.input.val_index_file,
                          test=ymir_cfg.ymir.input.candidate_index_file)

    in_dir = ymir_cfg.ymir.input.root_dir
    out_dir = ymir_cfg.ymir.output.root_dir
    # ann_dir = ymir_cfg.ymir.input.annotations_dir
    img_dir = ymir_cfg.ymir.input.assets_dir
    new_ann_files = dict()
    for split in ['train', 'val']:
        new_ann_files[split] = osp.join(out_dir, osp.relpath(ymir_ann_files[split], in_dir))

    new_ann_files['test'] = ymir_ann_files['test']
    # call before ddp, avoid multi-process problem, just to return new_ann_files
    logging.info(f'convert annotation dataset to: {new_ann_files}')
    if osp.exists(new_ann_files['train']) and not overwrite:
        return new_ann_files

    # note: if only exist blank area, we need generate a background class
    with_blank_area = train_with_black_area_or_not(ymir_cfg)
    class_names: List[str] = ymir_cfg.param.class_names

    for split in ['train', 'val']:
        with open(ymir_ann_files[split], 'r') as fp:
            lines = fp.readlines()

        imgname2imgpath = {}
        for line in tqdm(lines, desc='generate image filename to filepath dict'):
            img_path = line.split()[0]
            filename = os.path.basename(img_path)
            imgname2imgpath[filename] = img_path

        logging.info(f'images in ymir {split} index file: {len(imgname2imgpath)}')
        ann_json_file = lines[0].split()[1]
        coco_ann = coco.COCO(ann_json_file)

        img_ids = coco_ann.getImgIds()
        logging.info(f'images in coco {split} json file: {len(img_ids)}')
        fw = open(new_ann_files[split], 'w')
        for img_id in tqdm(img_ids, desc='convert coco annotations to training id mask'):
            filename = coco_ann.imgs[img_id]['file_name']
            # train-index file and val-index file share the same coco json file
            # so we need to filter the annotations according to index file
            if filename not in imgname2imgpath:
                continue

            ann_ids = coco_ann.getAnnIds(imgIds=[img_id])
            width = coco_ann.imgs[img_id]['width']
            height = coco_ann.imgs[img_id]['height']

            if with_blank_area:
                # background class id = 0
                training_id_mask = np.zeros(shape=(height, width), dtype=np.uint8)
            else:
                training_id_mask = np.ones(shape=(height, width), dtype=np.uint8) * 255  # type: ignore

            for ann_id in ann_ids:
                ann = coco_ann.anns[ann_id]
                mask = coco_ann.annToMask(ann)
                class_name = coco_ann.cats[ann['category_id']]['name']
                class_id = class_names.index(class_name)

                if with_blank_area:
                    # start from 1, class_name = ymir_background with class_id = 0
                    training_id_mask[mask == 1] = class_id + 1
                else:
                    training_id_mask[mask == 1] = class_id

            if os.path.isabs(filename):
                filename = os.path.relpath(path=filename, start=img_dir)
                img_path = filename
            else:
                img_path = imgname2imgpath[osp.basename(filename)]

            # use jpg will compression, use png without compression
            new_ann_path = osp.join(out_dir, 'annotations', osp.splitext(filename)[0] + '.png')
            os.makedirs(osp.dirname(new_ann_path), exist_ok=True)
            cv2.imwrite(new_ann_path, training_id_mask)
            fw.write(f'{img_path}\t{new_ann_path}\n')

        fw.close()

    return new_ann_files


def modify_mmcv_config(ymir_cfg: edict, mmcv_cfg: Config) -> None:
    """
    useful for training process
    - modify dataset config
    - modify model output channel
    - modify epochs, checkpoint, tensorboard config
    """

    def recursive_modify(mmcv_cfg: Union[Config, ConfigDict], attribute_key: str, attribute_value: Any):
        """
        recursive modify mmcv_cfg:
            1. mmcv_cfg.attribute_key to attribute_value
            2. mmcv_cfg.xxx.xxx.xxx.attribute_key to attribute_value (recursive)
            3. mmcv_cfg.xxx[i].attribute_key to attribute_value (i=0, 1, 2 ...)
            4. mmcv_cfg.xxx[i].xxx.xxx[j].attribute_key to attribute_value
        """
        for key in mmcv_cfg:
            if key == attribute_key:
                mmcv_cfg[key] = attribute_value
                logging.info(f'modify {mmcv_cfg}, {key} = {attribute_value}')
            elif isinstance(mmcv_cfg[key], (Config, ConfigDict)):
                recursive_modify(mmcv_cfg[key], attribute_key, attribute_value)
            elif isinstance(mmcv_cfg[key], Iterable):
                for cfg in mmcv_cfg[key]:
                    if isinstance(cfg, (Config, ConfigDict)):
                        recursive_modify(cfg, attribute_key, attribute_value)

    # modify dataset config
    ymir_ann_files = convert_annotation_dataset(ymir_cfg)

    # validation may augment the image and use more gpu
    # so set smaller samples_per_gpu for validation
    samples_per_gpu = ymir_cfg.param.samples_per_gpu
    workers_per_gpu = ymir_cfg.param.workers_per_gpu
    mmcv_cfg.data.samples_per_gpu = samples_per_gpu
    mmcv_cfg.data.workers_per_gpu = workers_per_gpu

    with_blank_area = train_with_black_area_or_not(ymir_cfg)
    mmcv_cfg.with_blank_area = with_blank_area  # save it for infer and mining
    if with_blank_area:
        assert 'ymir_background' not in ymir_cfg.param.class_names
        class_names = ['ymir_background'] + ymir_cfg.param.class_names
    else:
        class_names = ymir_cfg.param.class_names
    num_classes = len(class_names)
    recursive_modify(mmcv_cfg.model, 'num_classes', num_classes)

    for split in ['train', 'val', 'test']:
        ymir_dataset_cfg = dict(type='YmirDataset',
                                split=ymir_ann_files[split],
                                img_suffix='.jpg',
                                seg_map_suffix='.png',
                                img_dir=ymir_cfg.ymir.input.assets_dir,
                                ann_dir=ymir_cfg.ymir.input.annotations_dir,
                                classes=class_names,
                                palette=ymir_cfg.param.get('palette', None),
                                reduce_zero_label=False,
                                data_root=ymir_cfg.ymir.input.root_dir)
        # modify dataset config for `split`
        if split not in mmcv_cfg.data:
            continue

        mmcv_dataset_cfg = mmcv_cfg.data.get(split)

        if isinstance(mmcv_dataset_cfg, (list, tuple)):
            for x in mmcv_dataset_cfg:
                x.update(ymir_dataset_cfg)
        else:
            src_dataset_type = mmcv_dataset_cfg.type
            if src_dataset_type in ['MultiImageMixDataset', 'RepeatDataset']:
                mmcv_dataset_cfg.dataset.update(ymir_dataset_cfg)
            elif src_dataset_type in ['ConcatDataset']:
                for d in mmcv_dataset_cfg.datasets:
                    d.update(ymir_dataset_cfg)
            else:
                mmcv_dataset_cfg.update(ymir_dataset_cfg)

    if 'max_iters' in ymir_cfg.param:
        max_iters = ymir_cfg.param.max_iters
        if max_iters <= 0:
            pass
        elif 'max_iters' in mmcv_cfg.runner:
            mmcv_cfg.runner.max_iters = max_iters
        else:
            iter_runner = dict(type='IterBasedRunner', max_iters=max_iters)
            warnings.warn(f'modify {mmcv_cfg.runner} to {iter_runner}')
            mmcv_cfg.runner = iter_runner

    mmcv_cfg.checkpoint_config['out_dir'] = ymir_cfg.ymir.output.models_dir
    tensorboard_logger = dict(type='TensorboardLoggerHook', log_dir=ymir_cfg.ymir.output.tensorboard_dir)
    if len(mmcv_cfg.log_config['hooks']) <= 1:
        mmcv_cfg.log_config['hooks'].append(tensorboard_logger)
    else:
        mmcv_cfg.log_config['hooks'][1].update(tensorboard_logger)

    if 'interval' in ymir_cfg.param:
        interval = int(ymir_cfg.param.interval)
        if interval > 0:
            mmcv_cfg.evaluation.interval = min(interval, mmcv_cfg.runner.max_iters // 10)
    else:
        if 'max_iters' in mmcv_cfg.runner:
            interval = max(1, mmcv_cfg.runner.max_iters // 10)
        elif 'max_epoch' in mmcv_cfg.runner:
            interval = max(1, mmcv_cfg.runner.max_epochs // 10)
        else:
            assert False, f'unknown runner {mmcv_cfg.runner}'
        # modify evaluation and interval

        mmcv_cfg.evaluation.interval = interval

    mmcv_cfg.checkpoint_config.interval = mmcv_cfg.evaluation.interval

    # note some early weight files will be removed, check ymir result file and ensure the weight files be valid.
    mmcv_cfg.checkpoint_config.max_keep_ckpts = int(ymir_cfg.param.get('max_keep_ckpts', -1))
    mmcv_cfg.evaluation.save_best = 'mIoU'
    # fix DDP error
    mmcv_cfg.find_unused_parameters = True

    # set work dir
    mmcv_cfg.work_dir = ymir_cfg.ymir.output.models_dir

    # auto load offered weight file if not set by user!
    # maybe overwrite the default `load_from` from config file
    args_options = ymir_cfg.param.get("args_options", '')
    cfg_options = ymir_cfg.param.get("cfg_options", '')

    # if mmcv_cfg.load_from is None and mmcv_cfg.resume_from is None:
    if not (_find_any(args_options, ['--load-from', '--resume-from'])
            or _find_any(cfg_options, ['load_from', 'resume_from'])):  # noqa: W503
        weight_file = get_best_weight_file(ymir_cfg)
        if weight_file:
            mmcv_cfg.load_from = weight_file
        else:
            logging.warning('no weight file used for training!')


def write_last_ymir_result_file(cfg: edict, id: str = 'last'):
    """
    cfg: ymir merged config
    save_least_file: if True, save last weight file and config file
        if False, save log files with last weight file and config file

    load miou from json log file (*.log.json)
    """
    log_json_files = glob.glob(osp.join(cfg.ymir.output.models_dir, '*.log.json'))
    if len(log_json_files) == 0:
        raise Exception('no log json files found!')

    log_json_file = max(log_json_files, key=os.path.getctime)

    with open(log_json_file, 'r') as fr:
        lines = fr.readlines()

    for line in reversed(lines):
        val_result = json.loads(line)
        if val_result['mode'] == 'val':
            break

    last_miou = val_result['mIoU']
    save_least_file: bool = get_bool(cfg, key='save_least_file', default_value=True)

    # this file is soft link
    last_ckpt_path = osp.join(cfg.ymir.output.models_dir, 'latest.pth')
    if osp.islink(last_ckpt_path):
        last_ckpt_path = os.readlink(last_ckpt_path)

    evaluation_result = dict(mIoU=float(last_miou), mAcc=float(val_result['mAcc']), aAcc=float(val_result['aAcc']))
    if save_least_file:
        mmseg_config_files = glob.glob(osp.join(cfg.ymir.output.models_dir, '*.py'))

        write_ymir_training_result(cfg,
                                   evaluation_result=evaluation_result,
                                   files=mmseg_config_files + [last_ckpt_path],
                                   id=id)
    else:
        files = glob.glob(osp.join(cfg.ymir.output.models_dir, '*.py'))

        log_files = [f for f in files if not f.endswith('.pth')]

        write_ymir_training_result(cfg, evaluation_result=evaluation_result, files=log_files + [last_ckpt_path], id=id)
