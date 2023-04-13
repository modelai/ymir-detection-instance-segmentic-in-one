"""
Towards Fewer Annotations: Active Learning via Region Impurity and
    Prediction Uncertainty for Domain Adaptive Semantic Segmentation (CVPR 2022 Oral)

view code: https://github.com/BIT-DA/RIPU
"""

import os
import os.path as osp
import sys
from typing import Dict, List

import mmcv
import torch
import torch.distributed as dist
import torch.nn.functional as F
from easydict import EasyDict as edict
from mmcv.engine import collect_results_cpu
from mmcv.runner import init_dist, wrap_fp16_model
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import (YmirStage, get_bool, get_merged_config,
                           get_weight_files, write_ymir_monitor_process)

from mmseg.apis import init_segmentor
from ymir.tools.batch_infer import get_dataloader
from ymir.ymir_util import get_best_weight_file

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class RIPUMining(torch.nn.Module):

    def __init__(self, ymir_cfg: edict, class_number: int):
        super().__init__()
        self.ymir_cfg = ymir_cfg
        self.region_radius = int(ymir_cfg.param.ripu_region_radius)
        # note parameter: with_blank_area
        self.class_number = class_number
        self.image_topk = int(ymir_cfg.param.topk_superpixel_score)
        # ratio = float(ymir_cfg.param.ratio)

        kernel_size = 2 * self.region_radius + 1
        self.region_pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=self.region_radius)

        self.depthwise_conv = torch.nn.Conv2d(in_channels=self.class_number,
                                              out_channels=self.class_number,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=self.region_radius,
                                              bias=False,
                                              padding_mode='zeros',
                                              groups=self.class_number)

        weight = torch.ones((self.class_number, 1, kernel_size, kernel_size), dtype=torch.float32)
        weight = torch.nn.Parameter(weight)
        self.depthwise_conv.weight = weight
        self.depthwise_conv.requires_grad_(False)

    def get_region_uncertainty(self, logit: torch.Tensor) -> torch.Tensor:
        C = torch.tensor(logit.shape[1])
        entropy = -logit * torch.log(logit + 1e-6)  # BCHW
        uncertainty = torch.sum(entropy, dim=1, keepdim=True) / torch.log(C)  # B1HW

        region_uncertainty = self.region_pool(uncertainty)  # B1HW

        return region_uncertainty

    def get_region_impurity(self, logit: torch.Tensor) -> torch.Tensor:
        C = torch.tensor(logit.shape[1])
        predict = torch.argmax(logit, dim=1)  # BHW
        one_hot = F.one_hot(predict, num_classes=self.class_number).permute(
            (0, 3, 1, 2)).to(torch.float)  # BHW --> BHWC --> BCHW
        summary = self.depthwise_conv(one_hot)  # BCHW
        count = torch.sum(summary, dim=1, keepdim=True)  # B1CH
        dist = summary / count  # BCHW
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / torch.log(C)  # B1HW

        return region_impurity

    def get_region_score(self, logit: torch.Tensor) -> torch.Tensor:
        """
        logit: [B,C,H,W] prediction result with softmax/sigmoid
        """
        score = self.get_region_uncertainty(logit) * self.get_region_impurity(logit)  # B1HW

        return score

    def get_image_score(self, logit: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logit.shape
        score = self.get_region_score(logit).view(size=(B, 1 * H * W))  # B1HW
        topk = torch.topk(score, k=self.image_topk, dim=1, largest=True)  # BK
        image_score = torch.sum(topk.values, dim=1)  # B
        return image_score


def iter_fun(cfg, model, miner, idx, batch, N, monitor_gap) -> List[Dict]:
    """
    batch: Dict(img=[tensor,], img_metas=[List[DataContainer],])
    """
    batch_img = batch['img'][0]
    batch_meta = batch['img_metas'][0].data[0]
    device = torch.device('cuda:0' if RANK == -1 else f'cuda:{RANK}')
    batch_result = model.inference(batch_img.to(device), batch_meta, rescale=True)

    if idx % monitor_gap == 0 and RANK in [0, -1]:
        write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=idx / N, stage=YmirStage.TASK)

    scores = miner.get_image_score(batch_result)
    image_results: List[Dict] = []
    for idx, meta in enumerate(batch_meta):
        score = float(scores[idx])
        image_results.append(dict(image_filepath=meta['filename'], score=score))

    return image_results


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

    index_file_path = ymir_cfg.ymir.input.candidate_index_file
    with open(index_file_path, 'r') as f:
        num_image_all_rank = len(f.readlines())

    samples_per_gpu = int(ymir_cfg.param.samples_per_gpu)
    max_barrier_times = num_image_all_rank // WORLD_SIZE // samples_per_gpu
    mmcv_cfg = mmcv.Config.fromfile(config_files[0])
    mmcv_cfg.model.train_cfg = None
    model = init_segmentor(config=mmcv_cfg,
                           checkpoint=checkpoint_file,
                           device='cuda:0' if RANK == -1 else f'cuda:{RANK}')
    if mmcv_cfg.with_blank_area:
        class_num = len(ymir_cfg.param.class_names) + 1
    else:
        class_num = len(ymir_cfg.param.class_names)
    miner = RIPUMining(ymir_cfg, class_num)
    miner.to('cuda:0' if RANK == -1 else f'cuda:{RANK}')

    if get_bool(ymir_cfg, 'fp16', False):
        wrap_fp16_model(model)

    dataloader = get_dataloader(mmcv_cfg, ymir_cfg)
    N = len(dataloader)
    if N == 0:
        raise Exception('find empty dataloader')

    if RANK in [0, -1]:
        tbar = tqdm(dataloader, desc='obtain image prediction')
    else:
        tbar = dataloader

    monitor_gap = max(1, N // 1000)

    rank_image_result = []
    for idx, batch in enumerate(tbar):
        if idx < max_barrier_times and WORLD_SIZE > 1:
            dist.barrier()
        batch_image_result = iter_fun(ymir_cfg, model, miner, idx, batch, N, monitor_gap)
        rank_image_result.extend(batch_image_result)

    if WORLD_SIZE == 1:
        all_image_result = rank_image_result
    else:
        tmp_dir = osp.join(ymir_cfg.ymir.output.root_dir, 'tmp_dir')
        all_image_result = collect_results_cpu(rank_image_result, num_image_all_rank, tmp_dir)

    if RANK in [0, -1]:
        ymir_mining_result = []
        for mining_info in all_image_result:
            ymir_mining_result.append((mining_info['image_filepath'], mining_info['score']))

        rw.write_mining_result(mining_result=ymir_mining_result)

    return 0


if __name__ == '__main__':
    sys.exit(main())
