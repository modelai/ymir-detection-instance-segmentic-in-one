"""
ymir distribute infer/mining tool functions
"""
import math
import os
from typing import Callable, List

import cv2
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_images_for_rank(images):
    if WORLD_SIZE == 1:
        return images
    else:
        return images[RANK::WORLD_SIZE]


def get_dataloader(images):
    return YmirMapDataLoader(images)


class YmirMapDataLoader(Dataset):

    def __init__(self, images: List[str]):
        super().__init__()
        self.im_files = images

    def __getitem__(self, index):
        img_path = self.im_files[index]
        img = cv2.imread(img_path)
        return dict(filename=img_path, image=img)

    def __len__(self):
        return len(self.im_files)


class YmirIterDataLoader(IterableDataset):

    def __init__(self, images: List[str]):
        super().__init__()

        rank = int(os.getenv('RANK', 0))
        world_size = int(os.getenv('WORLD_SIZE', 1))

        N = len(images)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_index = 0
            num_workers_per_gpu = 1
            worker_id_per_gpu = 0
        else:
            num_workers_per_gpu = worker_info.num_workers
            worker_id_per_gpu = worker_info.id

        total_workers = num_workers_per_gpu * world_size
        per_worker = int(math.ceil(N / total_workers))
        worker_id = (rank * num_workers_per_gpu) + worker_id_per_gpu
        start_index = worker_id * per_worker
        end_index = min(start_index + per_worker, N)
        self.im_files = images[start_index:end_index]

    def __iter__(self):
        for img_path in self.im_files:
            img = cv2.imread(img_path)
            yield dict(filename=img_path, image=img)


class YmirDistribute(object):

    def __init__(self, init_dist_fn: Callable):
        """
        init_dist_fn: init distributed environment
        """
        if LOCAL_RANK != -1:
            init_dist_fn()

    def iter_begin(self, batch, batch_idx):
        pass

    def iter_end(self, batch, batch_idx):
        pass

    def epoch_start(self):
        pass

    def epoch_end(self):
        pass

    def run_dist(self, ymir_cfg: edict, batch_size: int, dataloader_fn: Callable, run_batch_fn: Callable):
        with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
            images = [line.strip() for line in f.readlines()]

        max_barrier_times = len(images) // WORLD_SIZE // batch_size
        rank_images = get_images_for_rank(images)

        dataset = dataloader_fn(rank_images)
        if RANK in [0, -1]:
            tbar = tqdm(dataset)
        else:
            tbar = dataset

        for idx, batch in enumerate(tbar):
            # filenames = batch['filename']
            images = batch['image']

            # results = self.iter_begin(batch, idx)
            if WORLD_SIZE > 1 and idx < max_barrier_times:
                dist.barrier()

            self.iter_end(batch, idx)

        self.epoch_end()
