import os
import warnings

import torch.distributed as dist
from tqdm import tqdm

from mmseg.apis.test import collect_results_gpu

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_images_for_rank(images):
    if WORLD_SIZE == 1:
        return images
    else:
        return images[RANK::WORLD_SIZE]


def run_dist(images_for_all_rank, iter_fun):
    """
    1. split images for each rank
    2. run iter_fun for each rank
    3. collect the result for each rank
    """
    max_barrier_times = len(images_for_all_rank) // WORLD_SIZE
    rank_images = get_images_for_rank(images_for_all_rank)

    if RANK in [0, -1]:
        tbar = tqdm(rank_images)
    else:
        tbar = rank_images

    results = []
    N = len(rank_images)

    if N > 10000:
        warnings.warn(f'infer dataset too large {N}>10000, may out of memory')

    monitor_gap = max(1, N // 100)
    for idx, image in enumerate(tbar):
        if WORLD_SIZE > 1 and idx < max_barrier_times:
            dist.barrier()

        result = iter_fun(idx, image, N, monitor_gap)
        results.append(result)

    if WORLD_SIZE > 1:
        results = collect_results_gpu(results, len(images_for_all_rank))

    return results
