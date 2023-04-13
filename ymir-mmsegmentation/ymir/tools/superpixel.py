"""
copy from https://github.com/cailile/Revisiting-Superpixels-for-Active-Learning/blob/master/scripts/extract_superpixels.py
"""

from math import sqrt
from typing import Union

import cv2
import numpy as np
from easydict import EasyDict as edict


class SuperPixelSLIC(object):
    """
    for 800 x 600 image
        SLIC: 172ms
        SLICO: 208ms
    """

    def __init__(self, img, region_size=32, algorithm='SLIC'):
        algorithm_dict = dict(SLIC=cv2.ximgproc.SLIC, SLICO=cv2.ximgproc.SLICO, MSLIC=cv2.ximgproc.MSLIC)
        self.slic_runner = cv2.ximgproc.createSuperpixelSLIC(img,
                                                             algorithm=algorithm_dict[algorithm.upper()],
                                                             region_size=region_size)
        self.slic_runner.iterate(10)

    def getLabels(self):  # noqa
        return self.slic_runner.getLabels()

    def getLabelContourMask(self):  # noqa
        return self.slic_runner.getLabelContourMask()

    def getNumberOfSuperpixels(self):  # noqa
        return self.slic_runner.getNumberOfSuperpixels()


class SuperPixelSEEDS(object):
    """
    191 ms for 800 x 600 image
    """

    def __init__(self, img, num_superpixels=1024):
        height, width, channel = img.shape
        self.slic_runner = cv2.ximgproc.createSuperpixelSEEDS(image_width=width,
                                                              image_height=height,
                                                              image_channels=channel,
                                                              num_superpixels=num_superpixels,
                                                              num_levels=5,
                                                              prior=3,
                                                              histogram_bins=5,
                                                              double_step=True)
        self.slic_runner.iterate(img, 10)

    def getLabels(self):  # noqa
        return self.slic_runner.getLabels()

    def getLabelContourMask(self):  # noqa
        return self.slic_runner.getLabelContourMask()

    def getNumberOfSuperpixels(self):  # noqa
        return self.slic_runner.getNumberOfSuperpixels()


def get_superpixel(cfg: edict, img: Union[str, np.ndarray], max_superpixels: int = 1024):
    if isinstance(img, str):
        img = cv2.imread(img)

    algorithm = cfg.param.superpixel_algorithm.lower()
    if algorithm == 'seeds':
        runner = SuperPixelSEEDS(img, num_superpixels=max_superpixels)
    elif algorithm in ['slic', 'slico', 'mslic']:
        height, width = img.shape[0:2]
        region_size = 2 * round(sqrt(height * width / max_superpixels))
        runner = SuperPixelSLIC(img, region_size=region_size, algorithm=algorithm)  # type: ignore
    else:
        available = ['slic', 'slico', 'mslic', 'seeds']
        raise Exception(f'unknown super pixel algorithm {algorithm}, not in {available}')

    num_superpixels = runner.getNumberOfSuperpixels()
    if num_superpixels > max_superpixels:
        raise Exception(f'number of superpixels {num_superpixels} > max {max_superpixels}')

    return runner.getLabels()
