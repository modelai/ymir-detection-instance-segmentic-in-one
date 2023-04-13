import os
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
from skimage import measure,draw,data
from PIL import Image
from tqdm import tqdm
from pycocotools import coco

def get_paired_coord(coord):
    points = None
    for i in range(0, len(coord), 2):
        point = np.array(coord[i: i+2], dtype=float).reshape(1, 2)
        if (points is None): points = point
        else: points = np.concatenate([points, point], axis=0)
    return points

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    # print(contours)
    # print(type(contours))
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

coco_ann = coco.COCO('/data1/yenanfei/segmentation/instances_val2017.json')
# coco_ann_orig = coco.COCO('/data1/wangjiaxin/cvdatasets/coco/annotations/instances_val2017.json')

img_ids = coco_ann.getImgIds()

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

for img_id in tqdm(img_ids, desc='convert coco annotations to training id mask'):
    filename = coco_ann.imgs[img_id]['file_name']
    img = cv2.imread('annotations/val-images/images/'+filename)
#     if filename != '14fa238c3b3fb81e23865a5abba2839daccdf23b.png':
#         continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_mask = np.array([0, 255, 255])
    ann_ids = coco_ann.getAnnIds(imgIds=[img_id])
    width = coco_ann.imgs[img_id]['width']
    height = coco_ann.imgs[img_id]['height']
    print(len(ann_ids))
    for ann_id in ann_ids:
        ann = coco_ann.anns[ann_id]
        ann_orig = coco_ann.anns[ann_id]
        # print(ann)
        # print(ann_orig)

        mask = coco_ann.annToMask(ann_orig)
        mask = mask.astype(np.bool)
        img[mask] = img[mask] * 0.7 + color_mask * 0.3
        # k=np.ones ((9,9),np.uint8)
        poly = binary_mask_to_polygon(mask)
        class_name = coco_ann.cats[ann['category_id']]['name']
        # print(len(poly))
        print(class_name)
        for i in poly:
                        
            poly0_0 = get_paired_coord(i)
            cv2.polylines(img, [np.array(poly0_0, dtype=np.int32)], True, compute_color_for_labels(int(ann['category_id'])),2)
        cv2.imshow('a',img)
        cv2.waitKey(0)