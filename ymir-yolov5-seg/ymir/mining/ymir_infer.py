"""use fake DDP to infer
1. split data with `images_rank = images[RANK::WORLD_SIZE]`
2. save splited result with `torch.save(results, f'results_{RANK}.pt')`
3. merge result
"""
import os
import sys
import warnings
from functools import partial

import torch
import torch.distributed as dist
import torch.utils.data as td
from easydict import EasyDict as edict
from tqdm import tqdm
from utils.general import scale_boxes,scale_segments,binary_mask_to_polygon,get_paired_coord
from ymir.mining.util import YmirDataset, load_image_file
from ymir.ymir_yolov5 import YmirYolov5,process_error
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process
from utils.segment.general import masks2segments, process_mask, process_mask_native,gen_anns_from_dets
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import cv2
import time


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def run(ymir_cfg: edict, ymir_yolov5: YmirYolov5):
    # eg: gpu_id = 1,3,5,7  for LOCAL_RANK = 2, will use gpu 5.
    gpu_id: str = str(ymir_cfg.param.get('gpu_id', 'cpu'))
    if gpu_id == '' or gpu_id == 'None':
        gpu_id = 'cpu'

    if gpu_id == 'cpu':
        device = 'cpu'
    else:
        gpu = LOCAL_RANK if LOCAL_RANK >= 0 else 0
        device = torch.device('cuda', gpu)
    ymir_yolov5.to(device)

    load_fn = partial(load_image_file, img_size=ymir_yolov5.img_size, stride=ymir_yolov5.stride)
    batch_size_per_gpu = ymir_yolov5.batch_size_per_gpu
    gpu_count = ymir_yolov5.gpu_count
    cpu_count: int = os.cpu_count() or 1
    num_workers_per_gpu = min([
        cpu_count // max(gpu_count, 1), batch_size_per_gpu if batch_size_per_gpu > 1 else 0,
        ymir_yolov5.num_workers_per_gpu
    ])

    with open(ymir_yolov5.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    max_barrier_times = len(images) // max(1, WORLD_SIZE) // batch_size_per_gpu
    # origin dataset
    if RANK != -1:
        images_rank = images[RANK::WORLD_SIZE]
    else:
        images_rank = images
    origin_dataset = YmirDataset(images_rank, load_fn=load_fn)
    origin_dataset_loader = td.DataLoader(origin_dataset,
                                          batch_size=batch_size_per_gpu,
                                          shuffle=False,
                                          sampler=None,
                                          num_workers=num_workers_per_gpu,
                                          pin_memory=ymir_yolov5.pin_memory,
                                          drop_last=False)

    results = []
    dataset_size = len(images_rank)
    monitor_gap = max(1, dataset_size // 1000 // batch_size_per_gpu)
    pbar = tqdm(origin_dataset_loader) if RANK == 0 else origin_dataset_loader
    for idx, batch in enumerate(pbar):
        # batch-level sync, avoid 30min time-out error
        if WORLD_SIZE > 1 and idx < max_barrier_times:
            dist.barrier()

        with torch.no_grad():
            pred,proto = ymir_yolov5.forward(batch['image'].float().to(device), nms=[True,False]) #[nms, nms_with_all_conf]

        if idx % monitor_gap == 0:
            write_ymir_monitor_process(ymir_cfg,
                                       task='infer',
                                       naive_stage_percent=idx * batch_size_per_gpu / dataset_size,
                                       stage=YmirStage.TASK)

        preprocess_image_shape = batch['image'].shape[2:]
        for idx, det in enumerate(pred):  # per image
            box_result_per_image = []
            seg_result_per_image = []


            image_file = batch['image_file'][idx]
            if len(det):

                origin_image_shape = (batch['origin_shape'][0][idx].item(), batch['origin_shape'][1][idx].item())
                # Rescale boxes from img_size to img size
                masks_prob,masks = process_mask(proto[idx], det[:, 6:], det[:, :4], preprocess_image_shape, upsample=True)  # HWC
                det[:, :4] = scale_boxes(preprocess_image_shape, det[:, :4], origin_image_shape).round()
                
                # masks = process_mask_native(proto[idx], det[:, 6:], det[:, :4], origin_image_shape[:2])  # HWC

                box_result_per_image.append(det)
    
                segments = [
                        scale_segments(preprocess_image_shape, x, origin_image_shape, normalize=False)
                        for x in reversed(masks2segments(masks))]
      
                seg_result_per_image.append(segments)



            # assert box_result_per_image.shape == seg_result_per_image.shape
            results.append(dict(image_file=image_file, result=[box_result_per_image,seg_result_per_image]))

    torch.save(results, f'/out/infer_results_{max(0,RANK)}.pt')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def main() -> int:
    ymir_cfg = get_merged_config()
    ymir_yolov5 = YmirYolov5(ymir_cfg)
    
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    run(ymir_cfg, ymir_yolov5)

    # wait all process to save the infer result
    if WORLD_SIZE > 1:
        dist.barrier()

    if RANK in [0, -1]:
        results = []
        for rank in range(WORLD_SIZE):
            results.append(torch.load(f'/out/infer_results_{rank}.pt'))

        ymir_infer_result = dict()
        
        for result in results:
            count = 0
            for img_data in result:
                
                img_file = img_data['image_file']
                # p0_img = cv2.imread(img_file)
                img = cv2.imread(img_file)
                img_shape = img.shape
                base_name = img_file.split('/')[-1]
                # anns = []
                bbox_result=[]
                segmentation_result=[]
                classes=[]
                # img_shapes=[]
                
                for i in range(len(img_data['result'][0])):
                    each_det = img_data['result'][0][i]
       

                    each_det_np = each_det.data.cpu().numpy()
                    
                    for j in range(each_det_np.shape[0]):
 
                        xmin, ymin, xmax, ymax, conf, cls = each_det_np[j, :6].tolist()
                        seg = img_data['result'][1][i][j].reshape(-1).tolist()

                        cls = int(cls)
                        line = [cls, *seg, conf]
                        if not seg or len(seg)<=4:
                            continue
                        if conf < ymir_yolov5.conf_thres:
                            continue
                        if int(cls) >= len(ymir_yolov5.class_names):
                            warnings.warn(f'class index {int(cls)} out of range for {ymir_yolov5.class_names}')
                            continue

                
                        box = [int(xmin), int(ymin), int(xmax - xmin),int(ymax - ymin)]
                        bbox_result.append([box, conf, cls])
                        seg_points = line[1:-1]
                        # poly0_0 = get_paired_coord(seg_points)


                        # cv2.polylines(p0_img, [np.array(poly0_0, dtype=np.int32)], True, compute_color_for_labels(int(cls)), 3)

                        compactedRLEs = maskUtils.frPyObjects([np.array(seg_points)], img_shape[0], img_shape[1])
                        compactedRLE = maskUtils.merge(compactedRLEs)
                        compactedRLE['counts'] = compactedRLE['counts'].decode('utf-8')
               
                        segmentation_result.append(compactedRLE)
                    # cv2.imwrite(f'./infer_result/{time.time()}_{str(cls)}.jpg',p0_img)
                if not segmentation_result:
                    count +=1
                ymir_infer_result = gen_anns_from_dets(img_file,img_shape,bbox_result,segmentation_result,ymir_cfg,ymir_infer_result)

        if 'annotations' not in ymir_infer_result:
            ymir_infer_result['annotations'] = []

        rw.write_infer_result(infer_result=ymir_infer_result,algorithm='segmentation')
    return 0


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        process_error(e)