"""
code based on https://github.com/chufengt/APIS
paper : https://arxiv.org/abs/2207.11493

use fake DDP to infer
1. split data with `images_rank = images[RANK::WORLD_SIZE]`
2. infer on the origin dataset
3. infer on the augmentation dataset
4. save splited mining result with `torch.save(results, f'/out/mining_results_{RANK}.pt')`
5. merge mining result
"""
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as td
from easydict import EasyDict as edict
from tqdm import tqdm
from ymir.mining.util import YmirDataset, load_image_file, observations
from ymir.ymir_yolov5 import YmirYolov5, process_error
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process
from utils.segment.general import masks2segments, process_mask, process_mask_native,gen_anns_from_dets
from utils.general import scale_boxes, scale_segments
from dropblock import DropBlock2D
import cv2

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
EPS = 1e-12



def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    drop_block = DropBlock2D(block_size=7, drop_prob=0.1)
    for index,m in enumerate(model.model.model.model):
        if index in [3,5,7]:
            m.dropblock = drop_block
            m.dropblock.train()
# @Author: Pieter Blok
# @Date:   2021-03-25 15:06:20
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-11-29 11:50:03

# This function is inspired by the uncertainty_aware_dropout function:
# https://github.com/RovelMan/active-learning-framework/blob/master/al_framework/strategies/dropout.py

def calculate_max_entropy(classes):
    least_confident = np.divide(np.ones(len(classes)), len(classes)).astype(np.float32)
    probs = torch.from_numpy(least_confident)
    max_entropy = torch.distributions.Categorical(probs).entropy()
    return max_entropy

def uncertainty(observations, iterations, max_entropy, device, mode = 'min'):
    """
    To calculate the uncertainty metrics on the observations
    observations {batch_idx:{obs_id:[dection]}}
    detection {"segments":[],"box":[]}
    """

    
    batch_points=[]
    # entropies_batches[batch_idx] = 
    for batch_idx , observation in observations.items():
        uncertainty_list = []
        for obs_id in observation.keys():
            
            detections = observation[obs_id]
            entropies = torch.stack([torch.distributions.Categorical(torch.tensor(detection['box'][1])).entropy() for detection in detections])
            entropies_norm = torch.stack([torch.divide(entropy, max_entropy.to(device)) for entropy in entropies])
            inv_entropies_norm = torch.stack([torch.subtract(torch.ones(1).to(device), entropy_norm) for entropy_norm in entropies_norm])
        
            mean_bbox = torch.mean(torch.stack([torch.tensor(detection['box'][0]) for detection in detections]), axis=0)
            mean_mask = torch.mean(torch.stack([torch.tensor(detection['segments']).flatten().type(torch.cuda.FloatTensor) for detection in detections]), axis=0)
            mean_mask[mean_mask < 0.25] = 0.0

            img = cv2.imread(detections[0]['file_name'])
            width, height,_ = img.shape
            mean_mask = mean_mask.reshape(-1, width, height).cuda()
       
            mask_IOUs = []
            for detection in detections:
                current_mask = torch.tensor(detection['segments']).permute(2,0,1).type(torch.cuda.FloatTensor)
   
                overlap = torch.logical_and(mean_mask, current_mask)
                union = torch.logical_or(mean_mask, current_mask)
                if union.sum() > 0:
                    IOU = torch.divide(overlap.sum(), union.sum())
                    mask_IOUs.append(IOU.unsqueeze(0))

            if len(mask_IOUs) > 0:
                mask_IOUs = torch.cat(mask_IOUs)
            else:
                mask_IOUs = torch.tensor([float('NaN')]).to(device)

            bbox_IOUs = []
            
            # mean_bbox = mean_bbox.squeeze(0)

            boxAArea = torch.multiply((mean_bbox[2] - mean_bbox[0] + 1), (mean_bbox[3] - mean_bbox[1] + 1))
            for detection in detections:
                current_bbox = torch.tensor(detection['box'][0])
                xA = torch.max(mean_bbox[0], current_bbox[0])
                yA = torch.max(mean_bbox[1], current_bbox[1])
                xB = torch.min(mean_bbox[2], current_bbox[2])
                yB = torch.min(mean_bbox[3], current_bbox[3])
                interArea = torch.multiply(torch.max(torch.tensor(0).to(device), xB - xA + 1), torch.max(torch.tensor(0).to(device), yB - yA + 1))
                boxBArea = torch.multiply((current_bbox[2] - current_bbox[0] + 1), (current_bbox[3] - current_bbox[1] + 1))
                bbox_IOU = torch.divide(interArea, (boxAArea + boxBArea - interArea))
                bbox_IOUs.append(bbox_IOU.unsqueeze(0))

            if len(bbox_IOUs) > 0:
                bbox_IOUs = torch.cat(bbox_IOUs)
            else:
                bbox_IOUs = torch.tensor([float('NaN')]).to(device)
            # print('mask_IOUs',mask_IOUs)
            # print('bbox_IOUs',bbox_IOUs)

            val_len = torch.tensor(len(detections)).to(device)
            outputs_len = torch.tensor(iterations).to(device)

            u_sem = torch.clamp(torch.mean(inv_entropies_norm), min=0, max=1)
            
            u_spl_m = torch.clamp(torch.divide(mask_IOUs.sum(), val_len), min=0, max=1)
            u_spl_b = torch.clamp(torch.divide(bbox_IOUs.sum(), val_len), min=0, max=1)
            u_spl = torch.multiply(u_spl_m, u_spl_b)

            u_sem_spl = torch.multiply(u_sem, u_spl)
            
            try:
                u_n = torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1)
            except:
                u_n = 0.0

            u_h = torch.multiply(u_sem_spl, u_n)
            uncertainty_list.append(u_h.unsqueeze(0))


        if uncertainty_list:
            uncertainty_list = torch.cat(uncertainty_list)

            if mode == 'min':
                uncertainty = torch.min(uncertainty_list)
            elif mode == 'mean':
                uncertainty = torch.mean(uncertainty_list)
            elif mode == 'max':
                uncertainty = torch.max(uncertainty_list)
            else:
                uncertainty = torch.mean(uncertainty_list)
                
        else:
            uncertainty = torch.tensor([float('NaN')]).to(device)

        points = uncertainty.detach().cpu().numpy().squeeze(0)
       
        batch_points.append(points.item() if not np.isnan(points) else float('-inf'))

    return batch_points


def run(ymir_cfg: edict, ymir_yolov5: YmirYolov5):
    # eg: gpu_id = 1,3,5,7  for LOCAL_RANK = 2, will use gpu 5.
    mcd_iterations = 5

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
    batch_size_per_gpu: int = ymir_yolov5.batch_size_per_gpu
    gpu_count: int = ymir_yolov5.gpu_count
    cpu_count: int = os.cpu_count() or 1
    num_workers_per_gpu = min([
        cpu_count // max(gpu_count, 1), batch_size_per_gpu if batch_size_per_gpu > 1 else 0,
        ymir_yolov5.num_workers_per_gpu
    ])

    with open(ymir_yolov5.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    max_barrier_times = (len(images) // max(1, WORLD_SIZE)) // batch_size_per_gpu
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

    mining_results = dict()
    dataset_size = len(images_rank)
    pbar = tqdm(origin_dataset_loader) if RANK in [0, -1] else origin_dataset_loader

    
    enable_dropout(ymir_yolov5)
    max_entropy = calculate_max_entropy(ymir_cfg.param.class_names)
    mining_results = dict()
    num_calss = len(ymir_cfg.param.class_names)
    for idx, batch in enumerate(pbar):
        batch_size = batch_size_per_gpu

        prediction_dict={"box":{i:[] for i in range(batch_size)},"segments":{i:[] for i in range(batch_size)},'image_file':{i:[] for i in range(batch_size)}}

        # batch-level sync, avoid 30min time-out error
        if WORLD_SIZE > 1 and idx < max_barrier_times:
            dist.barrier()
        if RANK in [-1, 0]:
            write_ymir_monitor_process(ymir_cfg, task='mining', naive_stage_percent=idx * batch_size_per_gpu / dataset_size, stage=YmirStage.TASK)
        
        for i in range(mcd_iterations):
         
            with torch.no_grad():
                pred,proto = ymir_yolov5.forward(batch['image'].float().to(device), nms=[False,True]) #[nms, nms_with_all_conf]

                


            preprocess_image_shape = batch['image'].shape[2:]

            for inner_idx, det in enumerate(pred):  # per image
                
                image_file = batch['image_file'][inner_idx]
                prediction_dict["image_file"][inner_idx].append(image_file)
                if len(det):
        
                    origin_image_shape = (batch['origin_shape'][0][inner_idx].item(), batch['origin_shape'][1][inner_idx].item())

                    # Rescale boxes from img_size to img size
                    masks_prob,masks = process_mask(proto[inner_idx], det[:, 5+num_calss:], det[:, :4], preprocess_image_shape, upsample=True)  # HWC
                    det[:, :4] = scale_boxes(preprocess_image_shape, det[:, :4], origin_image_shape).round()
                    # masks_prob,masks = process_mask(proto[inner_idx], det[:, 6:], det[:, :4], preprocess_image_shape, upsample=True)  # HWC
                    # det[:, :4] = scale_boxes(preprocess_image_shape, det[:, :4], origin_image_shape).round()
                    
                    # masks = process_mask_native(proto[idx], det[:, 6:], det[:, :4], origin_image_shape[:2])  # HWC
        
                    segments = [
                            scale_segments(preprocess_image_shape, x, origin_image_shape, normalize=False)
                            for x in reversed(masks2segments(masks))]

                    prediction_dict["box"][inner_idx].append(det[:, :5+num_calss])
                    prediction_dict["segments"][inner_idx].append(segments)
                    # prediction_dict["image_file"][inner_idx].append(image_file)

        obs = observations(prediction_dict, num_calss, ymir_cfg.param.iou_thres)
        img_uncertainty_batch = uncertainty(obs, mcd_iterations, max_entropy, device, mode='mean') ## reduce the iterations when facing a "CUDA out of memory" error
        
        for batch_idx in prediction_dict["image_file"].keys():
            if len(prediction_dict["image_file"][batch_idx]):
                mining_results[prediction_dict["image_file"][batch_idx][0]] = img_uncertainty_batch[batch_idx]

 

    torch.save(mining_results, f'/out/mining_results_{max(0,RANK)}.pt')


def main() -> int:
    ymir_cfg = get_merged_config()
    ymir_yolov5 = YmirYolov5(ymir_cfg)

    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    run(ymir_cfg, ymir_yolov5)

    # wait all process to save the mining result
    if WORLD_SIZE > 1:
        dist.barrier()

    if RANK in [0, -1]:
        results = []
        for rank in range(WORLD_SIZE):
            results.append(torch.load(f'/out/mining_results_{rank}.pt'))

        ymir_mining_result = []
        for result in results:
            for img_file, score in result.items():
                ymir_mining_result.append((img_file, score))
        rw.write_mining_result(mining_result=ymir_mining_result)
    return 0


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        process_error(e)