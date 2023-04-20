# ymir-detection-instance-segmentic-in-one镜像说明文档

- 支持算法类型： 目标检测， 实例分割， fastscnn语义分割

- 支持任务类型： 训练， 推理， 挖掘

- 版本信息

```
python: 3.8.8
pytorch: 1.8.0
torchvision: 0.9.0
cuda: 11.1
cudnn: 8
```


- 代码仓库 [modelai/ymir-detection-instance-segmentic-in-one](https://github.com/modelai/ymir-detection-instance-segmentic-in-one)

- 镜像地址

```
docker pull nanfei666/ymir-executor:ymir2.1.0-detection-instance-sementic-in-one
```

<details open>
<summary>sementic segmentation</summary>

## 性能表现

>参考 [fastscnn](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastscnn)

### Cityscapes

| Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                      | download                                                                                                                                                                                                                                                                                                                                               |
| -------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FastSCNN | FastSCNN | 512x1024  |  160000 | 3.3      | 56.45          | 70.96 | 72.65         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853.log.json) |
| HRNet    | HRNetV2p-W18-Small | 512x1024  |   40000 | 1.7      | 23.74          | 73.86 |         75.91 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18s_512x1024_40k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_40k_cityscapes/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216-93db27d0.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_40k_cityscapes/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216.log.json)     |
| HRNet    | HRNetV2p-W18       | 512x1024  |   40000 | 2.9      | 12.97          | 77.19 |         78.92 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_40k_cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_40k_cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216.log.json)         |
| HRNet    | HRNetV2p-W48       | 512x1024  |   40000 | 6.2      | 6.42           | 78.48 |         79.69 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_40k_cityscapes/fcn_hr48_512x1024_40k_cityscapes_20200601_014240-a989b146.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_40k_cityscapes/fcn_hr48_512x1024_40k_cityscapes_20200601_014240.log.json)         |
| HRNet    | HRNetV2p-W18-Small | 512x1024  |   80000 | -        | -              | 75.31 |         77.48 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18s_512x1024_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_80k_cityscapes/fcn_hr18s_512x1024_80k_cityscapes_20200601_202700-1462b75d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_80k_cityscapes/fcn_hr18s_512x1024_80k_cityscapes_20200601_202700.log.json)     |
| HRNet    | HRNetV2p-W18       | 512x1024  |   80000 | -        | -              | 78.65 |         80.35 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_80k_cityscapes/fcn_hr18_512x1024_80k_cityscapes_20200601_223255-4e7b345e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_80k_cityscapes/fcn_hr18_512x1024_80k_cityscapes_20200601_223255.log.json)         |
| HRNet    | HRNetV2p-W48       | 512x1024  |   80000 | -        | -              | 79.93 |         80.72 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr48_512x1024_80k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_80k_cityscapes/fcn_hr48_512x1024_80k_cityscapes_20200601_202606-58ea95d6.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_80k_cityscapes/fcn_hr48_512x1024_80k_cityscapes_20200601_202606.log.json)         |
| HRNet    | HRNetV2p-W18-Small | 512x1024  |  160000 | -        | -              | 76.31 |         78.31 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_160k_cityscapes/fcn_hr18s_512x1024_160k_cityscapes_20200602_190901-4a0797ea.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_160k_cityscapes/fcn_hr18s_512x1024_160k_cityscapes_20200602_190901.log.json) |
| HRNet    | HRNetV2p-W18       | 512x1024  |  160000 | -        | -              | 78.80 |         80.74 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18_512x1024_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_160k_cityscapes/fcn_hr18_512x1024_160k_cityscapes_20200602_190822-221e4a4f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_160k_cityscapes/fcn_hr18_512x1024_160k_cityscapes_20200602_190822.log.json)     |
| HRNet    | HRNetV2p-W48       | 512x1024  |  160000 | -        | -              | 80.65 |         81.92 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_160k_cityscapes/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_160k_cityscapes/fcn_hr48_512x1024_160k_cityscapes_20200602_190946.log.json)     |

> ## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| config_file |
| export_format | seg-coco:raw | 字符串| 受ymir后台处理，ymir分割数据集导出格式 | 禁止改变 |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| config_file | configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py | 文件路径 | mmlab配置文件 | 建议采用fastscnn系列, 参考[configs](https://github.com/modelai/ymir-mmsegmentation/tree/master/configs) |
| samples_per_gpu | 2 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 2 | 整数 | 每张GPU对应的数据读取进程数 | 采用默认值即可，若内存及CPU配置高，可适当增大 |
| max_iters | 20000 | 整数 | 数据集的训练批次 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| interval | 2000 | 整数 | 模型在验证集上评测的周期 | 采用默认值即可 |
| args_options | '' | 字符串 | 训练命令行参数 | 参考[tools/train.py]()
| cfg_options | '' | 字符串 | 训练命令行参数 | 参考 [tools/train.py]()
| save_least_file | True | 布尔型 | 是否只保存最优和最新的权重文件 | 设置为True |
| max_keep_ckpts | -1 | 整数 | 当save_least_file为False时，最多保存的权重文件数量 | 设置为k, 可保存k个最优权重和k个最新的权重文件，设置为-1可保存所有权重文件。|
| ignore_black_area | False | 布尔型 | 是否忽略未标注的区域 | 采用默认即可将空白区域当成背景进行训练 |

> ## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| samples_per_gpu | 2 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 2 | 整数 | 每张GPU对应的数据读取进程数 | 采用默认值即可，若内存及CPU配置高，可适当增大 |


> ## 挖掘参数

| 超参数 | 默认值 | 类型 | 可选值 | 说明 |
| hyper-parameter | default value | type | choices | note |
| - | - | - | - | - |
| mining_algorithm | RSAL | str | RSAL, RIPU | 挖掘算法名称 |
| superpixel_algorithm | slico | str | slico, slic, mslic, seeds | 超像素算法名称 |
| uncertainty_method | BvSB | str | BvSB | 不确定性计算方法名称 |
| shm_size | 128G | str | 128G | 容器可使用的共享内存大小 |
| max_superpixel_per_image | 1024 | int | 1024, ... | 一张图像中超像素的数量上限 |
| max_kept_mining_image | 5000 | int | 500, 1000, 2000, 5000, ... | 挖掘图像数量的上限 |
| topk_superpixel_score | 3 | int | 3, 5, 10, ... | 一张图像中采用的超像素数量 |
| class_balance | True | bool | True, False | 是否考虑各类标注的平衡性 |
| fp16 | True | bool | True, False | 是否采用fp16技术加速 |
| samples_per_gpu | 2 | int | 2, 4, ... | batch size per gpu |
| workers_per_gpu | 2 | int | 2 | num_workers per gpu |
| ripu_region_radius | 1 | int | 1, 2, 3  | ripu挖掘算法专用参数 |
</details>

<details>
<summary> instance segmentation</summary>

> ### 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 | 
| :----: | :----: | :----: | :----: |  :----: | 
| shm_size | 128G | 字符串 | 受ymir后台处理，docker image 可用共享内存 |  建议大小：镜像占用GPU数 * 32G | 
| export_format | seg-coco:raw | 字符串 | 受ymir后台处理，ymir数据集导出格式 |  - | 
| model | yolov5s-seg | 字符串 | yolov5模型，可选yolov5n-seg, yolov5s-seg, yolov5m-seg, yolov5l-seg等 |  建议：速度快选yolov5n-seg, 精度高选yolov5l-seg, yolov5x-seg, 平衡选yolov5s-seg或yolov5m-seg | 
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 |  建议大小：显存占用<50% 可增加2倍加快训练速度 | 
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数	 |  - |
| epochs | 100 | 整数 | 整个数据集的训练遍历次数 |  建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| img_size | 640 | 整数	 | 输入模型的图像分辨率 |  - |
| opset | 11 | 整数: int | onnx 导出参数 opset |  建议：一般不需要用到onnx，不必改 |
| args_options | '--exist-ok' | 字符串 | yolov5命令行参数 |  建议：专业用户可用yolov5所有命令行参数 |
| save_best_only | True | 布尔型 | 是否只保存最优模型 |  建议：为节省空间设为True即可 |
| save_period | 10 | 整数 | 保存模型的间隔 |  建议：当save_best_only为False时，可保存 epoch/save_period 个中间结果 |
| sync_bn | False | 布尔型 | 是否同步各gpu上的归一化层 |  建议：开启以提高训练稳定性及精度 |

> ### 推理参数
| 超参数 | 默认值 | 类型 | 说明 | 建议 | 
| :----: | :----: | :----: | :----: |  :----: | 
| shm_size | 128G | 字符串 | 受ymir后台处理，docker image 可用共享内存 |  建议大小：镜像占用GPU数 * 32G | 
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 |  建议大小：显存占用<50% 可增加2倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数	 |采用32的整数倍，224 = 32*7 以上大小- |
| img_size | 640 | 整数 | 模型的输入图像大小	 |  - |
| conf_thres | 	0.25 | 浮点数 | 置信度阈值	 |  - |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值	 |  采用默认值 |
| pin_memory | False | 布尔型	 | 是否为数据集单独固定内存?		 |  内存充足时改为True可加快数据集加载
 |

> ### 挖掘参数
| 超参数 | 默认值 | 类型 | 说明 | 建议 | 
| :----: | :----: | :----: | :----: |  :----: | 
| shm_size | 128G | 字符串 | 受ymir后台处理，docker image 可用共享内存 |  建议大小：镜像占用GPU数 * 32G | 
| mining_algorithm | apis | 字符串 | 挖掘算法名称，可选 apis, aldd, cald, maskAL	 |  - |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 |  建议大小：显存占用<50% 可增加2倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数	 |采用32的整数倍，224 = 32*7 以上大小- |
| img_size | 640 | 整数 | 模型的输入图像大小	 |  - |
| conf_thres | 	0.25 | 浮点数 | 置信度阈值	 |  - |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值	 |  采用默认值 |
| pin_memory | False | 布尔型	 | 是否为数据集单独固定内存?		 |  内存充足时改为True可加快数据集加载
 |
</details>
<details>
<summary> object detection</summary>

### 性能表现

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|[YOLOv5n]      |640  |28.0   |45.7   |**45** |**6.3**|**0.6**|**1.9**|**4.5**
|[YOLOv5s]      |640  |37.4   |56.8   |98     |6.4    |0.9    |7.2    |16.5
|[YOLOv5m]      |640  |45.4   |64.1   |224    |8.2    |1.7    |21.2   |49.0
|[YOLOv5l]      |640  |49.0   |67.3   |430    |10.1   |2.7    |46.5   |109.1
|[YOLOv5x]      |640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7
|                       |     |       |       |       |       |       |       |
|[YOLOv5n6]     |1280 |36.0   |54.4   |153    |8.1    |2.1    |3.2    |4.6
|[YOLOv5s6]     |1280 |44.8   |63.7   |385    |8.2    |3.6    |16.8   |12.6
|[YOLOv5m6]     |1280 |51.3   |69.3   |887    |11.1   |6.8    |35.7   |50.0
|[YOLOv5l6]     |1280 |53.7   |71.3   |1784   |15.8   |10.5   |76.8   |111.4

> ### 训练参数说明

- 一些参数由ymir后台生成，如 `gpu_id`, `class_names` 等参数
  - `gpu_id`: 使用的GPU硬件编号，如`0,1,2`，类型为 `str`。实际上对应的主机GPU随机，可能为`3,1,7`，镜像中只能感知并使用`0,1,2`作为设备ID。
  - `task_id`: ymir任务id, 类型为 `str`
  - `pretrained_model_params`: 预训练模型文件的路径，类型为 `List[str]`
  - `class_names`: 类别名，类型为 `List[str]`

- 一些参数由ymir后台进行处理，如 `shm_size`, `export_format`， 其中 `shm_size` 影响到docker镜像所能使用的共享内存，若过小会导致 `out of memory` 等错误。 `export_format` 会决定docker镜像中所看到数据的格式



| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| model | yolov5s | 字符串 | yolov5模型，可选yolov5n, yolov5s, yolov5m, yolov5l等 | 建议：速度快选yolov5n, 精度高选yolov5l, yolov5x, 平衡选yolov5s或yolov5m |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| epochs | 100 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| img_size | 640 | 整数 | 输入模型的图像分辨率 | - |
| opset | 11 | 整数 | onnx 导出参数 opset | 建议：一般不需要用到onnx，不必改 |
| args_options | '--exist-ok' | 字符串 | yolov5命令行参数 | 建议：专业用户可用yolov5所有命令行参数 |
| save_best_only | True | 布尔型 | 是否只保存最优模型 | 建议：为节省空间设为True即可 |
| save_period | 10 | 整数 | 保存模型的间隔 | 建议：当save_best_only为False时，可保存 `epoch/save_period` 个中间结果
| sync_bn | False | 布尔型 | 是否同步各gpu上的归一化层 | 建议：开启以提高训练稳定性及精度 |
| activate | '' | 字符串 | 激活函数，默认为nn.Hardswish(), 参考 [pytorch激活函数](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) | 可选值: ELU, Hardswish, LeakyReLU, PReLU, ReLU, ReLU6, SiLU, ... |

---

> ## 推理: infer

推理任务中，ymir后台会生成参数 `gpu_id`, `class_names`, `task_id` 与 `model_param_path`， 其中`model_param_path`与训练任务中的`pretrained_model_params`类似。

> ### 推理参数说明
| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| img_size | 640 | 整数 | 模型的输入图像大小 | 采用32的整数倍，224 = 32*7 以上大小 |
| conf_thres | 0.25 | 浮点数 | 置信度阈值 | 采用默认值 |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值 | 采用默认值 |
| batch_size_per_gpu | 16 | 整数| 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加1倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数| 每张GPU对应的数据读取进程数 | - |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| pin_memory | False | 布尔型 | 是否为数据集单独固定内存? | 内存充足时改为True可加快数据集加载 |


---

> ## 挖掘: mining

挖掘任务中，ymir后台会生成参数 `gpu_id`, `class_names`, `task_id` 与 `model_param_path`， 其中`model_param_path`与训练任务中的`pretrained_model_params`类似。推理与挖掘任务ymir后台生成的参数一样。

> ### 挖掘参数说明

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| img_size | 640 | 整数 | 模型的输入图像大小 | 采用32的整数倍，224 = 32*7 以上大小 |
| mining_algorithm | aldd | 字符串 | 挖掘算法名称，可选 random, aldd, cald, entropy | 建议单类检测采用aldd，多类检测采用entropy |
| class_distribution_scores | '' | List[float]的字符表示 | aldd算法的类别平衡参数 | 不用更改， 专业用户可根据各类比较调整，如对于4类检测，用 `1.0,1.0,0.1,0.2` 降低后两类的挖掘比重 |
| conf_thres | 0.25 | 浮点数 | 置信度阈值 | 采用默认值 |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值 | 采用默认值 |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加1倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| shm_size | 128G | 字符串 | 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| pin_memory | False | 布尔型 | 是否为数据集单独固定内存? | 内存充足时改为True可加快数据集加载 |

</details>

## 主要改动：main change log

- change activetion to `SiLU`

- change `export.py` in detecion & instance segmentation  to support RV1126

- change letterbox to `resize` directly