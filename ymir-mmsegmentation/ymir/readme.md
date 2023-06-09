# ymir-mmsegmentation镜像说明文档

- 支持任务类型： 训练， 推理， 挖掘

- 支持算法： fastscnn语义分割

- 版本信息

```
python: 3.8.8
pytorch: 1.8.0
torchvision: 0.9.0
cuda: 11.1
cudnn: 8
mmcv: 1.6.1
mmsegmentation: 0.27.0+
```

## 镜像信息

> 参考仓库[open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

- 代码仓库[modelai/ymir-mmsegmentation](https://github.com/modelai/ymir-mmsegmentation)

- 镜像地址

```
docker pull youdaoyzbx/ymir-executor:ymir2.1.0-mmseg-cu111-tmi
```

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

## 训练参数

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

## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| samples_per_gpu | 2 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 2 | 整数 | 每张GPU对应的数据读取进程数 | 采用默认值即可，若内存及CPU配置高，可适当增大 |


## 挖掘参数

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

## 镜像制作

- [YMIR语义分割镜像制作](https://ymir-executor-fork.readthedocs.io/zh/latest/image_segmentation/simple_semantic_seg_training/)

- [mmsegmentation简介](https://ymir-executor-fork.readthedocs.io/zh/latest/algorithms/mmseg/)
