# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from ymir_exc.util import (YmirStage, get_merged_config, write_ymir_monitor_process, write_ymir_training_result)


def write_best_ymir_result_file(ymir_cfg, runner) -> int:
    best_score = runner.meta['hook_msgs'].get('best_score', 0)
    best_ckpt_path = runner.meta['hook_msgs'].get('best_ckpt', '')
    if best_ckpt_path and best_score > 0:
        mmseg_config_files = glob.glob(osp.join(ymir_cfg.ymir.output.models_dir, '*.py'))

        # note some early weight files may be removed, use the same id to ensure the weight files be valid.
        evaluation_result = dict(mIoU=float(best_score))
        write_ymir_training_result(ymir_cfg,
                                   evaluation_result=evaluation_result,
                                   files=mmseg_config_files + [best_ckpt_path],
                                   id='best')
    return 0


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, pre_eval=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        self.ymir_cfg = get_merged_config()

        if efficient_test:
            warnings.warn('DeprecationWarning: ``efficient_test`` for evaluation hook '
                          'is deprecated, the evaluation hook is CPU memory friendly '
                          'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                          'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)
            write_best_ymir_result_file(self.ymir_cfg, runner)

    def after_train_epoch(self, runner):
        """Report the training process for ymir"""
        if self.by_epoch:
            monitor_interval = max(1, runner.max_epochs // 1000)
            if runner.epoch % monitor_interval == 0:
                write_ymir_monitor_process(self.ymir_cfg,
                                           task='training',
                                           naive_stage_percent=runner.epoch / runner.max_epochs,
                                           stage=YmirStage.TASK)
        super().after_train_epoch(runner)

    def after_train_iter(self, runner):
        if not self.by_epoch:
            monitor_interval = max(1, runner.max_iters // 1000)
            if runner.iter % monitor_interval == 0:
                write_ymir_monitor_process(self.ymir_cfg,
                                           task='training',
                                           naive_stage_percent=runner.iter / runner.max_iters,
                                           stage=YmirStage.TASK)
        super().after_train_iter(runner)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, pre_eval=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        self.ymir_cfg = get_merged_config()
        if efficient_test:
            warnings.warn('DeprecationWarning: ``efficient_test`` for evaluation hook '
                          'is deprecated, the evaluation hook is CPU memory friendly '
                          'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                          'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(runner.model,
                                 self.dataloader,
                                 tmpdir=tmpdir,
                                 gpu_collect=self.gpu_collect,
                                 pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
                write_best_ymir_result_file(self.ymir_cfg, runner)

    def after_train_epoch(self, runner):
        """Report the training process for ymir"""
        if self.by_epoch and runner.rank == 0:
            monitor_interval = max(1, runner.max_epochs // 1000)
            if runner.epoch % monitor_interval == 0:
                write_ymir_monitor_process(self.ymir_cfg,
                                           task='training',
                                           naive_stage_percent=runner.epoch / runner.max_epochs,
                                           stage=YmirStage.TASK)
        super().after_train_epoch(runner)

    def after_train_iter(self, runner):
        if not self.by_epoch and runner.rank == 0:
            monitor_interval = max(1, runner.max_iters // 1000)
            if runner.iter % monitor_interval == 0:
                write_ymir_monitor_process(self.ymir_cfg,
                                           task='training',
                                           naive_stage_percent=runner.iter / runner.max_iters,
                                           stage=YmirStage.TASK)
        super().after_train_iter(runner)
