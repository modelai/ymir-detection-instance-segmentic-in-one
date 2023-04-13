import os
import sys

from ymir_exc.executor import Executor
from ymir_exc.util import find_free_port, get_merged_config


def main() -> int:
    ymir_cfg = get_merged_config()
    gpu_id: str = str(ymir_cfg.param.get('gpu_id', '0'))
    gpu_count: int = ymir_cfg.param.get('gpu_count', None) or len(gpu_id.split(','))

    mining_cmd = 'python3 ymir/ymir_mining.py'
    if gpu_count <= 1:
        infer_cmd = 'python3 ymir/ymir_infer.py'
    else:
        port = find_free_port()
        dist_cmd = f'-m torch.distributed.launch --master_port {port} --nproc_per_node {gpu_count}'
        infer_cmd = f'python3 {dist_cmd} ymir/ymir_infer.py'

    apps = dict(training='python3 ymir/ymir_training.py', mining=mining_cmd, infer=infer_cmd)
    executor = Executor(apps)
    executor.start()

    return 0


if __name__ == '__main__':
    # fix mkl-service error, view https://github.com/pytorch/pytorch/issues/37377#issuecomment-629530272 for detail
    os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    sys.exit(main())
