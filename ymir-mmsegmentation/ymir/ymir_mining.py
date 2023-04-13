import subprocess
import sys

from ymir_exc.util import find_free_port, get_merged_config


def main() -> int:
    ymir_cfg = get_merged_config()
    gpu_id: str = str(ymir_cfg.param.get('gpu_id', '0'))
    gpu_count: int = ymir_cfg.param.get('gpu_count', None) or len(gpu_id.split(','))

    algorithm = ymir_cfg.param.get('mining_algorithm').lower()
    if gpu_count <= 1:
        command = f'python3 ymir/ymir_mining_{algorithm}.py'
    else:
        port = find_free_port()
        dist_cmd = f'-m torch.distributed.launch --master_port {port} --nproc_per_node {gpu_count}'
        command = f'python3 {dist_cmd} ymir/ymir_mining_{algorithm}.py'

    subprocess.run(command.split(), check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
