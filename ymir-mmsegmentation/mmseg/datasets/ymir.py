from typing import Any, Dict, List

from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class YmirDataset(CustomDataset):
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split) -> List[Dict]:
        with open(split, 'r') as fp:
            lines = fp.readlines()

        img_infos = []
        if len(lines[0].strip().split()) == 1:
            train_mode = False
        else:
            train_mode = True

        for line in lines:
            if not train_mode:
                img_info: Dict[str, Any] = dict(filename=line.strip())
            else:
                img_path, ann_path = line.strip().split()
                img_info = dict(filename=img_path, ann=dict(seg_map=ann_path))
            img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
