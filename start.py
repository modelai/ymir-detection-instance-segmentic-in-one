import os
import subprocess

from ymir_exc.util import get_merged_config
cfg = get_merged_config()
object_type = cfg.param.get('object_type',None)
assert object_type != None
if int(object_type) == 2:
    os.chdir('det-yolov5-tmi')
elif int(object_type) == 4:
    os.chdir('ymir-yolov5-seg')
elif int(object_type) == 3:
    os.chdir('ymir-mmsegmentation')
commands = ['python3']
commands.extend(['ymir/start.py'])
subprocess.run(commands, check=True)