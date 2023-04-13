
docker run -it --gpus all --rm --ipc host --shm-size='256g'\
 -v $PWD/:/data/ \
 -v /data1:/data1 \
 -v /data1/wudan/ymir-pro-dev/ymir-workplace4/sandbox/work_dir/TaskTypeTraining/t00000010000010464d71681367154/sub_task/t00000010000010464d71681367154/in:/in\
 -v /data1/wudan/ymir-pro-dev/ymir-workplace4/sandbox/work_dir/TaskTypeTraining/t00000010000010464d71681367154/sub_task/t00000010000010464d71681367154/out:/out\
   nanfei666/ymir-executorr:ymir2.1.0-detection-instance-sementic-in-one-v1 bash


  #  t0000001000002e089771681368401    mmseg
  # t00000010000010464d71681367154  instance