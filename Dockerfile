ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
# support YMIR=1.0.0, 1.1.0 or 1.2.0
ARG YMIR="1.1.0"


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8
ENV YMIR_VERSION=$YMIR
ENV YOLOV5_SEG_CONFIG_DIR='/app/ymir-yolov5-seg/data'
ENV YOLOV5_DETC_CONFIG_DIR='/app/det-yolov5-tmi/data'


# Install linux package
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev curl wget zip vim \
    build-essential ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG MMCV=1.6.1
RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]


# Copy file from host to docker and install requirements
COPY . /app
ENV FORCE_CUDA="1"


RUN mkdir /img-man && mkdir /img-man/det && mkdir /img-man/instance-seg && mkdir /img-man/semantic-seg \
&& mv /app/det-yolov5-tmi/ymir/img-man/*.yaml /img-man/det && mv /app/ymir-yolov5-seg/ymir/img-man/*.yaml /img-man/instance-seg && mv /app/ymir-mmsegmentation/ymir/img-man/*.yaml /img-man/semantic-seg
COPY manifest.yaml /img-man
COPY ./ymir-yolov5-seg/requirements.txt /workspace/

# install and requirements
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
# install ymir-exc sdk
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.1.0"
RUN pip install -r /app/ymir-mmsegmentation/requirements.txt && pip install -r /workspace/requirements.txt
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Download pretrained weight and font file
# RUN cd /app/det-yolov5-tmi && bash data/scripts/download_weights.sh \
#     && wget https://ultralytics.com/assets/Arial.ttf -O ${YOLOV5_DETC_CONFIG_DIR}/Arial.ttf

# RUN cd /app/ymir-yolov5-seg && bash data/scripts/download_weights.sh \
#     && wget https://ultralytics.com/assets/Arial.ttf -O ${YOLOV5_SEG_CONFIG_DIR}/Arial.ttf


# Make PYTHONPATH find local package
ENV PYTHONPATH=.

WORKDIR /app
RUN echo "python3 start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh