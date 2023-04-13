# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# docker build -t youdaoyzbx/ymir-executor:ymir1.3.0-mmseg-cu111-base -f docker/cuda111.dockerfile .
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-get update && apt-get install -y gnupg2
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install linux package
RUN sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && apt-get update && apt-get install -y build-essential ninja-build gnupg2 git libglib2.0-0 \
    libsm6 libxrender-dev libxext6 libgl1-mesa-glx curl wget zip vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV, 1.3.13 <= MMCV < 1.7.0
ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG MMCV=1.6.1
RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

# Install MMSegmentation
# RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
COPY . /mmsegmentation
WORKDIR /mmsegmentation
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONPATH=.
RUN mkdir -p /img-man && mv /mmsegmentation/ymir/img-man/*.yaml /img-man && \
    echo "python3 /mmsegmentation/ymir/start.py" > /usr/bin/start.sh

CMD bash /usr/bin/start.sh
