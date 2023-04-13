FROM youdaoyzbx/ymir-executor:ymir2.0.0-mmseg-cu111-base

COPY . /mmsegmentation
WORKDIR /mmsegmentation
ENV FORCE_CUDA="1"
# RUN pip install -r requirements.txt

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONPATH=.
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.1.0"
RUN mkdir -p /img-man && mv /mmsegmentation/ymir/img-man/*.yaml /img-man && \
    echo "python3 /mmsegmentation/ymir/start.py" > /usr/bin/start.sh

CMD bash /usr/bin/start.sh
