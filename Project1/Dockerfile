FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

# Needed for running tegrastats script from Docker. Container must have already connected to host for sshpass to work (answer yes to RSA question)
RUN apt-get update && apt-get -y install sshpass

RUN pip3 install --upgrade pip && pip3 install joblib termcolor

# Cache vgg16 pretrained network so we don't have to download it
COPY models/vgg16_bn-6c64b313.pth /root/.cache/torch/hub/checkpoints/

ENV PYTHONPATH /torch2trt

VOLUME ["/torch2trt"]

VOLUME ["/infer"]

WORKDIR /infer

CMD ["python3", "run-eval.py"]
