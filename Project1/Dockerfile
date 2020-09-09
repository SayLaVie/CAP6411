FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

RUN pip3 install joblib

# Cache vgg16 pretrained network so we don't have to download it
COPY models/vgg16_bn-6c64b313.pth /root/.cache/torch/hub/checkpoints/

VOLUME ["/infer"]

VOLUME ["/infer/models"]

VOLUME ["/infer/validation"]

WORKDIR /infer

CMD ["python3", "run-eval.py"]
