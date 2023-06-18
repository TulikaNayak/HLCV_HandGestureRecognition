FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt update \
   && apt install -y gcc

COPY requirements.txt .
RUN /bin/bash -c "pip install -r requirements.txt"
