ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN python -m pip install --no-cache-dir \
    transformers==4.52.4 \
    accelerate==1.4.0 \
    nvidia-ml-py==12.570.86 \
    Pillow==11.1.0 \
    && python -m pip freeze > /opt/environment.lock.txt

ENV HF_HOME=/cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

WORKDIR /work
