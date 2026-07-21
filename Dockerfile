# Build with a base image already on your server (no Docker Hub pull needed):
#   docker images                          # find a local image
#   docker build --build-arg BASE_IMAGE=<your-local-pytorch-image> -t ffa-nettest:latest .
#
# Example if you have nvidia pytorch image locally:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:23.10-py3 -t ffa-nettest:latest .
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.10-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace/FFA-Nettest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace/FFA-Nettest/net
CMD ["bash"]
