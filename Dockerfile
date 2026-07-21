# Build FFA-Net env on top of your local ssid image (no Docker Hub needed):
#   docker build --build-arg BASE_IMAGE=ssid:latest -t ffa-net:latest .
ARG BASE_IMAGE=ssid:latest
FROM ${BASE_IMAGE}

WORKDIR /workspace/FFA-Nettest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace/FFA-Nettest/net
CMD ["bash"]
