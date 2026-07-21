FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace/FFA-Nettest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and weights are mounted at runtime; default to eval shell
WORKDIR /workspace/FFA-Nettest/net
CMD ["bash"]
