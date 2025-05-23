FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DOCKER_CUDA_VERSION 11.8.0
# For A100, RTX4090, and H100
ENV TORCH_CUDA_ARCH_LIST "8.0;8.6;8.9;9.0"

ENV DEBIAN_FRONTEND noninteractive

ENV CUDA_HOME /usr/local/cuda
ENV SETUPTOOLS_USE_DISTUTILS stdlib

RUN rm -rf /var/lib/apt/lists/*
RUN apt update -y


# Build tools
RUN apt-get install -y gcc-12 g++-12 ninja-build cmake build-essential autoconf libtool automake git

# Utils
RUN apt-get install -y vim git bzip2 tmux wget tar htop

# Conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh && \
    chmod +x ./Anaconda3-2024.10-1-Linux-x86_64.sh && \
    ./Anaconda3-2024.10-1-Linux-x86_64.sh -b && \
    rm ./Anaconda3-2024.10-1-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH

RUN echo "$DOCKER_CUDA_VERSION" > /CUDA_VERSION

RUN conda init && conda install python=3.10 -y && conda install -c "nvidia/label/cuda-11.8.0" cuda -y
RUN pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118


WORKDIR /
RUN pip install transformers accelerate 'diffusers[torch]' datasets clean-fid torchmetrics

RUN echo "root:root" | chpasswd
RUN apt -y update
RUN apt install -y openssh-server
RUN ssh-keygen -A
RUN mkdir -p /run/sshd
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN apt install -y zsh unzip htop
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
RUN bash -c "source ~/.nvm/nvm.sh && nvm install 22 && npm install -g omniboard"

RUN curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
    gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
    --dearmor
RUN echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-8.0.list
RUN apt-get update && apt-get install -y mongodb-org

RUN pip install sacred PyYAML pymongo pyrallis image-reward \
    && pip install git+https://github.com/openai/CLIP.git

RUN conda init zsh && pip install sentencepiece


