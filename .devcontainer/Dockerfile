FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04

RUN apt update -y && \
    apt install -y \
    curl \
    gcc \
    git \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    patchelf \
    python3 \
    sudo

RUN groupadd -g 1005 vglusers && \
    useradd -ms /bin/bash vscode -u 1000 -g 1005 && \
    usermod -a -G video,sudo vscode

RUN echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

USER vscode
ADD --chown=vscode https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz /home/vscode/.mujoco/
RUN tar xfz /home/vscode/.mujoco/mujoco210-linux-x86_64.tar.gz -C /home/vscode/.mujoco/ && \
    rm /home/vscode/.mujoco/mujoco210-linux-x86_64.tar.gz
RUN curl -fsSL https://pixi.sh/install.sh | bash
