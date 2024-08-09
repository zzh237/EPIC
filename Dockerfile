FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04
# TODO multi-stage build, this is pretty similar to vscode

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

# RUN groupadd -g 1005 vglusers && \
#     useradd -ms /bin/bash vscode -u 1000 -g 1005 && \
#     usermod -a -G video,sudo vscode

# RUN echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode \
#     && chmod 0440 /etc/sudoers.d/vscode

# USER vscode
ADD https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz /root/.mujoco/
RUN tar xfz /root/.mujoco/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/ && \
    rm /root/.mujoco/mujoco210-linux-x86_64.tar.gz  
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/bin"
RUN curl -fsSL https://pixi.sh/install.sh | bash

COPY ./ /epic

WORKDIR /epic
RUN --mount=target=/root/.cache/rattler,type=cache /root/.pixi/bin/pixi install