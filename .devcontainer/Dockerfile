# syntax=docker/dockerfile:1.2
FROM --platform=linux/amd64 "mcr.microsoft.com/devcontainers/python:1-3.8-bookworm"

RUN install -d -o vscode -g vscode /home/vscode/.cache

RUN apt update -y && \
    apt install -y \
    gcc \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    patchelf \
    pipx

RUN pipx install poetry==1.8.3

ADD https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz /home/vscode/.mujoco/
RUN tar xfz /home/vscode/.mujoco/mujoco210-linux-x86_64.tar.gz -C /home/vscode/.mujoco/ && \
    rm /home/vscode/.mujoco/mujoco210-linux-x86_64.tar.gz
