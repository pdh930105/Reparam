FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3 as base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get clean
# this line change to ubuntu archive
RUN apt-get update -y
RUN apt-get install vim -y
RUN apt-get install tmux -y
RUN apt-get install sudo -y

RUN pip3 install timm
RUN pip3 install easydict

RUN addgroup --gid $GROUP_ID $USER_NAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME
RUN echo "$USER_NAME:$USER_NAME"|chpasswd
RUN adduser $USER_NAME sudo
RUN echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN pip install wandb
RUN pip install easydict
RUN pip install mmcv
RUN pip install mmpretrain
RUN apt-get install -y libgl1-mesa-glx

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
