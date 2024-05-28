ARG BASE_VERSION=23.11
FROM nvcr.io/nvidia/pytorch:${BASE_VERSION}-py3 as base

ENV DEBIAN_FRONTEND noninteractive

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN apt-get clean
# this line change to ubuntu archive
RUN apt-get update -y
RUN apt-get install vim -y
RUN apt-get install tmux -y
RUN apt-get install sudo -y

RUN addgroup --gid $GROUP_ID $USER_NAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME
RUN echo "$USER_NAME:$USER_NAME"|chpasswd
RUN adduser $USER_NAME sudo
RUN echo "$USER_NAME:$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN pip install pyhocon
RUN pip install timm
RUN pip install wandb
RUN pip install easydict
RUN pip install xlwt

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
