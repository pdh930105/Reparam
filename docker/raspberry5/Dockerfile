FROM ultralytics/ultralytics:latest-arm64 as base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get clean
# this line change to ubuntu archive
RUN apt-get update -y
RUN apt-get install vim -y
RUN apt-get install tmux -y
RUN pip install timm==1.0.3
RUN pip install easydict
RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.3/index.html
RUN pip install mmpretrain
RUN apt-get install -y libgl1-mesa-glx