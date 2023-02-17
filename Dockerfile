ARG CUDA_VERSION=11.4.0

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory
WORKDIR /app


# Install required packages
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y git ffmpeg

# install opencv
RUN apt-get update
RUN apt-get install -y python3-opencv

# install gnu time
RUN apt-get install time

# Install essential packages
Run apt-get install --no-install-recommends -y curl build-essential 


# Install pandoc
RUN apt-get install -y pandoc

# Install SoccerTrack and dependencies
COPY pyproject.toml .
RUN pip install -e git+https://github.com/AtomScott/SoccerTrack.git#egg=soccertrack
RUN pip install cython gdown pytorch-lightning pytest
RUN pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

WORKDIR /workspace