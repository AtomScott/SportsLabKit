ARG python_image_v="python:3.10-buster"
FROM ${python_image_v}

WORKDIR /workspace

RUN apt-get -y update
RUN apt-get -y upgrade

# Install ffmpeg
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# install opencv
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# install gnu time
RUN apt-get install time

# Install essential packages
Run apt-get install --no-install-recommends -y curl build-essential 

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y git

################
# Install pandoc
################

RUN apt-get install -y pandoc

EXPOSE 8000
EXPOSE 8080

COPY pyproject.toml .
RUN pip install poetry
RUN poetry install
RUN rm pyproject.toml
