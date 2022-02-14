# syntax=docker/dockerfile:1
FROM pytorchlightning/pytorch_lightning

# install opencv
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# install gnu time
RUN apt-get install time

# install python requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/AtomScott/Python-Object-Detection-Metrics.git
RUN pip install ./Python-Object-Detection-Metrics