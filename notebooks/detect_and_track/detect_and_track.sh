
# for size in 5120 2560 1280 640 320  
# do
# for model_name in yolov5n yolov5s yolov5m yolov5l yolov5x yolov5n6 yolov5m6 yolov5l6
# do
docker run \
    -v $PWD/:/PWD \
    -v /mnt:/mnt \
    atomscott/all-in-one:latest \
    /bin/bash -c " \
    python atom/SoccerTrack/notebooks/detect_and_track/detext_and_track.py \
    "