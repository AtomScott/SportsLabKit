
yolov5_path=/home/atom/MiRAI/submodules/yolov5
size=1920
docker run \
    --gpus all \
    --ipc=host \
    -v $PWD/:/PWD \
    -v /mnt:/mnt \
    -v /home:/home \
    atomscott/all-in-one:latest \
    /bin/bash -c " \
    cd $yolov5_path/ && \
    python detect.py \
            --device 1 \
            --exist-ok \
            --weights /mnt/share/atom/SoccerTrackDataset/drone_ball/yolov5/yolov5n-19205/weights/best.pt \
            --source /mnt/share/atom/SoccerTrackDataset/2022_02_19/drone/20220219_drone_merged/20220219_BvsTSC_drone_2.MP4 \
            --save-txt \
            --save-conf \
            --class 0 \
            --img $size \
            --project  /mnt/share/atom/SoccerTrackDataset/drone_ball/ \
            --name '20220219_BvsTSC_drone_2.MP4_yolov5-tta'\
            --conf-thres 0.1 \
            --augment \
            "
