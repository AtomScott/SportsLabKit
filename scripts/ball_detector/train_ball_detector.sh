dataset_name=soccer_dataset
model_name=yolov5n
size=1920

yolov5_path=/home/atom/MiRAI/submodules/yolov5

docker run \
    --gpus all \
    --ipc=host \
    -v $PWD/:/PWD \
    -v /mnt:/mnt \
    -v /home:/home \
    atomscott/all-in-one:latest \
    /bin/bash -c " \
    cd $yolov5_path/ && \
    python train.py \
        --data /mnt/share/atom/SoccerTrackDataset/drone_ball/soccertrack_ball_dataset.yaml \
        --cfg models/$model_name.yaml \
        --weights /$WEIGHT_DIR/yolov5/$model_name.pt \
        --img $size \
        --epochs 300 \
        --name $model_name-$size \
        --batch-size 16 \
        --project  /mnt/share/atom/SoccerTrackDataset/drone_ball/yolov5
    "

# docker run \
#     --gpus all \
#     --ipc=host \
#     -v $PWD/:/PWD \
#     -v /mnt:/mnt \
#     atomscott/all-in-one:latest \
#     /bin/bash -c " \
#     python /PWD/submodules/yolov5_20220131/train.py \
#         --data /PWD/submodules/yolov5/data/$dataset_name.yaml \
#         --cfg /PWD/submodules/yolov5_20220131/models/$model_name.yaml \
#         --weights /$WEIGHT_DIR/yolov5/$model_name.pt \
#         --img $size \
#         --epochs 500 \
#         --nosave \
#         --hyp /PWD/submodules/yolov5_20220131/runs/evolve/$model_name-$size-evolved_20220131/hyp_evolve.yaml \
#         --project $LOG_DIR/weights/ \
#         --name $model_name-$size-evolved_20220131 \
#     "

# python export.py --weights yolov5s.pt --include coreml 