# soccertrack root directory
git_root=$(git rev-parse --show-toplevel)

# date
dt=$(date '+%Y-%m-%d_%H-%M-%S')
python $git_root/external/yolov5/train.py \
    --project $git_root/logs/yolov5 \
    --name $dt \
    --data $git_root/data/yolov5/soccertrack_data.yaml \
    --weights $git_root/models/yolov5/yolov5s.pt \
    --seed 0 \
    --batch 4 \
    --img 640 \
    --epochs 10