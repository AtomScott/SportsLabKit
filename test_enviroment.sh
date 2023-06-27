conda create -y --name soccertrack-test-env python=3.10
conda activate soccertrack-test-env

python --version
python -m pip install -e .
python -m pip install torch torchvision pytorch-lightning
python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
python -m pip install ultralytics
python -m pip install git+https://github.com/openai/CLIP.git

python -m pytest tests

conda deactivate
conda remove -y --name soccertrack-test-env --all