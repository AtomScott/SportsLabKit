conda create -y --name soccertrack-test-env python=3.10
conda activate soccertrack-test-env

python --version
python -m pip install .
python -m pip install torch torchvision pytorch-lightning
python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
python -m pytest tests

conda deactivate
conda remove -y --name soccertrack-test-env --all