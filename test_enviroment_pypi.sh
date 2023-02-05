conda create -y --name soccertrack-test-env python=3.10
conda activate soccertrack-test-env

python --version
python -m pip install soccertrack
python -m pytest tests

conda deactivate
conda remove -y --name soccertrack-test-env --all