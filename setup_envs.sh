#!/bin/bash
conda create -n pyg_web_cpu python=3.7 -y
eval "$(conda shell.bash hook)"
conda activate pyg_web_cpu
conda install pytorch==1.10.0 torchvision torchaudio cpuonly -c pytorch -y
conda install rdkit -c rdkit -y
conda install -c dglteam dgl -y
conda install pyg -c pyg -y
conda install psutil -y
pip install git+https://github.com/bp-kelley/descriptastorus
pip install scikit-learn==1.0.2 xgboost==1.5.1 flask==2.0.2 flask-bootstrap==3.3.7.1 wtforms==3.0.1 DeepPurpose
git clone https://github.com/chemprop/chemprop.git ../chemprop
cd ../chemprop && pip install -e .
cd ../mfdd_prediction_web
pip install -e .
pip install -e web_core/admet_predictor/trimnet