# CODD-Pred: a web server for efficient target identification and bioactivity prediction of small molecules
## Overview
This repository contains the main files and model frameworks of front end and back end of our server(http://codd.iddd.group/).
## Requirements
    - rdkit == 2020.09.1
    - python == 3.7
    - flask == 2.0.2
    - pyg == 2.0.2
    - dgl == 0.8.0
    - dgllife == 0.2.9
    - pytorch == 1.10.0
    - torchvision == 0.7.0
    - scikit-learn == 1.0.2
## Creat a new environment in conda
    conda env create -f envs.yml 
## Build Remode apps in your flask web
    python app.py