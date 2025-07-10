#!/bin/bash

# 1. torch-geometric has dependency to torch-scatter without mentioning it in setup.py :(
# 2. torch-scatter's setup.py import torch, so it have to be installed already :(
# 3. ad-hoc solution: install first line of requirements.txt that is hardcoded to be torch :crazy:

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt