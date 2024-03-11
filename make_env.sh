#!/bin/bash
conda env create -n stainfuser python=3.10 -y
conda activate stainfuser
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install -r requirements.txt

pip install --no-dependencies diffusers["torch"]==0.26.3
pip install -U albumentations==1.3.0 --no-binary qudida,albumentations
pip install --no-dependencies tiatoolbox==1.4.1