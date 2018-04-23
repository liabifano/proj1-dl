#!/usr/bin/env bash

ENVNAME="deep"
conda env remove -yq -n $ENVNAME &> /dev/null
conda create -yq -n $ENVNAME --file conda.txt #1> /dev/null
source activate $ENVNAME
conda install pytorch torchvision -c pytorch
pip install -e .