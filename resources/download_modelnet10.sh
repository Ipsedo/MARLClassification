#!/bin/bash

if ! [[ -d "./downloaded" ]]; then
    mkdir downloaded
fi
cd downloaded

# Download ModelNet10 zip
if ! [[ -f "./ModelNet10.zip" ]]; then
    echo "Download ModelNet10 from http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
fi

# Extract ModelNet10
if ! [[ -d "./ModelNet10" ]]; then
    echo "Extract ModelNet10.zip"
    unzip ModelNet10.zip
fi