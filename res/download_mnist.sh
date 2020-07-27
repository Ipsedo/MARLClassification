#!/bin/bash

if ! [[ -d "./downloaded" ]]; then
    mkdir downloaded
fi
cd downloaded

# Download mnist PNGs
if ! [[ -f "./mnistzip.zip" ]]; then
    echo "Download MNIST png from Kaggle"
    kaggle datasets download -d playlist/mnistzip
fi

if ! [[ -d "./mnist_png" ]]; then
    echo "Extract mnistzip.zip"
    unzip mnistzip.zip
fi

cd mnist_png
if ! [[ -d "./all_png" ]]; then
    echo "Create all_png folder"
    mkdir all_png
    echo "Copy train img to all_png folder"
    cp -r train/* all_png
    echo "Copy eval img to all_png folder"
    cp -r valid/* all_png
fi
cd ../..