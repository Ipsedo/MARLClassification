#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

# Download mnist PNGs
if ! [[ -f "${SCRIPT_DIR}/downloaded/mnistzip.zip" ]]; then
    echo "Download MNIST png from Kaggle"
    kaggle datasets download -d playlist/mnistzip -p "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/mnist_png" ]]; then
    echo "Extract mnistzip.zip"
    unzip "${SCRIPT_DIR}/downloaded/mnistzip.zip" -d "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/mnist_png/all_png" ]]; then
    echo "Create all_png folder"
    mkdir "${SCRIPT_DIR}/downloaded/mnist_png/all_png"

    echo "Copy train img to all_png folder"
    cp -r "${SCRIPT_DIR}/downloaded/mnist_png/train/"* "${SCRIPT_DIR}/downloaded/mnist_png/all_png"

    echo "Copy eval img to all_png folder"
    cp -r "${SCRIPT_DIR}/downloaded/mnist_png/valid/"* "${SCRIPT_DIR}/downloaded/mnist_png/all_png"
fi
