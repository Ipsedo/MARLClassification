#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

# Download ModelNet10 zip
if ! [[ -f "${SCRIPT_DIR}/downloaded/ModelNet10.zip" ]]; then
    echo "Download ModelNet10 from http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip -P "${SCRIPT_DIR}/downloaded"
fi

# Extract ModelNet10
if ! [[ -d "${SCRIPT_DIR}/downloaded/ModelNet10" ]]; then
    echo "Extract ModelNet10.zip"
    unzip "${SCRIPT_DIR}/downloaded/ModelNet10.zip" -d "${SCRIPT_DIR}/downloaded"
fi