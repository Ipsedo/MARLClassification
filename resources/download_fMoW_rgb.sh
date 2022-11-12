#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/fMoW_dataset" ]]; then
  git clone https://github.com/fMoW/dataset.git "${SCRIPT_DIR}/downloaded/fMoW_dataset"
fi

#transmission-cli -w "${SCRIPT_DIR}/downloaded/fMoW_dataset" "${SCRIPT_DIR}/downloaded/fMoW_dataset/fMoW-rgb_trainval_v1.0.0.torrent"
aws s3 ls s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/