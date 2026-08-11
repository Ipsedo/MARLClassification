#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi


if ! [[ -f "${SCRIPT_DIR}/downloaded/kneemridataset.zip" ]]; then
    kaggle datasets download sohaibanwaar1203/kneemridataset -p "${SCRIPT_DIR}/downloaded/"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/kneemridataset" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded/kneemridataset"
    unzip "${SCRIPT_DIR}/downloaded/kneemridataset.zip" -d "${SCRIPT_DIR}/downloaded/kneemridataset"
fi