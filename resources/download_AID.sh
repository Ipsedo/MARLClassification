#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -f "${SCRIPT_DIR}/downloaded/aid-scene-classification-datasets.zip" ]]; then
  echo "aid-scene-classification-datasets.zip not found in ${SCRIPT_DIR}/downloaded"
  echo "Download aid-scene-classification-datasets.zip with Kaggle CLI"

  kaggle datasets download jiayuanchengala/aid-scene-classification-datasets -p "${SCRIPT_DIR}/downloaded/"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/AID/" ]]; then
  echo "AID folder not found in ${SCRIPT_DIR}/downloaded"
  echo "Unzip aid-scene-classification-datasets.zip"

  unzip "${SCRIPT_DIR}/downloaded/aid-scene-classification-datasets.zip" -d "${SCRIPT_DIR}/downloaded"
fi