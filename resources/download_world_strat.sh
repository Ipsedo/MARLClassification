#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -f "${SCRIPT_DIR}/downloaded/hr_dataset.tar.gz" ]]; then
  wget https://zenodo.org/record/6810792/files/hr_dataset.tar.gz -P "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/WroldStrat" ]]; then
  mkdir "${SCRIPT_DIR}/downloaded/WroldStrat"
  wget https://zenodo.org/record/6810792/files/metadata.csv -P "${SCRIPT_DIR}/downloaded"
  tar -xvzf "${SCRIPT_DIR}/downloaded/hr_dataset.tar.gz" -C "${SCRIPT_DIR}/downloaded/WroldStrat"
fi