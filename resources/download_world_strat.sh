#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -f "${SCRIPT_DIR}/downloaded/hr_dataset.tar.gz" ]]; then
  wget https://zenodo.org/record/6810792/files/hr_dataset.tar.gz
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/WroldStrat" ]]; then
  mkdir "${SCRIPT_DIR}/downloaded/WroldStrat"
  tar -xvzf "${SCRIPT_DIR}/downloaded/hr_dataset.tar.gz" -C "${SCRIPT_DIR}/downloaded/WroldStrat"
fi