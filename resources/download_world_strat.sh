#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -f "${SCRIPT_DIR}/downloaded/hr_dataset.tar.gz" ]]; then
  wget https://zenodo.org/record/6810792/files/hr_dataset.tar.gz -P "${SCRIPT_DIR}/downloaded"
fi

DEST_DIR="${SCRIPT_DIR}/downloaded/WorldStrat"

if ! [[ -d "${DEST_DIR}" ]]; then
  mkdir "${DEST_DIR}"
  wget https://zenodo.org/record/6810792/files/metadata.csv -P "${DEST_DIR}"
  tar -xvzf "${SCRIPT_DIR}/downloaded/hr_dataset.tar.gz" -C "${DEST_DIR}"
fi