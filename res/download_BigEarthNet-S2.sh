#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -f "${SCRIPT_DIR}/downloaded/BigEarthNet-S2-v1.0.tar.gz" ]]; then
  wget --no-check-certificate https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz -P "${SCRIPT_DIR}/downloaded"
fi