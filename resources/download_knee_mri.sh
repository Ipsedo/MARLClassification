#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/knee_mri" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded/knee_mri"
fi

if ! [[ -f "${SCRIPT_DIR}/downloaded/knee_mri/metadata.csv" ]]; then
    wget http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/metadata.csv -P "${SCRIPT_DIR}/downloaded/knee_mri"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/knee_mri/extracted" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded/knee_mri/extracted"
fi

for i in {01..10}; do
    if ! [[ -f "${SCRIPT_DIR}/downloaded/knee_mri/vol${i}.7z" ]]; then
        wget "http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/volumetric_data/vol${i}.7z" -P "${SCRIPT_DIR}/downloaded/knee_mri"
        7z x "${SCRIPT_DIR}/downloaded/knee_mri/vol${i}.7z" -o"${SCRIPT_DIR}/downloaded/knee_mri/extracted"
    fi
done