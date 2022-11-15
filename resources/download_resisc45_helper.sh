#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -f "${SCRIPT_DIR}/downloaded/NWPU-RESISC45.rar" ]]; then
  echo "NWPU-RESISC45.rar not found in /path/to/MARLClassification/res/downloaded"
  echo "Download NWPU-RESISC45.rar on the opened web page"
  firefox https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs

  echo "Then place it at ${SCRIPT_DIR}/downloaded folder and re-run this script"
else
  echo "Assume you have correctly downloaded and copied NWPU-RESISC45.rar to /path/to/MARLClassification/res/downloaded folder"

  if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    echo "Create downloaded dir"
    mkdir "${SCRIPT_DIR}/downloaded"
  fi

  echo "Extract NWPU-RESISC45.rar"

  unrar x "${SCRIPT_DIR}/downloaded/NWPU-RESISC45.rar" "${SCRIPT_DIR}/downloaded"
fi

echo "Finished, RESISC-45 data ready."
