#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -f "${SCRIPT_DIR}/downloaded/AID.zip" ]]; then
  echo "AID.zip not found in /path/to/MARLClassification/res/downloaded"
  echo "Download AID.zip on the opened web page"
  firefox https://captain-whu.github.io/AID/

  echo "Then place it at ${SCRIPT_DIR}/downloaded folder and re-run this script"
else
  unzip "${SCRIPT_DIR}/downloaded/AID.zip" -d "${SCRIPT_DIR}/downloaded"
fi