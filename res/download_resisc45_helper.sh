#!/usr/bin/env bash

if ! [[ -f "./downloaded/NWPU-RESISC45.rar" ]]; then
  echo "NWPU-RESISC45.rar not found in /path/to/MARLClassification/res/downloaded"
  echo "Download NWPU-RESISC45.rar on the opened web page"
  firefox https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs

  echo "Then place it at /path/to/MARLClassification/res/downloaded root"
  echo "Then re-run this script"
else
  echo "Assume you have correctly downloaded and copied NWPU-RESISC45.rar to /path/to/MARLClassification/res/downloaded root"

  if ! [[ -d "./downloaded" ]]; then
    echo "Create downloaded dir"
    mkdir downloaded
  fi

  echo "Go to download dir"
  cd downloaded
  echo "Extract NWPU-RESISC45.rar"
  unrar x -o- ./NWPU-RESISC45.rar

  echo "Go back in parent folder"
  cd ..
fi

echo "Finished, RESISC-45 data ready."
