#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -d "${SCRIPT_DIR}/downloaded/fMoW_dataset" ]]; then
  git clone https://github.com/fMoW/dataset.git "${SCRIPT_DIR}/downloaded/fMoW_dataset"
fi

REGION="eu-west-1"
AWS_URL="s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb"

if ! [[ -d "${SCRIPT_DIR}/downloaded/fMoW_dataset/dataset_rgb" ]]; then
  mkdir "${SCRIPT_DIR}/downloaded/fMoW_dataset/dataset_rgb"

  for cat in $( aws s3 ls --region="${REGION}" "${AWS_URL}/train/" | perl -pe 's/.+ ([^ ]+)$/$1/' ); do
    echo "${cat} will be downloaded"
    mkdir "${SCRIPT_DIR}/downloaded/fMoW_dataset/dataset_rgb/${cat}"

    for folder in $( aws s3 ls --region="${REGION}" "${AWS_URL}/train/${cat}" | perl -pe 's/.+ ([^ ]+)$/$1/' ); do
      first_jpg_file=$( aws s3 ls --region="${REGION}" "${AWS_URL}/train/${cat}${folder}" | grep _rgb.jpg | head -n 1 | perl -pe 's/.+ ([^ ]+)$/$1/' )
      first_jpg_json_file=$( aws s3 ls --region="${REGION}" "${AWS_URL}/train/${cat}${folder}" | grep _rgb.json | head -n 1 | perl -pe 's/.+ ([^ ]+)$/$1/' )

      aws s3 cp --region="${REGION}" "${AWS_URL}/train/${cat}${folder}${first_jpg_file}" "${SCRIPT_DIR}/downloaded/fMoW_dataset/dataset_rgb/${cat}"
      aws s3 cp --region="${REGION}" "${AWS_URL}/train/${cat}${folder}${first_jpg_json_file}" "${SCRIPT_DIR}/downloaded/fMoW_dataset/dataset_rgb/${cat}"
    done
  done
fi