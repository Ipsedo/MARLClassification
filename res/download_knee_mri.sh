#!/usr/bin/env bash

if ! [[ -d "./downloaded" ]]; then
    mkdir downloaded
fi
cd downloaded

if ! [[ -d "./knee_mri" ]]; then
    mkdir knee_mri
fi
cd knee_mri

if ! [[ -f "./metadata.csv" ]]; then
    wget http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/metadata.csv
fi

if ! [[ -d "./extracted" ]]; then
    mkdir extracted
fi

for i in {01..10}; do
    if ! [[ -f "./vol${i}.7z" ]]; then
        wget http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/volumetric_data/vol${i}.7z
    fi
    7z x vol${i}.7z -o./extracted
done


cd ../..