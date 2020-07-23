#!/bin/bash

if ! [[ -d "./downloaded" ]]; then
    mkdir downloaded
fi
cd downloaded

# Download mnist
if ! [[ -f "./mnist.pkl.gz" ]]; then
    wget http://deeplearning.net/data/mnist/mnist.pkl.gz
fi

cd ..