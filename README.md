# MARLClassification

PyTorch implementation of [Multi-Agent Image Classification via Reinforcement Learning](https://arxiv.org/abs/1905.04835) paper.

## TODO

- train on WorldStrat
- docstring

## Installation

You need to have `python` (at least version 3.14) and `uv` executables in your `PATH`.

First you can clone the project :

```bash
git clone https://github.com/Ipsedo/MARLClassification.git
```

Then install dependencies :

```bash
cd /path/to/MARLClassification
# install dependencies
uv sync
# download datasets using sh scripts in resources folder, ex : MNIST
./resources/download_mnist.sh
```

You may download datasets with bash scripts in `resources` folder.

## Usage

To run training :

```bash
cd /path/to/MARLClassification
# train on MNIST
python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action [[1,0],[-1,0],[0,1],[0,-1]] --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nmo 24 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 40 -o ./out/mnist_actor_critic
# train on NWPU-RESISC45
python -m marl_classification -a 16 --step 16 --cuda --run-id train_resisc45 train --action [[1,0],[-1,0],[0,1],[0,-1]] --ft-extr resisc45 --batch-size 8 --nb-class 45 --img-size 256 -d 2 --nb 256 --na 256 --nd 16 --f 12 --nm 64 --nmo 96 --nlb 384 --nla 384 --nb-epoch 50 --lr 1e-4 -o ./out/resisc45_actor_critic
# train on AID
python -m marl_classification -a 16 --step 16 --cuda --run-id train_aid train --action [[3,0],[-3,0],[0,3],[0,-3]] --ft-extr aid --batch-size 8 --nb-class 30 --img-size 600 -d 2 --nb 256 --na 256 --nd 16 --f 24 --nm 64 --nmo 96 --nlb 320 --nla 320 --nb-epoch 50 --lr 1e-4 -o ./out/aid_actor_critic
```

## Results

Training on MNIST (see `resources/trained_models/mnist`) :

```
Epoch 50
--------
[Train]
precision : 81.6%
recall    : 81.3%
--------
[Eval]
precision : 82.4%
recall    : 81.2%
```

Training on image dataset NWPU-RESISC45 (see `resources/trained_models/resisc45`) :

```
Epoch 50
--------
[Train]
precision : 71.4%
recall    : 71.8%
--------
[Eval]
precision : 68.7%
recall    : 67.8%
```

Training on image dataset AID (see `resources/trained_models/aid`) :

```
Epoch 50
--------
[Train]
precision : 80.9%
recall    : 80.6%
--------
[Eval]
precision : 73.7%
recall    : 72.5%
```


## Reference

[1]: https://arxiv.org/abs/1905.04835, _Hossein K. Mousavi, Mohammadreza Nazari, Martin Takáč, Nader Motee_ - 2019
