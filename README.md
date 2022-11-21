# MARLClassification

[Multi-Agent Image Classification via Reinforcement Learning](https://arxiv.org/abs/1905.04835)

## TODO
- train on AID
- docstring

## Results

Trained on MNIST (see `resources/trained_models/mnist`) :

|          | Loss            |  Train   |   Eval    |
|----------|-----------------|:--------:|:---------:|
| Epoch 0 | 1.5161 | 41%, 42% | 44%, 44% |
| Epoch 20 | 0.5385 | 80%, 80% | 80%, 79% |
| Epoch 39 | 0.5218 | 81%, 81% | 82%, 81% |

Train on image dataset NWPU-RESISC45 (see `resources/trained_models/resisc45`) :

|          | Loss |   Train   |   Eval    |
|----------| --- |:---------:|:---------:|
|          | | prec, rec | prec, rec |
| Epoch 0  | 2.6236 | 21%, 26%  | 25%, 26%  |
| Epoch 20 | 1.3262 | 57%, 57%  | 58%, 57%  |
| Epoch 49 | 0.9263 | 68%, 68%  | 66%, 65%  |

## Installation
```bash
$ cd /path/to/MARLClassification
$ # create and activate your virtual env
$ python -m venv venv
$ ./venv/bin/activate
$ # install requirements
$ pip install -r requirements.txt
$ # download datasets using sh scripts in resources folder, ex : MNIST
$ ./resources/download_mnist.sh
```

You may download datasets with bash scripts in `res` folder.
## Usage
To run training :
```bash
$ cd /path/to/MARLClassification
$ # train on MNIST
$ python -m marl_classification -a 3 --step 5 --cuda --run-id train_mnist train --action [[1,0],[-1,0],[0,1],[0,-1]] --img-size 28 --nb-class 10 -d 2 --f 6 --ft-extr mnist --nb 64 --na 64 --nm 16 --nd 8 --nlb 96 --nla 96 --batch-size 32 --lr 1e-3 --nb-epoch 40 -o ./out/mnist_actor_critic
$ # train on NWPU-RESISC45
$ python -m marl_classification -a 16 --step 16 --cuda --run-id train_resisc45 train --action [[1,0],[-1,0],[0,1],[0,-1]] --ft-extr resisc45 --batch-size 8 --nb-class 45 --img-size 256 -d 2 --nb 256 --na 256 --nd 16 --f 12 --nm 64 --nlb 384 --nla 384 --nb-epoch 50 --lr 1e-4 -o ./out/resisc45_actor_critic
$ # train on AID
$ python -m marl_classification -a 8 --step 16 --cuda --run-id train_aid train --action [[3,0],[-3,0],[0,3],[0,-3]] --ft-extr aid --batch-size 8 --nb-class 30 --img-size 600 -d 2 --nb 256 --na 256 --nd 16 --f 32 --nm 64 --nlb 320 --nla 320 --nb-epoch 50 --lr 1e-4 -o ./out/aid_actor_critic
```

## Reference

[1]: https://arxiv.org/abs/1905.04835, _Hossein K. Mousavi, Mohammadreza Nazari, Martin Takáč, Nader Motee_ - 2019
