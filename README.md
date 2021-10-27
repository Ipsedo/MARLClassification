# MARLClassification

[Multi-Agent Image Classification via Reinforcement Learning](https://arxiv.org/abs/1905.04835)

_Hossein K. Mousavi, Mohammadreza Nazari, Martin Takáč, Nader Motee, 2019_

## Description
Results reproduction of the above article : 98% on MNIST.

Extend to other image data NWPU-RESISC45 :

| | Loss | Train | Eval |
| --- | --- | :---: | :---: |
| | | prec, rec | prec, rec |
| Epoch 0 | -0.2638 | 6%, 9% | 12%, 14% |
| Epoch 10 | -0.1813 | 55%, 56% | 49%, 48% |
| Epoch 17 | -0.1613 | 63%, 64% | 54%, 53% |


## Usage
You may download datasets with bash scripts in `res` folder.

```bash
$ # train on MNIST
$ python main.py -a 3 --step 5 --cuda --run-id train_mnist train --action [[1,0],[-1,0],[0,1],[0,-1]] --img-size 28 --nb-class 10 -d 2 --f 5 --ft-extr mnist --nb 128 --na 128 --nm 32 --nd 8 --nlb 160 --nla 160 --batch-size 32 --lr 1e-3 --nb-epoch 40 --nr 1 --eps 0. --eps-dec 1. -o ./out/mnist
$ # train on NWPU-RESISC45
$ python main.py -a 20 --step 15 --cuda --run-id train_resisc45 train --action [[1,0],[-1,0],[0,1],[0,-1]] --ft-extr resisc45 --batch-size 8 --nb-class 45 --img-size 256 -d 2 --nb 1536 --na 1536 --nd 8 --f 10 --nm 256 --nlb 2048 --nla 2048 --nb-epoch 50 --nr 1 --learning-rate 2e-5 --eps 0. --eps-dec 1. -o ./out/resisc45
$ # train on Knee MRI
$ python main.py -a 20 --step 15 --cuda --run-id test_train_knee train --action [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] --ft-extr kneemri --batch-size 3 --nb-class 3 --img-size 320 -d 3 --nb 1536 --na 1536 --nd 16 --f 10 --nm 256 --nlb 2048 --nla 2048 --nb-epoch 50 --nr 1 --learning-rate 2e-5 --eps 0. --eps-dec 1. -o ./out/knee_test
```

## Requirements
torch, torchvision, torchnet, numpy, tqdm, matplotlib, pandas, mlflow