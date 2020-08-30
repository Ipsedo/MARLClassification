# MARLClassification

from this [article](https://arxiv.org/abs/1905.04835)

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
```bash
$ # train on NWPU-RESISC45
$ python main.py -a 10 --step 10 --cuda train --nb-action 4 --ft-extr resisc45 --batch-size 6 --nb-class 45 --img-size 256 -d 2 --n 1536 --nd 8 --f 10 --nm 256 --nl 2048 --nb-epoch 30 --nr 3 --learning-rate 3e-5 --eps 5e-2 -o ./out_train_resisc45_final
```

## Requirements
torch, torchvision, torchnet, numpy, tqdm, matplotlib