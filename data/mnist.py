import torch as th
import pickle
import gzip
from os.path import exists


def load_mnist():
    file = './res/downloaded/mnist.pkl.gz'

    assert exists(file), "You must download mnist dataset via download_mnist.sh script !"

    f = gzip.open(file, 'rb')

    u = pickle._Unpickler(f)
    u.encoding = 'latin1'

    train_set, valid_set, test_set = u.load()

    f.close()

    x_train = th.from_numpy(train_set[0]).view(-1, 28, 28)
    y_train = th.from_numpy(train_set[1])

    x_valid = th.from_numpy(valid_set[0]).view(-1, 28, 28)
    y_valid = th.from_numpy(valid_set[1])

    x_test = th.from_numpy(test_set[0]).view(-1, 28, 28)
    y_test = th.from_numpy(test_set[1])

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

