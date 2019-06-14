from environment.observation import obs_MNIST
from environment.transition import trans_MNIST
import torch as th


def test_MNIST_transition():
    img = th.rand(28, 28)

    a_1 = th.tensor([1, 0, 0, 0])
    a_2 = th.tensor([0, 1, 0, 0])
    a_3 = th.tensor([0, 0, 1, 0])
    a_4 = th.tensor([0, 0, 0, 1])

    print("First test :")
    pos_1 = th.tensor([0, 0])
    print(trans_MNIST(pos_1, a_1, 5, 28))
    print(trans_MNIST(pos_1, a_2, 5, 28))
    print(trans_MNIST(pos_1, a_3, 5, 28))
    print(trans_MNIST(pos_1, a_4, 5, 28))
    print()

    print("Snd test")
    pos_2 = th.tensor([27, 27])
    print(trans_MNIST(pos_2, a_1, 5, 28))
    print(trans_MNIST(pos_2, a_2, 5, 28))
    print(trans_MNIST(pos_2, a_3, 5, 28))
    print(trans_MNIST(pos_2, a_4, 5, 28))


def test_MNIST_obs():
    img = th.arange(0, 28 * 28).view(28, 28)

    print(img)
    print()

    pos = th.tensor([2, 2])

    print(obs_MNIST(img, pos, 3))
    print()

    try:
        pos_fail = th.tensor([-1, -1])
        print(obs_MNIST(img, pos_fail, 3))
    except Exception as e:
        print(e)

    try:
        pos_fail = th.tensor([24, 26])
        print(obs_MNIST(img, pos_fail, 3))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    #test_MNIST_transition()
    test_MNIST_obs()
