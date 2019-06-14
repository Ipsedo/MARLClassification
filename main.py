from environment.observation import obs_MNIST
from environment.transition import trans_MNIST
from environment.agent import Agent
import torch as th


def test_MNIST_transition():
    img = th.rand(28, 28)

    a_1 = th.tensor([1, 0, 0, 0])
    a_2 = th.tensor([0, 1, 0, 0])
    a_3 = th.tensor([0, 0, 1, 0])
    a_4 = th.tensor([0, 0, 0, 1])

    print("First test :")
    pos_1 = th.tensor([0, 0])
    print(trans_MNIST(pos_1, a_1, 5))
    print(trans_MNIST(pos_1, a_2, 5))
    print(trans_MNIST(pos_1, a_3, 5))
    print(trans_MNIST(pos_1, a_4, 5))
    print()

    print("Snd test")
    pos_2 = th.tensor([27, 27])
    print(trans_MNIST(pos_2, a_1, 5))
    print(trans_MNIST(pos_2, a_2, 5))
    print(trans_MNIST(pos_2, a_3, 5))
    print(trans_MNIST(pos_2, a_4, 5))


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


def test_agent_step():
    ag = []

    a1 = Agent(ag, th.tensor([1, 3]), 16, 5, 8, 2, 28, obs_MNIST, trans_MNIST)
    a2 = Agent(ag, th.tensor([1, 3]), 16, 5, 8, 2, 28, obs_MNIST, trans_MNIST)
    a3 = Agent(ag, th.tensor([1, 3]), 16, 5, 8, 2, 28, obs_MNIST, trans_MNIST)

    ag.append(a1)
    ag.append(a2)
    ag.append(a3)

    img = th.rand(28, 28)

    for a in ag:
        a.step(img)
    for a in ag:
        a.step_finished()

    print(a1.get_t_msg())


if __name__ == "__main__":
    #test_MNIST_transition()
    #test_MNIST_obs()
    test_agent_step()
