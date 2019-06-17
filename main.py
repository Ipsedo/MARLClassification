from environment.observation import obs_MNIST
from environment.transition import trans_MNIST
from environment.agent import Agent
from environment.core import step
import torch as th
from torch.nn import Softmax, MSELoss
from torchviz import make_dot, make_dot_from_trace


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

    nb_class = 10
    img_size = 28
    n = 16
    f = 5
    n_m = 8
    d = 2
    action_size = 2

    a1 = Agent(ag, th.tensor([1, 3]), n, f, n_m, d, img_size, action_size, nb_class, obs_MNIST, trans_MNIST)
    a2 = Agent(ag, th.tensor([5, 3]), n, f, n_m, d, img_size, action_size, nb_class, obs_MNIST, trans_MNIST)
    a3 = Agent(ag, th.tensor([10, 8]), n, f, n_m, d, img_size, action_size, nb_class, obs_MNIST, trans_MNIST)

    ag.append(a1)
    ag.append(a2)
    ag.append(a3)

    img = th.rand(28, 28)

    for a in ag:
        a.step(img)
    for a in ag:
        a.step_finished()

    print(a1.get_t_msg())


def test_core_step():
    ag = []

    nb_class = 10
    img_size = 28
    n = 16
    f = 5
    n_m = 8
    d = 2
    action_size = 2

    a1 = Agent(ag, th.tensor([1, 3]), n, f, n_m, d, img_size, action_size, nb_class, obs_MNIST, trans_MNIST)
    a2 = Agent(ag, th.tensor([5, 3]), n, f, n_m, d, img_size, action_size, nb_class, obs_MNIST, trans_MNIST)
    a3 = Agent(ag, th.tensor([10, 8]), n, f, n_m, d, img_size, action_size, nb_class, obs_MNIST, trans_MNIST)

    ag.append(a1)
    ag.append(a2)
    ag.append(a3)

    img = th.rand(28, 28)
    c = th.zeros(10)
    c[5] = 1

    sm = Softmax(dim=0)

    mse = MSELoss()

    params = []
    for a in ag:
        for n in a.get_networks():
            params += n.parameters()
    optim = th.optim.SGD(params, lr=1e-4)

    nb_epoch = 10

    for e in range(nb_epoch):
        optim.zero_grad()
        pred, proba = step(ag, img, 10, sm)
        r = mse(pred, c)
        Nr = 1
        loss = (th.log(proba) * r.detach() + r) / Nr

        loss.backward()
        optim.step()

    for a in ag:
        for n in a.get_networks():
            if hasattr(n, 'seq_lin'):
                if n.seq_lin[0].weight.grad is None:
                    print(n)
            elif hasattr(n, "lstm"):
                if n.lstm.weight_hh_l0.grad is None:
                    print(n)


if __name__ == "__main__":
    #test_MNIST_transition()
    #test_MNIST_obs()
    #test_agent_step()
    test_core_step()
