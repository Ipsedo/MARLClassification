import torch as th
import torch.nn as nn


def step(agents: list, img: th.Tensor, max_it: int, softmax: nn.Softmax, cuda: bool, random_walk: bool, nb_class: int):
    for a in agents:
        a.new_img(img.size(0))

    for t in range(max_it):
        for a in agents:
            a.step(img, random_walk)
        for a in agents:
            a.step_finished()

    q = th.zeros(img.size(0), nb_class)
    probas = th.zeros(len(agents), img.size(0))

    if cuda:
        q = q.cuda()
        probas = probas.cuda()

    for i, a in enumerate(agents):
        pred, proba = a.predict()
        probas[i, :] = proba
        q += pred

    q = q / len(agents)

    return softmax(q), probas


def detailled_step(agents: list, img: th.Tensor, max_it: int, softmax: nn.Softmax, cuda: bool, nb_class: int):
    for a in agents:
        a.new_img(img.size(0))

    pos = [[] for _ in range(len(agents))]

    for _ in range(max_it):
        for i, a in enumerate(agents):
            a.step(img, False)
            pos[i].append(a.p[0])
        for a in agents:
            a.step_finished()

    q = th.zeros(img.size(0), nb_class)

    if cuda:
        q = q.cuda()

    for a in agents:
        pred, _ = a.predict()
        q += pred

    q = q / len(agents)

    return softmax(q), pos
