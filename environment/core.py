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
