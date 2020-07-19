import torch as th
import torch.nn as nn

from .agent import Agent
from typing import List, Tuple


def step(agents: List[Agent], img: th.Tensor, max_it: int, softmax: nn.Softmax,
         cuda: bool, random_walk: bool, nb_class: int) -> Tuple[th.Tensor, th.Tensor]:
    """
    TODO

    :param agents:
    :type agents:
    :param img:
    :type img:
    :param max_it:
    :type max_it:
    :param softmax:
    :type softmax:
    :param cuda:
    :type cuda:
    :param random_walk:
    :type random_walk:
    :param nb_class:
    :type nb_class:
    :return:
    :rtype:
    """

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


def detailled_step(agents: List[Agent], img: th.Tensor, max_it: int, softmax: nn.Softmax,
                   cuda: bool, nb_class: int) -> Tuple[th.Tensor, List[List[th.Tensor]]]:
    """
    TODO

    :param agents:
    :type agents:
    :param img:
    :type img:
    :param max_it:
    :type max_it:
    :param softmax:
    :type softmax:
    :param cuda:
    :type cuda:
    :param nb_class:
    :type nb_class:
    :return:
    :rtype:
    """

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
