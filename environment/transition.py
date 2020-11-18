import torch as th

from functools import reduce
import operator as op

from typing import Tuple, List


def trans_2d_img(pos: th.Tensor, a_t_next: th.Tensor,
                 f: int, img_size: Tuple[int]) -> th.Tensor:
    """

    Check if MNIST transition is correct.
    Return next position if `a_t_next` is possible, current position otherwise.

    :param pos:
    :param a_t_next: Tensor of size = (2,)
    :param f:
    :param img_size:
    :return:
    """

    new_pos = pos.clone()

    idx = (new_pos[:, :, 0] + a_t_next[:, :, 0] >= 0) * \
          (new_pos[:, :, 0] + a_t_next[:, :, 0] + f < img_size[-1]) * \
          (new_pos[:, :, 1] + a_t_next[:, :, 1] >= 0) * \
          (new_pos[:, :, 1] + a_t_next[:, :, 1] + f < img_size[-2])

    idx = idx.unsqueeze(2).to(th.float)

    return idx * (new_pos + a_t_next) + (1 - idx) * new_pos


def trans_generic(
        pos: th.Tensor, a_t_next: th.Tensor,
        f: int, img_size: List[int]
) -> th.Tensor:
    new_pos = pos.clone()
    dim = new_pos.size(-1)

    idxs = []
    for d in range(dim):
        idx = (new_pos[:, :, d] + a_t_next[:, :, d] >= 0) * \
              (new_pos[:, :, d] + a_t_next[:, :, d] + f < img_size[d])
        idxs.append(idx)

    idx = reduce(op.mul, idxs)

    idx = idx.unsqueeze(2).to(th.float)

    return idx * (new_pos + a_t_next) + (1 - idx) * new_pos
