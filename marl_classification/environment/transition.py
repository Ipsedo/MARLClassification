import operator as op
from functools import reduce
from typing import List, cast

import torch as th


def trans_generic(
    pos: th.Tensor,
    a_t_next: th.Tensor,
    f: int,
    img_size: List[int],
) -> th.Tensor:
    new_pos = pos.clone()
    dim = new_pos.size(-1)

    idxs = []
    for d in range(dim):
        curr_idx = (new_pos[:, :, d] + a_t_next[:, :, d] >= 0) * (
            new_pos[:, :, d] + a_t_next[:, :, d] + f < img_size[d]
        )
        idxs.append(curr_idx)

    idx = reduce(op.mul, idxs)

    idx = idx.unsqueeze(2).to(th.float)

    return cast(th.Tensor, idx * (new_pos + a_t_next) + (1 - idx) * new_pos)
