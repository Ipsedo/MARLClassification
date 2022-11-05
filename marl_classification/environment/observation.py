import operator as op
from functools import reduce

import torch as th


def obs_generic(x: th.Tensor, pos: th.Tensor, f: int) -> th.Tensor:
    x_sizes = x.size()
    b_img, c = x_sizes[0], x_sizes[1]
    sizes = [s for s in x_sizes[2:]]

    nb_a, _, _ = pos.size()

    pos_min = pos
    pos_max = pos_min + f

    masks = []

    for d, s in enumerate(sizes):
        values = th.arange(0, s, device=pos.device)

        mask = (pos_min[:, :, d, None] <= values.view(1, 1, s)) & \
               (values.view(1, 1, s) < pos_max[:, :, d, None])

        for n_unsq in range(len(sizes) - 1):
            mask = mask.unsqueeze(-2) if n_unsq < d else mask.unsqueeze(-1)

        masks.append(mask)
    mask = reduce(op.and_, masks)
    mask = mask.unsqueeze(2)

    return x.unsqueeze(0).masked_select(mask) \
        .view(nb_a, b_img, c, *[f for _ in range(len(sizes))])
