import torch as th

from functools import reduce
import operator as op


def obs_2d_img(img: th.Tensor, pos: th.Tensor, f: int) -> th.Tensor:
    """
    TODO

    :param img:
    :type img:
    :param pos:
    :type pos:
    :param f:
    :type f:
    :return:
    :rtype:
    """

    nb_a, b_pos, d = pos.size()
    b_img, c, h, w = img.size()

    # pos.size == (nb_ag, batch_size, 2)
    pos_min = pos
    pos_max = pos_min + f

    values_x = th.arange(0, w, device=pos.device)
    mask_x = (pos_min[:, :, 0, None] <= values_x.view(1, 1, w)) & \
             (values_x.view(1, 1, w) < pos_max[:, :, 0, None])

    values_y = th.arange(0, h, device=pos.device)
    mask_y = (pos_min[:, :, 1, None] <= values_y.view(1, 1, h)) & \
             (values_y.view(1, 1, h) < pos_max[:, :, 1, None])

    mask = mask_x.unsqueeze(-2) & mask_y.unsqueeze(-1)

    return img.unsqueeze(0).masked_select(mask.unsqueeze(-3)) \
        .view(nb_a, b_img, c, f, f)


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
