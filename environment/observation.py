import torch as th


def obs_img(img: th.Tensor, pos: th.Tensor, f: int) -> th.Tensor:
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

    assert len(img.size()) == 4,\
        f"images.size() == (N, C, W, H), actual = {img.size()}"
    assert len(pos.size()) == 3,\
        f"Position size must be (nb_agents, batch_size, data_dim), actual = {pos.size()}"

    nb_a, b_pos, d = pos.size()
    b_img, c, h, w = img.size()

    assert (pos[:, :, 0] + f < w).all() and (pos[:, :, 0] >= 0).all(),\
        "Unbound obervation in first dim (pos = {}, f = {}, img_size = {}!".format(pos[:, 0], f, w)
    assert (pos[:, :, 1] + f < h).all() and (pos[:, :, 1] >= 0).all(),\
        "Unbound obervation in snd dim (pos = {}, f = {}, img_size = {}!".format(pos[:, 1], f, h)

    # pos.size == (nb_ag, batch_size, 2)
    pos_min = pos
    pos_max = pos_min + f

    values_x = th.arange(0, w, device=pos.device)
    mask_x = (pos_min[:, :, 0, None] <= values_x.view(1, 1, w)) &\
             (values_x.view(1, 1, w) < pos_max[:, :, 0, None])

    values_y = th.arange(0, h, device=pos.device)
    mask_y = (pos_min[:, :, 1, None] <= values_y.view(1, 1, h)) &\
             (values_y.view(1, 1, h) < pos_max[:, :, 1, None])

    mask = mask_x.unsqueeze(-2) & mask_y.unsqueeze(-1)

    return img.unsqueeze(0).masked_select(mask.unsqueeze(-3)).view(nb_a, b_img, c, f, f)
