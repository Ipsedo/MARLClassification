import torch as th


def obs_MNIST(img: th.Tensor, pos: th.Tensor, f: int) -> th.Tensor:
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

    assert len(img.size()) == 3, "Need 3-D input"
    assert len(pos.size()) == 3, "Position size must be (nb_agents, batch_size, data_dim)"
    assert (pos[:, :, 0] + f < img.size(1)).all() and (pos[:, :, 0] >= 0).all(),\
        "Unbound obervation in first dim (pos = {}, f = {}, img_size = {}!".format(pos[:, 0], f, img.size(1))
    assert (pos[:, :, 1] + f < img.size(2)).all() and (pos[:, :, 1] >= 0).all(),\
        "Unbound obervation in snd dim (pos = {}, f = {}, img_size = {}!".format(pos[:, 1], f, img.size(2))

    """res = []
    for i in range(img.size(0)):
        res.append(img[i, pos[i, 0]:pos[i, 0] + f, pos[i, 1]:pos[i, 1] + f])
    return th.stack(res)"""

    # pos.size == (nb_ag, batch_size, 2)
    pos_min = pos
    pos_max = pos_min + f

    values_x = th.arange(0, img.size(1), device=pos.device).view(1, 1, img.size(1)).repeat(pos.size(0), img.size(0), 1)
    mask_x = (pos_min[:, :, 0, None] <= values_x) & (values_x < pos_max[:, :, 0, None])

    values_y = th.arange(0, img.size(1), device=pos.device).view(1, 1, img.size(1)).repeat(pos.size(0), img.size(0), 1)
    mask_y = (pos_min[:, :, 1, None] <= values_y) & (values_y < pos_max[:, :, 1, None])

    mask_x = mask_x.view(pos.size(0), img.size(0), img.size(1), 1).repeat(1, 1, 1, img.size(1))
    mask_y = mask_y.view(pos.size(0), img.size(0), img.size(1), 1).repeat(1, 1, 1, img.size(1)).permute(0, 1, 3, 2)

    mask = mask_x & mask_y

    return img.unsqueeze(0).repeat(pos.size(0), 1, 1, 1).masked_select(mask).view(pos.size(0), img.size(0), f, f)
