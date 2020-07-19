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
    assert (pos[:, 0] + f < img.size(1)).any() and (pos[:, 0] >= 0).any(),\
        "Unbound obervation in first dim (pos = {}, f = {}, img_size = {}!".format(pos[:, 0], f, img.size(1))
    assert (pos[:, 1] + f < img.size(2)).any() and (pos[:, 1] >= 0).any(),\
        "Unbound obervation in snd dim (pos = {}, f = {}, img_size = {}!".format(pos[:, 1], f, img.size(2))

    res = []
    for i in range(img.size(0)):
        res.append(img[i, pos[i, 0]:pos[i, 0] + f, pos[i, 1]:pos[i, 1] + f])
    return th.stack(res)
