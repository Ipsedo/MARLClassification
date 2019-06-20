import torch as th


def trans_MNIST(pos: th.Tensor, a_t_next: th.Tensor, f: int, img_size: int) -> th.Tensor:
    """

    :param pos:
    :param a_t_next: Tensor of size = (2,)
    :param f:
    :param img_size:
    :return:
    """

    new_pos = pos + a_t_next

    if (new_pos[:, 0] < 0).any() or\
            (new_pos[:, 0] + f >= img_size).any() or\
            (new_pos[:, 1] < 0).any() or\
            (new_pos[:, 1] + f >= img_size).any():
        return pos

    return new_pos
