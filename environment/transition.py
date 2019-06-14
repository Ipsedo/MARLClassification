import torch as th


def trans_MNIST(pos: th.Tensor, a_t_p_one: th.Tensor, f: int, img_size: int) -> th.Tensor:
    """

    :param pos:
    :param a_t_p_one: Tensor of size = (4,)
        one_hot_vector [up, down, left, right]
    :param f:
    :param img_size:
    :return:
    """

    dir = th.zeros(2).to(th.long)
    if a_t_p_one[0]:
        # UP
        dir[1] = -1
        pass
    elif a_t_p_one[1]:
        # DOWN
        dir[1] = 1
        pass
    elif a_t_p_one[2]:
        # LEFT
        dir[0] = -1
        pass
    else:
        # RIGHT
        dir[0] = 1
        pass

    new_pos = pos + dir

    if new_pos[0] < 0 or new_pos[0] >= img_size or new_pos[1] < 0 or new_pos[1] >= img_size:
        return pos

    return pos + dir
