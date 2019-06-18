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

    dir = th.zeros(pos.size(0), 2)

    if pos.is_cuda:
        dir = dir.cuda()

    for i in range(pos.size(0)):
        if a_t_p_one[i, 0]:
            # UP
            dir[i, 1] = -1
        elif a_t_p_one[i, 1]:
            # DOWN
            dir[i, 1] = 1
        elif a_t_p_one[i, 2]:
            # LEFT
            dir[i, 0] = -1
        else:
            # RIGHT
            dir[i, 0] = 1

    new_pos = pos + dir

    if (new_pos[:, 0] < 0).any() or\
            (new_pos[:, 0] + f >= img_size).any() or\
            (new_pos[:, 1] < 0).any() or\
            (new_pos[:, 1] + f >= img_size).any():
        return pos

    return new_pos
