import torch as th


def obs_MNIST(img: th.Tensor, pos: th.Tensor, f: int) -> th.Tensor:
    assert len(img.size()) == 2, "Need 2-D input"
    assert pos[0] + f < img.size(0) and pos[0] >= 0, "Unbound obervation in first dim !"
    assert pos[1] + f < img.size(1) and pos[1] >= 0, "Unbound obervation in snd dim !"

    return img[pos[0]:pos[0] + f, pos[1]:pos[1] + f]
