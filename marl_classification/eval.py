from os import mkdir
from os.path import exists, isfile, isdir
from random import shuffle

import torch as th
import torchvision.transforms as tr
from torch.utils.data import Subset, DataLoader
from torchnet.meter import ConfusionMeter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .data import transforms as custom_tr
from .environment import (
    MultiAgent,
    obs_generic,
    trans_generic,
    episode
)
from .networks import ModelsWrapper
from .utils import MainOptions, EvalOptions, prec_rec, format_metric


def evaluation(
        main_options: MainOptions,
        eval_options: EvalOptions
) -> None:
    steps = main_options.step

    json_path = eval_options.json_path
    state_dict_path = eval_options.state_dict_path
    image_root = eval_options.image_root
    output_dir = eval_options.output_dir

    assert exists(json_path), \
        f"JSON path \"{json_path}\" does not exist"
    assert isfile(json_path), \
        f"\"{json_path}\" is not a file"

    assert exists(state_dict_path), \
        f"State dict path {state_dict_path} does not exist"
    assert isfile(state_dict_path), \
        f"{state_dict_path} is not a file"

    if exists(output_dir) and isdir(output_dir):
        print(f"File in {output_dir} will be overwritten")
    elif exists(output_dir) and not isdir(output_dir):
        raise Exception(f"\"{output_dir}\" is not a directory")
    else:
        print(f"Create \"{output_dir}\"")
        mkdir(output_dir)

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    img_dataset = ImageFolder(image_root, transform=img_pipeline)

    idx = list(range(len(img_dataset)))
    shuffle(idx)
    idx_test = idx[int(0.85 * len(idx)):]

    test_dataset = Subset(img_dataset, idx_test)

    nn_models = ModelsWrapper.from_json(json_path)
    nn_models.load_state_dict(th.load(state_dict_path))
    marl_m = MultiAgent.load_from(
        json_path, main_options.nb_agent,
        nn_models, obs_generic, trans_generic
    )

    data_loader = DataLoader(
        test_dataset, batch_size=eval_options.batch_size,
        shuffle=True, num_workers=8, drop_last=False
    )

    cuda = main_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    conf_meter = ConfusionMeter(nn_models.nb_class)

    for x, y in tqdm(data_loader):
        x, y = x.to(th.device(device_str)), y.to(th.device(device_str))

        preds, probas = episode(marl_m, x, 0., steps)

        conf_meter.add(preds.detach(), y)

    print(conf_meter.value())

    precs, recs = prec_rec(conf_meter)

    precs_str = format_metric(precs, img_dataset.class_to_idx)
    recs_str = format_metric(recs, img_dataset.class_to_idx)

    print(f"Precision : {precs_str}")
    print(f"Precision mean = {precs.mean()}")
    print(f"Recall : {recs_str}")
    print(f"Recall mean : {recs.mean()}")
