# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, isfile

import torch as th
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .core import MultiAgent, episode, obs_generic, trans_generic
from .metrics import ConfusionMeter, format_metric
from .networks import ModelsWrapper
from .options import EvalOptions, MainOptions


def evaluation(main_options: MainOptions, eval_options: EvalOptions) -> None:
    assert exists(
        eval_options.json_path
    ), f'JSON path "{eval_options.json_path}" does not exist'
    assert isfile(
        eval_options.json_path
    ), f'"{eval_options.json_path}" is not a file'

    assert exists(
        eval_options.state_dict_path
    ), f"State dict path {eval_options.state_dict_path} does not exist"
    assert isfile(
        eval_options.state_dict_path
    ), f"{eval_options.state_dict_path} is not a file"

    if exists(eval_options.output_dir) and isdir(eval_options.output_dir):
        print(f"File in {eval_options.output_dir} will be overwritten")
    elif exists(eval_options.output_dir) and not isdir(
        eval_options.output_dir
    ):
        raise NotADirectoryError(
            f'"{eval_options.output_dir}" is not a directory'
        )
    else:
        print(f'Create "{eval_options.output_dir}"')
        mkdir(eval_options.output_dir)

    img_pipeline = tr.Compose([tr.ToTensor()])

    test_dataset = ImageFolder(
        eval_options.dataset_path, transform=img_pipeline
    )

    nn_models = ModelsWrapper.from_json(eval_options.json_path)
    nn_models.load_state_dict(th.load(eval_options.state_dict_path))
    nn_models.eval()

    marl_m = MultiAgent.load_from(
        eval_options.json_path,
        main_options.nb_agent,
        nn_models,
        obs_generic,
        trans_generic,
    )

    data_loader = DataLoader(
        test_dataset,
        batch_size=eval_options.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )

    # pylint: disable=duplicate-code
    cuda = main_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"
    # pylint: enable=duplicate-code

    conf_meter = ConfusionMeter(nn_models.nb_class)

    for x, y in tqdm(data_loader):
        x, y = x.to(th.device(device_str)), y.to(th.device(device_str))

        preds, _ = episode(marl_m, x, main_options.step)

        # mean over agents
        conf_meter.add(preds.mean(dim=0).detach(), y)

    print(conf_meter.conf_mat())

    precs, recs = (conf_meter.precision(), conf_meter.recall())

    precs_str = format_metric(precs, test_dataset.class_to_idx)
    recs_str = format_metric(recs, test_dataset.class_to_idx)

    print(f"Precision : {precs_str}")
    print(f"Precision mean = {precs.mean()}")
    print(f"Recall : {recs_str}")
    print(f"Recall mean : {recs.mean()}")
