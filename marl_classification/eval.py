from os import mkdir
from os.path import exists, isdir, isfile

import torch as th
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .core import episode
from .metrics import ConfusionMeter, format_metric
from .options import EvalOptions, MainOptions
from .registry import default_image_pipeline
from .serde import MarlConfig


def eval_main(main_options: MainOptions, eval_options: EvalOptions) -> None:
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

    img_pipeline = default_image_pipeline()

    test_dataset = ImageFolder(
        eval_options.dataset_path, transform=img_pipeline
    )

    marl_config = MarlConfig.load_marl_config(eval_options.json_path)

    nn_models, marl_m, env = marl_config.build_marl(main_options.nb_agent)

    nn_models.load_state_dict(th.load(eval_options.state_dict_path))
    nn_models.eval()

    data_loader = DataLoader(
        test_dataset,
        batch_size=eval_options.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )

    device = th.device("cuda" if main_options.cuda else "cpu")
    nn_models.to(device)

    conf_meter = ConfusionMeter(nn_models.nb_class)

    with th.no_grad():
        for x, y in tqdm(data_loader):
            x, y = x.to(device), y.to(device)

            output = episode(marl_m, env, x, main_options.step)

            # mean over agents
            conf_meter.add(output.prediction.mean(dim=0).detach(), y)

    print(conf_meter.conf_mat())

    precs, recs = (conf_meter.precision(), conf_meter.recall())

    precs_str = format_metric(precs, test_dataset.class_to_idx)
    recs_str = format_metric(recs, test_dataset.class_to_idx)

    print(f"Precision : {precs_str}")
    print(f"Precision mean = {precs.mean()}")
    print(f"Recall : {recs_str}")
    print(f"Recall mean : {recs.mean()}")
