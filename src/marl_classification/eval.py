from os import mkdir
from os.path import exists, isdir, isfile

import torch as th
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .config import EvalConfig, MainConfig, ModelConfig
from .core import EpisodeSampler
from .metrics import ConfusionMeter, format_metric
from .registry import default_image_pipeline


def eval_main(main_config: MainConfig, eval_config: EvalConfig) -> None:
    assert exists(
        eval_config.json_path
    ), f'JSON path "{eval_config.json_path}" does not exist'
    assert isfile(
        eval_config.json_path
    ), f'"{eval_config.json_path}" is not a file'

    assert exists(
        eval_config.state_dict_path
    ), f"State dict path {eval_config.state_dict_path} does not exist"
    assert isfile(
        eval_config.state_dict_path
    ), f"{eval_config.state_dict_path} is not a file"

    if exists(eval_config.output_dir) and isdir(eval_config.output_dir):
        print(f"File in {eval_config.output_dir} will be overwritten")
    elif exists(eval_config.output_dir) and not isdir(eval_config.output_dir):
        raise NotADirectoryError(
            f'"{eval_config.output_dir}" is not a directory'
        )
    else:
        print(f'Create "{eval_config.output_dir}"')
        mkdir(eval_config.output_dir)

    img_pipeline = default_image_pipeline()

    test_dataset = ImageFolder(
        eval_config.dataset_path, transform=img_pipeline
    )

    marl_config = ModelConfig.load_marl_config(eval_config.json_path)

    nn_models, marl_m, env = marl_config.build_marl(main_config.nb_agent)

    nn_models.load_state_dict(th.load(eval_config.state_dict_path))
    nn_models.eval()

    data_loader = DataLoader(
        test_dataset,
        batch_size=eval_config.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )

    device = th.device("cuda" if main_config.cuda else "cpu")
    nn_models.to(device)

    episode_sampler = EpisodeSampler(marl_m, env, main_config.step)

    conf_meter = ConfusionMeter(nn_models.nb_class)

    with th.no_grad():
        for x, y in tqdm(data_loader):
            x, y = x.to(device), y.to(device)

            output = episode_sampler.run_episode_get_last_step(x)

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
