import torch
import yaml

from config import Param
from methods.manager import Manager
from methods.utils import setup_seed


def run(args):
    print(f"hyper-parameter configurations:")
    print(yaml.dump(args.__dict__, sort_keys=True, indent=4))

    setup_seed(args.seed)
    manager = Manager(args)
    manager.train(args)


if __name__ == "__main__":
    # Load configuration
    param = Param()
    args = param.args

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(args.device)
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device('cpu')
        args.n_gpu = 0

    # Task name
    args.task_name = args.dataname

    # rel_per_task
    args.rel_per_task = 8 if args.dataname == "FewRel" else 4

    # Run
    run(args)
