import os
import re
import json
import argparse
import random
import torch
import numpy as np

import src.commons.globals as glb
import src.experiment as exp

from types import SimpleNamespace as Namespace


class Arguments(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True, help='Provide the JSON config path with the parameters of your experiment')
        parser.add_argument('--replicable', action='store_true', help='If provided, a seed will be used to allow replicability')

        args = parser.parse_args()

        # Fields expected from the command line
        self.config = os.path.join(glb.PROJ_DIR, args.config)
        self.replicable = args.replicable

        assert os.path.exists(self.config) and self.config.endswith('.json'), 'The config path provided does not exist or is not a JSON file'

        # Read the parameters from the JSON file and skip comments
        with open(self.config, 'r') as f:
            params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])

        arguments = json.loads(params, object_hook=lambda d: Namespace(**d))

        # Must-have fields expected from the JSON config file
        self.experiment = arguments.experiment
        self.data = arguments.data
        self.model = arguments.model
        self.training = arguments.training
        self.optim = self.training.optim

        # Optim Args
        self.optim.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim.n_gpu = torch.cuda.device_count()

        # Checking that the JSON contains at least the fixed fields
        assert all([hasattr(self.data.text, name) for name in {'train', 'dev', 'test'}])
        assert all([hasattr(self.training, name) for name in {'epochs', 'per_gpu_train_batch_size', 'per_gpu_eval_batch_size', 'optim'}])
        assert all([hasattr(self.training.optim, name) for name in {'learning_rate', 'weight_decay'}])

        self._format_datapaths()
        self._add_extra_fields()


    def _format_datapaths(self):
        self.data.directory = os.path.join(glb.PROJ_DIR, self.data.directory)

        self.data.text.train = os.path.join(self.data.directory, self.data.text.train)
        self.data.text.dev = os.path.join(self.data.directory, self.data.text.dev)
        self.data.text.test = os.path.join(self.data.directory, self.data.text.test)

        if self.data.image.train is not None:
            self.data.image.train = os.path.join(self.data.directory, self.data.image.train)
            self.data.image.dev = os.path.join(self.data.directory, self.data.image.dev)
            self.data.image.test = os.path.join(self.data.directory, self.data.image.test)
        
        if self.data.caption.train is not None:
            self.data.caption.train = os.path.join(self.data.directory, self.data.caption.train)
            self.data.caption.dev = os.path.join(self.data.directory, self.data.caption.dev)
            self.data.caption.test = os.path.join(self.data.directory, self.data.caption.test)


    def _add_extra_fields(self):
        self.experiment.output_dir = os.path.join(glb.PROJ_DIR, self.experiment.output_dir, self.experiment.id)
        self.experiment.checkpoint_dir = os.path.join(self.experiment.output_dir, 'checkpoint')
        


def main():
    args = Arguments()

    if args.replicable:
        seed_num = 123
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)
            torch.backends.cudnn.deterministic = True

    print("[LOG] {}".format('=' * 40))
    print("[LOG] {: >15}: '{}'".format("Experiment ID", args.experiment.id))
    print("[LOG] {: >15}: '{}'".format("Description", args.experiment.description))
    for key, val in vars(args.data.text).items():
        print("[LOG] {: >15}: {}".format(key, val))
    print("[LOG] {: >15}: '{}'".format("Modeling", vars(args.model)))
    print("[LOG] {: >15}: '{}'".format("Training", vars(args.training)))
    print("[LOG] {: >15}: '{}'".format("GPUs avaliable", args.optim.n_gpu))
    print("[LOG] {}".format('=' * 40))

    exp.main(args)


if __name__ == '__main__':
    main()
