#!/usr/bin/env python3
import os
import sys

import fire
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tahiremovision import command, server


class TahiremoVision:
    def command(self, input_file: str, model_file: str = 'model.pth', input_size: int = 224):
        print(command(input_file, model_file, input_size))

    def server(self, config_file: str):
        with open(config_file) as f:
            config = yaml.load(f)
        print(server(**config))


def main():
    fire.Fire(TahiremoVision)


if __name__ == '__main__':
    main()
