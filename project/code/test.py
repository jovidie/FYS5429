import numpy as np
import torch
import argparse

from utils import parse_args, generate_dataset
from model import VanillaRNN


def test_parse_args():
    args = parse_args()
    print(args)


def test_generate_trajectories():
    pass


def test_vanilla_rnn_velocity():
    pass


def test_vanilla_rnn_velocity_head():
    pass


if __name__ == '__main__':
    test_parse_args()