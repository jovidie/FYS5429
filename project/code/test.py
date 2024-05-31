import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader

import argparse

from utils import parse_args, trajectories, synthetic_trajectories, new_generator
from model import VanillaRNN, NeuroRNN


def test_parse_args():
    args = parse_args()
    print(args)


def test_generate_trajectories():
    args = parse_args()
    model = VanillaRNN(args)

    data = trajectories(2, 10, 4)
    dl = DataLoader(data)


def test_vanilla_rnn_velocity():
    pass


def test_vanilla_rnn_velocity_head():
    args = parse_args()
    args.n_inputs = 4
    model = VanillaRNN(args)

    data = synthetic_trajectories(args.n_trajectories, args.seq_length)    
    dl = DataLoader(data)

    model.train(dl, verbose=True)

    unseen_data = synthetic_trajectories(args.batch_size, args.seq_length)
    unseen_loader = DataLoader(unseen_data, args.batch_size)

    x_true, y_true = next(iter(unseen_loader))
    y_tilde = model.predict((x_true, y_true))
    
    predicted_traj = y_tilde.detach().numpy()
    true_traj = y_true.detach().numpy()
    print(predicted_traj.shape, true_traj.shape)
    
    colors = sns.color_palette("mako", n_colors=args.batch_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    for i in range(args.batch_size):
        ax.plot(predicted_traj[i, :, 0], predicted_traj[i, :, 1], color=colors[i])
        ax.plot(true_traj[i, :, 0], true_traj[i, :, 1], "--", color=colors[i])
    
    # fig.savefig("../latex/figures/test_model.pdf", bbox_inches="tight")
    plt.show()

def test_neuro_rnn():
    args = parse_args()
    model = NeuroRNN(args)

    data = new_generator(args.n_trajectories, args.seq_length)    
    dl = DataLoader(data)

    model.train(dl, verbose=True)
    print(model.output.parameters)

    unseen_data = synthetic_trajectories(args.batch_size, args.seq_length)
    unseen_loader = DataLoader(unseen_data, args.batch_size)

    x_true, y_true = next(iter(unseen_loader))
    y_tilde = model.predict((x_true, y_true))
    
    predicted_traj = y_tilde.detach().numpy()
    true_traj = y_true.detach().numpy()
    
    colors = sns.color_palette("mako", n_colors=args.batch_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    for i in range(args.batch_size):
        ax.plot(predicted_traj[i, :, 0], predicted_traj[i, :, 1], color=colors[i])
        ax.plot(true_traj[i, :, 0], true_traj[i, :, 1], "--", color=colors[i])
    
    # fig.savefig("../latex/figures/test_model.pdf", bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    # test_parse_args()
    # test_generate_trajectories()
    # test_vanilla_rnn_velocity_head()
    test_neuro_rnn()