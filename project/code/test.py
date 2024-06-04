import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader

import argparse

from utils import parse_args, synthetic_dataset, load_dataset
from model import VanillaRNN, SimpleRNN, NeuroRNN


def test_parse_args():
    args = parse_args()
    print(args)


def test_generate_trajectories():
    vel_data = synthetic_dataset(n_trajectories=20, seq_length=20, features="vel")
    dl = DataLoader(vel_data, batch_size=10)
    assert len(dl) == 2 # Number of batches when batch_size=10
    assert len(dl.dataset) == 20

    vel_head_data = synthetic_dataset(n_trajectories=20, seq_length=20, features="vel_head")
    dl = DataLoader(vel_head_data, 10)
    assert len(dl) == 2
    assert len(dl.dataset) == 20

    # vel_exp_data = experimental_dataset(n_trajectories=1, seq_length=20, features="vel")
    # dl = DataLoader(vel_exp_data)
    # assert len(dl) == 1
    # assert len(dl.dataset) == 1

    # vel_head_exp_data = experimental_dataset(n_trajectories=1, seq_length=20, features="vel_head")
    # dl = DataLoader(vel_head_exp_data)
    # assert len(dl) == 1
    # assert len(dl.dataset) == 1


def test_simple_rnn_vel():
    n_inputs=2
    n_neurons=128
    n_outputs=2
    n_trajectories=100
    learning_rate=0.0001
    seq_length=20
    n_epochs=1000
    batch_size=20
    model = SimpleRNN(n_epochs=n_epochs)

    data = synthetic_dataset(n_trajectories, seq_length)    
    dl = DataLoader(data, batch_size)

    model.train(dl, verbose=True)

    unseen_data = synthetic_dataset(batch_size, seq_length)
    unseen_loader = DataLoader(unseen_data, batch_size)

    x_test, y_test = next(iter(unseen_loader))
    y_tilde = model.predict((x_test, y_test))
    
    predicted_traj = y_tilde.detach().numpy()
    true_traj = y_test.detach().numpy()
    print(predicted_traj.shape, true_traj.shape)
    
    colors = sns.color_palette("mako", n_colors=batch_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    for i in range(batch_size):
        ax.plot(predicted_traj[i, :, 0], predicted_traj[i, :, 1], color=colors[i])
        ax.plot(true_traj[i, :, 0], true_traj[i, :, 1], "--", color=colors[i])
    
    # fig.savefig("../latex/figures/test_model.pdf", bbox_inches="tight")
    plt.show()

def test_simple_rnn_vel_head():
    n_inputs=4
    n_neurons=128
    n_outputs=2
    n_trajectories=1000
    learning_rate=0.0001
    seq_length=20
    n_epochs=1000
    batch_size=20
    model = SimpleRNN(n_inputs=n_inputs, n_epochs=n_epochs)

    data = synthetic_dataset(n_trajectories, seq_length, features="vel_head")    
    dl = DataLoader(data, batch_size)

    model.train(dl, verbose=True)

    unseen_data = synthetic_dataset(batch_size, seq_length, features="vel_head")
    unseen_loader = DataLoader(unseen_data, batch_size)

    x_test, y_test = next(iter(unseen_loader))
    y_tilde = model.predict((x_test, y_test))
    
    predicted_traj = y_tilde.detach().numpy()
    true_traj = y_test.detach().numpy()
    print(predicted_traj.shape, true_traj.shape)
    
    colors = sns.color_palette("mako", n_colors=batch_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    for i in range(batch_size):
        ax.plot(predicted_traj[i, :, 0], predicted_traj[i, :, 1], color=colors[i])
        ax.plot(true_traj[i, :, 0], true_traj[i, :, 1], "--", color=colors[i])
    
    # fig.savefig("../latex/figures/test_model.pdf", bbox_inches="tight")
    plt.show()


def test_vanilla_rnn_velocity_head():
    args = parse_args()
    args.n_inputs = 4
    model = VanillaRNN(args)

    data = synthetic_dataset(args.n_trajectories, args.seq_length)    
    dl = DataLoader(data)

    model.train(dl, verbose=True)

    unseen_data = synthetic_dataset(args.batch_size, args.seq_length)
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

def test_neuro_rnn():
    args = parse_args()
    model = NeuroRNN(args)

    data = synthetic_dataset(args.n_trajectories, args.seq_length)    
    dl = DataLoader(data)

    model.train(dl, verbose=True)
    print(model.output.parameters)

    unseen_data = synthetic_dataset(args.batch_size, args.seq_length)
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


def test_learning_rate():
    epochs = np.arange(1000, 6000, 1000)
    learning_rates = np.array([1e-4, 1e-3, 1e-2])
    computed_loss = np.zeros((epochs.size, learning_rates.size))

    batch_size = 20
    data = load_dataset(100, 20)
    dl = DataLoader(data, batch_size)

    for i, n_epochs in enumerate(epochs):
        print(f"Epoch: {i+1}")
        for j, lr in enumerate(learning_rates):
            print(f"Learning rate: {lr}")
            model = SimpleRNN(n_epochs=n_epochs, learning_rate=lr)
            computed_loss[i, j] = model.train(dl)

    print(np.argmin(computed_loss))

if __name__ == '__main__':
    # test_parse_args()
    # test_generate_trajectories()
    # test_simple_rnn_vel()
    # test_simple_rnn_vel_head()
    # test_vanilla_rnn_velocity_head()
    # test_neuro_rnn()
    test_learning_rate()
