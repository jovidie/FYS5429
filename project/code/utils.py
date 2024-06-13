from dataclasses import dataclass

import torch 
from torch.nn import MSELoss

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from generate_data import generate_data



def plot_theme():
    sns.set_theme()
    params = {
        "font.family": "Serif",
        "font.serif": "Roman", 
        "text.usetex": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium"
    }
    plt.rcParams.update(params)


@dataclass
class VanillaRNNargs:
    n_inputs: int = 2
    n_neurons: int = 128
    n_outputs: int = 2

    seq_length: int = 20
    batch_size: int = 20

    lr: float = 0.001
    n_epochs: int = 10000
    loss_func = MSELoss(reduction="mean")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class NeuroRNNargs:
    n_inputs: int = 2
    n_gc: int = 128
    n_pc: int = 128
    n_outputs: int = 2

    seq_length: int = 20
    batch_size: int = 20

    lr: float = 0.001
    n_epochs: int = 10000
    loss_func = MSELoss(reduction="mean")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_inputs",
                        default=2,
                        type=int,
                        help="Number of input features")
    parser.add_argument("--n_neurons",
                        default=64,
                        type=int,
                        help="Number of neurons in hidden layer")
    parser.add_argument("--n_gc",
                        default=128,
                        type=int,
                        help="Number of neurons in MEC hidden layer")
    parser.add_argument("--n_pc",
                        default=64,
                        type=int,
                        help="Number of neurons in CA1 hidden layer")
    parser.add_argument("--n_layers",
                        default=1,
                        type=int,
                        help="Number of hidden layers")
    parser.add_argument("--n_outputs",
                        default=2,
                        type=int,
                        help="Number of output features")
    parser.add_argument("--lr",
                        default=0.001,
                        type=float,
                        help="Learning rate, tuning parameter in gradient descent")
    parser.add_argument("--n_epochs",
                        default=50,
                        type=int,
                        help="Number of training epochs")
    parser.add_argument("--seq_length",
                        default=20,
                        type=int,
                        help="Number of time steps in the trajectory")
    parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        help="Number of trajectories per batch")
    parser.add_argument("--n_trajectories",
                        default=100,
                        type=int,
                        help="Number of trajectories to generate")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for training model")
    parser.add_argument("--trajectory_dir",
                        default="../data/",
                        help="Directory for saving dataset")
    parser.add_argument("--model_dir",
                        default="../models/",
                        help="Directory for saving trained models")
    # Include param to specify load or generate data?
    
    args = parser.parse_args()

    return args


def load_data(
        n_trajectories=100, 
        seq_length=20, 
        features="vel", 
        type="features", 
        env="square"
):
    data_path = "../data/trajectories/"
    if os.path.exists(data_path):
        if type == "features":
            filename = f"{data_path}{features}_{n_trajectories}_{seq_length}.pt"
            joint_data = torch.load(f=filename)

        elif type == "env":
            filename = f"{data_path}env_{env}_{n_trajectories}_{seq_length}.pt"
            joint_data = torch.load(f=filename)

        elif type == "experimental":
            filename = f"{data_path}sargolini_{env}_{n_trajectories}_{seq_length}.pt"
            joint_data = torch.load(f=filename)

        return joint_data
    
    else:
        print("Data does not exist, generating data!")
        os.mkdir(data_path)
        generate_data()


def plot_trajectories(test_size, true, pred, save=False, filename=""):
    plot_theme()
    colors = sns.color_palette("mako", n_colors=test_size)
    fig, ax = plt.subplots(figsize=(4, 4))

    for i in range(test_size):
        ax.plot(pred[i, 0:-1, 0], pred[i, 0:-1, 1], color=colors[i])
        ax.plot(true[i, 1:, 0], true[i, 1:, 1], "--", color=colors[i])
    
    if save:
        path = f"../latex/figures/{filename}.pdf"
        fig.savefig(path, bbox_inches="tight")
    else:
        plt.show()