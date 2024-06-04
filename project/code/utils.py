from dataclasses import dataclass

import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

import torch 
from torch.nn import MSELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import argparse

path = "../latex/figures/"
ratinabox.stylize_plots()
ratinabox.autosave_plots = False
ratinabox.figure_directory = path 


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

    lr: float = 0.001
    n_epochs: int = 1000
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


def vel_data(n_trajectories, seq_length):
    X = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    for i in range(n_trajectories):
        env = Environment()
        ag = Agent(env)
            
        for _ in range(seq_length):
            ag.update()
        
        X[i, :, :] = torch.tensor(ag.history["vel"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(X, y)

    return joint_data


def vel_head_data(n_trajectories, seq_length):
    X = torch.zeros([n_trajectories, seq_length, 4], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    for i in range(n_trajectories):
        env = Environment()
        ag = Agent(env)
            
        for _ in range(seq_length):
            ag.update()
        
        X[i, :, :2] = torch.tensor(ag.history["vel"])
        X[i, :, 2:] = torch.tensor(ag.history["head_direction"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(X, y)

    return joint_data


def vel_exp_data(n_trajectories, seq_length):
    X = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    # Environment of Sargolini data
    for i in range(n_trajectories):
        env = Environment()
        ag = Agent(env)
        ag.import_trajectory(dataset="sargolini")

        for _ in range(seq_length):
            ag.update()

        X[i, :, :] = torch.tensor(ag.history["vel"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(X, y)

    return joint_data


def vel_head_exp_data(n_trajectories, seq_length):
    X = torch.zeros([n_trajectories, seq_length, 4], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    for i in range(n_trajectories):
        env = Environment()
        ag = Agent(env)
        ag.import_trajectory(dataset="sargolini")
            
        for _ in range(seq_length):
            ag.update()
        
        X[i, :, :2] = torch.tensor(ag.history["vel"])
        X[i, :, 2:] = torch.tensor(ag.history["head_direction"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(X, y)

    return joint_data


def synthetic_dataset(n_trajectories, seq_length, features="vel", save=False):
    try:
        if features == "vel":
            joint_data = vel_data(n_trajectories, seq_length)
            filename = f"../data/synthetic/vel_{n_trajectories}_{seq_length}.pt"
        elif features == "vel_head":
            joint_data = vel_head_data(n_trajectories, seq_length)
            filename = f"../data/synthetic/vel_head_{n_trajectories}_{seq_length}.pt"
    except ValueError:
        print(f"{features} is not a valid feature, enter vel or vel_head.")

    if save:
        torch.save(joint_data, filename)
    else:
        return joint_data
    

def experimental_dataset(n_trajectories, seq_length, features="vel", save=False):
    try:
        if features == "vel":
            joint_data = vel_exp_data(n_trajectories, seq_length)
            filename = f"../data/experimental/vel_{n_trajectories}_{seq_length}.pt"
        elif features == "vel_head":
            joint_data = vel_head_exp_data(n_trajectories, seq_length)
            filename = f"../data/experimental/vel_head_{n_trajectories}_{seq_length}.pt"
    except ValueError:
        print(f"{features} is not a valid feature, enter vel or vel_head.")

    if save:
        torch.save(joint_data, filename)
    else:
        return joint_data
    

def load_dataset(n_trajectories, seq_length, features="vel", synthetic=True):
    try:
        if synthetic:
            filename = f"../data/synthetic/{features}_{n_trajectories}_{seq_length}.pt"
            joint_data = torch.load(f=filename)
        else:
            filename = f"../data/experimental/{features}_{n_trajectories}_{seq_length}.pt"
            joint_data = torch.load(f=filename)

        return joint_data
    except FileNotFoundError:
        print("Dataset has not been generated!")


if __name__ == '__main__':
    # synthetic_dataset(10000, 20, features="vel", save=True)
    # synthetic_dataset(100, 20, features="vel_head", save=True)
    args = VanillaRNNargs(n_inputs=2)
    print(args.lr)