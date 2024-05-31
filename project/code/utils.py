import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

import torch 
from torch.utils.data import TensorDataset, DataLoader

import argparse

path = "../latex/figures/"
ratinabox.stylize_plots()
ratinabox.autosave_plots = False
ratinabox.figure_directory = path 


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
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="Tuning parameter in gradient descent")
    parser.add_argument("--n_epochs",
                        default=50,
                        type=int,
                        help="Number of training epochs")
    parser.add_argument("--seq_length",
                        default=20,
                        type=int,
                        help="Number of time steps in the trajectory")
    parser.add_argument("--batch_size",
                        default=10,
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


def synthetic_trajectories(batch_size, seq_length):
    x_1 = torch.zeros([batch_size, seq_length, 2], dtype=torch.float32)
    # x_2 = torch.zeros([batch_size, seq_length, 2], dtype=torch.float32)
    y = torch.zeros([batch_size, seq_length, 2], dtype=torch.float32)

    for i in range(batch_size):
        env = Environment()
        ag = Agent(env)
            
        for _ in range(seq_length):
            ag.update()
        
        x_1[i, :, :] = torch.tensor(ag.history["vel"])
        # x_2[i, :, :] = torch.tensor(ag.history["head_direction"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    # X = torch.cat((x_1, x_2), dim=2)

    # joint_data = TensorDataset(X, y)
    joint_data = TensorDataset(x_1, y)


    return joint_data


def trajectories(n_trajectories, seq_length, n_features=2):
    X = torch.zeros([n_trajectories, seq_length, n_features], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    if n_features==4:
        for i in range(n_trajectories):
            env = Environment()
            ag = Agent(env)
                
            for _ in range(seq_length):
                ag.update()
            X[i, :, :2] = torch.tensor(ag.history["vel"])
            X[i, :, 2:] = torch.tensor(ag.history["head_direction"])
            y[i, :, :] = torch.tensor(ag.history["pos"])
    else:
        for i in range(n_trajectories):
            env = Environment()
            ag = Agent(env)
                
            for _ in range(seq_length):
                ag.update()
            X[i, :, :2] = torch.tensor(ag.history["vel"])
            y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(X, y)

    return joint_data


def new_generator(n_trajectories, seq_length):
    x = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    env = Environment()
    for i in range(n_trajectories):
        ag = Agent(env)

        for _ in range(seq_length):
            ag.update()
        
        x[i, :, :] = torch.tensor(ag.history["vel"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(x, y)

    return joint_data


def experimental_trajectories(n_trajectories, seq_length):
    x = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    # Environment of Sargolini data
    env = Environment()
    for i in range(n_trajectories):
        ag = Agent(env)
        ag.import_trajectory(dataset="sargolini")

        for _ in range(seq_length):
            ag.update()

        x[i, :, :] = torch.tensor(ag.history["vel"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(x, y)

    return joint_data
    

def generate_dataset(batch_size, seq_length, synthetic=True, save=False):
    if synthetic is False:
        joint_data = experimental_trajectories(seq_length)
        filename = f"../data/trajectories_sargolini_{seq_length}.pt"

    else:
        joint_data = synthetic_trajectories(batch_size, seq_length)
        filename = f"../data/trajectories_{batch_size}_{seq_length}.pt"

    if save:
        torch.save(joint_data, f=filename)
    else:
        return joint_data
    

def new_generate_dataset(n_trajectories, seq_length, synthetic=True, save=False):
    if synthetic is False:
        joint_data = experimental_trajectories(seq_length)
        filename = f"../data/trajectories_sargolini_{seq_length}.pt"

    else:
        joint_data = new_generator(n_trajectories, seq_length)
        filename = f"../data/trajectories_{n_trajectories}_{seq_length}.pt"

    if save:
        torch.save(joint_data, f=filename)
    else:
        return joint_data


if __name__ == '__main__':
    generate_dataset(2, 20, save=True)