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

    parser.add_argument("--learning_rate",
                        # dest="accumulate",
                        default=0.001,
                        help="Tuning parameter in gradient descent")
    parser.add_argument("--n_epochs",
                        default=100,
                        help="Number of training epochs")
    parser.add_argument("--n_neurons",
                        default=64,
                        help="Number of neurons in hidden layer")
    parser.add_argument("--seq_length",
                        default=20,
                        help="Number of time steps in the trajectory")
    parser.add_argument("--batch_size",
                        default=10,
                        help="Number of trajectories per batch")
    parser.add_argument("--n_trajectories",
                        default=50,
                        help="Number of trajectories to generate")
    parser.add_argument("--trajectory_dir",
                        default="../data/",
                        help="Directory for saving dataset")
    parser.add_argument("--model_dir",
                        default="../models/",
                        help="Directory for saving trained models")
    # Include param to specify load or generate data?
    
    args = parser.parse_args()

    return args


def synthetic_trajectories(batch_size, seq_length, save=False):
    x = torch.zeros([batch_size, seq_length, 2], dtype=torch.float32)
    y = torch.zeros([batch_size, seq_length, 2], dtype=torch.float32)

    for i in range(batch_size):
        env = Environment()
        ag = Agent(env)
            
        for _ in range(seq_length):
            ag.update()
        
        x[i, :, :] = torch.tensor(ag.history["vel"])
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(x, y)

    return joint_data


def experimental_trajectories(seq_length, save=False):
    x = torch.zeros([1, seq_length, 2], dtype=torch.float32)
    y = torch.zeros([1, seq_length, 2], dtype=torch.float32)

    # Environment of Sargolini data
    env = Environment()
    ag = Agent(env)

    ag.import_trajectory(dataset="sargolini")

    for _ in range(seq_length):
        ag.update()

    x[0, :, :] = torch.tensor(ag.history["vel"])
    y[0, :, :] = torch.tensor(ag.history["pos"])

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
    


if __name__ == '__main__':
    generate_dataset(2, 20, save=True)