import numpy as np

import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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
    synthetic_dataset(n_trajectories=100, seq_length=20, features="vel", save=True)