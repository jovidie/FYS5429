import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

import torch 
from torch.utils.data import TensorDataset

import numpy as np
import contextlib

path = "../latex/figures/"
ratinabox.stylize_plots()
ratinabox.autosave_plots = False
ratinabox.figure_directory = path 


def vel_data(n_trajectories, seq_length, save=False):
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

    if save:
        filename = f"../data/trajectories/vel_{n_trajectories}_{seq_length}.pt"
        torch.save(joint_data, filename)
    else:
        return joint_data

    
def vel_head_data(n_trajectories, seq_length, save=False):
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

    if save:
        filename = f"../data/trajectories/vel_head_{n_trajectories}_{seq_length}.pt"
        torch.save(joint_data, filename)
    else:
        return joint_data
    
def vel_head_rot_data(n_trajectories, seq_length, save=False):
    X = torch.zeros([n_trajectories, seq_length, 5], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    for i in range(n_trajectories):
        env = Environment()
        ag = Agent(env)

        for _ in range(seq_length):
            ag.update()

        X[i, :, :2] = torch.tensor(ag.history["vel"])
        X[i, :, 2:4] = torch.tensor(ag.history["head_direction"])
        X[i, :, 4:] = torch.reshape(torch.tensor(ag.history["rot_vel"]), (seq_length, 1))
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(X, y)

    if save:
        filename = f"../data/trajectories/vel_head_rot_{n_trajectories}_{seq_length}.pt"
        torch.save(joint_data, filename)
    else:
        return joint_data 


def vel_head_rot_dist_data(n_trajectories, seq_length, save=False):
    X = torch.zeros([n_trajectories, seq_length, 6], dtype=torch.float32)
    y = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    for i in range(n_trajectories):
        env = Environment()
        ag = Agent(env)

        for _ in range(seq_length):
            ag.update()

        X[i, :, :2] = torch.tensor(ag.history["vel"])
        X[i, :, 2:4] = torch.tensor(ag.history["head_direction"])
        X[i, :, 4:5] = torch.reshape(torch.tensor(ag.history["rot_vel"]), (seq_length, 1))
        X[i, :, 5:] = torch.reshape(torch.tensor(ag.history["distance_travelled"]), (seq_length, 1))
        y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(X, y)
    
    if save:
        filename = f"../data/trajectories/vel_head_rot_dist_{n_trajectories}_{seq_length}.pt"
        torch.save(joint_data, filename)
    else:
        return joint_data
    

def experimental_data(n_trajectories, seq_length=20, save=False):
    # Environments 
    envs = {
        # Center all environments around (0, 0)
        "env1":[[0, 0],[0 , 1],[1, 1],[1, 0]], 
        "env2":[[0.5 + 0.45*np.cos(t),0.5 + 0.45*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]}

    # Environment 1
    X_1 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y_1 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    # Environment 2
    # X_2 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    # y_2 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    # Environment of Sargolini data is not clear
    for i in range(n_trajectories):
        env_1 = Environment(params={"boundary":envs["env1"]})
        # env_2 = Environment(params={"boundary":envs["env2"]})

        ag_1 = Agent(env_1)
        with contextlib.redirect_stdout(None):
            ag_1.import_trajectory(dataset="sargolini")
        # ag_2 = Agent(env_2)
        # ag_2.import_trajectory(dataset="sargolini")

        for _ in range(seq_length):
            ag_1.update()
            # ag_2.update()

        X_1[i, :, :] = torch.tensor(ag_1.history["vel"])
        y_1[i, :, :] = torch.tensor(ag_1.history["pos"])
        # X_2[i, :, :] = torch.tensor(ag_2.history["vel"])
        # y_2[i, :, :] = torch.tensor(ag_2.history["pos"])

    joint_data_1 = TensorDataset(X_1, y_1)
    # joint_data_2 = TensorDataset(X_2, y_2)

    if save:
        torch.save(joint_data_1, f"../data/trajectories/sargolini_square_{n_trajectories}_{seq_length}.pt")
        # torch.save(joint_data_2, f"../data/trajectories/sargolini_circle_{n_trajectories}_{seq_length}.pt")
    else:
        return joint_data_1#, joint_data_2
    

def synthetic_data(n_trajectories, seq_length=20, feature="vel", save=False):
    try:
        if feature == "vel":
            vel_data(n_trajectories, seq_length, save)
        elif feature == "vel_head":
            vel_head_data(n_trajectories, seq_length, save)
        elif feature == "vel_head_rot":
            vel_head_rot_data(n_trajectories, seq_length, save)
        elif feature == "vel_head_rot_dist":
            vel_head_rot_dist_data(n_trajectories, seq_length, save)
    except ValueError:
        print(f"{feature} is not a valid feature, enter vel, vel_head, vel_head_rot or vel_head_rot_dist.")


def environment_data(n_trajectories, seq_length=20, save=False):
    # Environments and walls
    envs = {
        # Need to center all the environments around (0, 0)
        "env1":[[-0.5, -0.5],[-0.5, 0.5],[0.5, 0.5],[0.5, -0.5]], 
        "env2":[[0.45*np.cos(t),0.45*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
        "env3":[[-0.8, -0.5],[-0.8, 0.5],[0.8, 0.5],[0.8, -0.5]]}
    walls = {"wall1":[[0.8, 0], [0.8, 0.5]]}

    # Environment 1
    X_1 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y_1 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    # Environment 2
    X_2 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y_2 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    # Environment 3
    X_3 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)
    y_3 = torch.zeros([n_trajectories, seq_length, 2], dtype=torch.float32)

    # Generate data
    for i in range(n_trajectories):
        env_1 = Environment(params={"boundary":envs["env1"]})
        env_2 = Environment(params={"boundary":envs["env2"]})
        env_3 = Environment(params={"boundary":envs["env3"]})
        env_3.add_wall(walls["wall1"])

        ag_1 = Agent(env_1)
        ag_2 = Agent(env_2)
        ag_3 = Agent(env_3)

        for _ in range(seq_length):
            ag_1.update()
            ag_2.update()
            ag_3.update()

        X_1[i, :, :] = torch.tensor(ag_1.history["vel"])
        y_1[i, :, :] = torch.tensor(ag_1.history["pos"])
        X_2[i, :, :] = torch.tensor(ag_2.history["vel"])
        y_2[i, :, :] = torch.tensor(ag_2.history["pos"])
        X_3[i, :, :] = torch.tensor(ag_3.history["vel"])
        y_3[i, :, :] = torch.tensor(ag_3.history["pos"])

    joint_data_2 = TensorDataset(X_2, y_2)
    joint_data_1 = TensorDataset(X_1, y_1)
    joint_data_3 = TensorDataset(X_3, y_3)

    if save:
        torch.save(joint_data_1, f"../data/trajectories/env_square_{n_trajectories}_{seq_length}.pt")
        torch.save(joint_data_2, f"../data/trajectories/env_circle_{n_trajectories}_{seq_length}.pt")
        torch.save(joint_data_3, f"../data/trajectories/env_rectangle_{n_trajectories}_{seq_length}.pt")
    else:
        return joint_data_1, joint_data_2, joint_data_3
    

def generate_data(
    trajectories = [100, 1000],
    sequences = [20, 30, 40, 50, 60, 70, 80],
    features = ["vel", "vel_head", "vel_head_rot", "vel_head_rot_dist"]
):
    
    print("Generating data...")
    for t in trajectories:
        # Different environments
        environment_data(t, save=True)

        # Experimental
        
        experimental_data(t, save=True)

        for s in sequences:
            vel_data(t, s, save=True)
        
        for f in features:
            synthetic_data(t, feature=f, save=True)
    print("Done!")