import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

import torch 
from torch.utils.data import TensorDataset, DataLoader

path = "../latex/figures/"
ratinabox.stylize_plots()
ratinabox.autosave_plots = False
ratinabox.figure_directory = path 

def generate_trajectories(batch_size, seq_length, save=False):
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
    if save:
        filename = f"../data/trajectories_{batch_size}.pt"
        torch.save(joint_data, f=filename)
    else:
        return joint_data
    

def new_generator(n_trajectories, seq_len, features=["vel"], save=False):
    # Create dataset with velocity
    if len(features) == 1:
        x = torch.zeros([n_trajectories, seq_len, 2], dtype=torch.float32)
        y = torch.zeros([n_trajectories, seq_len, 2], dtype=torch.float32)

        env = Environment()
        for i in range(n_trajectories):
            ag = Agent(env)

            for _ in range(seq_len):
                ag.update()
            
            # print(ag.history.keys())
            # ['t', 'pos', 'distance_travelled', 'vel', 'rot_vel', 'head_direction']
            x[i, :, :] = torch.tensor(ag.history["vel"])
            y[i, :, :] = torch.tensor(ag.history["pos"])

    joint_data = TensorDataset(x, y)

    return joint_data


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


if __name__ == '__main__':
    joint_data = new_generator(2, 10, save=False)
    # data_loader = DataLoader(dataset=joint_data, batch_size=2, shuffle=True)









