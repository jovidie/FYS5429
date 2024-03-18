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


if __name__ == '__main__':
    joint_data = generate_trajectories(2, 10, save=False)
    data_loader = DataLoader(dataset=joint_data, batch_size=2, shuffle=True)









