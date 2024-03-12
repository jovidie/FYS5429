import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

import pandas as pd 
import numpy as np

import torch 
import torch.nn as nn 

path = "../latex/figures/"
ratinabox.stylize_plots()
ratinabox.autosave_plots = False
ratinabox.figure_directory = path 

def generate_trajectories(batch_size, seq_length):
    filename = "../data/tensor.pt"
    data = torch.zeros([2, batch_size, seq_length, 2], dtype=torch.float32)
    for i in range(batch_size):
        env = Environment()
        ag = Agent(env)

        for _ in range(seq_length):
            ag.update()

        data[0, i, :, :] = torch.tensor(ag.history["vel"])
        data[1, i, :, :] = torch.tensor(ag.history["pos"])
    torch.save(data, filename)

def load_dataset(filename):
    pass

if __name__ == '__main__':
    generate_trajectories(2, 10)
