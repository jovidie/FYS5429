import os

import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
import pandas as pd 

ratinabox.stylize_plots()
ratinabox.autosave_plots = True
os.makedirs("../latex/figures", exist_ok=True)
ratinabox.figure_directory = "../latex/figures/"

def simulate():
    env = Environment()
    ag = Agent(env)

    for i in range(int(60 / ag.dt)):
        ag.update()
    
    ag.plot_trajectory()

    data = pd.DataFrame(ag.history)
    os.makedirs("../data", exist_ok=True)
    data.to_csv("../data/out.csv")


if __name__ == '__main__':
    simulate()