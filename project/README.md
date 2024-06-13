# Recurrent Neural Network and Predicting Trajectories of 'Rats in a Box'

The aim of this project is to explore recurrent neural network, and the effect of different architectures, in predicting trajectories given various input features and sequence lengths.

## Repo structure
```bash
.
├── code
│   ├── experiments.py
│   ├── generate_data.py
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── data
├── latex
│   ├── figures
│   ├── sections
│   ├── main.tex
│   └── references.bib
├── environment.yml  
└── README.md
```


## Requirements
To run the experiments, create conda environment from file
```bash
$ conda env create -f environment.yml
```
or install the following requirements
- python >= 3.11
- pip
- numpy
- matplotlib
- torch
- torcheval
- scikit-learn
- ratinabox
- seaborn
- tqdm


## Run experiments
Run all experiments from `code/` as
```bash
$ python main.py
```