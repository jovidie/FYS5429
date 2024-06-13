import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

import torch 
import torch.nn as nn 


# Vanilla RNN using parser dict or dataclass
class VanillaRNN(nn.Module):
    def __init__(self, args) -> None:
        super(VanillaRNN, self).__init__()
        self.n_inputs = args.n_inputs
        self.n_neurons = args.n_neurons
        self.n_outputs = args.n_outputs
        
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.device = args.device
        self.loss_func = args.loss_func

        self.encoder = nn.Linear(
                self.n_outputs, 
                self.n_neurons, 
                bias=False)
        self.rnn = nn.RNN(
                self.n_inputs, 
                self.n_neurons, 
                nonlinearity="relu", 
                bias=False, 
                batch_first=True)
        self.decoder = nn.Linear(
                self.n_neurons, 
                self.n_outputs, 
                bias=False)
        
    
    def forward(self, data):
        feats, x0 = data 
        h0 = self.encoder(x0[:, 0])
        out, _ = self.rnn(feats, h0[None])
        return out
    
    
    def predict(self, data):
        y_tilde = self.decoder(self.forward(data))
        return y_tilde
    

    def fit(self, train_data):
        self.train()
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.total_loss = np.zeros(self.n_epochs)
        n_data = len(train_data.dataset)

        progress = tqdm(range(self.n_epochs))
        for epoch in progress:
            train_loss = 0
            for x_batch, y_batch in train_data:
                self.optimizer.zero_grad()
                y_tilde = self.predict((x_batch, y_batch))
                # Calculate loss
                loss = self.loss_func(y_tilde[:, 0:-1], y_batch[:, 1:])
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.total_loss[epoch] = train_loss / n_data

            progress.set_description(f"Epoch [{epoch+1}/{self.n_epochs}]")
            progress.set_postfix(loss=self.total_loss[epoch])

        return self.total_loss  


    def test(self, test_data):
        self.eval()
        self.to(self.device)

        test_size = len(test_data.dataset)
        test_loss = []

        true = torch.zeros([test_size, self.seq_length, self.n_outputs], dtype=torch.float32)
        predicted = torch.zeros([test_size, self.seq_length, self.n_outputs], dtype=torch.float32)

        with torch.no_grad():
            for batch, (x_test, y_test) in enumerate(test_data):
                start_idx = batch*self.batch_size
                end_idx = start_idx + self.batch_size
                y_pred = self.predict((x_test, y_test))

                true[start_idx:end_idx] = y_test
                predicted[start_idx:end_idx] = y_pred

                loss = self.loss_func(y_pred[:, 0:-1], y_test[:, 1:])
                test_loss.append(loss.item())

        return test_loss, true, predicted


class NeuroRNN(nn.Module):
    def __init__(self, args) -> None:
        super(NeuroRNN, self).__init__()
        self.n_inputs = args.n_inputs
        self.n_gc = args.n_gc
        self.n_pc = args.n_pc
        self.n_outputs = args.n_outputs
        self.lr = args.lr

        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.loss_func = args.loss_func
        self.device = args.device

        # MEC -> Hippocampus Circuit
        self.mec_encoder = nn.Linear(
            self.n_outputs,
            self.n_gc,
            bias=False
        )
        self.ca1_encoder = nn.Linear(
            self.n_gc,
            self.n_pc,
            bias=False
        )
        self.mec = nn.RNN(
            self.n_inputs,
            self.n_gc,
            nonlinearity="relu",
            bias=False,
            batch_first=True
        )
        self.ca1 = nn.RNN(
            self.n_gc,
            self.n_pc,
            nonlinearity="relu",
            bias=False,
            batch_first=True
        )
        self.output = nn.Linear(
            self.n_pc,
            self.n_outputs,
            bias=False
        )



    def forward(self, data):
        vel, x0 = data 
        # init = torch.randn(self.batch_size, self.seq_length, self.n_gc)
        # print(f"Velocities: {vel.size()}")
        # print(f"Positions: {x0.size()}")
        # print(f"MEC encoder: {self.mec_encoder.parameters}")
        # print(f"MEC: {self.mec.parameters}")
        mec_h0 = self.mec_encoder(x0[:, 0])
        # print(f"MEC h0: {mec_h0.size()}")
        # print(f"CA1 encoder: {self.ca1_encoder.parameters}")
        ca1_h0 = self.ca1_encoder(mec_h0)
        # print(mec_h0[None].size())
        mec_out, mec_hn = self.mec(vel, mec_h0[None])
        ca1_out, ca1_hn = self.ca1(mec_out, ca1_h0[None])
        # print(f"Output CA1 weights: {ca1_hn.size()}")
        return ca1_out


    def predict(self, data):
        y_tilde = self.output(self.forward(data))
        return y_tilde

    
    def fit(self, train_data):
        self.train()
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.total_loss = np.zeros(self.n_epochs)
        n_data = len(train_data.dataset)

        progress = tqdm(range(self.n_epochs))
        for epoch in progress:
            train_loss = 0
            for x_batch, y_batch in train_data:
                self.optimizer.zero_grad()
                y_tilde = self.predict((x_batch, y_batch))

                loss = self.loss_func(y_tilde[:, 0:-1], y_batch[:, 1:])
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.total_loss[epoch] = train_loss / n_data
            
            progress.set_description(f"Epoch [{epoch+1}/{self.n_epochs}]")
            progress.set_postfix(loss=self.total_loss[epoch])

        return self.total_loss
    
    
    def test(self, test_data):
        self.eval()
        self.to(self.device)

        test_size = len(test_data.dataset)
        test_loss = []

        true = torch.zeros([test_size, self.seq_length, self.n_outputs], dtype=torch.float32)
        predicted = torch.zeros([test_size, self.seq_length, self.n_outputs], dtype=torch.float32)

        with torch.no_grad():
            for batch, (x_test, y_test) in enumerate(test_data):
                start_idx = batch*self.batch_size
                end_idx = start_idx + self.batch_size
                y_pred = self.predict((x_test, y_test))
                true[start_idx:end_idx] = y_test
                predicted[start_idx:end_idx] = y_pred

                loss = self.loss_func(y_pred[:, 0:-1], y_test[:, 1:])
                test_loss.append(loss.item())

        return test_loss, true, predicted


