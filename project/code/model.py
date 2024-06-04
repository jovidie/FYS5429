import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader

from utils import load_dataset, vel_data, VanillaRNNargs, plot_theme


class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length, batch_size):
        super(BasicRNN, self).__init__()
        # Input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.batch_size = batch_size

        # Network architecture
        # Use "encoder" and "decoder" to handle weights for input-hidden and hidden-hidden
        self.input_layer = nn.Linear(in_features=self.input_size,
                                     out_features=hidden_size, 
                                     bias=False)
        self.rnn = nn.RNN(input_size=self.input_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=1,
                          nonlinearity="relu",
                          bias=False,
                          batch_first=True)
        self.output_layer = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.output_size,
                                      bias=False)       


    def forward(self, data):
        # Give a starting position to initiate the hidden state
        vel, x0 = data 
        h0 = self.input_layer(x0[:, 0])
        # print(h0[None].size())
        out, _ = self.rnn(vel, h0[None])
        return out
    
    def forward_states(self, data):
        # Give a starting position to initiate the hidden state
        v, x0 = data 
        h0 = self.input_layer(x0[:, 0])
        # print(h0[None].size())
        out, states = self.rnn(v, h0[None])
        print(states.shape)
        return out, states

    # Have to consider the case where the network is only predicitng, and not 
    # training. Should separate the method into training and predicting?
    def predict(self, data):
        y_tilde = self.output_layer(self.forward(data))
        return y_tilde
    
    def train(self, learning_rate, num_epochs, data_loader, verbose=False):
        # Loss and optimizer
        self.loss_f = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # Get the predicted element equivalent to model(x_batch)[:, 0]
        for epoch in range(num_epochs):
            for x_batch, y_batch in data_loader:
                # Get the predicted element equivalent to model(x_batch)[:, 0]
                y_tilde = self.predict((x_batch, y_batch))
                # Calculate loss
                loss = self.loss_f(y_tilde[:, 0:-1], y_batch[:, 1:])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if verbose is True and epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss {loss.item():.6f}")
    
    def loss_function(self):
        # Define loss function which ensures conformal isometry?
        # Not needed for project-1
        pass

    # Consider implementing a separate method or class for training?



# Vanilla RNN using parser dict
class VanillaRNN(nn.Module):
    def __init__(self, args) -> None:
        super(VanillaRNN, self).__init__()
        self.n_inputs = args.n_inputs
        self.n_neurons = args.n_neurons
        self.n_outputs = args.n_outputs
        
        # self.seq_length = args.seq_length
        # self.batch_size = args.batch_size
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
    

    def fit(self, train_data, verbose=False):
        self.train()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.total_loss = np.zeros(self.n_epochs)
        n_data = len(train_data)
        for epoch in range(self.n_epochs):
            avg_loss = 0
            for batch, (x_batch, y_batch) in enumerate(train_data):
                self.optimizer.zero_grad()
                y_tilde = self.predict((x_batch, y_batch))
                # Calculate loss
                loss = self.loss_func(y_tilde[:, 0:-1], y_batch[:, 1:])
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()

            self.total_loss[epoch] = avg_loss / n_data
            if verbose is True and epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss {self.total_loss[epoch]:.6f}")
        return self.total_loss
    

    def test(self, test_data):
        self.eval()
        self.to(self.device)

        test_loss = 0
        test_size = len(test_data)
        with torch.no_grad():
            for batch, (x_test, y_test) in enumerate(test_data):
                y_pred = self.predict((x_test, y_test))
                loss = self.loss_func(y_pred[:, 0:-1], y_test[:, 1:])
                test_loss += loss.item()
        test_loss /= test_size
        return test_loss

# Finish RNN to use in project, should be a general model in terms of input
class SimpleRNN(nn.Module):
    def __init__(
            self, 
            n_inputs=2,
            n_neurons=128,
            n_outputs=2,
            learning_rate=0.0001,
            seq_length=20,
            n_epochs=50,
            batch_size=10
        ):

        super(SimpleRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.seq_length = seq_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size

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
        out, hn = self.rnn(feats, h0[None])
        return out
    
    def predict(self, data):
        y_tilde = self.decoder(self.forward(data))
        return y_tilde
    

    def compute_loss(self, y, y_tilde):
        pass
    

    def train(self, data, verbose=False):
        self.loss = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        self.total_loss = np.ndarray((self.n_epochs, len(data)))
        # self.total_eps = []

        # print(len(data), len(data.dataset))

        for epoch in range(self.n_epochs):
            for batch, (x_batch, y_batch) in enumerate(data):
                # print(f"Start epoch {epoch}, idx {batch}")
                self.optimizer.zero_grad()

                y_tilde = self.predict((x_batch, y_batch))

                loss = self.loss(y_tilde[:, 0:-1], y_batch[:, 1:])
                # loss_epoch.append(loss.item())
                # err_epoch.append(torch.sqrt(((y_tilde[:, 0:-1] - y_batch[:, 1:])**2).sum(-1)).mean())

                loss.backward()
                self.optimizer.step()

                # total_acc += ((y_tilde >= 0.5).float() == y_batch).float().sum().item()
                # total_loss += loss.item() * y_batch.size(0)
                self.total_loss[epoch, batch] = loss.item()
            # print(self.state_dict().keys())
            # > ['mec_encoder.weight', 'ca1_encoder.weight', 'mec.weight_ih_l0', 'mec.weight_hh_l0', 'ca1.weight_ih_l0', 'ca1.weight_hh_l0', 'output.weight']
            
            # self.total_loss.append(sum(loss_epoch) // len(loss_epoch))

            if verbose is True and epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss {loss.item():.6f}")
        
        # print(np.sum(self.total_loss, axis=1))
        return np.mean(self.total_loss)


class NeuroRNN(nn.Module):
    def __init__(self, args) -> None:
        super(NeuroRNN, self).__init__()
        self.n_gc = args.n_gc
        self.n_pc = args.n_pc
        self.n_inputs = args.n_inputs
        self.n_outputs = args.n_outputs
        self.lr = args.learning_rate

        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        # MEC -> Hippocampus Circuit
        self.mec_encoder = nn.Linear(
            self.n_inputs,
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
    
    def train(self, data_loader, verbose=False):
        # Loss and optimizer
        self.loss = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Get the predicted element equivalent to model(x_batch)[:, 0]
        for epoch in range(self.n_epochs):
            for x_batch, y_batch in data_loader:
                self.optimizer.zero_grad()

                y_tilde = self.predict((x_batch, y_batch))

                loss = self.loss(y_tilde[:, 0:-1], y_batch[:, 1:])
                loss.backward()
                self.optimizer.step()
            
            # print(self.state_dict().keys())
            # > ['mec_encoder.weight', 'ca1_encoder.weight', 'mec.weight_ih_l0', 'mec.weight_hh_l0', 'ca1.weight_ih_l0', 'ca1.weight_hh_l0', 'output.weight']
            
            if verbose is True and epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss {loss.item():.6f}")


if __name__ == '__main__':
    plot_theme()
    args = VanillaRNNargs(n_epochs=200)
    model = VanillaRNN(args)

    n_trajectories = 100
    seq_length = 20
    batch_size = 10
    test_size = 30

    train_data = load_dataset(n_trajectories, seq_length)
    dl = DataLoader(train_data, batch_size)

    loss = model.fit(dl, True)
    print(np.mean(loss))

    test_data = vel_data(test_size, seq_length)
    test_dl = DataLoader(test_data, test_size)

    # x_test, y_test = next(iter(test_dl))
    # y_tilde = model.predict((x_test, y_test))

    # pred = y_tilde.detach().numpy()
    # true = y_test.detach().numpy()
    
    test_loss = model.test(test_dl)
    print(test_loss)
    # plt.plot(loss)
    # plt.yscale("log")
    # plt.show()
    # colors = sns.color_palette("mako", n_colors=test_size)
    # fig, ax = plt.subplots(figsize=(6, 6))
    # # ax.set_xlim(0, 1)
    # # ax.set_ylim(0, 1)
    # print(pred.shape, true.shape)
    # for i in range(test_size):
    #     ax.plot(pred[i, :, 0], pred[i, :, 1], color=colors[i])
    #     ax.plot(true[i, :, 0], true[i, :, 1], "--", color=colors[i])
    
    # # fig.savefig("../latex/figures/test_model.pdf", bbox_inches="tight")
    # plt.show()

