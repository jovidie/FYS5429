import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length, batch_size):
        super(VanillaRNN, self).__init__()
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