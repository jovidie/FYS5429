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
        # Could put this into a helper function or use encoder/decoder
        vel, x0 = data 
        h0 = self.input_layer(x0[:, 0])
        # print(h0[None].size())
        out, _ = self.rnn(vel, h0[None])
        return out

    def predict(self, data):
        y_tilde = self.output_layer(self.forward(data))
        return y_tilde
    
    def loss_function(self):
        # Define loss function which ensures conformal isometry?
        # Not needed for project-1
        pass