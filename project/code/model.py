import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader


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
        self.n_layers = args.n_layers
        self.n_outputs = args.n_outputs
        self.learning_rate = args.learning_rate
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.init = nn.Linear(
                self.n_outputs, 
                self.n_neurons, 
                bias=False)
        # self.input_layer = nn.Linear(
        #         self.n_inputs, 
        #         self.n_neurons, 
        #         bias=False)
        self.rnn = nn.RNN(
                self.n_inputs, 
                self.n_neurons, 
                nonlinearity="relu", 
                bias=False, 
                batch_first=True)
        self.output_layer = nn.Linear(
                self.n_neurons, 
                self.n_outputs, 
                bias=False)
        
    
    def forward(self, data):
        vel, x0 = data 

        h0 = self.encoder(x0[:, 0])
        # print(h0[None].size())
        out, _ = self.rnn(vel, h0[None])
        return out
    
    def predict(self, data):
        y_tilde = self.output_layer(self.forward(data))
        return y_tilde
    
    def train(self, data_loader, verbose=False):
        # Loss and optimizer
        self.loss = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Get the predicted element equivalent to model(x_batch)[:, 0]
        for epoch in range(self.n_epochs):
            for x_batch, y_batch in data_loader:
                self.optimizer.zero_grad()
                # Get the predicted element equivalent to model(x_batch)[:, 0]
                y_tilde = self.predict((x_batch, y_batch))
                # Calculate loss
                loss = self.loss(y_tilde[:, 0:-1], y_batch[:, 1:])
                loss.backward()
                self.optimizer.step()
            
            if verbose is True and epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss {loss.item():.6f}")


# Finish RNN to use in project, should be a general model in terms of input
class PathRNN(VanillaRNN):
    def __init__(self, args):
        super(PathRNN, self).__init__(args)
        self.learning_rate = 0.02
        self.pre_input = nn.Linear(
                self.n_inputs, 
                self.n_outputs, 
                bias=False)
        
        self.input_layer = nn.Linear(
                self.n_outputs, 
                self.n_neurons, 
                bias=False)
        
        self.rnn = nn.RNN(
                self.n_outputs, 
                self.n_neurons, 
                nonlinearity="relu", 
                bias=False, 
                batch_first=True)
        
    def forward(self, data):
        feats, x0 = data 
        h0 = self.input_layer(x0[:, 0])
        # print(h0[None].size())
        vel = self.pre_input(feats)
        out, _ = self.rnn(vel, h0[None])
        return out
    


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