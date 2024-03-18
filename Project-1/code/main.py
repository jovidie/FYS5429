import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader

from model import VanillaRNN
from create_data import generate_trajectories


def main():
    learning_rate = 0.01
    num_epochs = 500
    input_size = 2
    hidden_size = 10
    output_size = 2
    batch_size = 10
    seq_length = 10

    rnn = VanillaRNN(input_size, hidden_size, output_size, seq_length, batch_size)
    loss_f = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    joint_data = generate_trajectories(batch_size, seq_length)
    data_loader = DataLoader(joint_data, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for x_batch, y_batch in data_loader:
            # Get the predicted element equivalent to model(x_batch)[:, 0]
            y_tilde = rnn.predict((x_batch, y_batch))
            # plt.plot(y_batch[:,0,:])
            # Calculate the loss using the correct dimension
            # loss = loss_f(y_tilde, y_batch)
            loss = loss_f(y_tilde[:, 0:-1], y_batch[:, 1:])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss {loss.item():.6f}")
    
    pred_traj = y_tilde.detach().numpy()
    true_traj = y_batch.detach().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(batch_size):
        ax.plot(pred_traj[i, :, 0], pred_traj[i, :, 1], "--")
        ax.plot(true_traj[i, :, 0], true_traj[i, :, 1])
    
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(2024)
    main()



