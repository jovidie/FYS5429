import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import VanillaRNN, NeuroRNN
from utils import VanillaRNNargs, NeuroRNNargs, plot_theme, load_data, plot_trajectories


def experiment_1():
    n_trajectories = 100
    seq_length = 20
    batch_size = 20

    epochs = np.array([100, 1000, 10000])
    learning_rates = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    train_loss = np.zeros((epochs.size, learning_rates.size, 2))

    data = load_data(n_trajectories, seq_length)
    dl = DataLoader(data, batch_size)

    for i, n_epochs in enumerate(epochs):
        for j, lr in enumerate(learning_rates):
            args = VanillaRNNargs(n_epochs=n_epochs, lr=lr, batch_size=batch_size)
            model = VanillaRNN(args)
            out = model.fit(dl, verbose=True)
            train_loss[i, j, :] = [np.mean(out), np.std(out)]
    
    loss = train_loss[:, :, 0]

    i, j = np.unravel_index(np.argmin(loss, axis=None), loss.shape)
    print(f"Optimal parameters at indices {i, j}:")
    print(f"\tEpoch = {epochs[i]}")
    print(f"\tLearning rate = {learning_rates[j]}")


def experiment_2():
    n_trajectories = 100
    seq_length = 20
    batch_size = 20

    train_size = int(n_trajectories*0.8)
    test_size = n_trajectories - train_size

    data_square = load_data(type="env", env="square")
    data_circle = load_data(type="env", env="circle")
    data_rectangle = load_data(type="env", env="rectangle")

    train_data_square, test_data_square = random_split(data_square, [train_size, test_size])
    train_data_circle, test_data_circle = random_split(data_circle, [train_size, test_size])
    train_data_rectangle, test_data_rectangle = random_split(data_rectangle, [train_size, test_size])

    train_data = [train_data_square, train_data_circle, train_data_rectangle]
    test_data = [test_data_square, test_data_circle, test_data_rectangle]

    args = VanillaRNNargs()
    square = VanillaRNN(args)
    circle = VanillaRNN(args)
    rectangle = VanillaRNN(args)

    train_loss = []
    test_loss = []

    for model in [square, circle, rectangle]:
        for i in range(3):
            dl_train = DataLoader(train_data[i], batch_size)
            loss = model.fit(dl_train)
            train_loss.append([np.mean(loss), np.std(loss)])

            dl_test = DataLoader(test_data[i])
            loss, true, pred = model.test(dl_test)
            test_loss.append([np.mean(loss), np.std(loss)])

    print("Mean Training loss and std: ")
    print(train_loss)

    print("Mean test loss and std: ")
    print(test_loss)

    # Training loss: 
    # [7.066814490062398e-06, 1.2305475479013239e-06, 3.668639920524583e-06, 8.011709573589486e-06, 1.145748810690037e-06, 4.963331519767565e-06, 7.041920494917236e-06, 1.187006197378082e-06, 4.088021100088213e-06]
    # Test loss: 
    # [0.00021785353683299036, 9.734457044032751e-05, 0.00028702012223220664, 0.00024916872371250065, 0.00020135033696533355, 0.0003445988457315252, 0.0001952258165601961, 5.844646151444977e-05, 0.0005142701649219816]


def experiment_3():
    n_trajectories = 100
    batch_size = 20
    n_epochs = 1000

    train_size = int(n_trajectories*0.8)
    test_size = n_trajectories - train_size

    data_exp = load_data(type="experimental")
    data_syn = load_data(type="features")

    train_data_syn, test_data_syn = random_split(data_syn, [train_size, test_size])
    train_data_exp, test_data_exp = random_split(data_exp, [train_size, test_size])

    args = VanillaRNNargs(n_epochs=n_epochs)
    model = VanillaRNN(args)

    dl_train = DataLoader(train_data_syn, batch_size)
    loss = model.fit(dl_train)
    train_loss = [np.mean(loss), np.std(loss)]

    dl_test_syn = DataLoader(test_data_syn)
    test_loss_syn, true, pred = model.test(dl_test_syn)
    test_loss_syn = [np.mean(test_loss_syn), np.std(test_loss_syn)]
    plot_trajectories(test_size, true, pred, save=True, filename=f"predict_synthetic")

    dl_test_exp = DataLoader(test_data_exp)
    test_loss_exp, true, pred = model.test(dl_test_exp)
    test_loss_exp = [np.mean(test_loss_exp), np.std(test_loss_exp)]
    plot_trajectories(test_size, true, pred, save=True, filename=f"predict_experimental")

    print(f"Mean train loss {train_loss[0]:.6f} and std {train_loss[1]:.6f}")
    print(f"Mean test loss {test_loss_syn[0]:.6f} and std {test_loss_syn[1]:.6f}, synthetic")
    print(f"Mean test loss {test_loss_exp[0]:.6f} and std {test_loss_exp[1]:.6f}, experimental")

    # Mean train loss 0.000133 and std 0.000915
    # Mean test loss 0.000130 and std 0.000345, synthetic
    # Mean test loss 0.199068 and std 0.220495, experimental


def experiment_4():
    trajectories = [100, 1000]
    sequences = [20, 30, 40, 50, 60, 70, 80]
    batch_size = 20
    n_epochs = 1000

    train_loss = np.zeros((len(trajectories), len(sequences), 2))
    test_loss = np.zeros((len(trajectories), len(sequences), 2))

    for i, n_trajectories in enumerate(trajectories):
        train_size = int(n_trajectories*0.8)
        test_size = n_trajectories - train_size

        for j, seq_length in enumerate(sequences):
            data = load_data(
                n_trajectories=n_trajectories,
                seq_length=seq_length, 
                type="features", 
                features="vel"
            )
            train, test = random_split(data, [train_size, test_size])

            args = VanillaRNNargs(
                seq_length=seq_length, 
                n_epochs=n_epochs,
                batch_size=batch_size
            )
            model = VanillaRNN(args)

            dl_train = DataLoader(train, batch_size)
            loss = model.fit(dl_train)
            train_loss[i, j, :] = np.mean(loss), np.std(loss)

            dl_test = DataLoader(test)
            loss, true, pred = model.test(dl_test)
            test_loss[i, j, :] = np.mean(loss), np.std(loss)

    print(f"Mean train loss {train_loss[:, :, 0]} and std {train_loss[:, :, 1]}")
    print(f"Mean test loss {test_loss[:, :, 0]} and std {test_loss[:, :, 1]}")

#     Train loss: [[1.18001702e-04 2.43397017e-04 4.87326180e-04 9.60213101e-04 1.37277395e-03 3.05756983e-03 2.50100700e-03]
#                   [1.83582800e-05 3.55920562e-05 8.28945633e-05 1.44345430e-04 1.72907279e-04 4.55887238e-04 2.48063939e-04]]
#     Test loss: [[1.50717824e-05 3.41076171e-05 4.76761197e-05 8.76334205e-05 1.17770873e-04 1.76274814e-03 2.32924093e-03]
#                   [8.74685320e-06 9.68198524e-07 1.20770481e-06 9.21109131e-07 1.70198337e-06 2.86765186e-06 4.19489197e-06]]

    # Mean train loss [[1.16707231e-04 1.93873333e-04 8.03189304e-04 8.11919989e-04
    # 8.38920623e-04 1.01402638e-03 2.15908669e-03]
    # [1.88621770e-05 3.12890478e-05 7.41834040e-05 1.46316779e-04
    # 1.45070726e-04 4.45271542e-04 1.31693036e-03]] and std [[0.00079835 0.00103607 0.00273543 0.00227416 0.00238702 0.00675562
    # 0.00416685]
    # [0.00024064 0.00033613 0.00057546 0.00090504 0.00079523 0.00142942
    # 0.00206527]]
    # Mean test loss [[1.57085616e-05 2.14621886e-04 4.36829319e-04 5.38065207e-04
    # 7.32046580e-03 1.53301139e-03 1.62235412e-03]
    # [3.39667467e-05 3.74550235e-06 8.31651935e-06 6.03106860e-06
    # 2.80487191e-04 9.77303463e-06 6.07143891e-03]] and std [[2.00526828e-05 4.62499425e-04 3.59406279e-04 4.85353345e-04
    # 4.52852329e-03 1.14716725e-03 1.17814413e-03]
    # [2.81555207e-05 4.05650734e-06 7.95866862e-06 5.86665317e-06
    # 2.31927361e-04 1.16059581e-05 7.56873286e-03]]

def experiment_5():
    trajectories = [100, 1000]
    features = ["vel", "vel_head", "vel_head_rot", "vel_head_rot_dist"]
    n_inputs = [2, 4, 5, 6]
    seq_length = 20
    batch_size = 20
    n_epochs = 1000

    train_loss = np.zeros((len(trajectories), len(features), 2))
    test_loss = np.zeros((len(trajectories), len(features), 2))

    for i, n_trajectories in enumerate(trajectories):
        train_size = int(n_trajectories*0.8)
        test_size = n_trajectories - train_size

        for j, feature in enumerate(features):
            data = load_data(
                n_trajectories=n_trajectories,
                type="features", 
                features=feature
            )
            train, test = random_split(data, [train_size, test_size])

            args = VanillaRNNargs(
                n_inputs=n_inputs[j],
                seq_length=seq_length, 
                n_epochs=n_epochs,
                batch_size=batch_size
            )
            model = VanillaRNN(args)

            dl_train = DataLoader(train, batch_size)
            loss = model.fit(dl_train)
            train_loss[i, j, :] = np.mean(loss), np.std(loss)

            dl_test = DataLoader(test)
            loss, true, pred = model.test(dl_test)
            test_loss[i, j, :] = np.mean(loss), np.std(loss)

    print(f"Mean train loss {train_loss[:, :, 0]} and std {train_loss[:, :, 1]}")
    print(f"Mean test loss {test_loss[:, :, 0]} and std {test_loss[:, :, 1]}")

# Train loss: [[1.42866112e-04 1.54621587e-04 1.18029548e-04 1.70700872e-04]
#             [2.22973723e-05 2.14224739e-05 2.39684707e-05 2.55888408e-05]]
# Test loss: [[8.13066086e-05 6.03779554e-06 6.01602253e-05 5.35132538e-05]
#             [4.37267158e-07 3.08092228e-07 6.11240361e-06 6.91885325e-06]]


def experiment_6():
    trajectories = [100, 1000]
    features = ["vel", "vel_head", "vel_head_rot", "vel_head_rot_dist"]
    n_inputs = [2, 4, 5, 6]
    seq_length = 20
    batch_size = 20
    n_epochs = 1000

    train_loss_vanilla = np.zeros((len(trajectories), len(features), 2))
    test_loss_vanilla = np.zeros((len(trajectories), len(features), 2))

    train_loss_neuro = np.zeros((len(trajectories), len(features), 2))
    test_loss_neuro = np.zeros((len(trajectories), len(features), 2))

    for i, n_trajectories in enumerate(trajectories):
        train_size = int(n_trajectories*0.8)
        test_size = n_trajectories - train_size

        for j, feature in enumerate(features):
            data = load_data(
                n_trajectories=n_trajectories,
                type="features", 
                features=feature
            )
            train, test = random_split(data, [train_size, test_size])

            args_vanilla = VanillaRNNargs(
                n_inputs=n_inputs[j],
                seq_length=seq_length, 
                n_epochs=n_epochs,
                batch_size=batch_size
            )
            args_neuro = NeuroRNNargs(
                n_inputs=n_inputs[j],
                seq_length=seq_length, 
                n_epochs=n_epochs,
                batch_size=batch_size
            )
            model_vanilla = VanillaRNN(args_vanilla)
            model_neuro = NeuroRNN(args_neuro)

            dl_train = DataLoader(train, batch_size)

            loss = model_vanilla.fit(dl_train)
            train_loss_vanilla[i, j, :] = np.mean(loss), np.std(loss)

            loss = model_neuro.fit(dl_train)
            train_loss_neuro[i, j, :] = np.mean(loss), np.std(loss)

            dl_test = DataLoader(test)

            loss, true, pred = model_vanilla.test(dl_test)
            test_loss_vanilla[i, j, :] = np.mean(loss), np.std(loss)
            loss, true, pred = model_neuro.test(dl_test)
            test_loss_neuro[i, j, :] = np.mean(loss), np.std(loss)

    print(f"Mean train loss vanilla {train_loss_vanilla[:, :, 0]} and std {train_loss_vanilla[:, :, 1]}") 
    print(f"Mean train loss neuro {train_loss_neuro[:, :, 0]} and std {train_loss_neuro[:, :, 1]}")
    print(f"Mean test loss vanilla {test_loss_vanilla[:, :, 0]} and std {test_loss_vanilla[:, :, 1]}")
    print(f"Mean test loss neuro {test_loss_neuro[:, :, 0]} and std {test_loss_neuro[:, :, 1]}")

# Train loss vanilla: [[1.15042700e-04 1.59911996e-04 1.52120509e-04 1.73859866e-04]
#                     [2.05384507e-05 2.43689052e-05 2.38358216e-05 2.43463378e-05]]
# Train loss neuro: [[1.00883413e-04 1.16579270e-04 1.17800835e-04 1.27882833e-04]
#                     [1.95908618e-05 1.43139137e-05 1.88605529e-05 2.18612601e-05]]
# Test loss vanilla: [[1.43100900e-05 2.46664946e-05 3.94575967e-05 9.25327069e-05]
#                     [1.13998792e-05 1.75406401e-07 8.28989041e-07 1.78903896e-06]]
# Test loss neuro: [[1.31720544e-06 3.65130254e-05 3.32964322e-05 3.84493731e-05]
#                     [1.89451349e-07 3.98076561e-06 1.58805975e-06 2.41200501e-06]]


    # Mean train loss vanilla [[1.50883961e-04 1.37259765e-04 1.58551048e-04 1.37377151e-04]
    # [2.04771456e-05 1.89890156e-05 2.20979584e-05 2.39434945e-05]] and std [[0.00089226 0.0008701  0.0008533  0.00069432]
    # [0.00028991 0.00021806 0.00027954 0.00032174]]
    # Mean train loss neuro [[9.01678954e-05 1.16706152e-04 1.21085828e-04 1.11050758e-04]
    # [1.56228919e-05 1.98963057e-05 1.97210819e-05 2.27914928e-05]] and std [[0.00074347 0.00079995 0.00067742 0.00070624]
    # [0.00020964 0.00022006 0.00020492 0.00030486]]
    # Mean test loss vanilla [[2.11340874e-04 7.30855159e-05 8.58813224e-04 2.94252326e-03]
    # [7.69873394e-05 1.39497357e-04 7.37540667e-06 1.75465310e-05]] and std [[1.29464961e-04 1.14732162e-04 1.38418192e-03 6.81152448e-03]
    # [5.87036943e-05 9.73848562e-05 9.40815377e-06 3.52716921e-05]]
    # Mean test loss neuro [[3.65248818e-04 5.84319026e-04 6.76866868e-04 4.95865762e-04]
    # [1.87319294e-05 1.10728683e-05 3.11810253e-05 1.81933266e-05]] and std [[2.64162883e-04 5.28126247e-04 6.91431940e-04 5.37840709e-04]
    # [1.40816516e-05 6.57685376e-06 2.90191213e-05 1.34580080e-05]]

