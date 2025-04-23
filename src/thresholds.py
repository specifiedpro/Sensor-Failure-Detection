# thresholds.py
import numpy as np
from tensorflow.keras import Model

def compute_thresholds(model, X_train, n_channels=20, z=4.0):
    """
    Compute channel-wise reconstruction thresholds based on training distribution.

    :param model: trained autoencoder
    :param X_train: shape=(N, 100, 20)
    :param n_channels: number of channels
    :param z: how many std dev away from mean
    :return: list of length n_channels, each threshold
    """
    train_pred = model.predict(X_train)
    # train_mae_loss => shape (N, 100, 20) => reduce mean over axis=1 => shape (N, 20)
    train_mae_loss = np.mean(np.abs(train_pred - X_train), axis=1)

    thresholds = []
    for ch in range(n_channels):
        mean_ = np.mean(train_mae_loss[:, ch])
        std_  = np.std(train_mae_loss[:, ch])
        channel_threshold = mean_ + z * std_
        thresholds.append(channel_threshold)
    return thresholds


def compute_channel_loss(model, X_test, n_channels=20):
    """
    Return list of length n_channels => each is the reconstruction error
    for that channel across all test windows.

    :param model: trained autoencoder
    :param X_test: shape=(N, 100, 20)
    :return: channel_loss, a list of lists [ch][time_slices]
    """
    test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(test_pred - X_test), axis=1)  # shape (N, 20)
    channel_loss = []
    for ch in range(n_channels):
        channel_loss.append( test_mae_loss[:, ch].tolist() )
    return channel_loss
