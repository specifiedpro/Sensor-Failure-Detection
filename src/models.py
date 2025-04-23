# models.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint

def build_trend_model(input_shape=(100, 20)):
    """
    Build an autoencoder for trend data.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Normalization(axis=-1),
        layers.Dense(32, activation='LeakyReLU'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='LeakyReLU'),
        layers.Dense(8, activation='LeakyReLU'),
        layers.Dense(4, activation='LeakyReLU'),
        layers.Dense(4, activation='LeakyReLU'),
        layers.Dense(8, activation='LeakyReLU'),
        layers.Dense(16, activation='LeakyReLU'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='LeakyReLU'),
        layers.Dense(input_shape[1])  # => output shape is (None, 100, 20) => the last dimension is 20
    ])
    model.compile(loss='mae', optimizer='adam')
    return model


def build_residual_model(input_shape=(100, 20)):
    """
    Build an autoencoder for residual data.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Normalization(axis=-1),
        layers.Dense(32, activation='LeakyReLU'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='LeakyReLU'),
        layers.Dense(8, activation='LeakyReLU'),
        layers.Dense(4, activation='LeakyReLU'),
        layers.Dense(4, activation='LeakyReLU'),
        layers.Dense(8, activation='LeakyReLU'),
        layers.Dense(16, activation='LeakyReLU'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='LeakyReLU'),
        layers.Dense(input_shape[1])
    ])
    model.compile(loss='mae', optimizer='adam')
    return model


def train_autoencoder(model, X_train, X_val, filepath=None, epochs=25, retrain=False):
    """
    :param model: compiled model
    :param X_train: shape=(?, 100, 20)
    :param X_val: shape=(?, 100, 20)
    :param filepath: If not None, path to save best weights
    :param epochs: number of training epochs
    :param retrain: if True, we skip the validation-based checkpoint logic 
    :return: trained model
    """
    callbacks = []
    if filepath and not retrain:
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                     verbose=0, save_best_only=True,
                                     save_weights_only=True, mode='min')
        earlystop = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.01, patience=5, mode='min')
        callbacks = [earlystop, checkpoint]

        model.fit(X_train, X_train,
                  validation_data=(X_val, X_val),
                  epochs=epochs,
                  verbose=0,
                  callbacks=callbacks)
    else:
        # If retrain = True, we might just combine X_train+X_val and train further
        model.fit(
            np.concatenate((X_train, X_val)),
            np.concatenate((X_train, X_val)),
            epochs=epochs,
            verbose=0
        )
    return model
