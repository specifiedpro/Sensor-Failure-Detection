# shaping.py
import numpy as np

def train_shape(train_trend_list, train_residual_list, window_size=100, overlap=50):
    """
    Prepare training windows from the trend/residual data.

    :param train_trend_list: list of length=20 each containing the trend array
    :param train_residual_list: list of length=20 each containing the residual array
    :param window_size: window size
    :param overlap: overlap size
    :return: (X_train_trend, X_train_residual) as np.array of shape [N, window_size, 20]
    """
    training_final_trend = []
    training_final_resid = []

    stride = window_size - overlap
    n_wins = (len(train_trend_list[0]) // stride) - 1  # as in your code

    for w_idx in range(n_wins):
        block_trend = []
        block_resid = []
        for t in range(window_size):
            row_trend = []
            row_resid = []
            for ch in range(20):
                val_trend = train_trend_list[ch][w_idx*stride + t]
                val_resid = train_residual_list[ch][w_idx*stride + t]
                row_trend.append(val_trend)
                row_resid.append(val_resid)
            block_trend.append(row_trend)
            block_resid.append(row_resid)
        training_final_trend.append(block_trend)
        training_final_resid.append(block_resid)

    X_train_trend = np.array(training_final_trend)
    X_train_residual = np.array(training_final_resid)
    return X_train_trend, X_train_residual


def val_shape(val_trend_list, val_residual_list, window_size=100, overlap=50):
    """
    Same shaping approach for validation data.
    """
    val_final_trend = []
    val_final_resid = []
    stride = window_size - overlap
    n_wins = (len(val_trend_list[0]) // stride) - 1
    for w_idx in range(n_wins):
        block_trend = []
        block_resid = []
        for t in range(window_size):
            row_trend = []
            row_resid = []
            for ch in range(20):
                row_trend.append(val_trend_list[ch][w_idx*stride + t])
                row_resid.append(val_residual_list[ch][w_idx*stride + t])
            block_trend.append(row_trend)
            block_resid.append(row_resid)
        val_final_trend.append(block_trend)
        val_final_resid.append(block_resid)

    X_val_trend = np.array(val_final_trend)
    X_val_residual = np.array(val_final_resid)
    return X_val_trend, X_val_residual


def test_shape(test_trend_list, test_residual_list, window_size=100, overlap=50):
    """
    Same shaping approach for test data.
    """
    test_final_trend = []
    test_final_resid = []
    stride = window_size - overlap
    n_wins = (len(test_trend_list[0]) // stride) - 1
    for w_idx in range(n_wins):
        block_trend = []
        block_resid = []
        for t in range(window_size):
            row_trend = []
            row_resid = []
            for ch in range(20):
                row_trend.append(test_trend_list[ch][w_idx*stride + t])
                row_resid.append(test_residual_list[ch][w_idx*stride + t])
            block_trend.append(row_trend)
            block_resid.append(row_resid)
        test_final_trend.append(block_trend)
        test_final_resid.append(block_resid)

    X_test_trend = np.array(test_final_trend)
    X_test_residual = np.array(test_final_resid)
    return X_test_trend, X_test_residual
