# data_generation.py
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from .brownian import Brownian

def get_simple_data(mu=10, sigma=1, p=3, s=1.0002, length=12600):
    """
    Generate synthetic time-series with both 'process' and 'sensor' anomalies.

    :param mu: Base mean
    :param sigma: Base std
    :param p: Factor for process anomaly variance
    :param s: Factor for sensor anomaly drift
    :param length: Length of each channel timeseries
    :return:
        signal: list of raw signals for each of 20 channels
        train_list: normalized training data (20 channels)
        test_list: normalized test data (20 channels)
        train_trend_list, train_residual_list, val_trend_list, val_residual_list, test_trend_list, test_residual_list
    """

    sensor_channels = [1, 2]
    process_channels = [0, 2]
    signal = []

    # 1) Build raw signals
    for ch in range(20):
        data = np.zeros(length)
        t_s = 10400
        t_p1, t_p2 = 10200+500, 10200+1000
        mu_hist, sigma_hist = [], []

        for t in range(length):
            # process anomaly region => variance p if t in [t_p1, t_p2)
            if ch in process_channels and (t_p1 <= t < t_p2):
                var = p
            else:
                var = sigma

            # sensor drift => multiply mean by s if t >= t_s
            if ch in sensor_channels and t >= t_s:
                if t == t_s:
                    this_mu = mu
                else:
                    this_mu = s * mu_hist[-1]
            else:
                this_mu = mu

            mu_hist.append(this_mu)
            sigma_hist.append(var)
            data[t] = np.random.normal(this_mu, var)

        signal.append(data)

    # 2) Train/Val/Test split => normalize by train stats
    train_list, val_list, test_list = [], [], []
    for i in range(20):
        raw_train = signal[i][0:8200]
        raw_val   = signal[i][8200:10400]
        raw_test  = signal[i][10400:12600]

        mean_ = np.mean(raw_train)
        std_  = np.std(raw_train)

        train_list.append((raw_train - mean_) / std_)
        val_list.append((raw_val - mean_) / std_)
        test_list.append((raw_test - mean_) / std_)

    # 3) Decompose => trend and residual
    def _get_trend_resid(x):
        """
        Returns the trend and residual from seasonal_decompose. Ignores NaN.
        """
        decomp = seasonal_decompose(x, period=100, model='additive')
        trend = [a for a in decomp.trend if not np.isnan(a)]
        resid = [r for r in decomp.resid if not np.isnan(r)]
        return trend, resid

    train_trend_list, train_residual_list = [], []
    val_trend_list,   val_residual_list   = [], []
    test_trend_list,  test_residual_list  = [], []

    for i in range(20):
        train_trend, train_resid = _get_trend_resid(train_list[i])
        val_trend,   val_resid   = _get_trend_resid(val_list[i])
        test_trend,  test_resid  = _get_trend_resid(test_list[i])

        train_trend_list.append(train_trend)
        train_residual_list.append(train_resid)
        val_trend_list.append(val_trend)
        val_residual_list.append(val_resid)
        test_trend_list.append(test_trend)
        test_residual_list.append(test_resid)

    return (
        signal,
        train_list,
        test_list,
        train_trend_list,
        train_residual_list,
        val_trend_list,
        val_residual_list,
        test_trend_list,
        test_residual_list
    )


def get_exp_1_data(process_level, sensor_level, gamma, t=0.01):
    """
    Example of generating signals for Experiment 1 with user-specified
    process level, sensor level, gamma, etc.
    This is just a placeholder for your specialized logic from the notebook.
    """
    # Copy the relevant code from your "Exp 1" function, or combine with Brownian, etc.
    # ...
    pass


def get_exp_2_data(...):
    """
    Similarly define for Experiment 2
    """
    pass


def get_exp_3_data(...):
    """
    Define for Experiment 3
    """
    pass


def get_exp_4_1_data(...):
    """
    ...
    """
    pass


def get_exp_4_2_data(...):
    """
    ...
    """
    pass
