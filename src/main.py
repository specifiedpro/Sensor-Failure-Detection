# main.py
import numpy as np
from .data_generation import get_simple_data
from .shaping import train_shape, val_shape, test_shape
from .models import build_trend_model, build_residual_model, train_autoencoder
from .thresholds import compute_thresholds, compute_channel_loss

def run_demo():
    """
    A small pipeline example that shows how to put everything together
    using get_simple_data(...).
    """

    # 1) Generate data
    (
        signal,
        train_list,
        test_list,
        train_trend_list,
        train_residual_list,
        val_trend_list,
        val_residual_list,
        test_trend_list,
        test_residual_list
    ) = get_simple_data(mu=10, sigma=1, p=3, s=1.0002, length=12600)

    # 2) Shaping
    X_train_trend, X_train_resid = train_shape(train_trend_list, train_residual_list)
    X_val_trend,   X_val_resid   = val_shape(val_trend_list, val_residual_list)
    X_test_trend,  X_test_resid  = test_shape(test_trend_list, test_residual_list)

    # 3) Build & train the models
    trend_model = build_trend_model()
    residual_model = build_residual_model()

    trend_model = train_autoencoder(trend_model, X_train_trend, X_val_trend,
                                    filepath='trend_model.hdf5',
                                    epochs=25, retrain=False)
    residual_model = train_autoencoder(residual_model, X_train_resid, X_val_resid,
                                       filepath='residual_model.hdf5',
                                       epochs=25, retrain=False)

    # 4) Compute thresholds (use combined train+val for final threshold calc)
    X_comb_trend = np.concatenate((X_train_trend, X_val_trend))
    X_comb_resid = np.concatenate((X_train_resid, X_val_resid))

    trend_thresholds = compute_thresholds(trend_model, X_comb_trend)
    resid_thresholds = compute_thresholds(residual_model, X_comb_resid)

    # 5) Evaluate on test
    trend_loss = compute_channel_loss(trend_model, X_test_trend)
    residual_loss = compute_channel_loss(residual_model, X_test_resid)

    # Now do your anomaly detection logic using trend_loss vs trend_thresholds, etc.
    print("Demo finished. Evaluate trend_loss/residual_loss as needed.")
    # ...
