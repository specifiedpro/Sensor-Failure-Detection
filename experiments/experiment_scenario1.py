# experiment_scenario1.py

import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.metrics import f1_score

from src.data_generation import get_simple_data
from src.shaping import train_shape, val_shape, test_shape
from src.models import build_trend_model, build_residual_model, train_autoencoder
from src.thresholds import compute_thresholds, compute_channel_loss

def run_scenario1(process_level=1.0, sensor_level=0.01, runs=30):
    """
    Example scenario 1 that repeats multiple runs with user-specified
    process_level, sensor_level, etc. Returns aggregated F1 results.
    """

    f1_trend_scores = []
    f1_resid_scores = []

    for seed in range(runs):
        np.random.seed(seed)

        # 1) Generate data
        signal, train_list, test_list, t_trend, t_resid, v_trend, v_resid, test_trend_list, test_resid_list = \
            get_simple_data(sigma=process_level, p=3, s=sensor_level, length=12600)

        # 2) Shape data
        X_train_trend, X_train_resid = train_shape(t_trend, t_resid)
        X_val_trend,   X_val_resid   = val_shape(v_trend, v_resid)
        X_test_trend,  X_test_resid  = test_shape(test_trend_list, test_resid_list)

        # 3) Build & train
        trend_model = build_trend_model()
        residual_model = build_residual_model()

        trend_model = train_autoencoder(trend_model, X_train_trend, X_val_trend,
                                        filepath=None, epochs=25, retrain=False)
        residual_model = train_autoencoder(residual_model, X_train_resid, X_val_resid,
                                           filepath=None, epochs=25, retrain=False)

        # 4) Threshold
        X_comb_trend = np.concatenate((X_train_trend, X_val_trend))
        X_comb_resid = np.concatenate((X_train_resid, X_val_resid))
        trend_thresh = compute_thresholds(trend_model, X_comb_trend)
        resid_thresh = compute_thresholds(residual_model, X_comb_resid)

        # 5) Test => channel loss
        test_trend_loss = compute_channel_loss(trend_model, X_test_trend)
        test_resid_loss = compute_channel_loss(residual_model, X_test_resid)

        # Suppose ground truth is known:
        # e.g., trend_y => which channels truly have a “trend anomaly”
        # residual_y => which channels truly have a “residual anomaly”
        # For demonstration:
        trend_y = [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        residual_y = [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # 6) Predict => if max channel loss > threshold => anomaly
        pred_trend = [0]*20
        pred_resid = [0]*20
        for ch in range(20):
            if max(test_trend_loss[ch]) > trend_thresh[ch]:
                pred_trend[ch] = 1
            if max(test_resid_loss[ch]) > resid_thresh[ch]:
                pred_resid[ch] = 1

        f1_trend = f1_score(trend_y, pred_trend, average='macro')
        f1_resid = f1_score(residual_y, pred_resid, average='macro')

        f1_trend_scores.append(f1_trend)
        f1_resid_scores.append(f1_resid)

    return {
        "process_level": process_level,
        "sensor_level": sensor_level,
        "f1_trend_mean": np.mean(f1_trend_scores),
        "f1_trend_std":  np.std(f1_trend_scores),
        "f1_resid_mean": np.mean(f1_resid_scores),
        "f1_resid_std":  np.std(f1_resid_scores),
    }


if __name__ == "__main__":
    # Example usage
    results = run_scenario1(process_level=1.0, sensor_level=0.015, runs=5)
    print(results)
