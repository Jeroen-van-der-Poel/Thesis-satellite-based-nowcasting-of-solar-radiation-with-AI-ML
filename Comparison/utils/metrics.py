import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim

def compute_rmse(pred, true):
    return np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))

def compute_rrmse(pred, true):
    rmse = compute_rmse(pred, true)
    mean_true = np.mean(true)
    return rmse / mean_true if mean_true != 0 else np.nan

def compute_mae(pred, true):
    return mean_absolute_error(true.flatten(), pred.flatten())

def compute_ssim(pred, true):
    pred = np.squeeze(pred)  # Remove singleton dimensions
    true = np.squeeze(true)
    ssim_scores = []

    for p, t in zip(pred, true):
        # If input is 3D (H, W, C), convert to 2D
        if p.ndim == 3:
            p = p[..., 0]
            t = t[..., 0]
        score = ssim(p, t, data_range=t.max() - t.min())
        ssim_scores.append(score)

    return np.mean(ssim_scores)

def compute_forecast_skill(pred, true, baseline):
    rmse_pred = compute_rmse(pred, true)
    rmse_base = compute_rmse(baseline, true)
    return 1 - (rmse_pred / rmse_base) if rmse_base > 0 else np.nan
