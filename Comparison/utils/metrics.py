import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim

def compute_rmse(pred, true):
    return np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))

# def compute_rrmse(pred, true):
#     rmse = compute_rmse(pred, true)
#     mean_true = np.mean(true)
#     return rmse / mean_true if mean_true != 0 else np.nan

def compute_rrmse(pred, target, eps=1e-5):
    rmse = compute_rmse(pred, target)
    denom = np.mean(target)
    # if abs(denom) < eps:
    #     return np.nan 
    return rmse / denom

def compute_mae(pred, true):
    return mean_absolute_error(true.flatten(), pred.flatten())

def compute_ssim(pred, true):
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    ssim_scores = []

    for p, t in zip(pred, true):
        if p.ndim == 3:
            p = p[..., 0]
            t = t[..., 0]

        # Check for valid content
        if np.isnan(p).any() or np.isnan(t).any():
            ssim_scores.append(np.nan)
            continue

        range_val = np.max(t) - np.min(t)
        if range_val < 1e-3:
            # Avoid computing SSIM on near-constant images
            ssim_scores.append(np.nan)
            continue

        try:
            score = ssim(p, t, data_range=range_val)
            ssim_scores.append(score)
        except Exception as e:
            print(f"SSIM computation failed: {e}")
            ssim_scores.append(np.nan)

    return np.nanmean(ssim_scores) if ssim_scores else np.nan


def compute_forecast_skill(pred, true, baseline):
    rmse_pred = compute_rmse(pred, true)
    rmse_base = compute_rmse(baseline, true)
    return 1 - (rmse_pred / rmse_base) if rmse_base > 0 else np.nan
