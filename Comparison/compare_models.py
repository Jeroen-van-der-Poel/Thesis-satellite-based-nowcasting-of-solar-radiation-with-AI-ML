import sys
from pathlib import Path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from utils.metrics import compute_rmse, compute_rrmse, compute_mae, compute_ssim, compute_forecast_skill
from utils.dataloader import load_earthformer_test_data # load_dgmr_test_data
#from utils.modelloader import load_dgmr_model, load_earthformer_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from EarthFormer.visualization.sevir.sevir_vis_seq import save_example_vis_results 
from EarthFormer.train import CuboidPLModule


def evaluate_earthformer(model, dataloader, visualize=False, visualization_indices=None, save_dir="./earthformer_vis"):
    os.makedirs(save_dir, exist_ok=True)

    if visualization_indices is None:
        visualization_indices = []

    metrics = {"rmse": [], "rrmse": [], "mae": [], "ssim": []}
    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating EarthFormer")):
        batch = batch.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] → [B, T, C, H, W]
        inputs = batch[:, :4]
        targets = batch[:, 4:]

        with torch.no_grad():
            preds = model(inputs)

        metrics["rmse"].append(compute_rmse(preds, targets))
        metrics["rrmse"].append(compute_rrmse(preds, targets))
        metrics["mae"].append(compute_mae(preds, targets))
        metrics["ssim"].append(compute_ssim(preds, targets))

        if visualize and idx in visualization_indices:
            # Permute to NTHWC format expected by visualizer
            x_vis = inputs.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
            y_vis = targets.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
            pred_vis = preds.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

            save_example_vis_results(
                save_dir=save_dir,
                save_prefix=f"earthformer_example_{idx}",
                in_seq=x_vis,
                target_seq=y_vis,
                pred_seq=pred_vis,
                label="EarthFormer",
                layout="NTHWC",
            )

    averages = {k: np.mean(v) for k, v in metrics.items()}
    return metrics, averages

# def evaluate_dgmr(model, test_data, visualize=False, visualization_indices=None, save_dir="./dgmr_vis"):
#     os.makedirs(save_dir, exist_ok=True)

#     if visualization_indices is None:
#         visualization_indices = []

#     metrics = {"rmse": [], "rrmse": [], "mae": [], "ssim": []}
#     for idx, batch in enumerate(tqdm(test_data, desc="Evaluating DGMR-SO")):
#         x, y = batch
#         preds = model(x, training=False)

#         metrics["rmse"].append(compute_rmse(preds, y))
#         metrics["rrmse"].append(compute_rrmse(preds, y))
#         metrics["mae"].append(compute_mae(preds, y))
#         metrics["ssim"].append(compute_ssim(preds, y))

#         if visualize and idx in visualization_indices:
#             save_example_vis_results(
#                 save_dir=save_dir,
#                 save_prefix=f"dgmr_example_{idx}",
#                 in_seq=x.detach().cpu().numpy(),
#                 target_seq=y.detach().cpu().numpy(),
#                 pred_seq=preds.detach().cpu().numpy(),
#                 label="DGMR-SO",
#                 layout="NTHWC",  
#             )

#     averages = {k: np.mean(v) for k, v in metrics.items()}
#     return metrics, averages


def plot_metrics(metrics_dict, model_name="Model"):
    for metric, values in metrics_dict.items():
        plt.figure()
        plt.plot(values)
        plt.title(f"{model_name} - {metric.upper()} per Batch")
        plt.xlabel("Batch")
        plt.ylabel(metric.upper())
        plt.grid(True)
        plt.tight_layout()
        plt.show() 


if __name__ == "__main__":
    #DGMR_CHECKPOINT = "/path/to/dgmr_checkpoint"
    #DGMR_TEST_PATH = "/path/to/dgmr/test_data"
    EARTHFORMER_CFG = "~/projects/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/config/train.yaml"
    EARTHFORMER_CHECKPOINT = "/projects/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/experiments/ef_v10/model-epoch=189.ckpt"

    print("Loading test data...")
    #dgmr_test_data = load_dgmr_test_data(DGMR_TEST_PATH)
    ef_test_loader = load_earthformer_test_data(EARTHFORMER_CFG)

    print("Loading models...")
    #dgmr_model = load_dgmr_model(DGMR_CHECKPOINT)
    ef_module = CuboidPLModule.load_from_checkpoint(EARTHFORMER_CHECKPOINT)
    ef_model = ef_module.torch_nn_module

    print("Running evaluation...")
    # dgmr_metrics, dgmr_results = evaluate_dgmr(
    #     dgmr_model,
    #     dgmr_test_data,
    #     visualize=True,
    #     visualization_indices=[0, 500, 1000, 1500],  
    #     save_dir="./vis/dgmr"
    # )
    # plot_metrics(dgmr_metrics, model_name="DGMR-SO")

    ef_metrics, ef_results = evaluate_earthformer(
        ef_model,
        ef_test_loader,
        visualize=True,
        visualization_indices=[0, 500, 1000, 1500],
        save_dir="./vis/earthformer"
    )
    plot_metrics(ef_metrics, model_name="EarthFormer")