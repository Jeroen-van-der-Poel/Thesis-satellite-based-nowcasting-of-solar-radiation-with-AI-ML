import sys
from pathlib import Path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from utils.metrics import compute_rmse, compute_rrmse, compute_mae, compute_ssim, compute_forecast_skill
from utils.dataloader import load_earthformer_test_data # load_dgmr_test_data
#from utils.modelloader import load_earthformer_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from EarthFormer.visualization.sevir.sevir_vis_seq import save_example_vis_results 
from EarthFormer.train import CuboidPLModule
from EarthFormer.h5LightningModule import H5LightningDataModule
from omegaconf import OmegaConf
from persistence import Persistence


def evaluate_earthformer(model, dataloader, visualize=False, visualization_indices=None, save_dir="./earthformer_vis"):
    import warnings
    os.makedirs(save_dir, exist_ok=True)
    if visualization_indices is None:
        visualization_indices = []

    metrics = {"rmse": [], "rrmse": [], "mae": [], "ssim": []}

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating EarthFormer")):
        inputs = batch[:, :4]
        targets = batch[:, 4:]

        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            preds = model(inputs)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        T = preds_np.shape[1]

        if idx == 0:
            for k in metrics:
                metrics[k] = [[] for _ in range(T)]

        for t in range(T):
            try:
                metrics["rmse"][t].append(compute_rmse(preds_np[:, t], targets_np[:, t]))
            except Exception as e:
                print(f"RMSE error at t={t}, batch={idx}: {e}")
                metrics["rmse"][t].append(np.nan)

            try:
                metrics["rrmse"][t].append(compute_rrmse(preds_np[:, t], targets_np[:, t]))
            except Exception as e:
                print(f"RRMSE error at t={t}, batch={idx}: {e}")
                metrics["rrmse"][t].append(np.nan)

            try:
                metrics["mae"][t].append(compute_mae(preds_np[:, t], targets_np[:, t]))
            except Exception as e:
                print(f"MAE error at t={t}, batch={idx}: {e}")
                metrics["mae"][t].append(np.nan)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    metrics["ssim"][t].append(compute_ssim(preds_np[:, t], targets_np[:, t]))
            except Exception as e:
                print(f"SSIM error at t={t}, batch={idx}: {e}")
                metrics["ssim"][t].append(np.nan)

        if visualize and idx in visualization_indices:
            save_example_vis_results(
                save_dir=save_dir,
                save_prefix=f"earthformer_example_{idx}",
                in_seq=inputs.detach().cpu().numpy(),
                target_seq=targets_np,
                pred_seq=preds_np,
                label="EarthFormer",
                layout="NTHWC",
                plot_stride=1,
                vis_hits_misses_fas=False,
                interval_real_time=15
            )

    averages = {k: np.nanmean([v for sub in metrics[k] for v in sub]) for k in metrics}

    for i in averages:
        print(f"Average - {i}: {averages[i]}")

    return metrics, averages

def evaluate_persistence(model, dataloader, visualize=False, visualization_indices=None, save_dir="./persistence_vis"):
    import warnings
    os.makedirs(save_dir, exist_ok=True)
    if visualization_indices is None:
        visualization_indices = []

    metrics = {"rmse": [], "rrmse": [], "mae": [], "ssim": []}

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Persistence")):
        inputs = batch[:, :4]
        targets = batch[:, 4:]

        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            preds, _ = model(inputs, targets)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        T = preds_np.shape[1]

        if idx == 0:
            for k in metrics:
                metrics[k] = [[] for _ in range(T)]

        for t in range(T):
            try:
                metrics["rmse"][t].append(compute_rmse(preds_np[:, t], targets_np[:, t]))
                metrics["rrmse"][t].append(compute_rrmse(preds_np[:, t], targets_np[:, t]))
                metrics["mae"][t].append(compute_mae(preds_np[:, t], targets_np[:, t]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    metrics["ssim"][t].append(compute_ssim(preds_np[:, t], targets_np[:, t]))
            except Exception as e:
                print(f"Metric error at t={t}, batch={idx}: {e}")
                for k in metrics:
                    metrics[k][t].append(np.nan)

        if visualize and idx in visualization_indices:
            save_example_vis_results(
                save_dir=save_dir,
                save_prefix=f"persistence_example_{idx}",
                in_seq=inputs.detach().cpu().numpy(),
                target_seq=targets_np,
                pred_seq=preds_np,
                label="Persistence",
                layout="NTHWC",
                plot_stride=1,
                vis_hits_misses_fas=False,
                interval_real_time=15
            )

    averages = {k: np.nanmean([v for sub in metrics[k] for v in sub]) for k in metrics}
    for i in averages:
        print(f"Average - {i}: {averages[i]}")
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


def plot_metrics(metrics_dict, model_name="Model", save_dir="./vis"):
    os.makedirs(save_dir, exist_ok=True)
    time_steps = np.arange(1, len(next(iter(metrics_dict.values()))) + 1) * 15

    for metric, values in metrics_dict.items():
        print(f"{metric}: {len(values)} intervals")
        avg_values = [np.nanmean(v) if len(v) > 0 else np.nan for v in values]

        plt.figure()
        plt.plot(time_steps[:len(avg_values)], avg_values, marker='o')
        plt.title(f"{model_name} - {metric.upper()} per 15-min Interval")
        plt.xlabel("Time (minutes)")
        plt.ylabel(metric.upper())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{metric}_15min.png', bbox_inches='tight')
        plt.show()
        plt.close()



if __name__ == "__main__":
    #DGMR_CHECKPOINT = "/path/to/dgmr_checkpoint"
    #DGMR_TEST_PATH = "/path/to/dgmr/test_data"
    EARTHFORMER_CFG = "../EarthFormer/config/train.yml"
    EARTHFORMER_CHECKPOINT = "../EarthFormer/experiments/ef_v12/checkpoints/model-epoch=026.ckpt"

    print("Loading test data...")
    #dgmr_test_data = load_dgmr_test_data(DGMR_TEST_PATH)

    print("Computing total_num_steps...")
    cfg = OmegaConf.load(EARTHFORMER_CFG)
    train_path = os.path.expanduser(cfg.dataset.train_path)
    val_path = os.path.expanduser(cfg.dataset.val_path)
    test_path = os.path.expanduser(cfg.dataset.test_path) 
    dm = H5LightningDataModule(
        train_path=train_path,
        val_path=val_path,  
        test_path=test_path,
        batch_size=cfg.optim.micro_batch_size,
        num_workers=8
    )
    dm.setup()
    num_train_samples = dm.num_train_samples
    total_batch_size = cfg.optim.total_batch_size
    max_epochs = cfg.optim.max_epochs

    total_num_steps = CuboidPLModule.get_total_num_steps(
        num_samples=num_train_samples,
        total_batch_size=total_batch_size,
        epoch=max_epochs
    )

    print("Loading models...")
    #dgmr_model = load_dgmr_model(DGMR_CHECKPOINT)
    ef_module = CuboidPLModule.load_from_checkpoint(
        checkpoint_path=EARTHFORMER_CHECKPOINT,
        total_num_steps=total_num_steps,
        oc_file=EARTHFORMER_CFG,
        save_dir="evaluation_run"
    )
    ef_model = ef_module.torch_nn_module
    ef_model.eval()
    ef_model.to("cuda" if torch.cuda.is_available() else "cpu")

    persistence_model = Persistence(layout="NTHWC")
    persistence_model.eval()
    persistence_model.to("cuda" if torch.cuda.is_available() else "cpu")

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
        dm.test_dataloader(),
        visualize=True,
        visualization_indices=[0, 500, 1000, 1500, 3000, 5000],
        save_dir="./vis/earthformer"
    )
    plot_metrics(ef_metrics, model_name="EarthFormer", save_dir="./vis/earthformer")

    p_metrics, p_results = evaluate_persistence(
        persistence_model,
        dm.test_dataloader(),
        visualize=True,
        visualization_indices=[0, 500, 1000, 1500, 3000, 5000],
        save_dir="./vis/persistence"
    )
    plot_metrics(p_metrics, model_name="Persistence", save_dir="./vis/persistence")