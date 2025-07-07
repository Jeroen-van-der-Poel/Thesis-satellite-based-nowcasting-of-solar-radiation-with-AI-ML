import sys
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from utils.metrics import compute_rmse, compute_rrmse, compute_mae, compute_ssim, compute_forecast_skill
from tqdm import tqdm
import matplotlib.pyplot as plt
# from EarthFormer.visualization.sevir.sevir_vis_seq import save_example_vis_results 
from EarthFormer.visualization.sevir.sevir_vis_seq_denorm import save_example_vis_results, save_comparison_vis_results
from EarthFormer.train import CuboidPLModule
from EarthFormer.h5LightningModule import H5LightningDataModule
from EarthFormer.Data.hdf5Dataset import HDF5NowcastingDataset
from omegaconf import OmegaConf
from persistence import Persistence
from utils.dgmr_wrapper import DGMRWrapper
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def evaluate_model(
    model_name,
    model,
    dataloader,
    inference_fn,
    visualize=False,
    visualization_indices=None,
    save_dir="./vis",
    sds_cs_dataset=None,
    denormalize=False
):
    os.makedirs(save_dir, exist_ok=True)
    if visualization_indices is None:
        visualization_indices = []

    metrics = {k: [] for k in ["rmse", "rrmse", "mae", "ssim", "fs"]}
    cached_samples = {} 

    for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
        inputs = batch[:, :4]
        targets = batch[:, 4:]
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        with torch.no_grad():
            if model_name == "DGMR-SO":
                preds, targets, preds_cropped, target_cropped, y_coords, x_coords = inference_fn(model, inputs, targets)
                preds_cropped_np = preds_cropped.detach().cpu().numpy().transpose(0, 1, 3, 2)
                target_cropped_np = target_cropped.detach().cpu().numpy().transpose(0, 1, 3, 2)
            else:
                preds, targets = inference_fn(model, inputs, targets)

        preds_np = preds.detach().cpu().numpy().transpose(0, 1, 3, 2)
        inputs_np = inputs.detach().cpu().numpy().transpose(0, 1, 3, 2)
        targets_np = targets.detach().cpu().numpy().transpose(0, 1, 3, 2)
        T = preds_np.shape[1]
        
        if denormalize and sds_cs_dataset is not None:
            sds_cs = sds_cs_dataset[idx]
            if torch.is_tensor(sds_cs):
                sds_cs = sds_cs.numpy()

            sds_cs_targets = sds_cs[4:]  
            sds_cs_targets = np.expand_dims(sds_cs_targets, axis=0) 
            sds_cs_targets = np.repeat(sds_cs_targets, preds_np.shape[0], axis=0)  
            sds_cs_inputs = sds_cs[:4]
            sds_cs_inputs = np.expand_dims(sds_cs_inputs, axis=0)  
            sds_cs_inputs = np.repeat(sds_cs_inputs, inputs_np.shape[0], axis=0) 

            inputs_np_1 = inputs_np
            inputs_np = inputs_np * sds_cs_inputs
            if preds_np.shape == targets_np.shape == sds_cs_targets.shape:
                preds_np = preds_np * sds_cs_targets
                targets_np = targets_np * sds_cs_targets
                if model_name == "DGMR-SO":
                    preds_cropped_np = preds_cropped_np * sds_cs_targets[:, :preds_cropped_np.shape[1], :preds_cropped_np.shape[2]]
                    target_cropped_np = target_cropped_np * sds_cs_targets[:, :target_cropped_np.shape[1], :target_cropped_np.shape[2]]
            else:
                raise ValueError(f"Shape mismatch between predictions and SDS clear sky targets at index {idx}")
            
        if idx == 0:
            for k in metrics:
                metrics[k] = [[] for _ in range(T)]

        for t in range(T):
            try:
                preds_np = np.clip(preds_np, 0, None)
                targets_np = np.clip(targets_np, 0, None)
                inputs_np = np.clip(inputs_np, 0, None)
                
                pred = preds_np[:, t]
                target = targets_np[:, t]

                if model_name == "DGMR-SO":
                    preds_cropped_np = np.clip(preds_cropped_np, 0, None)
                    target_cropped_np = np.clip(target_cropped_np, 0, None)
                    pred_crop = preds_cropped_np[:, t]
                    target_crop = target_cropped_np[:, t]
                    metrics["ssim"][t].append(compute_ssim(pred_crop, target_crop))

                    mask = (pred_crop > 0) & (target_crop > 0)
                    pred_masked = pred_crop[mask]
                    target_masked = target_crop[mask]
                else:
                    metrics["ssim"][t].append(compute_ssim(pred, target))
                    mask = (pred > 0) & (target > 0)
                    pred_masked = pred[mask]
                    target_masked = target[mask]

                metrics["rmse"][t].append(compute_rmse(pred_masked, target_masked))
                metrics["rrmse"][t].append(compute_rrmse(pred_masked, target_masked))
                metrics["mae"][t].append(compute_mae(pred_masked, target_masked))

                baseline = inputs_np_1[:, -1] * sds_cs_targets[:, t]
                if model_name == "DGMR-SO":
                    baseline_crop = np.array([
                        baseline[b,
                                y_coords[b]:y_coords[b] + preds_cropped_np.shape[2],
                                x_coords[b]:x_coords[b] + preds_cropped_np.shape[3]]
                        for b in range(baseline.shape[0])
                    ])
                    baseline_mask = (baseline_crop > 0)
                    baseline_masked = baseline_crop[baseline_mask]
                else:
                    baseline_mask= (baseline > 0)
                    baseline_masked = baseline[baseline_mask]
                metrics["fs"][t].append(compute_forecast_skill(pred_masked, target_masked, baseline_masked))

            except Exception as e:
                # print(f"Metric error at t={t}, batch={idx}: {e}")
                for k in metrics:
                    metrics[k][t].append(np.nan)

        if visualize and idx in visualization_indices:
            cached_samples[idx] = {
                "inputs_np": inputs_np,
                "targets_np": targets_np,
                "preds_np": preds_np
            }
            save_example_vis_results(
                save_dir=save_dir,
                save_prefix=f"{model_name.lower()}_example_{idx}",
                in_seq=inputs_np,
                target_seq=targets_np,
                pred_seq=preds_np,
                label=model_name,
                layout="NTHWC",
                plot_stride=1,
                vis_hits_misses_fas=False,
                interval_real_time=15
            )

    averages = {
        k: np.nanmean([float(x) for v in ts for x in v if np.isscalar(x) and not np.isnan(x)])
        if any(np.isscalar(x) and not np.isnan(x) for v in ts for x in v) else np.nan
        for k, ts in metrics.items()
    }
    for k, v in averages.items():
        print(f"{model_name} Average - {k}: {v}")

    return metrics, averages, cached_samples


def infer_earthformer(model, inputs, targets):
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)
    preds = model(inputs)
    return preds, targets

def infer_persistence(model, inputs, targets):
    inputs, targets = inputs.to(inputs.device), targets.to(inputs.device)
    preds, _ = model(inputs, targets)
    return preds, targets

def infer_dgmr(model, inputs, targets):
    inputs, targets = inputs.cpu(), targets.cpu()
    preds, targets, preds_cropped, targets_cropped, y_coords, x_coords = model(inputs, targets)
    return preds, targets, preds_cropped, targets_cropped, y_coords, x_coords


def plot_metrics(metrics_dict, model_name="Model", save_dir="./vis"):
    os.makedirs(save_dir, exist_ok=True)
    time_steps = np.arange(1, len(next(iter(metrics_dict.values()))) + 1) * 15

    for metric, values in metrics_dict.items():
        # print(f"{metric}: {len(values)} intervals")
        avg_values = [np.nanmean(v) if len(v) > 0 else np.nan for v in values]

        plt.figure()
        plt.plot(time_steps[:len(avg_values)], avg_values, marker='o')
        plt.title(f"{model_name} - {metric.upper()} per 15-min Interval")
        plt.xlabel("Time (minutes)")
        if metric == "mae" or metric == "rmse":
            plt.ylabel(f"{metric.upper()}  (W/m²)")
        elif metric == "rrmse":
            plt.ylabel(f"{metric.upper()} (%)")
        else:
            plt.ylabel(f"{metric.upper()}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{metric}_15min.png', bbox_inches='tight')
        plt.show()
        plt.close()

def plot_combined_metrics(metrics_list, model_names, save_dir="./vis/combined"):
    os.makedirs(save_dir, exist_ok=True)

    metrics_keys = metrics_list[0].keys()
    time_steps = np.arange(1, len(next(iter(metrics_list[0].values()))) + 1) * 15

    for metric in metrics_keys:
        plt.figure()
        for metrics, name in zip(metrics_list, model_names):
            values = metrics[metric]
            avg_values = [np.nanmean(v) if len(v) > 0 else np.nan for v in values]
            plt.plot(time_steps[:len(avg_values)], avg_values, marker='o', label=name)

        plt.title(f"{metric.upper()} per 15-min Interval")
        plt.xlabel("Time (minutes)")
        if metric == "mae" or metric == "rmse":
            plt.ylabel(f"{metric.upper()}  (W/m²)")
        elif metric == "rrmse":
            plt.ylabel(f"{metric.upper()} (%)")
        else:
            plt.ylabel(f"{metric.upper()}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{metric}_combined.png', bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    DGMR_CHECKPOINT_DIR = "../DGMR_SO/experiments/solar_nowcasting_v7/"
    EARTHFORMER_CFG = "../EarthFormer/config/train.yml"
    EARTHFORMER_CHECKPOINT = "../EarthFormer/experiments/ef_v23/checkpoints/model-epoch=189.ckpt"
    # EARTHFORMER_CHECKPOINT = "../EarthFormer/experiments/ef_v27/checkpoints/model-epoch=001.ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(EARTHFORMER_CFG)
    train_path = os.path.expanduser(cfg.dataset.train_path)
    val_path = os.path.expanduser(cfg.dataset.val_path)
    test_path = os.path.expanduser(cfg.dataset.test_path) 
    dm = H5LightningDataModule(
        train_path=train_path,
        val_path=val_path,  
        test_path=test_path,
        batch_size=8, # cfg.optim.micro_batch_size
        num_workers=2
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

    sds_cs_dataset = HDF5NowcastingDataset("/data1/h5data/test_data/test_data_cs_3.h5")
    print(f"Loaded {len(sds_cs_dataset)} SDS_CS samples to calculate SDS.")

    print("Loading models...")
    ef_module = CuboidPLModule.load_from_checkpoint(
        checkpoint_path=EARTHFORMER_CHECKPOINT,
        total_num_steps=total_num_steps,
        oc_file=EARTHFORMER_CFG,
        save_dir="evaluation_run"
    )
    ef_model = ef_module.torch_nn_module
    ef_model.eval()
    ef_model.to(device)

    persistence_model = Persistence(layout="NTHWC")
    persistence_model.eval()
    persistence_model.to(device)

    dgmr_model = DGMRWrapper(DGMR_CHECKPOINT_DIR)

    print("Evaluating Persistence...")
    p_metrics, p_results, p_cache = evaluate_model(
        "Persistence", 
        persistence_model, 
        dm.test_dataloader(),
        inference_fn=infer_persistence,
        visualize=True, 
        visualization_indices=[0, 800, 1250, 1500],
        save_dir="./vis/persistence",
        sds_cs_dataset=sds_cs_dataset,
        denormalize=True
    )
    plot_metrics(p_metrics, model_name="Persistence", save_dir="./vis/persistence")

    # print("Evaluating EarthFormer...")
    # ef_metrics, ef_results, ef_cache = evaluate_model(
    #     "EarthFormer", 
    #     ef_model, 
    #     dm.test_dataloader(),
    #     inference_fn=infer_earthformer,
    #     visualize=True, 
    #     visualization_indices=[0, 800, 1250, 1500],
    #     save_dir="./vis/earthformer",
    #     sds_cs_dataset=sds_cs_dataset,
    #     denormalize=True
    # )
    # plot_metrics(ef_metrics, model_name="EarthFormer", save_dir="./vis/earthformer")

    # print("Evaluating DGMR-SO...")
    # dgmr_metrics, dgmr_results, dgmr_cache = evaluate_model(
    #     "DGMR-SO", 
    #     dgmr_model, 
    #     dm.test_dataloader(),
    #     inference_fn=infer_dgmr,
    #     visualize=True, 
    #     visualization_indices=[0, 800, 1250, 1500],
    #     save_dir="./vis/dgmr",
    #     sds_cs_dataset=sds_cs_dataset,
    #     denormalize=True
    # )
    # plot_metrics(dgmr_metrics, model_name="DGMR-SO", save_dir="./vis/dgmr")

    # print("Plotting combined metrics...")
    # plot_combined_metrics(
    #     metrics_list=[ef_metrics, dgmr_metrics, p_metrics], 
    #     model_names=["EarthFormer", "DGMR-SO", "Persistence",], 
    #     save_dir="./vis/combined"
    # )

    # print("Saving side-by-side comparison visualizations...")
    # comparison_indices = [0, 800, 1250, 1500]
    # for idx in comparison_indices:
    #     inputs_np = ef_cache[idx]["inputs_np"] 
    #     targets_np = ef_cache[idx]["targets_np"]

    #     ef_preds_np = ef_cache[idx]["preds_np"]
    #     dgmr_preds_np = dgmr_cache[idx]["preds_np"]
    #     p_preds_np = p_cache[idx]["preds_np"]

    #     save_comparison_vis_results(
    #         save_dir="./vis/combined",
    #         save_prefix=f"comparison_example_{idx:04d}",
    #         in_seq=inputs_np,
    #         target_seq=targets_np,
    #         pred_seq_list=[ef_preds_np, dgmr_preds_np, p_preds_np],
    #         label_list=["EarthFormer", "DGMR-SO", "Persistence"],
    #         layout="NTHWC",
    #         interval_real_time=15,
    #         plot_stride=1,
    #         vis_hits_misses_fas=False
    #     )