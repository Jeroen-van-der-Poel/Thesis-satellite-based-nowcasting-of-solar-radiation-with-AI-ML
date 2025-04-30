
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from solar_datamodule import SolarLightningDataModule 
from cuboid_transformer.cuboid_transformer import CuboidTransformerModel
import pytorch_lightning as pl
from omegaconf import OmegaConf
from argparse import Namespace
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import os
import numpy as np
from earthformer.visualization.sevir.sevir_vis_seq import save_example_vis_results

def warmup_lambda(warmup_steps, min_lr_ratio=0.0):
    def f(step):
        if step < warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * step / float(max(1, warmup_steps))
        else:
            return 1.0
    return f

class EarthFormerPLModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg.optim.lr
        self.wd = cfg.optim.wd
        self.loss_fn = torch.nn.MSELoss()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.save_dir = os.path.join("experiments", cfg.logging.logging_prefix)
        os.makedirs(self.save_dir, exist_ok=True)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def forward(self, in_seq, out_seq):
        output = self.model(in_seq)
        loss = F.mse_loss(output, out_seq)
        return output, loss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        preds, loss = self(x, y)
        self.train_mse.update(preds, y)
        self.train_mae.update(preds, y)
        if batch_idx == 0:
            self.save_vis(x, y, preds, batch_idx, mode="train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_mse_epoch", self.train_mse.compute())
        self.log("train_mae_epoch", self.train_mae.compute())
        self.train_mse.reset()
        self.train_mae.reset()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        preds, loss = self(x, y)
        self.val_mse.update(preds, y)
        self.val_mae.update(preds, y)
        if batch_idx == 0:
            self.save_vis(x, y, preds, batch_idx, mode="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outs):
        self.log("val_mse_epoch", self.val_mse.compute())
        self.log("val_mae_epoch", self.val_mae.compute())
        self.val_mse.reset()
        self.val_mae.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        warmup_steps = int(self.cfg.optim.warmup_percentage * self.cfg.optim.max_epochs)
        warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda(warmup_steps, self.cfg.optim.warmup_min_lr_ratio))
        cosine_sched = CosineAnnealingLR(optimizer, T_max=self.cfg.optim.max_epochs - warmup_steps,
                                         eta_min=self.cfg.optim.min_lr_ratio * self.cfg.optim.lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def save_vis(self, x, y, pred, batch_idx, mode):
        save_example_vis_results(
            save_dir=self.example_save_dir,
            save_prefix=f'{mode}_epoch_{self.current_epoch}_batch_{batch_idx}',
            in_seq=x.detach().cpu().numpy(),
            target_seq=y.detach().cpu().numpy(),
            pred_seq=pred.detach().cpu().numpy(),
            layout=self.cfg.layout.layout,
            plot_stride=self.cfg.vis.plot_stride,
            label=self.cfg.logging.logging_prefix,
            interval_real_time=self.cfg.dataset.interval_real_time
        )

def main(args):
    seed_everything(42)
    cfg = OmegaConf.load(args.cfg)
    datamodule = SolarLightningDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=cfg.optim.micro_batch_size,
        num_workers=args.num_workers
    )
    model = CuboidTransformerModel(**cfg.model)
    pl_module = EarthFormerPLModule(model=model, cfg=cfg)
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=cfg.optim.save_top_k),
        LearningRateMonitor(logging_interval="epoch")
    ]
    if cfg.optim.early_stop:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=cfg.optim.early_stop_patience,
            mode=cfg.optim.early_stop_mode,
            verbose=True
        ))
    loggers = [
        CSVLogger("logs/", name=cfg.logging.logging_prefix),
        TensorBoardLogger("logs/", name=cfg.logging.logging_prefix)
    ]
    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=cfg.optim.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.optim.gradient_clip_val
    )
    trainer.fit(pl_module, datamodule=datamodule)

if __name__ == "__main__":
    from argparse import Namespace
    cfg_path = "./config/train.yml"
    train_path = "/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data"
    val_path = '/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data'
    num_gpus = 1
    num_workers = 4
    args = Namespace(
        cfg=cfg_path,
        train_path=train_path,
        val_path=val_path,
        gpus=num_gpus,
        num_workers=num_workers
    )
    main(args)
