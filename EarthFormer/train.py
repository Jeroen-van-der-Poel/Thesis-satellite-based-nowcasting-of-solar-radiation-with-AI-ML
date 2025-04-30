import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from solar_datamodule import SolarLightningDataModule 
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
import pytorch_lightning as pl
from omegaconf import OmegaConf
from argparse import Namespace

class EarthFormerPLModule(pl.LightningModule):
    def __init__(self, model, lr, wd):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr
        self.wd = wd

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

def main(args):
    seed_everything(42)

    # Load YAML config
    cfg = OmegaConf.load(args.cfg)

    # Prepare data module
    datamodule = SolarLightningDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=cfg.optim.micro_batch_size,
        num_workers=args.num_workers
    )

    model = CuboidTransformerModel(**cfg.model)
    # Wrap in LightningModule
    pl_module = EarthFormerPLModule(model=model, lr=cfg.optim.lr, wd=cfg.optim.wd)

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
        log_every_n_steps=10
    )

    trainer.fit(pl_module, datamodule=datamodule)

if __name__ == "__main__":
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