from EarthFormer.netCDFLightningModule import NetCDFLightningDataModule
from omegaconf import OmegaConf
# from DGMR_SO.Data.data_pipeline import Dataset
from pathlib import Path
import os

# def load_dgmr_test_data(path=None, batch_size=16):
#     test_data, test_aug = Dataset(Path(path), batch_size=batch_size)
#     return test_data

def load_earthformer_test_data(cfg_path, batch_size=4):
    oc = OmegaConf.load(cfg_path)
    dataset_oc = OmegaConf.to_object(oc.dataset)
    dm = NetCDFLightningDataModule(
        train_path=os.path.expanduser(dataset_oc["train_path"]),
        test_path=os.path.expanduser(dataset_oc["test_path"]),
        batch_size=batch_size,
        num_workers=0,
    )
    dm.setup()
    return dm.test_dataloader()