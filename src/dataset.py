import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.ecg_constants import WALL_NAMES, WALL_TO_SUBCLASS


class PTBXLProcessedDataset(Dataset):
    """
    `python -m src.prepare_data` ile üretilen önbellekten (data/processed) okuyan MI/Normal Dataset.

    split : "train" | "val" | "test"  (strat_fold 1–8 / 9 / 10)
    Döndürür: (sinyal (12, 1000) float32, hedef sözlüğü)
      hedef["mi"]    : skaler, 1 = MI, 0 = Normal
      hedef["walls"] : (3,) Anterior/Inferior/Lateral MI (AMI/IMI/LMI) çoklu etiket — kriz tipi
      hedef["mu"], hedef["sigma"] : (12,) z-score istatistikleri (mV cinsinden RMSE için)
    balanced=True: Rapor Md. 2.1'deki dengeli MI / Normal alt kümesi (her bölümde MI sayısı kadar Normal).
    balanced=False: tüm MI ve Normal kayıtlar (label >= 0). X.npy bellek eşlemeli (mmap) açılır;
    ~1 GB'lık dosya RAM'e tümüyle yüklenmez.
    """
    def __init__(self, split="train", processed_dir="data/processed", transform=None, balanced=True):
        meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"), index_col="ecg_id")
        mask = (meta["split"] == split) & (meta["label"] >= 0)
        if balanced:
            mask &= meta["balanced"] == 1
        mask = mask.to_numpy()

        self.indices = np.flatnonzero(mask)
        self.X = np.load(os.path.join(processed_dir, "X.npy"), mmap_mode="r")
        self.mu = np.load(os.path.join(processed_dir, "mu.npy"))[self.indices]
        self.sigma = np.load(os.path.join(processed_dir, "sigma.npy"))[self.indices]
        self.meta = meta.iloc[self.indices]
        self.labels = self.meta["label"].to_numpy(dtype=np.float32)
        self.walls = self.meta[[WALL_TO_SUBCLASS[w] for w in WALL_NAMES]].to_numpy(dtype=np.float32)
        self.ecg_ids = self.meta.index.to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        signal = torch.from_numpy(np.array(self.X[self.indices[idx]], dtype=np.float32))
        if self.transform:
            signal = self.transform(signal)
        target = {
            "mi": torch.tensor(self.labels[idx]),
            "walls": torch.from_numpy(self.walls[idx]),
            "mu": torch.from_numpy(self.mu[idx]),
            "sigma": torch.from_numpy(self.sigma[idx]),
        }
        return signal, target

    def positive_fraction(self):
        return float(self.labels.mean()) if len(self.labels) else 0.0
