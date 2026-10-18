"""
Küçük demo veri seti: arayüzün (dashboard) tam PTB-XL önbelleği (data/processed, ~1 GB) olmadan
çalışabilmesi için. Her senaryodan gerçek test kayıtları (strat_fold 10) + eğitim popülasyonu genlik istatistiği.

Çıktı (~10 MB, repoda izlenir):
    data/demo/X.npy, mu.npy, sigma.npy, meta.csv   seçilen test kayıtları (data/processed ile aynı biçim)
    data/demo/pop_sigma.npy                         eğitim kümesinde derivasyon başına medyan σ (12 değer)

Backend (src/main.py) data/processed yoksa otomatik olarak data/demo'yu kullanır.
Çalıştırma (data/processed hazırken): python -m src.make_demo_data
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.dataset import population_sigma

PER_CONDITION = 40
ALWAYS_INCLUDE = [16038]  # testlerde kullanılan kayıt (Anterior + Inferior MI)

SELECTORS = {
    "normal": lambda m: m["label"] == 0,
    "mi": lambda m: m["label"] == 1,
    "ant": lambda m: (m["label"] == 1) & (m["AMI"] == 1),
    "inf": lambda m: (m["label"] == 1) & (m["IMI"] == 1),
    "lat": lambda m: (m["label"] == 1) & (m["LMI"] == 1),
    "sttc": lambda m: (m["label"] == -1) & (m["STTC"] == 1),
}


def main():
    parser = argparse.ArgumentParser(description="Arayüz için küçük demo veri seti")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/demo")
    parser.add_argument("--per-condition", type=int, default=PER_CONDITION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = pd.read_csv(os.path.join(args.processed_dir, "meta.csv"))
    test = meta["split"] == "test"
    rng = np.random.default_rng(args.seed)
    rows = set(np.flatnonzero((test & meta["ecg_id"].isin(ALWAYS_INCLUDE)).to_numpy()))
    for name, select in SELECTORS.items():
        pool = np.flatnonzero((test & select(meta)).to_numpy())
        rows.update(rng.choice(pool, size=min(args.per_condition, len(pool)), replace=False).tolist())
    rows = np.array(sorted(rows))

    X = np.load(os.path.join(args.processed_dir, "X.npy"), mmap_mode="r")
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X.npy"), np.asarray(X[rows], dtype=np.float32))
    for name in ["mu", "sigma"]:
        np.save(os.path.join(args.out_dir, f"{name}.npy"), np.load(os.path.join(args.processed_dir, f"{name}.npy"))[rows])
    np.save(os.path.join(args.out_dir, "pop_sigma.npy"), population_sigma(args.processed_dir))
    meta.iloc[rows].to_csv(os.path.join(args.out_dir, "meta.csv"), index=False)

    demo = meta.iloc[rows]
    counts = {name: int((select(demo)).sum()) for name, select in SELECTORS.items()}
    size = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in os.listdir(args.out_dir)) / 1e6
    print(f"Demo seti: {len(rows)} test kaydı, {size:.1f} MB -> {args.out_dir}/   senaryo başına: {counts}")


if __name__ == "__main__":
    main()
