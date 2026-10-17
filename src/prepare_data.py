"""
PTB-XL veri hazırlama hattı (TÜBİTAK Raporu İP1, Md. 2.1).

Ne yapar:
  1. ptbxl_database.csv + scp_statements.csv okunur, SCP kodları tanı sınıflarına çevrilir.
  2. İkili görev etiketi üretilir (label sütunu):
       1  = MI      : tanısal süper sınıfları arasında MI olan kayıt (MI+STTC gibi eş tanılar dahil)
       0  = Normal  : tek tanısal süper sınıfı NORM olan kayıt
      -1  = diğer   : MI içermeyen patolojik kayıtlar veya tanısal kodu olmayanlar (sınıflandırmaya girmez)
  3. 21.799 kaydın tamamı 100 Hz (records100) olarak okunur.
  4. Her kayda 0.5–45 Hz Butterworth filtre + derivasyon bazlı Z-score uygulanır.
  5. Sonuç tek seferlik bir önbelleğe yazılır (eğitim her seferinde 21 bin dosyayı tekrar okumaz):

     data/processed/
       X.npy          (N, 12, 1000) float32   filtrelenmiş + Z-score sinyaller
       mu.npy         (N, 12)       float32   Z-score ortalaması (mV'a geri dönüş için)
       sigma.npy      (N, 12)       float32   Z-score std'si
       meta.csv       ecg_id, strat_fold, split, label, süper sınıf ve MI alt sınıf sütunları

Bölümleme: PTB-XL'in resmi önerisi — strat_fold 1–8 eğitim, 9 doğrulama, 10 test.
Aynı hastanın kayıtları aynı fold'da olduğundan hasta sızıntısı (data leakage) oluşmaz.

Çalıştırma (proje kök klasöründen):
    python -m src.prepare_data
    python -m src.prepare_data --limit 200 --out-dir data/processed_debug   # hızlı deneme
"""
import argparse
import ast
import os
import time

import numpy as np
import pandas as pd
import wfdb

from src.preprocess import preprocess_ecg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
MI_SUBCLASSES = ["AMI", "IMI", "LMI", "PMI"]  # anatomik duvar (XAI için saklanır)
LABEL_NAMES = {0: "Normal", 1: "MI"}
FS = 100
SEQ_LEN = 1000


def find_ptbxl_root(data_dir):
    candidate = os.path.join(data_dir, "ptb-xl-1.0.3")
    root = candidate if os.path.exists(candidate) else data_dir
    if not os.path.exists(os.path.join(root, "ptbxl_database.csv")):
        raise FileNotFoundError(f"ptbxl_database.csv bulunamadı: {root}")
    return root


def build_labels(root):
    """SCP kodlarını süper sınıf / MI alt sınıfı multi-hot sütunlarına ve ikili MI-Normal etiketine çevirir."""
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp["diagnostic"] == 1]

    def classes_of(codes, column):
        return {scp.loc[c, column] for c in codes if c in scp.index and isinstance(scp.loc[c, column], str)}

    for name in SUPERCLASSES:
        df[name] = df["scp_codes"].apply(lambda c, n=name: int(n in classes_of(c, "diagnostic_class")))
    for name in MI_SUBCLASSES:
        df[name] = df["scp_codes"].apply(lambda c, n=name: int(n in classes_of(c, "diagnostic_subclass")))

    is_mi = df["MI"] == 1
    is_pure_norm = (df["NORM"] == 1) & (df[SUPERCLASSES].sum(axis=1) == 1)
    df["label"] = np.select([is_mi, is_pure_norm], [1, 0], default=-1)

    df["split"] = np.select(
        [df["strat_fold"] <= 8, df["strat_fold"] == 9],
        ["train", "val"],
        default="test",
    )
    return df


def load_record(root, rel_path):
    signal, fields = wfdb.rdsamp(os.path.join(root, rel_path))
    names = [n.upper() for n in fields["sig_name"]]
    if names != [n.upper() for n in LEAD_NAMES]:
        raise ValueError(f"Beklenmeyen derivasyon sırası: {fields['sig_name']}")
    if fields["fs"] != FS or signal.shape != (SEQ_LEN, 12):
        raise ValueError(f"Beklenmeyen biçim: fs={fields['fs']} shape={signal.shape}")
    return signal.T  # (12, 1000)


def main():
    parser = argparse.ArgumentParser(description="PTB-XL ön işleme ve önbellek oluşturma")
    parser.add_argument("--data-dir", default="data/ptbxl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=None, help="Yalnızca ilk N kaydı işle (deneme için)")
    args = parser.parse_args()

    root = find_ptbxl_root(args.data_dir)
    df = build_labels(root)
    if args.limit:
        df = df.head(args.limit)

    n = len(df)
    print(f"[1/3] Etiketler hazır. Kayıt sayısı: {n}")

    X = np.zeros((n, 12, SEQ_LEN), dtype=np.float32)
    mu = np.zeros((n, 12), dtype=np.float32)
    sigma = np.zeros((n, 12), dtype=np.float32)
    ok = np.ones(n, dtype=bool)

    print(f"[2/3] Sinyaller okunuyor + filtre ({FS} Hz, 0.5–45 Hz) + Z-score...")
    t0 = time.time()
    for i, rel_path in enumerate(df["filename_lr"]):
        try:
            raw = load_record(root, rel_path)
            if not np.all(np.isfinite(raw)):
                raise ValueError("NaN/Inf değer içeriyor")
            X[i], mu[i], sigma[i] = preprocess_ecg(raw, fs=FS)
        except Exception as exc:  # bozuk kayıt: atla, raporla
            ok[i] = False
            print(f"  ! {rel_path} atlandı: {exc}")
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  ({time.time() - t0:.0f} sn)")

    X, mu, sigma, df = X[ok], mu[ok], sigma[ok], df[ok]

    print("[3/3] Önbellek yazılıyor...")
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "mu.npy"), mu)
    np.save(os.path.join(args.out_dir, "sigma.npy"), sigma)
    meta_cols = ["patient_id", "strat_fold", "split", "filename_lr", "label"] + SUPERCLASSES + MI_SUBCLASSES
    df[meta_cols].to_csv(os.path.join(args.out_dir, "meta.csv"))

    # Özet tablo (rapor için)
    print("\n=== Özet ===")
    print(f"X: {X.shape}  ({X.nbytes / 1e9:.2f} GB)   atlanan kayıt: {int((~ok).sum())}")
    summary = pd.crosstab(df["split"], df["label"].map({**LABEL_NAMES, -1: "diğer"}), margins=True, margins_name="toplam")
    print(summary.reindex([s for s in ["train", "val", "test", "toplam"] if s in summary.index]).to_string())
    print(f"\nÖnbellek kaydedildi: {args.out_dir}/")


if __name__ == "__main__":
    main()
