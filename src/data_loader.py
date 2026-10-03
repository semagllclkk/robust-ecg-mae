import os
import pandas as pd
import numpy as np
import wfdb
import urllib.request
import zipfile

def download_ptbxl_demo(data_dir='data/raw'):
    """
    Eger ptbxl_database.csv yoksa PhysioNet uzerinden indirir.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'ptbxl_database.csv')
    
    if not os.path.exists(csv_path):
        print("PTB-XL Veri seti bulunamadi. PhysioNet'ten indiriliyor (bu işlem birkaç saniye sürebilir)...")
        url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
        try:
            urllib.request.urlretrieve(url, csv_path)
            print("ptbxl_database.csv basariyla indirildi!")
        except Exception as e:
            print(f"Indirme hatasi: {e}")
            return None

    df = pd.read_csv(csv_path, index_col='ecg_id')
    print(f"\n--- Veri Seti OzetBilgisi ---")
    print(f"Toplam EKG Kaydi Sayisi: {len(df)}")
    return df

if __name__ == '__main__':
    df = download_ptbxl_demo()
    if df is not None:
        print("\nIlk 5 Satir:")
        print(df[['patient_id', 'age', 'sex', 'height', 'weight']].head())