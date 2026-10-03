import os
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import urllib.request

def download_sample_signal(data_dir='data/raw'):
    """
    Örnek bir PTB-XL EKG sinyal dosyasını (hea ve dat) indirir.
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/records100/00000/00001_lr"
    
    hea_path = os.path.join(data_dir, "00001_lr.hea")
    dat_path = os.path.join(data_dir, "00001_lr.dat")
    
    if not os.path.exists(hea_path):
        print("Örnek sinyal dosyaları indiriliyor...")
        urllib.request.urlretrieve(base_url + ".hea", hea_path)
        urllib.request.urlretrieve(base_url + ".dat", dat_path)
        print("Sinyal dosyaları başarıyla indirildi!")

def plot_sample_ecg(data_dir='data/raw'):
    """
    İndirilen EKG sinyalini okur ve 12 kanaldan bir tanesini ekrana çizer.
    """
    record_path = os.path.join(data_dir, "00001_lr")
    record = wfdb.rdrecord(record_path)
    
    signals = record.p_signal
    channels = record.sig_name
    fs = record.fs
    
    print(f"Sinyal Boyutu: {signals.shape} (Örnek Sayısı x Kanal Sayısı)")
    print(f"Örnekleme Frekansı: {fs} Hz")
    print(f"Kanal İsimleri: {channels}")
    
    # Ilk kanalı (Lead I) çizdirelim
    plt.figure(figsize=(12, 4))
    plt.plot(signals[:1000, 0], color='crimson', linewidth=1.5)
    plt.title(f"PTB-XL Örnek EKG Sinyali ({channels[0]} Kanalı)")
    plt.xlabel("Örnek (Sample)")
    plt.ylabel("Genlik (mV)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "sample_ecg.png"))
    print("EKG grafiği 'data/raw/sample_ecg.png' olarak kaydedildi!")
    plt.show()

if __name__ == '__main__':
    download_sample_signal()
    plot_sample_ecg()