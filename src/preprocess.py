import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os
import wfdb

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """
    EKG sinyaline Butterworth Bandpass filtresi uygular.
    0.5 Hz altındaki (solunum gürültüsü) ve 40 Hz üstündeki (kas/elektrik gürültüsü) bileşenleri temizler.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def test_filtering(data_dir='data/raw'):
    record_path = os.path.join(data_dir, "00001_lr")
    if not os.path.exists(record_path + ".hea"):
        print("Sinyal dosyası bulunamadı. Önce data_loader.py çalıştırılmalı!")
        return
        
    record = wfdb.rdrecord(record_path)
    raw_signal = record.p_signal[:1000, 0] # Ilk 1000 ornek, Lead I
    
    # Filtre uygula
    filtered_signal = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=40.0, fs=record.fs)
    
    # Karşılaştırma grafiği çiz
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(raw_signal, color='gray', label='Ham Sinyal (Raw)')
    plt.title("Ham EKG Sinyali")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal, color='green', label='Filtrelenmiş Sinyal (Bandpass 0.5-40 Hz)')
    plt.title("Filtrelenmiş Temiz EKG Sinyali")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "filtered_ecg_comparison.png"))
    print("Filtreleme karşılaştırma grafiği 'data/raw/filtered_ecg_comparison.png' olarak kaydedildi!")
    plt.show()

if __name__ == '__main__':
    test_filtering()