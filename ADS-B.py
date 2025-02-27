import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks


def read_wav_iq(filepath):
    """
    WAV dosyasından IQ verisini okur ve kompleks formata çevirir.
    """
    fs, data = wavfile.read(filepath)

    # Stereo kontrolü (IQ verisi iki kanal olmalı)
    if data.ndim == 2:
        I = data[:, 0]
        Q = data[:, 1]
        iq_samples = I.astype(np.float32) + 1j * Q.astype(np.float32)
    else:
        raise ValueError("Veri tek kanal görünüyor, IQ verisi stereo olmalıdır.")

    return fs, iq_samples


def adaptive_threshold(amplitude):
    """
    Genliğin ortalaması ve standart sapmasına dayalı dinamik threshold belirler.
    Şu an mean + 3.5 * std kullanıyoruz (daha agresif bir eşikleme)
    """
    mean_amp = np.mean(amplitude)
    std_amp = np.std(amplitude)
    return mean_amp + (3.5 * std_amp)  #  Daha katı threshold!


def detect_preamble(iq_data, fs, min_preamble_distance=350):
    """
    IQ verisinde ADS-B Mode S preamble tespiti yapar.

    Parametreler:
      iq_data         : Kompleks IQ sinyali (numpy array)
      fs             : Örnekleme frekansı (Hz) (2.4e6 gibi)
      min_preamble_distance: Minimum preamble mesafesi (örnek cinsinden)

    Dönüş:
      preamble_indices: Preamble başlangıç indeksleri
    """
    # 1) Genlik hesapla
    amplitude = np.abs(iq_data)

    # 2) Dinamik eşik belirle
    threshold = adaptive_threshold(amplitude)  #  Daha sıkı threshold uygulanıyor

    # 3) Eşiği geçen zirveleri (peaks) bul
    peaks, _ = find_peaks(amplitude, height=threshold, distance=int(fs * 0.5e-6))

    # 4) Gerçek preamble olup olmadığını test et
    preamble_candidates = []
    for i in range(len(peaks) - 3):
        p1, p2, p3, p4 = peaks[i:i + 4]

        if (
                abs(p2 - p1) == int(fs * 1e-6) and  # 1.0 µs aralık
                abs(p3 - p1) == int(fs * 3.5e-6) and  # 3.5 µs aralık
                abs(p4 - p1) == int(fs * 4.5e-6)  # 4.5 µs aralık
        ):
            # En son eklenen preamble ile arasında en az 350 örnek mesafe olmalı
            if len(preamble_candidates) == 0 or (p1 - preamble_candidates[-1]) > min_preamble_distance:
                # Ekstra güvenlik: Eğer bu preamble threshold'un altında ise, eklemiyoruz
                avg_preamble_amp = np.mean(amplitude[p1:p1 + 20])  # İlk 20 örneğin ortalaması
                if avg_preamble_amp > threshold:
                    preamble_candidates.append(p1)

    return np.array(preamble_candidates)


# 📌 WAV dosyasını oku
dosya_yolu = r"C:\Users\Can Tekin\OneDrive\Belgeler\adsb.2021.wav"
fs, iq_samples = read_wav_iq(dosya_yolu)

# 📌 Preamble tespiti (Threshold DİNAMİK, daha sıkı, min mesafe artırıldı)
preamble_indices = detect_preamble(iq_samples, fs)

# 📌 Sonuçları yazdır
print("Bulunan preamble sayısı:", len(preamble_indices))

# 📌 Grafikte işaretleyelim
plt.figure(figsize=(10, 4))
plt.plot(np.abs(iq_samples[:50000]), label="Amplitude")
plt.scatter(preamble_indices, np.abs(iq_samples[preamble_indices]), color='r', marker='x', label="Preambles")
plt.title("ADS-B Mode S Preamble Tespiti (Daha Katı Threshold ile)")
plt.xlabel("Örnek Numarası")
plt.ylabel("Genlik")
plt.legend()
plt.show()
