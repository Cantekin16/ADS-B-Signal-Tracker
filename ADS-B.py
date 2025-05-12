import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, resample_poly, correlate
import matplotlib.pyplot as plt
import pyModeS as pms


# =============================================================================
# Parametreler
# =============================================================================
INPUT_SAMPLE_RATE = 2400000  # Hz; orijinal WAV dosyasının örnekleme hızı
TARGET_SAMPLE_RATE = 2000000 # Hz; downsampling sonrası kullanılacak örnekleme hızı
BITS_PER_MESSAGE = 112       # ADS‑B Extended Squitter mesajı uzunluğu (112 bit)
BIT_DURATION_US = 1          # Her bit 1 mikro saniye

# Preamble ayarları (ADS‑B preamble tipik olarak 8 µs):
PREAMBLE_DURATION_US = 8
SAMPLES_PER_US = TARGET_SAMPLE_RATE / 1e6   # Örneğin: 2 örnek/µs (hedef 2 MHz)
PREAMBLE_LENGTH = int(PREAMBLE_DURATION_US * SAMPLES_PER_US)

# Preamble maskesi: ADS‑B preamble’da tipik darbe pozisyonları 0, 1, 3 ve 4.5 µs
preamble_mask = np.zeros(PREAMBLE_LENGTH)
for pos in [0, 1, 3, 4.5]:
    idx = int(round(pos * SAMPLES_PER_US))
    if idx < PREAMBLE_LENGTH:
        preamble_mask[idx] = 1

# Korelasyon için eşik: Maksimum korelasyonun %80’i
THRESHOLD_RATIO = 0.8
# Mesaj başlangıcında ofset denemeleri (örnek cinsinden)
OFFSET_RANGE = [-1, 0, 1, 2, 3]

# =============================================================================
# 1. IQ Verisinin Okunması ve Kompleks Sinyal Oluşturulması
# =============================================================================
wav_filename = r"C:\Users\Can Tekin\OneDrive\Belgeler\adsb.2021.wav"
sample_rate, data = wavfile.read(wav_filename)
print("Original Sample Rate:", sample_rate)
print("Data shape:", data.shape)

if data.ndim < 2 or data.shape[1] < 2:
    raise ValueError("Stereo IQ verisi bekleniyor (en az iki kanal).")

I = data[:, 0].astype(np.float64)
Q = data[:, 1].astype(np.float64)
iq_signal = I + 1j * Q

# =============================================================================
# 2. Band-Pass Filtreleme (FIR)
# =============================================================================
numtaps = 101
lowcut = 200e3     # 200 kHz
highcut = 1000e3   # 1000 kHz
nyquist = INPUT_SAMPLE_RATE / 2
bp_coeff = firwin(numtaps, [lowcut/nyquist, highcut/nyquist], pass_zero=False)
filtered_signal = lfilter(bp_coeff, 1.0, iq_signal)
print("Band-pass filtre uygulandı.")

# =============================================================================
# 3. Downsampling (Polyphase Resampling)
# =============================================================================
# 2.4 MHz'den 2 MHz’ye geçiş: up=5, down=6 (hesaba göre 5/6 oranı)
resampled_signal = resample_poly(filtered_signal, up=5, down=6)
new_sample_rate = TARGET_SAMPLE_RATE
print("Resampled Signal Length:", len(resampled_signal))

# =============================================================================
# 4. Sinyalin Envelope’unun Hesaplanması
# =============================================================================
envelope = np.abs(resampled_signal)

# (Opsiyonel: envelope grafiğini görmek)
# plt.figure(figsize=(12, 4))
# plt.plot(envelope)
# plt.title("Signal Envelope")
# plt.xlabel("Örnek")
# plt.ylabel("Genlik")
# plt.show()

# =============================================================================
# 5. Preamble Tespiti (Cross-Correlation)
# =============================================================================
corr = correlate(envelope, preamble_mask, mode='valid')
max_corr = np.max(corr)
threshold = THRESHOLD_RATIO * max_corr
candidate_indices = np.where(corr > threshold)[0]

# Aday preamble indeksleri arasında minimum 1 ms (new_sample_rate * 0.001) ayrım koyuyoruz
MIN_SEPARATION = int(new_sample_rate * 0.001)
preamble_indices = []
last_idx = -MIN_SEPARATION
for idx in candidate_indices:
    if idx - last_idx >= MIN_SEPARATION:
         preamble_indices.append(idx)
         last_idx = idx
print("Aday preamble indeksleri:", preamble_indices)

# =============================================================================
# 6. Bit Demodülasyonu
# =============================================================================
# ADS‑B için, TARGET_SAMPLE_RATE=2 MHz ve her bit 1 µs -> bit başına yaklaşık 2 örnek
samples_per_bit = int(new_sample_rate * BIT_DURATION_US * 1e-6)  # genellikle 2

# İnterpolasyon faktörü: 2 örnek üzerinden daha hassas karar için faktörü 4 ile (2*4 = 8 örnek)
INTERP_FACTOR = 4

def demodulate_message(envelope, start_idx, samples_per_bit, num_bits, interp_factor=4):
    """
    Belirtilen indeksten itibaren num_bits bit demodüle eder.
    Her bit penceresindeki 2 örnek, lineer interpolasyonla (örneğin, 8 örneğe çıkartılarak)
    iki yarının ortalamaları karşılaştırılır.
    """
    bits = ""
    for i in range(num_bits):
        s = start_idx + i * samples_per_bit
        e = s + samples_per_bit
        if e > len(envelope):
            return None
        bit_samples = envelope[s:e]
        # İnterpolasyon: orijinal örnekleri daha yüksek çözünürlükte ele alıyoruz.
        x_orig = np.arange(samples_per_bit)
        x_interp = np.linspace(0, samples_per_bit, num=samples_per_bit * interp_factor, endpoint=False)
        interp_vals = np.interp(x_interp, x_orig, bit_samples)
        half = len(interp_vals) // 2
        first_half = np.mean(interp_vals[:half])
        second_half = np.mean(interp_vals[half:])
        bits += "1" if first_half > second_half else "0"
    return bits

# =============================================================================
# 7. Mesaj Demodülasyonu ve Decode İşlemi (pyModeS ile)
# =============================================================================
messages = []
for preamble_idx in preamble_indices:
    base_msg_start = preamble_idx + PREAMBLE_LENGTH
    valid_found = False
    for offset in OFFSET_RANGE:
        msg_start = base_msg_start + offset
        if msg_start + BITS_PER_MESSAGE * samples_per_bit > len(envelope):
            continue
        bit_seq = demodulate_message(envelope, msg_start, samples_per_bit, BITS_PER_MESSAGE, interp_factor=INTERP_FACTOR)
        if bit_seq is None or len(bit_seq) != BITS_PER_MESSAGE:
            continue
        try:
            msg_int = int(bit_seq, 2)
            msg_hex = format(msg_int, '028X')
        except Exception as e:
            print("Hex dönüşüm hatası:", e)
            continue
        try:
            df = pms.common.df(msg_hex)
            crc_val = pms.common.crc(msg_hex)
        except Exception as e:
            print("pyModeS DF/CRC hesaplama hatası:", e)
            continue
        # Geçerli ADS‑B mesajı: DF 17 ve CRC 0 olmalı
        if df == 17 and crc_val == 0:
            try:
                icao = pms.adsb.icao(msg_hex)
            except Exception:
                icao = None
            try:
                tc = pms.adsb.tc(msg_hex)
            except Exception:
                tc = None
            messages.append({
                "preamble_idx": preamble_idx,
                "offset": offset,
                "bit_sequence": bit_seq,
                "msg_hex": msg_hex,
                "df": df,
                "icao": icao,
                "tc": tc
            })
            valid_found = True
            break  # Uygun ofset bulunduysa diğer denemelere gerek yok.
    if not valid_found:
        print(f"Preamble {preamble_idx}: Geçerli mesaj bulunamadı.")

# =============================================================================
# 8. Sonuçların Yazdırılması
# =============================================================================
if not messages:
    print("Geçerli DF=17 ve CRC=0 olan ADS-B mesajı tespit edilemedi.")
else:
    print("Tespit edilen geçerli ADS-B mesajları:")
    for msg in messages:
        print("\nPreamble Index:", msg["preamble_idx"])
        print("Offset (örnek):", msg["offset"])
        print("Hex Mesaj:", msg["msg_hex"])
        print("DF:", msg["df"], "ICAO:", msg["icao"], "TC:", msg["tc"])
