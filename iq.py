#!/usr/bin/env python3
# iq_adsb.py — ADS‑B IQ işleme adım adım

import argparse
import numpy as np
from scipy.signal import firwin, lfilter, fftconvolve, find_peaks
import pyModeS as pms
import pandas as pd

def read_iq(path, dtype=np.complex64):
    iq = np.fromfile(path, dtype=dtype)
    if dtype != np.complex64:
        iq = iq.astype(np.float32).view(np.complex64)
    return iq

def bandpass_filter(iq, fs, f_low=100e3, f_high=None, numtaps=129):
    nyq = fs/2
    if f_high is None or f_high >= nyq:
        f_high = nyq * 0.9
    if f_low is None or f_low <= 0:
        taps = firwin(numtaps, f_high, fs=fs)
    else:
        taps = firwin(numtaps, [f_low, f_high], pass_zero=False, fs=fs)
    return lfilter(taps, 1.0, iq)

def detect_preamble(env, fs):
    preamble = np.array([1,0,1,0,1,0,1,0], dtype=float)
    corr = fftconvolve(env, preamble[::-1], mode='valid')
    thr = np.max(corr)*0.8
    dist = int(8 * fs * 1e-6)
    peaks, _ = find_peaks(corr, height=thr, distance=dist)
    return peaks

def demod_bits(env, pre_idxs, fs, thr):
    spb = int(fs * 1e-6)
    msgs = []
    for idx in pre_idxs:
        start = idx + int(8 * fs * 1e-6)
        bits = []
        for i in range(112):
            s = start + i*spb
            bits.append(1 if s < len(env) and env[s] > thr else 0)
        msgs.append(bits)
    return msgs

def main():
    parser = argparse.ArgumentParser(description="ADS‑B IQ işleme adım adım")
    parser.add_argument("filepath", nargs="?", help="IQ dosyasının tam yolu")
    parser.add_argument("--dtype", choices=["c64","c16","c8"], default="c64",
                        help="c64=complex64, c16=int16, c8=int8")
    parser.add_argument("--fs", type=float, default=2e6, help="Örnekleme hızı (Hz)")
    parser.add_argument("--f_low", type=float, default=100e3, help="Filtre alt (Hz)")
    parser.add_argument("--f_high", type=float, default=None, help="Filtre üst (Hz)")
    args = parser.parse_args()

    if not args.filepath:
        args.filepath = input("IQ dosyasının tam yolunu girin: ")

    # 1) IQ oku
    dtype_map = {"c64": np.complex64, "c16": np.int16, "c8": np.int8}
    iq = read_iq(args.filepath, dtype_map[args.dtype])
    print(f"[1] IQ okundu: {len(iq)} örnek, fs={args.fs/1e6:.1f} MHz")

    # 2) Filtre uygula
    filt = bandpass_filter(iq, args.fs, f_low=args.f_low, f_high=args.f_high)
    hi = (args.f_high or args.fs/2*0.9)/1e6
    print(f"[2] Filtre uygulandı: {args.f_low/1e6:.1f}–{hi:.1f} MHz")

    # 3) Zarf çıkar
    env = np.abs(filt)
    print("[3] Zarf çıkarıldı.")

    # 4) Dinamik eşik
    mu, sigma = env.mean(), env.std()
    thr = mu + 3*sigma
    print(f"[4] Eşik belirlendi: {thr:.3f} (μ={mu:.3f}, σ={sigma:.3f})")

    # 5) Preamble tespiti
    pre_idxs = detect_preamble(env, args.fs)
    print("[5] Preamble indeksleri:", pre_idxs[:5])

    # 6) Bit demodülasyonu
    msgs = demod_bits(env, pre_idxs, args.fs, thr)
    print(f"[6] {len(msgs)} mesaj dizisi elde edildi.")

    # 7) CRC kontrolü ve parse
    print("[7] Geçerli mesajlar işleniyor...")
    parsed = []
    for bits in msgs:
        # 112 bit → 14 byte
        raw_bytes = bytes(int("".join(str(b) for b in bits[i:i+8]), 2)
                          for i in range(0, 112, 8))
        # hex string’e dönüştür
        raw_hex = raw_bytes.hex().upper()
        # CRC sıfır mı?
        if pms.crc(raw_hex) != 0:
            continue
        try:
            rec = pms.parse(raw_hex)
            if rec:
                parsed.append(rec)
        except Exception:
            pass

    # 8) Sonuçları CSV’ye yaz
    if parsed:
        df = pd.DataFrame(parsed)
        df.to_csv("adsb_messages.csv", index=False)
        print(f"[8] {len(df)} kayıt adsb_messages.csv dosyasına yazıldı.")
    else:
        print("[8] Geçerli mesaj bulunamadı.")

if __name__ == "__main__":
    main()
