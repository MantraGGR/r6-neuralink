"""
EEG Live Reader — reads filtered EEG samples from the Arduino (running
Upside Down Labs' EEGFilter.ino) over serial, plots them live, and
computes rolling band power (delta/theta/alpha/beta) via FFT.

Setup:
    pip install pyserial numpy matplotlib scipy

Before running:
    1. Upload EEGFilter.ino to your Arduino (see BioAmp-EXG-Pill repo).
    2. Close the Arduino Serial Monitor/Plotter — only one program can
       read the serial port at a time.
    3. Find your port:
         Mac:  ls /dev/tty.usbmodem*   (or /dev/tty.usbserial*)
       Update SERIAL_PORT below to match.
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ---- CONFIG — edit these ----
SERIAL_PORT = "/dev/tty.usbmodem101"   # <-- change to your actual port
BAUD_RATE = 9600                       # matches EEGFilter.ino's Serial.begin()
SAMPLE_RATE = 256                        # Hz, per BioAmp EEGFilter.ino
WINDOW_SECONDS = 4                       # how much history to show/analyze
# ------------------------------

WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS
buffer = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}


def connect_serial():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected. Waiting for data...")
    return ser


def read_sample(ser):
    """Read one line, parse it as a float. Returns None if invalid."""
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "":
            return None
        return float(line)
    except (ValueError, UnicodeDecodeError):
        return None


def band_power(signal, fs, band):
    """Compute power in a given frequency band using FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    return np.sum(fft_vals[mask])


def main():
    ser = connect_serial()

    fig, (ax_wave, ax_bands) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Live EEG — BioAmp EXG Pill")

    line_wave, = ax_wave.plot(np.zeros(WINDOW_SIZE))
    ax_wave.set_title("Raw filtered signal")
    ax_wave.set_ylim(-500, 500)  # adjust based on your actual signal range

    band_names = list(BANDS.keys())
    bar_container = ax_bands.bar(band_names, [0] * len(band_names))
    ax_bands.set_title("Band power (relative)")
    ax_bands.set_ylim(0, 1)  # will auto-rescale below

    def update(frame):
        # Drain whatever samples are waiting in the serial buffer
        while ser.in_waiting:
            val = read_sample(ser)
            if val is not None:
                buffer.append(val)

        signal = np.array(buffer)
        line_wave.set_ydata(signal)

        # Compute band powers over the current window
        powers = [band_power(signal, SAMPLE_RATE, BANDS[b]) for b in band_names]
        total = sum(powers) or 1  # avoid divide-by-zero
        normalized = [p / total for p in powers]

        for bar, p in zip(bar_container, normalized):
            bar.set_height(p)
        ax_bands.set_ylim(0, max(normalized) * 1.2 if max(normalized) > 0 else 1)

        return [line_wave, *bar_container]

    ani = FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

    ser.close()


if __name__ == "__main__":
    main()