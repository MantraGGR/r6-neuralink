
"""
brain_model_live.py — Live EEG-driven 3D brain heatmap.
 
Reads real-time filtered EEG from the Arduino (running EEGFilter.ino) and
drives the intensity of a fixed activity "blob" on the 3D brain model.
 
HONESTY NOTE (see CLAUDE.md): this does NOT localize activity to a real
anatomical source. With 1-4 electrodes there's no way to do real source
localization. The blob is placed at a fixed point approximating "near the
forehead electrode" and its INTENSITY (not position) reflects your real,
live signal. This is illustrative, not diagnostic.
 
Setup:
    pip install pyserial numpy mne pyvistaqt
 
Before running:
    1. Upload EEGFilter.ino to your Arduino, close Serial Monitor/Plotter.
    2. Find your port: `ls /dev/tty.usbmodem*` (Mac)
    3. Update SERIAL_PORT below.
"""
 
import os
os.environ["MNE_3D_BACKEND"] = "pyvistaqt"
 
import time
from collections import deque
from pathlib import Path
 
import numpy as np
import serial
import mne
 
# ---- CONFIG — edit these ----
SERIAL_PORT = "/dev/tty.usbmodem101"   # <-- your Arduino's port
BAUD_RATE = 115200                       # must match EEGFilter.ino's Serial.begin()
SAMPLE_RATE = 256                        # Hz, per EEGFilter.ino
WINDOW_SECONDS = 1.5                      # rolling window used for band power
UPDATE_HZ = 10                            # how often the brain redraws (keep modest)
BLOB_SIGMA = 15                           # spatial "spread" of the activity blob
BAND = (8, 13)                            # alpha band, Hz — swap for another band if you like
# ------------------------------
 
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
 
 
def connect_serial():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected. Waiting for data...")
    return ser
 
 
def read_sample(ser):
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        return float(line) if line else None
    except (ValueError, UnicodeDecodeError):
        return None
 
 
def band_power(signal, fs, band):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    return np.sum(fft_vals[mask])
 
 
def pick_frontal_vertex(coords):
    """Pick the most anterior (forward-facing) vertex as our stand-in
    for 'near the forehead electrode'. Approximate, not clinically real."""
    return int(np.argmax(coords[:, 1]))  # +Y is anterior in surface RAS
 
 
def main():
    ser = connect_serial()
 
    print("Loading brain model...")
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = Path(fs_dir).parent
 
    brain = mne.viz.Brain(
        "fsaverage",
        hemi="both",
        surf="inflated",
        subjects_dir=subjects_dir,
        cortex="low_contrast",
        background="black",
        size=1200,
    )
    brain.show_view("dorsal")
    print("Brain loaded.")
 
    # Precompute a fixed Gaussian "shape" per hemisphere, centered near
    # the frontal pole. We'll scale this shape by the live signal each tick.
    blob_shapes = {}
    for hemi in ("lh", "rh"):
        coords = brain.geo[hemi].coords
        center_idx = pick_frontal_vertex(coords)
        center = coords[center_idx]
        distance = np.linalg.norm(coords - center, axis=1)
        shape = np.exp(-(distance ** 2) / (2 * BLOB_SIGMA ** 2))
        blob_shapes[hemi] = (coords, shape)
 
    buffer = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
 
    # Rolling min/max for normalization — auto-calibrates to your signal
    # range over time rather than assuming fixed units.
    running_min, running_max = None, None
 
    print("Starting live visualization. Close the eyes / try focusing to see it react.")
    print("Press Ctrl+C in this terminal to stop.")
 
    tick_interval = 1.0 / UPDATE_HZ
    last_update = 0.0
 
    added_data = False
 
    try:
        while True:
            # Drain whatever's waiting on serial
            while ser.in_waiting:
                val = read_sample(ser)
                if val is not None:
                    buffer.append(val)
 
            now = time.time()
            if now - last_update < tick_interval:
                time.sleep(0.005)
                continue
            last_update = now
 
            signal = np.array(buffer)
            power = band_power(signal, SAMPLE_RATE, BAND)
 
            # Update rolling normalization range
            if running_min is None or power < running_min:
                running_min = power
            if running_max is None or power > running_max:
                running_max = power
            span = (running_max - running_min) or 1.0
            normalized = (power - running_min) / span
            normalized = float(np.clip(normalized, 0.0, 1.0))
 
            # Redraw both hemispheres scaled by the live value.
            if added_data:
                brain.remove_data()
            for hemi, (coords, shape) in blob_shapes.items():
                scaled = shape * normalized
                brain.add_data(
                    scaled,
                    hemi=hemi,
                    vertices=np.arange(len(coords)),
                    fmin=0.0,
                    fmid=0.5,
                    fmax=1.0,
                    colormap="hot",
                    alpha=1.0,
                )
            added_data = True
 
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ser.close()
 
    input("Press ENTER to close the brain window...")
 
 
if __name__ == "__main__":
    main()
 
