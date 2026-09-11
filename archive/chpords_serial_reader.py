"""
chords_serial_reader.py — reads the ACTUAL binary packet protocol used by
the board (reverse-engineered from Chords-Web's Connection.tsx), instead
of assuming plain-text lines.

Protocol:
    SYNC_BYTE1 = 0xC7
    SYNC_BYTE2 = 0x7C
    HEADER_LENGTH = 3      (sync1, sync2, counter)
    END_BYTE = 0x01
    Each channel = 2 bytes, big-endian: (high << 8) | low
    PACKET_LENGTH = NUM_CHANNELS * 2 + HEADER_LENGTH + 1

Handshake (matches Chords-Web):
    1. Open port at 230400 baud
    2. Send "WHORU\n", read the device's name response
    3. Send "START\n" to begin streaming
    4. (Send "STOP\n" when done, if you want a clean shutdown)

Setup:
    pip install pyserial numpy matplotlib
"""

import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ---- CONFIG ----
SERIAL_PORT = "/dev/tty.usbmodem101"   # <-- update if it drifts again
BAUD_RATE = 230400
NUM_CHANNELS = 6                        # confirmed from raw packet structure (A0-A5)
SIGNAL_CHANNELS = [0, 1, 2]             # A0, A1, A2 — your three BioAmp Pills
SYNC_BYTE1 = 0xC7
SYNC_BYTE2 = 0x7C
END_BYTE = 0x01
HEADER_LENGTH = 3
PACKET_LENGTH = NUM_CHANNELS * 2 + HEADER_LENGTH + 1
WINDOW_SECONDS = 4
SAMPLE_RATE_GUESS = 256   # used only for band-power display scaling
# -----------------

WINDOW_SIZE = SAMPLE_RATE_GUESS * WINDOW_SECONDS
buffers_plot = {ch: deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for ch in SIGNAL_CHANNELS}

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def connect_and_handshake():
    print(f"Opening {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # let the board reset after opening the port (normal Arduino behavior)

    ser.reset_input_buffer()
    ser.write(b"WHORU\n")
    time.sleep(0.3)
    who = ser.read(ser.in_waiting or 1)
    print(f"Device responded to WHORU: {who!r}")

    ser.write(b"START\n")
    time.sleep(0.3)
    print("Sent START. Streaming should begin now.")
    return ser


def band_power(signal, fs, band):
    n = len(signal)
    if n < 8:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    return np.sum(fft_vals[mask])


def read_packets(ser, raw_buffer):
    """Read whatever bytes are available, parse any complete packets found,
    return a list of channel-value tuples for each valid packet."""
    if ser.in_waiting:
        raw_buffer.extend(ser.read(ser.in_waiting))

    packets = []
    while len(raw_buffer) >= PACKET_LENGTH:
        # find sync bytes
        sync_index = None
        for i in range(len(raw_buffer) - 1):
            if raw_buffer[i] == SYNC_BYTE1 and raw_buffer[i + 1] == SYNC_BYTE2:
                sync_index = i
                break

        if sync_index is None:
            raw_buffer.clear()
            break

        if sync_index + PACKET_LENGTH > len(raw_buffer):
            # not enough bytes yet for a full packet starting here
            if sync_index > 0:
                del raw_buffer[:sync_index]
            break

        end_index = sync_index + PACKET_LENGTH - 1
        if raw_buffer[end_index] == END_BYTE:
            packet = raw_buffer[sync_index:sync_index + PACKET_LENGTH]
            counter = packet[2]
            channels = []
            for ch in range(NUM_CHANNELS):
                high = packet[ch * 2 + HEADER_LENGTH]
                low = packet[ch * 2 + HEADER_LENGTH + 1]
                value = (high << 8) | low
                channels.append(value)
            packets.append((counter, channels))
            del raw_buffer[:end_index + 1]
        else:
            del raw_buffer[:sync_index + 1]

    return packets


def main():
    ser = connect_and_handshake()
    raw_buffer = bytearray()

    fig, axes = plt.subplots(len(SIGNAL_CHANNELS) + 1, 1, figsize=(10, 9))
    fig.suptitle("Live signal — Chords protocol (3 channels)")

    wave_axes = axes[:-1]
    ax_bands = axes[-1]

    lines = {}
    for ax, ch in zip(wave_axes, SIGNAL_CHANNELS):
        line, = ax.plot(np.zeros(WINDOW_SIZE))
        ax.set_title(f"Channel A{ch} (raw)")
        lines[ch] = line

    band_names = list(BANDS.keys())
    bar_container = ax_bands.bar(band_names, [0] * len(band_names))
    ax_bands.set_title("Channel A0 band power (relative)")
    ax_bands.set_ylim(0, 1)

    def update(frame):
        packets = read_packets(ser, raw_buffer)
        for counter, channels in packets:
            for ch in SIGNAL_CHANNELS:
                buffers_plot[ch].append(channels[ch])

        artists = []
        for ch in SIGNAL_CHANNELS:
            signal = np.array(buffers_plot[ch])
            lines[ch].set_ydata(signal)
            ax = wave_axes[SIGNAL_CHANNELS.index(ch)]
            if len(signal):
                ax.set_ylim(signal.min() - 10, signal.max() + 10)
            artists.append(lines[ch])

        # Band power computed from the first channel (A0) as the primary signal
        primary_signal = np.array(buffers_plot[SIGNAL_CHANNELS[0]])
        powers = [band_power(primary_signal, SAMPLE_RATE_GUESS, BANDS[b]) for b in band_names]
        total = sum(powers) or 1
        normalized = [p / total for p in powers]
        for bar, p in zip(bar_container, normalized):
            bar.set_height(p)
        ax_bands.set_ylim(0, max(normalized) * 1.2 if max(normalized) > 0 else 1)
        artists.extend(bar_container)

        return artists

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    ser.write(b"STOP\n")
    ser.close()


if __name__ == "__main__":
    main()