"""
chords_protocol.py — Shared serial protocol code for the Upside Down Labs
"Chords" binary packet format (sync bytes 0xC7 0x7C, 6 channel slots per
packet). Used by both brain_model_live_chords.py and debug_signal.py so
the two can never disagree about how a packet is parsed.

Only SIGNAL_CHANNEL (A0) carries real signal on this rig — see the
honesty note in brain_model_live_chords.py for why.
"""

import time

import numpy as np
import serial

# ---- CONFIG ----
SERIAL_PORT = "/dev/tty.usbmodem1101"    # <-- update if it drifts
BAUD_RATE = 230400
NUM_CHANNELS = 6                         # confirmed from raw packet structure (A0-A5)
SIGNAL_CHANNEL = 0                       # A0 — the only lead with real signal
SYNC_BYTE1 = 0xC7
SYNC_BYTE2 = 0x7C
END_BYTE = 0x01
HEADER_LENGTH = 3
PACKET_LENGTH = NUM_CHANNELS * 2 + HEADER_LENGTH + 1

SAMPLE_RATE_GUESS = 256
BAND = (8, 13)   # alpha band
# -----------------


def connect_and_handshake():
    print(f"Opening {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(b"WHORU\n")
    time.sleep(0.3)
    ser.read(ser.in_waiting or 1)  # drain handshake response, don't need it
    ser.write(b"START\n")
    time.sleep(0.3)
    print("Connected and streaming.")
    return ser


def read_packets(ser, raw_buffer):
    if ser.in_waiting:
        raw_buffer.extend(ser.read(ser.in_waiting))

    packets = []
    sync_pattern = bytes([SYNC_BYTE1, SYNC_BYTE2])

    while True:
        sync_index = raw_buffer.find(sync_pattern)  # fast C-level search, not a Python loop

        if sync_index == -1:
            # No sync anywhere in the buffer. Keep the last byte in case it's
            # half of a split sync pattern, drop the rest.
            if len(raw_buffer) > 1:
                del raw_buffer[:-1]
            break

        if sync_index > 0:
            del raw_buffer[:sync_index]  # drop junk before the sync

        if len(raw_buffer) < PACKET_LENGTH:
            break  # not enough bytes yet for a full packet — wait for more

        if raw_buffer[PACKET_LENGTH - 1] == END_BYTE:
            packet = raw_buffer[:PACKET_LENGTH]
            channels = []
            for ch in range(NUM_CHANNELS):  # fixed, tiny loop (6 iterations) — negligible cost
                high = packet[ch * 2 + HEADER_LENGTH]
                low = packet[ch * 2 + HEADER_LENGTH + 1]
                channels.append((high << 8) | low)
            packets.append(channels)
            del raw_buffer[:PACKET_LENGTH]
        else:
            # Sync bytes matched by coincidence but end byte didn't line up —
            # skip past this false match and keep searching.
            del raw_buffer[:2]

    return packets


def band_power(signal, fs, band):
    n = len(signal)
    if n < 8:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    return np.sum(fft_vals[mask])
