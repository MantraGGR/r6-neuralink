"""
debug_signal.py — Sanity-check the EEG signal without the 3D brain model.

Answers one question: is real data coming in on A0, and does it move the
way you'd expect (e.g. alpha power rising when you close your eyes)?
No mne/pyvistaqt here, so it starts instantly and can't be confused by
anything going wrong in the brain visualization itself.

Prints raw packet values to the terminal AND live-plots the A0 waveform +
rolling alpha-band power with matplotlib.

Setup:
    pip install pyserial numpy matplotlib
"""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from chords_protocol import (
    BAND,
    SAMPLE_RATE_GUESS,
    SIGNAL_CHANNEL,
    band_power,
    connect_and_handshake,
    read_packets,
)

WINDOW_SECONDS = 3
PRINT_EVERY_N_PACKETS = 32   # ~8 Hz of terminal output at 256 Hz sampling

WINDOW_SIZE = int(SAMPLE_RATE_GUESS * WINDOW_SECONDS)


def main():
    ser = connect_and_handshake()
    raw_buffer = bytearray()

    signal_buffer = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
    power_history = deque([0.0] * 100, maxlen=100)

    fig, (ax_wave, ax_power) = plt.subplots(2, 1, figsize=(9, 6))

    (wave_line,) = ax_wave.plot(np.zeros(WINDOW_SIZE))
    ax_wave.set_title(f"Raw A{SIGNAL_CHANNEL} waveform")
    # Auto-scaled rather than a fixed 0-4095: the Uno R4 Minima's ADC can
    # run at higher resolution than the classic 12-bit assumption, so a
    # hardcoded ceiling would clip or misrepresent the real range.

    (power_line,) = ax_power.plot(np.zeros(100))
    ax_power.set_title(f"Alpha band power ({BAND[0]}-{BAND[1]} Hz), rolling")

    plt.tight_layout()
    plt.show(block=False)

    packet_count = 0
    print("Streaming. Ctrl+C to stop.\n")
    print(f"{'packet #':>10}  {'A0 (raw)':>10}  {'all channels'}")

    try:
        while True:
            packets = read_packets(ser, raw_buffer)

            for channels in packets:
                packet_count += 1
                signal_buffer.append(channels[SIGNAL_CHANNEL])

                if packet_count % PRINT_EVERY_N_PACKETS == 0:
                    print(f"{packet_count:>10}  {channels[SIGNAL_CHANNEL]:>10}  {channels}")

            if not packets:
                continue

            power = band_power(np.array(signal_buffer), SAMPLE_RATE_GUESS, BAND)
            power_history.append(power)

            wave_line.set_ydata(signal_buffer)
            ax_wave.set_ylim(min(signal_buffer) - 10, max(signal_buffer) + 10)

            power_line.set_ydata(power_history)
            ax_power.set_ylim(0, max(power_history) * 1.1 or 1.0)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ser.write(b"STOP\n")
        ser.close()


if __name__ == "__main__":
    main()
