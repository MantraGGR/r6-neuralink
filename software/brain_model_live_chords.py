"""
brain_model_live_chords.py — Live single-channel EEG-driven 3D brain heatmap.

Uses the real Chords binary packet protocol (sync bytes 0xC7 0x7C, 6
channel slots streamed per packet) to drive one activity blob on the 3D
brain model. Only ONE channel (A0) carries real signal: this rig is a
single BioAmp EXG Pill with the standard 3-lead differential montage —
signal electrode on the forehead, reference behind the ear, ground at
another forehead point. The reference and ground leads are return paths
the amplifier needs to work at all, not independent recording channels,
so there is nothing meaningful on A1-A5 to visualize.

HONESTY NOTE (see CLAUDE.md): this does NOT localize activity to a real
anatomical source. The forehead electrode sits on the midline, not over
either hemisphere specifically, so the same live value is mirrored as a
frontal blob on BOTH hemispheres — a fixed, illustrative stand-in for
"the signal electrode is roughly here," not real source localization.
Only the blobs' shared INTENSITY reflects your real, live signal.

Setup:
    pip install pyserial numpy mne pyvistaqt
"""

import os
os.environ["MNE_3D_BACKEND"] = "pyvistaqt"

import time
from collections import deque
from pathlib import Path

import numpy as np
import mne
from matplotlib.colors import LinearSegmentedColormap

from chords_protocol import (
    BAND,
    SAMPLE_RATE_GUESS,
    SIGNAL_CHANNEL,
    band_power,
    connect_and_handshake,
    read_packets,
)

# ---- CONFIG ----
WINDOW_SECONDS = 1.5
UPDATE_HZ = 20
BLOB_SIGMA = 15
NORMALIZATION_WINDOW_SECONDS = 8    # how far back to look when rescaling power to 0-1
NORMALIZATION_PERCENTILES = (10, 90)   # robust "low"/"high" — ignores rare noise spikes
SENSITIVITY_GAMMA = 0.5             # <1 boosts mid-range values so moderate changes are visible

# Discrete "eyes closed" event, not a continuous dial: OFF until val crosses
# THRESHOLD_ON, then ON until it drops back below THRESHOLD_OFF. The gap
# between the two (hysteresis) stops it flickering on/off from noise
# sitting right at a single boundary.
THRESHOLD_ON = 0.65
THRESHOLD_OFF = 0.45
# -----------------

# Purple (baseline) -> red (peak activation) — much easier to see against
# the black background than "hot", which is near-invisible at low values.
ACTIVATION_CMAP = LinearSegmentedColormap.from_list("purple_red", ["#3b0764", "#ff0000"])

WINDOW_SIZE = int(SAMPLE_RATE_GUESS * WINDOW_SECONDS)


def make_blob(coords, center_idx, sigma):
    center = coords[center_idx]
    distance = np.linalg.norm(coords - center, axis=1)
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


def pick_vertex(coords, mode):
    """mode: 'frontal' = most anterior (+Y), 'posterior' = most posterior (-Y)"""
    if mode == "frontal":
        return int(np.argmax(coords[:, 1]))
    else:
        return int(np.argmin(coords[:, 1]))


def main():
    print("Loading brain model first (this can take a while on first run)...")
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
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
    print("Brain loaded.\n")

    # Only NOW open the serial connection — nothing was buffering during
    # the slow model load, so there's no backlog to flush when we start.
    ser = connect_and_handshake()
    raw_buffer = bytearray()

    # Electrode -> location mapping (illustrative, see honesty note above):
    # the forehead signal electrode sits on the midline, so its one real
    # channel is mirrored as a frontal blob on both hemispheres.
    lh_coords = brain.geo["lh"].coords
    rh_coords = brain.geo["rh"].coords

    blob_lh = make_blob(lh_coords, pick_vertex(lh_coords, "frontal"), BLOB_SIGMA)
    blob_rh = make_blob(rh_coords, pick_vertex(rh_coords, "frontal"), BLOB_SIGMA)

    # Add each hemisphere's data layer ONCE. Calling remove_data()+add_data()
    # on every tick (the original approach) tears down and rebuilds the full
    # ~164k-vertex mesh overlay each time — benchmarked at ~120ms per update,
    # i.e. under 10 fps and unable to keep up with UPDATE_HZ. Instead we keep
    # the same layer alive and push new scalar values into it directly via
    # mne's internal overlay update path (~20ms/update, the same call
    # brain.set_time_point() uses under the hood for animated data).
    brain.add_data(
        np.zeros(len(lh_coords)), hemi="lh", vertices=np.arange(len(lh_coords)),
        fmin=0.0, fmid=0.5, fmax=1.0, colormap=ACTIVATION_CMAP, alpha=1.0,
    )
    brain.add_data(
        np.zeros(len(rh_coords)), hemi="rh", vertices=np.arange(len(rh_coords)),
        fmin=0.0, fmid=0.5, fmax=1.0, colormap=ACTIVATION_CMAP, alpha=1.0,
    )

    # add_data() resets the camera to the model's default "lateral" view
    # (one hemisphere only) each time it's called, so this must happen
    # after the last add_data() call, not before. "frontal" (not "dorsal")
    # because the frontal-pole blob curves steeply out of view from directly
    # overhead — from the front it's fully, clearly visible.
    brain.show_view("frontal")

    def update_hemi_data(hemi, array):
        brain._data[hemi]["array"] = array
        brain._layered_meshes[hemi].update_overlay(name="data", scalars=array)
        # _update() -> plotter.update() just schedules a Qt repaint; it does
        # NOT call VTK's Render(). That's why dragging the mouse "wakes up"
        # the window (VTK's interactor calls Render() on mouse events) but
        # our own updates otherwise sit there until you interact with it.
        # plotter.render() calls render_window.Render() directly, so the
        # heatmap actually draws every tick without needing a click.
        brain._renderer._update()
        brain.plotter.render()
        brain.plotter.app.processEvents()

    # Rolling sample buffer for the one real channel, plus a rolling window
    # of recent power readings used to rescale to 0-1.
    channel_buffer = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
    power_history = deque(maxlen=int(NORMALIZATION_WINDOW_SECONDS * UPDATE_HZ))

    def normalized_power():
        signal = np.array(channel_buffer)
        power = band_power(signal, SAMPLE_RATE_GUESS, BAND)
        power_history.append(power)
        # A window that only ever expands (all-time min/max) gets permanently
        # blown out by a single noise spike — e.g. touching a wire on
        # startup — and everything real after that looks tiny by comparison
        # and renders as flat purple forever. Using only the last
        # NORMALIZATION_WINDOW_SECONDS lets old spikes age out so the scale
        # stays matched to your current baseline.
        #
        # Using the true min/max (rather than percentiles) also means one
        # rare noise spike sets the top of the scale, so real but smaller
        # changes barely move — percentiles ignore that outlier tail, and
        # the gamma curve then boosts mid-range values so they're clearly
        # visible instead of needing to hit the very top to look "lit up".
        lo, hi = np.percentile(power_history, NORMALIZATION_PERCENTILES)
        span = (hi - lo) or 1.0
        val = np.clip((power - lo) / span, 0.0, 1.0)
        return float(val ** SENSITIVITY_GAMMA)

    print("Starting live single-channel visualization.")
    print("Press 'P' in the brain window to pause/resume live updates. Ctrl+C in terminal to quit.")

    paused = {"value": False}

    def toggle_pause():
        paused["value"] = not paused["value"]
        state = "PAUSED" if paused["value"] else "RESUMED"
        print(f"[{state}] live updates {'stopped' if paused['value'] else 'running'}")

    # Bind the toggle to a keypress on the render window. MNE's Brain
    # exposes the underlying PyVista plotter slightly differently across
    # versions, so try both known attribute paths.
    try:
        brain.plotter.add_key_event("p", toggle_pause)
    except AttributeError:
        try:
            brain._renderer.plotter.add_key_event("p", toggle_pause)
        except Exception:
            print("Could not bind keyboard toggle — pause/resume via keypress won't work, "
                  "but the script will still run normally.")

    tick_interval = 1.0 / UPDATE_HZ
    last_update = 0.0
    tick_count = 0
    detected = False

    try:
        while True:
            packets = read_packets(ser, raw_buffer)

            if paused["value"]:
                # Still drain the buffer so it doesn't build up a backlog,
                # but don't feed it into the channel buffers or redraw.
                # Keep pumping Qt here too — otherwise the window freezes
                # solid, and the 'P' keypress needed to unpause would never
                # even get delivered.
                brain.plotter.app.processEvents()
                time.sleep(0.05)
                continue

            for channels in packets:
                channel_buffer.append(channels[SIGNAL_CHANNEL])

            now = time.time()
            if now - last_update < tick_interval:
                brain.plotter.app.processEvents()
                time.sleep(0.005)
                continue
            last_update = now

            val = normalized_power()

            # Discrete "eyes closed" event with hysteresis: once ON, val has
            # to drop all the way below THRESHOLD_OFF (not just back under
            # THRESHOLD_ON) before it counts as OFF again, so noise sitting
            # near one boundary doesn't make it flicker.
            if not detected and val >= THRESHOLD_ON:
                detected = True
                print(f">>> ACTION (val={val:.2f})")
            elif detected and val <= THRESHOLD_OFF:
                detected = False
                print(f"<<< back to baseline (val={val:.2f})")

            tick_count += 1
            if tick_count % UPDATE_HZ == 0:  # once a second
                raw = channel_buffer[-1]
                lo, hi = np.percentile(power_history, NORMALIZATION_PERCENTILES)
                print(f"A0={raw:>5}  alpha_power={power_history[-1]:.3e}  "
                      f"p{NORMALIZATION_PERCENTILES[0]}-p{NORMALIZATION_PERCENTILES[1]}=[{lo:.3e}, {hi:.3e}]  "
                      f"val={val:.2f}  detected={detected}")

            # Discrete indicator, not a continuous dial: full intensity when
            # detected, otherwise baseline. Mirrored on both hemispheres —
            # see honesty note at the top of this file re: the midline
            # forehead electrode.
            blob_val = 1.0 if detected else 0.0
            lh_combined = np.clip(blob_lh * blob_val, 0.0, 1.0)
            rh_combined = np.clip(blob_rh * blob_val, 0.0, 1.0)

            update_hemi_data("lh", lh_combined)
            update_hemi_data("rh", rh_combined)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ser.write(b"STOP\n")
        ser.close()

    input("Press ENTER to close the brain window...")


if __name__ == "__main__":
    main()