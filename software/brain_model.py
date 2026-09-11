import os
os.environ["MNE_3D_BACKEND"] = "pyvistaqt"

import numpy as np
import mne
from pathlib import Path
import time
from matplotlib.colors import LinearSegmentedColormap

# Purple (baseline) -> red (peak activation) — much easier to see against
# the black background than "hot", which is near-invisible at low values.
ACTIVATION_CMAP = LinearSegmentedColormap.from_list("purple_red", ["#3b0764", "#ff0000"])

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

print("Brain loaded.")

def make_frames(coords, source_indices):
    """Build a (n_vertices, n_times) activity array for one hemisphere."""
    frames = np.empty((len(coords), len(source_indices)))
    for i, src_idx in enumerate(source_indices):
        center = coords[src_idx]
        distance = np.linalg.norm(coords - center, axis=1)
        activity = np.exp(-(distance ** 2) / (2 * 30 ** 2))
        frames[:, i] = activity ** 0.5
    return frames

n_steps = 100

for hemi in ("lh", "rh"):
    coords = brain.geo[hemi].coords
    source_indices = np.linspace(0, len(coords) - 1, n_steps).astype(int)
    frames = make_frames(coords, source_indices)

    brain.add_data(
        frames,
        hemi=hemi,
        vertices=np.arange(len(coords)),
        fmin=0.0,
        fmid=0.5,
        fmax=1.0,
        colormap=ACTIVATION_CMAP,
        alpha=1.0,
        time=np.arange(n_steps),
    )

print("🔥 Activity source added to both hemispheres.")

# add_data() resets the camera to the model's default "lateral" view each
# time it's called, so this must happen after the last add_data() call —
# set_time_point() below doesn't touch the camera, so it sticks from here on.
brain.show_view("dorsal")

print("Starting neural activity simulation...")

for i in range(n_steps):
    brain.set_time_point(i)
    # set_time_point() updates the mesh data but never calls VTK's Render()
    # itself, and Qt only repaints when its event loop runs — neither
    # happens on its own in a tight loop like this, so the window would
    # otherwise sit frozen showing stale data until you drag/click it
    # (mouse interaction is what was making it "wake up" and redraw).
    brain.plotter.render()
    brain.plotter.app.processEvents()
    time.sleep(0.05)

input("Press ENTER to close...")