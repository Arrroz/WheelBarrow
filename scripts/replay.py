"""Replay a logged experiment in the viewer.

Reads the joint columns of a CSV written by an experiment and plays the
recorded motion back. Nothing is simulated, so what you see is exactly the
data that was analysed.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import model, viewer

LOGS = pathlib.Path(__file__).resolve().parents[1] / "logs"

LOG = "wheeled_biped_creep_A.csv"
SPEED = 1.0


def model_name_from(path):
    """The model whose name the log filename starts with."""
    stems = sorted((p.stem for p in model.MODELS.glob("*.xml")), key=len, reverse=True)
    for stem in stems:
        if path.stem.startswith(stem):
            return stem
    raise SystemExit(f"{path.name} does not start with a known model name")


def main():
    path = LOGS / LOG
    data = np.genfromtxt(path, delimiter=",", names=True)

    m, d = model.load(model_name_from(path))
    names = model.joint_names(m)
    missing = [n for n in names if n not in data.dtype.names]
    if missing:
        raise SystemExit(f"{path.name} has no columns for joints: {missing}")

    qpos = np.column_stack([data[n] for n in names])
    if qpos.shape[1] != m.nq:
        raise SystemExit(
            f"log holds {qpos.shape[1]} joint values but the model has {m.nq} "
            "coordinates; the log cannot describe a full pose"
        )
    print(f"replaying {path.name}: {len(qpos)} frames, {data['t'][-1]:.1f} s")
    viewer.replay(m, d, data["t"], qpos, speed=SPEED)


if __name__ == "__main__":
    main()
