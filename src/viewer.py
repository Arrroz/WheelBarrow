"""Interactive viewing and replay.

An experiment shown in the viewer is paced to wall-clock time, so a 30 s run
takes 30 s of real time.
"""

import time

import mujoco
import numpy as np


def apply_perturb(m, d, viewer):
    """Feed the viewer's mouse perturbation into the dynamics."""
    pert = viewer.perturb
    if pert.select > 0:
        # the perturbation force scales with localmass; zero means no force
        pert.localmass = m.body_mass[pert.select]
    mujoco.mjv_applyPerturbForce(m, d, pert)


def replay(m, d, times, qpos, speed=1.0):
    """Play back a recorded qpos trajectory. No physics is stepped."""
    from mujoco import viewer as mj_viewer

    times = np.asarray(times, dtype=float)
    with mj_viewer.launch_passive(m, d) as viewer:
        started = time.time()
        for t, q in zip(times, qpos):
            if not viewer.is_running():
                break
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            viewer.sync()
            lag = t / speed - (time.time() - started)
            if lag > 0:
                time.sleep(lag)
