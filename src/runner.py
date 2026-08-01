"""Stepping a model forward, with optional live viewing."""

import time

import mujoco

from .viewer import apply_perturb

FRAME_TIME = 1.0 / 60.0


def run(m, d, control=None, duration=None, sample_hz=None, on_sample=None, view=False):
    """Step the model, optionally showing the viewer and sampling as it goes.

    control(m, d) writes into d.ctrl before each step. on_sample(m, d) returns
    one logged record and is called at sample_hz. Returns the list of records.
    A duration of None runs until the viewer is closed.
    """
    if duration is None and not view:
        raise ValueError("duration is required when the viewer is off")

    total_steps = None if duration is None else int(round(duration / m.opt.timestep))
    decimation = (
        max(1, int(round(1.0 / (sample_hz * m.opt.timestep)))) if sample_hz else None
    )
    samples = []

    def step(i):
        if control is not None:
            control(m, d)
        mujoco.mj_step(m, d)
        if decimation and on_sample is not None and i % decimation == 0:
            samples.append(on_sample(m, d))

    if not view:
        for i in range(total_steps):
            step(i)
        return samples

    from mujoco import viewer as mj_viewer

    # Draw one frame per FRAME_TIME, stepping physics in between until simulated
    # time has caught up with the wall clock.
    with mj_viewer.launch_passive(m, d) as viewer:
        i = 0
        start_wall, start_sim = time.time(), d.time
        while viewer.is_running() and (total_steps is None or i < total_steps):
            frame_end = time.time() + FRAME_TIME
            while time.time() < frame_end:
                if total_steps is not None and i >= total_steps:
                    break
                if d.time - start_sim > time.time() - start_wall:
                    break
                d.xfrc_applied[:] = 0.0
                apply_perturb(m, d, viewer)
                step(i)
                i += 1
            viewer.sync()
            lag = frame_end - time.time()
            if lag > 0:
                time.sleep(lag)
    return samples
