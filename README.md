# WheelBarrow

A 3D MuJoCo simulation platform for legged-wheeled robots, built for a distributed control
architecture in which a robot is decomposed into a body and legs, each leg interfacing with the
body through a wrench.

## Environment

- Python 3.10.11, project-local venv (`.venv/`, gitignored).
- Pinned dependencies (`requirements.txt`): `mujoco==3.11.0`, `numpy==2.2.6`, `matplotlib==3.10.9`.

## Layout

```
models/          # MJCF
src/             # extraction utilities, sim helpers
tests/           # verification tests (pytest)
scripts/         # runnable experiments (creep test, etc.)
logs/            # gitignored
```

## Models

### Wheeled biped

One robot currently defined in `models/`: a biped with two legs, each leg ending in a driven
wheel.

- Wheels are modeled as `sphere` geoms, giving a single clean contact point against the ground
  plane; cylinder and box geoms have contact-frame quirks that aren't worth fighting. Each wheel's
  `<inertial>` is set explicitly to the thin-disk formula (about the spin axis, and about the two
  transverse axes), rather than the solid-sphere inertia MuJoCo would otherwise compute from the
  sphere geom - the contact shape and the rotational inertia are deliberately decoupled.
- Box-shaped links (body, thigh, shin) have no defined thickness along the third (non-planar)
  axis; it is set equal to each link's own in-plane width. This choice doesn't affect the
  simulated dynamics: the inertia about the pitch axis depends only on the other two dimensions of
  the box, and the robot has no roll/yaw degrees of freedom to make the third axis reachable.
- No joint is angle-limited and no actuator has a ctrlrange/forcerange.
- Contact restitution is not modeled; MuJoCo's default inelastic contact is used.
