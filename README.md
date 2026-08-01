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

## Running

Experiments are plain scripts, run with the venv interpreter:

```
.venv\Scripts\python.exe scripts\creep_test.py
```

Each experiment has a `VIEW` constant at the top. Setting it to `True` opens the interactive
viewer and paces the run to wall-clock time, so a 30 s experiment takes 30 s of real time.
In the viewer, double-click a body to select it, then ctrl-right-drag to push it or
ctrl-left-drag to twist it.

`scripts/replay.py` plays a logged run back in the viewer without simulating anything. It reads
the model name from the start of the log filename, so logs stay tied to the robot that produced
them.

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

## Sign conventions

`src/dynamics.py` reads the terms of MuJoCo's equation of motion:

```
M(q) * qacc + qfrc_bias = qfrc_applied + J(q)^T * f
```

Two properties of this form are worth stating, because both are easy to assume wrongly:

- `qfrc_bias` carries the Coriolis, centrifugal **and** gravitational terms together in one
  vector. There is no separate gravity term to read.
- It sits on the same side as the inertial term, not with the applied forces. A robot at rest
  under no actuation therefore accelerates as `qacc = -M^-1 * qfrc_bias`, and holding it still
  takes `qfrc_applied = qfrc_bias` rather than its negative.

Each of these signs is pinned by a test in `tests/test_dynamics.py`:

| what it fixes | test |
| --- | --- |
| `qfrc_bias` on the left, `qfrc_applied` on the right | `test_applied_force_cancelling_bias_leaves_no_acceleration` |
| that the above is not satisfied by the opposite sign | `test_wrong_sign_on_bias_does_accelerate` |
| `J` as the derivative of a world position with respect to the generalized coordinates, so `J^T` maps world forces to joint efforts | `test_jacobian_matches_finite_differences` |
| `M` symmetric and positive definite | `test_mass_matrix_is_symmetric_and_positive_definite` |
| gravity conservative and consistent with the above | `test_energy_is_conserved_without_contact_or_actuation` |

The second row is what makes the first meaningful: cancelling the bias is only evidence of a
correct sign if the opposite sign demonstrably fails, so that test asserts the wrong sign produces
a large acceleration.

Energy conservation is measured under a fourth-order integrator, because conservation is a
property of the dynamics rather than of the integrator. The first-order integrator the model runs
with loses a few percent of total energy over five seconds, scaling linearly with the timestep.
