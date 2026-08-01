"""Dynamics quantities of a model in its current state.

Every function here reads what mujoco has already computed, so the caller must
have brought the state up to date with mj_forward or mj_step first.

The equation of motion is

    M(q) * qacc + qfrc_bias = qfrc_applied + J(q)^T * f

where qfrc_bias holds the Coriolis, centrifugal and gravitational terms
together, and f are the constraint forces.
"""

import mujoco
import numpy as np


def jacobian(m, d, body=None, site=None):
    """Translational and rotational jacobians of a body frame or a site.

    Each is 3 x nv, mapping generalized velocities to linear and angular
    velocity in world coordinates. Exactly one of body or site is expected.
    """
    if (body is None) == (site is None):
        raise ValueError("pass exactly one of body or site")

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    if site is not None:
        mujoco.mj_jacSite(m, d, jacp, jacr, m.site(site).id)
    else:
        mujoco.mj_jacBody(m, d, jacp, jacr, m.body(body).id)
    return jacp, jacr


def mass_matrix(m, d):
    """The nv x nv mass matrix, expanded from mujoco's sparse storage."""
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    return full


def bias_forces(m, d):
    """Coriolis, centrifugal and gravitational forces, as one nv vector."""
    return d.qfrc_bias.copy()
