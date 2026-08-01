"""Verification of the dynamics quantities in src/dynamics.py.

The getters themselves are thin wrappers; these tests are the point of the
module. They check the jacobian against finite differences, the mass matrix
against its defining properties, the sign convention of the equation of motion,
and energy conservation.
"""

import pathlib
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import dynamics, model

MODEL = "wheeled_biped"
BODY = "rear_wheel"

ENERGY_TOLERANCE = 1e-9  # relative drift over 5 s


def loaded():
    m, d = model.load(MODEL)
    # every joint of this model has one coordinate, so a qpos index and a
    # velocity index refer to the same joint
    assert m.nq == m.nv
    return m, d


def free_floating():
    """The model with contacts off, so only the articulated dynamics act."""
    m, d = loaded()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d.ctrl[:] = 0.0
    return m, d


def test_jacobian_matches_finite_differences():
    m, d = loaded()
    body_id = m.body(BODY).id
    rng = np.random.default_rng(0)
    step = 1e-6

    for _ in range(3):
        d.qpos[:] = rng.uniform(-1.0, 1.0, m.nq)
        mujoco.mj_forward(m, d)
        analytic, _ = dynamics.jacobian(m, d, body=BODY)

        numeric = np.zeros_like(analytic)
        reference = d.qpos.copy()
        for i in range(m.nv):
            d.qpos[i] = reference[i] + step
            mujoco.mj_kinematics(m, d)
            ahead = d.xpos[body_id].copy()

            d.qpos[i] = reference[i] - step
            mujoco.mj_kinematics(m, d)
            behind = d.xpos[body_id].copy()

            d.qpos[i] = reference[i]
            numeric[:, i] = (ahead - behind) / (2.0 * step)

        assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-9)


def test_jacobian_needs_exactly_one_target():
    m, d = loaded()
    with pytest.raises(ValueError):
        dynamics.jacobian(m, d)
    with pytest.raises(ValueError):
        dynamics.jacobian(m, d, body=BODY, site=BODY)


def test_mass_matrix_is_symmetric_and_positive_definite():
    m, d = loaded()
    rng = np.random.default_rng(1)

    for _ in range(5):
        d.qpos[:] = rng.uniform(-np.pi, np.pi, m.nq)
        mujoco.mj_forward(m, d)
        mass = dynamics.mass_matrix(m, d)

        assert mass.shape == (m.nv, m.nv)
        assert np.allclose(mass, mass.T, atol=1e-12)
        np.linalg.cholesky(mass)  # raises unless positive definite


def test_applied_force_cancelling_bias_leaves_no_acceleration():
    """M*qacc + qfrc_bias = qfrc_applied, so equal terms must give qacc = 0."""
    m, d = free_floating()
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    d.qfrc_applied[:] = dynamics.bias_forces(m, d)
    mujoco.mj_step(m, d)

    assert np.max(np.abs(d.qacc)) < 1e-9


def test_wrong_sign_on_bias_does_accelerate():
    """The cancellation above only means something if the opposite sign fails."""
    m, d = free_floating()
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    d.qfrc_applied[:] = -dynamics.bias_forces(m, d)
    mujoco.mj_step(m, d)

    assert np.max(np.abs(d.qacc)) > 1.0


def test_energy_is_conserved_without_contact_or_actuation():
    """Conservation is a property of the dynamics, so it is measured with an
    integrator accurate enough not to dominate the result. The integrator the
    model ships with is first order and loses a few percent over this span.
    """
    m, d = free_floating()
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    assert np.all(m.dof_damping == 0.0)

    d.qvel[:] = 0.3
    mujoco.mj_forward(m, d)
    before = d.energy.sum()

    for _ in range(int(round(5.0 / m.opt.timestep))):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    after = d.energy.sum()

    assert abs(after - before) / abs(before) < ENERGY_TOLERANCE
