"""Controller that holds the body at a reference pose through its legs."""

import mujoco
import numpy as np


class BodyController:

    def __init__(self, m, d, leg_controllers, kp=20, ki=0, kd=20):
        self.leg_controllers = leg_controllers
        self.kp, self.ki, self.kd = kp, ki, kd

        self.body = m.body("body").id
        self.ref_pos = d.xipos[self.body].copy()
        self.ref_quat = d.xquat[self.body].copy()
        self.mass = m.body_mass[self.body]
        self.integral = np.zeros(6)

    def __call__(self, m, d):
        b = self.body

        # Find pose and velocity errors
        quat_inv, quat_err, rot_err = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(quat_inv, d.xquat[b])
        mujoco.mju_mulQuat(quat_err, self.ref_quat, quat_inv)
        mujoco.mju_quat2Vel(rot_err, quat_err, 1)
        error = np.concatenate((self.ref_pos - d.xipos[b], rot_err))

        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        w = vel[:3]
        derror = -np.concatenate((vel[3:], w))

        # Get reference acceleration as the output of a PID controller
        self.integral += error * m.opt.timestep
        acc_ref = self.kp * error + self.ki * self.integral + self.kd * derror

        # Get required wrench as a function of the reference acceleration
        R = d.ximat[b].reshape(3, 3)
        moi = R @ np.diag(m.body_inertia[b]) @ R.T
        wrench_ref = np.concatenate((
            self.mass * (acc_ref[:3] - m.opt.gravity),
            moi @ acc_ref[3:] + np.cross(w, moi @ w),
        ))

        # Solve the wrench distribution problem
        nLegs = len(self.leg_controllers)
        JL = np.zeros((3 * nLegs, 6 * nLegs))
        kL = np.zeros((3 * nLegs,))
        for i, lc in enumerate(self.leg_controllers):
            lc.update_matrices(m, d)
            Jl = np.block([lc.link_jacobians[-1].T, lc.rotation_jacobians[-1].T])
            JL[3 * i:3 * (i + 1), 6 * i:6 * (i + 1)] = Jl[:3, :]

            kl = lc.gravity_efforts - lc.cc_matrix @ lc.joint_velocities
            kL[3 * i:3 * (i + 1)] = kl[:3]

        gamma = np.tile(np.eye(6), nLegs)
        lsVec = np.concatenate((wrench_ref, kL))
        lsMat = np.concatenate([gamma, JL])
        wrenches, _, _, _ = np.linalg.lstsq(lsMat, lsVec, rcond=None)

        for i, lc in enumerate(self.leg_controllers):
            wrench = wrenches[6 * i:6 * (i + 1)]
            lc.update(d, wrench)
