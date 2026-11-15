"""Controller that holds the torso at a reference pose through its legs."""

import numpy as np


class BodyController:

    def __init__(self, m, d, leg_controllers, kp=20, ki=0, kd=20):
        self.leg_controllers = leg_controllers
        self.kp, self.ki, self.kd = kp, ki, kd

        base = [m.joint(n) for n in ("base_x", "base_z", "base_pitch")]
        self.qadr = [j.qposadr[0] for j in base]
        self.dofadr = [j.dofadr[0] for j in base]
        self.ref = d.qpos[self.qadr].copy()

        torso = m.body("torso").id
        self.mass = m.body_mass[torso]
        self.moi = m.body_inertia[torso][1]
        self.integral = np.zeros(3)

    def __call__(self, m, d):
        # Find pose and velocity errors
        error = self.ref - d.qpos[self.qadr]
        derror = -d.qvel[self.dofadr]

        # Get reference acceleration as the output of a PID controller
        self.integral += error * m.opt.timestep
        acc_ref = self.kp * error + self.ki * self.integral + self.kd * derror

        # Get required wrench as a function of the reference acceleration
        gravity = np.append(m.opt.gravity[[0, 2]], 0)
        wrench_ref = np.diag([self.mass, self.mass, self.moi]) @ (acc_ref - gravity)

        # Solve the wrench distribution problem
        nLegs = len(self.leg_controllers)
        JL = np.zeros((nLegs, 3 * nLegs))
        kL = np.zeros((nLegs,))
        for i, lc in enumerate(self.leg_controllers):
            lc.update_matrices(m, d)
            Jl = np.block([lc.link_jacobians[-1].T, lc.rotation_jacobians[-1].T])
            JL[i, 3 * i:3 * (i + 1)] = Jl[0, :]
            kL[i] = (lc.gravity_efforts - lc.cc_matrix @ lc.joint_velocities)[0]

        gamma = np.tile(np.eye(3), nLegs)
        lsVec = np.concatenate((wrench_ref, kL))
        lsMat = np.concatenate([gamma, JL])
        wrenches, _, _, _ = np.linalg.lstsq(lsMat, lsVec, rcond=None)

        for i, lc in enumerate(self.leg_controllers):
            lc.update(d, wrenches[3 * i:3 * (i + 1)])
