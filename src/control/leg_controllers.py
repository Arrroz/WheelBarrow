"""Controller for a leg that supports the torso through its wheel."""

import numpy as np

from src import model

R90 = np.array([[0, -1], [1, 0]])


class SupportingLegController:

    def __init__(self, m, prefix):
        names = ("ankle", "knee", "hip")
        self.joints = [m.joint(f"{prefix}_{n}").id for n in names]
        self.actuators = [model.actuator_for(m, f"{prefix}_{n}") for n in names]
        self.wheel_hinges = model.chain_hinges(m, f"{prefix}_wheel")
        self.links = [m.body(f"{prefix}_{n}").id for n in ("wheel", "shin", "thigh")]
        self.links.append(m.body("torso").id)

        self.wheel_geom = m.geom(f"{prefix}_wheel").id
        self.foot_radius = m.geom_size[self.wheel_geom][0]

        self.n_joints = len(self.joints)
        self.n_links = len(self.links)

    def update_matrices(self, m, d):
        n = self.n_links

        # Joints' velocities
        self.joint_velocities = np.zeros(n)
        self.joint_velocities[0] = sum(d.qvel[m.jnt_dofadr[j]] for j in self.wheel_hinges)
        self.joint_velocities[1:] = -d.qvel[m.jnt_dofadr[self.joints]]

        # Frames
        contact = d.geom_xpos[self.wheel_geom][[0, 2]] - [0, self.foot_radius]
        chain_frames = [contact] + [d.xanchor[j][[0, 2]] for j in self.joints]

        # Frames' velocities
        chain_frames_velocities = [np.array([-self.foot_radius * self.joint_velocities[0], 0])]
        for i in range(1, n):
            jacobian = np.zeros((2, n))
            for j in range(i):
                jacobian[:, j] = R90 @ (chain_frames[i] - chain_frames[j])
            chain_frames_velocities.append(jacobian @ self.joint_velocities)

        # Jacobians
        self.link_jacobians = [np.zeros((2, n)) for _ in range(n)]
        self.rotation_jacobians = [np.zeros((1, n)) for _ in range(n)]
        for i, link in enumerate(self.links):
            p = d.xipos[link][[0, 2]]
            for j in range(i + 1):
                self.link_jacobians[i][:, j] = R90 @ (p - chain_frames[j])
                self.rotation_jacobians[i][0, j] = 1.0

        # Jacobians' derivatives
        self.link_jacobian_derivatives = [np.zeros((2, n)) for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                self.link_jacobian_derivatives[i][:, j] = R90 @ (
                    self.link_jacobians[i] @ self.joint_velocities - chain_frames_velocities[j]
                )

        # Centrifugal and Coriolis effects, and gravity efforts, over the leg links only
        gravity = m.opt.gravity[[0, 2]]
        self.cc_matrix = np.zeros((n, n))
        self.gravity_efforts = np.zeros(n)
        for i in range(n - 1):
            mass = m.body_mass[self.links[i]]
            self.cc_matrix += self.link_jacobians[i].T @ self.link_jacobian_derivatives[i] * mass
            self.gravity_efforts += self.link_jacobians[i].T @ gravity * mass

    def update(self, d, body_wrench_ref):
        body_efforts = -np.vstack((self.link_jacobians[-1], self.rotation_jacobians[-1])).T @ body_wrench_ref
        joint_efforts = self.gravity_efforts - self.cc_matrix @ self.joint_velocities + body_efforts
        for i in range(self.n_joints):
            d.ctrl[self.actuators[i]] = joint_efforts[i + 1]
