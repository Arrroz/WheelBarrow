"""Controller for a leg that supports the body through its foot."""

import mujoco
import numpy as np

from src import model

GROUND_NORMAL = np.array([0.0, 0.0, 1.0])


class SupportingLegController:

    def __init__(self, m, joints, foot_geom):
        self.joints = [m.joint(j).id for j in joints]
        self.actuators = [model.actuator_for(m, j) for j in joints]
        self.foot_geom = m.geom(foot_geom).id
        self.foot_radius = m.geom_size[self.foot_geom][0]
        self.links = [m.geom_bodyid[self.foot_geom]]
        self.links += [m.body_parentid[m.jnt_bodyid[j]] for j in self.joints]

        self.n_joints = len(self.joints)
        self.n_links = len(self.links)

    def update_matrices(self, m, d):
        # Columns: foot rotations about world x, y, z, then one per joint
        n = 3 + self.n_joints
        foot_offset = self.foot_radius * GROUND_NORMAL

        # Joints' velocities
        foot_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.links[0], foot_velocity, 0)
        self.joint_velocities = np.concatenate((foot_velocity[:3], -d.qvel[m.jnt_dofadr[self.joints]]))

        # Frames
        axes = np.vstack((np.eye(3), d.xaxis[self.joints]))
        contact = d.geom_xpos[self.foot_geom] - foot_offset
        chain_frames = np.vstack((np.tile(contact, (3, 1)), d.xanchor[self.joints]))

        def jacobian(p, cols):
            J = np.zeros((3, n))
            J[:, :cols] = np.cross(axes[:cols], p - chain_frames[:cols]).T
            return J

        # Jacobians
        self.chain_frames_jacobians = [jacobian(chain_frames[3 + i], 3 + i) for i in range(self.n_joints)]
        self.link_jacobians = [jacobian(d.xipos[link], 3 + i) for i, link in enumerate(self.links)]
        self.rotation_jacobians = [np.zeros((3, n)) for _ in range(self.n_links)]
        for i in range(self.n_links):
            self.rotation_jacobians[i][:, :3 + i] = axes[:3 + i].T

        # Frames' velocities
        chain_frames_velocities = np.zeros((n, 3))
        chain_frames_velocities[:3] = np.cross(self.joint_velocities[:3], foot_offset)
        for i in range(self.n_joints):
            chain_frames_velocities[3 + i] = self.chain_frames_jacobians[i] @ self.joint_velocities

        # Axes' rates
        axis_rates = np.zeros((n, 3))
        for i in range(self.n_joints):
            axis_rates[3 + i] = np.cross(self.rotation_jacobians[i] @ self.joint_velocities, axes[3 + i])

        # Jacobians' derivatives
        self.link_jacobian_derivatives = [np.zeros((3, n)) for _ in range(self.n_links)]
        self.rotation_jacobian_derivatives = [np.zeros((3, n)) for _ in range(self.n_links)]
        for i, link in enumerate(self.links):
            cols = 3 + i
            p = d.xipos[link]
            v = self.link_jacobians[i] @ self.joint_velocities
            self.link_jacobian_derivatives[i][:, :cols] = (
                np.cross(axis_rates[:cols], p - chain_frames[:cols])
                + np.cross(axes[:cols], v - chain_frames_velocities[:cols])
            ).T
            self.rotation_jacobian_derivatives[i][:, :cols] = axis_rates[:cols].T

        # Centrifugal and Coriolis Effects
        self.cc_matrix = np.zeros((n, n))
        for i in range(self.n_links - 1):
            R = d.ximat[self.links[i]].reshape(3, 3)
            moi = R @ np.diag(m.body_inertia[self.links[i]]) @ R.T
            w_skew = np.cross(np.eye(3), self.rotation_jacobians[i] @ self.joint_velocities)
            self.cc_matrix += (
                self.link_jacobians[i].T @ self.link_jacobian_derivatives[i] * m.body_mass[self.links[i]]
                + self.rotation_jacobians[i].T
                @ (moi @ self.rotation_jacobian_derivatives[i] + w_skew @ moi @ self.rotation_jacobians[i])
            )

        # Gravity Efforts
        self.gravity_efforts = np.zeros(n)
        for i in range(self.n_links - 1):
            self.gravity_efforts += self.link_jacobians[i].T @ m.opt.gravity * m.body_mass[self.links[i]]

    def update(self, d, body_wrench_ref):
        body_efforts = -np.vstack((self.link_jacobians[-1], self.rotation_jacobians[-1])).T @ body_wrench_ref
        joint_efforts = self.gravity_efforts - self.cc_matrix @ self.joint_velocities + body_efforts
        for i in range(self.n_joints):
            d.ctrl[self.actuators[i]] = joint_efforts[3 + i]
