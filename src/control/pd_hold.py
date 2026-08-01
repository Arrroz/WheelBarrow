"""Holding joints at fixed angles."""

from ..model import actuator_for

KP, KD = 2000.0, 100.0


class PDHold:
    """Holds joints at fixed angles, defaulting to their current values."""

    def __init__(self, m, d, joints, kp=KP, kd=KD, targets=None):
        self.kp, self.kd = kp, kd
        self.joints = list(joints)
        self.qadr = {n: m.joint(n).qposadr[0] for n in self.joints}
        self.dadr = {n: m.joint(n).dofadr[0] for n in self.joints}
        self.ctrl = {n: actuator_for(m, n) for n in self.joints}
        source = d.qpos if targets is None else targets
        self.targets = {n: source[self.qadr[n]] for n in self.joints}

    def __call__(self, m, d):
        # m is unused but keeps the control(m, d) signature the runner expects
        for n in self.joints:
            error = self.targets[n] - d.qpos[self.qadr[n]]
            d.ctrl[self.ctrl[n]] = self.kp * error - self.kd * d.qvel[self.dadr[n]]
