"""Loading models and looking up their joints and actuators."""

import pathlib

import mujoco

MODELS = pathlib.Path(__file__).resolve().parents[1] / "models"


def load(name, keyframe=0):
    """Load a model from models/ and put it in a keyframe pose."""
    m = mujoco.MjModel.from_xml_path(str(MODELS / f"{name}.xml"))
    d = mujoco.MjData(m)
    if keyframe is not None:
        mujoco.mj_resetDataKeyframe(m, d, keyframe)
    mujoco.mj_forward(m, d)
    return m, d


def joint_names(m):
    return [m.joint(i).name for i in range(m.njnt)]


def joints_in_group(m, group):
    """Names of the joints tagged with a model group."""
    return [m.joint(i).name for i in range(m.njnt) if m.jnt_group[i] == group]


def geoms_in_group(m, group):
    """Names of the geoms tagged with a model group."""
    return [m.geom(i).name for i in range(m.ngeom) if m.geom_group[i] == group]


def chain_hinges(m, body_name):
    """Indices of the hinge joints between a body and the world.

    Their angles sum to the body's absolute rotation whenever every hinge in
    the chain shares an axis, which avoids the wrapping of a rotation read out
    of the body frame.
    """
    joints = []
    body = m.body(body_name).id
    while body > 0:
        start = m.body_jntadr[body]
        for j in range(start, start + m.body_jntnum[body]):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                joints.append(j)
        body = m.body_parentid[body]
    return joints


def actuator_for(m, joint_name):
    """Index of the actuator driving a joint, found by transmission target."""
    jid = m.joint(joint_name).id
    for i in range(m.nu):
        if m.actuator_trnid[i, 0] == jid:
            return i
    raise ValueError(f"no actuator drives joint {joint_name!r}")
