"""Hold a robot's body at its standing pose, in the viewer.

Each geom in the feet group gets a leg, made of the hinges between the foot
and the body.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import model, runner
from src.control.body_controllers import BodyController
from src.control.leg_controllers import SupportingLegController

MODEL = "quadruped"
FEET_GROUP = 2

m, d = model.load(MODEL)
body = m.body("body").id
legs = []
for foot in model.geoms_in_group(m, FEET_GROUP):
    hinges = model.chain_hinges(m, m.body(m.geom(foot).bodyid[0]).name)
    joints = [m.joint(j).name for j in hinges if m.jnt_bodyid[j] != body]
    legs.append(SupportingLegController(m, joints, foot))
control = BodyController(m, d, legs)
runner.run(m, d, control, view=True)
