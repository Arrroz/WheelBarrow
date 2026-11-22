"""Hold the wheeled biped's torso at its standing pose, in the viewer."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import model, runner
from src.control.body_controllers import BodyController
from src.control.leg_controllers import SupportingLegController

m, d = model.load("wheeled_biped")
legs = [
    SupportingLegController(m, [f"{p}_ankle", f"{p}_knee", f"{p}_hip"], f"{p}_wheel")
    for p in ("rear", "front")
]
control = BodyController(m, d, legs)
runner.run(m, d, control, view=True)
