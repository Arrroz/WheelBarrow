"""Standing creep test.

Holds the robot in its nominal standing pose for 30 s and measures whether the
wheel-ground contact drifts. Contact regularization can produce non-physical
creep and slip under sustained load, which would silently corrupt any result
that depends on steady-state force distribution.

Two variants:
  A - wheels free: zero torque on the wheel hinges, so the contact carries no
      tangential load and any drift means a spurious force.
  B - wheels held: the wheel hinges are PD-held, loading friction against the
      legs' sag and exposing frictional creep.

Writes one CSV and one plot per variant to logs/.
"""

import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import model, runner
from src.control.pd_hold import PDHold

LOGS = pathlib.Path(__file__).resolve().parents[1] / "logs"

MODEL = "wheeled_biped"
LEG_GROUP, WHEEL_GROUP = 1, 2

DURATION = 30.0
LOG_HZ = 100
SETTLE_TIME = 2.0  # transient discarded when measuring drift

# The wheels' small spin inertia makes the default damping unstable at this
# timestep; the hold diverges somewhere between 20 and 30.
WHEEL_KD = 10.0

# Watch the run in the viewer. Paced to wall-clock time, so each variant takes
# DURATION seconds of real time.
VIEW = False


def contact_state(m, d, wheel_geoms):
    """Penetration depth and normal force for each wheel's floor contact.

    Penetration is mujoco's signed distance, so negative means the geoms
    overlap. Wheels out of contact report zeros.
    """
    pen = {g: 0.0 for g in wheel_geoms}
    normal = {g: 0.0 for g in wheel_geoms}
    force = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            if g in pen:
                mujoco.mj_contactForce(m, d, i, force)
                pen[g] = c.dist
                normal[g] = force[0]
    return pen, normal


def run(variant):
    m, d = model.load(MODEL)
    legs = model.joints_in_group(m, LEG_GROUP)
    wheels = model.joints_in_group(m, WHEEL_GROUP)
    leg_hold = PDHold(m, d, legs)
    wheel_hold = PDHold(m, d, wheels, kd=WHEEL_KD) if variant == "B" else None

    def hold(m, d):
        leg_hold(m, d)
        if wheel_hold is not None:
            wheel_hold(m, d)

    joints = model.joint_names(m)
    qadr = [m.joint(j).qposadr[0] for j in joints]
    wheel_geoms = [m.geom(g).id for g in model.geoms_in_group(m, WHEEL_GROUP)]

    # a wheel rolls on the body its geom belongs to
    wheel_bodies = [m.body(m.geom_bodyid[g]).name for g in wheel_geoms]
    spin_adr = [
        [m.jnt_qposadr[j] for j in model.chain_hinges(m, b)] for b in wheel_bodies
    ]

    def sample(m, d):
        pen, normal = contact_state(m, d, wheel_geoms)
        return (
            [d.time]
            + [d.qpos[a] for a in qadr]
            + [pen[g] for g in wheel_geoms]
            + [normal[g] for g in wheel_geoms]
            + [d.xpos[m.body(b).id][0] for b in wheel_bodies]
            + [sum(d.qpos[a] for a in adr) for adr in spin_adr]
        )

    rows = runner.run(
        m,
        d,
        control=hold,
        duration=DURATION,
        sample_hz=LOG_HZ,
        on_sample=sample,
        view=VIEW,
    )
    names = [m.geom(g).name for g in wheel_geoms]
    header = (
        ["t"]
        + joints
        + [f"{n}_penetration" for n in names]
        + [f"{n}_normal_force" for n in names]
        + [f"{n}_x" for n in names]
        + [f"{n}_spin" for n in names]
    )
    return m, np.array(rows), header, names


def summarise(t, series, settle_idx):
    """Total excursion, post-settle drift, and the slope of that drift."""
    tail_t, tail = t[settle_idx:], series[settle_idx:]
    return dict(
        total=series[-1] - series[0],
        drift=tail[-1] - tail[0],
        slope=np.polyfit(tail_t, tail, 1)[0],
    )


def report(variant, m, data, header, wheels):
    t = data[:, 0]
    col = {n: i for i, n in enumerate(header)}
    settle_idx = int(SETTLE_TIME * LOG_HZ)

    print(f"\n{'=' * 72}")
    print(f"Variant {variant} - wheels {'free' if variant == 'A' else 'held'}")
    print(f"{'=' * 72}")
    print(
        f"{'quantity':<22}{'total (t=0..30)':>18}{'drift (post-settle)':>21}{'slope':>14}"
    )

    for label, key, scale, unit in [
        ("base x", "base_x", 1e3, "mm"),
        ("base z", "base_z", 1e3, "mm"),
        ("base pitch", "base_pitch", np.degrees(1.0), "deg"),
        ("rear wheel angle", "rear_ankle", np.degrees(1.0), "deg"),
        ("front wheel angle", "front_ankle", np.degrees(1.0), "deg"),
    ]:
        s = summarise(t, data[:, col[key]] * scale, settle_idx)
        print(
            f"{label:<22}{s['total']:>15.4f} {unit:<2}{s['drift']:>18.4f} {unit:<2}"
            f"{s['slope']:>10.2e} {unit}/s"
        )

    print()
    for w in wheels:
        pen = data[:, col[f"{w}_penetration"]] * 1e3
        normal = data[:, col[f"{w}_normal_force"]]
        print(
            f"{w:<22}penetration final {pen[-1]:8.4f} mm   "
            f"drift {pen[-1] - pen[settle_idx]:8.5f} mm   "
            f"normal force {normal[-1]:7.3f} N"
        )

    # A wheel that rolls without slipping advances by r * spin. Whatever is left
    # over slid along the ground.
    print()
    for w in wheels:
        r = m.geom(w).size[0]
        x = data[:, col[f"{w}_x"]]
        spin = data[:, col[f"{w}_spin"]]
        total = r * (spin[-1] - spin[0]) - (x[-1] - x[0])
        drift = r * (spin[-1] - spin[settle_idx]) - (x[-1] - x[settle_idx])
        print(
            f"{w:<22}slip at contact   total {total * 1e3:8.4f} mm   "
            f"post-settle {drift * 1e3:9.5f} mm"
        )

    weight = m.body_mass.sum() * abs(m.opt.gravity[2])
    total_normal = sum(data[-1, col[f"{w}_normal_force"]] for w in wheels)
    print(
        f"\nnormal force sum {total_normal:.3f} N vs weight {weight:.3f} N "
        f"(residual {total_normal - weight:+.4f} N)"
    )


def plot(variant, data, header, wheels):
    t = data[:, 0]
    col = {n: i for i, n in enumerate(header)}
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax[0].plot(t, data[:, col["base_x"]] * 1e3, lw=1.2)
    ax[0].set_ylabel("base x [mm]")
    ax[1].plot(t, data[:, col["base_z"]] * 1e3, lw=1.2, color="tab:orange")
    ax[1].set_ylabel("base z [mm]")
    for w in wheels:
        ax[2].plot(t, data[:, col[f"{w}_penetration"]] * 1e3, lw=1.2, label=w)
    ax[2].set_ylabel("penetration [mm]")
    ax[2].set_xlabel("time [s]")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)

    fig.suptitle(
        f"Creep test, variant {variant} "
        f"({'wheels free' if variant == 'A' else 'wheels held'})"
    )
    fig.tight_layout()
    path = LOGS / f"{MODEL}_creep_{variant}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    LOGS.mkdir(exist_ok=True)
    for variant in ("A", "B"):
        m, data, header, wheels = run(variant)

        csv_path = LOGS / f"{MODEL}_creep_{variant}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

        report(variant, m, data, header, wheels)
        png_path = plot(variant, data, header, wheels)
        print(f"\nwrote {csv_path.name}, {png_path.name}")


if __name__ == "__main__":
    main()
