# Driving the two SO-101 arms with the original Telegrip stack

This is *not* the in-repo `telegrip` benchmark method
(`src/sim_benchmark/methods/telegrip_split.py`, our reimplementation of their
split IK). This runs the **unmodified upstream Telegrip repository**
([DipFlip/telegrip](https://github.com/DipFlip/telegrip)) — its web UI,
WebXR VR streaming, keyboard control, PyBullet IK, and LeRobot motor I/O —
against our two arms. The only code this repo contributes is the entry
script `tool/telegrip_native.py`, which locates the checkout, bridges our
port/calibration setup into what Telegrip expects, and calls their CLI.

## One-time setup

```bash
# 1. Clone Telegrip next to this repo (same level as ../lerobot)
git clone https://github.com/DipFlip/telegrip ../telegrip

# 2. Install its one missing dependency into our venv
venv/bin/pip install pybullet
# (torch, scipy, numpy, websockets, pynput, pyyaml are already present;
#  LeRobot is our editable ../lerobot checkout, which has the
#  lerobot.robots.so_follower API Telegrip imports — verified compatible.)
```

No `pip install -e ../telegrip` is needed — the entry script puts the
checkout on `sys.path`. Nothing in their repo is modified; Telegrip will
generate `cert.pem`/`key.pem` (self-signed SSL for WebXR) inside
`../telegrip/` on first run.

## What the entry script bridges

| Telegrip expects | This rig provides | Bridge |
|---|---|---|
| `left`/`right` arm serial ports (its `config.yaml` defaults are `/dev/ttySO100blue|red`) | `src/conf/robot.yaml`: `PORT_ID_0` = follower_0 = **RIGHT**, `PORT_ID_1` = follower_1 = **LEFT** | passed as `--left-port`/`--right-port` CLI overrides |
| LeRobot calibrations for robot ids `left_follower`/`right_follower` under `$HF_LEROBOT_HOME/calibration/robots/so_follower/` | `src/calibration_files/follower_{0,1}.json` (same draccus `dict[str, MotorCalibration]` format) | copied on launch (existing differing files are kept unless `--refresh-calibration`) |
| SO-100 URDF + PyBullet link names (`Fixed_Jaw_tip`, joints named `1`–`6`) | SO-101 arms | **left as their bundled SO-100 model** — kinematics are close enough for teleop; do *not* point `--urdf` at our SO-101 URDF (their joint/link name map won't match) |

## Running

```bash
source setup.sh   # serial permissions + env

# Full stack: web UI (https://<this-machine>:8443), VR WebSocket,
# keyboard control, PyBullet viz
venv/bin/python tool/telegrip_native.py

# Useful variants (flags after ours are forwarded verbatim to Telegrip)
venv/bin/python tool/telegrip_native.py --autoconnect          # engage motors on startup
venv/bin/python tool/telegrip_native.py --no-viz               # headless PyBullet
venv/bin/python tool/telegrip_native.py --no-robot             # visualization only, no motors
venv/bin/python tool/telegrip_native.py --log-level info       # verbose
venv/bin/python tool/telegrip_native.py --left-port /dev/ttyACM3   # port override
```

Then:

1. Open `https://<printed-ip>:8443` in a browser (accept the self-signed
   cert) — the web UI shows arm status; click **Connect Robot** (or use
   `--autoconnect`).
2. On the Quest, open the *same* address in the headset browser and enter
   the WebXR app. Grip = engage, controller motion drives the EE; trigger
   drives the gripper. Keyboard control works from the web UI too
   (WASD/arrow-style keys, see their README).

## Gotchas

- **Arm sides**: Telegrip's "left"/"right" are mapped so its LEFT arm is
  our follower_1 and its RIGHT is follower_0 — same convention as the rest
  of this repo. If the physical arms respond swapped, verify with
  `tool/check_mirror.py`, don't guess.
- **Calibration direction is one-way**: the script copies *our* files →
  LeRobot's cache. If you recalibrate inside Telegrip/LeRobot, the cache
  copy changes but `src/calibration_files/` does not; the script warns on
  mismatch and keeps the cache copy unless `--refresh-calibration`.
- Telegrip saves UI settings back into `../telegrip/config.yaml` — that's
  their repo behaving as designed; our port overrides are re-applied on
  every launch so a stale yaml can't flip the arms.
- The firewall must allow TCP 8443 (HTTPS/UI) and 8442 (VR WebSocket) on
  the LAN for the headset to connect.


## Integrated `telegrip` method vs this native entry point

The repo offers Telegrip two ways, and they are different things:

| | Integrated method (`--method telegrip`) | Native stack (`tool/telegrip_native.py`) |
|---|---|---|
| What runs | **Our reimplementation** of Telegrip's split-IK algorithm (`src/sim_benchmark/methods/telegrip_split.py`), inside our teleop pipeline | The **unmodified upstream Telegrip program** — its own UI, input path, IK, and motor I/O; our script only bridges ports/calibrations then hands over |
| Input path | Our Quest reader → One-Euro filter → clutch → armplane target construction → envelope policy | Telegrip's WebXR browser stream (or its keyboard control); none of our filtering, clutching, or envelope handling |
| IK | Analytic wrist (recovered from the absolute target rotation) + 3-joint position DLS on our simulator/robot kinematics, plus our joint-space rate limiter | PyBullet position IK on their bundled SO-100 model, wrist driven by controller rotation *deltas*, their own clamping |
| Envelope / safety | Full workspace-envelope policies, rate limiter, our calibration checks | Whatever upstream Telegrip does (no envelope) |
| What it is for | Comparing Telegrip's *algorithm* against the other IK methods under identical input processing (benchmark + user-study conditions C1–C3) | Comparing the *whole system* end-to-end as users would actually install it (user-study condition C4) |

In short: the integrated method isolates Telegrip's IK idea inside our
pipeline so the comparison is apples-to-apples; the native entry point
runs their complete product untouched so the comparison is
system-versus-system. The paper's method section describes the
integrated port and its deliberate departures from upstream; the user
study uses both.
