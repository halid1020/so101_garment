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

On startup Telegrip prints the URL to open (the machine's LAN IP is filled
in automatically):

```
🤖 telegrip starting...
📱 Open the UI in your browser on:
   https://192.168.0.41:8443
📱 Then go to the same address on your VR headset browser
```

Open that address in a desktop browser first (accept the self-signed
cert) — the web UI shows arm status; click **Connect Robot** (or launch
with `--autoconnect`). Then follow *Connecting the VR headset* below to
drive the arms from a Meta Quest.

## Connecting the VR headset

This is the part that trips people up. Telegrip runs its own **WebXR** app
straight from the headset's built-in browser — there is nothing to install
on the Quest. But two things must line up: the Quest has to reach the
laptop over the LAN, and it has to accept the laptop's self-signed
certificate on **both** ports Telegrip uses.

### What Telegrip serves (source-verified)

Telegrip starts two servers, both bound to `0.0.0.0` (all interfaces) with
the **same self-signed certificate** (`../telegrip/cert.pem`, CN=`localhost`,
generated on first run):

| Server | Port | Protocol | Purpose |
|---|---|---|---|
| Web UI / WebXR page | `8443` | `https://` | the page the browser loads; `Start`/`Connect` buttons, keyboard control |
| VR control channel | `8442` | `wss://` (secure WebSocket) | controller pose/grip/trigger stream from the headset to the arms |

The WebXR page is loaded from `:8443`, but its JavaScript then opens a
secure WebSocket to `wss://<same-host>:8442`. WebXR only runs in a **secure
context**, so both ports must be HTTPS/WSS — which is why Telegrip ships a
self-signed cert and why the certificate has to be trusted on *each* port
separately (see step 4; this is the single most common failure).

Override the ports with `--https-port` / `--ws-port` if 8442/8443 are
taken; the page always derives the WebSocket port from its own bundle, so
if you move `--ws-port` off 8442 you must also rebuild/patch the upstream
`web-ui/vr_app.js` (it hard-codes `8442`). In practice, leave them.

### Prerequisites

1. **Same network.** Put the Quest on the *same* Wi-Fi/LAN as the laptop.
   Guest or "isolated" Wi-Fi networks block device-to-device traffic and
   will never work — the Quest must be able to reach the laptop's LAN IP.
2. **Internet on the Quest too.** The page pulls the A-Frame WebXR library
   from a CDN (`https://aframe.io/...`, hard-coded in
   `../telegrip/web-ui/index.html`). If the Quest has LAN access but no
   route to the internet, the 3D scene never loads and the *Start* button
   never appears.
3. **Find the laptop's LAN IP** (Telegrip already prints it, but to check
   independently):

   ```bash
   hostname -I | awk '{print $1}'      # first address is usually the LAN one
   ip addr show                        # full list, per interface
   ```

   Pick the `192.168.x.x` / `10.x.x.x` address on the interface that
   carries your Wi-Fi/LAN, not `127.0.0.1` and not the `172.17.x.x` Docker
   bridge.
4. **Ports actually listening / not firewalled.** Confirm both servers are
   up and bound to all interfaces (not just localhost):

   ```bash
   ss -tlnp | grep -E '8442|8443'
   # expect two LISTEN lines on 0.0.0.0:8442 and 0.0.0.0:8443
   ```

   If the Quest can't reach them but the lines are present, a host firewall
   is the likely blocker. With `ufw` active you would need
   `sudo ufw allow 8442,8443/tcp` — but this rig assumes **no sudo**; if you
   can't change the firewall, connect both devices to a network where the
   host firewall already permits LAN traffic.

### Happy path (numbered)

1. **Launch the stack** on the laptop and note the printed
   `https://<ip>:8443` URL:

   ```bash
   source setup.sh
   venv/bin/python tool/telegrip_native.py --autoconnect
   ```

2. **Accept the cert for the UI port.** In the Quest's browser open
   `https://<ip>:8443`. Because the certificate is self-signed the browser
   shows a security warning — tap **Advanced → Proceed to <ip> (unsafe)**
   (wording varies by browser). The Telegrip UI then loads.
3. **Accept the cert for the WebSocket port too — the classic gotcha.**
   The UI page will try to open `wss://<ip>:8442` in the background, and the
   browser silently refuses that connection until the *same* cert is
   trusted on 8442 as well. In the headset browser open a second tab at
   `https://<ip>:8442` and do the same **Advanced → Proceed** dance. You
   will land on a bare page (or an "Upgrade Required" message — that is
   normal; 8442 is a WebSocket endpoint, not a web page). Telegrip even
   prints a warning in its terminal when a browser hits 8442, confirming
   you reached it. Now return to the `:8443` tab and reload it.
4. **Start the VR session.** With the cert trusted on both ports and the
   controllers powered on, a green **Start Controller Tracking** button
   appears in the middle of the page (it only shows if the browser reports
   WebXR immersive support). Tap it. It first connects the arms if they are
   not already engaged, then requests the immersive session — **accept the
   headset's permission prompt** to enter VR/passthrough. You are now in the
   Telegrip WebXR app; you should see your two controllers with little RGB
   axis gizmos, and the UI's **VR Connected** indicator on the desktop
   turns on.
5. **Drive the arms.** Per controller, independently (left controller →
   left arm, right controller → right arm):
   - **Hold the grip button** to engage that arm. While held, the arm's
     gripper tip tracks your controller's *relative* motion from where you
     gripped; roll/pitch of the controller map onto the wrist. Release grip
     to freeze the arm.
   - **Trigger** actuates the gripper (hold to change state). Note: upstream
     Telegrip's own UI text and its server code disagree on the
     open-vs-close direction, so confirm the polarity live on the rig and
     don't trust either doc blindly.
   - There is **no envelope/limit safety** on this native path beyond
     Telegrip's own joint clamping — this is the whole system as upstream
     ships it (user-study condition C4). Keep clear of the arms.
6. **Exit** by taking off / exiting the immersive session (the *Start*
   button reappears), or `Ctrl+C` in the terminal for a graceful shutdown.

Keyboard control also works from the desktop UI without any headset
(WASD-style keys, listed in the UI and upstream README) — handy for a quick
check that the arms move before bringing the Quest in.

### Troubleshooting the connection

- **Page loads but "Start Controller Tracking" is greyed out / never
  appears.** The browser isn't offering WebXR. Causes, in order: (a) you
  opened over `http://` or a context the browser doesn't treat as secure —
  it must be `https://` with the cert accepted; (b) the A-Frame CDN didn't
  load because the Quest has no internet; (c) the controllers are off or not
  paired. Open the browser console if you can — "WebXR not supported" vs a
  failed script load tells them apart.
- **In VR but nothing streams / "VR Connected" stays off.** This is almost
  always step 3: the `wss://<ip>:8442` connection is blocked by the untrusted
  cert on 8442. Re-open `https://<ip>:8442` in the headset, accept the cert,
  reload `:8443`. (Confirm from the laptop with the Telegrip terminal — it
  logs `VR client connected` when the socket actually opens.)
- **Page unreachable in the headset.** Wrong IP (use `hostname -I`, first
  address), or the two devices are on different subnets / a guest Wi-Fi that
  isolates clients, or a host firewall. Verify the servers are listening on
  `0.0.0.0` (not `127.0.0.1`) with the `ss` command above; if they show
  `127.0.0.1` you passed `--host 127.0.0.1` — drop it (default is
  `0.0.0.0`).
- **Controllers connect and "VR Connected" is on, but the arms don't
  move.** That is this repo's side, not Telegrip's browser stack: the motors
  aren't engaged. Launch with `--autoconnect`, or click **Connect Robot** in
  the desktop UI, and check `tool/telegrip_native.py`'s own startup printout
  (it echoes the LEFT/RIGHT ports and the calibration copy) for a port or
  calibration error. If the arms move *swapped*, see the arm-side note under
  *Gotchas*.

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
