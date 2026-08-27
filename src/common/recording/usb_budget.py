"""How many camera streams fit on each USB bus, and which selection does not.

Every camera on this rig is a USB 2.0 device, so each one lands on a 480 Mbit/s
bus whatever socket it is plugged into, and each bus has ONE isochronous
bandwidth budget shared by everything on it. That budget, not the cameras and not
the code, is what limits how many streams can record at once.

MEASURED at 640x480 MJPG, across every trial run on this rig so far:

=========================================  ====================================
trial                                      result
=========================================  ====================================
each camera alone                          all seven fine, 15.9-26.4 fps
all seven together, two controllers        exactly two refused, five at full rate
all seven, tactile at 320x240              still two refused
one controller, its four cameras           three run, one refused
the other controller, its three cameras    two run, one refused
the four tactile cameras alone             all four fine, 28.4-28.7 fps
central + two tactile, one controller      one refused
five streams over THREE controllers        all five deliver
=========================================  ====================================

Two things in that table matter more than the counts. First, WHICH stream is
refused is random -- it is whichever loses the race to reserve bandwidth, so the
same configuration fails differently on consecutive runs and looks like a flaky
camera rather than a budget. A refused stream opens normally and then delivers
nothing at all. Second, the per-controller capacity is NOT a stream count,
because the cameras do not cost the same.

ONE cost model reproduces every row above: a TACTILE camera costs TWO units, a
plain RGB camera costs ONE, and a controller carries FOUR.

===================================  =====  =========  =====================
selection on one controller          units  predicted  observed
===================================  =====  =========  =====================
central + wrist + 1 tactile          4      fits       three ran
wrist + 2 tactile                    5      over       two ran
2 tactile                            4      fits       both ran
central + 2 tactile                  5      over       one refused
central + wrist + 2 tactile          6      over       three ran
===================================  =====  =========  =====================

Asking for less does not help, and the reason is worth recording so nobody
tries it twice. ``VIDIOC_ENUM_FRAMEINTERVALS`` reports exactly ONE frame
interval per format and size on every camera here -- the tactile cameras offer
60 fps at 640x480 MJPG and 30 fps at 320x240, nothing else; the wrist cameras
offer 30 fps and nothing else. There is no lower rate to select, which is also
why OpenCV's ``CAP_PROP_FPS`` appears to be ignored: the request has nowhere to
go. And halving the frame SIZE, which does move the camera to its 30 fps mode,
still bought no extra stream.

The obvious software lever is the uvcvideo FIX_BANDWIDTH quirk, which makes the
driver compute the real need instead of trusting the camera's declaration.
MEASURED on this rig, with the module reloaded and every device re-enumerated
under ``quirks=128``: it changes nothing at all -- the same two streams are
refused and the rest keep the same rates. The likely reason is that uvcvideo
skips that fixup for compressed formats, and everything here is captured as
MJPEG. So it is not a remedy, and :func:`quirks_active` exists to report the
state rather than to promise one.

What remains is physical: a stream needs a controller with room. That is what
this rig now does -- the overhead camera on a controller of its own and a pair
of tactile cameras on each of two others, which is four units, four units and
one, and all five deliver together.

The grouping here is deliberately pure and string-only: it reads the stable
by-path aliases already stored in ``sensor_map.yaml`` and never opens a device,
so the console and the preflight can both warn about a selection BEFORE anything
is plugged, opened or recorded.
"""

from __future__ import annotations

from pathlib import Path

# What one 480 Mbit/s bus carries, in the units of the cost model above. Upper
# bounds, not promises: a selection at exactly the capacity is the largest that
# was measured to work, not one the bus guarantees. Do not raise them to make a
# warning go away -- the authority on whether a stream actually got its
# bandwidth is the runtime check in ``cameras.CameraCapture``, which fails an
# open that delivers no frame. These numbers only buy the operator a warning
# first, before a session rather than during one.
BUS_CAPACITY_UNITS = 4
TACTILE_STREAM_UNITS = 2
RGB_STREAM_UNITS = 1

# How a stream name says which it is. The tactile cameras are named for the
# gripper finger they sit on (left_arm_right_gripper and the other three), and
# that suffix is the whole rule -- kept here as string work so this module stays
# free of the camera stack. ``test_usb_budget`` pins it against the real list in
# ``tool.test_sensor_rates`` so the two cannot drift apart.
_TACTILE_SUFFIX = "_gripper"

# Where the kernel reports the uvcvideo quirk mask.
QUIRKS_PATH = "/sys/module/uvcvideo/parameters/quirks"

# UVC_QUIRK_FIX_BANDWIDTH: compute the real bandwidth need rather than trusting
# the camera's declared one.
QUIRK_FIX_BANDWIDTH = 128

# The kernel prints this for a module parameter that was never set.
_QUIRKS_UNSET = 0xFFFFFFFF


def controller_of(node: "str | int | None") -> "str | None":
    """The USB controller a by-path device alias hangs off, or None. Pure.

    ``/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.1.2:1.0-video-index0`` belongs to
    controller ``pci-0000:05:00.4``. Anything that is not a by-path alias -- a
    bare ``/dev/videoN``, an integer index, an unassigned stream -- has no
    knowable controller and returns None, because the kernel's numbering says
    nothing about which bus a device is on.
    """
    if node is None or isinstance(node, int):
        return None
    name = Path(str(node)).name
    if not name.startswith("pci-"):
        return None
    # Both spellings appear in /dev/v4l/by-path for the same device.
    for sep in ("-usb-", "-usbv2-"):
        head, found, _tail = name.partition(sep)
        if found:
            return head
    return None


def group_by_controller(
    camera_nodes: "dict[str, str | int | None]",
) -> "dict[str, list[str]]":
    """Map controller -> the stream names sharing it. Pure -- unit-tested.

    Streams whose controller cannot be determined are left out entirely rather
    than pooled under a placeholder: counting them together would invent a bus
    that does not exist and warn about a crowd that is not there.
    """
    groups: dict[str, list[str]] = {}
    for name, node in sorted(camera_nodes.items()):
        controller = controller_of(node)
        if controller is not None:
            groups.setdefault(controller, []).append(name)
    return groups


def is_tactile_stream(name: str) -> bool:
    """Whether this stream is a fingertip (tactile) camera. Pure."""
    return str(name).endswith(_TACTILE_SUFFIX)


def stream_units(name: str) -> int:
    """What one stream costs its controller, in the model above. Pure."""
    return TACTILE_STREAM_UNITS if is_tactile_stream(name) else RGB_STREAM_UNITS


def bus_units(names: "list[str]") -> int:
    """What a set of streams costs the controller they share. Pure."""
    return sum(stream_units(n) for n in names)


def _describe(names: "list[str]") -> str:
    """The selection priced out, so the operator can see where it went over."""
    return ", ".join(f"{n} ({stream_units(n)})" for n in names)


def budget_warnings(
    groups: "dict[str, list[str]]",
    capacity: int = BUS_CAPACITY_UNITS,
) -> "list[str]":
    """One operator-readable line per over-subscribed bus. Pure -- unit-tested.

    Named streams and their cost, not just a count: when the bus refuses one of
    them at random, the operator needs to know which set was competing and
    which of them are the expensive ones to move.
    """
    warnings = []
    for controller, names in sorted(groups.items()):
        units = bus_units(names)
        if units <= capacity:
            continue
        warnings.append(
            f"USB controller {controller} is over-subscribed: {_describe(names)} "
            f"— {units} units against about {capacity} it carries (a fingertip "
            f"camera costs {TACTILE_STREAM_UNITS}, a colour camera "
            f"{RGB_STREAM_UNITS}, so two fingertip views fill a controller). "
            "Expect some of them to open and then deliver nothing, and which "
            "ones is random"
        )
    return warnings


def selection_warnings(
    selected: "set[str]",
    camera_nodes: "dict[str, str | int | None]",
    capacity: int = BUS_CAPACITY_UNITS,
) -> "list[str]":
    """The warnings for just the streams a session is about to record. Pure."""
    chosen = {n: d for n, d in camera_nodes.items() if n in selected}
    return budget_warnings(group_by_controller(chosen), capacity)


def quirks_active(path: "str | Path" = QUIRKS_PATH) -> "bool | None":
    """Whether uvcvideo's FIX_BANDWIDTH quirk is on; None if it cannot be read.

    Reported, not relied upon: on this rig the quirk made no difference to how
    many streams fit (see the module docstring). It is worth surfacing anyway,
    because it is the first thing anyone meeting this ceiling reaches for, and
    knowing it is already on saves trying it again.

    None and False mean different things to an operator -- "no uvcvideo module
    here" is not "the quirk is off" -- so they stay distinct.
    """
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None
    try:
        mask = int(raw)
    except ValueError:
        return None
    if mask == _QUIRKS_UNSET or mask < 0:
        return False
    return bool(mask & QUIRK_FIX_BANDWIDTH)
