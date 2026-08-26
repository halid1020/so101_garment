"""How many camera streams fit on each USB bus, and which selection does not.

Every camera on this rig is a USB 2.0 device, so each one lands on a 480 Mbit/s
bus whatever socket it is plugged into, and each bus has ONE isochronous
bandwidth budget shared by everything on it. That budget, not the cameras and not
the code, is what limits how many streams can record at once.

MEASURED with the rig's seven streams at 640x480 MJPG, two controllers
(``pci-0000:05:00.4`` carrying four cameras, ``pci-0000:06:00.4`` carrying three):

===================================  ==========================================
trial                                result
===================================  ==========================================
each camera alone                    all seven fine, 15.9-26.4 fps
all seven together                   exactly two refused, five at 21-27 fps
all seven together at 15 fps         still two refused
all seven together at 320x240        still two refused
one bus alone, its four cameras      three run, one refused
one bus alone, its three cameras     two run, one refused
the four tactile cameras alone       all four fine, 28.4-28.7 fps
===================================  ==========================================

Two things in that table matter more than the counts. First, WHICH streams are
refused is random -- it is whichever loses the race to reserve bandwidth, so the
same configuration fails differently on consecutive runs and looks like a flaky
camera rather than a budget. Second, asking for less does not help: these cameras
declare a fixed bandwidth need regardless of the format negotiated, so neither a
lower frame rate nor a smaller frame buys a single extra stream.

The obvious software lever is the uvcvideo FIX_BANDWIDTH quirk, which makes the
driver compute the real need instead of trusting the camera's declaration.
MEASURED on this rig, with the module reloaded and every device re-enumerated
under ``quirks=128``: it changes nothing at all -- the same two streams are
refused and the rest keep the same rates. So it is not a remedy here, and
:func:`quirks_active` exists to report the state rather than to promise one. The
ceiling above is what holds with the quirk on or off.

The grouping here is deliberately pure and string-only: it reads the stable
by-path aliases already stored in ``sensor_map.yaml`` and never opens a device,
so the console and the preflight can both warn about a selection BEFORE anything
is plugged, opened or recorded.
"""

from __future__ import annotations

from pathlib import Path

# Streams per 480 Mbit/s bus, measured (see the table above) with the uvcvideo
# FIX_BANDWIDTH quirk both off and on -- it made no difference. An upper bound,
# not a promise: one of the two buses fitted three streams and the other managed
# only two, so a selection at exactly this limit can still lose one. Do not raise
# it to make a warning go away -- the authority on whether a stream actually got
# its bandwidth is the runtime check in ``cameras.CameraCapture``, which fails an
# open that delivers no frame. This number only buys the operator a warning first.
DEFAULT_PER_BUS_LIMIT = 3

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


def budget_warnings(
    groups: "dict[str, list[str]]",
    per_bus_limit: int = DEFAULT_PER_BUS_LIMIT,
) -> "list[str]":
    """One operator-readable line per over-subscribed bus. Pure -- unit-tested.

    Named streams, not just a count: when the bus refuses one of them at random,
    the operator needs to know which set was competing.
    """
    warnings = []
    for controller, names in sorted(groups.items()):
        if len(names) <= per_bus_limit:
            continue
        warnings.append(
            f"{len(names)} camera streams share USB controller {controller} "
            f"({', '.join(names)}) but only about {per_bus_limit} fit its "
            "bandwidth — expect some of them to open and then deliver nothing, "
            "and which ones is random"
        )
    return warnings


def selection_warnings(
    selected: "set[str]",
    camera_nodes: "dict[str, str | int | None]",
    per_bus_limit: int = DEFAULT_PER_BUS_LIMIT,
) -> "list[str]":
    """The warnings for just the streams a session is about to record. Pure."""
    chosen = {n: d for n, d in camera_nodes.items() if n in selected}
    return budget_warnings(group_by_controller(chosen), per_bus_limit)


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
