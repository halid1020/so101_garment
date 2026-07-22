"""Set firmware wrist-roll limits to protect the tactile-camera cables.

The visual-tactile cameras are mounted on the grippers; a free
wrist_roll rotation twists their USB cables. This tool records a
cable-safe band interactively — torque is disabled, you rotate the
wrist by hand through the FULL safe arc while a live min/pos/max table
streams, then press Enter — and writes the band (minus a safety
margin) into the servo's EEPROM ``Min/Max_Position_Limit`` registers.

The firmware then refuses to pass the band ends no matter what
commands it: teleop, homing moves, scripts, or mistakes. The limits
persist across power cycles and across this repo's normal startup
(``configure_motors`` never touches them).

Usage:

    venv/bin/python tool/set_wrist_roll_limits.py --arm right
    venv/bin/python tool/set_wrist_roll_limits.py --arm right --reset

Caveats:
- Re-running ``lerobot-calibrate`` on a follower rewrites wrist_roll's
  limits to the full turn (0-4095) — re-run this tool afterwards.
- The calibration JSON deliberately stays at 0-4095: its range defines
  the zero of every degree reading, and shifting it would silently
  break the HW->URDF offsets in configs.py. The EEPROM registers are
  the only thing this tool writes.
"""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

# Encoder geometry (STS3215, matching lerobot's DEGREES normalisation:
# degrees = (ticks - mid) * 360 / MAX_RES with mid from the 0-4095
# calibration range).
MAX_RES = 4095
FULL_MIN = 0
FULL_MAX = 4095
_MID = (FULL_MIN + FULL_MAX) / 2.0

# Reject a band whose BOTH ends hug the encoder wrap: firmware limits
# are a plain min<max window, so a safe arc crossing 0/4095 cannot be
# expressed — the wrist must be re-homed away from the wrap first.
WRAP_GUARD_TICKS = 50
# A cable-safe band narrower than this (after margin) is almost
# certainly a recording mistake (barely rotated before pressing Enter).
MIN_BAND_DEG = 30.0


def deg_to_ticks(deg: float) -> int:
    return round(deg * MAX_RES / 360.0)


def ticks_to_hw_deg(ticks: float) -> float:
    """Raw tick -> hardware degrees (the frame sync_read reports in)."""
    return (ticks - _MID) * 360.0 / MAX_RES


def apply_margin(min_t: int, max_t: int, margin_deg: float) -> tuple[int, int]:
    """Shrink the recorded band by ``margin_deg`` on each side."""
    m = deg_to_ticks(margin_deg)
    return min_t + m, max_t - m


def validate_band(min_t: int, max_t: int, present: int) -> list[str]:
    """Sanity-check a candidate limit band. Returns problems (empty = OK)."""
    problems = []
    if min_t <= FULL_MIN + WRAP_GUARD_TICKS and max_t >= FULL_MAX - WRAP_GUARD_TICKS:
        problems.append(
            "the recorded arc hugs the 0/4095 encoder wrap on BOTH ends — "
            "a wrapping band cannot be expressed as firmware min<max "
            "limits. Re-home wrist_roll (lerobot-calibrate mid-range "
            "step) so the safe arc sits away from the wrap, then retry."
        )
    if ticks_to_hw_deg(max_t) - ticks_to_hw_deg(min_t) < MIN_BAND_DEG:
        problems.append(
            f"band is narrower than {MIN_BAND_DEG:.0f} deg after the margin "
            "— rotate through the FULL cable-safe arc before pressing Enter."
        )
    if not min_t <= present <= max_t:
        problems.append(
            "the wrist's current position lies OUTSIDE the candidate band — "
            "leave the wrist inside the safe arc when finishing."
        )
    return problems


def _print_limits(label: str, min_t: int, max_t: int) -> None:
    print(
        f"  {label}: [{min_t}, {max_t}] ticks  "
        f"= [{ticks_to_hw_deg(min_t):+.1f}, {ticks_to_hw_deg(max_t):+.1f}] hw deg"
    )


def _read_limits(bus) -> tuple[int, int]:
    return (
        int(bus.read("Min_Position_Limit", "wrist_roll", normalize=False)),
        int(bus.read("Max_Position_Limit", "wrist_roll", normalize=False)),
    )


def _write_limits(bus, min_t: int, max_t: int) -> None:
    """Write both EEPROM registers and verify by reading back."""
    bus.write("Min_Position_Limit", "wrist_roll", min_t, normalize=False)
    bus.write("Max_Position_Limit", "wrist_roll", max_t, normalize=False)
    got = _read_limits(bus)
    if got != (min_t, max_t):
        raise SystemExit(
            f"❌ readback mismatch: wrote [{min_t}, {max_t}], motor reports "
            f"{list(got)} — power-cycle the arm and retry."
        )
    _print_limits("✓ limits now", *got)


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        choices=["right", "left"],
        default="right",
        help="which follower arm (right=follower_0, the verified mapping)",
    )
    parser.add_argument(
        "--margin-deg",
        type=float,
        default=5.0,
        help="safety margin kept inside each recorded extreme (degrees)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="restore the full-turn limits (0-4095) instead of recording",
    )
    args = parser.parse_args()

    from common.follower_bus import connect_follower_bus

    bus = connect_follower_bus(args.arm)
    try:
        cur = _read_limits(bus)
        present = int(
            bus.sync_read("Present_Position", ["wrist_roll"], normalize=False)[
                "wrist_roll"
            ]
        )
        print()
        _print_limits("current EEPROM limits", *cur)
        print(
            f"  present position: {present} ticks "
            f"= {ticks_to_hw_deg(present):+.1f} hw deg"
        )

        if args.reset:
            if not _confirm(
                "\nRestore FULL-TURN limits (removes the cable protection)?"
            ):
                print("aborted — nothing written")
                return
            _write_limits(bus, FULL_MIN, FULL_MAX)
            return

        print(
            "\nRotate the wrist by hand through the FULL cable-safe arc —\n"
            "slowly, end to end, stopping where the tactile-camera cable\n"
            "would start to strain. Press Enter when done.\n"
        )
        mins, maxes = bus.record_ranges_of_motion(["wrist_roll"])
        rec_min, rec_max = int(mins["wrist_roll"]), int(maxes["wrist_roll"])
        lim_min, lim_max = apply_margin(rec_min, rec_max, args.margin_deg)
        present = int(
            bus.sync_read("Present_Position", ["wrist_roll"], normalize=False)[
                "wrist_roll"
            ]
        )

        print()
        _print_limits("recorded arc", rec_min, rec_max)
        _print_limits(f"with {args.margin_deg:g} deg margin", lim_min, lim_max)

        problems = validate_band(lim_min, lim_max, present)
        if problems:
            print()
            for p in problems:
                print(f"❌ {p}")
            raise SystemExit(1)

        if not _confirm("\nWrite these limits to the servo EEPROM?"):
            print("aborted — nothing written")
            return
        _write_limits(bus, lim_min, lim_max)
        print(
            "\nDone. Note: re-running lerobot-calibrate on this follower\n"
            "resets wrist_roll to full turn — re-run this tool afterwards."
        )
    finally:
        try:
            bus.disconnect()
        except Exception as e:  # noqa: BLE001 — cleanup must not raise
            print(f"⚠️  bus disconnect failed: {e}")


if __name__ == "__main__":
    main()
