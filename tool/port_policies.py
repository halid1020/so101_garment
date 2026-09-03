#!/usr/bin/env python
"""Re-derive the ported policies from the LeRobot checkout.

Run after bumping ``LEROBOT_COMMIT``. Every rule lives in
``so101_policies._port``; this only applies them and reports what moved.

    venv/bin/python tool/port_policies.py [--check]

``--check`` writes nothing and exits non-zero if any ported file is out of
date -- the same thing ``test/unit/test_policy_ports.py`` asserts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from so101_policies._port import port_text, ported_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    stale = []
    for upstream, ours, name, kind in ported_files():
        if not upstream.is_file():
            print(f"❌ upstream missing: {upstream}")
            return 2
        want = port_text(upstream.read_text(), name, kind)
        have = ours.read_text() if ours.is_file() else None
        if have == want:
            print(f"  = {ours.relative_to(Path.cwd()) if ours.is_absolute() else ours}")
            continue
        stale.append(ours)
        if args.check:
            print(f"  ✗ out of date: {ours}")
        else:
            ours.write_text(want)
            print(f"  ↻ rewrote: {ours}")

    if args.check and stale:
        print(f"\n❌ {len(stale)} ported file(s) out of date — run without --check")
        return 1
    print(f"\n✅ {len(list(ported_files()))} ported files in step with upstream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
