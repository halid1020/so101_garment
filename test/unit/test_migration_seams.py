"""Two seams the move to `actoris_harena` opened, and a guard for each.

The package holds the logic; this repo holds the RIG. `common/rig_profile.py`
is the one place the rig hands the package its facts -- where its destinations
file is, which cameras compose, which action columns are grippers -- and it runs
as a side effect of importing `common`. Anything that reaches the package by
another route gets an unconfigured copy.

Both failure modes were live on this branch:

* **Loud.** `tool/train_launch.py` imported `actoris_harena.training.*` and
  never `common`, so every subcommand died on "no train_destinations.yaml has
  been declared".
* **Quiet, and worse.** The camera profile reads EMPTY rather than raising, so
  `all` expands without knowing what a composite is, `short_name` stops eliding
  `wrist_camera_`, and a run directory gets a different slug from the one the
  console and the cluster drivers produce -- a run that trains fine and lands
  where nothing looks for it.

The second seam is the shell drivers. Their Python lives inside heredocs, which
no test collects and no linter reads, so three of them kept importing
`common.recording.dataset_edit` for a module that had moved. `long_vla_real.sh`
guards that heredoc with `|| exit 2`, so training died on EVERY destination --
slurm, ssh and local alike -- while the whole unit tier stayed green.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_migration_seams
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Package surfaces that only answer correctly once the rig has declared itself.
RIG_CONFIGURED = (
    "actoris_harena.training.destinations",
    "actoris_harena.training.runs",
    "actoris_harena.recording.camera_profile",
    "actoris_harena.recording.dataset_view",
    "actoris_harena.action_layout",
)

#: Subpackages that moved out of this repo. A `from common.<x>` naming one of
#: these is a stale import, wherever it appears.
MOVED = ("recording", "training", "analysis", "deploy", "web")


def shell_scripts() -> "list[Path]":
    return sorted(
        p
        for d in ("hpc", "test/system")
        for p in (REPO / d).rglob("*.sh")
        if p.is_file()
    )


class ToolsDeclareTheRigTest(unittest.TestCase):
    def test_every_tool_that_needs_the_rig_imports_common(self):
        offenders = []
        for path in sorted((REPO / "tool").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if not any(surface in text for surface in RIG_CONFIGURED):
                continue
            if not re.search(r"^\s*(import common\b|from common\b)", text, re.M):
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            "these reach a rig-configured part of actoris_harena without "
            "importing `common`, so rig_profile never runs: "
            f"{', '.join(offenders)}. Add `import common  # noqa: F401`.",
        )

    def test_the_guard_would_notice(self):
        # A guard that cannot fail is not a guard. Prove the detector fires on
        # text shaped like the bug it exists for.
        bad = "from actoris_harena.training.destinations import destination\n"
        self.assertTrue(any(s in bad for s in RIG_CONFIGURED))
        self.assertIsNone(re.search(r"^\s*(import common\b|from common\b)", bad, re.M))


class ShellDriversImportTheMovedPackageTest(unittest.TestCase):
    """The heredocs no linter reads and no test collects."""

    def test_no_script_imports_a_module_that_moved(self):
        stale = []
        pattern = re.compile(r"from common\.(%s)\b" % "|".join(MOVED))
        for path in shell_scripts():
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if pattern.search(line):
                    stale.append(f"{path.relative_to(REPO)}:{number}")
        self.assertEqual(
            stale,
            [],
            "these import a subpackage that moved to actoris_harena; the "
            "heredoc dies at runtime and, in long_vla_real.sh, takes the whole "
            f"training run with it: {', '.join(stale)}",
        )

    def test_the_drivers_are_actually_being_read(self):
        # Guards against the list of directories going empty or wrong, which
        # would make the test above pass by checking nothing.
        scripts = shell_scripts()
        self.assertGreater(len(scripts), 3)
        names = {p.name for p in scripts}
        self.assertIn("long_vla_real.sh", names)


if __name__ == "__main__":
    unittest.main()
