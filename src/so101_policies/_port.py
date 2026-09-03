"""How the three ported policies are derived from LeRobot's, in one place.

``act``, ``diffusion`` and ``pi05`` are not rewritten here -- they are the
upstream modules moved into this repo so we own and can edit them. Keeping them
mechanically derived from upstream buys two things that a hand-edited copy would
lose within a week:

* **Checkpoints stay interchangeable.** The module tree is untouched, so the
  ``state_dict`` keys are identical and the finished 80 000-step ACT checkpoint
  loads into either implementation. ``test/unit/test_policy_ports.py`` proves it
  on the real file, not in principle.
* **A LeRobot bump is one command.** ``tool/port_policies.py`` re-derives every
  ported file; ``test_policy_ports.py`` fails the moment one drifts by hand.

Everything the port changes is below. Nothing else may change -- if a port needs
a real edit, it stops being a port and the rule for that file comes out of here.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Where the LeRobot source checkout lives, relative to this repo.
UPSTREAM = (
    Path(__file__).resolve().parents[2].parent
    / "lerobot"
    / "src"
    / "lerobot"
    / "policies"
)

#: ported directory -> (upstream type, upstream config class, upstream policy class, our stem)
PORTS = {
    "act": ("act", "ACTConfig", "ACTPolicy", "So101Act"),
    "diffusion": ("diffusion", "DiffusionConfig", "DiffusionPolicy", "So101Diffusion"),
    "pi05": ("pi05", "PI05Config", "PI05Policy", "So101Pi05"),
}

#: A ported module imports from ``lerobot.policies`` what it used to reach by a
#: relative import, because only its home moved.
_RELATIVE = (
    (r"^(\s*)from \.\.pretrained import", r"\1from lerobot.policies.pretrained import"),
    (r"^(\s*)from \.\.utils import", r"\1from lerobot.policies.utils import"),
    (r"^(\s*)from \.\.rtc\.", r"\1from lerobot.policies.rtc."),
    (r"^(\s*)from \.\.pi_gemma import", r"\1from lerobot.policies.pi_gemma import"),
)

_CONFIG_TAIL = """

# The body above is the upstream module, unchanged, so that a checkpoint trained
# by either implementation loads into the other. Only the class name moved --
# LeRobot derives the policy class from it mechanically -- and this alias keeps
# the sibling modules' imports reading as they do upstream.
{old_config} = {stem}Config
"""

_PROCESSOR_TAIL = """

# lerobot.policies.factory._make_processors_from_policy_config looks for
# make_<registered type>_pre_post_processors in this module, by name.
make_so101_{typ}_pre_post_processors = make_{typ}_pre_post_processors
"""


def port_text(text: str, name: str, kind: str) -> str:
    """Return the ported form of one upstream file. Pure.

    ``kind`` is ``configuration``, ``modeling`` or ``processor``.
    """
    typ, old_config, old_policy, stem = PORTS[name]
    for pattern, repl in _RELATIVE:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)

    if kind == "configuration":
        text = text.replace(
            f'@PreTrainedConfig.register_subclass("{typ}")',
            f'@PreTrainedConfig.register_subclass("so101_{typ}")',
        )
        text = re.sub(
            rf"^class {old_config}\(", f"class {stem}Config(", text, flags=re.MULTILINE
        )
        return text + _CONFIG_TAIL.format(old_config=old_config, stem=stem)

    if kind == "modeling":
        text = re.sub(
            rf"^class {old_policy}\(", f"class {stem}Policy(", text, flags=re.MULTILINE
        )
        return re.sub(
            rf'^(    name = )"{typ}"$', rf'\1"so101_{typ}"', text, flags=re.MULTILINE
        )

    if kind == "processor":
        # ProcessorStepRegistry is a GLOBAL namespace, so a ported step whose name
        # is unchanged collides with the upstream one the moment both are imported
        # -- which is exactly what the equivalence test does. This is the only
        # rename that is not cosmetic.
        text = text.replace(
            'ProcessorStepRegistry.register(name="pi05_',
            'ProcessorStepRegistry.register(name="so101_pi05_',
        )
        return text + _PROCESSOR_TAIL.format(typ=typ)

    raise ValueError(f"unknown kind {kind!r}")


def ported_files() -> "list[tuple[Path, Path, str, str]]":
    """(upstream path, our path, port name, kind) for every ported file."""
    here = Path(__file__).resolve().parent
    out = []
    for name in PORTS:
        for kind in ("configuration", "modeling", "processor"):
            stem = f"{kind}_{name}.py"
            out.append((UPSTREAM / name / stem, here / name / stem, name, kind))
    return out
