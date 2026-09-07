"""Every policy this rig trains, implemented here rather than in LeRobot.

Importing this package REGISTERS each policy with LeRobot's draccus registry,
which is the whole point of it: ``lerobot.configs.parser.wrap`` loads a package
named by ``--policy.discover_packages_path`` before draccus parses anything, and
``PreTrainedConfig.register_subclass`` fires as a side effect of that import.
After it, ``lerobot.policies.factory.get_policy_class`` resolves our names by
the same route as its own, so one command trains any of them::

    lerobot-train --policy.discover_packages_path=so101_policies \
                  --policy.type=so101_act ...

LeRobot then derives the rest of the wiring from the config class NAME, purely
mechanically (``policies/factory.py:606``), so the naming is a contract and not
a style:

    so101_policies/<x>/configuration_<x>.py   So101<X>Config, registered "so101_<x>"
    so101_policies/<x>/modeling_<x>.py        So101<X>Policy
    so101_policies/<x>/processor_<x>.py       make_so101_<x>_pre_post_processors

Three of these -- act, diffusion, pi05 -- are PORTS: the upstream module tree
moved here unchanged, so their ``state_dict`` keys are identical to LeRobot's
and every checkpoint already trained still loads. ``test/unit/test_policy_ports.py``
holds them to that. The rest are ours.
"""

from so101_policies.act.configuration_act import So101ActConfig
from so101_policies.act.modeling_act import So101ActPolicy
from so101_policies.act_crop.configuration_act_crop import So101ActCropConfig
from so101_policies.act_crop.modeling_act_crop import So101ActCropPolicy
from so101_policies.diffusion.configuration_diffusion import So101DiffusionConfig
from so101_policies.diffusion.modeling_diffusion import So101DiffusionPolicy
from so101_policies.diffusion_crop.configuration_diffusion_crop import (
    So101DiffusionCropConfig,
)
from so101_policies.diffusion_crop.modeling_diffusion_crop import (
    So101DiffusionCropPolicy,
)
from so101_policies.dreamzero.configuration_dreamzero import So101DreamzeroConfig
from so101_policies.dreamzero.modeling_dreamzero import So101DreamzeroPolicy
from so101_policies.flowmatch.configuration_flowmatch import So101FlowmatchConfig
from so101_policies.flowmatch.modeling_flowmatch import So101FlowmatchPolicy
from so101_policies.pi05.configuration_pi05 import So101Pi05Config
from so101_policies.pi05.modeling_pi05 import So101Pi05Policy
from so101_policies.pi05_crop.configuration_pi05_crop import So101Pi05CropConfig
from so101_policies.pi05_crop.modeling_pi05_crop import So101Pi05CropPolicy

#: Registered type -> the LeRobot type it was ported from. A checkpoint written
#: by either side of a pair carries the other's name in ``config.json``, and
#: this is what lets one be loaded as the other.
PORTED_FROM = {
    "so101_act": "act",
    "so101_diffusion": "diffusion",
    "so101_pi05": "pi05",
}

__all__ = [
    "PORTED_FROM",
    "So101ActConfig",
    "So101ActCropConfig",
    "So101ActCropPolicy",
    "So101ActPolicy",
    "So101DiffusionConfig",
    "So101DiffusionCropConfig",
    "So101DiffusionCropPolicy",
    "So101DiffusionPolicy",
    "So101DreamzeroConfig",
    "So101DreamzeroPolicy",
    "So101FlowmatchConfig",
    "So101FlowmatchPolicy",
    "So101Pi05Config",
    "So101Pi05CropConfig",
    "So101Pi05CropPolicy",
    "So101Pi05Policy",
]
