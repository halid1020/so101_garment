"""A shim. The policies moved to ``actoris_harena.policies``.

WHY THIS EXISTS AND WHEN IT CAN GO. ``--policy.discover_packages_path`` names an
importable package, and that string is written into things that outlive this
change: a Slurm job script submitted last week, a ``train_config.json`` a running
job reads when it resumes, a run-matrix row. A job resuming on CREATE with
``--policy.discover_packages_path=so101_policies`` must still find something that
registers the policies, or it dies at the point it was meant to recover.

Importing this registers them, exactly as before, from their new home. Every name
still resolves -- the new ``harena_*`` and the legacy ``so101_*``, which are
registered as aliases of the same classes.

REMOVE IT once no submitted job can still be resumed against it. The one that
prompted it was on CREATE, submitted 2026-09-07, naming so101_act_crop,
so101_diffusion_crop and so101_pi05_crop; ``tool/train_launch.py --status`` says
whether it has finished.
"""

# Importing the package IS the registration -- it imports every config and
# modeling module, and each carries its @register_subclass decorator. That is the
# same mechanism `--policy.discover_packages_path` relies on, which is why a shim
# that merely re-exported names would not have been enough.
import actoris_harena.policies as _policies  # noqa: F401

__all__: "list[str]" = []
