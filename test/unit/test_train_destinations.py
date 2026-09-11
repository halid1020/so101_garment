"""Where a training run may be sent, and the commands that would send it.

No machine is contacted here. What is worth testing about a destination is what
can be got wrong while writing one down -- a path that would run a second
command on the far end, a Slurm entry with no partition, a ``~`` that must NOT
be expanded on this machine -- and what an operator should be told when a box
does not answer. All of that is decided before any connection exists.

The shipped ``src/conf/train_destinations.yaml`` is checked too: it is a
tracked file that every launch reads, so a typo in it is a broken rig, not a
broken test.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_train_destinations
"""

import tempfile
import unittest
from pathlib import Path

import yaml
from actoris_harena.training.destinations import (
    KINDS,
    load_destinations,
    rsync_argv,
    ssh_argv,
)

GOOD = {
    "box": {
        "ssh": "box",
        "kind": "ssh",
        "repo": "~/project/so101_garment",
        "scratch": "~/.cache/huggingface/lerobot",
        "stage": "{scratch}/local",
    }
}


def write(entry) -> Path:
    path = Path(tempfile.mkdtemp()) / "dest.yaml"
    path.write_text(yaml.safe_dump(entry))
    return path


class TestTheShippedFile(unittest.TestCase):
    """The one every launch reads."""

    def setUp(self):
        self.dests = load_destinations()

    def test_it_loads(self):
        self.assertTrue(self.dests)

    def test_both_machines_this_repo_trains_on_are_there(self):
        self.assertIn("create", self.dests)
        self.assertIn("thanos", self.dests)

    def test_every_entry_names_a_kind_that_has_an_executor(self):
        for name, dest in self.dests.items():
            self.assertIn(dest["kind"], KINDS, name)

    def test_the_measured_pi05_ceiling_is_recorded_for_the_cluster(self):
        # Not a preference: batch 8 raised OutOfMemoryError at 39.22 GiB on a
        # 40 GB A100. Losing this number costs a whole reservation.
        self.assertEqual(self.dests["create"]["limits"]["pi05"]["batch"], 4)

    def test_the_measured_pi05_ceiling_is_recorded_for_the_gpu_box_too(self):
        # MEASURED on the 24.5 GiB card, five months after the CREATE figure
        # above and on a smaller one: batch 2 raised OutOfMemoryError at 23.45
        # of 23.55 GiB; batch 1 ran at 20710 MiB for 30 000 steps. Also the
        # reason there is no gradient-accumulation equivalent to reach for --
        # lerobot-train has none, so batch 1 is batch 1.
        self.assertEqual(self.dests["thanos"]["limits"]["pi05"]["batch"], 1)

    def test_the_measured_diffusion_ceiling_is_recorded_for_the_gpu_box(self):
        # NOT about VRAM, which is why it took a crash to find. Diffusion at
        # batch 8 uses under a third of this card -- but batch 32 took the whole
        # BOX down within a minute and batch 16 within about six, both at 24
        # loader workers. The run that finished, and which every diffusion
        # comparison on this rig is drawn against, was batch 8 at 8 workers,
        # read back from its own train_config.json. A row asking for 32 must be
        # refused, not merely discouraged by a comment elsewhere in the file.
        self.assertEqual(self.dests["thanos"]["limits"]["diffusion"]["batch"], 8)

    def test_the_gpu_box_claims_no_ceiling_it_has_not_measured(self):
        # An unmeasured limit written down as if it were measured is worse than
        # none: it would refuse rows that fit and pass rows that do not. act at
        # batch 8 leaves better than half this card free and has never taken the
        # box down, so it has none -- and neither do the policies nobody has run
        # here yet. Adding a name to this set means someone measured it.
        self.assertEqual(set(self.dests["thanos"]["limits"]), {"pi05", "diffusion"})


class TestTheLocalMachine(unittest.TestCase):
    """``kind: local`` -- the machine the console is running on.

    Not a lesser destination: the same driver, the same lock, the same run
    directories and the same manifest. The only difference is that there is no
    ssh in front of the command, and that is deliberately the ONE place the kind
    is consulted -- staging, the manifest, the dispatch, the status and the stop
    then all work on it unchanged.
    """

    def dest(self, **over):
        entry = {
            "kind": "local",
            "repo": ".",
            "scratch": "~/.cache/huggingface/lerobot",
            "stage": "{scratch}/local",
            **over,
        }
        return load_destinations(write({"here": entry}))["here"]

    def test_local_is_a_kind(self):
        self.assertIn("local", KINDS)

    def test_it_needs_no_host(self):
        # There is no host to name, and naming one would be a lie the launcher
        # would then try to reach.
        self.assertEqual(self.dest()["ssh"], "-")

    def test_every_other_kind_still_needs_one(self):
        with self.assertRaises(ValueError) as caught:
            load_destinations(
                write(
                    {
                        "box": {
                            "kind": "ssh",
                            "repo": ".",
                            "scratch": "/s",
                            "stage": "{scratch}/l",
                        }
                    }
                )
            )
        self.assertIn("needs an 'ssh'", str(caught.exception))

    def test_a_dot_repo_means_this_checkout(self):
        # The only local path that is the same on every machine the console
        # runs on -- and it has to be resolved here, because the driver is
        # started from wherever the console happened to be launched.
        repo = Path(self.dest()["repo"])
        self.assertTrue(repo.is_absolute())
        self.assertTrue((repo / "hpc" / "gpu_box_run.sh").is_file())

    def test_a_command_runs_in_a_shell_and_not_over_ssh(self):
        self.assertEqual(ssh_argv(self.dest(), "echo hi"), ["bash", "-lc", "echo hi"])

    def test_rsync_needs_no_host_prefix(self):
        self.assertNotIn(":", rsync_argv("/data/ds", self.dest())[-1])

    def test_a_second_output_root_may_be_named(self):
        # A console started through setup.sh puts SO101_OUTPUT_DIR in the repo,
        # so a run started here lands there rather than in the cache.
        dest = self.dest(outputs="{repo}/outputs")
        self.assertEqual(dest["outputs"], "{repo}/outputs")

    def test_an_output_path_is_checked_like_every_other_path(self):
        with self.assertRaises(ValueError):
            self.dest(outputs="{repo}/out; rm -rf /")
