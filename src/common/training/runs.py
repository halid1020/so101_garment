"""Which log, on which machine, and how little of it has to travel.

``common.training.progress`` turns a log into a curve. This decides where that
log is, gets it off the machine that holds it, and finds the runs nobody told us
about -- the nine finished directories on CREATE were all started from a
terminal, and a viewer that listed only what the console launched would show an
empty page next to a full drive.

Three things here are deliberate.

**One round trip, not four.** A remote read is a whole SSH, so each question is
one command whose answer carries its own section markers. The log itself is
filtered ON THE FAR SIDE: the live ACT log on thanos is 5.8 MB of tqdm frames
and 723 useful lines, and after the filter it is 190 KB -- roughly constant as
the run grows, because what grows is the part that is dropped.

**The tqdm frame rides along.** After ``\\r`` is translated, a metric line and
the progress frame that precedes it are the SAME line, so grepping for the
metric line brings the exact step with it. Matching progress frames separately
looked equivalent and shipped 1.7 MB instead of 190 KB.

**Standard output only.** CREATE prints an MFA banner and a QR code on stderr at
every login. Anything that falls back to stderr when stdout is empty -- which is
what "no such job" looks like -- renders that banner as a status.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from common.training.destinations import ssh_argv

SENTINEL = "@@so101:"

# What a filtered log must keep: the metric lines (with their progress frame),
# the exact checkpoint steps, the end, and any way a run can go wrong. ASCII
# only -- this reaches a remote shell, and the driver's tick and cross do not
# survive every locale on the way.
KEEP_PATTERN = (
    "step:|Checkpoint policy after step|End of training|Traceback"
    "|Error|error:|row [0-9]+ FAILED|row [0-9]+ done|reusing checkpoint"
    # A resumed run appends to the log it already had, so without this line the
    # curve would show one continuous run where in fact the box rebooted and the
    # driver picked the checkpoint back up.
    "|resuming"
)

# 20 000 lines is 2 000 000 steps at the log_freq this repo uses; the longest
# run here is 100 000. A run past that would lose its early points, and with
# them the ordinal the CREATE step count is derived from.
BODY_LINES = 20000
HEAD_BYTES = 20000
READ_TIMEOUT_S = 90.0
DISCOVER_TIMEOUT_S = 60.0

# A run directory is `<dataset>__<slug>`, and a slug joins camera names with
# `+`. These names reach a remote shell inside a command, so what they may
# contain is checked here rather than escaped: a directory that cannot be named
# safely is one this rig did not create.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class RunsError(ValueError):
    """A name that cannot be sent, or an answer that cannot be read."""


def check_name(name: str, what: str = "name") -> str:
    if not _NAME_RE.match(str(name or "")):
        raise RunsError(
            f"{what} {name!r} is not a plain name. Run directories and policies "
            "reach the remote shell inside a command, so they may hold only "
            "letters, digits and . _ + -"
        )
    return str(name)


# ── Where the runs are ───────────────────────────────────────────────────────


def out_roots(dest: "dict[str, Any]") -> "list[str]":
    """Every directory on that machine that may hold run directories.

    Normally one: the drivers both put ``SO101_OUTPUT_DIR`` at
    ``<scratch>/so101_outputs``. A destination may name a second in ``outputs``,
    which is what makes the local machine work -- a console started through
    ``setup.sh`` has ``SO101_OUTPUT_DIR`` pointing into the repo, so runs
    started here land beside the older ones rather than in the cache.
    """
    roots = [f"{dest['scratch']}/so101_outputs"]
    extra = dest.get("outputs")
    if extra:
        formatted = str(extra).format(scratch=dest["scratch"], repo=dest["repo"])
        if formatted not in roots:
            roots.append(formatted)
    return roots


def view_dir_name(dataset: str, cameras: str, info: "dict[str, Any] | None") -> str:
    """``<dataset>__<slug>`` -- the directory both drivers name a run after.

    The slug is the camera view's, computed the way ``make_camera_view`` does,
    because that is the name the run directory actually gets. Joining the row's
    camera list would be close and wrong: ``central,wrist_left`` is a row, and
    ``central+wrist_left`` is the directory.
    """
    if info is None:
        return f"{dataset}__{cameras.replace(',', '+')}"
    from common.recording.dataset_view import split_selection, view_slug

    keep, composites = split_selection(info, str(cameras or "all").split(","))
    slug = "+".join(filter(None, [view_slug(info, keep) if keep else "", *composites]))
    return f"{dataset}__{slug}"


def run_dir_names(record: "dict[str, Any]") -> "list[str]":
    """Every directory a recorded run may have written into.

    ``--run-tag`` when one was given, else what was recorded at launch, else the
    camera view derived from the row -- which is what the runs launched before
    this branch, and everything started from a terminal, have to fall back to.
    """
    tag = record.get("run_tag")
    if tag:
        return [str(tag)]
    recorded = record.get("run_dir") or record.get("run_dirs")
    if recorded:
        return (
            [str(recorded)] if isinstance(recorded, str) else [str(r) for r in recorded]
        )
    dataset = record["dataset"]
    return [
        view_dir_name(dataset, str(c), None) for c in (record.get("cameras") or ["all"])
    ]


def run_dir_name(record: "dict[str, Any]") -> str:
    """The first directory of ``run_dir_names``. Convenience."""
    return run_dir_names(record)[0]


def log_path(out_root: str, run: str, policy: str) -> str:
    return f"{out_root}/vla_real_long/{run}/logs/train_{policy}.log"


# ── The commands ─────────────────────────────────────────────────────────────


def read_command(path: str) -> str:
    """One command that returns everything needed to draw a run. Pure.

    The sections are the configuration lerobot dumps before training, the
    filtered metric lines, the raw tail (where a traceback that matched nothing
    still shows), and the file's size and modification time against the
    machine's OWN clock -- which is how staleness is judged, because the
    timestamps inside the log carry no timezone.
    """
    return (
        f"L={path}; "
        'if [ ! -f "$L" ]; then echo \'' + SENTINEL + "missing'; exit 0; fi; "
        "echo '" + SENTINEL + f'head\'; head -c {HEAD_BYTES} "$L"; echo; '
        "echo '" + SENTINEL + "body'; "
        f"tr '\\r' '\\n' < \"$L\" | grep -aE '{KEEP_PATTERN}' | tail -n {BODY_LINES}; "
        "echo '" + SENTINEL + "tail'; "
        # The `echo` after each section is load-bearing: a progress bar's last
        # frame ends WITHOUT a newline, so the next marker landed on the end of
        # it -- which put `@@so101:stat` in the text shown to the operator and
        # lost the file's age, which is the whole basis for calling a run
        # stalled.
        "tail -c 4000 \"$L\" | tr '\\r' '\\n' | tail -n 4; echo; "
        "echo '" + SENTINEL + "stat'; stat -c '%s %Y' \"$L\"; date +%s"
    )


def discover_command(dest: "dict[str, Any]") -> str:
    """Every run directory and training log on a machine, in one command. Pure.

    Also reports what ``checkpoints/last`` resolves to, which is the only exact
    step written anywhere structured, and the machine's clock, so an age can be
    computed without comparing two machines' timezones.
    """
    parts = []
    for root in out_roots(dest):
        parts.append(
            f"for f in {root}/vla_real_long/*/logs/train_*.log; do "
            '[ -f "$f" ] && stat -c "LOG %n %s %Y" "$f"; done; '
            f"for c in {root}/vla_real_long/*/train/*/checkpoints/last; do "
            '[ -e "$c" ] && echo "CKPT $c $(basename "$(readlink -f "$c")")"; done'
        )
    return "; ".join(parts) + "; echo '" + SENTINEL + "now'; date +%s"


def command_argv(dest: "dict[str, Any]", command: str) -> "list[str]":
    """How to run a shell command on that machine. Pure.

    The local machine is not a special case anywhere else in this feature: the
    same driver, the same lock, the same run directories, and here the same
    command -- only without the SSH in front of it.
    """
    if dest.get("kind") == "local":
        return ["bash", "-lc", command]
    return ssh_argv(dest, command)


# ── Running them ─────────────────────────────────────────────────────────────


def run_command(
    dest: "dict[str, Any]", command: str, timeout: float = READ_TIMEOUT_S
) -> "tuple[str, str | None]":
    """``(stdout, problem)``. Never raises, and never returns stderr as output.

    stderr is reported only as a problem, never as an answer: CREATE greets
    every login with an MFA banner and a QR code on stderr, and a caller that
    substitutes it for empty output shows the operator a QR code where a status
    should be.
    """
    argv = command_argv(dest, command)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            # `head -c` counts BYTES, and a progress bar is drawn with
            # three-byte block characters, so the head of a log is routinely
            # cut through the middle of one. Without this the whole read raises
            # UnicodeDecodeError and the run cannot be watched at all --
            # measured on a real local run, at byte 20012.
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", f"{dest.get('name', 'the machine')} did not answer in {timeout:.0f}s"
    except OSError as exc:
        return "", f"could not run {argv[0]}: {exc}"
    if proc.returncode != 0 and not proc.stdout.strip():
        return "", (proc.stderr or "").strip() or f"exit {proc.returncode}"
    return proc.stdout, None


# ── Reading the answers ──────────────────────────────────────────────────────


def parse_sections(text: str) -> "dict[str, str]":
    """Split a sentinel-delimited answer. Pure. Unknown sections are kept."""
    out: "dict[str, str]" = {}
    current = ""
    lines: "list[str]" = []
    for line in text.splitlines():
        if line.startswith(SENTINEL):
            if current:
                out[current] = "\n".join(lines)
            current = line[len(SENTINEL) :].strip()
            lines = []
        elif current:
            lines.append(line)
    if current:
        out[current] = "\n".join(lines)
    return out


def read_log(dest: "dict[str, Any]", path: str) -> "dict[str, Any]":
    """Fetch and parse one training log. Blocking; never raises."""
    from common.training.progress import parse_log, parse_markers, parse_tqdm, summarise

    text, problem = run_command(dest, read_command(path))
    if problem:
        return {"ok": False, "problem": problem, "path": path}
    sections = parse_sections(text)
    if "missing" in sections or "body" not in sections:
        return {"ok": False, "problem": f"no log at {path}", "path": path}

    age = None
    stat = (sections.get("stat") or "").split()
    if len(stat) >= 3:
        try:
            age = max(0.0, float(stat[2]) - float(stat[1]))
        except ValueError:
            age = None

    # The curve comes from the filtered body ALONE. The tail overlaps it -- it
    # is the last few lines of the same file -- and appending it counted those
    # points twice, which put a finished 30 000-step run at 30 200 and its
    # fraction over one. The tail is still read, for two things the body cannot
    # give: the freshest progress frame (written every step, where a metric line
    # is written every hundredth) and a traceback whose wording matched no
    # filter.
    parsed = parse_log(sections["body"], head_text=sections.get("head"))
    tail_text = sections.get("tail", "").replace("\r", "\n")
    if tail_text.strip():
        frames = parse_tqdm(tail_text)
        if frames:
            parsed["last_frame"] = frames[-1]
        marks = parse_markers(tail_text)
        parsed["finished"] = parsed["finished"] or marks["finished"]
        parsed["failure"] = parsed["failure"] or marks["failure"]
    return {
        "ok": True,
        "path": path,
        "summary": summarise(parsed, age_s=age),
        "points": parsed["points"],
        "problems": parsed["problems"],
        "failure": parsed["failure"],
        "checkpoints": parsed["checkpoints"],
        "tail": sections.get("tail", "").strip(),
    }


_LOG_LINE_RE = re.compile(
    r"^LOG (?P<path>\S+)/vla_real_long/(?P<run>[^/]+)/logs/train_(?P<policy>[^/.]+)\.log"
    r" (?P<size>\d+) (?P<mtime>\d+)$"
)
_CKPT_RE = re.compile(
    r"^CKPT \S+/vla_real_long/(?P<run>[^/]+)/train/(?P<policy>[^/]+)"
    r"/checkpoints/last (?P<step>\S+)$"
)


def parse_discovery(text: str) -> "list[dict[str, Any]]":
    """The listing -> one entry per (run, policy) found on that machine. Pure."""
    sections = parse_sections(text)
    now = None
    try:
        now = float((sections.get("now") or "").split()[0])
    except (IndexError, ValueError):
        now = None
    body = text.split(SENTINEL, 1)[0]

    found: "dict[tuple, dict[str, Any]]" = {}
    for line in body.splitlines():
        match = _LOG_LINE_RE.match(line.strip())
        if match:
            key = (match["run"], match["policy"])
            mtime = float(match["mtime"])
            found.setdefault(
                key,
                {
                    "run": match["run"],
                    "policy": match["policy"],
                    "out_root": match["path"],
                    "size": int(match["size"]),
                    "mtime": mtime,
                    "age_s": (now - mtime) if now else None,
                    "checkpoint": None,
                },
            )
            continue
        ckpt = _CKPT_RE.match(line.strip())
        if ckpt:
            key = (ckpt["run"], ckpt["policy"])
            entry = found.get(key)
            step = ckpt["step"]
            if entry is not None:
                entry["checkpoint"] = int(step) if step.isdigit() else step
    return sorted(found.values(), key=lambda e: (-(e["mtime"] or 0), e["run"]))


def discover(dest: "dict[str, Any]") -> "dict[str, Any]":
    """What training this machine holds. Blocking; never raises."""
    text, problem = run_command(dest, discover_command(dest), DISCOVER_TIMEOUT_S)
    if problem:
        return {"ok": False, "problem": problem, "runs": []}
    return {"ok": True, "problem": None, "runs": parse_discovery(text)}


# ── What this machine can do (the local destination) ─────────────────────────

# The rule test/system/long_vla_real.sh applies: below this it picks the CPU,
# and an 80 000-step run that quietly becomes a CPU run is worse than one that
# is refused, because it looks like it is working for a week.
MIN_GPU_GB = 8.0


def local_capability(python: "str | Path | None" = None) -> "dict[str, Any]":
    """This machine's GPU, measured now rather than read from a file.

    The destinations file is checked in and this console is meant to run on more
    than one machine, so a VRAM figure written there would be a fact about
    whoever wrote it. Ask torch instead.
    """
    exe = str(
        python or (Path(__file__).resolve().parents[3] / "venv" / "bin" / "python")
    )
    probe = (
        "import json,torch\n"
        "d={'cuda':torch.cuda.is_available()}\n"
        "if d['cuda']:\n"
        "    p=torch.cuda.get_device_properties(0)\n"
        "    d['gpu']=p.name; d['vram_gb']=round(p.total_memory/1e9,2)\n"
        "print(json.dumps(d))"
    )
    try:
        proc = subprocess.run(
            [exe, "-c", probe], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "problem": f"could not ask torch: {exc}"}
    if proc.returncode != 0:
        return {"ok": False, "problem": (proc.stderr or "torch would not load").strip()}
    try:
        import json

        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        return {"ok": False, "problem": f"could not read torch's answer: {exc}"}

    vram = float(data.get("vram_gb") or 0.0)
    device = "cuda" if data.get("cuda") and vram >= MIN_GPU_GB else "cpu"
    return {
        "ok": True,
        "problem": None,
        "cuda": bool(data.get("cuda")),
        "gpu": data.get("gpu"),
        "vram_gb": vram or None,
        "device": device,
        "why": _device_why(bool(data.get("cuda")), vram),
    }


def _device_why(cuda: bool, vram: float) -> str:
    if not cuda:
        return "torch cannot use a GPU here, so training would run on the CPU"
    if vram < MIN_GPU_GB:
        return (
            f"this GPU has {vram:.1f} GB and the driver wants at least "
            f"{MIN_GPU_GB:.0f} GB, so training would fall back to the CPU"
        )
    return f"{vram:.1f} GB of VRAM"


def with_measurements(dest: "dict[str, Any]") -> "dict[str, Any]":
    """The destination, plus what the machine says about itself. Blocking.

    Only the local machine is asked, and only because it is the one whose
    hardware this repo cannot write down: ``train_destinations.yaml`` is checked
    in and the console is meant to run on more than one machine. A remote
    destination's ceilings are MEASURED and recorded in that file, which is the
    right place for a number somebody had to observe.
    """
    if dest.get("kind") != "local":
        return dest
    return {**dest, "measured": local_capability()}
