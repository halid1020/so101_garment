#!/usr/bin/env bash
# =====================================================================
# Bring trained checkpoints BACK from a cluster run, to the machine that will
# serve them. The return leg of hpc/stage_datasets.sh.
#
# A real policy is evaluated ON THE ROBOT, so a finished training run is of no
# use where it was produced. This copies each run's FINAL checkpoint (and its
# report) out of the cluster's scratch into the layout tool/policy_server.py
# and tool/run_policy_real.py expect: one directory per (dataset, policy).
#
#   bash hpc/fetch_policies.sh --from k1234567@<create-login-host> --list
#   bash hpc/fetch_policies.sh --from k1234567@<create-login-host> \
#       --dest <gpu-host>:project/so101_garment/outputs/policies
#   bash hpc/fetch_policies.sh --from ... --dest ~/outputs/policies \
#       --datasets fold-short --only diffusion
#
# Options:
#   --from USER@HOST  cluster login host  (omit: the runs are on this machine)
#   --scratch DIR     node-visible scratch root   (default /scratch/users/<user>)
#   --dest DEST       where to put them: a local dir or HOST:dir  (required
#                     unless --list)
#   --datasets a,b    only these datasets
#   --only a,b        only these policies (act, diffusion)
#   --as NAME         name the destination after NAME instead of the run
#                     directory (one run at a time). A run resubmitted without
#                     a --run-name is called after its job id, which says
#                     nothing about what it trained.
#   --bridge          copy through THIS machine instead of pulling on the
#                     destination (see below)
#   --list            print what the cluster holds and transfer nothing
#   --dry-run         run rsync in --dry-run mode (still lists and checks)
#
# The whole run directory is tens of gigabytes -- every intermediate checkpoint
# and its optimiser state. Only `checkpoints/last/pretrained_model` is fetched,
# which is what a checkpoint IS for inference: about 200 MB for ACT, 1.1 GB for
# diffusion.
#
# ROUTING. The transfer runs on the machine that will HOLD the weights: with a
# remote --dest, rsync is invoked there and pulls from the cluster over your
# forwarded SSH agent (`ssh -A`), so nothing lands on this machine and no key is
# ever copied to the destination. If the destination cannot reach the cluster --
# a cluster reachable only from inside its own network, for instance -- pass
# --bridge and the bytes come through here instead, in two hops, staged in a
# temporary directory that is removed on the way out. That stage needs room for
# the largest checkpoint, so set $TMPDIR if /tmp is small (a tmpfs, typically).
#
# Every fetched checkpoint is then read back from the destination and reported:
# policy type, observation steps in, actions out, and the CAMERA NAMES it was
# trained on. Those names must be the rig's, or the checkpoint cannot be run
# there -- and that is a training/collection mismatch, discovered here for the
# price of one `cat` rather than at the first rollout.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_SUBDIR="so101_outputs/vla_real_long"

FROM=""
SCRATCH="${SO101_SCRATCH:-}"
DEST=""
FILTER_DATASETS=""
FILTER_POLICIES=""
AS=""
BRIDGE=0
LIST=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2;;
        --scratch) SCRATCH="$2"; shift 2;;
        --dest) DEST="$2"; shift 2;;
        --datasets) FILTER_DATASETS="$2"; shift 2;;
        --only) FILTER_POLICIES="$2"; shift 2;;
        --as) AS="$2"; shift 2;;
        --bridge) BRIDGE=1; shift;;
        --list) LIST=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        -h|--help) sed -n '2,51p' "${BASH_SOURCE[0]}"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [ -z "$SCRATCH" ]; then
    user="${FROM%@*}"
    [ -n "$FROM" ] && [ "$user" != "$FROM" ] || user="${USER:-$(id -un)}"
    SCRATCH="/scratch/users/$user"
fi
RUNS_ROOT="${SCRATCH%/}/$RUNS_SUBDIR"

if [ "$LIST" = "0" ] && [ -z "$DEST" ]; then
    echo "❌ --dest is required (a local directory, or HOST:dir)" >&2; exit 2
fi

# A destination of the form HOST:dir is remote; anything else is a local path.
DEST_HOST=""
DEST_PATH="${DEST%/}"
if [[ "$DEST" =~ ^[^/:]+: ]]; then
    DEST_HOST="${DEST%%:*}"
    DEST_PATH="${DEST#*:}"
    DEST_PATH="${DEST_PATH%/}"
fi

PY="$REPO_ROOT/venv/bin/python"
[ -x "$PY" ] || PY="python3"

# ---- running a snippet on the cluster, or on the destination ---------
run_src() {  # run_src <script> [arg...]   -- on the cluster
    local script="$1"; shift
    if [ -n "$FROM" ]; then
        ssh -o BatchMode=yes "$FROM" "bash -s -- $(printf '%q ' "$@")" <<<"$script"
    else
        bash -s -- "$@" <<<"$script"
    fi
}

run_dest() {  # run_dest <script> [arg...]  -- where the weights will live
    local script="$1"; shift
    if [ -n "$DEST_HOST" ]; then
        ssh -o BatchMode=yes "$DEST_HOST" "bash -s -- $(printf '%q ' "$@")" <<<"$script"
    else
        bash -s -- "$@" <<<"$script"
    fi
}

# What a fetched checkpoint says about itself, read from its own config.json.
READ_CONFIG='
import json
import sys

cfg = json.load(sys.stdin)
prefix = "observation.images."
feats = cfg.get("input_features") or {}
cameras = sorted(k[len(prefix):] for k in feats if k.startswith(prefix))
action = (cfg.get("output_features") or {}).get("action") or {}
dim = (action.get("shape") or ["?"])[0]
steps = cfg.get("n_action_steps") or cfg.get("chunk_size") or "?"
print("   %s: %s observation step(s) in, %s action(s) out, %s-D"
      % (cfg.get("type", "?"), cfg.get("n_obs_steps", 1), steps, dim))
print("   cameras: " + (", ".join(cameras) or "(none)"))
'

# ---- what the cluster holds -----------------------------------------
# One line per (dataset, policy): whether the FINAL checkpoint is there, how
# big it is, and whether the run wrote a report. A run cancelled at its wall
# time has checkpoints but no `last`, and says so here instead of failing a
# transfer half-way.
SCAN='
root="$1"
[ -d "$root" ] || { echo "NOROOT" ; exit 0; }
for run in "$root"/*/; do
    ds="$(basename "$run")"
    [ -d "${run}train" ] || continue
    for pol_dir in "${run}train"/*/; do
        pol="$(basename "$pol_dir")"
        ckpt="${pol_dir}checkpoints/last/pretrained_model"
        report="no"
        [ -f "${run}results_${pol}.md" ] && report="yes"
        if [ -f "$ckpt/model.safetensors" ]; then
            printf "%s\t%s\tready\t%s\t%s\n" "$ds" "$pol" \
                "$(du -sh "$ckpt" | cut -f1)" "$report"
        else
            printf "%s\t%s\tunfinished\t-\t%s\n" "$ds" "$pol" "$report"
        fi
    done
done
'

echo "== reading ${FROM:+$FROM:}$RUNS_ROOT =="
INVENTORY="$(run_src "$SCAN" "$RUNS_ROOT")"
if [ "$INVENTORY" = "NOROOT" ]; then
    echo "❌ no run directory at ${FROM:+$FROM:}$RUNS_ROOT" >&2
    echo "   pass --scratch (or \$SO101_SCRATCH) pointing at the cluster's scratch root." >&2
    exit 2
fi
if [ -z "$INVENTORY" ]; then
    echo "❌ no training runs under ${FROM:+$FROM:}$RUNS_ROOT" >&2; exit 2
fi

selected() {  # selected <dataset> <policy>  -- against the two filters
    local ds="$1" pol="$2"
    # A camera-ablation run is named <dataset>__<camera slug>, so --datasets
    # matches either the whole run name or the dataset it was built from:
    # `--datasets cube-pnp-new` takes all three arms, `cube-pnp-new__all` one.
    local base="${ds%%__*}"
    if [ -n "$FILTER_DATASETS" ] \
       && ! [[ ",$FILTER_DATASETS," == *",$ds,"* ]] \
       && ! [[ ",$FILTER_DATASETS," == *",$base,"* ]]; then
        return 1
    fi
    if [ -n "$FILTER_POLICIES" ] && ! [[ ",$FILTER_POLICIES," == *",$pol,"* ]]; then
        return 1
    fi
    return 0
}

printf '   %-22s %-10s %-11s %8s  %s\n' dataset policy state size report
FETCH=()
while IFS=$'\t' read -r ds pol state size report; do
    [ -n "$ds" ] || continue
    mark="  "
    if selected "$ds" "$pol" && [ "$state" = "ready" ]; then
        mark=" →"
        FETCH+=("$ds:$pol:$report")
    fi
    printf '%s %-22s %-10s %-11s %8s  %s\n' "$mark" "$ds" "$pol" "$state" "$size" "$report"
done <<<"$INVENTORY"

if [ "$LIST" = "1" ]; then
    echo
    echo "✓ listed only. Fetch them with --dest <dir|HOST:dir>."
    exit 0
fi
if [ "${#FETCH[@]}" -eq 0 ]; then
    echo >&2
    echo "❌ nothing to fetch: no finished checkpoint matches the filters." >&2
    exit 2
fi
if [ -n "$AS" ]; then
    runs="$(printf '%s\n' "${FETCH[@]}" | cut -d: -f1 | sort -u | wc -l)"
    if [ "$runs" -ne 1 ]; then
        echo >&2
        echo "❌ --as renames one run's destination, but $runs runs are selected." >&2
        echo "   Narrow it with --datasets." >&2
        exit 2
    fi
fi

# ---- check the route before moving gigabytes -------------------------
RSYNC_OPTS=(-avP)
[ "$DRY_RUN" = "1" ] && RSYNC_OPTS+=(--dry-run)
SRC_PREFIX="${FROM:+$FROM:}"
DIRECT=0
if [ -n "$DEST_HOST" ] && [ -n "$FROM" ] && [ "$BRIDGE" = "0" ]; then
    echo
    echo "== can $DEST_HOST reach $FROM itself? =="
    if ssh -A -o BatchMode=yes "$DEST_HOST" \
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 $(printf '%q' "$FROM") true" \
        >/dev/null 2>&1; then
        DIRECT=1
        echo "   ✓ yes — pulling there, over your forwarded agent"
    else
        echo "❌ $DEST_HOST cannot open an SSH session to $FROM." >&2
        echo "   Either your agent is not forwarding a key the cluster accepts" >&2
        echo "   (check \`ssh-add -l\`), or the cluster is not reachable from there." >&2
        echo "   Re-run with --bridge to route the copy through this machine." >&2
        exit 2
    fi
fi

BRIDGE_DIR=""
cleanup() {
    # An EXIT trap decides the script's status, so it must not end on a
    # false test: a clean run would exit 1 with nothing wrong.
    if [ -n "$BRIDGE_DIR" ]; then rm -rf "$BRIDGE_DIR"; fi
}
trap cleanup EXIT

echo
echo "== fetching ${#FETCH[@]} checkpoint(s) -> ${DEST} =="
for entry in "${FETCH[@]}"; do
    ds="${entry%%:*}"; rest="${entry#*:}"; pol="${rest%%:*}"; report="${rest#*:}"
    src="$RUNS_ROOT/$ds/train/$pol/checkpoints/last/pretrained_model"
    out="$DEST_PATH/${AS:-$ds}-$pol"
    echo "-- $ds / $pol"
    [ "$DRY_RUN" = "1" ] || run_dest 'mkdir -p "$1"' "$out"
    if [ "$DIRECT" = "1" ]; then
        ssh -A "$DEST_HOST" "rsync ${RSYNC_OPTS[*]} \
            -e 'ssh -o StrictHostKeyChecking=accept-new' \
            $(printf '%q' "$FROM:$src/") $(printf '%q' "$out/")"
        if [ "$report" = "yes" ]; then
            ssh -A "$DEST_HOST" "rsync ${RSYNC_OPTS[*]} \
                -e 'ssh -o StrictHostKeyChecking=accept-new' \
                $(printf '%q' "$FROM:$RUNS_ROOT/$ds/results_$pol.md") $(printf '%q' "$out/")"
        fi
    elif [ -n "$DEST_HOST" ]; then
        # Through this machine: pull, then push. Same bytes, one more hop.
        [ -n "$BRIDGE_DIR" ] || BRIDGE_DIR="$(mktemp -d)"
        stage="$BRIDGE_DIR/$ds-$pol"
        mkdir -p "$stage"
        rsync "${RSYNC_OPTS[@]}" "$SRC_PREFIX$src/" "$stage/"
        if [ "$report" = "yes" ]; then
            rsync "${RSYNC_OPTS[@]}" \
                "$SRC_PREFIX$RUNS_ROOT/$ds/results_$pol.md" "$stage/"
        fi
        rsync "${RSYNC_OPTS[@]}" "$stage/" "$DEST_HOST:$out/"
        rm -rf "$stage"
    else
        rsync "${RSYNC_OPTS[@]}" "$SRC_PREFIX$src/" "$out/"
        if [ "$report" = "yes" ]; then
            rsync "${RSYNC_OPTS[@]}" \
                "$SRC_PREFIX$RUNS_ROOT/$ds/results_$pol.md" "$out/"
        fi
    fi
done

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "✓ dry run — nothing transferred."
    exit 0
fi

# ---- read each one back, where it now lives --------------------------
echo
echo "== what arrived =="
FAILED=0
for entry in "${FETCH[@]}"; do
    ds="${entry%%:*}"; rest="${entry#*:}"; pol="${rest%%:*}"
    out="$DEST_PATH/${AS:-$ds}-$pol"
    echo "-- ${AS:-$ds}-$pol"
    if ! run_dest '[ -f "$1/config.json" ] && [ -f "$1/model.safetensors" ]' "$out"; then
        echo "   ❌ incomplete: config.json or model.safetensors is missing" >&2
        FAILED=1
        continue
    fi
    run_dest 'cat "$1/config.json"' "$out" | "$PY" -c "$READ_CONFIG"
done

echo
[ "$FAILED" = "0" ] || { echo "❌ some checkpoints did not arrive intact." >&2; exit 1; }
first_ds="${AS:-${FETCH[0]%%:*}}"
first_pol="${FETCH[0]#*:}"; first_pol="${first_pol%%:*}"
echo "✓ fetched. Those camera names must be the rig's."
echo "  Serve one where it now lives${DEST_HOST:+ (on $DEST_HOST)}:"
echo "      venv/bin/python tool/policy_server.py --checkpoint $DEST_PATH/$first_ds-$first_pol"
echo "  then, on the rig (see documents/remote_policy_inference.md):"
echo "      ssh -N -L 8765:127.0.0.1:8765 ${DEST_HOST:-<gpu-host>}"
echo "      venv/bin/python tool/run_policy_real.py --server http://127.0.0.1:8765 \\"
echo "          --task \"<the task it was trained on>\" --dry-run"
