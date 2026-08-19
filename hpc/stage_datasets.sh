#!/usr/bin/env bash
# =====================================================================
# Stage collected datasets from the COLLECTION BOX to a cluster's scratch.
#
# Run this on the machine that holds the recordings (not on the cluster). It
# copies whole dataset directories into the cluster's $HF_LEROBOT_HOME/local,
# which is where hpc/create_real_vla.sbatch expects to find them.
#
#   bash hpc/stage_datasets.sh --dir /mnt/seagate/so101 \
#       --dest k1234567@<create-login-host>:/scratch/users/k1234567/hf_lerobot/local \
#       cube-pnp fold-short
#
# Options:
#   --dir DIR     collection directory holding the datasets   (required)
#   --dest DEST   rsync destination, local path or user@host:path (required)
#   --dry-run     run rsync in --dry-run mode (still checks the datasets)
#
# Every named dataset is checked BEFORE anything is transferred: it must be a
# real dataset, and it must have no episodes still marked for deletion. Marked
# episodes are only flagged until the dataset is compacted -- they are still on
# disk, so staging one would ship takes the operator threw away, and the
# training driver would refuse the dataset on arrival anyway.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DIR=""
DEST=""
DRY_RUN=0
DATASETS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DIR="$2"; shift 2;;
        --dest) DEST="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0;;
        -*) echo "Unknown arg: $1" >&2; exit 2;;
        *) DATASETS+=("$1"); shift;;
    esac
done

[ -n "$DIR" ] || { echo "❌ --dir is required (the collection directory)" >&2; exit 2; }
[ -n "$DEST" ] || { echo "❌ --dest is required (e.g. user@host:<scratch>/hf_lerobot/local)" >&2; exit 2; }
if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "❌ name at least one dataset. Available under $DIR:" >&2
    ls -1 "$DIR" 2>/dev/null | sed 's/^/   /' >&2 || true
    exit 2
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/setup.sh" >/dev/null
fi
PY="$REPO_ROOT/venv/bin/python"

# ---- check everything before transferring anything -------------------
echo "== checking =="
for ds in "${DATASETS[@]}"; do
    root="$DIR/$ds"
    if [ ! -f "$root/meta/info.json" ]; then
        echo "❌ not a dataset: $root (no meta/info.json)" >&2; exit 2
    fi
    "$PY" - "$root" <<'PY' || exit 2
import sys
from common.recording.dataset_edit import read_soft_deleted

marked = read_soft_deleted(sys.argv[1])
if marked:
    print(
        f"❌ {sys.argv[1]} has {len(marked)} episode(s) marked for deletion that are\n"
        f"   still on disk: {marked}\n"
        "   Open tool/rig_web.py --allow-delete and press 'Remove for good'\n"
        "   (or Restore them) before staging.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
    printf '   ✓ %-22s %s\n' "$ds" "$(du -sh "$root" | cut -f1)"
done

# ---- transfer --------------------------------------------------------
RSYNC=(rsync -avP)
[ "$DRY_RUN" = "1" ] && RSYNC+=(--dry-run)

echo
echo "== staging ${#DATASETS[@]} dataset(s) -> $DEST =="
for ds in "${DATASETS[@]}"; do
    echo "-- $ds"
    "${RSYNC[@]}" "$DIR/$ds" "$DEST/"
done

echo
if [ "$DRY_RUN" = "1" ]; then
    echo "✓ dry run — nothing transferred."
else
    echo "✓ staged. On the cluster, confirm and submit:"
    echo "    ls <scratch>/hf_lerobot/local/${DATASETS[0]}/meta/info.json"
    echo "    bash hpc/submit_real.sh --datasets $(IFS=,; echo "${DATASETS[*]}")"
fi
