#!/usr/bin/env bash
# =====================================================================
# Submit real-data VLA training to KCL CREATE (Slurm) from a run manifest.
#
# Run this on a CREATE *login* node, from the repo. It reads hpc/runs.tsv --
# one row per (dataset, policy, cameras) training run -- filters it, checks every
# selected dataset is staged, and submits the rows as Slurm job arrays, one
# array task per row. Nothing in the repo has to be edited to change which
# datasets are trained: edit the manifest, or filter it on the command line.
#
#   bash hpc/submit_real.sh                          # everything in runs.tsv
#   bash hpc/submit_real.sh --datasets cube-pnp      # one dataset
#   bash hpc/submit_real.sh --only act               # one policy
#   bash hpc/submit_real.sh --cameras all            # one arm of the ablation
#   bash hpc/submit_real.sh --dry-run                # print the sbatch commands
#
# Options:
#   --manifest F      run matrix to read           (default hpc/runs.tsv)
#   --datasets a,b    only these datasets
#   --only a,b        only these policies          (act, diffusion, pi05)
#   --cameras SET     only rows whose cameras column is EXACTLY this
#                     (`all`, or the row's own comma list -- not a list of sets)
#   --scratch DIR     node-visible scratch root    (default /scratch/users/$USER)
#   --partition P     Slurm partition              (default: the sbatch's own)
#   --account A       Slurm account, if enforced
#   --exclude NODES   Slurm nodes to keep off, comma-separated. Use it when a
#                     node has an unhealthy GPU: the driver refuses one that
#                     allocated a GPU torch cannot open, and names the node.
#                     NOT $SBATCH_EXCLUDE -- CREATE's Slurm ignores that.
#   --concurrency N   cap simultaneously running array tasks
#   --job-name N      Slurm job name               (default real_vla)
#   --dry-run         prepare everything, print the sbatch commands, submit none
#
# Rows are grouped by their `hours` column and each group is submitted as its
# own array with that wall time, so a short cell does not queue behind a long
# reservation. Each group's rows are SNAPSHOTTED under
# <scratch>/so101_outputs/submissions/<stamp>/ and the array reads the snapshot,
# so editing the manifest afterwards cannot shift the indices of a queued array.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST="$REPO_ROOT/hpc/runs.tsv"
FILTER_DATASETS=""
FILTER_POLICIES=""
FILTER_CAMERAS=""
EXCLUDE=""
SCRATCH="${SO101_SCRATCH:-/scratch/users/${USER:-$(id -un)}}"
PARTITION=""
ACCOUNT=""
CONCURRENCY=""
JOB_NAME="real_vla"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2;;
        --datasets) FILTER_DATASETS="$2"; shift 2;;
        --only) FILTER_POLICIES="$2"; shift 2;;
        --cameras) FILTER_CAMERAS="$2"; shift 2;;
        --exclude) EXCLUDE="$2"; shift 2;;
        --scratch) SCRATCH="$2"; shift 2;;
        --partition) PARTITION="$2"; shift 2;;
        --account) ACCOUNT="$2"; shift 2;;
        --concurrency) CONCURRENCY="$2"; shift 2;;
        --job-name) JOB_NAME="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

[ -f "$MANIFEST" ] || { echo "❌ no manifest at $MANIFEST" >&2; exit 2; }

# Print a command the way a person can paste it back: quote only the words
# that need it, so the export list's commas stay readable.
print_cmd() {
    local word out=""
    for word in "$@"; do
        case "$word" in
            *[[:space:]\'\"\$\`]*) word="'"'"'${word}'"'"'";;
        esac
        out+="$word "
    done
    printf '%s\n' "$out"
}

in_list() {  # value, comma-list ("" = everything matches)
    [ -z "$2" ] && return 0
    case ",$2," in *",$1,"*) return 0;; *) return 1;; esac
}

# ---- read + filter the manifest --------------------------------------
ROWS=()
LINE_NO=0
while IFS= read -r line || [ -n "$line" ]; do
    LINE_NO=$((LINE_NO + 1))
    [[ "$line" =~ ^[[:space:]]*(#.*)?$ ]] && continue
    # `extra` is the last column and may contain spaces: read takes the rest.
    read -r ds policy cameras steps batch hours slots extra <<<"$line"
    if [ -z "${hours:-}" ]; then
        echo "❌ $MANIFEST:$LINE_NO needs 7 fields (dataset policy cameras steps batch hours slots): $line" >&2
        exit 2
    fi
    # The `slots` column was added after the first matrices were written, and a
    # row from before it has SEVEN fields -- so `read` puts the old `extra` into
    # `slots` and leaves `extra` empty. Catching it on the empty `extra` rather
    # than on an empty `slots` is what tells the two apart; reading such a row
    # leniently would pass a string of lerobot-train flags as a camera mapping.
    if [ -z "${extra:-}" ]; then
        echo "❌ $MANIFEST:$LINE_NO has 7 fields, not 8 -- the 'slots' column was added." >&2
        echo "   Put a '-' before the last column: $line" >&2
        exit 2
    fi
    # This list MIRRORS common.training.matrix.POLICY_NAMES, which is the one
    # registry the console and tool/train_launch.py read. It is repeated here
    # because this runs on a login node before anything imports our Python, and
    # it drifts the day a policy is added -- so test/unit/test_hpc_manifest.py
    # asserts the two are the same set, which is the only reason a copy is
    # allowed to exist.
    case "$policy" in
        act|diffusion|pi05|fastwam) ;;
        so101_act|so101_diffusion|so101_pi05|so101_flowmatch|so101_dreamzero) ;;
        so101_act_crop|so101_diffusion_crop|so101_pi05_crop) ;;
        *) echo "❌ $MANIFEST:$LINE_NO unknown policy '$policy'" >&2
           echo "   want: act|diffusion|pi05|fastwam, or one of the repo-local" >&2
           echo "   so101_act|so101_diffusion|so101_pi05|so101_flowmatch|so101_dreamzero," >&2
           echo "   or a cropped-tactile variant so101_act_crop|so101_diffusion_crop|so101_pi05_crop" >&2
           exit 2;;
    esac
    # A space here would silently shift every later column into `extra`, so the
    # row would submit and train the wrong thing. Refuse it at the door.
    case "$cameras" in
        ''|*[[:space:]]*) echo "❌ $MANIFEST:$LINE_NO cameras must be 'all' or a comma list with no spaces: '$cameras'" >&2; exit 2;;
    esac
    case "$hours" in
        ''|*[!0-9]*) echo "❌ $MANIFEST:$LINE_NO hours must be a whole number: '$hours'" >&2; exit 2;;
    esac
    # steps and batch are checked for the same reason the cameras column is: a
    # stray space anywhere in the row shifts every later column left, and the
    # row would still submit -- training the wrong size for the wrong time. A
    # non-numeric steps/batch is the first place that shift becomes visible.
    for field in steps:"$steps" batch:"$batch"; do
        case "${field#*:}" in
            -|*[!0-9]*)
                [ "${field#*:}" = "-" ] && continue
                echo "❌ $MANIFEST:$LINE_NO ${field%%:*} must be a whole number or '-': '${field#*:}'" >&2
                echo "   (a space in the cameras column shifts every later column -- check that first)" >&2
                exit 2;;
        esac
    done
    # A space here shifts `extra` along, and unlike the numeric columns there is
    # nothing after it to notice. So the column's SHAPE is checked instead: `-`
    # or comma-joined camera=slot pairs and nothing else, which also refuses the
    # trailing comma a shifted row leaves behind.
    if [ "$slots" != "-" ] && \
       ! [[ "$slots" =~ ^[A-Za-z0-9_]+=[A-Za-z0-9_]+(,[A-Za-z0-9_]+=[A-Za-z0-9_]+)*$ ]]; then
        echo "❌ $MANIFEST:$LINE_NO slots must be '-' or comma-joined camera=slot pairs with no spaces: '$slots'" >&2
        exit 2
    fi
    in_list "$ds" "$FILTER_DATASETS" || continue
    in_list "$policy" "$FILTER_POLICIES" || continue
    # NOT in_list: a cameras value is ITSELF a comma list, so `central,wrist_x`
    # would be read as two alternatives and match both rows. Exact match only.
    # An `if`, not an && chain: this file runs under `set -e`, where a chain
    # whose last evaluated test is false takes the script down with it.
    if [ -n "$FILTER_CAMERAS" ] && [ "$cameras" != "$FILTER_CAMERAS" ]; then
        continue
    fi
    ROWS+=("$ds	$policy	$cameras	${steps:--}	${batch:--}	$hours	${slots:--}	${extra:--}")
done < "$MANIFEST"

if [ "${#ROWS[@]}" -eq 0 ]; then
    echo "❌ no rows selected from $MANIFEST" >&2
    [ -n "$FILTER_DATASETS" ] && echo "   --datasets $FILTER_DATASETS" >&2
    [ -n "$FILTER_POLICIES" ] && echo "   --only $FILTER_POLICIES" >&2
    [ -n "$FILTER_CAMERAS" ] && echo "   --cameras $FILTER_CAMERAS" >&2
    exit 2
fi

# ---- scratch + staged datasets ---------------------------------------
if [ ! -d "$SCRATCH" ]; then
    echo "❌ scratch root not found: $SCRATCH" >&2
    echo "   find yours with:  ls -d /scratch/users/\$USER || ls /scratch" >&2
    echo "   then pass --scratch <dir> (or export SO101_SCRATCH)." >&2
    exit 2
fi
STAGE_DIR="$SCRATCH/hf_lerobot/local"

MISSING=()
for row in "${ROWS[@]}"; do
    ds="${row%%	*}"
    [ -f "$STAGE_DIR/$ds/meta/info.json" ] && continue
    case " ${MISSING[*]-} " in *" $ds "*) ;; *) MISSING+=("$ds");; esac
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "❌ not staged under $STAGE_DIR: ${MISSING[*]}" >&2
    echo "   From the COLLECTION BOX, stage them first:" >&2
    echo "     bash hpc/stage_datasets.sh --dir <collection-dir> \\" >&2
    echo "       --dest ${USER:-<user>}@<create-login-host>:$STAGE_DIR ${MISSING[*]}" >&2
    exit 1
fi

# ---- snapshot what we submit -----------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUB_DIR="$SCRATCH/so101_outputs/submissions/$STAMP"
mkdir -p "$SUB_DIR"

HOURS_SET="$(printf '%s\n' "${ROWS[@]}" | cut -f6 | sort -n -u)"

echo "manifest : $MANIFEST"
echo "scratch  : $SCRATCH"
echo "snapshot : $SUB_DIR"
echo

for hours in $HOURS_SET; do
    group_file="$SUB_DIR/runs_${hours}h.tsv"
    index_file="$SUB_DIR/index_${hours}h.md"
    {
        echo "# dataset	policy	cameras	steps	batch	hours	slots	extra"
        printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v h="$hours" '$6 == h'
    } > "$group_file"
    n="$(awk -F'\t' 'NF && $1 !~ /^#/' "$group_file" | wc -l)"

    {
        echo "# Array index -> run  (${hours} h group, submitted $STAMP)"
        echo
        echo "| array id | dataset | policy | cameras | steps | batch | slots |"
        echo "|---------:|---------|--------|---------|-------|-------|-------|"
        awk -F'\t' 'NF && $1 !~ /^#/ {printf "| %d | %s | %s | %s | %s | %s | %s |\n", NR-2, $1, $2, $3, $4, $5, $7}' "$group_file"
    } > "$index_file"

    range="0-$((n - 1))"
    [ -n "$CONCURRENCY" ] && range="$range%$CONCURRENCY"

    cmd=(sbatch
        --job-name="$JOB_NAME"
        --array="$range"
        --time="${hours}:00:00")
    [ -n "$PARTITION" ] && cmd+=(--partition="$PARTITION")
    [ -n "$ACCOUNT" ] && cmd+=(--account="$ACCOUNT")
    [ -n "$EXCLUDE" ] && cmd+=(--exclude="$EXCLUDE")
    cmd+=(--export="ALL,SO101_REPO_ROOT=$REPO_ROOT,SO101_SCRATCH=$SCRATCH,SO101_MANIFEST=$group_file"
        "$REPO_ROOT/hpc/create_real_vla.sbatch")

    echo "== ${hours} h group — $n run(s), see $index_file"
    awk -F'\t' 'NF && $1 !~ /^#/ {printf "   [%d] %s %s  cameras=%s\n", NR-2, $1, $2, $3}' "$group_file"
    if [ "$DRY_RUN" = "1" ]; then
        printf '   DRY-RUN would submit: '; print_cmd "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
    echo
done

if [ "$DRY_RUN" = "1" ]; then
    echo "✓ dry run — nothing submitted. Drop --dry-run to submit."
else
    echo "✓ submitted. Watch with:  squeue --me"
    echo "  Logs land in the submit dir as ${JOB_NAME}-<arrayjobid>_<taskid>.out;"
    echo "  the index_*.md above says which task is which run."
fi
