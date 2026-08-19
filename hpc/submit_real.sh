#!/usr/bin/env bash
# =====================================================================
# Submit real-data VLA training to KCL CREATE (Slurm) from a run manifest.
#
# Run this on a CREATE *login* node, from the repo. It reads hpc/runs.tsv --
# one row per (dataset, policy) training run -- filters it, checks every
# selected dataset is staged, and submits the rows as Slurm job arrays, one
# array task per row. Nothing in the repo has to be edited to change which
# datasets are trained: edit the manifest, or filter it on the command line.
#
#   bash hpc/submit_real.sh                          # everything in runs.tsv
#   bash hpc/submit_real.sh --datasets cube-pnp      # one dataset
#   bash hpc/submit_real.sh --only act               # one policy
#   bash hpc/submit_real.sh --dry-run                # print the sbatch commands
#
# Options:
#   --manifest F      run matrix to read           (default hpc/runs.tsv)
#   --datasets a,b    only these datasets
#   --only a,b        only these policies          (act, diffusion)
#   --scratch DIR     node-visible scratch root    (default /scratch/users/$USER)
#   --partition P     Slurm partition              (default: the sbatch's own)
#   --account A       Slurm account, if enforced
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
    read -r ds policy steps batch hours extra <<<"$line"
    if [ -z "${hours:-}" ]; then
        echo "❌ $MANIFEST:$LINE_NO needs 5 fields (dataset policy steps batch hours): $line" >&2
        exit 2
    fi
    case "$policy" in
        act|diffusion) ;;
        *) echo "❌ $MANIFEST:$LINE_NO unknown policy '$policy' (want act|diffusion)" >&2; exit 2;;
    esac
    case "$hours" in
        ''|*[!0-9]*) echo "❌ $MANIFEST:$LINE_NO hours must be a whole number: '$hours'" >&2; exit 2;;
    esac
    in_list "$ds" "$FILTER_DATASETS" || continue
    in_list "$policy" "$FILTER_POLICIES" || continue
    ROWS+=("$ds	$policy	${steps:--}	${batch:--}	$hours	${extra:--}")
done < "$MANIFEST"

if [ "${#ROWS[@]}" -eq 0 ]; then
    echo "❌ no rows selected from $MANIFEST" >&2
    [ -n "$FILTER_DATASETS" ] && echo "   --datasets $FILTER_DATASETS" >&2
    [ -n "$FILTER_POLICIES" ] && echo "   --only $FILTER_POLICIES" >&2
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

HOURS_SET="$(printf '%s\n' "${ROWS[@]}" | cut -f5 | sort -n -u)"

echo "manifest : $MANIFEST"
echo "scratch  : $SCRATCH"
echo "snapshot : $SUB_DIR"
echo

for hours in $HOURS_SET; do
    group_file="$SUB_DIR/runs_${hours}h.tsv"
    index_file="$SUB_DIR/index_${hours}h.md"
    {
        echo "# dataset	policy	steps	batch	hours	extra"
        printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v h="$hours" '$5 == h'
    } > "$group_file"
    n="$(awk -F'\t' 'NF && $1 !~ /^#/' "$group_file" | wc -l)"

    {
        echo "# Array index -> run  (${hours} h group, submitted $STAMP)"
        echo
        echo "| array id | dataset | policy | steps | batch |"
        echo "|---------:|---------|--------|-------|-------|"
        awk -F'\t' 'NF && $1 !~ /^#/ {printf "| %d | %s | %s | %s | %s |\n", NR-2, $1, $2, $3, $4}' "$group_file"
    } > "$index_file"

    range="0-$((n - 1))"
    [ -n "$CONCURRENCY" ] && range="$range%$CONCURRENCY"

    cmd=(sbatch
        --job-name="$JOB_NAME"
        --array="$range"
        --time="${hours}:00:00")
    [ -n "$PARTITION" ] && cmd+=(--partition="$PARTITION")
    [ -n "$ACCOUNT" ] && cmd+=(--account="$ACCOUNT")
    cmd+=(--export="ALL,SO101_REPO_ROOT=$REPO_ROOT,SO101_SCRATCH=$SCRATCH,SO101_MANIFEST=$group_file"
        "$REPO_ROOT/hpc/create_real_vla.sbatch")

    echo "== ${hours} h group — $n run(s), see $index_file"
    awk -F'\t' 'NF && $1 !~ /^#/ {printf "   [%d] %s %s\n", NR-2, $1, $2}' "$group_file"
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
