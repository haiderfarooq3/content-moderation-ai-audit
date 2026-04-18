#!/usr/bin/env zsh
# Run parts 2, 3, 4, 5 sequentially, each as jupyter nbconvert --execute, logging to a file per part.
set -eu
cd "$(dirname "$0")/.."

LOG_DIR="outputs/exec_logs"
mkdir -p "$LOG_DIR"

run_one() {
    local part="$1"
    local log="$LOG_DIR/$part.log"
    echo "=== $(date '+%H:%M:%S')  executing $part.ipynb ==="  | tee -a "$log"
    TOKENIZERS_PARALLELISM=false .venv/bin/jupyter nbconvert \
        --to notebook --execute "$part.ipynb" --output "$part.ipynb" \
        --ExecutePreprocessor.timeout=7200 \
        --ExecutePreprocessor.kernel_name=jigsaw-py312 \
        >> "$log" 2>&1
    echo "=== $(date '+%H:%M:%S')  $part.ipynb OK ==="        | tee -a "$log"
}

for p in part2 part3 part4 part5; do
    run_one "$p"
done
echo "ALL PARTS DONE $(date '+%H:%M:%S')"
