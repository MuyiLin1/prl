#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Assemble a self-contained bundle of the Sokoban DSL+HC labelling code that
# can be scp'd to the cluster and dropped into the repo. Run from the repo root:
#
#     bash scripts/sokoban/cluster/package_bundle.sh
#
# Produces dist/sokoban_hc_bundle/ containing:
#   tree/        -> repo-relative files to rsync into the cluster repo
#   README.md    -> install + run instructions
#   run_sokoban_hc.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

BUNDLE="dist/sokoban_hc_bundle"
TREE="${BUNDLE}/tree"
rm -rf "${BUNDLE}"
mkdir -p "${TREE}"

FILES=(
    "prog_policies/sokoban/__init__.py"
    "prog_policies/sokoban/environment.py"
    "prog_policies/sokoban/dsl.py"
    "prog_policies/sokoban_tasks/__init__.py"
    "prog_policies/sokoban_tasks/boxoban.py"
    "prog_policies/utils/__init__.py"
    "scripts/sokoban/hc_label.py"
    "scripts/sokoban/sokoban_core.py"
    "scripts/sokoban/cluster/run_sokoban_hc.sh"
    "data/sokoban_fastpath_labels.csv"
)

for f in "${FILES[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: missing ${f}" >&2
        exit 1
    fi
    mkdir -p "${TREE}/$(dirname "${f}")"
    cp "${f}" "${TREE}/${f}"
done

cp scripts/sokoban/cluster/README.md "${BUNDLE}/README.md"
cp scripts/sokoban/cluster/run_sokoban_hc.sh "${BUNDLE}/run_sokoban_hc.sh"

echo "Bundle assembled at ${BUNDLE}"
echo "Files:"
( cd "${BUNDLE}" && find . -type f | sort | sed 's/^/  /' )
echo
echo "Next: scp -r ${BUNDLE} <cluster>:~/  then follow ${BUNDLE}/README.md"
