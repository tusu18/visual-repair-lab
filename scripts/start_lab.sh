#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv_dir=${VISUAL_REPAIR_VENV:-"$project_dir/.venv"}
if [ ! -x "$venv_dir/bin/visual-repair-lab" ]; then
  echo 'Install first: bash scripts/install.sh' >&2
  exit 1
fi
cd "$project_dir"
mkdir -p var
exec "$venv_dir/bin/visual-repair-lab" serve --bank banks/train.json --database var/episodes.db --port "${VISUAL_REPAIR_PORT:-8822}"
