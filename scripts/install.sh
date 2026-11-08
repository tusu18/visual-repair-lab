#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${VISUAL_REPAIR_PYTHON:-python3}
venv_dir=${VISUAL_REPAIR_VENV:-"$project_dir/.venv"}
"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip --disable-pip-version-check --no-cache-dir install -r "$project_dir/requirements.lock" "$project_dir"
"$venv_dir/bin/python" -m pip --disable-pip-version-check --no-cache-dir check
"$venv_dir/bin/visual-repair-lab" --help

