#!/usr/bin/env bash
set -euo pipefail

repo_url="https://github.com/amazon-science/chronos-forecasting.git"
commit="8589d1988e9676817548e9626738ff06b6ca6370"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
source_dir="$project_dir/external/chronos-forecasting"
patch_file="$project_dir/patches/chronos2_forecast_embeds.patch"

if [[ ! -d "$source_dir/.git" ]]; then
    mkdir -p "$project_dir/external"
    git init "$source_dir"
    git -C "$source_dir" remote add origin "$repo_url"
fi

if ! git -C "$source_dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$source_dir" fetch --depth 1 origin "$commit"
fi

current_commit="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || true)"
if [[ "$current_commit" != "$commit" ]]; then
    if [[ -n "$(git -C "$source_dir" status --porcelain 2>/dev/null)" ]]; then
        echo "Chronos checkout has local changes; refusing to overwrite it." >&2
        exit 1
    fi
    git -C "$source_dir" checkout --detach "$commit"
fi

if git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "Chronos forecast-embedding patch is already applied."
elif git -C "$source_dir" apply --check "$patch_file"; then
    git -C "$source_dir" apply "$patch_file"
    echo "Applied Chronos forecast-embedding patch."
else
    echo "Pinned Chronos source does not match the expected patch." >&2
    exit 1
fi

uv pip install --python "$project_dir/.venv/bin/python" --editable "$source_dir"
echo "Chronos is pinned at $commit and installed editable."
