#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$root_dir"

required_dirs=(
  "backend"
  "frontend"
  "configs"
  "datasets"
  "metrics"
  "templates"
  "scripts"
  "docs"
)

required_files=(
  "README.md"
  "LICENSE"
  ".gitignore"
)

fail=false

echo "Checking required directories..."
for d in "${required_dirs[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "Missing directory: $d" >&2
    fail=true
  else
    echo "OK: $d" >&2
  fi
done

echo "Checking required files..."
for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing file: $f" >&2
    fail=true
  else
    echo "OK: $f" >&2
  fi
done

if [[ "$fail" == true ]]; then
  echo "Project structure test: FAIL" >&2
  exit 1
fi

echo "Project structure test: PASS"
exit 0
