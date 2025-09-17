#!/bin/bash

set -euo pipefail

rst_in="changelog.rst"

# Fallback: output everything after the first '^^^^' underline
fallback() {
  awk '
    found { print; next }
    /^[[:space:]]*\^{4,}[[:space:]]*$/ { found=1; next }
  ' "'"$rst_in"'" | pandoc --from=rst --to=markdown --wrap=none
}

# No tag provided -> fallback
if [[ -z "${1:-}" ]]; then
  fallback
  exit 0
fi

tag=$(echo "$1" | sed  -e 's/\./\\\./g')
pcregrep -M "^${tag}.*\n\^\^\^\^+.*\n(.*\n)+?(\^\^\^\^+|^---+)$" ${rst_in} \
  | tail -n +3 \
  | head -n -2 \
  | pandoc --from=rst --to=markdown --wrap=none
