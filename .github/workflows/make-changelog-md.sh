#!/bin/bash

set -euo pipefail

rst_in="changelog.rst"

# No tag provided -> fallback
if [[ -z "${1:-}" ]]; then
  pcregrep -M "^\^{4,}.*\n[\s\S]*" ${rst_in} \
  | tail -n +2 \
  | pandoc --from=rst --to=markdown --wrap=none
  exit 0
fi

tag=$(echo "$1" | sed  -e 's/\./\\\./g')
pcregrep -M "^${tag}.*\n\^\^\^\^+.*\n(.*\n)+?(\^\^\^\^+|^---+)$" ${rst_in} \
  | tail -n +3 \
  | head -n -2 \
  | pandoc --from=rst --to=markdown --wrap=none
