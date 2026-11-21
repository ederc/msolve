#!/bin/bash
# usage: bench.sh REF NEW [SET] [extra] -- snapshot the current msolve build as NEW and compare to snapshot REF
# Snapshots land in bench/snap-$NEW (git-ignored); results in bench/$NEW.{json,log}.
# Corpus defaults to $MSCORPUS, else /home/ederc/repos/grobx-corpus/tuning.
set -e
D=$(cd "$(dirname "$0")" && pwd)                 # this bench/ dir
M=$(cd "$D/.." && pwd)                           # repo root
ref=$1; new=$2
SET=${3:-${MSCORPUS:-/home/ederc/repos/grobx-corpus/tuning}}
extra=${4:-}
snap=$D/snap-$new
mkdir -p "$snap"
cp "$M/.libs/msolve" "$snap/msolve.bin"
cp -P "$M"/src/neogb/.libs/libneogb.so* "$snap/"
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s/msolve.bin "$@"\n' "$snap" "$snap" > "$snap/msolve"
chmod +x "$snap/msolve"
cd "$D"
python3 -I run.py "$new.json" "$SET" \
  "$ref=$D/snap-$ref/msolve $extra -g 2 -t 1 -f {in} -o {out}" \
  "$new=$D/snap-$new/msolve $extra -g 2 -t 1 -f {in} -o {out}" \
  --jobs 4 --timeout 120 --repeat 2 --keep > "$new.log" 2>&1
python3 -I cmp.py "$new.json" "$ref" "$new" 6
python3 -I cmpmem.py "$new.json" "$ref" "$new"
python3 -I same.py "$new.json" "$ref" "$new"
