#!/usr/bin/env bash

# Groebner bases over Q whose traces are learned guided by a signature-
# based run (--signature-f4 3) must be the same as with traces learned by
# F4 alone.

source test/diff/diff_source.sh

for file in kat7-qq cyclic5-qq henrion5-qq bug-2nd-prime-bad; do
    $(pwd)/msolve -f input_files/$file.ms -o test/diff/$file.gdl.res \
          --random-seed $seed \
          --signature-f4 3 \
          -g 2 -t 1
    if [ $? -gt 0 ]; then
        rm -f test/diff/$file.gdl.res
        print_exit 1
    fi
    diff test/diff/$file.gdl.res output_files/$file.g2.res
    if [ $? -gt 0 ]; then
        rm -f test/diff/$file.gdl.res
        print_exit 2
    fi
    rm test/diff/$file.gdl.res
done

normal_exit
