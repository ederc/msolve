#!/usr/bin/env bash

# Groebner bases over Q lifted from their images modulo primes by the
# mixed radix digits of the coefficients (--gb-lifting 1) must be the
# same as with the default lifting.

source test/diff/diff_source.sh

for file in kat7-qq cyclic5-qq henrion5-qq bug-2nd-prime-bad; do
    $(pwd)/msolve -f input_files/$file.ms -o test/diff/$file.gbl.res \
          --random-seed $seed \
          --gb-lifting 1 \
          -g 2 -t 1
    if [ $? -gt 0 ]; then
        rm -f test/diff/$file.gbl.res
        print_exit 1
    fi
    diff test/diff/$file.gbl.res output_files/$file.g2.res
    if [ $? -gt 0 ]; then
        rm -f test/diff/$file.gbl.res
        print_exit 2
    fi
    rm test/diff/$file.gbl.res
done

normal_exit
