#!/usr/bin/env bash

# With several threads the traces, of F4 and of the signature-based F4,
# are applied at several primes in parallel and the polynomials are
# lifted in parallel, the Groebner bases over Q must be the same.

source test/diff/diff_source.sh

for sig in -1 1; do
    for file in kat7-qq cyclic5-qq henrion5-qq bug-2nd-prime-bad; do
        $(pwd)/msolve -f input_files/$file.ms \
              -o test/diff/$file.thr.res \
              --random-seed $seed \
              --signature-f4 $sig \
              -g 2 -t 4
        if [ $? -gt 0 ]; then
            rm -f test/diff/$file.thr.res
            print_exit 1
        fi
        diff test/diff/$file.thr.res output_files/$file.g2.res
        if [ $? -gt 0 ]; then
            rm -f test/diff/$file.thr.res
            print_exit 2
        fi
        rm test/diff/$file.thr.res
    done
done

normal_exit
