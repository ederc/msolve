#!/usr/bin/env bash

# The traces, of F4 and of the signature-based F4, are applied at eight
# primes at once where this pays off, which the small inputs of the tests
# never reach. With NEOGB_MULTI_FORCE set they are applied this way
# wherever possible, the Groebner bases over Q must be the same.

source test/diff/diff_source.sh

for sig in -1 2; do
    for file in kat7-qq cyclic5-qq henrion5-qq bug-2nd-prime-bad; do
        NEOGB_MULTI_FORCE=1 $(pwd)/msolve -f input_files/$file.ms \
              -o test/diff/$file.mp.res \
              --random-seed $seed \
              --signature-f4 $sig \
              -g 2 -t 1
        if [ $? -gt 0 ]; then
            rm -f test/diff/$file.mp.res
            print_exit 1
        fi
        diff test/diff/$file.mp.res output_files/$file.g2.res
        if [ $? -gt 0 ]; then
            rm -f test/diff/$file.mp.res
            print_exit 2
        fi
        rm test/diff/$file.mp.res
    done
done

normal_exit
