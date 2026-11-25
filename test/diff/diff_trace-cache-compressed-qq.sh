#!/usr/bin/env bash

# The matrices stored in the trace are compressed once the uncompressed
# ones exceed a limit, which only large computations reach. With the
# limit set to zero all of them are compressed, the Groebner bases over Q
# must be the same. The matrices of katsura-8 have enough rows for the
# compressed headers of their rows to grow.

source test/diff/diff_source.sh

for file in kat7-qq cyclic5-qq henrion5-qq bug-2nd-prime-bad; do
    NEOGB_TC_RAW_MAX=0 $(pwd)/msolve -f input_files/$file.ms \
          -o test/diff/$file.tcc.res \
          --random-seed $seed \
          -g 2 -t 1
    if [ $? -gt 0 ]; then
        rm -f test/diff/$file.tcc.res
        print_exit 1
    fi
    diff test/diff/$file.tcc.res output_files/$file.g2.res
    if [ $? -gt 0 ]; then
        rm -f test/diff/$file.tcc.res
        print_exit 2
    fi
    rm test/diff/$file.tcc.res
done

file=kat8-qq-truncate
$(pwd)/msolve -f input_files/$file.ms -o test/diff/$file.tc.res \
      --random-seed $seed -g 2 -t 1
if [ $? -gt 0 ]; then
    rm -f test/diff/$file.tc.res
    print_exit 3
fi
NEOGB_TC_RAW_MAX=0 $(pwd)/msolve -f input_files/$file.ms \
      -o test/diff/$file.tcc.res --random-seed $seed -g 2 -t 1
if [ $? -gt 0 ]; then
    rm -f test/diff/$file.tc.res test/diff/$file.tcc.res
    print_exit 4
fi
diff test/diff/$file.tcc.res test/diff/$file.tc.res
if [ $? -gt 0 ]; then
    rm -f test/diff/$file.tc.res test/diff/$file.tcc.res
    print_exit 5
fi
rm test/diff/$file.tc.res test/diff/$file.tcc.res

normal_exit
