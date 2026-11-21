#!/usr/bin/env bash

# QQ lifting with a learned signature-F4 record: the application phase
# must replay the record at 61-bit primes (see
# docs/superpowers/specs/2026-11-21-61bit-sigf4-tracer-replay-design.md).
# Checks that
# 1. the Groebner basis over Q is unchanged (same expected file as the
#    plain -g 2 run), and
# 2. the diagnostics contain the wide-application marker line
#    "application phase at 61-bit primes".
#
# The record is only learned for --signature-f4 > 0.  We use cyclic5-qq
# with --signature-f4 2: mode 2 forbids the handover of the
# signature-based F4 to F4, so the learned record covers the whole run
# and is replayed at every application prime (see the Task 4 ruling in
# .superpowers/sdd/2026-11-21-61bit-sigf4-replay/progress.md).

file=cyclic5-qq

source test/diff/diff_source.sh

$(pwd)/msolve -f input_files/$file.ms -o test/diff/$file.wide.res \
      --random-seed $seed \
      --signature-f4 2 \
      -g 2 -v 2 -t 1 > test/diff/$file.wide.log 2>&1
if [ $? -gt 0 ]; then
    rm -f test/diff/$file.wide.res test/diff/$file.wide.log
    print_exit 1
fi

diff test/diff/$file.wide.res output_files/$file.g2.res
if [ $? -gt 0 ]; then
    rm -f test/diff/$file.wide.res test/diff/$file.wide.log
    print_exit 2
fi
rm test/diff/$file.wide.res

grep -q "application phase at 61-bit primes" test/diff/$file.wide.log
if [ $? -gt 0 ]; then
    rm -f test/diff/$file.wide.log
    print_exit 3
fi
rm test/diff/$file.wide.log

normal_exit
