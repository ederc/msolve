# the reduced Groebner basis by the signature-based F4 (option
# --signature-f4 1), which is only run on request over GF(16) resp.
# GF(256), with one and with two threads
for t in 1 2; do
    $(pwd)/msolve -f input_files/$file.ms -o test/diff/$file.8$t.res \
          --random-seed $seed \
          -g 2 -l 2 -t $t --signature-f4 1
    if [ $? -gt 0 ]; then
        print_exit $((80+2*t-1))
    fi

    diff test/diff/$file.8$t.res output_files/$file.g2.res
    if [ $? -gt 0 ]; then
        print_exit $((80+2*t))
    fi

    rm test/diff/$file.8$t.res
done
