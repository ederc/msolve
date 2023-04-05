/* This file is part of msolve.
 *
 * msolve is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * msolve is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with msolve.  If not, see <https://www.gnu.org/licenses/>
 *
 * Authors:
 * Jérémy Berthomieu
 * Christian Eder
 * Mohab Safey El Din */


#include "engine.h"

int initialize_gba_input_data(
        bs_t **bsp,
        ht_t **htp,
        stat_t **stp,
        /* input values */
        const int32_t *lens,
        const int32_t *exps,
        const void *cfs,
        uint32_t field_char,
        int32_t mon_order,
        int32_t elim_block_len,
        int32_t nr_vars,
        int32_t nr_gens,
        int32_t nr_nf,
        int32_t ht_size,
        int32_t nr_threads,
        int32_t max_nr_pairs,
        int32_t reset_ht,
        int32_t la_option,
        int32_t use_signatures,
        int32_t reduce_gb,
        int32_t pbm_file,
        int32_t info_level
        )
{
    bs_t *bs    = *bsp;
    ht_t *ht    = *htp;
    stat_t *st  = *stp;

    /* initialize stuff */
    st  = initialize_statistics();

    int *invalid_gens   =   NULL;
    int res = validate_input_data(&invalid_gens, cfs, lens, &field_char, &mon_order,
            &elim_block_len, &nr_vars, &nr_gens, &nr_nf, &ht_size, &nr_threads,
            &max_nr_pairs, &reset_ht, &la_option, &use_signatures,
            &reduce_gb, &info_level);

    /* all data is corrupt */
    if (res == -1) {
        free(invalid_gens);
        return res;
    }

    /* checks and set all meta data. if a nonzero value is returned then
     * some of the input data is corrupted. */
    if (check_and_set_meta_data(st, lens, exps, cfs, invalid_gens,
                field_char, mon_order, elim_block_len, nr_vars, nr_gens,
                nr_nf, ht_size, nr_threads, max_nr_pairs, reset_ht, la_option,
                use_signatures, reduce_gb, pbm_file, info_level)) {
        return 0;
    }


    /* initialize basis */
    bs  = initialize_basis(st);
    /* initialize basis hash table */
    ht = initialize_hash_table(st);

    import_input_data(bs, ht, st, lens, exps, cfs, invalid_gens);

    if (st->info_level > 0) {
      print_initial_statistics(stderr, st);
    }

    /* for faster divisibility checks, needs to be done after we have
     * read some input data for applying heuristics */
    calculate_divmask(ht);

    /* sort initial elements, smallest lead term first */
    sort_r(bs->hm, (unsigned long)bs->ld, sizeof(hm_t *),
            initial_input_cmp, ht);
    /* normalize input generators */
    if (st->fc > 0) {
        normalize_initial_basis(bs, st->fc);
    } else {
        if (st->fc == 0) {
            remove_content_of_initial_basis(bs);
        }
    }

    *bsp  = bs;
    *htp  = ht;
    *stp  = st;

    free(invalid_gens);

    return 1;
}

int core_gba(
        bs_t **bsp,
        ht_t **htp,
        stat_t **stp
        )
{
    if ((*stp)->use_signatures == 0) {
        return core_f4(bsp, htp, stp);
    } else {
        return core_sba_schreyer(bsp, htp, stp);
    }
}

int64_t export_results_from_gba(
    /* return values */
    int32_t *bld,   /* basis load */
    int32_t **blen, /* length of each poly in basis */
    int32_t **bexp, /* basis exponent vectors */
    void **bcf,     /* coefficients of basis elements */
    void *(*mallocp) (size_t),
    bs_t **bsp,
    ht_t **bhtp,
    stat_t **stp
    )
{
    if ((*stp)->use_signatures == 0) {
        return export_results_from_f4(bld, blen, bexp, bcf,
                mallocp, bsp, bhtp, stp);
    } else {
        exit(1);
    }
}

bs_t *gba_trace_learning_phase(
        trace_t *trace,           /* trace of the GB Algorithm */
        ht_t * tht,               /* trace hash table for multipliers */
        const bs_t * const ggb,   /* global basis */
        ht_t *gbht,               /* global basis hash table, generated
                                   * in this run, used in upcoming runs */
        stat_t *gst,              /* global statistics */
        const int32_t fc          /* characteristic of field */
        )
{
    if (gst->use_signatures == 0) {
        return f4_trace_learning_phase(trace, tht, ggb, gbht, gst, fc);
    } else {
        exit(1);
    }
}

bs_t *gba_trace_application_phase(
        trace_t *trace,           /* trace of the GB Algorithm */
        ht_t * tht,               /* trace hash table for multipliers */
        const bs_t * const ggb,   /* global basis */
        ht_t *lbht,               /* global basis hash table, generated
                                   * in this run, used in upcoming runs */
        stat_t *gst,              /* global statistics */
        const int32_t fc          /* characteristic of field */
        )
{
    if (gst->use_signatures == 0) {
        return f4_trace_application_phase(trace, tht, ggb, lbht, gst, fc);
    } else {
        exit(1);
    }
}
