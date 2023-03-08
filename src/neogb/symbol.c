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


#include "data.h"

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/* select_all_pairs() is unused at the moment */
#if 0 
static void select_all_spairs(
        mat_t *mat,
        const bs_t * const bs,
        ps_t *psl,
        stat_t *st,
        ht_t *sht,
        ht_t *bht,
        ht_t *tht
        )
{
    len_t i, j, k, l, nps, npd, nrr = 0, ntr = 0;
    hm_t *b;
    len_t load  = 0;
    hi_t lcm;
    len_t *gens;
    exp_t *elcm, *eb;
    exp_t *etmp = bht->ev[0];

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    spair_t *ps     = psl->p;
    const len_t nv  = bht->nv;

    /* sort pair set */
    sort_r(ps, (unsigned long)psl->ld, sizeof(spair_t), spair_degree_cmp, bht);

    /* select pairs of this degree respecting maximal selection size mnsel */
    npd  = psl->ld;
    sort_r(ps, (unsigned long)npd, sizeof(spair_t), spair_cmp, bht);
    /* now do maximal selection if it applies */

    nps = psl->ld;
    
    if (st->info_level > 1) {
        printf("%3d  %6d %7d", 0, nps, psl->ld);
        fflush(stdout);
    }
    /* statistics */
    st->num_pairsred  +=  nps;
    /* list for generators */
    gens  = (len_t *)malloc(2 * (unsigned long)nps * sizeof(len_t));
    /* preset matrix meta data */
    mat->rr       = (hm_t **)malloc(2 * (unsigned long)nps * sizeof(hm_t *));
    hm_t **rrows  = mat->rr;
    mat->tr       = (hm_t **)malloc(2 * (unsigned long)nps * sizeof(hm_t *));
    hm_t **trows  = mat->tr;
    mat->sz = 2 * nps;
    mat->nc = mat->ncl = mat->ncr = 0;
    mat->nr = 0;

    int ctr = 0;


    i = 0;

    while (i < nps) {
        /* ncols initially counts number of different lcms */
        mat->nc++;
        load  = 0;
        lcm   = ps[i].lcm;
        j = i;

        while (j < nps && ps[j].lcm == lcm) {
            gens[load++] = ps[j].gen1;
            gens[load++] = ps[j].gen2;
            ++j;
        }
        /* sort gens set */
        qsort(gens, (unsigned long)load, sizeof(len_t), gens_cmp);

        len_t prev  = -1;

        /* first element with given lcm goes into reducer part of matrix,
         * all remaining ones go to to be reduced part */
        prev  = gens[0];
        /* printf("prev %u / %u\n", prev, bs->ld); */
        /* ev might change when enlarging the hash table during insertion of a new
            * row in the matrix, thus we have to reset elcm inside the for loop */
        elcm  = bht->ev[lcm];
        b     = bs->hm[prev];
        eb    = bht->ev[b[OFFSET]];
        for (l = 0; l <= nv; ++l) {
            etmp[l]   =   (exp_t)(elcm[l] - eb[l]);
        }
        const hi_t h    = bht->hd[lcm].val - bht->hd[b[OFFSET]].val;
        /* note that we use index mat->nc and not mat->nr since for each new
         * lcm we add exactly one row to mat->rr */
        rrows[nrr]  = multiplied_poly_to_matrix_row(sht, bht, h, etmp, b);
        /* track trace information ? */
        if (tht != NULL) { 
           rrows[nrr][BINDEX]  = prev;
            if (tht->eld == tht->esz-1) {
                enlarge_hash_table(tht);
            }
            rrows[nrr][MULT]    = insert_in_hash_table(etmp, tht);
        }

        /* mark lcm column as lead term column */
        sht->hd[rrows[nrr++][OFFSET]].idx = 2; 
        /* still we have to increase the number of rows */
        mat->nr++;
        for (k = 1; k < load; ++k) {
            /* check sorted list for doubles */
            if (gens[k] ==  prev) {
                continue;
            }
            prev  = gens[k];
            /* ev might change when enlarging the hash table during insertion of a new
             * row in the matrix, thus we have to reset elcm inside the for loop */
            elcm  = bht->ev[lcm];
            if (elcm[0] > 0) {
                /* printf("pair with lcm ");
                 * for (int ii = 0; ii < nv; ++ii) {
                 *     printf("%u ", elcm[ii]);
                 * }
                 * printf("\n"); */
            }
            b     = bs->hm[prev];
            eb    = bht->ev[b[OFFSET]];
            for (l = 0; l <= nv; ++l) {
                etmp[l]   =   (exp_t)(elcm[l] - eb[l]);
            }
            const hi_t h  = bht->hd[lcm].val - bht->hd[b[OFFSET]].val;
            trows[ntr] = multiplied_poly_to_matrix_row(sht, bht, h, etmp, b);
            /* track trace information ? */
            if (tht != NULL) {
                trows[ntr][BINDEX]  = prev;
                if (tht->eld == tht->esz-1) {
                    enlarge_hash_table(tht);
                }
                trows[ntr][MULT]    = insert_in_hash_table(etmp, tht);
            }
            /* mark lcm column as lead term column */
            sht->hd[trows[ntr++][OFFSET]].idx = 2;
            mat->nr++;
        }
        ctr++;
        i = j;
    }
    /* printf("nc %u | nr %u || %u\n", mat->nc, mat->nr, sht->eld); */
    /* printf("%u pairs in degree %u\n", ctr, md); */
    /* clear ht-ev[0] */
    memset(bht->ev[0], 0, (unsigned long)nv * sizeof(exp_t));
    /* fix rows to be reduced */
    mat->tr = realloc(mat->tr, (unsigned long)(mat->nr - mat->nc) * sizeof(hm_t *));

    st->num_rowsred +=  mat->nr - mat->nc;
    st->current_deg =   etmp[DEG];

    free(gens);

    /* remove selected spairs from pairset */
    memmove(ps, ps+nps, (unsigned long)(psl->ld-nps) * sizeof(spair_t));
    psl->ld -=  nps;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->select_ctime  +=  ct1 - ct0;
    st->select_rtime  +=  rt1 - rt0;
}
#endif

/* selection of spairs, at the moment only selection
by minial degree of the spairs is supported */
static void select_spairs(
        mat_t *mat,
        ht_t *ht,
        ps_t *ps,
        const bs_t * const bs,
        stat_t *st
        )
{
    len_t i, j, k, l;
    hm_t *b;
    len_t load = 0;
    hi_t lcm;
    len_t *gens;
    exp_t *elcm, *eb;
    exp_t etmp[ht->evl];

    /* timings */
    double ct = cputime();
    double rt = realtime();

    /* get pairs */
    spair_t *p = ps->p;

    /* get real exponent vector length, see data.h for more information */
    const len_t evl = ht->evl;

    const len_t pld = ps->ld;

    /* sort pair set */
    sort_r(p, (unsigned long)pld, sizeof(spair_t), spair_cmp, ht);

    /* get minimal degree */
    const deg_t md = p[0].deg;

    /* get index i of last spair of lowest degree */
    for (i = 0; i < pld; ++i) {
        if (p[i].deg > md) {
            break;
        }
    }

    /* check against maximal selection option, but get
       all of spairs of the same last lcm in the matrix */
    if (i > st->mnsel) {
        i   = st->mnsel;
        lcm = p[i].lcm;
        while (i+1 < pld && p[i+1].lcm == lcm) {
            i++;
        }
    }
    /* statistics */
    if (st->info_level > 1) {
        printf("%3d  %6d %7d", md, i, pld);
        fflush(stdout);
    }
    st->num_pairsred += i;

    const len_t nps = i;

    /* list for generators */
    gens  = (len_t *)malloc(2 * (unsigned long)nps * sizeof(len_t));

    /* preset matrix meta data, tuples of (multiplier, basis index) */
    mat->rrd  = (len_t *)malloc(4 * (unsigned long)nps * sizeof(len_t));
    mat->trd  = (len_t *)malloc(4 * (unsigned long)nps * sizeof(len_t));
    hm_t *rrd = mat->rrd;
    hm_t *trd = mat->trd;
    mat->sz   = 2 * nps;
    mat->nc   = mat->ncl = mat->ncr = 0;
    mat->nr   = 0;

    len_t rpos = 0, tpos = 0;

    /* reset local hashes load */
    ht->lhld = 0;

    i = 0;

    while (i < nps) {
        /* ncols initially counts number of different lcms */
        mat->nc++;
        load  = 0;
        lcm = p[i].lcm;
        j = i;

        while (j < nps && p[j].lcm == lcm) {
            gens[load++] = p[j].gen1;
            gens[load++] = p[j].gen2;
            ++j;
        }
        /* sort gens set */
        qsort(gens, (unsigned long)load, sizeof(len_t), gens_cmp);

        len_t prev  = -1;

        /* first element with given lcm goes into reducer part of matrix,
         * all remaining ones go to to be reduced part */
        prev  = gens[0];
        /* ev might change when enlarging the hash table during insertion of a new
         * row in the matrix, thus we have to reset elcm inside the for loop */
        elcm  = ht->ev[lcm];
        b     = bs->hm[prev];
        eb    = ht->ev[b[OFFSET]];
        for (l = 0; l < evl; ++l) {
            etmp[l]   =   (exp_t)(elcm[l] - eb[l]);
        }
        const hi_t h    = ht->hd[lcm].val - ht->hd[b[OFFSET]].val;
        /* note that we use index mat->nc and not mat->nr since for each new
         * lcm we add exactly one row to mat->rr */
        multiplied_poly_to_hash_table(ht, h, etmp, b);
#if PARALLEL_HASHING
        hm_t mul = check_insert_in_hash_table(etmp, h, ht);
#else
        hm_t mul = insert_in_hash_table(etmp, ht);
#endif
        rrd[rpos++] = mul;
        rrd[rpos++] = prev;
        mat->nr++;

        for (k = 1; k < load; ++k) {
            /* check sorted list for doubles */
            if (gens[k] ==  prev) {
                continue;
            }
            prev  = gens[k];
            /* ev might change when enlarging the hash table during insertion of a new
             * row in the matrix, thus we have to reset elcm inside the for loop */
            elcm  = ht->ev[lcm];
            if (elcm[0] > 0) {
            }
            b     = bs->hm[prev];
            eb    = ht->ev[b[OFFSET]];
            for (l = 0; l < evl; ++l) {
                etmp[l]   =   (exp_t)(elcm[l] - eb[l]);
            }
            const hi_t h  = ht->hd[lcm].val - ht->hd[b[OFFSET]].val;
            multiplied_poly_to_hash_table(ht, h, etmp, b);
#if PARALLEL_HASHING
            hm_t mul = check_insert_in_hash_table(etmp, h, ht);
#else
            hm_t mul = insert_in_hash_table(etmp, ht);
#endif
            trd[tpos++] = mul;
            trd[tpos++] = prev;
            mat->nr++;
        }
        i = j;
    }

    /* memory stuff */
    mat->nrl = mat->nr - mat->nc;
    mat->nru = mat->nc;
    trd = realloc(trd, (unsigned long)2*mat->nrl * sizeof(hm_t));

    mat->trd = trd;
    memset(ht->ev[0], 0, (unsigned long)evl * sizeof(exp_t));
    /* fix rows to be reduced */

    free(gens);

    memmove(p, p+nps, (unsigned long)(ps->ld-nps) * sizeof(spair_t));
    ps->p  =  p;
    ps->ld -= nps;

    /* statistics */
    st->num_rowsred +=  mat->nrl;
    st->current_deg =   md;

    /* timings */
    st->select_ctime += cputime() - ct;
    st->select_rtime += realtime() - rt;
}

static void select_spairs_by_minimal_degree(
        mat_t *mat,
        const bs_t * const bs,
        ps_t *psl,
        stat_t *st,
        ht_t *sht,
        ht_t *bht,
        ht_t *tht
        )
{
    len_t i, j, k, l, md, nps, npd, nrr = 0, ntr = 0;
    hm_t *b;
    len_t load = 0;
    hi_t lcm;
    len_t *gens;
    exp_t *elcm, *eb;
    exp_t *etmp = bht->ev[0];

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    spair_t *ps     = psl->p;
    const len_t evl = bht->evl;

    /* sort pair set */
    sort_r(ps, (unsigned long)psl->ld, sizeof(spair_t), spair_cmp, bht);
    /* get minimal degree */
    md  = ps[0].deg;

    /* select pairs of this degree respecting maximal selection size mnsel */
#if 0
    printf("pair set sorted for symbolic preprocessing\n");
    int pctr = 0;
    deg_t degtest = 0;
    for (i = 0; i < psl->ld; ++i) {
        if (degtest != ps[i].deg) {
            printf("%d elements\n", pctr);
            printf("-- degree %d --\n", ps[i].deg);
            degtest = ps[i].deg;
            pctr = 0;
        }
        printf("%d --> deg %d --> [%u,%u]", i, ps[i].deg, ps[i].gen1, ps[i].gen2);
        for (int jj = 0; jj < evl; ++jj) {
            printf("%d ", bht->ev[ps[i].lcm][jj]);
        }
        pctr++;
        printf("\n");
    }
    printf("\n");
#endif
    for (i = 0; i < psl->ld; ++i) {
        if (ps[i].deg > md) {
            break;
        }
    }
    npd  = i;
    /* printf("npd %d\n", npd); */
    /* sort_r(ps, (unsigned long)npd, sizeof(spair_t), spair_cmp, bht); */
    /* now do maximal selection if it applies */
    
    /* if we stopped due to maximal selection size we still get the following
     * pairs of the same lcm in this matrix */
    if (npd > st->mnsel) {
        nps = st->mnsel;
        lcm = ps[nps].lcm;
        while (nps < npd && ps[nps+1].lcm == lcm) {
            nps++;
        }
    } else {
        nps = npd;
    }
    if (st->info_level > 1) {
        printf("%3d  %6d %7d", md, nps, psl->ld);
        fflush(stdout);
    }
    /* statistics */
    st->num_pairsred  +=  nps;
    /* list for generators */
    gens  = (len_t *)malloc(2 * (unsigned long)nps * sizeof(len_t));
    /* preset matrix meta data */
    mat->rr       = (hm_t **)malloc(2 * (unsigned long)nps * sizeof(hm_t *));
    hm_t **rrows  = mat->rr;
    mat->tr       = (hm_t **)malloc(2 * (unsigned long)nps * sizeof(hm_t *));
    hm_t **trows  = mat->tr;
    mat->sz = 2 * nps;
    mat->nc = mat->ncl = mat->ncr = 0;
    mat->nr = 0;

    int ctr = 0;

    i = 0;

    while (i < nps) {
        /* ncols initially counts number of different lcms */
        mat->nc++;
        load  = 0;
        lcm   = ps[i].lcm;
        j = i;

        while (j < nps && ps[j].lcm == lcm) {
            gens[load++] = ps[j].gen1;
            gens[load++] = ps[j].gen2;
            ++j;
        }
        /* sort gens set */
        qsort(gens, (unsigned long)load, sizeof(len_t), gens_cmp);

        len_t prev  = -1;

        /* first element with given lcm goes into reducer part of matrix,
         * all remaining ones go to to be reduced part */
        prev  = gens[0];
        /* ev might change when enlarging the hash table during insertion of a new
            * row in the matrix, thus we have to reset elcm inside the for loop */
        elcm  = bht->ev[lcm];
        b     = bs->hm[prev];
        eb    = bht->ev[b[OFFSET]];
        for (l = 0; l < evl; ++l) {
            etmp[l]   =   (exp_t)(elcm[l] - eb[l]);
        }
        const hi_t h    = bht->hd[lcm].val - bht->hd[b[OFFSET]].val;
        /* note that we use index mat->nc and not mat->nr since for each new
         * lcm we add exactly one row to mat->rr */
        /* hm_t lm = multiplied_poly_to_hash_table(sht, h, etmp, b); */
/* #if PARALLEL_HASHING
        hm_t mul = check_insert_in_hash_table(etmp, h, sht);
#else
        hm_t mul = insert_in_hash_table(etmp, sht);
#endif */
        /* track trace information ? */
        if (tht != NULL) { 
           rrows[nrr][BINDEX]  = prev;
            if (tht->eld == tht->esz-1) {
                enlarge_hash_table(tht);
            }
#if PARALLEL_HASHING
            rrows[nrr][MULT]    = check_insert_in_hash_table(etmp, h, tht);
#else
            rrows[nrr][MULT]    = insert_in_hash_table(etmp, tht);
#endif
        }

        /* mark lcm column as lead term column */
        /* sht->idx[lm] = 2;  */
        /* still we have to increase the number of rows */
        mat->nr++;
        for (k = 1; k < load; ++k) {
            /* check sorted list for doubles */
            if (gens[k] ==  prev) {
                continue;
            }
            prev  = gens[k];
            /* ev might change when enlarging the hash table during insertion of a new
             * row in the matrix, thus we have to reset elcm inside the for loop */
            elcm  = bht->ev[lcm];
            if (elcm[0] > 0) {
            }
            b     = bs->hm[prev];
            eb    = bht->ev[b[OFFSET]];
            for (l = 0; l < evl; ++l) {
                etmp[l]   =   (exp_t)(elcm[l] - eb[l]);
            }
            const hi_t h  = bht->hd[lcm].val - bht->hd[b[OFFSET]].val;
			/* hm_t lm = multiplied_poly_to_hash_table(sht, bht, h, etmp, b); */
/* #if PARALLEL_HASHING
			hm_t mul = check_insert_in_hash_table(etmp, h, sht);
#else
			hm_t mul = insert_in_hash_table(etmp, sht);
#endif */
            /* track trace information ? */
            if (tht != NULL) {
                trows[ntr][BINDEX]  = prev;
                if (tht->eld == tht->esz-1) {
                    enlarge_hash_table(tht);
                }
#if PARALLEL_HASHING
                trows[ntr][MULT]    = check_insert_in_hash_table(etmp, h, tht);
#else
                trows[ntr][MULT]    = insert_in_hash_table(etmp, tht);
#endif
            }

            mat->nr++;
        }
        ctr++;
        i = j;
    }
    /* printf("%u pairs in degree %u\n", ctr, md); */
    /* clear ht-ev[0] */
    memset(bht->ev[0], 0, (unsigned long)evl * sizeof(exp_t));
    /* fix rows to be reduced */
    /* mat->tr = realloc(mat->tr, (unsigned long)(mat->nr - mat->nc) * sizeof(hm_t *)); */

    st->num_rowsred +=  mat->nr - mat->nc;
    st->current_deg =   md;

    free(gens);

    /* remove selected spairs from pairset */
    memmove(ps, ps+nps, (unsigned long)(psl->ld-nps) * sizeof(spair_t));
    psl->ld -=  nps;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->select_ctime  +=  ct1 - ct0;
    st->select_rtime  +=  rt1 - rt0;
}

/* write elements straight to sat, not to a matrix */
static void select_saturation(
        bs_t *sat,
        mat_t *mat,
        stat_t *st,
        ht_t *sht,
        ht_t *bht
        )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();


    /* preset matrix meta data */
    mat->rr = (hm_t **)malloc(100 * sizeof(hm_t *));
    mat->tr = NULL;

    mat->sz = 100;
    mat->nc = mat->ncl = mat->ncr = 0;
    mat->nr = 0;

    /* for (i=0; i < sat->hm[0][LENGTH]; ++i) {
     *     printf("%u | ", sat->cf_32[sat->hm[0][COEFFS]][i]);
     *     for (len_t j = 0; j < bht->nv; ++j) {
     *         printf("%u ", bht->ev[sat->hm[0][OFFSET+i]][j]);
     *     }
     *     printf(" ||| ");
     * }
     * printf("\n"); */
    /* move hashes of sat data from bht to sht for linear algebra */
    /* for (i = 0; i < sat->ld; ++i) {
     *     for (j = OFFSET; j < sat->hm[i][LENGTH]+OFFSET; ++j) {
     *         sat->hm[i][j] = insert_in_hash_table(
     *                 bht->ev[sat->hm[i][j]], sht);
     *     }
     * } */

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->select_ctime  +=  ct1 - ct0;
    st->select_rtime  +=  rt1 - rt0;
}


static void select_tbr(
        const bs_t * const tbr,
        const exp_t * const mul,
        const len_t start,
        mat_t *mat,
        stat_t *st,
        ht_t *sht,
        ht_t *bht,
        ht_t *tht
        )
{
    len_t i;

    len_t ntr = 0;

    /* preset matrix meta data */
    mat->rr       = (hm_t **)malloc(100 * sizeof(hm_t *));
    mat->tr       = (hm_t **)malloc((unsigned long)tbr->ld * sizeof(hm_t *));
    hm_t **trows  = mat->tr;

    mat->sz = 100;
    mat->nc = mat->ncl = mat->ncr = 0;
    mat->nr = 0;

    /* always take all elements in tbr and
     * multiply them by the given multiple */
    for (i = start; i < tbr->ld; ++i) {
        const hm_t *b   = tbr->hm[i];
        /* remove the multiplier business for the moment, no need
         * and it corrupts a bit the sht size for efficient matrix
         * generation */
        /* const hi_t mulh = insert_in_hash_table(mul, sht);
         * const hi_t h    = sht->hd[mulh].val;
         * const deg_t d   = sht->hd[mulh].deg; */
        const hi_t h    = 0;
        trows[ntr++]    = multiplied_poly_to_matrix_row(
                sht, bht, h, mul, b);
        mat->nr++;
    }
}


static inline void find_multiplied_reducer_data(
        const len_t idx,
        ht_t *ht,
        mat_t *mat,
        const bs_t * const bs
        )
{
    len_t i, k;

    bi_t dp; /* divisor polynomial basis index */
    hm_t *b  = NULL;
    exp_t *f = NULL;

    const hm_t m = ht->lh[idx];

    /* printf("searching reducer for ");
    for (int j = 0; j < ht->evl; ++j) {
        printf("%d ", ht->ev[m][j]);
    }
    printf("\n"); */

    const len_t evl = ht->evl;

    const exp_t * const e = ht->ev[m];

    const hd_t hdm    = ht->hd[m];
    const len_t lml   = bs->lml;
    const sdm_t ns    = ~hdm.sdm;

    const sdm_t * const lms = bs->lm;
    const bl_t * const lmps = bs->lmps;

    exp_t *etmp = ht->ev[0];
    const hd_t * const hdb  = ht->hd;
    exp_t * const * const evb = ht->ev;

    dp = ht->div[m];
    i  = 0;
    if (dp > 0 && bs->red[dp] == 0) {
        b = bs->hm[dp];
        f = evb[b[OFFSET]];
        for (k=0; k < evl; ++k) {
            etmp[k] = (exp_t)(e[k]-f[k]);
        }
    } else {
start:
        while (i < lml && lms[i] & ns) {
            i++;
        }
        if (i < lml) {
            b = bs->hm[lmps[i]];
            f = evb[b[OFFSET]];
            for (k=0; k < evl; ++k) {
                etmp[k] = (exp_t)(e[k]-f[k]);
                if (etmp[k] < 0) {
                    i++;
                    goto start;
                }
            }
            dp = lmps[i];
        }
    }
    if (i < lml) {
        const hi_t h  = hdm.val - hdb[b[OFFSET]].val;
#if PARALLEL_HASHING
        hm_t mul = check_insert_in_hash_table(etmp, h, ht);
#else
        hm_t mul = insert_in_hash_table(etmp, ht);
#endif
        multiplied_poly_to_hash_table(ht, h, etmp, b);
        mat->rrd[2*mat->nru]   = mul;
        mat->rrd[2*mat->nru+1] = dp;
        mat->nru++;
        ht->div[m] = dp;
    }
}

static inline void find_multiplied_reducer_no_row(
        const bs_t * const bs,
        const hm_t m,
        const ht_t * const bht,
        len_t *nr,
        hm_t *pre_rr,
        ht_t *sht,
        ht_t *tht
        )
{
    len_t i, k;

    const len_t rr  = *nr;

    const len_t evl = bht->evl;

    const exp_t * const e  = sht->ev[m];

    const hd_t hdm    = sht->hd[m];
    const len_t lml   = bs->lml;
    const sdm_t ns    = ~hdm.sdm;

    const sdm_t * const lms = bs->lm;
    const bl_t * const lmps = bs->lmps;

    exp_t *etmp = bht->ev[0];
    exp_t * const * const evb = bht->ev;

    i = 0;
start:
    while (i < lml && lms[i] & ns) {
        i++;
    }
    if (i < lml) {
        const hm_t *b = bs->hm[lmps[i]];
        const exp_t * const f = evb[b[OFFSET]];
        for (k=0; k < evl; ++k) {
            etmp[k] = (exp_t)(e[k]-f[k]);
            if (etmp[k] < 0) {
                i++;
                goto start;
            }
        }
		/* hm_t prev = lmps[i];
        const hi_t h  = hdm.val - hdb[b[OFFSET]].val; */
        /* hm_t lm = multiplied_poly_to_hash_table(sht, bht, h, etmp, b);
#if PARALLEL_HASHING
        hm_t mul = check_insert_in_hash_table(etmp, h, sht);
#else
        hm_t mul = insert_in_hash_table(etmp, sht);
#endif */
        /* track trace information ? */
        /* if (tht != NULL) {
            rows[rr][BINDEX]  = lmps[i];
            if (tht->eld == tht->esz-1) {
                enlarge_hash_table(tht);
            }
#if PARALLEL_HASHING
            rows[rr][MULT]    = check_insert_in_hash_table(etmp, h, tht);
#else
            rows[rr][MULT]    = insert_in_hash_table(etmp, tht);
#endif
        } */
		/* pre_rr[2*(*nr)]		= mul;
		pre_rr[2*(*nr)+1]	= prev; */
        sht->idx[m]  = 2;
        *nr             = rr + 1;
    }
}
static inline void find_multiplied_reducer(
        const bs_t * const bs,
        const hm_t m,
        const ht_t * const bht,
        len_t *nr,
        hm_t **rows,
        ht_t *sht,
        ht_t *tht
        )
{
    len_t i, k;

    const len_t rr  = *nr;

    const len_t evl = bht->evl;

    const exp_t * const e  = sht->ev[m];

    const hd_t hdm    = sht->hd[m];
    const len_t lml   = bs->lml;
    const sdm_t ns    = ~hdm.sdm;

    const sdm_t * const lms = bs->lm;
    const bl_t * const lmps = bs->lmps;

    exp_t *etmp = bht->ev[0];
    const hd_t * const hdb  = bht->hd;
    exp_t * const * const evb = bht->ev;

    i = 0;
start:
    while (i < lml && lms[i] & ns) {
        i++;
    }
    if (i < lml) {
        const hm_t *b = bs->hm[lmps[i]];
        const exp_t * const f = evb[b[OFFSET]];
        for (k=0; k < evl; ++k) {
            etmp[k] = (exp_t)(e[k]-f[k]);
            if (etmp[k] < 0) {
                i++;
                goto start;
            }
        }
        const hi_t h  = hdm.val - hdb[b[OFFSET]].val;
        rows[rr]  = multiplied_poly_to_matrix_row(sht, bht, h, etmp, b);
        /* track trace information ? */
        if (tht != NULL) {
            rows[rr][BINDEX]  = lmps[i];
            if (tht->eld == tht->esz-1) {
                enlarge_hash_table(tht);
            }
#if PARALLEL_HASHING
            rows[rr][MULT]    = check_insert_in_hash_table(etmp, h, tht);
#else
            rows[rr][MULT]    = insert_in_hash_table(etmp, tht);
#endif
        }
        sht->idx[m]  = 2;
        *nr             = rr + 1;
    }
}

static void symbolic_preprocessing_new(
        mat_t *mat,
        ht_t *ht,
        const bs_t * const bs,
        stat_t *st
        )
{
    /* timings */
    double ct = cputime();
    double rt = realtime();

    len_t i = 0, j = 0;

    while (i < ht->lhld) {
        while (mat->sz - mat->nru < ht->lhld - i) {
            mat->sz *= 2;
            mat->rrd = realloc(mat->rrd,
                    (unsigned long)2 * mat->sz * sizeof(hm_t));
        }
        j = i;
        i = ht->lhld;
        const len_t lld = i;

        for (; j < lld; ++j) {
            if (ht->idx[ht->lh[j]] == 1) {
                find_multiplied_reducer_data(j, ht, mat, bs);
                ht->idx[ht->lh[j]] = 2;
                mat->nc++;
            }
        }
    }
    mat->rrd = realloc(mat->rrd, (unsigned long)2*mat->nru * sizeof(hm_t));
    mat->nr  = mat->nrl + mat->nru;
    mat->sz  = mat->nr;

    /* timings */
    st->symbol_ctime += cputime() - ct;
    st->symbol_rtime += realtime() - rt;
}

static void symbolic_preprocessing(
        mat_t *mat,
        const bs_t * const bs,
        stat_t *st,
        ht_t *sht,
        ht_t *tht,
        const ht_t * const bht
        )
{
#if 0
    hl_t i;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* at the moment we have as many reducers as we have different lcms */
    len_t nrr = mat->nc; 

    /* note that we have already counted the different lcms, i.e.
     * ncols until this step. moreover, we have also already marked
     * the corresponding hash indices to represent lead terms. so
     * we only have to do the bookkeeping for newly added reducers
     * in the following. */

    const hl_t oesld = sht->eld;
    const len_t onrr  = mat->nc;
    i = 1;
    /* we only have to check if idx is set for the elements already set
     * when selecting spairs, afterwards (second for loop) we do not
     * have to do this check */
    while (mat->sz <= nrr + oesld) {
        mat->sz *=  2;
        mat->pre_rr =   realloc(mat->pre_rr, (unsigned long)mat->sz * 2 * sizeof(hm_t));
    }
    for (; i < oesld; ++i) {
        if (!sht->hd[i].idx) {
            sht->hd[i].idx = 1;
            mat->nc++;
            find_multiplied_reducer_no_row(bs, i, bht, &nrr, mat->pre_rr, sht, tht);
        }
    }
    for (; i < sht->eld; ++i) {
        if (mat->sz == nrr) {
            mat->sz *=  2;
            mat->rr  =  realloc(mat->rr, (unsigned long)mat->sz * sizeof(hm_t *));
        }
        sht->hd[i].idx = 1;
        mat->nc++;
        find_multiplied_reducer_no_row(bs, i, bht, &nrr, mat->pre_rr, sht, tht);
    }
    /* realloc to real size */
    mat->pre_rr   =   realloc(mat->pre_rr, (unsigned long)2*nrr * sizeof(hm_t));
    mat->nr   +=  nrr - onrr;
    mat->nrl  =   mat->nr - nrr;
    mat->nru  =   nrr;
    mat->sz   =   mat->nr;
    mat->rbal =   mat->nrl;

    /* initialize memory for reducer bit arrays for tracing information */
    mat->rba  = (rba_t **)malloc((unsigned long)mat->rbal * sizeof(rba_t *));
    const unsigned long len = nrr / 32 + ((nrr % 32) != 0);
    for (i = 0; i < mat->nrl; ++i) {
        mat->rba[i] = (rba_t *)calloc(len, sizeof(rba_t));
    }

    /* statistics */
    st->max_sht_size  = st->max_sht_size > sht->esz ?
        st->max_sht_size : sht->esz;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->symbol_ctime  +=  ct1 - ct0;
    st->symbol_rtime  +=  rt1 - rt0;
#endif
}

static void generate_matrix_from_trace(
        mat_t *mat,
        const trace_t * const trace,
        const len_t idx,
        const bs_t * const bs,
        stat_t *st,
        ht_t *sht,
        const ht_t * const bht,
        const ht_t * const tht
        )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    len_t i, nr;
    hm_t *b;
    exp_t *emul;
    hi_t h;

    td_t td       = trace->td[idx];
    mat->rr       = (hm_t **)malloc((unsigned long)td.rld * sizeof(hm_t *));
    hm_t **rrows  = mat->rr;
    mat->tr       = (hm_t **)malloc((unsigned long)td.tld * sizeof(hm_t *));
    hm_t **trows  = mat->tr;
    mat->rba      = (rba_t **)malloc((unsigned long)td.tld * sizeof(rba_t *));
    rba_t **rba   = mat->rba;

    /* reducer rows, i.e. AB part */
    i   = 0;
    nr  = 0;
    while (i < td.rld) {
        b     = bs->hm[td.rri[i++]];
        emul  = tht->ev[td.rri[i]];
        h     = tht->hd[td.rri[i++]].val;

        rrows[nr] = multiplied_poly_to_matrix_row(sht, bht, h, emul, b);
        sht->idx[rrows[nr][OFFSET]] = 2;
        ++nr;

    }
    /* to be reduced rows, i.e. CD part */
    i   = 0;
    nr  = 0;
    while (i < td.tld) {
        b     = bs->hm[td.tri[i++]];
        emul  = tht->ev[td.tri[i]];
        h     = tht->hd[td.tri[i]].val;

        trows[nr] = multiplied_poly_to_matrix_row(sht, bht, h, emul, b);
        /* At the moment rba is unused */
        rba[nr]   = td.rba[i/2];
        i++;
        nr++;
    }
    /* meta data for matrix */
    mat->nru  = td.rld/2;
    mat->nrl  = td.tld/2;
    mat->nr   = mat->sz = mat->nru + mat->nrl;
    mat->nc   = sht->eld-1;

    /* statistics */
    st->max_sht_size  = st->max_sht_size > sht->esz ?
        st->max_sht_size : sht->esz;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->symbol_ctime  +=  ct1 - ct0;
    st->symbol_rtime  +=  rt1 - rt0;
}

static void generate_saturation_reducer_rows_from_trace(
        mat_t *mat,
        const trace_t * const trace,
        const len_t idx,
        const bs_t * const bs,
        stat_t *st,
        ht_t *sht,
        const ht_t * const bht,
        const ht_t * const tht
        )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    len_t i, nr;
    hm_t *b;
    exp_t *emul;
    hi_t h;

    ts_t ts       = trace->ts[idx];
    mat->rr       = (hm_t **)malloc((unsigned long)ts.rld * sizeof(hm_t *));
    hm_t **rrows  = mat->rr;

    /* reducer rows, i.e. AB part */
    i   = 0;
    nr  = 0;
    while (i < ts.rld) {
        b     = bs->hm[ts.rri[i++]];
        emul  = tht->ev[ts.rri[i]];
        h     = tht->hd[ts.rri[i++]].val;

        rrows[nr] = multiplied_poly_to_matrix_row(sht, bht, h, emul, b);
        sht->idx[rrows[nr][OFFSET]]= 2;
        ++nr;

    }
    /* meta data for matrix */
    mat->nru  = ts.rld/2;
    mat->nr   = mat->sz = mat->nru + mat->nrl;
    mat->nc   = sht->eld-1;

    /* statistics */
    st->max_sht_size  = st->max_sht_size > sht->esz ?
        st->max_sht_size : sht->esz;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->symbol_ctime  +=  ct1 - ct0;
    st->symbol_rtime  +=  rt1 - rt0;
}
