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

#include "../msolve/streams.h"


#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/* select_all_pairs() is unused at the moment */
#if 0
static void select_all_spairs(
        mat_t *mat,
        const bs_t * const bs,
        ps_t *psl,
        md_t *st,
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
    exp_t etmp[bht->evl];

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
        fprintf(VERBSTREAM, "%3d  %6d %7d", 0, nps, psl->ld);
        fflush(VERBSTREAM);
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
by minial degree of the spairs is supported

NOTE: The pair list is already sorted! */
static int32_t select_spairs_by_minimal_degree(
        mat_t *mat,
        bs_t *bs,
        md_t *md
        )
{
    len_t i, j, k, l, mdeg, nps, npd;
    ht_t *bht   = bs->ht;
    ps_t *psl   = md->ps;
    ht_t *sht   = md->ht;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    spair_t *ps     = psl->p;
    const len_t evl = bht->evl;

    /* sort pair set */
    sort_r(ps, (unsigned long)psl->ld, sizeof(spair_t), spair_cmp, bht);
    /* get minimal degree */
    mdeg  = ps[0].deg;

    /* compute a truncated GB? Check maximal degree. */
    if (md->max_gb_degree < mdeg) {
        return 1;
    }

    /* select pairs of this degree respecting maximal selection size mnsel */
    for (i = 0; i < psl->ld; ++i) {
        if (ps[i].deg > mdeg) {
            break;
        }
    }
    npd  = i;

    /* if we stopped due to maximal selection size we still get the following
     * pairs of the same lcm in this matrix */
    if (npd > md->mnsel) {
        nps = md->mnsel;
        const hi_t lcmsel = ps[nps].lcm;
        while (nps < npd && ps[nps+1].lcm == lcmsel) {
            nps++;
        }
    } else {
        nps = npd;
    }
    if (md->info_level > 1) {
        fprintf(VERBSTREAM, "%3d  %6d %7d", mdeg, nps, psl->ld);
        fflush(VERBSTREAM);
    }
    /* statistics */
    md->num_pairsred  +=  nps;
    /* list for generators (2 entries per selected pair, one slice per
     * lcm group) */
    len_t *gens  = (len_t *)malloc(2 * (unsigned long)nps * sizeof(len_t));
    /* preset matrix meta data */
    mat->rr       = (hm_t **)malloc(2 * (unsigned long)nps * sizeof(hm_t *));
    hm_t **rrows  = mat->rr;
    mat->tr       = (hm_t **)malloc(2 * (unsigned long)nps * sizeof(hm_t *));
    hm_t **trows  = mat->tr;
    mat->sz = 2 * nps;
    mat->nc = mat->ncl = mat->ncr = 0;
    mat->nr = 0;

    /* group boundaries: group g covers the pairs [gs[g], gs[g+1]),
     * nlcms is the number of groups (i.e. of different lcms) */
    len_t *gs    = (len_t *)malloc((unsigned long)(nps+1) * sizeof(len_t));
    len_t nlcms  = 0;
    i = 0;
    while (i < nps) {
        gs[nlcms++] = i;
        const hi_t lcm1 = ps[i].lcm;
        while (i < nps && ps[i].lcm == lcm1) {
            i++;
        }
    }
    gs[nlcms] = nps;

    /* per group: number of distinct generators (one matrix row each)
     * and the total number of terms (each row term is inserted into sht) */
    len_t *nd     = (len_t *)malloc((unsigned long)nlcms * sizeof(len_t));
    len_t *tcount = (len_t *)malloc((unsigned long)nlcms * sizeof(len_t));

    /* phase A: sort the generators of each group and count the
     * distinct ones. The generator slice of group i occupies
     * gens[2*s .. 2*e) (two entries per pair). */
#pragma omp parallel for num_threads(md->nthrds) schedule(dynamic) \
    private(j, k)
    for (i = 0; i < nlcms; ++i) {
        const len_t s    = gs[i];
        const len_t e    = gs[i+1];
        const len_t ngen = 2 * (e - s);
        for (j = s; j < e; ++j) {
            gens[2*j]   = ps[j].gen1;
            gens[2*j+1] = ps[j].gen2;
        }
        qsort(gens + 2*s, (unsigned long)ngen, sizeof(len_t), gens_cmp);
        len_t prev = gens[2*s];
        len_t ndg  = 1;
        len_t tc   = bs->hm[prev][LENGTH];
        for (k = 1; k < ngen; ++k) {
            if (gens[2*s+k] == prev) {
                continue;
            }
            prev = gens[2*s+k];
            ndg++;
            tc += bs->hm[prev][LENGTH];
        }
        nd[i]     = ndg;
        tcount[i] = tc;
    }

    /* prefix sums: trows offset per group, total number of rows and terms */
    len_t *pre_t     = (len_t *)malloc((unsigned long)nlcms * sizeof(len_t));
    len_t total_nd   = 0;
    len_t total_terms = 0;
    for (i = 0; i < nlcms; ++i) {
        pre_t[i]     = total_nd;
        total_nd    += nd[i] - 1;
        total_terms += tcount[i];
    }

    /* phase B: build the rows, one thread per lcm group.
     * The first distinct generator of a group goes into the reducer
     * part of the matrix, all remaining ones go to the to-be-reduced
     * part. All rows of a group have the group lcm as lead monomial.
     *
     * A parallel pass requires that sht does not resize (and does not
     * overflow) while it runs, so before each pass the table is
     * pre-enlarged with a factor 2 headroom (which also covers the
     * slots reserved but lost in concurrent insert races). If a pass
     * overflows anyway, its rows are discarded and the pass is
     * repeated on the enlarged table. */
    int ovflag = 0;
    for (;;) {
    while (sht->esz < sht->eld + 2ULL * (hl_t)total_terms + 1) {
        enlarge_hash_table(sht);
    }
    /* pre-enlarge bht for the tracer multiplier insertions
     * (one insertion per matrix row: nlcms reducer rows plus
     * total_nd to-be-reduced rows) */
    if (md->trace_level == LEARN_TRACER) {
        while (bht->esz - bht->eld < 2ULL * ((hl_t)total_nd + (hl_t)nlcms) + 1) {
            enlarge_hash_table(bht);
        }
    }
    ovflag = 0;

#pragma omp parallel for num_threads(md->nthrds) schedule(dynamic) \
    private(k, l)
    for (i = 0; i < nlcms; ++i) {
        const len_t s    = gs[i];
        const len_t e    = gs[i+1];
        const len_t ngen = 2 * (e - s);
        const hi_t lcmg  = ps[s].lcm;
        const exp_t * const elcm = bht->ev[lcmg];
        len_t prev = gens[2*s];

        /* reducer row */
        const hm_t *b  = bs->hm[prev];
        exp_t etmp[evl];
        const exp_t *eb = bht->ev[b[OFFSET]];
        for (l = 0; l < evl; ++l) {
            etmp[l] = (exp_t)(elcm[l] - eb[l]);
        }
        int ov = 0;
        hm_t *row = ts_multiplied_poly_to_matrix_row(
                sht, bht, etmp, b, NULL, NULL, &ov);
        if (md->trace_level == LEARN_TRACER) {
            row[BINDEX]  = prev;
            row[MULT]    = thread_safe_insert_in_hash_table(etmp, bht, NULL);
        }
        /* mark lcm column as lead term column */
        __atomic_store_n(&sht->hd[row[OFFSET]].idx, 2, __ATOMIC_RELAXED);
        rrows[i] = row;
        if (ov) {
            __atomic_store_n(&ovflag, 1, __ATOMIC_RELAXED);
        }

        len_t tn = 0;
        for (k = 1; k < ngen; ++k) {
            /* check sorted list for doubles */
            if (gens[2*s+k] == prev) {
                continue;
            }
            prev  = gens[2*s+k];
            b     = bs->hm[prev];
            eb    = bht->ev[b[OFFSET]];
            for (l = 0; l < evl; ++l) {
                etmp[l] = (exp_t)(elcm[l] - eb[l]);
            }
            ov = 0;
            row = ts_multiplied_poly_to_matrix_row(
                    sht, bht, etmp, b, NULL, NULL, &ov);
            if (ov) {
                __atomic_store_n(&ovflag, 1, __ATOMIC_RELAXED);
            }
            if (md->trace_level == LEARN_TRACER) {
                row[BINDEX]  = prev;
                row[MULT]    = thread_safe_insert_in_hash_table(etmp, bht, NULL);
            }
            /* mark lcm column as lead term column */
            __atomic_store_n(&sht->hd[row[OFFSET]].idx, 2, __ATOMIC_RELAXED);
            trows[pre_t[i] + tn] = row;
            tn++;
        }
    }

    if (!ovflag) {
        break;
    }
    /* the pass overflowed: discard the rows it built and repeat it
     * on the (now enlarged) table */
    for (i = 0; i < nlcms; ++i) {
        free(rrows[i]);
        for (k = 0; k < nd[i]-1; ++k) {
            free(trows[pre_t[i]+k]);
        }
    }
    }

    /* ncols counts the number of different lcms; the total number
     * of rows is one reducer row plus the trows per lcm group */
    mat->nc = nlcms;
    mat->nr = total_nd + nlcms;

    /* fix rows to be reduced */
    mat->tr = realloc(mat->tr, (unsigned long)(mat->nr - mat->nc) * sizeof(hm_t *));

    md->num_rowsred +=  mat->nr - mat->nc;
    md->current_deg =   mdeg;
    mat->cd         =   mdeg;

    free(gens);
    free(gs);
    free(nd);
    free(tcount);
    free(pre_t);

    /* remove selected spairs from pairset */
    memmove(ps, ps+nps, (unsigned long)(psl->ld-nps) * sizeof(spair_t));
    psl->ld -=  nps;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    md->select_ctime  +=  ct1 - ct0;
    md->select_rtime  +=  rt1 - rt0;

    return 0;
}

/* write elements straight to sat, not to a matrix */
static void select_saturation(
        bs_t *sat,
        mat_t *mat,
        md_t *st,
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
        md_t *st,
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


static inline void find_multiplied_reducer(
        bs_t *bs,
        const hm_t m,
        len_t *nr,
        hm_t **rows,
        ht_t *sht,
        const md_t * const md
        )
{
    len_t i, k;

    ht_t *bht = bs->ht;

    const len_t rr  = *nr;

    const len_t evl = bht->evl;

    const exp_t * const e  = sht->ev[m];

    const hd_t hdm    = sht->hd[m];
    const len_t lml   = bs->lml;
    const sdm_t ns    = ~hdm.sdm;

    const sdm_t * const lms = bs->lm;
    const bl_t * const lmps = bs->lmps;

    exp_t etmp[bht->evl];
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
            if (e[k] < f[k]) {
                i++;
                goto start;
            }
            etmp[k] = (exp_t)(e[k]-f[k]);
        }

        const hi_t h  = hdm.val - hdb[b[OFFSET]].val;
        rows[rr]  = multiplied_poly_to_matrix_row(sht, bht, h, etmp, b);
        /* track trace information ? */
        if (md->trace_level == LEARN_TRACER) {
            rows[rr][BINDEX]  = lmps[i];
            if (bht->eld == bht->esz-1) {
                enlarge_hash_table(bht);
            }
            rows[rr][MULT]    = insert_in_hash_table(etmp, bht);
        }
        sht->hd[m].idx  = 2;
        *nr             = rr + 1;
    }
}

/* thread-safe version of find_multiplied_reducer() for use inside an
 * OpenMP parallel region. The table sht must be pre-enlarged so that
 * no resize can occur. If q and q_tail are not NULL, every newly
 * created sht entry is appended to the queue q (atomically).
 * Returns 1 if a reducer was found (a row was added), 0 if no reducer
 * exists, and -1 if the table sht overflowed (no row was added, m has
 * to be processed again). */
static int32_t ts_find_multiplied_reducer(
        const bs_t *bs,
        const hm_t m,
        hl_t *nr,
        hm_t **rows,
        ht_t *sht,
        const md_t * const md,
        qent_t *q,
        hl_t *q_tail
        )
{
    len_t i, k;

    ht_t *bht = bs->ht;

    const len_t evl = bht->evl;

    const exp_t * const e  = sht->ev[m];

    const hd_t hdm    = sht->hd[m];
    const len_t lml   = bs->lml;
    const sdm_t ns    = ~hdm.sdm;

    const sdm_t * const lms = bs->lm;
    const bl_t * const lmps = bs->lmps;

    exp_t etmp[bht->evl];
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
            if (e[k] < f[k]) {
                i++;
                goto start;
            }
            etmp[k] = (exp_t)(e[k]-f[k]);
        }

        int ov = 0;
        hm_t *row = ts_multiplied_poly_to_matrix_row(
                sht, bht, etmp, b, q, q_tail, &ov);
        if (ov) {
            free(row);
            return -1;
        }
        const hl_t rr = __atomic_fetch_add(nr, 1, __ATOMIC_RELAXED);
        rows[rr] = row;
        /* track trace information ? */
        if (md->trace_level == LEARN_TRACER) {
            row[BINDEX]  = lmps[i];
            row[MULT]    = thread_safe_insert_in_hash_table(etmp, bht, NULL);
        }
        __atomic_store_n(&sht->hd[m].idx, 2, __ATOMIC_RELAXED);
        return 1;
    }
    return 0;
}

static void symbolic_preprocessing(
        mat_t *mat,
        bs_t *bs,
        md_t *md
        )
{
    hl_t i;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* at the moment we have as many reducers as we have different lcms */
    len_t nrr = mat->nc;

    ht_t *sht = md->ht;
    ht_t *bht = bs->ht;

    /* note that we have already counted the different lcms, i.e.
     * ncols until this step. moreover, we have also already marked
     * the corresponding hash indices to represent lead terms. so
     * we only have to do the bookkeeping for newly added reducers
     * in the following. */

    const hl_t oesld = sht->eld;
    const len_t onrr  = mat->nc;

    /* ---- parallel version ----
     *
     * The monomials form a closure: processing a monomial may
     * create new monomials (the terms of its multiplier row) that
     * have to be processed as well. We maintain a shared work
     * queue of sht entries; every entry is enqueued exactly once
     * (by its creator) and processed exactly once. The reducer of
     * a monomial is the first dividing lead monomial in lmps
     * order, so the closure (and hence the set of rows) does not
     * depend on the processing order.
     *
     * A parallel phase requires that sht does not resize (and
     * does not overflow) while it runs, so before each phase the
     * table is enlarged with a factor 2 headroom. If the table
     * overflows during a phase, the overflowing monomials are
     * processed serially (which may enlarge sht on demand) and a
     * new parallel phase starts on the larger table. Since each
     * overflow at least doubles the table, the loop terminates. */

     qent_t *q  = NULL;
     hi_t *ovf  = (hi_t *)calloc((unsigned long)md->nthrds, sizeof(hi_t));
     hl_t q_head = 0;
     hl_t q_tail = 0;
     hl_t q_cap  = 0;

     for (;;) {
        /* ensure a factor 2 headroom in sht so that no resize
         * (and no overflow) can occur during the parallel phase */
        while (sht->esz < 2ULL * (hl_t)sht->eld + 64) {
            enlarge_hash_table(sht);
        }
        const hl_t cap = sht->esz;

        /* grow the row array so it can hold one row per table
         * entry (at most one row is added per processed entry) */
        while (mat->sz <= (hl_t)onrr + cap) {
            mat->sz *=  2;
            mat->rr =   realloc(mat->rr, (unsigned long)mat->sz * sizeof(hm_t *));
            if (mat->rr == NULL) {
                fprintf(ERRSTREAM, "Allocating memory for matrix failed,\n");
                fprintf(ERRSTREAM, "segmentation fault will follow.\n");
            }
        }

        /* pre-enlarge bht for the tracer multiplier insertions
         * (at most one insertion per processed entry, plus the
         * slots lost in concurrent insert races) */
        if (md->trace_level == LEARN_TRACER) {
            while (bht->esz - bht->eld < 2ULL * cap + 1) {
                enlarge_hash_table(bht);
            }
        }

        /* grow the work queue to cover the whole table; new slots
         * must start with flag 0 ("not yet produced"), the memory
         * may otherwise hold garbage that a consumer's flag spin
         * would mistake for a published slot */
        if (q_cap < cap) {
            const hl_t old_cap = q_cap;
            q_cap = cap;
            q = (qent_t *)realloc(q, (unsigned long)q_cap * sizeof(qent_t));
            if (q != NULL) {
                memset(q + old_cap, 0,
                       (q_cap - old_cap) * sizeof(qent_t));
            }
        }

        /* enqueue all not-yet-counted entries of the current
         * table (only needed in the first iteration); the plain
         * stores are published by the omp barrier below */
        if (q_head == 0 && q_tail == 0) {
            for (i = 1; i < oesld; ++i) {
                if (!sht->hd[i].idx) {
                    q[q_tail].v = (hi_t)i;
                    q[q_tail].flag = 1;
                    q_tail++;
                }
            }
        }

        for (i = 0; i < (hl_t)md->nthrds; ++i) {
            ovf[i] = 0;
        }

        hl_t pnrr     = nrr;
        hl_t nc_inc   = 0;
        hl_t q_idle   = 0;
        int overflow  = 0;

        /* each thread pops entries from the shared queue. A slot
         * is claimed with a CAS on q_head only after seeing that
         * the slot has been produced (q_tail has passed it), and
         * producers publish slot values with a per-slot ready
         * flag, so a consumer never observes a stale slot value.
         * A thread that finds the queue empty does not leave
         * right away: other threads may still append, so it
         * waits until the queue is refilled or all threads are
         * idle. */
        #pragma omp parallel num_threads(md->nthrds)
        {
            int my_ovf = 0;
            for (;;) {
                if (__atomic_load_n(&overflow, __ATOMIC_ACQUIRE)) {
                    break;
                }
                hl_t pos = __atomic_load_n(&q_head, __ATOMIC_ACQUIRE);
                if (pos >= __atomic_load_n(&q_tail, __ATOMIC_ACQUIRE)) {
                    /* the queue is empty; a concurrent producer
                     * may still append, so wait until the queue
                     * is refilled or no thread is active any
                     * more. q_idle counts threads that are
                     * waiting here or have already retired from
                     * this phase: it reaches nthrds exactly when
                     * every thread is idle, and it stays there. */
                    __atomic_add_fetch(&q_idle, 1, __ATOMIC_RELAXED);
                    int reenter = 0;
                    for (;;) {
                        if (__atomic_load_n(&overflow,
                                __ATOMIC_ACQUIRE)) {
                            break;
                        }
                        if (__atomic_load_n(&q_tail,
                                __ATOMIC_ACQUIRE)
                                > __atomic_load_n(&q_head,
                                      __ATOMIC_ACQUIRE)) {
                            /* refilled: become active again */
                            __atomic_sub_fetch(&q_idle, 1,
                                    __ATOMIC_RELAXED);
                            reenter = 1;
                            break;
                        }
                        if (__atomic_load_n(&q_idle,
                                __ATOMIC_ACQUIRE)
                                >= (hl_t)md->nthrds) {
                            break;
                        }
                    }
                    if (reenter
                            && !__atomic_load_n(&overflow,
                                 __ATOMIC_ACQUIRE)) {
                        continue;
                    }
                    break;
                }
                if (!__atomic_compare_exchange_n(
                        &q_head, &pos, pos + 1, 0,
                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                    continue;
                }
                /* wait until the producer of slot pos published
                 * its value */
                while (!__atomic_load_n(&q[pos].flag,
                        __ATOMIC_ACQUIRE)) {
                }
                __atomic_store_n(&q[pos].flag, 0, __ATOMIC_RELAXED);
                const hi_t m = q[pos].v;
                if (__atomic_load_n(&sht->hd[m].idx,
                        __ATOMIC_ACQUIRE) == 2) {
                    continue;
                }
                __atomic_add_fetch(&nc_inc, 1, __ATOMIC_RELAXED);
                const int32_t r = ts_find_multiplied_reducer(
                        bs, m, &pnrr, mat->rr, sht, md, q, &q_tail);
                if (r < 0) {
                    /* sht overflowed: this monomial is processed
                     * serially after the barrier */
                    __atomic_fetch_sub(&nc_inc, 1, __ATOMIC_RELAXED);
                    __atomic_store_n(&overflow, 1, __ATOMIC_RELEASE);
                    my_ovf = m;
                    break;
                }
            }
            if (my_ovf) {
                __atomic_store_n(&ovf[omp_get_thread_num()], my_ovf,
                        __ATOMIC_RELEASE);
            }
        }

        /* continue the row counter with the parallel part */
        nrr    = (len_t)pnrr;
        mat->nc += (len_t)nc_inc;

        /* serial processing of the overflowing monomials; the
         * serial row builder enlarges sht on demand, and the
         * entries created by the new rows are enqueued */
        for (i = 0; i < (hl_t)md->nthrds; ++i) {
            const hi_t m = __atomic_load_n(&ovf[i], __ATOMIC_ACQUIRE);
            if (!m || sht->hd[m].idx == 2) {
                continue;
            }
            sht->hd[m].idx = 1;
            mat->nc++;
            if (mat->sz == nrr) {
                mat->sz *= 2;
                mat->rr  =  realloc(mat->rr, (unsigned long)mat->sz * sizeof(hm_t *));
                if (mat->rr == NULL) {
                    fprintf(ERRSTREAM, "Allocating memory for matrix failed,\n");
                    fprintf(ERRSTREAM, "segmentation fault will follow.\n");
                }
            }
            const hl_t before = sht->eld;
            find_multiplied_reducer(bs, m, &nrr, mat->rr, sht, md);
            if (q_tail + (sht->eld - before) > q_cap) {
                const hl_t old_cap = q_cap;
                q_cap = 2 * q_cap + (sht->eld - before);
                q = (qent_t *)realloc(q,
                        (unsigned long)q_cap * sizeof(qent_t));
                if (q != NULL) {
                    memset(q + old_cap, 0,
                           (q_cap - old_cap) * sizeof(qent_t));
                }
            }
            /* plain stores; published by the next omp barrier */
            for (hl_t s2 = before; s2 < sht->eld; ++s2) {
                q[q_tail].v = (hi_t)s2;
                q[q_tail].flag = 1;
                q_tail++;
            }
        }

        if (q_head >= q_tail) {
            /* queue exhausted: all monomials of the closure
             * have been processed */
            break;
        }
        /* otherwise start a new parallel phase on the (possibly
         * enlarged) table */
    }

    free(q);
    free(ovf);

    /* realloc to real size */
    mat->rr   =   realloc(mat->rr, (unsigned long)nrr * sizeof(hm_t *));
    mat->nr   +=  nrr - onrr;
    mat->nrl  =   mat->nr - nrr;
    mat->nru  =   nrr;
    mat->sz   =   mat->nr;
    mat->rbal =   mat->nrl;

    /* initialize memory for reducer bit arrays for tracing information */
    mat->rba  = (rba_t **)malloc((uint64_t)mat->rbal * sizeof(rba_t *));
    const uint64_t len = nrr / 32 + ((nrr % 32) != 0);
    for (i = 0; i < mat->nrl; ++i) {
        mat->rba[i] = (rba_t *)calloc(len, sizeof(rba_t));
    }

    /* statistics */
    md->max_sht_size  = md->max_sht_size > sht->esz ?
        md->max_sht_size : sht->esz;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    md->symbol_ctime  +=  ct1 - ct0;
    md->symbol_rtime  +=  rt1 - rt0;
}

static void generate_matrix_from_trace(
        mat_t *mat,
        const bs_t * const bs,
        md_t *md
        )
{
    /* timings */
    double ct, rt;
    ct = cputime();
    rt = realtime();

    len_t i, nr;
    hm_t *b;
    exp_t *emul;
    hi_t h;

    const len_t idx = md->trace_rd;

    td_t td   = md->tr->td[idx];
    ht_t *bht = bs->ht;
    ht_t *sht = md->ht;

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
        emul  = bht->ev[td.rri[i]];
        h     = bht->hd[td.rri[i++]].val;


        rrows[nr] = multiplied_poly_to_matrix_row(sht, bht, h, emul, b);
        sht->hd[rrows[nr][OFFSET]].idx = 2;
        ++nr;

    }
    /* to be reduced rows, i.e. CD part */
    i   = 0;
    nr  = 0;
    while (i < td.tld) {
        b     = bs->hm[td.tri[i++]];
        emul  = bht->ev[td.tri[i]];
        h     = bht->hd[td.tri[i]].val;
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
    md->max_sht_size  = md->max_sht_size > sht->esz ?
        md->max_sht_size : sht->esz;

    /* timings */
    md->tracer_ctime += cputime() - ct;
    md->tracer_rtime += realtime() - rt;

    print_current_trace_meta_data(md);
}

static void generate_saturation_reducer_rows_from_trace(
        mat_t *mat,
        const trace_t * const trace,
        const len_t idx,
        const bs_t * const bs,
        md_t *st,
        ht_t *sht,
        const ht_t * const bht
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
        emul  = bht->ev[ts.rri[i]];
        h     = bht->hd[ts.rri[i++]].val;

        rrows[nr] = multiplied_poly_to_matrix_row(sht, bht, h, emul, b);
        sht->hd[rrows[nr][OFFSET]].idx = 2;
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

static int preprocessing(
        mat_t *mat,
        bs_t *bs,
        md_t *md
        )
{
    if (md->trace_level != APPLY_TRACER) {
        if (select_spairs_by_minimal_degree(mat, bs, md)) {
            return 1;
        }
        symbolic_preprocessing(mat, bs, md);
    } else {
        generate_matrix_from_trace(mat, bs, md);
    }
    if (mat->nc == 0) {
        return 1;
    } else {
        return 0;
    }
}
