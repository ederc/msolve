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

/* after calling this procedure we have column indices instead of exponent
 * hashes in the polynomials resp. rows. moreover, we have sorted each row
 * by pivots / non-pivots. thus we get already an A|B splicing of the
 * initial matrix. this is a first step for receiving a full GBLA matrix. */
static void convert_multipliers_to_columns(
        hi_t **hcmp,
        bs_t *sat,
        stat_t *st,
        ht_t *ht
        )
{
    hl_t i;

    hi_t *hcm = *hcmp;
    /* clear ht-ev[0] */
    memset(ht->ev[0], 0, (unsigned long)ht->nv * sizeof(exp_t));

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* all elements in the sht hash table represent
     * exactly one column of the matrix */
    hcm = realloc(hcm, (unsigned long)sat->ld * sizeof(hi_t));
    for (i = 0; i < sat->ld; ++i) {
        hcm[i]  = sat->hm[i][MULT];
    }
    sort_r(hcm, (unsigned long)sat->ld, sizeof(hi_t), hcm_cmp, ht);

    /* printf("hcmm\n");
     * for (int ii=0; ii<sat->ld; ++ii) {
     *     printf("hcmm[%d] = %d | idx %u | ", ii, ht->hd[hcm[ii]].idx, hcm[ii]);
     *     for (int jj = 0; jj < ht->nv; ++jj) {
     *         printf("%d ", ht->ev[hcm[ii]][jj]);
     *     }
     *     printf("\n");
     * } */

    /* store the other direction (hash -> column) */
    for (i = 0; i < sat->ld; ++i) {
        ht->idx[hcm[i]]  = (hi_t)i;
    }

    /* map column positions to mul entries*/
    for (i = 0; i < sat->ld; ++i) {
        sat->hm[i][MULT]  =  ht->idx[sat->hm[i][MULT]];
    }
    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
    *hcmp = hcm;
}

static void convert_hashes_to_columns_sat(
        hi_t **hcmp,
        mat_t *mat,
        bs_t *sat,
        stat_t *st,
        ht_t *sht
        )
{
    hl_t i;
    hi_t j, k;
    hm_t *row;
    int64_t nterms = 0;

    hi_t *hcm = *hcmp;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    len_t hi;

    const len_t mnr = mat->nr;
    const hl_t esld = sht->eld;
    len_t *hds      = sht->idx;
    hm_t **rrows    = mat->rr;

    /* all elements in the sht hash table represent
     * exactly one column of the matrix */
    hcm = realloc(hcm, (esld-1) * sizeof(hi_t));
    for (k = 0, j = 0, i = 1; i < esld; ++i) {
        hi  = hds[i];

        hcm[j++]  = i;
        if (hi == 2) {
            k++;
        }
    }
    sort_r(hcm, (unsigned long)j, sizeof(hi_t), hcm_cmp, sht);

    /* printf("hcm\n");
     * for (int ii=0; ii<j; ++ii) {
     *     printf("hcm[%d] = %d | idx %u | deg %u |", ii, hcm[ii], hds[hcm[ii]].idx, sht->ev[hcm[ii]][DEG]+sht->ev[hcm[ii]][sht->ebl]);
     *     for (int jj = 0; jj < sht->evl; ++jj) {
     *         printf("%d ", sht->ev[hcm[ii]][jj]);
     *     }
     *     printf("\n");
     * } */

    mat->ncl  = k;
    mat->ncr  = (len_t)esld - 1 - mat->ncl;

    st->num_rowsred +=  sat->ld;

    /* store the other direction (hash -> column) */
    const hi_t ld = (hi_t)(esld - 1);
    for (k = 0; k < ld; ++k) {
        hds[hcm[k]]  = (hi_t)k;
    }

    /* map column positions to reducer matrix */
#pragma omp parallel for num_threads(st->nthrds) private(k, j, row)
    for (k = 0; k < mat->nru; ++k) {
        const len_t os  = rrows[k][PRELOOP];
        const len_t len = rrows[k][LENGTH];
        row = rrows[k] + OFFSET;
        for (j = 0; j < os; ++j) {
            row[j]  = hds[row[j]];
        }
        for (; j < len; j += UNROLL) {
            row[j]    = hds[row[j]];
            row[j+1]  = hds[row[j+1]];
            row[j+2]  = hds[row[j+2]];
            row[j+3]  = hds[row[j+3]];
        }
    }
    for (k = 0; k < mat->nru; ++k) {
        nterms  +=  rrows[k][LENGTH];
    }
    /* map column positions to saturation elements */
#pragma omp parallel for num_threads(st->nthrds) private(k, j, row)
    for (k = 0; k < sat->ld; ++k) {
        const len_t os  = sat->hm[k][PRELOOP];
        const len_t len = sat->hm[k][LENGTH];
        row = sat->hm[k] + OFFSET;
        for (j = 0; j < os; ++j) {
            row[j]  = hds[row[j]];
        }
        for (; j < len; j += UNROLL) {
            row[j]    = hds[row[j]];
            row[j+1]  = hds[row[j+1]];
            row[j+2]  = hds[row[j+2]];
            row[j+3]  = hds[row[j+3]];
        }
    }
    for (k = 0; k < mat->nrl; ++k) {
        nterms  +=  sat->hm[k][LENGTH];
    }

    /* next we sort each row by the new colum order due
     * to known / unkown pivots */

    /* NOTE: As strange as it may sound, we do not need to sort the rows.
     * When reducing, we copy them to dense rows, there we copy the coefficients
     * at the right place and reduce then. For the reducers itself it is not
     * important in which order the terms are represented as long as the first
     * term is the lead term, which is always true. Once a row is finally reduced
     * it is copied back to a sparse representation, now in the correct term
     * order since it is coming from the correctly sorted dense row. So all newly
     * added elements have all their terms sorted correctly w.r.t. the given
     * monomial order. */

    /* compute density of matrix */
    nterms  *=  100; /* for percentage */
    double density = (double)nterms / (double)mnr / (double)mat->nc;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
    if (st->info_level > 1) {
        printf(" %7d x %-7d %8.2f%%", mat->nr + sat->ld, mat->nc, density);
        fflush(stdout);
    }
    *hcmp = hcm;
}


static void sba_convert_hashes_to_columns(
        hi_t **hcmp,
        smat_t *smat,
        stat_t *st,
        ht_t *ht
        )
{
    len_t i, j, k;

    hm_t *row;
    int64_t nterms = 0;

    hi_t *hcm = *hcmp;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    const len_t nr = smat->cld;
    const hl_t eld = ht->eld;
    len_t *hd      = ht->idx;
    hm_t **cr      = smat->cr;

    hcm = realloc(hcm, (unsigned long)eld * sizeof(hi_t));
    k = 0;
    for (i = 0; i < nr; ++i) {
        const len_t len = SM_OFFSET + cr[i][SM_LEN];
        for (j = SM_OFFSET; j < len; ++j) {
            if (hd[cr[i][j]]== 0) {
                hd[cr[i][j]]= 1;
                hcm[k++] = cr[i][j];
            }
        }
    }

    hcm = realloc(hcm, (unsigned long)k * sizeof(hi_t));
    sort_r(hcm, (unsigned long)k, sizeof(hi_t), hcm_cmp, ht);

    smat->nc = k;

    /* printf("hcm\n");
     * for (int ii=0; ii<j; ++ii) {
     *     printf("hcm[%d] = %d | idx %u | deg %u |", ii, hcm[ii], hds[hcm[ii]].idx, sht->ev[hcm[ii]][DEG]+sht->ev[hcm[ii]][sht->ebl]);
     *     for (int jj = 0; jj < sht->evl; ++jj) {
     *         printf("%d ", sht->ev[hcm[ii]][jj]);
     *     }
     *     printf("\n");
     * } */

    /* store the other direction (hash -> column) */
    const hi_t ld = k;
    for (i = 0; i < ld; ++i) {
        hd[hcm[i]] = (hi_t)i;
    }

    /* map column positions to matrix rows */
#pragma omp parallel for num_threads(st->nthrds) private(k, j, row)
    for (i = 0; i < nr; ++i) {
        const len_t os  = cr[i][SM_PRE];
        const len_t len = cr[i][SM_LEN];
        row = cr[i] + SM_OFFSET;
        for (j = 0; j < os; ++j) {
            row[j]  = hd[row[j]];
        }
        for (; j < len; j += UNROLL) {
            row[j]    = hd[row[j]];
            row[j+1]  = hd[row[j+1]];
            row[j+2]  = hd[row[j+2]];
            row[j+3]  = hd[row[j+3]];
        }
        nterms += len;
    }

    /* compute density of matrix */
    nterms  *=  100; /* for percentage */
    double density = (double)nterms / (double)nr / (double)smat->nc;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
    if (st->info_level > 1) {
        printf("%4d    %7d x %-7d %8.2f%%", smat->cd, smat->cld, smat->nc, density);
        fflush(stdout);
    }
    *hcmp = hcm;
}


static void convert_hashes_to_columns_no_matrix(
        ht_t *ht,
        const bs_t * const bs,
        stat_t *st
        )
{
    /* timings */
    double ct = cputime();
    double rt = realtime();

    ht->lh = realloc(ht->lh, (unsigned long)ht->lhld * sizeof(len_t));
    for (int ii = 0; ii < ht->lhld; ++ii) {
        printf("lh[%d] = %u\n", ii, ht->lh[ii]);
    }
    sort_r(ht->lh, (unsigned long)ht->lhld, sizeof(hi_t), hcm_cmp, ht);
    printf("sorted\n-------------------\n");
    for (int ii = 0; ii < ht->lhld; ++ii) {
        printf("lh[%d] = %u | ", ii, ht->lh[ii]);
        for (int jj = 0; jj < ht->evl; ++jj) {
            printf("%u ", ht->ev[ht->lh[ii]][jj]);
        }
        printf("\n");
    }

    /* store the other direction (hash -> column) */
    for (len_t i = 0; i < ht->lhld; ++i) {
        ht->idx[ht->lh[i]]  = (hi_t)i;
    }

    /* timings */
    st->convert_ctime += cputime() - ct;
    st->convert_rtime += realtime() - rt;
}

/* When generating the column differences information for the rows we do a bit
of playing around with different sizes for packing the information as much
as possible to decrease memory usage. Please see the corresponding documentation
at the various stages of the code below as well as the macro definitions in
src/neogb/data.h for more information. */
static void generate_matrix_row(
        mat_t *mat,
        const len_t idx,
        const hm_t mul,
        const exp_t * const emul,
        const hm_t * const poly,
        const ht_t * const ht,
        const bs_t * const bs
        )
{
    len_t i, j, k, d;
    const len_t len = poly[LENGTH];
    const len_t * const hi = ht->idx;

    /* allocate memory for row:
        - OFFSET for meta data
        - len for possible long column differences
        - len/RATIO for column differences
        - len%RATIO > 0 for len not divisible by RATIO */
    const unsigned long rlen = OFFSET + len + len/RATIO + (len%RATIO > 0);
    len_t *row = calloc(rlen, sizeof(len_t));

    /* set meta data */
    row[DEG]     = ht->hd[mul].deg + poly[DEG];
    row[BINDEX]  = poly[BINDEX];
    row[MULT]    = mul;
    row[COEFFS]  = poly[COEFFS];
    row[PRELOOP] = poly[PRELOOP];
    row[LENGTH]  = poly[LENGTH];

    /* write column difference data */
    k = 0;
    j = 0; /* counts number of column differences >= 2^BSCD - 1 */
    cd_t *cd   = (cd_t *)(row + OFFSET);
    len_t *lcd = row + (rlen - len);
    for (i = 0; i < len; ++i) {
        const len_t idx  = hi[get_multiplied_monomial(
                                mul, emul, poly[OFFSET+i], ht)];
        d = idx - k;
        if (d < SCD) {
            cd[i] = (cd_t)d;
        } else {
            cd[i]    = (cd_t)SCD;
            lcd[j++] = d;
        }
        k = idx;
    }
    printf("cd = ");
    for (i = 0; i < len; ++i) {
        printf("%u ", cd[i]);
    }
    printf("\n");
    /* get rid of unused space for long column differences */
    printf("len%ratio %d --> %d\n", len%RATIO, len%RATIO>0);
    printf("row[LENGTH] %d\n", row[LENGTH]);
    printf("rlen %lu\n", rlen);
    printf("j %d\n", j);
    row = realloc(row, (rlen - (len - j)) * sizeof(len_t));
    printf("new length %d\n",(rlen - (len - j)));
    mat->row[idx] = row;
}

static void generate_reducer_matrix_part(
        mat_t *mat,
        const ht_t * const ht,
        const bs_t * const bs,
        stat_t *st
        )
{
    len_t i = 0, j = 0;

    const len_t *rrd       = mat->rrd;
    const len_t * const hi = ht->idx;

    /* we directly allocate space for all rows, not only for the
    known pivots, but also for the later on newly computed ones */
    mat->row   = calloc((unsigned long)mat->nc, sizeof(len_t *));
    mat->op    = calloc((unsigned long)mat->nru, sizeof(len_t *));

    switch (st->ff_bits) {
        case 8:
            mat->cf_8 = calloc((unsigned long)mat->nc, sizeof(cf8_t *));
            for (i = 0; i < mat->nru; ++i) {
                const hm_t mul    = rrd[2*i];
                const exp_t *emul = ht->ev[mul];
                const hm_t *poly  = bs->hm[rrd[2*i+1]];
                /* get multiplied leading term to insert at right place */
                const len_t idx = hi[get_multiplied_monomial(
                                        mul, emul, poly[OFFSET], ht)];
                generate_matrix_row(mat, idx, mul, emul, poly, ht, bs);
                mat->cf_8[idx] = bs->cf_8[mat->row[idx][COEFFS]];
                mat->op[j++] = mat->row[idx];
            }
            break;
        case 16:
            mat->cf_16 = calloc((unsigned long)mat->nc, sizeof(cf16_t *));
            for (i = 0; i < mat->nru; ++i) {
                const hm_t mul    = rrd[2*i];
                const exp_t *emul = ht->ev[mul];
                const hm_t *poly  = bs->hm[rrd[2*i+1]];
                /* get multiplied leading term to insert at right place */
                const len_t idx = hi[get_multiplied_monomial(
                                        mul, emul, poly[OFFSET], ht)];
                generate_matrix_row(mat, idx, mul, emul, poly, ht, bs);
                mat->cf_16[idx] = bs->cf_16[mat->row[idx][COEFFS]];
                mat->op[j++] = mat->row[idx];
            }
            break;
        case 32:
            mat->cf_32 = calloc((unsigned long)mat->nc, sizeof(cf32_t *));
            for (i = 0; i < mat->nru; ++i) {
                const hm_t mul    = rrd[2*i];
                const exp_t *emul = ht->ev[mul];
                const hm_t *poly  = bs->hm[rrd[2*i+1]];
                /* get multiplied leading term to insert at right place */
                const len_t idx = hi[get_multiplied_monomial(
                                        mul, emul, poly[OFFSET], ht)];
                generate_matrix_row(mat, idx, mul, emul, poly, ht, bs);
                mat->cf_32[idx] = bs->cf_32[mat->row[idx][COEFFS]];
                mat->op[j++] = mat->row[idx];
            }
            break;
        default:
            fprintf(stderr, "ff_bits not correctly set in generate_reducer_matrix_part().\n");
            exit(1);
    }
}


static void convert_hashes_to_columns(
        hi_t **hcmp,
        mat_t *mat,
        stat_t *st,
        ht_t *sht
        )
{
    hl_t i;
    hi_t j, k;
    hm_t *row;
    int64_t nterms = 0;

    hi_t *hcm = *hcmp;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    len_t hi;

    const len_t mnr = mat->nr;
    const hl_t esld = sht->eld;
    len_t *hds      = sht->idx;
    hm_t **rrows    = mat->rr;
    hm_t **trows    = mat->tr;

    /* all elements in the sht hash table represent
     * exactly one column of the matrix */
    hcm = realloc(hcm, (esld-1) * sizeof(hi_t));
    for (k = 0, j = 0, i = 1; i < esld; ++i) {
        hi  = hds[i];

        hcm[j++]  = i;
        if (hi == 2) {
            k++;
        }
    }
    sort_r(hcm, (unsigned long)j, sizeof(hi_t), hcm_cmp, sht);

    /* printf("hcm\n");
     * for (int ii=0; ii<j; ++ii) {
     *     printf("hcm[%d] = %d | idx %u | deg %u |", ii, hcm[ii], hds[hcm[ii]].idx, sht->ev[hcm[ii]][DEG]+sht->ev[hcm[ii]][sht->ebl]);
     *     for (int jj = 0; jj < sht->evl; ++jj) {
     *         printf("%d ", sht->ev[hcm[ii]][jj]);
     *     }
     *     printf("\n");
     * } */

    mat->ncl  = k;
    mat->ncr  = (len_t)esld - 1 - mat->ncl;

    st->num_rowsred +=  mat->nrl;

    /* store the other direction (hash -> column) */
    const hi_t ld = (hi_t)(esld - 1);
    for (k = 0; k < ld; ++k) {
        hds[hcm[k]]= (hi_t)k;
    }

    /* map column positions to matrix rows */
#pragma omp parallel for num_threads(st->nthrds) private(k, j, row)
    for (k = 0; k < mat->nru; ++k) {
        const len_t os  = rrows[k][PRELOOP];
        const len_t len = rrows[k][LENGTH];
        row = rrows[k] + OFFSET;
		len_t prev = 0;
		len_t tmp;
        for (j = 0; j < os; ++j) {
			tmp = row[j];
            row[j]  = hds[row[j]] - prev;
			prev	= hds[tmp];
        }
        for (; j < len; j += UNROLL) {
			tmp = row[j];
            row[j]		= hds[row[j]] - prev;
			prev		= hds[tmp];
			tmp = row[j+1];
            row[j+1]	= hds[row[j+1]] - prev;
			prev		= hds[tmp];
			tmp = row[j+2];
            row[j+2]	= hds[row[j+2]] - prev;
			prev		= hds[tmp];
			tmp = row[j+3];
            row[j+3]	= hds[row[j+3]] - prev;
			prev		= hds[tmp];
        }
    }
    for (k = 0; k < mat->nru; ++k) {
        nterms  +=  rrows[k][LENGTH];
    }
#pragma omp parallel for num_threads(st->nthrds) private(k, j, row)
    for (k = 0; k < mat->nrl; ++k) {
        const len_t os  = trows[k][PRELOOP];
        const len_t len = trows[k][LENGTH];
        row = trows[k] + OFFSET;
		len_t prev = 0;
		len_t tmp;
        for (j = 0; j < os; ++j) {
			tmp = row[j];
            row[j]  = hds[row[j]] - prev;
			prev	= hds[tmp];
        }
        for (; j < len; j += UNROLL) {
			tmp = row[j];
            row[j]		= hds[row[j]] - prev;
			prev		= hds[tmp];
			tmp = row[j+1];
            row[j+1]	= hds[row[j+1]] - prev;
			prev		= hds[tmp];
			tmp = row[j+2];
            row[j+2]	= hds[row[j+2]] - prev;
			prev		= hds[tmp];
			tmp = row[j+3];
            row[j+3]	= hds[row[j+3]] - prev;
			prev		= hds[tmp];
        }
    }
    for (k = 0; k < mat->nrl; ++k) {
        nterms  +=  trows[k][LENGTH];
    }

    /* next we sort each row by the new colum order due
     * to known / unkown pivots */

    /* NOTE: As strange as it may sound, we do not need to sort the rows.
     * When reducing, we copy them to dense rows, there we copy the coefficients
     * at the right place and reduce then. For the reducers itself it is not
     * important in which order the terms are represented as long as the first
     * term is the lead term, which is always true. Once a row is finally reduced
     * it is copied back to a sparse representation, now in the correct term
     * order since it is coming from the correctly sorted dense row. So all newly
     * added elements have all their terms sorted correctly w.r.t. the given
     * monomial order. */

    /* compute density of matrix */
    nterms  *=  100; /* for percentage */
    double density = (double)nterms / (double)mnr / (double)mat->nc;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
    if (st->info_level > 1) {
        printf(" %7d x %-7d %8.2f%%", mat->nr, mat->nc, density);
        fflush(stdout);
    }
    *hcmp = hcm;
}

static void sba_convert_columns_to_hashes(
        smat_t *smat,
        const hi_t * const hcm
        )
{
    len_t i, j;

    for (i = 0; i < smat->cld; ++i) {
        const len_t len = smat->cr[i][SM_LEN] + SM_OFFSET;
        for (j = SM_OFFSET; j < len; ++j) {
            smat->cr[i][j] = hcm[smat->cr[i][j]];
        }
    }
}

static void convert_columns_to_hashes(
        bs_t *bs,
        const hi_t * const hcm,
        const hi_t * const hcmm
        )
{
    len_t i, j;

    for (i = 0; i < bs->ld; ++i) {
        if (bs->hm[i] != NULL) {
            for (j = OFFSET; j < bs->hm[i][LENGTH]+OFFSET; ++j) {
                bs->hm[i][j]  = hcm[bs->hm[i][j]];
            }
            bs->hm[i][MULT] = hcmm[bs->hm[i][MULT]];
        }
    }
}

/* add_kernel_elements_to_basis() is unused at the moment */
#if 0
static void add_kernel_elements_to_basis(
        bs_t *sat,
        bs_t *bs,
        bs_t *kernel,
        const ht_t * const ht,
        const hi_t * const hcm,
        stat_t *st
        )
{
    len_t *terms  = (len_t *)calloc((unsigned long)sat->ld, sizeof(len_t));
    len_t nterms  = 0;
    len_t i, j, k;

    len_t ctr       = 0;
    const len_t bld = bs->ld;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* fix size of basis for entering new elements directly */
    check_enlarge_basis(bs, kernel->ld);

    /* we need to sort the kernel elements first (in order to track
     * redundancy correctly) */
    hm_t **rows = (hm_t **)calloc((unsigned long)kernel->ld, sizeof(hm_t *));
    k = 0;
    for (i = 0; i < kernel->ld; ++i) {
        /* printf("kernel[%u] = %p\n", i, kernel->hm[i]); */
        rows[k++]     = kernel->hm[i];
        kernel->hm[i] = NULL;
    }
    sort_matrix_rows_increasing(rows, k);
    /* only for 32 bit at the moment */
    for (i = 0; i < kernel->ld; ++i) {
        bs->cf_32[bld+ctr]              = kernel->cf_32[rows[i][COEFFS]];
        kernel->cf_32[rows[i][COEFFS]]  = NULL;
        bs->hm[bld+ctr]                 = rows[i];
        bs->hm[bld+ctr][COEFFS]         = bld+ctr;
        j = OFFSET;
next_j:
        for (; j < bs->hm[bld+ctr][LENGTH]+OFFSET; ++j) {
            bs->hm[bld+ctr][j] = hcm[bs->hm[bld+ctr][j]];
            if (nterms != 0) {
                for (int kk = 0; kk < nterms; ++kk) {
                    if (terms[kk] == bs->hm[bld+ctr][j]) {
                        j++;
                        goto next_j;
                    }
                }
            }
            terms[nterms] = bs->hm[bld+ctr][j];
            nterms++;
        }
        if (ht->ev[bs->hm[bld+ctr][OFFSET]][DEG] == 0) {
            bs->constant  = 1;
        }
        /* printf("new element from kernel (%u): length %u | ", bld+ctr, bs->hm[bld+ctr][LENGTH]);
         * for (int kk=0; kk<bs->hm[bld+ctr][LENGTH]; ++kk) {
         *     printf("%u | ", bs->cf_32[bld+ctr][kk]);
         *     printf("%u | ", ht->ev[bs->hm[bld+ctr][OFFSET+kk]][DEG]);
         *     for (int jj=0; jj < ht->nv; ++jj) {
         *         printf("%u ", ht->ev[bs->hm[bld+ctr][OFFSET+kk]][jj]);
         *     }
         *     printf(" || ");
         * }
         * printf("\n"); */
        ctr++;
    }
    /* printf("%u of %u terms are used for kernel elements, %.2f\n", nterms, sat->ld, (float)nterms / (float)sat->ld); */
    free(terms);
    free(rows);
    rows  = NULL;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
}
#endif

static void return_normal_forms_to_basis(
        mat_t *mat,
        bs_t *bs,
        ht_t *bht,
        const ht_t * const sht,
        const hi_t * const hcm,
        stat_t *st
        )
{
    len_t i;

    const len_t np  = mat->np;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();
    /* fix size of basis for entering new elements directly */
    check_enlarge_basis(bs, mat->np, st);

    hm_t **rows = mat->tr;

    /* only for 32 bit at the moment */
    for (i = 0; i < np; ++i) {
        if (rows[i] != NULL) {
            insert_in_basis_hash_table_pivots(rows[i], bht, sht, hcm, st);
            bs->cf_32[bs->ld] = mat->cf_32[rows[i][COEFFS]];
            rows[i][COEFFS]   = bs->ld;
            bs->hm[bs->ld]    = rows[i];
        } else {
            bs->cf_32[bs->ld] = NULL;
            bs->hm[bs->ld]    = NULL;
        }
        bs->lmps[bs->ld]  = bs->ld;
        bs->lml++;
        bs->ld++;
    }

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
}

static void convert_sparse_matrix_rows_to_basis_elements(
        const int sort,
        mat_t *mat,
        bs_t *bs,
        ht_t *bht,
        const ht_t * const sht,
        const hi_t * const hcm,
        stat_t *st
        )
{
    len_t i, j, k;
    deg_t deg;

    const len_t bl  = bs->ld;
    const len_t np  = mat->np;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* fix size of basis for entering new elements directly */
    check_enlarge_basis(bs, mat->np, st);

    hm_t **rows = mat->tr;

    for (k = 0; k < np; ++k) {
        /* We first insert the highest leading monomial element to the basis
         * for a better Gebauer-Moeller application when updating the pair
         * set later on. */
        if (sort == -1) {
            i   =   np - 1 - k;
        } else {
            i = k;
        }
        insert_in_basis_hash_table_pivots(rows[i], bht, sht, hcm, st);
        deg = bht->hd[rows[i][OFFSET]].deg;
        if (st->nev > 0) {
            const len_t len = rows[i][LENGTH]+OFFSET;
            for (j = OFFSET+1; j < len; ++j) {
                if (deg < bht->hd[rows[i][j]].deg) {
                    deg = bht->hd[rows[i][j]].deg;
                }
            }
        }
        switch (st->ff_bits) {
            case 0:
                bs->cf_qq[bl+k] = mat->cf_qq[rows[i][COEFFS]];
                break;
            case 8:
                bs->cf_8[bl+k]  = mat->cf_8[rows[i][COEFFS]];
                break;
            case 16:
                bs->cf_16[bl+k] = mat->cf_16[rows[i][COEFFS]];
                break;
            case 32:
                bs->cf_32[bl+k] = mat->cf_32[rows[i][COEFFS]];
                break;
            default:
                bs->cf_32[bl+k] = mat->cf_32[rows[i][COEFFS]];
                break;
        }
        rows[i][COEFFS]   = bl+k;
        bs->hm[bl+k]      = rows[i];
        bs->hm[bl+k][DEG] = deg;
        if (deg == 0) {
            bs->constant  = 1;
        }
#if 0
        if (st->ff_bits == 32) {
            printf("new element (%u): length %u | degree %d | ", bl+k, bs->hm[bl+k][LENGTH], bs->hm[bl+k][DEG]);
            int kk = 0;
            for (int kk=0; kk<bs->hm[bl+k][LENGTH]; ++kk) {
            /* printf("%u | ", bs->cf_32[bl+k][kk]); */
            for (int jj=0; jj < bht->evl; ++jj) {
                printf("%u ", bht->ev[bs->hm[bl+k][OFFSET+kk]][jj]);
            }
            printf(" || ");
            }
            printf("\n");
        }
        if (st->ff_bits == 16) {
            printf("new element (%u): length %u | degree %d (difference %d) | ", bl+k, bs->hm[bl+k][LENGTH], bs->hm[bl+k][DEG],
                    bs->hm[bl+k][DEG] - bht->hd[bs->hm[bl+k][OFFSET]].deg);
            int kk = 0;
            for (int kk=0; kk<bs->hm[bl+k][LENGTH]; ++kk) {
            printf("%u | ", bs->cf_16[bl+k][kk]);
            for (int jj=0; jj < bht->evl; ++jj) {
                printf("%u ", bht->ev[bs->hm[bl+k][OFFSET+kk]][jj]);
            }
            printf(" || ");
            }
            printf("\n");
        }
        if (st->ff_bits == 8) {
            printf("new element (%u): length %u | degree %d (difference %d) | ", bl+k, bs->hm[bl+k][LENGTH], bs->hm[bl+k][DEG],
                    bs->hm[bl+k][DEG] - bht->hd[bs->hm[bl+k][OFFSET]].deg);
            int kk = 0;
            for (int kk=0; kk<bs->hm[bl+k][LENGTH]; ++kk) {
            printf("%u | ", bs->cf_8[bl+k][kk]);
            for (int jj=0; jj < bht->evl; ++jj) {
                printf("%u ", bht->ev[bs->hm[bl+k][OFFSET+kk]][jj]);
            }
            printf(" || ");
            }
            printf("\n");
        }
#endif
    }

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
}

static void convert_sparse_cd_matrix_rows_to_basis_elements(
        mat_t *mat,
        bs_t *bs,
        const ht_t * const ht,
        stat_t *st
        )
{
    /* timings */
    double ct = cputime();
    double rt = realtime();

    len_t i;

    const len_t np = mat->np;
    const len_t bl = bs->ld;

    const len_t * const lh = ht->lh;

    printf("bld %u, bsz %u\n", bs->ld, bs->sz);
    check_enlarge_basis(bs, mat->np, st);
    printf("mat->np %u\n", mat->np);
    printf("bld %u, bsz %u\n", bs->ld, bs->sz);

#pragma omp parallel for num_threads(st->nthrds) private(i)
    for (i = 0; i < np; ++i) {
        printf("cp[%d] = %p\n", i, mat->cp[i]);
        len_t k = 0;
        len_t pos = 0;

        const len_t len = mat->cp[i][LENGTH];

        const cd_t * const cd   = (cd_t *)(mat->cp[i] + OFFSET);
        const len_t * const lcd = mat->cp[i] + (OFFSET + len/RATIO + (len%RATIO > 0));

        hm_t *poly = calloc((unsigned long)len + OFFSET, sizeof(hm_t));
        hm_t *p = poly + OFFSET;
        for (len_t j = 0; j < len; ++j) {
            printf("cd[%u] = %u\n", j, cd[j]);
            pos = cd[j] != SCD ? pos + cd[j] : pos + lcd[k++];
            printf("lh[%u] = %u\n", pos, lh[pos]);
            p[j] = lh[pos];
            printf("poly[%u] = %u\n", j, poly[j]);
        }
        poly[LENGTH]  = len;
        poly[PRELOOP] = mat->cp[i][PRELOOP];
        poly[COEFFS]  = bl+i;
        poly[DEG]     = ht->hd[poly[OFFSET]].deg;

        /* check for degree of polynomial if we are using an elimination order */
        if (st->nev != 0) {
            const hd_t * const hd = ht->hd;
            deg_t deg = poly[DEG];
            for (len_t j = OFFSET; j < OFFSET+len; ++j) {
                deg = deg < hd[poly[j]].deg ? hd[poly[j]].deg : deg;
            }
            poly[DEG] = deg;
        }
        free(mat->cp[i]);
        mat->cp[i] = NULL;

        switch (st->ff_bits) {
            case 0:
                bs->cf_qq[bl+i] = mat->cf_qq[poly[COEFFS]];
                break;
            case 8:
                bs->cf_8[bl+i]  = mat->cf_8[poly[COEFFS]];
                break;
            case 16:
                bs->cf_16[bl+i] = mat->cf_16[poly[COEFFS]];
                break;
            case 32:
                bs->cf_32[bl+i] = mat->cf_32[poly[COEFFS]];
                break;
            default:
                bs->cf_32[bl+i] = mat->cf_32[poly[COEFFS]];
                break;
        }
        bs->hm[bl+i] = poly;
        if (poly[DEG] == 0) {
            bs->constant  = 1;
        }
#if 1
        if (st->ff_bits == 32) {
            printf("new element (%u): length %u | degree %d | ", bl+i, bs->hm[bl+i][LENGTH], bs->hm[bl+i][DEG]);
            int kk = 0;
            for (int kk=0; kk<bs->hm[bl+i][LENGTH]; ++kk) {
            printf("%u | ", bs->cf_32[bl+i][kk]);
            for (int jj=0; jj < ht->evl; ++jj) {
                printf("%u ", ht->ev[bs->hm[bl+i][OFFSET+kk]][jj]);
            }
            printf(" || ");
            }
            printf("\n");
        }
#endif
    }
    /* timings */
    st->convert_ctime += cputime() - ct;
    st->convert_rtime += realtime() - rt;
}

static void convert_sparse_matrix_rows_to_basis_elements_use_sht(
        const int sort,
        mat_t *mat,
        bs_t *bs,
        const ht_t * const sht,
        const hi_t * const hcm,
        stat_t *st
        )
{
    len_t i, j, k;
    deg_t deg;
    hm_t *row;

    const len_t bl  = bs->ld;
    const len_t np  = mat->np;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* fix size of basis for entering new elements directly */
    check_enlarge_basis(bs, mat->np, st);

    hm_t **rows = mat->tr;

    for (k = 0; k < np; ++k) {
        /* We first insert the highest leading monomial element to the basis
         * for a better Gebauer-Moeller application when updating the pair
         * set later on. */
        if (sort == -1) {
            i   =   np - 1 - k;
        } else {
            i = k;
        }
        row = rows[i];
        deg = sht->hd[hcm[rows[i][OFFSET]]].deg;
        const len_t len = rows[i][LENGTH]+OFFSET;
		len_t prev = 0;
        if (st->nev ==  0) {
            for (j = OFFSET; j < len; ++j) {
                prev += row[j];
				row[j]  = hcm[prev];
            }
        } else {
            for (j = OFFSET; j < len; ++j) {
				prev += row[j];
                row[j]  = hcm[prev];
                if (deg < sht->hd[row[j]].deg) {
                    deg = sht->hd[row[j]].deg;
                }
            }
        }
        switch (st->ff_bits) {
            case 0:
                bs->cf_qq[bl+k] = mat->cf_qq[rows[i][COEFFS]];
                break;
            case 8:
                bs->cf_8[bl+k]  = mat->cf_8[rows[i][COEFFS]];
                break;
            case 16:
                bs->cf_16[bl+k] = mat->cf_16[rows[i][COEFFS]];
                break;
            case 32:
                bs->cf_32[bl+k] = mat->cf_32[rows[i][COEFFS]];
                break;
            default:
                bs->cf_32[bl+k] = mat->cf_32[rows[i][COEFFS]];
                break;
        }
        rows[i][COEFFS]   = bl+k;
        bs->hm[bl+k]      = rows[i];
        bs->hm[bl+k][DEG] = deg;
        if (deg == 0) {
            bs->constant  = 1;
        }
#if 0
        if (st->ff_bits == 32) {
            printf("new element (%u): length %u | degree %d | ", bl+i, bs->hm[bl+i][LENGTH], bs->hm[bl+i][DEG]);
            int kk = 0;
            /* for (int kk=0; kk<bs->hm[bl+i][LENGTH]; ++kk) { */
            printf("%u | ", bs->cf_32[bl+i][kk]);
            for (int jj=0; jj < sht->evl; ++jj) {
                printf("%u ", sht->ev[bs->hm[bl+i][OFFSET+kk]][jj]);
            }
            /* printf(" || ");
             * } */
            printf("\n");
        }
#endif
    }

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->convert_ctime +=  ct1 - ct0;
    st->convert_rtime +=  rt1 - rt0;
}
