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

/* That's also enough if AVX512 is avaialable on the system */
#if defined HAVE_AVX2
#include <immintrin.h>
#elif defined __aarch64__
#include <arm_neon.h>
#endif

static inline cf24_t *normalize_sparse_matrix_row_ff_24(
        cf24_t *row,
        const len_t os,
        const len_t len,
        const double fc
        )
{
    len_t i;
    double tmp;

    const cf32_t fc32  = (cf32_t)fc;
    const double inv   = (double)mod_p_inverse_32(row[0], fc32);
    const double mod   = (double)fc;
    const double invmod = (double)1.0/mod;

    for (i = 0; i < os; ++i) {
        tmp1    = inv * row[i];
        row[i]  = (cf24_t)(tmp1 - floor(tmp1 * invmod) * mod);
    }
    for (i = os; i < len; i += OFFSET) {
        tmp1     = inv * row[i];
        tmp2     = inv * row[i+1];
        tmp3     = inv * row[i+2];
        tmp4     = inv * row[i+3];
        row[i]   = (cf24_t)(tmp1 - floor(tmp1 * invmod) * mod);
        row[i+1] = (cf24_t)(tmp2 - floor(tmp2 * invmod) * mod);
        row[i+2] = (cf24_t)(tmp3 - floor(tmp3 * invmod) * mod);
        row[i+3] = (cf24_t)(tmp4 - floor(tmp4 * invmod) * mod);
    }

    return row;
}

static hm_t *reduce_dense_row_by_known_pivots_sparse_ff_24(
        double *dr,
        mat_t *mat,
        const bs_t * const bs,
        hm_t * const * const pivs,
        const hi_t dpiv,    /* pivot of dense row at the beginning */
        const hm_t tmp_pos, /* position of new coeffs array in tmpcf */
        const len_t mh,     /* multiplier hash for tracing */
        const len_t bi,     /* basis index of generating element */
        const len_t tr,     /* trace data? */
        const uint32_t fc
        )
{
    hi_t i, j, k;
    hm_t *dts;
    cf24_t *cfs;
    int64_t np = -1;
    const float mod             = (float)fc;
    const float invmod          = pow(mod, -1);
    const len_t ncols           = mat->nc;
    const len_t ncl             = mat->ncl;
    cf24_t * const * const mcf  = mat->cf_24;
    uint8_t of[mat->ncr];

    rba_t *rba;
    if (tr > 0) {
        rba = mat->rba[tmp_pos];
    } else {
        rba = NULL;
    }

    k = 0;
    for (i = dpiv; i < ncols; ++i) {
        /* if (dr[i] != 0) {
            dr[i] = dr[i] % mod;
        } */
        if (dr[i] == 0) {
            continue;
        }
        if (pivs[i] == NULL) {
            if (np == -1) {
                np  = i;
            }
            k++;
            continue;
        }
        /* found reducer row, get multiplier */
        const uint32_t mul = (uint32_t)(fc - dr[i]);
        dts   = pivs[i];
        if (i < ncl) {
            /* set corresponding bit of reducer in reducer bit array */
            if (tr > 0) {
                rba[i/32] |= 1U << (i % 32);
            }
        }
        cfs   = mcf[dts[COEFFS]];
        const len_t os  = dts[PRELOOP];
        const len_t len = dts[LENGTH];
        const hm_t * const ds  = dts + OFFSET;
        for (j = 0; j < len; ++j) {
            dr[ds[j]] +=  mul * cfs[j];
            if (ds[j] > ncl) {
                of[ds[j]-ncl]++;
            }
            if (of[ds[j]] > 31) {
                dr[ds[j]] = dr[ds[j]] - floor(dr[ds[j]] * invmod) * mod;
                of[ds[j]] = 0;
            }
        }
        /* for (; j < len; j += UNROLL) {
            dr[ds[j]]   +=  mul * cfs[j];
            dr[ds[j+1]] +=  mul * cfs[j+1];
            dr[ds[j+2]] +=  mul * cfs[j+2];
            dr[ds[j+3]] +=  mul * cfs[j+3];
        } */
        dr[i] = 0;
    }
    if (k == 0) {
        return NULL;
    }

    hm_t *row   = (hm_t *)malloc((unsigned long)(k+OFFSET) * sizeof(hm_t));
    cf24_t *cf  = (cf24_t *)malloc((unsigned long)(k) * sizeof(cf24_t));
    j = 0;
    hm_t *rs = row + OFFSET;
    for (i = ncl; i < ncols; ++i) {
        if (dr[i] != 0) {
            rs[j] = (hm_t)i;
            cf[j] = (cf24_t)dr[i];
            j++;
        }
    }
    row[COEFFS]   = tmp_pos;
    row[PRELOOP]  = j % UNROLL;
    row[LENGTH]   = j;
    mat->cf_24[tmp_pos]  = cf;

    return row;
}

static void exact_sparse_reduced_echelon_form_ff_24(
        mat_t *mat,
        const bs_t * const tbr,
        const bs_t * const bs,
        md_t *st
        )
{
    len_t i = 0, j, k;
    hi_t sc    = 0;    /* starting column */

    const len_t ncols = mat->nc;
    const len_t nrl   = mat->nrl;
    const len_t ncr   = mat->ncr;
    const len_t ncl   = mat->ncl;

    const int32_t nthrds = st->in_final_reduction_step == 1 ? 1 : st->nthrds;

    len_t bad_prime = 0;

    /* we fill in all known lead terms in pivs */
    hm_t **pivs   = (hm_t **)calloc((unsigned long)ncols, sizeof(hm_t *));
    if (st->in_final_reduction_step == 0) {
        memcpy(pivs, mat->rr, (unsigned long)mat->nru * sizeof(hm_t *));
    } else {
        for (i = 0;  i < mat->nru; ++i) {
            pivs[mat->rr[i][OFFSET]] = mat->rr[i];
        }
    }
    j = nrl;
    for (i = 0; i < mat->nru; ++i) {
        mat->cf_24[j]      = bs->cf_24[mat->rr[i][COEFFS]];
        mat->rr[i][COEFFS] = j;
        ++j;
    }


    /* unkown pivot rows we have to reduce with the known pivots first */
    hm_t **upivs  = mat->tr;

    double *dr  = (double *)malloc(
            (unsigned long)(nthrds * ncols) * sizeof(double));
    /* mo need to have any sharing dependencies on parallel computation,
     * no data to be synchronized at this step of the linear algebra */
#pragma omp parallel for num_threads(nthrds) \
    private(i, j, k, sc) \
    schedule(dynamic)
    for (i = 0; i < nrl; ++i) {
        if (bad_prime == 0) {
            double *drl    = dr + (omp_get_thread_num() * ncols);
            hm_t *npiv      = upivs[i];
            cf24_t *cfs     = tbr->cf_24[npiv[COEFFS]];
            const len_t bi  = npiv[BINDEX];
            const len_t mh  = npiv[MULT];
            const len_t os  = npiv[PRELOOP];
            const len_t len = npiv[LENGTH];
            const hm_t * const ds = npiv + OFFSET;
            k = 0;
            memset(drl, 0, (unsigned long)ncols * sizeof(double));
            for (j = 0; j < os; ++j) {
                drl[ds[j]]  = (double)cfs[j];
            }
            for (; j < len; j += UNROLL) {
                drl[ds[j]]    = (double)cfs[j];
                drl[ds[j+1]]  = (double)cfs[j+1];
                drl[ds[j+2]]  = (double)cfs[j+2];
                drl[ds[j+3]]  = (double)cfs[j+3];
            }
            cfs = NULL;
            do {
                /* If we do normal form computations the first monomial in the polynomial might not
                be a known pivot, thus setting it to npiv[OFFSET] can lead to wrong results. */
                sc  = st->nf == 0 ? npiv[OFFSET] : 0;
                free(npiv);
                npiv  = NULL;
                free(cfs);
                cfs = NULL;
                npiv  = mat->tr[i] = reduce_dense_row_by_known_pivots_sparse_ff_24(
                        drl, mat, bs, pivs, sc, i, mh, bi, st->trace_level == LEARN_TRACER, st->fc);
                if (st->nf > 0) {
                    if (!npiv) {
                        mat->tr[i]  = NULL;
                        break;
                    }
                    mat->tr[i]  = npiv;
                    cfs = mat->cf_24[npiv[COEFFS]];
                    break;
                } else {
                    if (!npiv) {
                        break;
                    }
                    /* normalize coefficient array
                     * NOTE: this has to be done here, otherwise the reduction may
                     * lead to wrong results in a parallel computation since other
                     * threads might directly use the new pivot once it is synced. */
                    if (mat->cf_24[npiv[COEFFS]][0] != 1) {
                        normalize_sparse_matrix_row_ff_24(
                                mat->cf_24[npiv[COEFFS]], npiv[PRELOOP], npiv[LENGTH], st->fc);
                    }
                    k   = __sync_bool_compare_and_swap(&pivs[npiv[OFFSET]], NULL, npiv);
                    cfs = mat->cf_24[npiv[COEFFS]];
                }
            } while (!k);
        }
    }

    if (bad_prime == 1) {
        for (i = 0; i < ncl+ncr; ++i) {
            free(pivs[i]);
            pivs[i] = NULL;
        }
        mat->np = 0;
        if (st->info_level > 0) {
            fprintf(stderr, "Zero reduction while applying tracer, bad prime.\n");
        }
        return;
    }

    /* construct the trace */
    if (st->trace_level == LEARN_TRACER && st->in_final_reduction_step == 0) {
        construct_trace(st->tr, mat);
    }

    /* we do not need the old pivots anymore */
    for (i = 0; i < ncl; ++i) {
        free(pivs[i]);
        pivs[i] = NULL;
    }

    len_t npivs = 0; /* number of new pivots */

    if (st->nf == 0 && st->in_final_reduction_step == 0) {
        dr      = realloc(dr, (unsigned long)ncols * sizeof(double));
        mat->tr = realloc(mat->tr, (unsigned long)ncr * sizeof(hm_t *));

        /* interreduce new pivots */
        cf24_t *cfs;
        hm_t cf_array_pos;
        for (i = 0; i < ncr; ++i) {
            k = ncols-1-i;
            if (pivs[k]) {
                memset(dr, 0, (unsigned long)ncols * sizeof(double));
                cfs = mat->cf_24[pivs[k][COEFFS]];
                cf_array_pos    = pivs[k][COEFFS];
                const len_t bi  = pivs[k][BINDEX];
                const len_t mh  = pivs[k][MULT];
                const len_t os  = pivs[k][PRELOOP];
                const len_t len = pivs[k][LENGTH];
                const hm_t * const ds = pivs[k] + OFFSET;
                sc  = ds[0];
                for (j = 0; j < os; ++j) {
                    dr[ds[j]] = (double)cfs[j];
                }
                for (; j < len; j += UNROLL) {
                    dr[ds[j]]    = (double)cfs[j];
                    dr[ds[j+1]]  = (double)cfs[j+1];
                    dr[ds[j+2]]  = (double)cfs[j+2];
                    dr[ds[j+3]]  = (double)cfs[j+3];
                }
                free(pivs[k]);
                free(cfs);
                pivs[k] = NULL;
                pivs[k] = mat->tr[npivs++] =
                    reduce_dense_row_by_known_pivots_sparse_ff_24(
                            dr, mat, bs, pivs, sc, cf_array_pos, mh, bi, 0, st->fc);
            }
        }
        mat->tr = realloc(mat->tr, (unsigned long)npivs * sizeof(hi_t *));
        st->np = mat->np = mat->nr = mat->sz = npivs;
    } else {
        st->np = mat->np = mat->nr = mat->sz = nrl;
    }
    free(pivs);
    pivs  = NULL;
    free(dr);
    dr  = NULL;
}

/* In f4: tbr == bs
in nf: tbr are the polynomials to be reduced w.r.t. bs */
static void exact_sparse_linear_algebra_ff_24(
        mat_t *mat,
        const bs_t * const tbr,
        const bs_t * const bs,
        md_t *st
        )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* allocate temporary storage space for sparse
     * coefficients of all pivot rows */
    mat->cf_24  = realloc(mat->cf_24,
            (unsigned long)mat->nr * sizeof(cf24_t *));
    exact_sparse_reduced_echelon_form_ff_24(mat, tbr, bs, st);

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->la_ctime  +=  ct1 - ct0;
    st->la_rtime  +=  rt1 - rt0;

    st->num_zerored += (mat->nrl - mat->np);
    if (st->info_level > 1) {
        printf("%9d new %7d zero", mat->np, mat->nrl - mat->np);
        fflush(stdout);
    }
}

static void interreduce_matrix_rows_ff_24(
        mat_t *mat,
        bs_t *bs,
        md_t *st,
        const int free_basis
        )
{
    len_t i, j, k, l;

    const len_t nrows = mat->nr;
    const len_t ncols = mat->nc;

    /* adjust displaying timings for statistic printout */
    if (st->info_level > 1) {
        printf("                          ");
    }

    /* for interreduction steps like the final basis reduction we
    need to allocate memory for rba here, even so we do not use
    it at all */
    mat->rba  = (rba_t **)malloc((unsigned long)ncols * sizeof(rba_t *));
    const unsigned long len = ncols / 32 + ((ncols % 32) != 0);
    for (i = 0; i < ncols; ++i) {
        mat->rba[i] = (rba_t *)calloc(len, sizeof(rba_t));
    }

    mat->tr = realloc(mat->tr, (unsigned long)ncols * sizeof(hm_t *));

    mat->cf_24  = realloc(mat->cf_24,
            (unsigned long)ncols * sizeof(cf16_t *));
    memset(mat->cf_24, 0, (unsigned long)ncols * sizeof(cf24_t *));
    hm_t **pivs = (hm_t **)calloc((unsigned long)ncols, sizeof(hm_t *));
    /* copy coefficient arrays from basis in matrix, maybe
     * several rows need the same coefficient arrays, but we
     * cannot share them here. */
    for (i = 0; i < nrows; ++i) {
        pivs[mat->rr[i][OFFSET]]  = mat->rr[i];
    }

    double *dr = (double *)malloc((unsigned long)ncols * sizeof(double));
    /* interreduce new pivots */
    cf24_t *cfs;
    /* starting column, coefficient array position in tmpcf */
    hm_t sc;
    k = nrows - 1;
    for (i = 0; i < ncols; ++i) {
        l = ncols-1-i;
        if (pivs[l] != NULL) {
            memset(dr, 0, (unsigned long)ncols * sizeof(double));
            cfs = bs->cf_24[pivs[l][COEFFS]];
            const len_t bi  = pivs[l][BINDEX];
            const len_t mh  = pivs[l][MULT];
            const len_t os  = pivs[l][PRELOOP];
            const len_t len = pivs[l][LENGTH];
            const hm_t * const ds = pivs[l] + OFFSET;
            sc  = ds[0];
            for (j = 0; j < os; ++j) {
                dr[ds[j]] = (double)cfs[j];
            }
            for (; j < len; j += UNROLL) {
                dr[ds[j]]   = (double)cfs[j];
                dr[ds[j+1]] = (double)cfs[j+1];
                dr[ds[j+2]] = (double)cfs[j+2];
                dr[ds[j+3]] = (double)cfs[j+3];
            }
            free(pivs[l]);
            pivs[l] = NULL;
            pivs[l] = mat->tr[k--] =
                reduce_dense_row_by_known_pivots_sparse_ff_24(
                        dr, mat, bs, pivs, sc, l, mh, bi, 0, st->fc);
        }
    }
    for (i = 0; i < ncols; ++i) {
        free(mat->rba[i]);
        mat->rba[i] = NULL;
    }
    if (free_basis != 0) {
        /* free now all polynomials in the basis and reset bs->ld to 0. */
        free_basis_elements(bs);
    }
    free(mat->rr);
    mat->rr = NULL;
    st->np = mat->np = nrows;
    free(pivs);
    free(dr);
}

/* 
TODO: Implement below functions for 24 bit.
 */
#if 0
static void probabilistic_sparse_reduced_echelon_form_ff_16(
        mat_t *mat,
        const bs_t * const bs,
        md_t *st
        )
{
    len_t i = 0, j, k, l, m;

    const len_t ncols = mat->nc;
    const len_t nrl   = mat->nrl;
    const len_t ncr   = mat->ncr;
    const len_t ncl   = mat->ncl;

    /* we fill in all known lead terms in pivs */
    hm_t **pivs   = (hm_t **)calloc((unsigned long)ncols, sizeof(hm_t *));
    memcpy(pivs, mat->rr, (unsigned long)mat->nru * sizeof(hm_t *));
    j = nrl;
    for (i = 0; i < mat->nru; ++i) {
        mat->cf_16[j]      = bs->cf_16[mat->rr[i][COEFFS]];
        mat->rr[i][COEFFS] = j;
        ++j;
    }

    /* unkown pivot rows we have to reduce with the known pivots first */
    hm_t **upivs  = mat->tr;

    const uint32_t fc   = st->fc;
    const int64_t mod2  = (int64_t)fc * fc;

    /* compute rows per block */
    const len_t nb  = (len_t)(floor(sqrt(nrl/3)))+1;
    const len_t rem = (nrl % nb == 0) ? 0 : 1;
    const len_t rpb = (nrl / nb) + rem;

    int64_t *dr   = (int64_t *)malloc(
        (unsigned long)(st->nthrds * ncols) * sizeof(int64_t));
    int64_t *mul  = (int64_t *)malloc(
        (unsigned long)(st->nthrds * rpb) * sizeof(int64_t));

    /* mo need to have any sharing dependencies on parallel computation,
     * no data to be synchronized at this step of the linear algebra */
#pragma omp parallel for num_threads(st->nthrds) \
    private(i, j, k, l, m) \
    schedule(dynamic)
    for (i = 0; i < nb; ++i) {
        int64_t *drl  = dr + (omp_get_thread_num() * ncols);
        int64_t *mull = mul + (omp_get_thread_num() * rpb);
        const int32_t nbl   = (int32_t) (nrl > (i+1)*rpb ? (i+1)*rpb : nrl);
        const int32_t nrbl  = (int32_t) (nbl - i*rpb);
        if (nrbl != 0) {
            hm_t *npiv  = NULL;
            cf16_t *cfs;
            /* starting column, offset, coefficient array position in tmpcf */
            hm_t sc, cfp;
            len_t bctr  = 0;
            while (bctr < nrbl) {
                cfp = bctr + i*rpb;
                sc  = 0;

                /* fill random value array */
                for (j = 0; j < nrbl; ++j) {
                    mull[j] = (int64_t)rand() % fc;
                }
                /* generate one dense row as random linear combination
                 * of the rows of the block */
                memset(drl, 0, (unsigned long)ncols * sizeof(int64_t));

                for (k = 0, m = i*rpb; m < nbl; ++k, ++m) {
                    npiv  = upivs[m];
                    cfs   = bs->cf_16[npiv[COEFFS]];
                    const len_t os  = npiv[PRELOOP];
                    const len_t len = npiv[LENGTH];
                    const hm_t * const ds = npiv + OFFSET;
                    sc    = sc < ds[0] ? sc : ds[0];

                    for (l = 0; l < os; ++l) {
                        drl[ds[l]]  -=  mull[k] * cfs[l];
                        drl[ds[l]]  +=  (drl[ds[l]] >> 63) & mod2;
                    }
                    for (; l < len; l += UNROLL) {
                        drl[ds[l]]    -=  mull[k] * cfs[l];
                        drl[ds[l]]    +=  (drl[ds[l]] >> 63) & mod2;
                        drl[ds[l+1]]  -=  mull[k] * cfs[l+1];
                        drl[ds[l+1]]  +=  (drl[ds[l+1]] >> 63) & mod2;
                        drl[ds[l+2]]  -=  mull[k] * cfs[l+2];
                        drl[ds[l+2]]  +=  (drl[ds[l+2]] >> 63) & mod2;
                        drl[ds[l+3]]  -=  mull[k] * cfs[l+3];
                        drl[ds[l+3]]  +=  (drl[ds[l+3]] >> 63) & mod2;
                    }
                }
                k     = 0;
                cfs   = NULL;
                npiv  = NULL;
                /* do the reduction */
                do {
                    free(cfs);
                    cfs = NULL;
                    free(npiv);
                    npiv  = NULL;
                    npiv  = reduce_dense_row_by_known_pivots_sparse_ff_16(
                            drl, mat, bs, pivs, sc, cfp, 0, 0, 0, st->fc);
                    if (!npiv) {
                        bctr  = nrbl;
                        break;
                    }
                    /* normalize coefficient array
                    * NOTE: this has to be done here, otherwise the reduction may
                    * lead to wrong results in a parallel computation since other
                    * threads might directly use the new pivot once it is synced. */
                    if (mat->cf_16[npiv[COEFFS]][0] != 1) {
                        normalize_sparse_matrix_row_ff_16(
                                mat->cf_16[npiv[COEFFS]], npiv[PRELOOP], npiv[LENGTH], st->fc);
                    }
                    cfs = mat->cf_16[npiv[COEFFS]];
                    sc  = npiv[OFFSET];
                    k   = __sync_bool_compare_and_swap(&pivs[npiv[OFFSET]], NULL, npiv);
                } while (!k);
                bctr++;
            }
            for (j = i*rpb; j < nbl; ++j) {
                free(upivs[j]);
                upivs[j]  = NULL;
            }
        }
    }
    free(mul);
    mul   = NULL;

    /* construct the trace */
    if (st->trace_level == LEARN_TRACER && st->in_final_reduction_step == 0) {
        construct_trace(st->tr, mat);
    }

    /* we do not need the old pivots anymore */
    for (i = 0; i < ncl; ++i) {
        free(pivs[i]);
        pivs[i] = NULL;
    }

    len_t npivs = 0; /* number of new pivots */

    dr      = realloc(dr, (unsigned long)ncols * sizeof(int64_t));
    mat->tr = realloc(mat->tr, (unsigned long)ncr * sizeof(hm_t *));

    /* interreduce new pivots */
    cf16_t *cfs;
    /* starting column, coefficient array position in tmpcf */
    hm_t sc, cfp;
    for (i = 0; i < ncr; ++i) {
        k = ncols-1-i;
        if (pivs[k]) {
            memset(dr, 0, (unsigned long)ncols * sizeof(int64_t));
            cfs = mat->cf_16[pivs[k][COEFFS]];
            cfp = pivs[k][COEFFS];
            const len_t bi  = pivs[k][BINDEX];
            const len_t mh  = pivs[k][MULT];
            const len_t os  = pivs[k][PRELOOP];
            const len_t len = pivs[k][LENGTH];
            const hm_t * const ds = pivs[k] + OFFSET;
            sc  = ds[0];
            for (j = 0; j < os; ++j) {
                dr[ds[j]] = (int64_t)cfs[j];
            }
            for (; j < len; j += UNROLL) {
                dr[ds[j]]   = (int64_t)cfs[j];
                dr[ds[j+1]] = (int64_t)cfs[j+1];
                dr[ds[j+2]] = (int64_t)cfs[j+2];
                dr[ds[j+3]] = (int64_t)cfs[j+3];
            }
            free(pivs[k]);
            free(cfs);
            pivs[k] = NULL;
            pivs[k] = mat->tr[npivs++] =
                reduce_dense_row_by_known_pivots_sparse_ff_16(
                        dr, mat, bs, pivs, sc, cfp, mh, bi, 0, st->fc);
        }
    }
    free(pivs);
    pivs  = NULL;

    free(dr);
    dr  = NULL;

    mat->tr = realloc(mat->tr, (unsigned long)npivs * sizeof(hi_t *));
    st->np = mat->np = mat->nr = mat->sz = npivs;
}

static void convert_to_sparse_matrix_rows_ff_16(
        mat_t *mat,
        cf16_t * const * const dm
        )
{
    if (mat->np == 0) {
        return;
    }

    len_t i, j, k, l, m;
    cf16_t *cfs;
    hm_t *dts, *dss;

    const len_t ncr = mat->ncr;
    const len_t ncl = mat->ncl;

    mat->tr     = realloc(mat->tr, (unsigned long)mat->np * sizeof(hm_t *));
    mat->cf_16  = realloc(mat->cf_16,
            (unsigned long)mat->np * sizeof(cf16_t *));

    l = 0;
    for (i = 0; i < ncr; ++i) {
        m = ncr-1-i;
        if (dm[m] != NULL) {
            cfs = malloc((unsigned long)(ncr-m) * sizeof(cf16_t));
            dts = malloc((unsigned long)(ncr-m+OFFSET) * sizeof(hm_t));
            const hm_t len    = ncr-m;
            const hm_t os     = len % UNROLL;
            const hm_t shift  = ncl+m;
            dss = dts + OFFSET;

            for (k = 0, j = 0; j < os; ++j) {
                if (dm[m][j] != 0) {
                    cfs[k]    = dm[m][j];
                    dss[k++]  = j+shift;
                }
            }
            for (; j < len; j += UNROLL) {
                if (dm[m][j] != 0) {
                    cfs[k]    = dm[m][j];
                    dss[k++]  = j+shift;
                }
                if (dm[m][j+1] != 0) {
                    cfs[k]    = dm[m][j+1];
                    dss[k++]  = j+1+shift;
                }
                if (dm[m][j+2] != 0) {
                    cfs[k]    = dm[m][j+2];
                    dss[k++]  = j+2+shift;
                }
                if (dm[m][j+3] != 0) {
                    cfs[k]    = dm[m][j+3];
                    dss[k++]  = j+3+shift;
                }
            }

            /* store meta data in first entries */
            dts[COEFFS]   = l; /* position of coefficient array in tmpcf */
            dts[PRELOOP]  = k % UNROLL;
            dts[LENGTH]   = k;

            /* adjust memory usage */
            dts = realloc(dts, (unsigned long)(k+OFFSET) * sizeof(hm_t));
            cfs = realloc(cfs, (unsigned long)k * sizeof(cf16_t));

            /* link to basis */
            mat->tr[l]    = dts;
            mat->cf_16[l] = cfs;
            l++;
        }
    }
}

/* NOTE: this note is about the different linear algebra implementations:
 * exact and probabilistic linear algebra differ only in the last,
 * dense reduction step: the reduction of CD via AB is sparse and
 * the same for both. this generates a dense D' part which is then
 * either reduced via exact linear algebra or via probabilistic
 * linear algebra */
static void probabilistic_sparse_linear_algebra_ff_16(
        mat_t *mat,
        const bs_t * const tbr,
        const bs_t * const bs,
        md_t *st
        )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* allocate temporary storage space for sparse
     * coefficients of all pivot rows */
    mat->cf_16 = realloc(mat->cf_16,
            (unsigned long)mat->nr * sizeof(cf16_t *));
    probabilistic_sparse_reduced_echelon_form_ff_16(mat, bs, st);

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->la_ctime  +=  ct1 - ct0;
    st->la_rtime  +=  rt1 - rt0;

    st->num_zerored += (mat->nrl - mat->np);
    if (st->info_level > 1) {
        printf("%9d new %7d zero", mat->np, mat->nrl - mat->np);
        fflush(stdout);
    }
}
#endif
