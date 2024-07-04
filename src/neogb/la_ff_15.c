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

static inline cf16_t *normalize_sparse_matrix_row_ff_15(
        cf16_t *row,
        const len_t os,
        const len_t len,
        const uint32_t fc
        )
{
    len_t i;

    const uint16_t fc16  = (uint16_t)fc;
    const uint16_t inv   = mod_p_inverse_16(row[0], fc16);

    for (i = 0; i < os; ++i) {
        row[i]  = (cf16_t)(((uint32_t)row[i] * inv) % fc16);
    }
    /* we need to set i to os since os < 1 is possible */
    for (i = os; i < len; i += UNROLL) {
        row[i]   = (cf16_t)(((uint32_t)row[i] * inv) % fc16);
        row[i+1] = (cf16_t)(((uint32_t)row[i+1] * inv) % fc16);
        row[i+2] = (cf16_t)(((uint32_t)row[i+2] * inv) % fc16);
        row[i+3] = (cf16_t)(((uint32_t)row[i+3] * inv) % fc16);
    }
    row[0]  = 1;

    return row;
}

static hm_t *reduce_dense_row_by_known_pivots_sparse_ff_15(
        int32_t *dr,
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
    cf16_t *cfs;
    int64_t np = -1;
    const int32_t mod           = (int32_t)fc;
    const int32_t mod2          = mod*mod;
    const len_t ncols           = mat->nc;
    const len_t ncl             = mat->ncl;
    cf16_t * const * const mcf  = mat->cf_16;

    rba_t *rba;
    if (tr > 0) {
        rba = mat->rba[tmp_pos];
    } else {
        rba = NULL;
    }
#if defined HAVE_AVX512_F
    uint32_t mone32   = (uint32_t)0xFFFFFFFF;
    uint16_t mone16   = (uint16_t)0xFFFF;
    uint32_t mone16h  = (uint32_t)0xFFFF0000;
    __m512i mask32    = _mm512_set1_epi64(mone32);
    __m512i mask16    = _mm512_set1_epi32(mone16);
    __m512i mask16h   = _mm512_set1_epi32(mone16h);

    int64_t res[8] __attribute__((aligned(64)));
    __m512i redv, mulv, prodh, prodl, prod, drv, resv;
#elif defined HAVE_AVX2
    uint32_t mone32   = (uint32_t)0xFFFFFFFF;
    uint16_t mone16   = (uint16_t)0xFFFF;
    uint32_t mone16h  = (uint32_t)0xFFFF0000;
    __m256i mask32    = _mm256_set1_epi64x(mone32);
    __m256i mask16    = _mm256_set1_epi32(mone16);
    __m256i mask16h   = _mm256_set1_epi32(mone16h);

    int64_t res[4] __attribute__((aligned(32)));
    __m256i redv, mulv, prodh, prodl, prod, drv, resv;
#elif defined __abarch64__
    const int32x4_t mod2v = vmovq_n_s32(mod2);
    uint32_t tmp[4] __attribute__((aligned(32)));
    uint16x8_t redv;
    uint32x4_t drv, mask, resv;
#endif

    k = 0;
    for (i = dpiv; i < ncols; ++i) {
        if (dr[i] != 0) {
            dr[i] = dr[i] % mod;
        }
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
        if (i < ncl) {
            /* set corresponding bit of reducer in reducer bit array */
            if (tr > 0) {
                rba[i/32] |= 1U << (i % 32);
            }
        }
        /* found reducer row, get multiplier */
        const int32_t mul = (int32_t)(dr[i]);
        dts   = pivs[i];
        cfs   = mcf[dts[COEFFS]];
#if defined __abarch64__
        const len_t len       = dts[LENGTH];
        const len_t os        = len % 8;
        const hm_t * const ds = dts + OFFSET;
        const int16_t mul16   = (int16_t)(dr[i]);
        const int16x4_t mulv  = vmov_n_s16(mul16);
        for (j = 0; j < os; ++j) {
            dr[ds[j]] -=  mul * cfs[j];
            dr[ds[j]] +=  (dr[ds[j]] >> 31) & mod2;
        }
        for (; j < len; j += 8) {
            redv = vld1q_s16((int16_t *)(cfs)+j);
            tmp[0] = dr[ds[j]];
            tmp[1] = dr[ds[j+1]];
            tmp[2] = dr[ds[j+2]];
            tmp[3] = dr[ds[j+3]];
            drv  = vld1q_s32(tmp);
            /* multiply and subtract */
            resv = vmlsl_s16(drv, vget_low_s16(redv), mulv);
            mask = vreinterpretq_s32_u32(vcltzq_s32(resv));
            resv = vaddq_s32(resv, vandq_s32(mask, mod2v));
            vst1q_s32(tmp, resv);
            dr[ds[j]]   = tmp[0];
            dr[ds[j+1]] = tmp[1];
            dr[ds[j+2]] = tmp[2];
            dr[ds[j+3]] = tmp[3];
            tmp[0] = dr[ds[j+4]];
            tmp[1] = dr[ds[j+5]];
            tmp[2] = dr[ds[j+6]];
            tmp[3] = dr[ds[j+7]];
            drv  = vld1q_s32(tmp);
            resv = vmlsl_s16(drv, vget_high_s16(redv), mulv);
            mask = vreinterpretq_s32_u32(vcltzq_s32(resv));
            resv = vaddq_s32(resv, vandq_s32(mask, mod2v));
            vst1q_s32(tmp, resv);
            dr[ds[j+4]] = tmp[0];
            dr[ds[j+5]] = tmp[1];
            dr[ds[j+6]] = tmp[2];
            dr[ds[j+7]] = tmp[3];
        }

#else
        const len_t os  = dts[PRELOOP];
        const len_t len = dts[LENGTH];
        const hm_t * const ds  = dts + OFFSET;
        for (j = 0; j < os; ++j) {
            dr[ds[j]] -=  mul * cfs[j];
            dr[ds[j]]   +=  (dr[ds[j]] >> 31) & mod2;
        }
        for (; j < len; j += UNROLL) {
            dr[ds[j]]   -=  mul * cfs[j];
            dr[ds[j+1]] -=  mul * cfs[j+1];
            dr[ds[j+2]] -=  mul * cfs[j+2];
            dr[ds[j+3]] -=  mul * cfs[j+3];
            dr[ds[j]]   +=  (dr[ds[j]] >> 31) & mod2;
            dr[ds[j+1]] +=  (dr[ds[j+1]] >> 31) & mod2;
            dr[ds[j+2]] +=  (dr[ds[j+2]] >> 31) & mod2;
            dr[ds[j+3]] +=  (dr[ds[j+3]] >> 31) & mod2;
        }
#endif
        dr[i] = 0;
    }
    if (k == 0) {
        return NULL;
    }

    hm_t *row   = (hm_t *)malloc((unsigned long)(k+OFFSET) * sizeof(hm_t));
    cf16_t *cf  = (cf16_t *)malloc((unsigned long)(k) * sizeof(cf16_t));
    j = 0;
    hm_t *rs = row + OFFSET;
    for (i = ncl; i < ncols; ++i) {
        if (dr[i] != 0) {
            rs[j] = (hm_t)i;
            cf[j] = (cf16_t)dr[i];
            j++;
        }
    }
    row[COEFFS]   = tmp_pos;
    row[PRELOOP]  = j % UNROLL;
    row[LENGTH]   = j;
    mat->cf_16[tmp_pos]  = cf;

    return row;
}

static void exact_sparse_reduced_echelon_form_ff_15(
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
        mat->cf_16[j]      = bs->cf_16[mat->rr[i][COEFFS]];
        mat->rr[i][COEFFS] = j;
        ++j;
    }


    /* unkown pivot rows we have to reduce with the known pivots first */
    hm_t **upivs  = mat->tr;

    int32_t *dr  = (int32_t *)malloc(
            (unsigned long)(nthrds * ncols) * sizeof(int32_t));
    /* mo need to have any sharing dependencies on parallel computation,
     * no data to be synchronized at this step of the linear algebra */
#pragma omp parallel for num_threads(nthrds) \
    private(i, j, k, sc) \
    schedule(dynamic)
    for (i = 0; i < nrl; ++i) {
        if (bad_prime == 0) {
            int32_t *drl    = dr + (omp_get_thread_num() * ncols);
            hm_t *npiv      = upivs[i];
            cf16_t *cfs     = tbr->cf_16[npiv[COEFFS]];
            const len_t bi  = npiv[BINDEX];
            const len_t mh  = npiv[MULT];
            const len_t os  = npiv[PRELOOP];
            const len_t len = npiv[LENGTH];
            const hm_t * const ds = npiv + OFFSET;
            k = 0;
            memset(drl, 0, (unsigned long)ncols * sizeof(int32_t));
            for (j = 0; j < os; ++j) {
                drl[ds[j]]  = (int32_t)cfs[j];
            }
            for (; j < len; j += UNROLL) {
                drl[ds[j]]    = (int32_t)cfs[j];
                drl[ds[j+1]]  = (int32_t)cfs[j+1];
                drl[ds[j+2]]  = (int32_t)cfs[j+2];
                drl[ds[j+3]]  = (int32_t)cfs[j+3];
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
                npiv  = mat->tr[i] = reduce_dense_row_by_known_pivots_sparse_ff_15(
                        drl, mat, bs, pivs, sc, i, mh, bi, st->trace_level == LEARN_TRACER, st->fc);
                if (st->nf > 0) {
                    if (!npiv) {
                        mat->tr[i]  = NULL;
                        break;
                    }
                    mat->tr[i]  = npiv;
                    cfs = mat->cf_16[npiv[COEFFS]];
                    break;
                } else {
                    if (!npiv) {
                        break;
                    }
                    /* normalize coefficient array
                     * NOTE: this has to be done here, otherwise the reduction may
                     * lead to wrong results in a parallel computation since other
                     * threads might directly use the new pivot once it is synced. */
                    if (mat->cf_16[npiv[COEFFS]][0] != 1) {
                        normalize_sparse_matrix_row_ff_15(
                                mat->cf_16[npiv[COEFFS]], npiv[PRELOOP], npiv[LENGTH], st->fc);
                    }
                    k   = __sync_bool_compare_and_swap(&pivs[npiv[OFFSET]], NULL, npiv);
                    cfs = mat->cf_16[npiv[COEFFS]];
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
        dr      = realloc(dr, (unsigned long)ncols * sizeof(int32_t));
        mat->tr = realloc(mat->tr, (unsigned long)ncr * sizeof(hm_t *));

        /* interreduce new pivots */
        cf16_t *cfs;
        hm_t cf_array_pos;
        for (i = 0; i < ncr; ++i) {
            k = ncols-1-i;
            if (pivs[k]) {
                memset(dr, 0, (unsigned long)ncols * sizeof(int32_t));
                cfs = mat->cf_16[pivs[k][COEFFS]];
                cf_array_pos    = pivs[k][COEFFS];
                const len_t bi  = pivs[k][BINDEX];
                const len_t mh  = pivs[k][MULT];
                const len_t os  = pivs[k][PRELOOP];
                const len_t len = pivs[k][LENGTH];
                const hm_t * const ds = pivs[k] + OFFSET;
                sc  = ds[0];
                for (j = 0; j < os; ++j) {
                    dr[ds[j]] = (int32_t)cfs[j];
                }
                for (; j < len; j += UNROLL) {
                    dr[ds[j]]    = (int32_t)cfs[j];
                    dr[ds[j+1]]  = (int32_t)cfs[j+1];
                    dr[ds[j+2]]  = (int32_t)cfs[j+2];
                    dr[ds[j+3]]  = (int32_t)cfs[j+3];
                }
                free(pivs[k]);
                free(cfs);
                pivs[k] = NULL;
                pivs[k] = mat->tr[npivs++] =
                    reduce_dense_row_by_known_pivots_sparse_ff_15(
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
static void exact_sparse_linear_algebra_ff_15(
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
    mat->cf_16  = realloc(mat->cf_16,
            (unsigned long)mat->nr * sizeof(cf16_t *));
    exact_sparse_reduced_echelon_form_ff_15(mat, tbr, bs, st);

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

static void interreduce_matrix_rows_ff_15(
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

    mat->cf_16  = realloc(mat->cf_16,
            (unsigned long)ncols * sizeof(cf16_t *));
    memset(mat->cf_16, 0, (unsigned long)ncols * sizeof(cf16_t *));
    hm_t **pivs = (hm_t **)calloc((unsigned long)ncols, sizeof(hm_t *));
    /* copy coefficient arrays from basis in matrix, maybe
     * several rows need the same coefficient arrays, but we
     * cannot share them here. */
    for (i = 0; i < nrows; ++i) {
        pivs[mat->rr[i][OFFSET]]  = mat->rr[i];
    }

    int32_t *dr = (int32_t *)malloc((unsigned long)ncols * sizeof(int32_t));
    /* interreduce new pivots */
    cf16_t *cfs;
    /* starting column, coefficient array position in tmpcf */
    hm_t sc;
    k = nrows - 1;
    for (i = 0; i < ncols; ++i) {
        l = ncols-1-i;
        if (pivs[l] != NULL) {
            memset(dr, 0, (unsigned long)ncols * sizeof(int32_t));
            cfs = bs->cf_16[pivs[l][COEFFS]];
            const len_t bi  = pivs[l][BINDEX];
            const len_t mh  = pivs[l][MULT];
            const len_t os  = pivs[l][PRELOOP];
            const len_t len = pivs[l][LENGTH];
            const hm_t * const ds = pivs[l] + OFFSET;
            sc  = ds[0];
            for (j = 0; j < os; ++j) {
                dr[ds[j]] = (int32_t)cfs[j];
            }
            for (; j < len; j += UNROLL) {
                dr[ds[j]]   = (int32_t)cfs[j];
                dr[ds[j+1]] = (int32_t)cfs[j+1];
                dr[ds[j+2]] = (int32_t)cfs[j+2];
                dr[ds[j+3]] = (int32_t)cfs[j+3];
            }
            free(pivs[l]);
            pivs[l] = NULL;
            pivs[l] = mat->tr[k--] =
                reduce_dense_row_by_known_pivots_sparse_ff_15(
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
