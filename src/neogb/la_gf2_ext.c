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
 * Vincent Neiger
 * Mohab Safey El Din */

#include "data.h"
#include "../msolve/streams.h"
#include "gf2_ext.h"

/* That's also enough if AVX512 is avaialable on the system */
#if defined HAVE_AVX2
#include <immintrin.h>
#elif defined __aarch64__
#include <arm_neon.h>
#endif

// Normalisation of sparse rows
static inline void normalize_sparse_row_gf_16(
        cf2_ext_t * row,
        const len_t os,
        const hm_t len,
        const uint32_t fc
        )
{
    len_t j;

    const cf2_ext_t inv = gf16_inv(row[0]);

    for (j = 0; j < os; ++j) {
        row[j]  =   gf16_mul(row[j], inv);
    }
    for (j = os; j < len; j += UNROLL) {
        row[j]    =   gf16_mul(row[j], inv);
        row[j+1]  =   gf16_mul(row[j+1], inv);
        row[j+2]  =   gf16_mul(row[j+2], inv);
        row[j+3]  =   gf16_mul(row[j+3], inv);
    }
}

static inline void normalize_sparse_row_gf_256(
        cf2_ext_t * row,
        const len_t os,
        const hm_t len,
        const uint32_t fc
        )
{
    len_t j;

    const cf2_ext_t inv = gf256_inv(row[0]);

    for (j = 0; j < os; ++j) {
        row[j]  =   gf256_mul(row[j], inv);
    }
    for (j = os; j < len; j += UNROLL) {
        row[j]    =   gf256_mul(row[j], inv);
        row[j+1]  =   gf256_mul(row[j+1], inv);
        row[j+2]  =   gf256_mul(row[j+2], inv);
        row[j+3]  =   gf256_mul(row[j+3], inv);
    }
}

// Dense row by sparse pivot reduction
static hm_t *reduce_dense_row_by_known_pivots_sparse_gf_16(
        cf2_ext_t *dr,
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
    cf2_ext_t *cfs;
    int64_t np = -1;
    const len_t ncols           = mat->nc;
    const len_t ncl             = mat->ncl;
    cf2_ext_t * const * const mcf  = mat->cf2_ext;

// #if defined HAVE_AVX512_F
//     __m512i mask1 = _mm512_set1_epi64(0x000000000000FFFF);
//     __m512i mask2 = _mm512_set1_epi64(0x00000000FFFF0000);
//     __m512i mask3 = _mm512_set1_epi64(0x0000FFFF00000000);
//     __m512i mask4 = _mm512_set1_epi64(0xFFFF000000000000);
//     __m512i mask8 = _mm512_set1_epi16(0x00FF);
//     int64_t res[8] __attribute__((aligned(64)));
//     __m512i redv, mulv, prod, drv, resv;
// #elif defined HAVE_AVX2
//     __m256i mask1 = _mm256_set1_epi64x(0x000000000000FFFF);
//     __m256i mask2 = _mm256_set1_epi64x(0x00000000FFFF0000);
//     __m256i mask3 = _mm256_set1_epi64x(0x0000FFFF00000000);
//     __m256i mask4 = _mm256_set1_epi64x(0xFFFF000000000000);
//     __m256i mask8 = _mm256_set1_epi16(0x00FF);
//     int64_t res[4] __attribute__((aligned(32)));
//     __m256i redv, mulv, prod, drv, resv;
// #elif defined __aarch64__
//     uint64_t tmp[2] __attribute__((aligned(32)));
//     uint16x8_t prodv;
//     uint32x4_t prodvl, prodvh;
//     uint8x16_t redv;
//     uint64x2_t drv, resv;
// #endif

    rba_t *rba;
    if (tr > 0) {
        rba = mat->rba[tmp_pos];
    } else {
        rba = NULL;
    }
    k = 0;
    for (i = dpiv; i < ncols; ++i) {
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
        dts   = pivs[i];
        if (i < ncl) {
            /* set corresponding bit of reducer in reducer bit array */
            if (tr > 0) {
                rba[i/32] |= 1U << (i % 32);
            }
        }
        cfs   = mcf[dts[COEFFS]];
// #if defined HAVE_AVX512_F
// #elif defined HAVE_AVX2
// #elif defined __aarch64__
// #else
        const len_t os  = dts[PRELOOP];
        const len_t len = dts[LENGTH];
        const hm_t * const ds  = dts + OFFSET;
        const cf2_ext_t *restrict mul = gf16_mul_row(dr[i]);
        for (j = 0; j < os; ++j) {
            dr[ds[j]] ^= mul[cfs[j]];
        }
        for (; j < len; j += UNROLL) {
            dr[ds[j]] ^= mul[cfs[j]];
            dr[ds[j+1]] ^= mul[cfs[j+1]];
            dr[ds[j+2]] ^= mul[cfs[j+2]];
            dr[ds[j+3]] ^= mul[cfs[j+3]];
        }
// #endif
        dr[i] = 0;
    }
    if (k == 0) {
        return NULL;
    }
    hm_t *row   = (hm_t *)malloc((uint64_t)(k+OFFSET) * sizeof(hm_t));
    cf2_ext_t *cf  = (cf2_ext_t *)malloc((uint64_t)(k) * sizeof(cf2_ext_t));
    j = 0;
    hm_t *rs = row + OFFSET;
    for (i = ncl; i < ncols; ++i) {
        if (dr[i] != 0) {
            rs[j] = (hm_t)i;
            cf[j] = dr[i];
            j++;
        }
    }
    row[BINDEX]   = bi;
    row[MULT]     = mh;
    row[COEFFS]   = tmp_pos;
    row[PRELOOP]  = j % UNROLL;
    row[LENGTH]   = j;
    mat->cf2_ext[tmp_pos]  = cf;

    return row;
}
        
static hm_t *reduce_dense_row_by_known_pivots_sparse_gf_256(
        cf2_ext_t *dr,
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
    cf2_ext_t *cfs;
    int64_t np = -1;
    const len_t ncols           = mat->nc;
    const len_t ncl             = mat->ncl;
    cf2_ext_t * const * const mcf  = mat->cf2_ext;

// #if defined HAVE_AVX512_F
//     __m512i mask1 = _mm512_set1_epi64(0x000000000000FFFF);
//     __m512i mask2 = _mm512_set1_epi64(0x00000000FFFF0000);
//     __m512i mask3 = _mm512_set1_epi64(0x0000FFFF00000000);
//     __m512i mask4 = _mm512_set1_epi64(0xFFFF000000000000);
//     __m512i mask8 = _mm512_set1_epi16(0x00FF);
//     int64_t res[8] __attribute__((aligned(64)));
//     __m512i redv, mulv, prod, drv, resv;
// #elif defined HAVE_AVX2
//     __m256i mask1 = _mm256_set1_epi64x(0x000000000000FFFF);
//     __m256i mask2 = _mm256_set1_epi64x(0x00000000FFFF0000);
//     __m256i mask3 = _mm256_set1_epi64x(0x0000FFFF00000000);
//     __m256i mask4 = _mm256_set1_epi64x(0xFFFF000000000000);
//     __m256i mask8 = _mm256_set1_epi16(0x00FF);
//     int64_t res[4] __attribute__((aligned(32)));
//     __m256i redv, mulv, prod, drv, resv;
// #elif defined __aarch64__
//     uint64_t tmp[2] __attribute__((aligned(32)));
//     uint16x8_t prodv;
//     uint32x4_t prodvl, prodvh;
//     uint8x16_t redv;
//     uint64x2_t drv, resv;
// #endif

    rba_t *rba;
    if (tr > 0) {
        rba = mat->rba[tmp_pos];
    } else {
        rba = NULL;
    }
    k = 0;
    for (i = dpiv; i < ncols; ++i) {
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
        // const cf2_ext_t mul= dr[i];
        dts   = pivs[i];
        if (i < ncl) {
            /* set corresponding bit of reducer in reducer bit array */
            if (tr > 0) {
                rba[i/32] |= 1U << (i % 32);
            }
        }
        cfs   = mcf[dts[COEFFS]];
// #if defined HAVE_AVX512_F
// #elif defined HAVE_AVX2
// #elif defined __aarch64__
// #else
        const len_t os  = dts[PRELOOP];
        const len_t len = dts[LENGTH];
        const hm_t * const ds  = dts + OFFSET;
        const cf2_ext_t *restrict mul = gf256_mul_row(dr[i]);
        for (j = 0; j < os; ++j) {
            dr[ds[j]] ^= mul[cfs[j]];
        }
        for (; j < len; j += UNROLL) {
            dr[ds[j]] ^= mul[cfs[j]];
            dr[ds[j+1]] ^= mul[cfs[j+1]];
            dr[ds[j+2]] ^= mul[cfs[j+2]];
            dr[ds[j+3]] ^= mul[cfs[j+3]];
        }
// #endif
        dr[i] = 0;
    }
    if (k == 0) {
        return NULL;
    }
    hm_t *row   = (hm_t *)malloc((uint64_t)(k+OFFSET) * sizeof(hm_t));
    cf2_ext_t *cf  = (cf2_ext_t *)malloc((uint64_t)(k) * sizeof(cf2_ext_t));
    j = 0;
    hm_t *rs = row + OFFSET;
    for (i = ncl; i < ncols; ++i) {
        if (dr[i] != 0) {
            rs[j] = (hm_t)i;
            cf[j] = dr[i];
            j++;
        }
    }
    row[BINDEX]   = bi;
    row[MULT]     = mh;
    row[COEFFS]   = tmp_pos;
    row[PRELOOP]  = j % UNROLL;
    row[LENGTH]   = j;
    mat->cf2_ext[tmp_pos]  = cf;

    return row;
}
        


// Final rows interreduction
static void interreduce_matrix_rows_gf_16(
        mat_t *mat,
        bs_t *bs,
        md_t *st,
        int free_basis
        )
{
    len_t i, j, k, l;

    const len_t nrows = mat->nr;
    const len_t ncols = mat->nc;

    /* adjust displaying timings for statistic printout */
    if (st->info_level > 1) {
        fprintf(VERBSTREAM, "                          ");
    }

    /* for interreduction steps like the final basis reduction we
    need to allocate memory for rba here, even so we do not use
    it at all */
    mat->rba  = (rba_t **)malloc((uint64_t)ncols * sizeof(rba_t *));
    const uint64_t len = ncols / 32 + ((ncols % 32) != 0);
    for (i = 0; i < ncols; ++i) {
        mat->rba[i] = (rba_t *)calloc(len, sizeof(rba_t));
    }

    mat->tr = realloc(mat->tr, (uint64_t)ncols * sizeof(hm_t *));

    mat->cf2_ext  = realloc(mat->cf2_ext,
            (uint64_t)ncols * sizeof(cf2_ext_t *));
    memset(mat->cf2_ext, 0, (uint64_t)ncols * sizeof(cf2_ext_t *));
    hm_t **pivs = (hm_t **)calloc((uint64_t)ncols, sizeof(hm_t *));
    /* copy coefficient arrays from basis in matrix, maybe
     * several rows need the same coefficient arrays, but we
     * cannot share them here. */
    for (i = 0; i < nrows; ++i) {
        pivs[mat->rr[i][OFFSET]]  = mat->rr[i];
    }

    cf2_ext_t *dr = (cf2_ext_t *)malloc((uint64_t)ncols * sizeof(cf2_ext_t));
    /* interreduce new pivots */
    cf2_ext_t *cfs;
    /* starting column, coefficient array position in tmpcf */
    hm_t sc;
    k = nrows - 1;
    for (i = 0; i < ncols; ++i) {
        l = ncols-1-i;
        if (pivs[l] != NULL) {
            memset(dr, 0, (uint64_t)ncols * sizeof(cf2_ext_t));
            cfs = bs->cf2_ext[pivs[l][COEFFS]];
            const len_t bi  = pivs[l][BINDEX];
            const len_t mh  = pivs[l][MULT];
            const len_t os  = pivs[l][PRELOOP];
            const len_t len = pivs[l][LENGTH];
            const hm_t * const ds = pivs[l] + OFFSET;
            sc  = ds[0];
            for (j = 0; j < os; ++j) {
                dr[ds[j]] = cfs[j];
            }
            for (; j < len; j += UNROLL) {
                dr[ds[j]]   = cfs[j];
                dr[ds[j+1]] = cfs[j+1];
                dr[ds[j+2]] = cfs[j+2];
                dr[ds[j+3]] = cfs[j+3];
            }
            free(pivs[l]);
            pivs[l] = NULL;
            pivs[l] = mat->tr[k--] =
                reduce_dense_row_by_known_pivots_sparse_gf_16(
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

static void interreduce_matrix_rows_gf_256(
        mat_t *mat,
        bs_t *bs,
        md_t *st,
        int free_basis
        )
{
    len_t i, j, k, l;

    const len_t nrows = mat->nr;
    const len_t ncols = mat->nc;

    /* adjust displaying timings for statistic printout */
    if (st->info_level > 1) {
        fprintf(VERBSTREAM, "                          ");
    }

    /* for interreduction steps like the final basis reduction we
    need to allocate memory for rba here, even so we do not use
    it at all */
    mat->rba  = (rba_t **)malloc((uint64_t)ncols * sizeof(rba_t *));
    const uint64_t len = ncols / 32 + ((ncols % 32) != 0);
    for (i = 0; i < ncols; ++i) {
        mat->rba[i] = (rba_t *)calloc(len, sizeof(rba_t));
    }

    mat->tr = realloc(mat->tr, (uint64_t)ncols * sizeof(hm_t *));

    mat->cf2_ext  = realloc(mat->cf2_ext,
            (uint64_t)ncols * sizeof(cf2_ext_t *));
    memset(mat->cf2_ext, 0, (uint64_t)ncols * sizeof(cf2_ext_t *));
    hm_t **pivs = (hm_t **)calloc((uint64_t)ncols, sizeof(hm_t *));
    /* copy coefficient arrays from basis in matrix, maybe
     * several rows need the same coefficient arrays, but we
     * cannot share them here. */
    for (i = 0; i < nrows; ++i) {
        pivs[mat->rr[i][OFFSET]]  = mat->rr[i];
    }

    cf2_ext_t *dr = (cf2_ext_t *)malloc((uint64_t)ncols * sizeof(cf2_ext_t));
    /* interreduce new pivots */
    cf2_ext_t *cfs;
    /* starting column, coefficient array position in tmpcf */
    hm_t sc;
    k = nrows - 1;
    for (i = 0; i < ncols; ++i) {
        l = ncols-1-i;
        if (pivs[l] != NULL) {
            memset(dr, 0, (uint64_t)ncols * sizeof(cf2_ext_t));
            cfs = bs->cf2_ext[pivs[l][COEFFS]];
            const len_t bi  = pivs[l][BINDEX];
            const len_t mh  = pivs[l][MULT];
            const len_t os  = pivs[l][PRELOOP];
            const len_t len = pivs[l][LENGTH];
            const hm_t * const ds = pivs[l] + OFFSET;
            sc  = ds[0];
            for (j = 0; j < os; ++j) {
                dr[ds[j]] = cfs[j];
            }
            for (; j < len; j += UNROLL) {
                dr[ds[j]]   = cfs[j];
                dr[ds[j+1]] = cfs[j+1];
                dr[ds[j+2]] = cfs[j+2];
                dr[ds[j+3]] = cfs[j+3];
            }
            free(pivs[l]);
            pivs[l] = NULL;
            pivs[l] = mat->tr[k--] =
                reduce_dense_row_by_known_pivots_sparse_gf_256(
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

// exact sparse reduced echelon form
static void exact_sparse_reduced_echelon_form_gf_16(
        mat_t *mat,
        const bs_t * const tbr,
        const bs_t * const bs,
        md_t *st
        )
{
    len_t i = 0, j, k;
    hi_t sc = 0;    /* starting column */

    const len_t ncols = mat->nc;
    const len_t nrl   = mat->nrl;
    const len_t ncr   = mat->ncr;
    const len_t ncl   = mat->ncl;

    const int32_t nthrds = st->in_final_reduction_step == 1 ? 1 : st->nthrds;

    len_t bad_prime = 0;

    /* we fill in all known lead terms in pivs */
    hm_t **pivs   = (hm_t **)calloc((uint64_t)ncols, sizeof(hm_t *));
    if (st->in_final_reduction_step == 0) {
        memcpy(pivs, mat->rr, (uint64_t)mat->nru * sizeof(hm_t *));
    } else {
        for (i = 0;  i < mat->nru; ++i) {
            pivs[mat->rr[i][OFFSET]] = mat->rr[i];
        }
    }
    j = nrl;
    for (i = 0; i < mat->nru; ++i) {
        mat->cf2_ext[j]      = bs->cf2_ext[mat->rr[i][COEFFS]];
        mat->rr[i][COEFFS] = j;
        ++j;
    }

    /* unkown pivot rows we have to reduce with the known pivots first */
    hm_t **upivs  = mat->tr;

    cf2_ext_t *dr  = (cf2_ext_t *)malloc(
            (uint64_t)ncols * nthrds * sizeof(cf2_ext_t));
    /* mo need to have any sharing dependencies on parallel computation,
     * no data to be synchronized at this step of the linear algebra */
#pragma omp parallel for num_threads(nthrds) \
    private(i, j, k, sc) \
    schedule(dynamic)
    for (i = 0; i < nrl; ++i) {
        if (bad_prime == 0) {
            cf2_ext_t *drl  = dr + (omp_get_thread_num() * (uint64_t)ncols);
            hm_t *npiv      = upivs[i];
            cf2_ext_t *cfs      = tbr->cf2_ext[npiv[COEFFS]];
            const len_t os  = npiv[PRELOOP];
            const len_t len = npiv[LENGTH];
            const len_t bi  = npiv[BINDEX];
            const len_t mh  = npiv[MULT];
            const hm_t * const ds = npiv + OFFSET;
            k = 0;
            memset(drl, 0, (uint64_t)ncols * sizeof(cf2_ext_t));
            for (j = 0; j < os; ++j) {
                drl[ds[j]]  = cfs[j];
            }
            for (; j < len; j += UNROLL) {
                drl[ds[j]]    = cfs[j];
                drl[ds[j+1]]  = cfs[j+1];
                drl[ds[j+2]]  = cfs[j+2];
                drl[ds[j+3]]  = cfs[j+3];
            }
            cfs = NULL;
            do {
                /* If we do normal form computations the first monomial in the polynomial might not
                be a known pivot, thus setting it to npiv[OFFSET] can lead to wrong results. */
                sc  = st->nf == 0 ? npiv[OFFSET] : 0;
                free(npiv);
                free(cfs);
                npiv  = mat->tr[i] = reduce_dense_row_by_known_pivots_sparse_gf_16(
                        drl, mat, bs, pivs, sc, i, mh, bi, st->trace_level == LEARN_TRACER, st->fc);
                if (st->nf > 0) {
                    if (!npiv) {
                        mat->tr[i]  = NULL;
                        break;
                    }
                    mat->tr[i]  = npiv;
                    cfs = mat->cf2_ext[npiv[COEFFS]];
                    break;
                } else {
                    if (!npiv) {
                        if (st->trace_level == APPLY_TRACER) {
                            bad_prime = 1;
                        }
                        break;
                    }
                    /* normalize coefficient array
                     * NOTE: this has to be done here, otherwise the reduction may
                     * lead to wrong results in a parallel computation since other
                     * threads might directly use the new pivot once it is synced. */
                    if (mat->cf2_ext[npiv[COEFFS]][0] != 1) {
                        normalize_sparse_row_gf_16(
                                mat->cf2_ext[npiv[COEFFS]], npiv[PRELOOP], npiv[LENGTH], st->fc);
                    }
                    k   = __sync_bool_compare_and_swap(&pivs[npiv[OFFSET]], NULL, npiv);
                    cfs = mat->cf2_ext[npiv[COEFFS]];
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
            fprintf(ERRSTREAM, "Zero reduction while applying tracer, bad prime.\n");
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
        dr      = realloc(dr, (uint64_t)ncols * sizeof(int64_t));
        mat->tr = realloc(mat->tr, (uint64_t)ncr * sizeof(hm_t *));

        /* interreduce new pivots */
        cf2_ext_t *cfs;
        hm_t cf_array_pos;
        for (i = 0; i < ncr; ++i) {
            k = ncols-1-i;
            if (pivs[k]) {
                memset(dr, 0, (uint64_t)ncols * sizeof(cf2_ext_t));
                cfs = mat->cf2_ext[pivs[k][COEFFS]];
                cf_array_pos    = pivs[k][COEFFS];
                const len_t os  = pivs[k][PRELOOP];
                const len_t len = pivs[k][LENGTH];
                const len_t bi  = pivs[k][BINDEX];
                const len_t mh  = pivs[k][MULT];
                const hm_t * const ds = pivs[k] + OFFSET;
                sc  = ds[0];
                for (j = 0; j < os; ++j) {
                    dr[ds[j]] = cfs[j];
                }
                for (; j < len; j += UNROLL) {
                    dr[ds[j]]    = cfs[j];
                    dr[ds[j+1]]  = cfs[j+1];
                    dr[ds[j+2]]  = cfs[j+2];
                    dr[ds[j+3]]  = cfs[j+3];
                }
                free(pivs[k]);
                free(cfs);
                pivs[k] = NULL;
                pivs[k] = mat->tr[npivs++] =
                    reduce_dense_row_by_known_pivots_sparse_gf_16(
                        dr, mat, bs, pivs, sc, cf_array_pos, mh, bi, 0, st->fc);
            }
        }
        mat->tr = realloc(mat->tr, (uint64_t)npivs * sizeof(hi_t *));
        st->np = mat->np = mat->nr = mat->sz = npivs;
    } else {
        st->np = mat->np = mat->nr = mat->sz = nrl;
    }
    free(pivs);
    pivs  = NULL;
    free(dr);
    dr  = NULL;
}

static void exact_sparse_reduced_echelon_form_gf_256(
        mat_t *mat,
        const bs_t * const tbr,
        const bs_t * const bs,
        md_t *st
        )
{
    len_t i = 0, j, k;
    hi_t sc = 0;    /* starting column */

    const len_t ncols = mat->nc;
    const len_t nrl   = mat->nrl;
    const len_t ncr   = mat->ncr;
    const len_t ncl   = mat->ncl;

    const int32_t nthrds = st->in_final_reduction_step == 1 ? 1 : st->nthrds;

    len_t bad_prime = 0;

    /* we fill in all known lead terms in pivs */
    hm_t **pivs   = (hm_t **)calloc((uint64_t)ncols, sizeof(hm_t *));
    if (st->in_final_reduction_step == 0) {
        memcpy(pivs, mat->rr, (uint64_t)mat->nru * sizeof(hm_t *));
    } else {
        for (i = 0;  i < mat->nru; ++i) {
            pivs[mat->rr[i][OFFSET]] = mat->rr[i];
        }
    }
    j = nrl;
    for (i = 0; i < mat->nru; ++i) {
        mat->cf2_ext[j]      = bs->cf2_ext[mat->rr[i][COEFFS]];
        mat->rr[i][COEFFS] = j;
        ++j;
    }

    /* unkown pivot rows we have to reduce with the known pivots first */
    hm_t **upivs  = mat->tr;

    cf2_ext_t *dr  = (cf2_ext_t *)malloc(
            (uint64_t)ncols * nthrds * sizeof(cf2_ext_t));
    /* mo need to have any sharing dependencies on parallel computation,
     * no data to be synchronized at this step of the linear algebra */
#pragma omp parallel for num_threads(nthrds) \
    private(i, j, k, sc) \
    schedule(dynamic)
    for (i = 0; i < nrl; ++i) {
        if (bad_prime == 0) {
            cf2_ext_t *drl  = dr + (omp_get_thread_num() * (uint64_t)ncols);
            hm_t *npiv      = upivs[i];
            cf2_ext_t *cfs      = tbr->cf2_ext[npiv[COEFFS]];
            const len_t os  = npiv[PRELOOP];
            const len_t len = npiv[LENGTH];
            const len_t bi  = npiv[BINDEX];
            const len_t mh  = npiv[MULT];
            const hm_t * const ds = npiv + OFFSET;
            k = 0;
            memset(drl, 0, (uint64_t)ncols * sizeof(cf2_ext_t));
            for (j = 0; j < os; ++j) {
                drl[ds[j]]  = cfs[j];
            }
            for (; j < len; j += UNROLL) {
                drl[ds[j]]    = cfs[j];
                drl[ds[j+1]]  = cfs[j+1];
                drl[ds[j+2]]  = cfs[j+2];
                drl[ds[j+3]]  = cfs[j+3];
            }
            cfs = NULL;
            do {
                /* If we do normal form computations the first monomial in the polynomial might not
                be a known pivot, thus setting it to npiv[OFFSET] can lead to wrong results. */
                sc  = st->nf == 0 ? npiv[OFFSET] : 0;
                free(npiv);
                free(cfs);
                npiv  = mat->tr[i] = reduce_dense_row_by_known_pivots_sparse_gf_256(
                        drl, mat, bs, pivs, sc, i, mh, bi, st->trace_level == LEARN_TRACER, st->fc);
                if (st->nf > 0) {
                    if (!npiv) {
                        mat->tr[i]  = NULL;
                        break;
                    }
                    mat->tr[i]  = npiv;
                    cfs = mat->cf2_ext[npiv[COEFFS]];
                    break;
                } else {
                    if (!npiv) {
                        if (st->trace_level == APPLY_TRACER) {
                            bad_prime = 1;
                        }
                        break;
                    }
                    /* normalize coefficient array
                     * NOTE: this has to be done here, otherwise the reduction may
                     * lead to wrong results in a parallel computation since other
                     * threads might directly use the new pivot once it is synced. */
                    if (mat->cf2_ext[npiv[COEFFS]][0] != 1) {
                        normalize_sparse_row_gf_256(
                                mat->cf2_ext[npiv[COEFFS]], npiv[PRELOOP], npiv[LENGTH], st->fc);
                    }
                    k   = __sync_bool_compare_and_swap(&pivs[npiv[OFFSET]], NULL, npiv);
                    cfs = mat->cf2_ext[npiv[COEFFS]];
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
            fprintf(ERRSTREAM, "Zero reduction while applying tracer, bad prime.\n");
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
        dr      = realloc(dr, (uint64_t)ncols * sizeof(int64_t));
        mat->tr = realloc(mat->tr, (uint64_t)ncr * sizeof(hm_t *));

        /* interreduce new pivots */
        cf2_ext_t *cfs;
        hm_t cf_array_pos;
        for (i = 0; i < ncr; ++i) {
            k = ncols-1-i;
            if (pivs[k]) {
                memset(dr, 0, (uint64_t)ncols * sizeof(cf2_ext_t));
                cfs = mat->cf2_ext[pivs[k][COEFFS]];
                cf_array_pos    = pivs[k][COEFFS];
                const len_t os  = pivs[k][PRELOOP];
                const len_t len = pivs[k][LENGTH];
                const len_t bi  = pivs[k][BINDEX];
                const len_t mh  = pivs[k][MULT];
                const hm_t * const ds = pivs[k] + OFFSET;
                sc  = ds[0];
                for (j = 0; j < os; ++j) {
                    dr[ds[j]] = cfs[j];
                }
                for (; j < len; j += UNROLL) {
                    dr[ds[j]]    = cfs[j];
                    dr[ds[j+1]]  = cfs[j+1];
                    dr[ds[j+2]]  = cfs[j+2];
                    dr[ds[j+3]]  = cfs[j+3];
                }
                free(pivs[k]);
                free(cfs);
                pivs[k] = NULL;
                pivs[k] = mat->tr[npivs++] =
                    reduce_dense_row_by_known_pivots_sparse_gf_256(
                        dr, mat, bs, pivs, sc, cf_array_pos, mh, bi, 0, st->fc);
            }
        }
        mat->tr = realloc(mat->tr, (uint64_t)npivs * sizeof(hi_t *));
        st->np = mat->np = mat->nr = mat->sz = npivs;
    } else {
        st->np = mat->np = mat->nr = mat->sz = nrl;
    }
    free(pivs);
    pivs  = NULL;
    free(dr);
    dr  = NULL;
}

// exact sparse linear algebra
static void exact_sparse_linear_algebra_gf_16(
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
    mat->cf2_ext  = realloc(mat->cf2_ext,
            (uint64_t)mat->nr * sizeof(cf2_ext_t *));
    exact_sparse_reduced_echelon_form_gf_16(mat, tbr, bs, st);

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->la_ctime  +=  ct1 - ct0;
    st->la_rtime  +=  rt1 - rt0;

    st->num_zerored += (mat->nrl - mat->np);
    if (st->info_level > 1) {
        fprintf(VERBSTREAM, "%9d new %7d zero", mat->np, mat->nrl - mat->np);
        fflush(VERBSTREAM);
    }
}

static void exact_sparse_linear_algebra_gf_256(
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
    mat->cf2_ext  = realloc(mat->cf2_ext,
            (uint64_t)mat->nr * sizeof(cf2_ext_t *));
    exact_sparse_reduced_echelon_form_gf_256(mat, tbr, bs, st);

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->la_ctime  +=  ct1 - ct0;
    st->la_rtime  +=  rt1 - rt0;

    st->num_zerored += (mat->nrl - mat->np);
    if (st->info_level > 1) {
        fprintf(VERBSTREAM, "%9d new %7d zero", mat->np, mat->nrl - mat->np);
        fflush(VERBSTREAM);
    }
}
