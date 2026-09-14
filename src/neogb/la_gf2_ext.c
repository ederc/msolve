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
static inline void normalize_sparse_gf_16(
        cf2_ext_t * row,
        const hm_t len,
        const uint32_t fc
        )
{
    len_t j;

    const cf2_ext_t inv = gf16_inv(row[0]);
    const len_t os    = len % UNROLL;

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

static inline void normalize_sparse_gf_256(
        cf2_ext_t * row,
        const hm_t len,
        const uint32_t fc
        )
{
    len_t j;

    const cf2_ext_t inv = gf256_inv(row[0]);
    const len_t os    = len % UNROLL;

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
    const cf2_ext_t gfc         = (cf2_ext_t)fc;

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
        const cf2_ext_t mul= gfc - dr[i];
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
        for (j = 0; j < os; ++j) {
            dr[ds[j]] =  gf16_mul_add_table(dr[ds[j]], cfs[j], mul);
        }
        for (; j < len; j += UNROLL) {
            dr[ds[j]]   =  gf16_mul_add_table(dr[ds[j]], cfs[j], mul);
            dr[ds[j+1]] =  gf16_mul_add_table(dr[ds[j+1]], cfs[j+1], mul);
            dr[ds[j+2]] =  gf16_mul_add_table(dr[ds[j+2]], cfs[j+2], mul);
            dr[ds[j+3]] =  gf16_mul_add_table(dr[ds[j+3]], cfs[j+3], mul);
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
    const cf2_ext_t gfc         = (cf2_ext_t)fc;

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
        const cf2_ext_t mul= gfc - dr[i];
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
        for (j = 0; j < os; ++j) {
            dr[ds[j]] =  gf256_mul_add_table(dr[ds[j]], cfs[j], mul);
        }
        for (; j < len; j += UNROLL) {
            dr[ds[j]]   =  gf256_mul_add_table(dr[ds[j]], cfs[j], mul);
            dr[ds[j+1]] =  gf256_mul_add_table(dr[ds[j+1]], cfs[j+1], mul);
            dr[ds[j+2]] =  gf256_mul_add_table(dr[ds[j+2]], cfs[j+2], mul);
            dr[ds[j+3]] =  gf256_mul_add_table(dr[ds[j+3]], cfs[j+3], mul);
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
