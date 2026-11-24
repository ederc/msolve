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

#ifndef GB_TOOLS_H
#define GB_TOOLS_H

#include <time.h>
#include "data.h"

#if defined(__BMI2__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* cpu time */
double cputime(
    void
    );

/* wall time */
double realtime(
    void
    );

static inline uint8_t mod_p_inverse_8(
        const int16_t val,
        const int16_t p
        )
{
    int16_t a, b, c, d, e, f;
    a =   p;
    b =   val % p;
    /* if b < 0 we shift correspondingly */
    b +=  (b >> 15) & p;
    c =   1;
    d =   0;

    while (b != 0) {
        f = b;
        e = a/f;
        b = a - e*f;
        a = f;
        f = c;
        c = d - e*f;
        d = f;
    }

    /* if d < 0 we shift correspondingly */
    d +=  (d >> 15) & p;

    return d;
}

static inline uint16_t mod_p_inverse_16(
        const int32_t val,
        const int32_t p
        )
{
    int32_t a, b, c, d, e, f;
    a =   p;
    b =   val % p;
    /* if b < 0 we shift correspondingly */
    b +=  (b >> 31) & p;
    c =   1;
    d =   0;

    while (b != 0) {
        f = b;
        e = a/f;
        b = a - e*f;
        a = f;
        f = c;
        c = d - e*f;
        d = f;
    }

    /* if d < 0 we shift correspondingly */
    d +=  (d >> 31) & p;

    return (uint16_t)d;
}

/* Barrett reduction: returns x mod p for 0 <= x < 2^64, where
 * m = barrett_multiplier(p) = floor((2^64-1) / p). The quotient estimate
 * is at most one too small, so one correction step suffices. This is
 * much cheaper than a 64 bit division in the linear algebra kernels. */
/* the first column i >= from with dr[i] != 0, or to if there is none;
 * the dense rows of the linear algebra are mostly zero, so the blocks of
 * zero entries are skipped with SIMD compares */
static inline uint32_t next_nonzero_column(
        const int64_t * const dr,
        uint32_t from,
        const uint32_t to
        )
{
#if defined(__AVX512F__)
    while (from + 8 <= to) {
        const __mmask8 m = _mm512_test_epi64_mask(
                _mm512_loadu_si512((const void *)(dr + from)),
                _mm512_set1_epi64(-1));
        if (m != 0) {
            return from + (uint32_t)__builtin_ctz((unsigned)m);
        }
        from += 8;
    }
#elif defined(__AVX2__)
    while (from + 4 <= to) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(dr + from));
        const int m = _mm256_movemask_pd(_mm256_castsi256_pd(
                    _mm256_cmpeq_epi64(v, _mm256_setzero_si256())));
        if (m != 0xF) {
            return from + (uint32_t)__builtin_ctz((unsigned)(~m & 0xF));
        }
        from += 4;
    }
#endif
    while (from < to && dr[from] == 0) {
        from++;
    }
    return from;
}

static inline uint64_t barrett_multiplier(
        const uint64_t p
        )
{
    return UINT64_MAX / p;
}

static inline uint64_t barrett_reduce(
        const uint64_t x,
        const uint64_t p,
        const uint64_t m
        )
{
    const uint64_t q  = (uint64_t)(((unsigned __int128)x * m) >> 64);
    uint64_t r        = x - q * p;
    return r >= p ? r - p : r;
}

static inline uint32_t mod_p_inverse_32(
        const int64_t val,
        const int64_t p
        )
{
    int64_t a, b, c, d, e, f;
    a =   p;
    b =   val % p;
    /* if b < 0 we shift correspondingly */
    b +=  (b >> 63) & p;
    c =   1;
    d =   0;

    while (b != 0) {
        f = b;
        e = a/f;
        b = a - e*f;
        a = f;
        f = c;
        c = d - e*f;
        d = f;
    }

    /* if d < 0 we shift correspondingly */
    d +=  (d >> 63) & p;

    return d;
}

static inline uint64_t mod_p_inverse_64(
        const int64_t val,
        const int64_t p
        )
{
    int64_t a, b, c, d, e, f;
    a =   p;
    b =   val % p;
    /* if b < 0 we shift correspondingly */
    b +=  (b >> 63) & p;
    c =   1;
    d =   0;

    while (b != 0) {
        f = b;
        e = a/f;
        b = a - e*f;
        a = f;
        f = c;
        c = d - e*f;
        d = f;
    }

    /* if d < 0 we shift correspondingly */
    d +=  (d >> 63) & p;

    return (uint64_t)d;
}

/* a * b mod p for a, b < p < 2^63 */
static inline uint64_t mul_mod_64(
        const uint64_t a,
        const uint64_t b,
        const uint64_t p
        )
{
    return (uint64_t)(((unsigned __int128)a * b) % p);
}

/* Shoup's pre-inverse mu = floor(x * 2^64 / p) of a fixed multiplicand
 * x < p; mu < 2^64 since x < p. */
static inline uint64_t shoup_mu(
        const uint64_t p,
        const uint64_t x
        )
{
    return (uint64_t)(((unsigned __int128)x << 64) / p);
}

/* Shoup's quotient q = floor(mu * y / 2^64) for mu = shoup_mu(p, x)
 * and any y < 2^64: with wrapping 64-bit arithmetic x * y - q * p lies
 * in [0, 2p) for x < p < 2^63 and is x*y mod p or that value plus p. */
static inline uint64_t shoup_mul(
        const uint64_t mu,
        const uint64_t y
        )
{
    return (uint64_t)(((unsigned __int128)mu * y) >> 64);
}

/* the high word of mu * y; the BMI2 mulx instruction computes it in one
 * step without tying the mul accumulator pair, which measurably shortens
 * the dependency chain of the dense row subtraction below */
static inline uint64_t mulx_hi(
        const uint64_t mu,
        const uint64_t y
        )
{
#if defined(__BMI2__)
    unsigned long long hi;
    _mulx_u64(mu, y, &hi);
    return (uint64_t)hi;
#else
    return shoup_mul(mu, y);
#endif
}

/* subtracts v times the pivot row (pr, pf) of length pl, but its lead
 * entry, from the dense row dr; pf holds the pivot coefficients. The
 * Shoup pre-inverse psi = floor(v * 2^64 / p) of the multiplier v < p
 * is computed once per application, the per-element quotient is
 * mulhi(pf[j], psi) (harvey-2014 §3). Dense row entries are kept in
 * [0, 2p) as redundant representatives: the difference of two entries
 * in [0, 2p) is corrected by 2p on borrow, which is exact and
 * overflow-safe for all p < 2^62. */
static inline void sg_sub_row_64(
        uint64_t *dr,
        const hm_t * const pr,
        const cf64_t * const pf,
        const len_t pl,
        const uint64_t v,
        const uint64_t p
        )
{
    len_t j;
    const uint64_t twice = 2 * p;
    const uint64_t psi   = shoup_mu(p, v);
    for (j = 1; j < pl; ++j) {
        const uint64_t q = mulx_hi(psi, pf[j]);
        const uint64_t d = v * pf[j] - q * p; /* in [0, 2p) */
        const uint64_t e = dr[pr[j]];
        /* e - d plus 2p on borrow: the borrow generates the correction
         * mask, no conditional branch (harvey-2014 §2) */
        const uint64_t diff = e - d;
        const uint64_t brw  = e < d;
        dr[pr[j]] = diff + (((uint64_t)0 - brw) & twice);
    }
}
#endif
