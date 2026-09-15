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

#ifndef GF2EXT_H
#define GF2EXT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Representations:
 *
 * GF(2^4): x^4 + x + 1
 *   A nibble abc... represents polynomial bits in the usual way.
 *   Only the low four bits are significant.
 *
 * GF(2^8): x^8 + x^4 + x^3 + x + 1 (AES polynomial, 0x11B).
 */

/* Addition and subtraction are identical in characteristic two. */
static inline uint8_t gf16_add(uint8_t a, uint8_t b)
{
    return (uint8_t)((a ^ b) & 0x0Fu);
}

static inline uint8_t gf16_sub(uint8_t a, uint8_t b)
{
    return gf16_add(a, b);
}

static inline uint8_t gf256_add(uint8_t a, uint8_t b)
{
    return (uint8_t)(a ^ b);
}

static inline uint8_t gf256_sub(uint8_t a, uint8_t b)
{
    return (uint8_t)(a ^ b);
}

/*
 * Convert a random 32-bit value to the canonical byte representation.
 *
 * If x is uniformly random over 32-bit values, these outputs are
 * uniformly distributed over GF(16) and GF(256), respectively.
 */
static inline uint8_t gf16_from_i32(int32_t x)
{
    return (uint8_t)((uint32_t)x & UINT32_C(0x0F));
}

static inline uint8_t gf256_from_i32(int32_t x)
{
    return (uint8_t)((uint32_t)x & UINT32_C(0xFF));
}

static inline uint8_t gf16_from_integer(int32_t x)
{
    return (uint8_t)((uint32_t)x & 1u);
}

static inline uint8_t gf256_from_integer(int32_t x)
{
    return (uint8_t)((uint32_t)x & 1u);
}

/*
 * Multiply in GF(2^4), reducing modulo x^4 + x + 1.
 *
 * This has a fixed iteration count and no data-dependent branches or
 * memory accesses. Inputs should be canonical nibbles; masking below
 * also makes it tolerate high input bits.
 */
static inline uint8_t gf16_mul(uint8_t a, uint8_t b)
{
    uint8_t p = 0;

    a &= 0x0Fu;
    b &= 0x0Fu;

    for (unsigned i = 0; i < 4; ++i) {
        uint8_t bmask = (uint8_t)-(uint8_t)(b & 1u);
        uint8_t high  = (uint8_t)(a >> 3);

        p ^= (uint8_t)(a & bmask);
        a = (uint8_t)((a << 1) & 0x0Fu);

        /*
         * x^4 = x + 1 modulo x^4 + x + 1.
         * Therefore reduction XORs with binary 0011.
         */
        a ^= (uint8_t)(0x03u & (uint8_t)-high);
        b >>= 1;
    }

    return (uint8_t)(p & 0x0Fu);
}

/*
 * Multiply in GF(2^8), reducing modulo the AES polynomial 0x11B.
 */
static inline uint8_t gf256_mul(uint8_t a, uint8_t b)
{
    uint8_t p = 0;

    for (unsigned i = 0; i < 8; ++i) {
        uint8_t bmask = (uint8_t)-(uint8_t)(b & 1u);
        uint8_t high  = (uint8_t)(a >> 7);

        p ^= (uint8_t)(a & bmask);
        a = (uint8_t)(a << 1);

        /*
         * x^8 = x^4 + x^3 + x + 1 modulo 0x11B.
         * The low-byte reduction constant is 0x1B.
         */
        a ^= (uint8_t)(0x1Bu & (uint8_t)-high);
        b >>= 1;
    }

    return p;
}

/*
 * Fused multiply-add:
 *
 *     result = acc + a*b = acc XOR (a*b)
 *
 * These implementations integrate the accumulator into the multiplication
 * loop, avoiding a separate final XOR.
 */
static inline uint8_t
gf16_mul_add(uint8_t acc, uint8_t a, uint8_t b)
{
    uint8_t p = (uint8_t)(acc & 0x0Fu);

    a &= 0x0Fu;
    b &= 0x0Fu;

    for (unsigned i = 0; i < 4; ++i) {
        uint8_t bmask = (uint8_t)-(uint8_t)(b & 1u);
        uint8_t high  = (uint8_t)(a >> 3);

        p ^= (uint8_t)(a & bmask);

        a = (uint8_t)((a << 1) & 0x0Fu);
        a ^= (uint8_t)(0x03u & (uint8_t)-high);
        b >>= 1;
    }

    return (uint8_t)(p & 0x0Fu);
}

static inline uint8_t
gf256_mul_add(uint8_t acc, uint8_t a, uint8_t b)
{
    uint8_t p = acc;

    for (unsigned i = 0; i < 8; ++i) {
        uint8_t bmask = (uint8_t)-(uint8_t)(b & 1u);
        uint8_t high  = (uint8_t)(a >> 7);

        p ^= (uint8_t)(a & bmask);

        a = (uint8_t)(a << 1);
        a ^= (uint8_t)(0x1Bu & (uint8_t)-high);
        b >>= 1;
    }

    return p;
}

uint8_t gf16_mul_add_table(uint8_t acc, uint8_t a, uint8_t b);
uint8_t gf256_mul_add_table(uint8_t acc, uint8_t a, uint8_t b);

void gf16_mul_add_region(uint8_t *dst, const uint8_t *src,
                         size_t count, uint8_t coefficient);

void gf256_mul_add_region(uint8_t *dst, const uint8_t *src,
                          size_t count, uint8_t coefficient);

void gf16_mul_add_region_table(uint8_t *dst, const uint8_t *src,
                               size_t count, uint8_t coefficient);

void gf256_mul_add_region_table(uint8_t *dst, const uint8_t *src,
                                size_t count, uint8_t coefficient);


static inline uint8_t gf16_square(uint8_t a)
{
    return gf16_mul(a, a);
}

static inline uint8_t gf256_square(uint8_t a)
{
    return gf256_mul(a, a);
}

/*
 * Return a^(-1), with inverse(0) defined as 0.
 *
 * For nonzero a in GF(2^m):
 *     a^(-1) = a^(2^m - 2)
 */
static inline uint8_t gf16_inv(uint8_t a)
{
    /*
     * 14 = 12 + 2:
     * a2  = a^2
     * a3  = a^3
     * a6  = a^6
     * a12 = a^12
     * result = a^14
     */
    uint8_t a2  = gf16_square(a);
    uint8_t a3  = gf16_mul(a2, a);
    uint8_t a6  = gf16_square(a3);
    uint8_t a12 = gf16_square(a6);

    return gf16_mul(a12, a2);
}

static inline uint8_t gf256_inv(uint8_t a)
{
    /*
     * Addition chain for exponent 254:
     *
     * a2   = a^2
     * a3   = a^3
     * a6   = a^6
     * a12  = a^12
     * a15  = a^15
     * a30  = a^30
     * a60  = a^60
     * a120 = a^120
     * a240 = a^240
     * a252 = a^252
     * result = a^254
     *
     * Seven squarings and four general multiplications.
     */
    uint8_t a2   = gf256_square(a);
    uint8_t a3   = gf256_mul(a2, a);
    uint8_t a6   = gf256_square(a3);
    uint8_t a12  = gf256_square(a6);
    uint8_t a15  = gf256_mul(a12, a3);
    uint8_t a30  = gf256_square(a15);
    uint8_t a60  = gf256_square(a30);
    uint8_t a120 = gf256_square(a60);
    uint8_t a240 = gf256_square(a120);
    uint8_t a252 = gf256_mul(a240, a12);

    return gf256_mul(a252, a2);
}

/*
 * Checked division.
 *
 * Returns false when b == 0. The output is set to zero in that case.
 */
static inline bool gf16_div(uint8_t a, uint8_t b, uint8_t *out)
{
    if ((b & 0x0Fu) == 0) {
        *out = 0;
        return false;
    }

    *out = gf16_mul(a, gf16_inv(b));
    return true;
}

static inline bool gf256_div(uint8_t a, uint8_t b, uint8_t *out)
{
    if (b == 0) {
        *out = 0;
        return false;
    }

    *out = gf256_mul(a, gf256_inv(b));
    return true;
}

/*
 * Optional lookup tables.
 *
 * Call gf_tables_init() once before using these functions. Initialization
 * must finish before multiple threads begin accessing the tables.
 *
 * These functions are not cache-side-channel resistant.
 */
void gf_tables_init(void);

uint8_t gf16_mul_table(uint8_t a, uint8_t b);
uint8_t gf16_square_table(uint8_t a);
uint8_t gf16_inv_table(uint8_t a);

uint8_t gf256_mul_table(uint8_t a, uint8_t b);
uint8_t gf256_square_table(uint8_t a);
uint8_t gf256_inv_table(uint8_t a);

#endif /* GF2EXT_H */
