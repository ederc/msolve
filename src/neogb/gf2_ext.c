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

#include "gf2_ext.h"

#include <assert.h>

/*
 * GF(16):  16 * 16 = 256 bytes
 * GF(256): 256 * 256 = 65536 bytes
 */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Alignas(64)
#endif
static uint8_t gf16_mul_lut[16u * 16u];

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Alignas(64)
#endif
static uint8_t gf256_mul_lut[256u * 256u];

static uint8_t gf16_inv_lut[16];
static uint8_t gf256_inv_lut[256];

static int tables_initialized;

void gf_tables_init(void)
{
    for (unsigned a = 0; a < 16; ++a) {
        for (unsigned b = 0; b < 16; ++b) {
            gf16_mul_lut[(a << 4) | b] =
                gf16_mul((uint8_t)a, (uint8_t)b);
        }
    }

    for (unsigned a = 0; a < 256; ++a) {
        for (unsigned b = 0; b < 256; ++b) {
            gf256_mul_lut[(a << 8) | b] =
                gf256_mul((uint8_t)a, (uint8_t)b);
        }
    }

    gf16_inv_lut[0] = 0;
    for (unsigned a = 1; a < 16; ++a)
        gf16_inv_lut[a] = gf16_inv((uint8_t)a);

    gf256_inv_lut[0] = 0;
    for (unsigned a = 1; a < 256; ++a)
        gf256_inv_lut[a] = gf256_inv((uint8_t)a);

    tables_initialized = 1;
}

uint8_t gf16_mul_table(uint8_t a, uint8_t b)
{
    assert(tables_initialized);
    return gf16_mul_lut[((unsigned)(a & 0x0Fu) << 4) |
                        (unsigned)(b & 0x0Fu)];
}

uint8_t gf16_square_table(uint8_t a)
{
    a &= 0x0Fu;
    return gf16_mul_table(a, a);
}

uint8_t gf16_inv_table(uint8_t a)
{
    assert(tables_initialized);
    return gf16_inv_lut[a & 0x0Fu];
}

uint8_t gf256_mul_table(uint8_t a, uint8_t b)
{
    assert(tables_initialized);
    return gf256_mul_lut[((unsigned)a << 8) | (unsigned)b];
}

uint8_t gf256_square_table(uint8_t a)
{
    return gf256_mul_table(a, a);
}

uint8_t gf256_inv_table(uint8_t a)
{
    assert(tables_initialized);
    return gf256_inv_lut[a];
}

uint8_t
gf16_mul_add_table(uint8_t acc, uint8_t a, uint8_t b)
{
    unsigned index;

    assert(tables_initialized);

    index = ((unsigned)(a & 0x0Fu) << 4) |
            (unsigned)(b & 0x0Fu);

    return (uint8_t)((acc ^ gf16_mul_lut[index]) & 0x0Fu);
}

uint8_t
gf256_mul_add_table(uint8_t acc, uint8_t a, uint8_t b)
{
    unsigned index;

    assert(tables_initialized);

    index = ((unsigned)a << 8) | (unsigned)b;

    return (uint8_t)(acc ^ gf256_mul_lut[index]);
}

void
gf16_mul_add_region(uint8_t *dst, const uint8_t *src,
                    size_t count, uint8_t coefficient)
{
    coefficient &= 0x0Fu;

    if (coefficient == 0)
        return;

    if (coefficient == 1) {
        for (size_t i = 0; i < count; ++i)
            dst[i] = (uint8_t)((dst[i] ^ src[i]) & 0x0Fu);
        return;
    }

    for (size_t i = 0; i < count; ++i)
        dst[i] = gf16_mul_add(dst[i], coefficient, src[i]);
}

void
gf256_mul_add_region(uint8_t *dst, const uint8_t *src,
                     size_t count, uint8_t coefficient)
{
    if (coefficient == 0)
        return;

    if (coefficient == 1) {
        for (size_t i = 0; i < count; ++i)
            dst[i] ^= src[i];
        return;
    }

    for (size_t i = 0; i < count; ++i)
        dst[i] = gf256_mul_add(dst[i], coefficient, src[i]);
}

void
gf16_mul_add_region_table(uint8_t *dst, const uint8_t *src,
                          size_t count, uint8_t coefficient)
{
    const uint8_t *row;

    assert(tables_initialized);

    coefficient &= 0x0Fu;

    if (coefficient == 0)
        return;

    if (coefficient == 1) {
        for (size_t i = 0; i < count; ++i)
            dst[i] = (uint8_t)((dst[i] ^ src[i]) & 0x0Fu);
        return;
    }

    /*
     * A fixed coefficient selects one 16-byte multiplication row,
     * which remains cache-resident.
     */
    row = &gf16_mul_lut[(unsigned)coefficient << 4];

    for (size_t i = 0; i < count; ++i)
        dst[i] = (uint8_t)((dst[i] ^ row[src[i] & 0x0Fu]) & 0x0Fu);
}

void
gf256_mul_add_region_table(uint8_t *dst, const uint8_t *src,
                           size_t count, uint8_t coefficient)
{
    const uint8_t *row;

    assert(tables_initialized);

    if (coefficient == 0)
        return;

    if (coefficient == 1) {
        for (size_t i = 0; i < count; ++i)
            dst[i] ^= src[i];
        return;
    }

    /*
     * Since the coefficient is fixed, only one 256-byte row of the
     * 64 KiB multiplication table is accessed.
     */
    row = &gf256_mul_lut[(unsigned)coefficient << 8];

    for (size_t i = 0; i < count; ++i)
        dst[i] ^= row[src[i]];
}
