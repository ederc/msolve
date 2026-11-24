/* Unit test for the 64-bit Shoup arithmetic used by the wide
 * signature-F4 replay: shoup_mu / shoup_mul (tools.h) and the dense-row
 * update sg_sub_row_64. shoup_mu(p, x) = floor(x*2^64/p) and with
 * q = floor(mu*y/2^64) the wrapping difference x*y - q*p lies in
 * [0, 2p) and is x*y mod p or that plus p for 0 < x < p < 2^62 and any
 * y < 2^64. sg_sub_row_64 must keep dense-row entries in [0, 2p) under
 * arbitrary many subtractions of pivot rows. */
#include <stdio.h>
#include <stdlib.h>
#include "../../../src/neogb/tools.h"

/* deterministic xorshift64* */
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t rng(void)
{
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static int check_pair(const uint64_t p, const uint64_t x, const uint64_t y)
{
    /* q = floor(v*psi/2^64); d = x*y - q*p (wrapping u64) lies in
     * [0, 2p) and is x*y mod p or that plus p */
    const uint64_t q = shoup_mul(shoup_mu(p, x), y);
    const uint64_t d = x * y - q * p;
    const uint64_t ref = mul_mod_64(x, y, p);
    if (d != ref && d != ref + p) {
        printf("FAIL p=%llu x=%llu y=%llu: shoup=%llu ref=%llu\n",
               (unsigned long long)p, (unsigned long long)x,
               (unsigned long long)y, (unsigned long long)d,
               (unsigned long long)ref);
        return 0;
    }
    return 1;
}

int main(void)
{
    /* the Mersenne prime 2^61-1, the boundary-relevant prime 2^62-57,
     * and a mid-range prime */
    const uint64_t primes[] = {
        2305843009213693951ULL,               /* 2^61 - 1 */
        4611686018427387847ULL,               /* 2^62 - 57 */
        3458764513820540933ULL                /* mid-range */
    };
    const int nprimes = (int)(sizeof(primes) / sizeof(primes[0]));

    for (int pi = 0; pi < nprimes; ++pi) {
        const uint64_t p = primes[pi];

        /* randomized: 10^5 pairs */
        for (int i = 0; i < 100000; ++i) {
            const uint64_t x = rng() % p;
            const uint64_t y = rng();
            if (!check_pair(p, x, y)) {
                return 1;
            }
        }

        /* edges */
        const uint64_t xs[] = {1, p - 1, 2, p / 2, p - 2};
        const uint64_t ys[] = {1, p - 1, p / 2, 2, UINT64_MAX};
        for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); ++i) {
            for (size_t j = 0; j < sizeof(ys) / sizeof(ys[0]); ++j) {
                if (!check_pair(p, xs[i], ys[j])) {
                    return 1;
                }
            }
        }
    }

    /* sg_sub_row_64: 1000 chained random subtractions from a zero row
     * must keep every entry in [0, 2p) */
    for (int pi = 0; pi < nprimes; ++pi) {
        const uint64_t p = primes[pi];
        const len_t ncols = 512;
        uint64_t *dr = (uint64_t *)calloc(ncols, sizeof(uint64_t));
        cf64_t *pf = (cf64_t *)malloc(ncols * sizeof(cf64_t));
        hm_t *pr = (hm_t *)malloc(ncols * sizeof(hm_t));
        for (len_t j = 0; j < ncols; ++j) {
            pr[j]  = (hm_t)j;
            pf[j]  = rng() % p;              /* canonical pivot coefficient */
        }
        for (int i = 0; i < 1000; ++i) {
            const uint64_t v = rng() % p; /* consumed dense value */
            sg_sub_row_64(dr, pr, pf, ncols, v, p);
            for (len_t j = 0; j < ncols; ++j) {
                if (dr[j] >= 2 * p) {
                    printf("FAIL sg_sub_row_64: dr[%d]=%llu >= 2p=%llu (p=%llu, iter %d)\n",
                           (int)j, (unsigned long long)dr[j],
                           (unsigned long long)(2 * p),
                           (unsigned long long)p, i);
                    return 1;
                }
            }
        }
        /* edges: a zero multiplier is the identity on the dense row,
         * and a zero pivot coefficient leaves its column untouched */
        uint64_t *snap = (uint64_t *)malloc(ncols * sizeof(uint64_t));
        memcpy(snap, dr, ncols * sizeof(uint64_t));
        sg_sub_row_64(dr, pr, pf, ncols, 0, p);
        for (len_t j = 0; j < ncols; ++j) {
            if (dr[j] != snap[j]) {
                printf("FAIL sg_sub_row_64 edge v=0: dr[%d] changed\n", (int)j);
                return 1;
            }
        }
        pf[ncols / 2] = 0;
        sg_sub_row_64(dr, pr, pf, ncols, rng() % p, p);
        if (dr[ncols / 2] != snap[ncols / 2]) {
            printf("FAIL sg_sub_row_64 edge pf=0: dr[%d] changed\n",
                   (int)(ncols / 2));
            return 1;
        }
        for (len_t j = 0; j < ncols; ++j) {
            if (dr[j] >= 2 * p) {
                printf("FAIL sg_sub_row_64 edge: dr[%d]=%llu >= 2p=%llu\n",
                       (int)j, (unsigned long long)dr[j],
                       (unsigned long long)(2 * p));
                return 1;
            }
        }
        free(snap);
        free(dr);
        free(pf);
        free(pr);
    }

    printf("shoup_64: all checks passed\n");
    return 0;
}
