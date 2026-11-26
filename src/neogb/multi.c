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

/* Applying a learned trace at up to ML primes below 2^31 at once.
 *
 * The rounds of a trace are the same for all good primes, only the
 * coefficients differ. Once the matrices of all rounds are stored in the
 * trace, see tc_t, the trace is applied at ML primes in one pass: each
 * coefficient of a basis element is an array of ML coefficients, one per
 * prime or lane, and the dense rows of the linear algebra hold ML entries
 * per column side by side, so that a pivot row is read once for all
 * primes and its products are computed for all lanes by one vector
 * instruction. All lanes share the supports of the basis elements, those
 * of the elements stored in the trace: a coefficient vanishing at one
 * prime is a zero entry of its lane. A lane whose new element has
 * another lead term, or a term outside of the stored support, is the one
 * of a bad prime and is dropped; the other lanes go on. */

#define ML 8

typedef struct ml_t ml_t;
struct ml_t
{
    len_t nl;           /* number of lanes */
    uint64_t p[ML];     /* the primes, 1 for unused lanes */
    uint64_t big[ML];   /* the largest multiple of p below 2^63 */
    uint64_t mu[ML];    /* floor(2^64 / p) for the reductions mod p */
    uint32_t act;       /* the lanes still alive */
    len_t ld;           /* basis elements */
    len_t sz;
    hm_t **hm;          /* the elements, COEFFS is the index of cf */
    uint32_t **cf;      /* their coefficients, ML per term */
    len_t ncf;          /* size of cf */
};

/* x mod p for x < 2^63 by the pre-inverse mu = floor(2^64 / p): the
 * quotient mulhi(x, mu) is at most one too small */
static inline uint64_t ml_mod(
        const uint64_t x,
        const uint64_t p,
        const uint64_t mu
        )
{
    const uint64_t q = (uint64_t)(((unsigned __int128)x * mu) >> 64);
    const uint64_t r = x - q * p;
    return r >= p ? r - p : r;
}

/* multiplies the coefficients of all active lanes of the row by the
 * inverse of their first coefficient; a lane whose first coefficient is
 * zero is dropped */
static void ml_normalize(
        ml_t *ml,
        uint32_t *cf,
        const len_t len
        )
{
    len_t j, l;
    for (l = 0; l < ML; ++l) {
        if (!(ml->act >> l & 1)) {
            continue;
        }
        if (cf[l] == 0) {
            ml->act &= ~((uint32_t)1 << l);
            continue;
        }
        const uint64_t inv = mod_p_inverse_32((int64_t)cf[l], (int64_t)ml->p[l]);
        for (j = 0; j < len; ++j) {
            cf[(unsigned long)j*ML + l] = (uint32_t)
                (((uint64_t)cf[(unsigned long)j*ML + l] * inv) % ml->p[l]);
        }
    }
}

/* adds m times the pivot row (pr, pc) of length pl but its first term to
 * the dense row dr, the entries of each lane stay below big */
static inline void ml_add_row(
        uint64_t *dr,
        const hm_t * const pr,
        const uint32_t * const pc,
        const len_t pl,
        const uint64_t * const m,
        const uint64_t * const big
        )
{
    len_t j;
#if defined HAVE_AVX512_F
    const __m512i vm = _mm512_loadu_si512((const void *)m);
    const __m512i vb = _mm512_loadu_si512((const void *)big);
    for (j = 1; j < pl; ++j) {
        uint64_t *y = dr + (unsigned long)pr[j] * ML;
        const __m512i c = _mm512_cvtepu32_epi64(
                _mm256_loadu_si256((const __m256i *)(pc + (unsigned long)j * ML)));
        __m512i t = _mm512_add_epi64(_mm512_loadu_si512((const void *)y),
                _mm512_mul_epu32(vm, c));
        const __mmask8 o = _mm512_cmpge_epu64_mask(t, vb);
        t = _mm512_mask_sub_epi64(t, o, t, vb);
        _mm512_storeu_si512((void *)y, t);
    }
#elif defined HAVE_AVX2
    /* the lanes in two halves of four; AVX2 compares 64-bit integers
     * only as signed ones, with the sign bits flipped t >= big becomes
     * t > big - 1 */
    const __m256i sg  = _mm256_set1_epi64x((long long)((uint64_t)1 << 63));
    const __m256i m0  = _mm256_loadu_si256((const __m256i *)m);
    const __m256i m1  = _mm256_loadu_si256((const __m256i *)(m + 4));
    const __m256i b0  = _mm256_loadu_si256((const __m256i *)big);
    const __m256i b1  = _mm256_loadu_si256((const __m256i *)(big + 4));
    const __m256i one = _mm256_set1_epi64x(1);
    const __m256i c0  = _mm256_xor_si256(_mm256_sub_epi64(b0, one), sg);
    const __m256i c1  = _mm256_xor_si256(_mm256_sub_epi64(b1, one), sg);
    for (j = 1; j < pl; ++j) {
        __m256i *y = (__m256i *)(dr + (unsigned long)pr[j] * ML);
        const uint32_t *c = pc + (unsigned long)j * ML;
        const __m256i v0 = _mm256_cvtepu32_epi64(_mm_loadu_si128((const __m128i *)c));
        const __m256i v1 = _mm256_cvtepu32_epi64(_mm_loadu_si128((const __m128i *)(c + 4)));
        __m256i t0 = _mm256_add_epi64(_mm256_loadu_si256(y), _mm256_mul_epu32(m0, v0));
        __m256i t1 = _mm256_add_epi64(_mm256_loadu_si256(y + 1), _mm256_mul_epu32(m1, v1));
        const __m256i o0 = _mm256_cmpgt_epi64(_mm256_xor_si256(t0, sg), c0);
        const __m256i o1 = _mm256_cmpgt_epi64(_mm256_xor_si256(t1, sg), c1);
        t0 = _mm256_sub_epi64(t0, _mm256_and_si256(o0, b0));
        t1 = _mm256_sub_epi64(t1, _mm256_and_si256(o1, b1));
        _mm256_storeu_si256(y, t0);
        _mm256_storeu_si256(y + 1, t1);
    }
#elif defined __aarch64__
    /* the lanes in four pairs, vmull_u32 multiplies two 32-bit lanes to
     * 64-bit products; the multipliers are below 2^32 */
    len_t l;
    uint32x2_t mv[ML/2];
    uint64x2_t bv[ML/2];
    for (l = 0; l < ML/2; ++l) {
        mv[l] = vmovn_u64(vld1q_u64(m + 2*l));
        bv[l] = vld1q_u64(big + 2*l);
    }
    for (j = 1; j < pl; ++j) {
        uint64_t *y = dr + (unsigned long)pr[j] * ML;
        const uint32_t *c = pc + (unsigned long)j * ML;
        for (l = 0; l < ML/2; ++l) {
            uint64x2_t t = vaddq_u64(vld1q_u64(y + 2*l),
                    vmull_u32(mv[l], vld1_u32(c + 2*l)));
            const uint64x2_t o = vcgeq_u64(t, bv[l]);
            t = vsubq_u64(t, vandq_u64(o, bv[l]));
            vst1q_u64(y + 2*l, t);
        }
    }
#else
    len_t l;
    for (j = 1; j < pl; ++j) {
        uint64_t *y = dr + (unsigned long)pr[j] * ML;
        const uint32_t *c = pc + (unsigned long)j * ML;
        for (l = 0; l < ML; ++l) {
            const uint64_t t = y[l] + m[l] * c[l];
            y[l] = t >= big[l] ? t - big[l] : t;
        }
    }
#endif
}

/* the rows of the matrix stored in the trace, decompressed if needed */
static hm_t **ml_rows(
        const tc_t * const tc,
        hm_t **buf
        )
{
    len_t i;
    const len_t nr = tc->nru + tc->nrl;
    hm_t **rows = (hm_t **)malloc((unsigned long)(nr + 1) * sizeof(hm_t *));
    *buf = NULL;
    if (tc->raw) {
        for (i = 0; i < nr; ++i) {
            rows[i] = (hm_t *)(tc->rows + tc->ro[i]);
        }
        return rows;
    }
    unsigned long sz = 0;
    for (i = 0; i < nr; ++i) {
        sz += cached_row_length(tc, i) + OFFSET;
    }
    *buf = (hm_t *)malloc(sz * sizeof(hm_t));
    sz = 0;
    for (i = 0; i < nr; ++i) {
        rows[i] = *buf + sz;
        decompress_row(rows[i], tc->rows + tc->ro[i]);
        sz += rows[i][LENGTH] + OFFSET;
    }
    return rows;
}

/* reduces the dense row dr from column sc on by the pivots, piv[c] is the
 * row of the pivot of column c and pcf[c] its coefficients. Writes the
 * columns of the remainder to rc and their coefficients, ML per column, to
 * rv, returns their number. dr is zero afterwards. */
static len_t ml_reduce_row(
        ml_t *ml,
        uint64_t *dr,
        const len_t sc,
        const len_t nc,
        hm_t * const *piv,
        uint32_t * const *pcf,
        hm_t *rc,
        uint32_t *rv
        )
{
    len_t c, l, nr = 0;
    uint64_t m[ML];
    for (c = sc; c < nc; ++c) {
        uint64_t * const x = dr + (unsigned long)c * ML;
        uint64_t any = 0;
        for (l = 0; l < ML; ++l) {
            any |= x[l];
        }
        if (any == 0) {
            continue;
        }
        any = 0;
        for (l = 0; l < ML; ++l) {
            const uint64_t v = ml_mod(x[l], ml->p[l], ml->mu[l]);
            x[l] = 0;
            m[l] = (v == 0 || !(ml->act >> l & 1)) ? 0 : ml->p[l] - v;
            any |= m[l];
        }
        if (any == 0) {
            continue;
        }
        if (piv[c] != NULL) {
            ml_add_row(dr, piv[c] + OFFSET, pcf[c], piv[c][LENGTH], m, ml->big);
            continue;
        }
        rc[nr] = c;
        for (l = 0; l < ML; ++l) {
            rv[(unsigned long)nr*ML + l] = (uint32_t)(m[l] == 0 ? 0 : ml->p[l] - m[l]);
        }
        nr++;
    }
    return nr;
}

/* applies the matrix of round rd stored in the trace, the new elements get
 * the lead terms nlms[0], ..., nlms[nn-1] and are appended to the basis;
 * for the final reduction (nlms == NULL) the reduced basis elements are
 * written to fin by decreasing lead columns, as in reduce_final_basis(). Returns the number of
 * new elements, -1 if the round does not apply. */
static int64_t ml_round(
        ml_t *ml,
        const tc_t * const tc,
        const hm_t * const nlms,
        const len_t nn,
        const trace_t * const tr,
        ht_t *bht,
        hm_t ***finp,
        uint32_t ***fcfp
        )
{
    len_t i, j, k, l;

    const len_t nru = tc->nru;
    const len_t nrl = tc->nrl;
    const len_t ncl = tc->ncl;
    const len_t nc  = tc->ncl + tc->ncr;
    /* the positions of the columns in the basis hash table, looked up if
     * the matrix was stored with another hash table */
    hm_t *hcl = NULL;
    const hm_t *hc = tc->hc;
    if (hc == NULL) {
        hcl = (hm_t *)malloc((unsigned long)(tc->ncr + 1) * sizeof(hm_t));
        for (j = 0; j < tc->ncr; ++j) {
            if (bht->eld == bht->esz-1) {
                enlarge_hash_table(bht);
            }
            hcl[j] = insert_in_hash_table(tc->ev + (unsigned long)j * bht->evl, bht);
        }
        hc = hcl;
    }
    hm_t *buf   = NULL;
    hm_t **rows = ml_rows(tc, &buf);

    hm_t **piv      = (hm_t **)calloc((unsigned long)nc + 1, sizeof(hm_t *));
    uint32_t **pcf  = (uint32_t **)calloc((unsigned long)nc + 1, sizeof(uint32_t *));
    for (i = 0; i < nru; ++i) {
        const hm_t e = tc->rcf != NULL ? tc->rcf[i] : rows[i][COEFFS];
        piv[rows[i][OFFSET]] = rows[i];
        pcf[rows[i][OFFSET]] = ml->cf[e];
    }
    uint64_t *dr = (uint64_t *)calloc((unsigned long)nc * ML, sizeof(uint64_t));
    hm_t *rc     = (hm_t *)malloc((unsigned long)(nc + 1) * sizeof(hm_t));
    uint32_t *rv = (uint32_t *)malloc((unsigned long)(nc + 1) * ML * sizeof(uint32_t));
    /* the new pivots */
    hm_t **np       = (hm_t **)malloc((unsigned long)(nrl + 1) * sizeof(hm_t *));
    uint32_t **npc  = (uint32_t **)malloc((unsigned long)(nrl + 1) * sizeof(uint32_t *));
    len_t nnp = 0;
    int ok = 1;
    for (i = 0; i < nrl && ok; ++i) {
        const hm_t * const r  = rows[nru+i];
        const uint32_t * const cf = ml->cf[r[COEFFS]];
        const len_t len = r[LENGTH];
        hm_t sc = nc;
        for (j = 0; j < len; ++j) {
            const hm_t c = r[OFFSET+j];
            for (l = 0; l < ML; ++l) {
                dr[(unsigned long)c*ML + l] = cf[(unsigned long)j*ML + l];
            }
            if (c < sc) {
                sc = c;
            }
        }
        const len_t nr = ml_reduce_row(ml, dr, sc, nc, piv, pcf, rc, rv);
        if (nr == 0) {
            /* reduces to zero in all lanes: no good prime */
            ok = 0;
            break;
        }
        /* lanes without the lead term of the remainder are bad */
        for (l = 0; l < ML; ++l) {
            if ((ml->act >> l & 1) && rv[l] == 0) {
                ml->act &= ~((uint32_t)1 << l);
            }
        }
        if (ml->act == 0) {
            ok = 0;
            break;
        }
        hm_t *row = (hm_t *)malloc((unsigned long)(nr + OFFSET) * sizeof(hm_t));
        row[LENGTH]  = nr;
        row[PRELOOP] = nr % UNROLL;
        memcpy(row + OFFSET, rc, (unsigned long)nr * sizeof(hm_t));
        uint32_t *pc = (uint32_t *)malloc((unsigned long)nr * ML * sizeof(uint32_t));
        memcpy(pc, rv, (unsigned long)nr * ML * sizeof(uint32_t));
        ml_normalize(ml, pc, nr);
        piv[rc[0]] = row;
        pcf[rc[0]] = pc;
        np[nnp]  = row;
        npc[nnp] = pc;
        nnp++;
    }
    /* interreduce the new pivots, the ones leading farthest right first */
    for (k = nc; ok && k > ncl; --k) {
        const len_t c = k - 1;
        if (piv[c] == NULL || c < ncl) {
            continue;
        }
        hm_t *row = piv[c];
        uint32_t *pc = pcf[c];
        const len_t len = row[LENGTH];
        int red = 0;
        for (j = 1; j < len; ++j) {
            if (piv[row[OFFSET+j]] != NULL) {
                red = 1;
                break;
            }
        }
        if (!red) {
            continue;
        }
        for (j = 1; j < len; ++j) {
            for (l = 0; l < ML; ++l) {
                dr[(unsigned long)row[OFFSET+j]*ML + l] = pc[(unsigned long)j*ML + l];
            }
        }
        /* keep the lead term, reduce the rest */
        const len_t nr = ml_reduce_row(ml, dr, c + 1, nc, piv, pcf, rc + 1, rv + ML);
        rc[0] = c;
        for (l = 0; l < ML; ++l) {
            rv[l] = pc[l];
        }
        hm_t *nrow = (hm_t *)malloc((unsigned long)(nr + 1 + OFFSET) * sizeof(hm_t));
        nrow[LENGTH]  = nr + 1;
        nrow[PRELOOP] = (nr + 1) % UNROLL;
        memcpy(nrow + OFFSET, rc, (unsigned long)(nr + 1) * sizeof(hm_t));
        uint32_t *npcf = (uint32_t *)malloc((unsigned long)(nr + 1) * ML * sizeof(uint32_t));
        memcpy(npcf, rv, (unsigned long)(nr + 1) * ML * sizeof(uint32_t));
        for (i = 0; i < nnp; ++i) {
            if (np[i] == row) {
                np[i]  = nrow;
                npc[i] = npcf;
            }
        }
        free(row);
        free(pc);
        piv[c] = nrow;
        pcf[c] = npcf;
    }

    int64_t ret = -1;
    if (ok && nlms != NULL) {
        /* the new elements in the order of the trace */
        if (nnp == nn) {
            ret = nn;
            for (k = 0; k < nn && ret >= 0; ++k) {
                for (i = 0; i < nnp; ++i) {
                    if (np[i] != NULL && np[i][OFFSET] >= ncl
                            && hc[np[i][OFFSET] - ncl] == nlms[k]) {
                        break;
                    }
                }
                if (i == nnp) {
                    ret = -1;
                    break;
                }
                hm_t *row    = np[i];
                uint32_t *pc = npc[i];
                np[i] = NULL;
                const len_t bi = ml->ld;
                /* the support of the element: the one stored in the trace
                 * if there is one, lanes with terms outside of it are bad */
                const hm_t * const ref = bi < tr->bsz ? tr->bsp[bi] : NULL;
                hm_t *er;
                uint32_t *ec;
                if (ref != NULL) {
                    const len_t rl = ref[0];
                    er = (hm_t *)malloc((unsigned long)(rl + OFFSET) * sizeof(hm_t));
                    ec = (uint32_t *)calloc((unsigned long)rl * ML, sizeof(uint32_t));
                    er[LENGTH]  = rl;
                    er[PRELOOP] = rl % UNROLL;
                    memcpy(er + OFFSET, ref + 2, (unsigned long)rl * sizeof(hm_t));
                    len_t q = 0;
                    for (j = 0; j < row[LENGTH]; ++j) {
                        const hm_t h = hc[row[OFFSET+j] - ncl];
                        while (q < rl && ref[2+q] != h) {
                            ++q;
                        }
                        if (q == rl) {
                            /* a term outside of the stored support */
                            for (l = 0; l < ML; ++l) {
                                if (pc[(unsigned long)j*ML + l] != 0) {
                                    ml->act &= ~((uint32_t)1 << l);
                                }
                            }
                            q = 0;
                            continue;
                        }
                        memcpy(ec + (unsigned long)q * ML, pc + (unsigned long)j * ML,
                                ML * sizeof(uint32_t));
                    }
                    free(row);
                    free(pc);
                } else {
                    er = row;
                    ec = pc;
                    for (j = 0; j < row[LENGTH]; ++j) {
                        er[OFFSET+j] = hc[row[OFFSET+j] - ncl];
                    }
                }
                er[COEFFS] = bi;
                er[DEG]    = bht->hd[er[OFFSET]].deg;
                if (ml->ld >= ml->sz) {
                    ml->sz *= 2;
                    ml->hm  = realloc(ml->hm, (unsigned long)ml->sz * sizeof(hm_t *));
                }
                if (bi >= ml->ncf) {
                    const len_t o = ml->ncf;
                    ml->ncf = 2 * (bi + 1);
                    ml->cf  = realloc(ml->cf, (unsigned long)ml->ncf * sizeof(uint32_t *));
                    memset(ml->cf + o, 0, (unsigned long)(ml->ncf - o) * sizeof(uint32_t *));
                }
                ml->hm[ml->ld++] = er;
                ml->cf[bi]       = ec;
            }
        }
    } else if (ok) {
        /* the final reduction: the reduced elements by their lead columns */
        hm_t **fin    = (hm_t **)malloc((unsigned long)(nnp + 1) * sizeof(hm_t *));
        uint32_t **fc = (uint32_t **)malloc((unsigned long)(nnp + 1) * sizeof(uint32_t *));
        len_t nf = 0;
        /* as sort_matrix_rows_increasing(): the largest column first */
        for (k = nc; k > 0; --k) {
            for (i = 0; i < nnp; ++i) {
                if (np[i] != NULL && np[i][OFFSET] == k - 1) {
                    break;
                }
            }
            if (i < nnp) {
                hm_t *row = np[i];
                for (j = 0; j < row[LENGTH]; ++j) {
                    row[OFFSET+j] = row[OFFSET+j] < ncl ? 0
                        : hc[row[OFFSET+j] - ncl];
                }
                fin[nf] = row;
                fc[nf]  = npc[i];
                np[i]   = NULL;
                nf++;
            }
        }
        *finp = fin;
        *fcfp = fc;
        ret   = nf;
    }
    /* the new pivots not taken over */
    for (i = 0; i < nnp; ++i) {
        if (np[i] != NULL) {
            free(np[i]);
            free(npc[i]);
        }
    }
    free(np);
    free(npc);
    free(piv);
    free(pcf);
    free(dr);
    free(rc);
    free(rv);
    free(rows);
    free(buf);
    free(hcl);
    return ret;
}

/* the reduced basis of lane l, its elements without the terms whose
 * coefficients vanish at the prime of the lane */
static bs_t *ml_lane_basis(
        const ml_t * const ml,
        hm_t * const *fin,
        uint32_t * const *fcf,
        const len_t nf,
        const len_t l,
        ht_t *bht
        )
{
    len_t i, j;

    bs_t *bs    = (bs_t *)calloc(1, sizeof(bs_t));
    const len_t sz = nf > 0 ? nf : 1;
    bs->ht      = bht;
    bs->ght     = 1;
    bs->ld      = bs->lml = nf;
    bs->sz      = sz;
    bs->hm      = (hm_t **)malloc((unsigned long)sz * sizeof(hm_t *));
    bs->cf_32   = (cf32_t **)malloc((unsigned long)sz * sizeof(cf32_t *));
    bs->lmps    = (bl_t *)malloc((unsigned long)sz * sizeof(bl_t));
    bs->lm      = (sdm_t *)malloc((unsigned long)sz * sizeof(sdm_t));
    bs->red     = (int8_t *)calloc((unsigned long)sz, sizeof(int8_t));
    for (i = 0; i < nf; ++i) {
        const hm_t * const r = fin[i];
        len_t n = 0;
        for (j = 0; j < r[LENGTH]; ++j) {
            n += fcf[i][(unsigned long)j*ML + l] != 0;
        }
        hm_t *row  = (hm_t *)malloc((unsigned long)(n + OFFSET) * sizeof(hm_t));
        cf32_t *cf = (cf32_t *)malloc((unsigned long)(n + 1) * sizeof(cf32_t));
        n = 0;
        for (j = 0; j < r[LENGTH]; ++j) {
            const uint32_t v = fcf[i][(unsigned long)j*ML + l];
            if (v != 0) {
                row[OFFSET+n] = r[OFFSET+j];
                cf[n++]       = v;
            }
        }
        row[LENGTH]  = n;
        row[PRELOOP] = n % UNROLL;
        row[COEFFS]  = i;
        row[MULT]    = 0;
        row[BINDEX]  = 0;
        row[DEG]     = bht->hd[row[OFFSET]].deg;
        if (row[DEG] == 0) {
            bs->constant = 1;
        }
        bs->hm[i]     = row;
        bs->cf_32[i]  = cf;
        bs->lmps[i]   = i;
        bs->lm[i]     = bht->hd[row[OFFSET]].sdm;
    }
    return bs;
}

/* the reduced input generators modulo p, as the applications of the trace
 * of a signature-based run start from them, see initialize_f4(); NULL if
 * p is bad */
static bs_t *ml_echelon_inputs(
        bs_t *gbs,
        const md_t * const gmd,
        const uint64_t p,
        len_t *ng
        )
{
    md_t *md  = copy_meta_data(gmd, p);
    md->hcm   = (hi_t *)malloc(sizeof(hi_t));
    bs_t *bs  = copy_basis_mod_p(gbs, md);
    normalize_initial_basis(bs, p);
    md->ht    = initialize_secondary_hash_table(bs->ht, md);
    mat_t *mat = (mat_t *)calloc(1, sizeof(mat_t));
    bs->ld    = 0;
    const int32_t bad = echelonize_input_generators(mat, bs, md);
    bs->ld    = md->ngens;
    *ng       = md->ngens;
    free(mat);
    free_meta_data(&md);
    if (bad) {
        free_basis_and_only_local_hash_table_data(&bs);
        return NULL;
    }
    return bs;
}

/* writes the coefficients of lane l of the polynomial (h, c) of length n
 * to cf at the positions of its terms in the support r, led by its length;
 * returns 0 if it has a nonzero term outside of r */
static int ml_place(
        uint32_t *cf,
        const hm_t * const r,
        const hm_t * const h,
        const uint32_t * const c,
        const len_t n,
        const len_t l
        )
{
    len_t j, q = 0;
    const len_t rl = r[0];
    for (j = 0; j < n; ++j) {
        if (c[j] == 0) {
            continue;
        }
        while (q < rl && r[1+q] != h[j]) {
            ++q;
        }
        if (q == rl) {
            return 0;
        }
        cf[(unsigned long)q*ML + l] = c[j];
    }
    return 1;
}

/* applies the trace of a signature-based run, see sig_f4_replay(), at the
 * primes of the lanes of ml, whose basis is empty; returns 0 if it does not
 * apply. The basis gets the elements of the run in the order of
 * sg_load_basis(). */
static int ml_sig_rounds(
        ml_t *ml,
        bs_t *gbs,
        const md_t * const gmd,
        const sg_rec_t * const rec
        )
{
    len_t i, j, k, l, r;

    /* the reduced input generators in each lane */
    bs_t *lb[ML];
    len_t ng = 0;
    for (l = 0; l < ML; ++l) {
        lb[l] = NULL;
        if (!(ml->act >> l & 1)) {
            continue;
        }
        len_t n = 0;
        lb[l] = ml_echelon_inputs(gbs, gmd, ml->p[l], &n);
        if (lb[l] == NULL || (ng > 0 && n != ng)) {
            ml->act &= ~((uint32_t)1 << l);
            continue;
        }
        ng = n;
    }
    int ok = ml->act != 0 && rec->gs != NULL && rec->ng == ng;
    len_t nel = ng;
    for (r = 0; ok && r < rec->nrd; ++r) {
        const sg_rrd_t * const rd = rec->rd + r;
        if (rd->n > 0 && (rd->rc == NULL || rd->ec == NULL || rd->eb == NULL)) {
            ok = 0;
        }
        nel += rd->nn;
    }
    len_t l0 = 0;
    while (l0 < ML && lb[l0] == NULL) {
        ++l0;
    }
    len_t *gi = ok ? sg_generator_order(lb[l0], ng) : NULL;
    if (ok) {
        ml->sz  = nel + 1;
        ml->hm  = (hm_t **)calloc((unsigned long)ml->sz, sizeof(hm_t *));
        ml->ncf = nel + 1;
        ml->cf  = (uint32_t **)calloc((unsigned long)ml->ncf, sizeof(uint32_t *));
        /* the generators in the order of their signatures, with the
         * supports of the run */
        for (i = 0; i < ng; ++i) {
            const hm_t * const gs = rec->gs[i];
            const len_t len = gs[0];
            hm_t *row = (hm_t *)malloc((unsigned long)(len + OFFSET) * sizeof(hm_t));
            memcpy(row + OFFSET, gs + 1, (unsigned long)len * sizeof(hm_t));
            row[LENGTH]  = len;
            row[PRELOOP] = len % UNROLL;
            row[COEFFS]  = i;
            row[MULT]    = 0;
            row[BINDEX]  = 0;
            row[DEG]     = gbs->ht->hd[row[OFFSET]].deg;
            uint32_t *cf = (uint32_t *)calloc((unsigned long)len * ML + 1, sizeof(uint32_t));
            for (l = 0; l < ML; ++l) {
                if (lb[l] == NULL || !(ml->act >> l & 1)) {
                    continue;
                }
                const hm_t * const g = lb[l]->hm[gi[i]];
                if (!ml_place(cf, gs, g + OFFSET, lb[l]->cf_32[g[COEFFS]],
                            g[LENGTH], l) || cf[l] != 1) {
                    ml->act &= ~((uint32_t)1 << l);
                }
            }
            ml->hm[i] = row;
            ml->cf[i] = cf;
        }
        ml->ld = ng;
    }
    for (l = 0; l < ML; ++l) {
        if (lb[l] != NULL) {
            free_basis_and_only_local_hash_table_data(&lb[l]);
        }
    }
    free(gi);
    if (!ok || ml->act == 0) {
        return 0;
    }

    /* the rounds of the run */
    for (r = 0; r < rec->nrd && ml->act != 0; ++r) {
        const sg_rrd_t * const rd = rec->rd + r;
        const len_t nr = rd->n;
        if (nr == 0) {
            continue;
        }
        const len_t nc = rd->nc;
        hm_t **piv      = (hm_t **)calloc((unsigned long)nc + 1, sizeof(hm_t *));
        uint32_t **pcf  = (uint32_t **)calloc((unsigned long)nc + 1, sizeof(uint32_t *));
        hm_t **own      = (hm_t **)calloc((unsigned long)(nr + 1), sizeof(hm_t *));
        uint64_t *dr    = (uint64_t *)calloc((unsigned long)nc * ML, sizeof(uint64_t));
        hm_t *rc        = (hm_t *)malloc((unsigned long)(nc + 1) * sizeof(hm_t));
        uint32_t *rv    = (uint32_t *)malloc((unsigned long)(nc + 1) * ML * sizeof(uint32_t));
        len_t nn = 0;
        for (i = 0; i < nr && ok; ++i) {
            const len_t src = rd->src[i];
            const len_t e   = (src & SG_GEN) ? (src & ~SG_GEN) : src + ng;
            /* the recorded row is read in place */
            hm_t *row = rd->rc[i];
            const len_t len = row[LENGTH];
            const uint32_t * const cf = ml->cf[e];
            const hm_t lc = row[OFFSET];
            if (rd->kind[i] == 0) {
                if (piv[lc] != NULL) {
                    ok = 0;
                    break;
                }
                piv[lc] = row;
                pcf[lc] = (uint32_t *)cf;
                continue;
            }
            for (j = 0; j < len; ++j) {
                for (l = 0; l < ML; ++l) {
                    dr[(unsigned long)row[OFFSET+j]*ML + l] = cf[(unsigned long)j*ML + l];
                }
            }
            const len_t nrem = ml_reduce_row(ml, dr, lc, nc, piv, pcf, rc, rv);
            if (nn >= rd->nn || nrem == 0) {
                ok = 0;
                break;
            }
            /* the remainder in the columns of the run, lanes with terms
             * elsewhere or without its lead term are bad */
            const hm_t * const ec = rd->ec[nn];
            const len_t el = ec[0];
            uint32_t *ncf = (uint32_t *)calloc((unsigned long)el * ML + 1, sizeof(uint32_t));
            for (l = 0; l < ML; ++l) {
                if (!(ml->act >> l & 1)) {
                    continue;
                }
                uint32_t c[nrem + 1];
                for (k = 0; k < nrem; ++k) {
                    c[k] = rv[(unsigned long)k*ML + l];
                }
                if (!ml_place(ncf, ec, rc, c, nrem, l) || ncf[l] == 0) {
                    ml->act &= ~((uint32_t)1 << l);
                }
            }
            if (ml->act == 0) {
                free(ncf);
                ok = 0;
                break;
            }
            ml_normalize(ml, ncf, el);
            /* the new element: the pivot of its lead column in this round,
             * with the support of the run in the basis */
            hm_t *prow = (hm_t *)malloc((unsigned long)(el + OFFSET) * sizeof(hm_t));
            prow[LENGTH]  = el;
            prow[PRELOOP] = el % UNROLL;
            memcpy(prow + OFFSET, ec + 1, (unsigned long)el * sizeof(hm_t));
            piv[ec[1]] = prow;
            pcf[ec[1]] = ncf;
            own[i] = prow;
            const hm_t * const eb = rd->eb[nn];
            hm_t *er = (hm_t *)malloc((unsigned long)(el + OFFSET) * sizeof(hm_t));
            memcpy(er + OFFSET, eb + 1, (unsigned long)el * sizeof(hm_t));
            er[LENGTH]  = el;
            er[PRELOOP] = el % UNROLL;
            er[COEFFS]  = ml->ld;
            er[MULT]    = 0;
            er[BINDEX]  = 0;
            er[DEG]     = gbs->ht->hd[er[OFFSET]].deg;
            ml->hm[ml->ld] = er;
            ml->cf[ml->ld] = ncf;
            ml->ld++;
            nn++;
        }
        if (ok && nn != rd->nn) {
            ok = 0;
        }
        for (k = 0; k < nr; ++k) {
            free(own[k]);
        }
        free(own);
        free(piv);
        free(pcf);
        free(dr);
        free(rc);
        free(rv);
        if (!ok) {
            return 0;
        }
    }
    return ml->act != 0;
}

/* A trace of F4 is applied without new monomials in the hash table
 * once the positions of the columns of all its matrices in it are known,
 * as after an application on that hash table, see ml_round(). A trace of
 * a signature-based run reduces the input generators first, which works
 * on that hash table, too. */
int core_f4_multi_read_only(
        const md_t * const gmd
        )
{
    const trace_t * const tr = gmd->tr;
    if (tr == NULL || tr->sgr != NULL || tr->tc == NULL) {
        return 0;
    }
    for (len_t r = 0; r <= tr->ltd; ++r) {
        const tc_t * const tc = __atomic_load_n(tr->tc + r, __ATOMIC_ACQUIRE);
        if (tc == NULL || tc->hc == NULL) {
            return 0;
        }
    }
    return 1;
}

/* Applies the trace gmd->tr at the primes p[0], ..., p[n-1] < 2^31,
 * n <= ML, in one pass, see above. All matrices of the trace must be
 * stored, as after its first application. out[l] gets the reduced basis
 * modulo p[l], or NULL if p[l] is a bad prime. Returns 0 if the trace
 * cannot be applied this way, then out is not set. */
int core_f4_multi(
        bs_t *gbs,
        md_t *gmd,
        const uint64_t * const p,
        const len_t n,
        bs_t **out
        )
{
    len_t i, j, l, r;

    trace_t *tr = gmd->tr;
    if (tr == NULL || (tr->sgr == NULL && tr->ech) || !tr->fin
            || tr->tc == NULL || n == 0 || n > ML || gbs->ht->boolean
            || gmd->nev > 0) {
        return 0;
    }
    for (r = 0; r <= tr->ltd; ++r) {
        const tc_t * const tc = __atomic_load_n(tr->tc + r, __ATOMIC_ACQUIRE);
        if (tc == NULL) {
            return 0;
        }
    }
    for (l = 0; l < n; ++l) {
        if (p[l] >= ((uint64_t)1 << 31)) {
            return 0;
        }
    }
    ht_t *bht = gbs->ht;
    ml_t ml;
    memset(&ml, 0, sizeof(ml_t));
    ml.nl = n;
    for (l = 0; l < ML; ++l) {
        ml.p[l]   = l < n ? p[l] : p[0];
        ml.big[l] = ((uint64_t)1 << 63) / ml.p[l] * ml.p[l];
        ml.mu[l]  = (uint64_t)(((unsigned __int128)1 << 64) / ml.p[l]);
    }
    ml.act = ((uint32_t)1 << n) - 1;

    if (tr->sgr != NULL) {
        /* the rounds of the signature-based run, then its final reduction */
        int ok = ml_sig_rounds(&ml, gbs, gmd, (const sg_rec_t *)tr->sgr);
        hm_t **fin    = NULL;
        uint32_t **fc = NULL;
        int64_t nf    = -1;
        if (ok && ml.act != 0) {
            nf = ml_round(&ml, tr->tc[tr->ltd], NULL, 0, tr, bht, &fin, &fc);
        }
        for (l = 0; l < n; ++l) {
            out[l] = nf >= 0 && (ml.act >> l & 1)
                ? ml_lane_basis(&ml, fin, fc, (len_t)nf, l, bht) : NULL;
        }
        for (i = 0; nf > 0 && i < (len_t)nf; ++i) {
            free(fin[i]);
            free(fc[i]);
        }
        free(fin);
        free(fc);
        for (i = 0; i < ml.ld; ++i) {
            free(ml.hm[i]);
        }
        for (i = 0; i < ml.ncf; ++i) {
            free(ml.cf[i]);
        }
        free(ml.hm);
        free(ml.cf);
        /* the trace applies, bad lanes or not */
        return 1;
    }

    /* the input generators, monic in each lane */
    const len_t ng = (len_t)gmd->ngens;
    len_t ne = ng + 1;
    for (r = 0; r < tr->ltd; ++r) {
        ne += tr->td[r].nlm;
    }
    ml.sz  = ne;
    ml.hm  = (hm_t **)malloc((unsigned long)ml.sz * sizeof(hm_t *));
    ml.ncf = ne;
    ml.cf  = (uint32_t **)calloc((unsigned long)ml.ncf, sizeof(uint32_t *));
    for (i = 0; i < ng; ++i) {
        const hm_t * const g = gbs->hm[i];
        const len_t len = g[LENGTH];
        const len_t idx = g[COEFFS];
        ml.hm[i] = (hm_t *)malloc((unsigned long)(len + OFFSET) * sizeof(hm_t));
        memcpy(ml.hm[i], g, (unsigned long)(len + OFFSET) * sizeof(hm_t));
        uint32_t *cf = (uint32_t *)malloc((unsigned long)(len + 1) * ML * sizeof(uint32_t));
        for (j = 0; j < len; ++j) {
            for (l = 0; l < ML; ++l) {
                cf[(unsigned long)j*ML + l] = (uint32_t)mpz_fdiv_ui(
                        gbs->cf_qq[idx][j], ml.p[l]);
            }
        }
        ml_normalize(&ml, cf, len);
        ml.cf[idx] = cf;
    }
    ml.ld = ng;
    if (tr->gperm != NULL) {
        hm_t **h = (hm_t **)malloc((unsigned long)ng * sizeof(hm_t *));
        for (i = 0; i < ng; ++i) {
            h[i] = ml.hm[tr->gperm[i]];
        }
        memcpy(ml.hm, h, (unsigned long)ng * sizeof(hm_t *));
        free(h);
    }

    int ok = 1;
    for (r = 0; r < tr->ltd && ok && ml.act != 0; ++r) {
        const td_t * const td = tr->td + r;
        ok = ml_round(&ml, tr->tc[r], td->nlms, td->nlm, tr, bht,
                NULL, NULL) == (int64_t)td->nlm;
    }
    hm_t **fin    = NULL;
    uint32_t **fc = NULL;
    int64_t nf    = -1;
    if (ok && ml.act != 0) {
        nf = ml_round(&ml, tr->tc[tr->ltd], NULL, 0, tr, bht, &fin, &fc);
    }
    if (nf >= 0) {
        for (l = 0; l < n; ++l) {
            out[l] = (ml.act >> l & 1)
                ? ml_lane_basis(&ml, fin, fc, (len_t)nf, l, bht) : NULL;
        }
        for (i = 0; i < (len_t)nf; ++i) {
            free(fin[i]);
            free(fc[i]);
        }
    } else {
        for (l = 0; l < n; ++l) {
            out[l] = NULL;
        }
    }
    free(fin);
    free(fc);
    for (i = 0; i < ml.ld; ++i) {
        free(ml.hm[i]);
    }
    for (i = 0; i < ml.ncf; ++i) {
        free(ml.cf[i]);
    }
    free(ml.hm);
    free(ml.cf);
    return 1;
}
