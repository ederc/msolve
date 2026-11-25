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
    if (tc->hc == NULL) {
        return -1;
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
                            && tc->hc[np[i][OFFSET] - ncl] == nlms[k]) {
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
                        const hm_t h = tc->hc[row[OFFSET+j] - ncl];
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
                        er[OFFSET+j] = tc->hc[row[OFFSET+j] - ncl];
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
                        : tc->hc[row[OFFSET+j] - ncl];
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
    if (tr == NULL || tr->sgr != NULL || tr->ech || !tr->fin
            || tr->tc == NULL || n == 0 || n > ML || gbs->ht->boolean
            || gmd->nev > 0) {
        return 0;
    }
    for (r = 0; r <= tr->ltd; ++r) {
        const tc_t * const tc = __atomic_load_n(tr->tc + r, __ATOMIC_ACQUIRE);
        if (tc == NULL || tc->hc == NULL) {
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
