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

/* Signature-based F4: a batched F5 whose rounds are F4 matrices.
 *
 * Each input generator f_i gets the signature e_i, a multiple t*e_i of it
 * is compared in Schreyer's order: by the monomial t*lm(f_i), then by the
 * index, e_i > e_j for i > j. Each round takes all pending signatures of
 * least degree, the signatures of the S-pairs, and builds one matrix:
 * the row at a signature s is a*g for the basis element g of largest
 * sig-lead ratio whose signature divides s (rewriting). Symbolic
 * preprocessing adds for each monomial a reducer, a multiple of the
 * basis element of least sig-lead ratio whose lead term divides it. All
 * rows are sorted by signature and reduced in this order, each row only
 * by pivots of smaller signature (F4-RB). A row reducing to zero gives a
 * syzygy signature, any other reduced row a new element. Syzygy
 * signatures also come from the Koszul syzygies of the input generators
 * and from S-pairs with coprime lead terms; no pending signature divisible
 * by a syzygy signature of the same index is kept.
 *
 * Where the signature run does not pay off, it hands over: the plain F4
 * algorithm starts from the input generators. The tests read the S-pairs
 * an update would form against the work of the round,
 * the zero reductions and the divisible new elements at the first rounds
 * whose signature degree passes the lead degree by 2 resp. 4, and the
 * divisible new elements of each round.
 *
 * References: Eder, Faugere, A survey on signature-based algorithms for
 * computing Groebner bases (2017), sections 6, 7, 13, 14; Roune,
 * Stillman, Practical Groebner basis computation (2012); Arri, Perry,
 * The F5 criterion revised (2011). */

/* handover parameters */
#define SG_PAIRS        1000  /* least S-pairs of an update to be tested */
/* S-pairs per multiply-add handing over, see sg_hand_over() */
#define SG_PAIR_WORK_NUM 3
#define SG_PAIR_WORK_DEN 2
/* a round is stopped once its reductions take more than SG_WORK_GROWTH
 * times the multiply-adds of the round before and more than SG_WORK_MIN,
 * see sg_round(). Dense quadratic systems over GF(65521) grow by up to 22
 * times in their fifth degree, where F4 has many more zero reductions,
 * while the rounds handing over better grow by more than 27 times. */
#define SG_WORK_GROWTH 24
#define SG_WORK_MIN ((uint64_t)1 << 27)
#define SG_EARLY        2     /* gap between signature and lead degree */
#define SG_GAP          4
#define SG_EARLY_ROUND  25    /* least new elements of the early round */
/* the early round hands over for fewer zero reductions than this
 * percentage of its new elements, see sg_hand_over() */
#define SG_EARLY_ZERO        25
#define SG_EARLY_ZERO_LEARN  40
#define SG_OUTGROWN_MIN 5     /* least new elements of a round tested */

/* Packed monomials: the exponents of the variables in lanes of SG_LB bits,
 * SG_LPW lanes in a 64-bit word, the low lanes first, the lanes past the
 * last variable 0. Every lane stays below 2^(SG_LB-1), so operations on
 * whole words act on all lanes at once without carries or borrows between
 * lanes: a product is a sum of words, divisibility the absence of a borrow
 * in the difference. A degree past SG_LMAX hands over to F4. */
#ifdef EXP32
#define SG_LB   32
#else
#define SG_LB   16
#endif
#define SG_LPW  (64 / SG_LB)
#define SG_LANE (~(uint64_t)0 >> (64 - SG_LB))
#define SG_ONES (~(uint64_t)0 / SG_LANE)
#define SG_HIGH (SG_ONES << (SG_LB - 1))
#define SG_LMAX ((deg_t)((SG_LANE >> 1)))

/* the top bit of each lane where the lane of a is at least the one of b */
static inline uint64_t sg_ge(
        const uint64_t a,
        const uint64_t b
        )
{
    return ((a | SG_HIGH) - b) & SG_HIGH;
}

/* each lane whose top bit is set in h filled with ones */
static inline uint64_t sg_widen(
        const uint64_t h
        )
{
    return (h >> (SG_LB - 1)) * SG_LANE;
}

static inline uint64_t sg_max(
        const uint64_t a,
        const uint64_t b
        )
{
    const uint64_t m = sg_widen(sg_ge(a, b));
    return (a & m) | (b & ~m);
}

/* the top bit of each nonzero lane */
static inline uint64_t sg_nonzero(
        const uint64_t a
        )
{
    return (a + (SG_HIGH - SG_ONES)) & SG_HIGH;
}

/* the sum of the lanes */
static inline deg_t sg_lane_sum(
        uint64_t w
        )
{
#if SG_LB == 16
    w = (w & 0x0000FFFF0000FFFFULL) + ((w >> 16) & 0x0000FFFF0000FFFFULL);
#endif
    return (deg_t)((w & 0xFFFFFFFFULL) + (w >> 32));
}

/* does a divide b? */
static inline int sg_divides_packed(
        const uint64_t * const a,
        const uint64_t * const b,
        const len_t nw
        )
{
    len_t i;
    for (i = 0; i < nw; ++i) {
        if ((b[i] - a[i]) & SG_HIGH) {
            return 0;
        }
    }
    return 1;
}

/* packs the variables of the exponent vector e, returns its degree */
static inline deg_t sg_pack(
        const exp_t * const e,
        uint64_t *w,
        const len_t evl,
        const len_t nw
        )
{
    len_t i;
    deg_t d = 0;
    memset(w, 0, (unsigned long)nw * sizeof(uint64_t));
    for (i = 1; i < evl; ++i) {
        w[(i-1) / SG_LPW] |= (uint64_t)e[i] << (SG_LB * ((i-1) % SG_LPW));
        d += e[i];
    }
    return d;
}

/* the exponent vector of the packed monomial w, with its degree */
static inline void sg_unpack(
        const uint64_t * const w,
        exp_t *e,
        const len_t evl
        )
{
    len_t i;
    deg_t d = 0;
    for (i = 1; i < evl; ++i) {
        e[i] = (exp_t)((w[(i-1) / SG_LPW] >> (SG_LB * ((i-1) % SG_LPW))) & SG_LANE);
        d   += e[i];
    }
    e[0] = (exp_t)d;
}

/* the packed lead term and module monomial of the signature of element n */
#define SG_PL(sg, n) ((sg)->pw + (unsigned long)(n) * 2 * (sg)->nw)
#define SG_PS(sg, n) ((sg)->pw + ((unsigned long)(n) * 2 + 1) * (sg)->nw)

/* the syzygy signatures of one index: their packed module monomials, each
 * with a divisor mask of 64 bits */
typedef struct sg_syz_t sg_syz_t;
struct sg_syz_t
{
    uint64_t *m;  /* divisor masks */
    uint64_t *w;  /* packed monomials, one after the other */
    len_t ld;
    len_t sz;
};

typedef struct sg_el_t sg_el_t;
struct sg_el_t
{
    hm_t *p;      /* terms as hashes of the basis hash table, decreasing,
                   * with the header of basis elements */
    cf32_t *cf;   /* monic coefficients */
    len_t ix;     /* index of the signature */
    hm_t sm;      /* module monomial of the signature */
    hm_t sc;      /* Schreyer monomial sm * lm(f_ix) */
    uint64_t tau; /* rank of the class (sig-lead ratio, index): elements
                   * compare as their classes iff their ranks do */
};

typedef struct sg_sig_t sg_sig_t;
struct sg_sig_t
{
    len_t ix;
    hm_t sm;
    hm_t sc;
    deg_t deg;    /* degree of sc */
};

typedef struct sg_row_t sg_row_t;
struct sg_row_t
{
    hm_t *r;            /* terms, first as hashes of the symbolic hash
                         * table, then as columns */
    const cf32_t *cf;   /* coefficients of the multiplied element */
    len_t ix;
    hm_t sm;            /* the signature in the round's table sg->rst, its
                         * Schreyer signature is derived where the row
                         * gives a new element */
    int8_t reducer;     /* 1 for rows from symbolic preprocessing */
    int8_t gen;         /* 1 for a given row, a reductant of the round */
    len_t id;           /* build order: the given rows first, then the
                         * rows of the symbolic preprocessing */
    len_t src;          /* the element multiplied, SG_GEN | i for the input
                         * generator i */
    hm_t mul;           /* the multiplier in the basis hash table where the
                         * run is recorded */
    hm_t *pr;           /* if not NULL, the remainder of a row to be
                         * reduced modulo the reducer rows before it, with
                         * its coefficients in pv, see
                         * sg_reduce_by_reducers() */
    cf32_t *pv;
};

/* The record of a run for the application at other primes: for each round
 * the rows that set a pivot or gave a new element, in the order of their
 * reduction, with the expected lead terms of the new elements. Rows
 * reducing to zero are not recorded. */
#define SG_GEN ((len_t)1 << 31)

typedef struct sg_rrd_t sg_rrd_t;
struct sg_rrd_t
{
    len_t n;        /* rows */
    len_t sz;
    len_t *src;     /* element multiplied, see sg_row_t */
    hm_t *mul;      /* multiplier in the basis hash table */
    uint8_t *kind;  /* 0: sets the pivot of its lead column,
                     * 1: reduced by the pivots before it, a new element */
    len_t nn;       /* new elements */
    hm_t *lm;       /* their lead terms in the basis hash table */
    /* the matrix of the run, applied where the elements have the supports
     * of the run: the columns of each row, the columns and terms of each
     * new element, each array led by its length */
    len_t nc;       /* columns */
    hm_t **rc;
    hm_t **ec;
    hm_t **eb;
};

typedef struct sg_rec_t sg_rec_t;
struct sg_rec_t
{
    len_t nrd;
    len_t szrd;
    sg_rrd_t *rd;
    len_t ng;       /* the supports of the input generators, led by their */
    hm_t **gs;      /* lengths, in the order of their signatures */
};

static void sg_rec_free(
        void *recp
        )
{
    len_t i;
    sg_rec_t *rec = (sg_rec_t *)recp;
    if (rec == NULL) {
        return;
    }
    for (i = 0; i < rec->nrd; ++i) {
        len_t j;
        for (j = 0; j < rec->rd[i].n; ++j) {
            free(rec->rd[i].rc[j]);
        }
        for (j = 0; j < rec->rd[i].nn; ++j) {
            free(rec->rd[i].ec[j]);
            free(rec->rd[i].eb[j]);
        }
        free(rec->rd[i].rc);
        free(rec->rd[i].ec);
        free(rec->rd[i].eb);
        free(rec->rd[i].src);
        free(rec->rd[i].mul);
        free(rec->rd[i].kind);
        free(rec->rd[i].lm);
    }
    free(rec->rd);
    for (i = 0; i < rec->ng; ++i) {
        free(rec->gs[i]);
    }
    free(rec->gs);
    free(rec);
}

/* The lead terms of the rows of the record which gave new elements, in
 * the basis hash table bht, sorted and without repetitions: these rows
 * are rows of S-pairs, so their lead terms are the lcms of the S-pairs
 * the run did not find to reduce to zero. Their number is stored in n. */
hm_t *sig_record_lead_terms(
        const void * const recp,
        ht_t *bht,
        len_t *n
        )
{
    len_t i, j, r;
    const sg_rec_t * const rec = (const sg_rec_t *)recp;
    const len_t evl = bht->evl;

    /* the lead terms of the new elements in the order of the elements */
    len_t nel = 0, ne = 0, nl = 0, szl = 0;
    for (r = 0; r < rec->nrd; ++r) {
        nel += rec->rd[r].nn;
    }
    hm_t *elm = (hm_t *)malloc((unsigned long)(nel + 1) * sizeof(hm_t));
    for (r = 0; r < rec->nrd; ++r) {
        for (i = 0; i < rec->rd[r].nn; ++i) {
            elm[ne++] = rec->rd[r].lm[i];
        }
    }
    hm_t *l = NULL;
    exp_t *e = (exp_t *)malloc((unsigned long)evl * sizeof(exp_t));
    for (r = 0; r < rec->nrd; ++r) {
        const sg_rrd_t * const rd = rec->rd + r;
        for (i = 0; i < rd->n; ++i) {
            if (rd->kind[i] == 0) {
                continue;
            }
            const len_t src = rd->src[i];
            hm_t lt;
            if (src & SG_GEN) {
                if (rec->gs == NULL) {
                    continue;
                }
                lt = rec->gs[src & ~SG_GEN][1];
            } else {
                if (src >= nel) {
                    continue;
                }
                lt = elm[src];
            }
            const exp_t * const em = bht->ev[rd->mul[i]];
            const exp_t * const el = bht->ev[lt];
            for (j = 0; j < evl; ++j) {
                e[j] = em[j] + el[j];
            }
            if (bht->eld == bht->esz-1) {
                enlarge_hash_table(bht);
            }
            if (nl == szl) {
                szl = szl > 0 ? 2 * szl : 256;
                l   = (hm_t *)realloc(l, (unsigned long)szl * sizeof(hm_t));
            }
            l[nl++] = insert_in_hash_table(e, bht);
        }
    }
    free(e);
    free(elm);
    if (nl > 0) {
        qsort(l, (unsigned long)nl, sizeof(hm_t), hash_cmp);
        for (i = 1, j = 1; i < nl; ++i) {
            if (l[i] != l[j-1]) {
                l[j++] = l[i];
            }
        }
        nl = j;
    }
    *n = nl;
    return l;
}

/* a copy of the n entries h[0], ..., h[n-1] led by n */
static hm_t *sg_copy_terms(
        const hm_t * const h,
        const len_t n
        )
{
    hm_t *c = (hm_t *)malloc((unsigned long)(n + 1) * sizeof(hm_t));
    c[0] = n;
    memcpy(c + 1, h, (unsigned long)n * sizeof(hm_t));
    return c;
}

/* a copy of the row, with its header, read in place by the replays */
static hm_t *sg_copy_row(
        const hm_t * const row
        )
{
    const len_t n = row[LENGTH];
    hm_t *c = (hm_t *)calloc((unsigned long)(n + OFFSET), sizeof(hm_t));
    c[LENGTH]  = n;
    c[PRELOOP] = n % UNROLL;
    memcpy(c + OFFSET, row + OFFSET, (unsigned long)n * sizeof(hm_t));
    return c;
}

static void sg_rec_add(
        sg_rrd_t *rd,
        const len_t src,
        const hm_t mul,
        const uint8_t kind,
        const hm_t * const row
        )
{
    if (rd->n >= rd->sz) {
        rd->sz   = rd->sz > 0 ? 2 * rd->sz : 64;
        rd->rc   = realloc(rd->rc, (unsigned long)rd->sz * sizeof(hm_t *));
        rd->src  = realloc(rd->src, (unsigned long)rd->sz * sizeof(len_t));
        rd->mul  = realloc(rd->mul, (unsigned long)rd->sz * sizeof(hm_t));
        rd->kind = realloc(rd->kind, (unsigned long)rd->sz * sizeof(uint8_t));
    }
    rd->src[rd->n]  = src;
    rd->mul[rd->n]  = mul;
    rd->kind[rd->n] = kind;
    rd->rc[rd->n]   = sg_copy_row(row);
    rd->n++;
}

typedef struct sg_t sg_t;
struct sg_t
{
    sg_el_t *e;         /* elements of the signature basis */
    len_t ld;
    len_t sz;
    len_t **ixe;        /* elements of each index */
    len_t *ixld;
    len_t *ixsz;
    sg_el_t *g;         /* input generators with signature e_i */
    len_t ng;
    sg_syz_t *syz;      /* syzygy signatures of each index */
    len_t *syzhit;      /* the syzygy of each index that divided last */
    sg_sig_t *ps;       /* pending signatures */
    len_t pld;
    len_t psz;
    uint32_t fc;
    ht_t *bht;
    ht_t *sht;
    /* the signatures of the rows of a round: most rows are reducer rows,
     * their signatures are not kept in bht */
    ht_t *rst;
    exp_t *e0, *e1, *e2, *e3; /* scratch exponent vectors */
    /* handover state */
    int early_done;
    int gap_done;
    /* statistics */
    uint64_t nsubs;     /* multiply-adds of the current round */
    uint64_t psubs;     /* multiply-adds of the previous round */
    len_t nzero;        /* rows reduced to zero in the current round */
    len_t ndiv;         /* new elements whose lead term is divisible
                         * by a lead term from before the round */
    len_t maxgap;       /* largest signature degree minus lead degree */
    uint64_t cnt_pairs, cnt_singular, cnt_syz, cnt_koszul, cnt_queued, cnt_base;
    uint64_t **tri;     /* triangle of bits, see sg_tri_get() */
    uint64_t nearly;    /* rows dropped before the matrix is built */
    uint64_t seed;      /* random state of the zero tests */
    uint64_t ntested;   /* rows found zero by random combinations */
    sg_rec_t *rec;      /* the record of the run, NULL if not recorded */
    len_t *rord;        /* the elements by increasing rank */
    len_t nw;           /* words of a packed monomial */
    uint64_t *gpl;      /* packed lead terms of the input generators */
    uint64_t *pw;       /* packed lead term and packed module monomial of
                         * the signature of each element */
    int wide;           /* 1 if a degree passes SG_LMAX */
    int nthrds;         /* threads forming the S-pairs, see sg_new_pairs() */
    len_t trisz;
};

static inline hm_t sg_insert(
        const exp_t *e,
        ht_t *ht
        )
{
    if (ht->eld >= ht->esz) {
        enlarge_hash_table(ht);
    }
    return insert_in_hash_table(e, ht);
}

/* degree reverse lexicographical comparison of exponent vectors,
 * as monomial_cmp_drl() */
static inline int sg_cmp_exp(
        const exp_t * const a,
        const exp_t * const b,
        const len_t evl
        )
{
    len_t i;
    if (a[0] != b[0]) {
        return a[0] > b[0] ? 1 : -1;
    }
    for (i = evl-1; i > 0; --i) {
        if (a[i] != b[i]) {
            return a[i] < b[i] ? 1 : -1;
        }
    }
    return 0;
}

static inline int sg_divides_exp(
        const exp_t * const a, /* divisor */
        const exp_t * const b,
        const len_t evl
        )
{
    len_t i;
    for (i = 0; i < evl; ++i) {
        if (a[i] > b[i]) {
            return 0;
        }
    }
    return 1;
}

/* signature comparison in Schreyer's order */
static inline int sg_cmp_sig(
        const len_t ia,
        const hm_t sca,
        const len_t ib,
        const hm_t scb,
        const ht_t * const ht
        )
{
    const int c = monomial_cmp_drl(sca, scb, ht);
    if (c != 0) {
        return c > 0 ? 1 : -1;
    }
    return ia == ib ? 0 : (ia > ib ? 1 : -1);
}

/* compares the sig-lead ratios sc_a/lm(a) and sc_b/lm(b) */
static inline int sg_cmp_ratio(
        const sg_el_t * const a,
        const sg_el_t * const b,
        sg_t *sg
        )
{
    len_t j;
    const ht_t * const ht = sg->bht;
    const len_t evl = ht->evl;
    const exp_t * const sa = ht->ev[a->sc];
    const exp_t * const sb = ht->ev[b->sc];
    const exp_t * const la = ht->ev[a->p[OFFSET]];
    const exp_t * const lb = ht->ev[b->p[OFFSET]];
    for (j = 0; j < evl; ++j) {
        sg->e0[j] = sa[j] + lb[j];
        sg->e1[j] = sb[j] + la[j];
    }
    const int c = sg_cmp_exp(sg->e0, sg->e1, evl);
    if (c != 0) {
        return c;
    }
    return a->ix == b->ix ? 0 : (a->ix > b->ix ? 1 : -1);
}

/* divisor mask of 64 bits: with n <= 64 variables each variable takes
 * 64/n bits, bit k set where its exponent passes k, past 64 variables
 * variable i sets bit i modulo 64 where its exponent is not zero. The
 * thresholds do not depend on the input, so the masks keep filtering at
 * higher degrees. */
static inline uint64_t sg_mask(
        const uint64_t * const w,
        const len_t nv
        )
{
    len_t i;
    uint64_t m = 0;
    if (nv > 64) {
        for (i = 0; i < nv; ++i) {
            if ((w[i / SG_LPW] >> (SG_LB * (i % SG_LPW))) & SG_LANE) {
                m |= (uint64_t)1 << (i % 64);
            }
        }
        return m;
    }
    const len_t b = 64 / nv;
    for (i = 0; i < nv; ++i) {
        const uint64_t e = (w[i / SG_LPW] >> (SG_LB * (i % SG_LPW))) & SG_LANE;
        const len_t x    = e < b ? (len_t)e : b;
        const uint64_t f = x >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << x) - 1);
        m |= f << (i * b);
    }
    return m;
}

/* is the signature with index ix and packed module monomial w divisible
 * by a syzygy signature? */
static inline int sg_is_syzygy(
        const sg_t * const sg,
        const len_t ix,
        const uint64_t * const w
        )
{
    len_t i;
    const len_t nw  = sg->nw;
    const sg_syz_t * const s = sg->syz + ix;
    if (s->ld == 0) {
        return 0;
    }
    const uint64_t nm = ~sg_mask(w, sg->bht->nv);
    /* the syzygy that divided last is tried first */
    const len_t h = sg->syzhit[ix];
    if (!(s->m[h] & nm) && sg_divides_packed(s->w + (unsigned long)h * nw, w, nw)) {
        return 1;
    }
    for (i = 0; i < s->ld; ++i) {
        if (s->m[i] & nm) {
            continue;
        }
        if (sg_divides_packed(s->w + (unsigned long)i * nw, w, nw)) {
            /* the syzygies dividing often move to the front */
            const len_t f = i / 2;
            if (f < i) {
                sg_syz_t *u = ((sg_t *)sg)->syz + ix;
                len_t j;
                const uint64_t tm = u->m[f];
                u->m[f] = u->m[i];
                u->m[i] = tm;
                for (j = 0; j < nw; ++j) {
                    const uint64_t t = u->w[(unsigned long)f * nw + j];
                    u->w[(unsigned long)f * nw + j] = u->w[(unsigned long)i * nw + j];
                    u->w[(unsigned long)i * nw + j] = t;
                }
            }
            ((sg_t *)sg)->syzhit[ix] = f;
            return 1;
        }
    }
    return 0;
}

/* sg_is_syzygy() without reordering the syzygies, for several threads */
static inline int sg_is_syzygy_ro(
        const sg_t * const sg,
        const len_t ix,
        const uint64_t * const w
        )
{
    len_t i;
    const len_t nw  = sg->nw;
    const sg_syz_t * const s = sg->syz + ix;
    const uint64_t nm = ~sg_mask(w, sg->bht->nv);
    for (i = 0; i < s->ld; ++i) {
        if (!(s->m[i] & nm)
                && sg_divides_packed(s->w + (unsigned long)i * nw, w, nw)) {
            return 1;
        }
    }
    return 0;
}

static void sg_add_syzygy(
        sg_t *sg,
        const len_t ix,
        const uint64_t * const w
        )
{
    if (sg_is_syzygy(sg, ix, w)) {
        return;
    }
    const len_t nw = sg->nw;
    sg_syz_t *s = sg->syz + ix;
    /* the syzygies stay minimal: the multiples of the new one go */
    const uint64_t m = sg_mask(w, sg->bht->nv);
    len_t i, k = 0;
    for (i = 0; i < s->ld; ++i) {
        const uint64_t * const f = s->w + (unsigned long)i * nw;
        if (!(m & ~s->m[i]) && sg_divides_packed(w, f, nw)) {
            continue;
        }
        if (k < i) {
            s->m[k] = s->m[i];
            memmove(s->w + (unsigned long)k * nw, f, (unsigned long)nw * sizeof(uint64_t));
        }
        k++;
    }
    s->ld = k;
    if (sg->syzhit[ix] >= k) {
        sg->syzhit[ix] = 0;
    }
    if (s->ld >= s->sz) {
        s->sz = s->sz > 0 ? 2 * s->sz : 8;
        s->m  = realloc(s->m, (unsigned long)s->sz * sizeof(uint64_t));
        s->w  = realloc(s->w, (unsigned long)s->sz * nw * sizeof(uint64_t));
    }
    memcpy(s->w + (unsigned long)s->ld * nw, w, (unsigned long)nw * sizeof(uint64_t));
    s->m[s->ld] = m;
    s->ld++;
}

/* the same for an exponent vector */
static inline int sg_is_syzygy_exp(
        sg_t *sg,
        const len_t ix,
        const exp_t * const e
        )
{
    uint64_t w[sg->nw];
    sg_pack(e, w, sg->bht->evl, sg->nw);
    return sg_is_syzygy(sg, ix, w);
}

static void sg_add_syzygy_exp(
        sg_t *sg,
        const len_t ix,
        const exp_t * const e
        )
{
    uint64_t w[sg->nw];
    if (sg_pack(e, w, sg->bht->evl, sg->nw) > SG_LMAX) {
        sg->wide = 1;
        return;
    }
    sg_add_syzygy(sg, ix, w);
}

static void sg_add_pending(
        sg_t *sg,
        const len_t ix,
        const exp_t * const sm,
        const exp_t * const sc
        )
{
    if (sg->pld >= sg->psz) {
        sg->psz = 2 * sg->psz;
        sg->ps  = realloc(sg->ps, (unsigned long)sg->psz * sizeof(sg_sig_t));
    }
    sg_sig_t *s = sg->ps + sg->pld;
    s->ix  = ix;
    /* the degree of the signature itself: the rounds take the pending
     * signatures of least degree (eder-2013a §3 -- no polynomial degree
     * preselection, the sugar degree of the signature) */
    s->deg = sm[0];
    /* sm and sc may point into the hash table, the insertions may
     * move it */
    memcpy(sg->e2, sm, (unsigned long)sg->bht->evl * sizeof(exp_t));
    memcpy(sg->e3, sc, (unsigned long)sg->bht->evl * sizeof(exp_t));
    s->sm  = sg_insert(sg->e2, sg->bht);
    s->sc  = sg_insert(sg->e3, sg->bht);
    sg->pld++;
}

/* inserts the element n in the elements ordered by their classes (sig-lead
 * ratio, index) and gives it the rank of its class: the one of an equal
 * class, otherwise the middle of the ranks of its neighbours; where there
 * is no room between them, all classes are ranked anew with equal gaps */
static void sg_rank_element(
        sg_t *sg,
        const len_t n
        )
{
    len_t lo = 0, hi = n, i;
    sg_el_t *e = sg->e;
    int c = 1;
    while (lo < hi) {
        const len_t mid = lo + (hi - lo) / 2;
        c = sg_cmp_ratio(e + n, e + sg->rord[mid], sg);
        if (c == 0) {
            lo = mid;
            break;
        }
        if (c > 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    memmove(sg->rord + lo + 1, sg->rord + lo, (unsigned long)(n - lo) * sizeof(len_t));
    sg->rord[lo] = n;
    if (c == 0 && lo < hi) {
        e[n].tau = e[sg->rord[lo+1]].tau;
        return;
    }
    const uint64_t a = lo > 0 ? e[sg->rord[lo-1]].tau : 0;
    const uint64_t b = lo < n ? e[sg->rord[lo+1]].tau : UINT64_MAX;
    if (b - a >= 2) {
        e[n].tau = a + (b - a) / 2;
        return;
    }
    /* rank all classes anew */
    len_t ncl = 0;
    for (i = 0; i <= n; ++i) {
        if (i == 0 || sg_cmp_ratio(e + sg->rord[i], e + sg->rord[i-1], sg) != 0) {
            ncl++;
        }
    }
    const uint64_t step = UINT64_MAX / (ncl + 1);
    uint64_t t = 0;
    for (i = 0; i <= n; ++i) {
        if (i == 0 || sg_cmp_ratio(e + sg->rord[i], e + sg->rord[i-1], sg) != 0) {
            t += step;
        }
        e[sg->rord[i]].tau = t;
    }
}

static void sg_add_element(
        sg_t *sg,
        hm_t *p,
        cf32_t *cf,
        const len_t ix,
        const hm_t sm,
        const hm_t sc
        )
{
    if (sg->ld >= sg->sz) {
        sg->sz   = 2 * sg->sz;
        sg->e    = realloc(sg->e, (unsigned long)sg->sz * sizeof(sg_el_t));
        sg->rord = realloc(sg->rord, (unsigned long)sg->sz * sizeof(len_t));
        sg->pw   = realloc(sg->pw, (unsigned long)sg->sz * 2 * sg->nw * sizeof(uint64_t));
    }
    if (sg_pack(sg->bht->ev[p[OFFSET]], SG_PL(sg, sg->ld), sg->bht->evl, sg->nw) > SG_LMAX
            || sg_pack(sg->bht->ev[sm], SG_PS(sg, sg->ld), sg->bht->evl, sg->nw) > SG_LMAX) {
        sg->wide = 1;
    }
    sg_el_t *e = sg->e + sg->ld;
    e->p  = p;
    e->cf = cf;
    e->ix = ix;
    e->sm = sm;
    e->sc = sc;
    sg_rank_element(sg, sg->ld);
    if (sg->ixld[ix] >= sg->ixsz[ix]) {
        sg->ixsz[ix] = sg->ixsz[ix] > 0 ? 2 * sg->ixsz[ix] : 8;
        sg->ixe[ix]  = realloc(sg->ixe[ix],
                (unsigned long)sg->ixsz[ix] * sizeof(len_t));
    }
    sg->ixe[ix][sg->ixld[ix]++] = sg->ld;
    sg->ld++;
}

/* the syzygy f_i * h - h * e_i of the new element h = sg->e[n] and each
 * input generator f_i, written as module element: its terms are
 * lm(f_i) * sig(h) with Schreyer monomial lm(f_i) * sc(h) and lm(h) * e_i
 * with Schreyer monomial lm(h) * lm(f_i), the larger one is its
 * signature. If both are the same module term they may cancel. */
static void sg_principal_syzygies(
        sg_t *sg,
        const len_t n
        )
{
    len_t i, j;
    ht_t *ht         = sg->bht;
    const len_t nw   = sg->nw;
    const sg_el_t *h = sg->e + n;
    uint64_t w[nw];
    /* lm(f_i) * sc(h) against lm(h) * lm(f_i) is sc(h) against lm(h) */
    const int c0 = sg_cmp_exp(ht->ev[h->sc], ht->ev[h->p[OFFSET]], ht->evl);
    const deg_t dsm = ht->hd[h->sm].deg;

    for (i = 0; i < sg->ng; ++i) {
        int c = c0;
        if (c == 0) {
            if (h->ix == i) {
                continue;
            }
            c = h->ix > i ? 1 : -1;
        }
        if (c > 0) {
            if (dsm + ht->hd[sg->g[i].p[OFFSET]].deg > SG_LMAX) {
                sg->wide = 1;
                return;
            }
            const uint64_t * const a = sg->gpl + (unsigned long)i * nw;
            const uint64_t * const b = SG_PS(sg, n);
            for (j = 0; j < nw; ++j) {
                w[j] = a[j] + b[j];
            }
            sg_add_syzygy(sg, h->ix, w);
        } else {
            sg_add_syzygy(sg, i, SG_PL(sg, n));
        }
    }
}

/* the signature of the S-pair of the elements a and b: returns 0 for a
 * singular S-pair, otherwise sets the index and the module monomial of
 * the signature and if the lead terms are coprime. With L the lcm of the
 * lead terms, the two candidates are L times the sig-lead ratios of a
 * and b, so the ranks of their classes tell which one is larger, and only
 * that one is formed. */
static inline int sg_pair_signature(
        const sg_t * const sg,
        const len_t a,
        const len_t b,
        len_t *ix,
        uint64_t *sm,
        int *coprime
        )
{
    len_t j;
    const len_t nw = sg->nw;
    const sg_el_t * const g = sg->e + a;
    const sg_el_t * const h = sg->e + b;
    if (g->tau == h->tau) {
        return 0;
    }
    const len_t x = g->tau > h->tau ? a : b;
    const len_t y = g->tau > h->tau ? b : a;
    const uint64_t * const lx = SG_PL(sg, x);
    const uint64_t * const ly = SG_PL(sg, y);
    const uint64_t * const sx = SG_PS(sg, x);
    deg_t d   = 0;
    uint64_t cp = 0;
    for (j = 0; j < nw; ++j) {
        /* each lane of sm is sx + max(lx, ly) - lx, the degree of sm
         * bounds them */
        sm[j] = sx[j] + sg_max(lx[j], ly[j]) - lx[j];
        d    += sg_lane_sum(sm[j]);
        cp   |= sg_nonzero(lx[j]) & sg_nonzero(ly[j]);
    }
    if (d > SG_LMAX) {
        ((sg_t *)sg)->wide = 1;
        return 0;
    }
    *ix      = sg->e[x].ix;
    *coprime = cp == 0;
    return 1;
}

/* the triangle of bits: is the signature of the S-pair of a and b known
 * to be a syzygy signature? */
static inline int sg_tri_get(
        const sg_t * const sg,
        len_t a,
        len_t b
        )
{
    if (a < b) {
        const len_t t = a;
        a = b;
        b = t;
    }
    return (sg->tri[a][b / 64] >> (b % 64)) & 1;
}

static inline void sg_tri_set(
        sg_t *sg,
        const len_t a,  /* a > b */
        const len_t b
        )
{
    sg->tri[a][b / 64] |= (uint64_t)1 << (b % 64);
}

/* whether a base divisor eliminates the S-pair of n and i, see
 * sg_new_pairs() */
static inline int sg_base_divisor_eliminates(
        const sg_t * const sg,
        const len_t n,
        const len_t i,
        const len_t high,
        const len_t bd,
        const exp_t * const v
        )
{
    len_t j;
    const ht_t * const ht = sg->bht;
    const len_t evl = ht->evl;
    if (high != n && i != high && sg->e[i].tau > sg->e[high].tau
            && sg->e[i].tau > sg->e[n].tau && sg_tri_get(sg, high, i)) {
        return 1;
    }
    if (bd != n && i != bd && sg->e[i].tau < sg->e[bd].tau
            && sg->e[i].tau < sg->e[n].tau && sg_tri_get(sg, bd, i)) {
        const exp_t * const elg = ht->ev[sg->e[i].p[OFFSET]];
        for (j = 1; j < evl; ++j) {
            if (elg[j] > v[j]) {
                break;
            }
        }
        if (j == evl) {
            return 1;
        }
    }
    return 0;
}

/* enters the S-pair of n and i of signature (ix, sm) which is no known
 * syzygy signature: as a syzygy signature for coprime leading monomials,
 * otherwise as a pending signature; esm and sc are scratch vectors */
static inline void sg_enter_pair(
        sg_t *sg,
        const len_t n,
        const len_t i,
        const len_t ix,
        const uint64_t * const sm,
        const int coprime,
        exp_t *esm,
        exp_t *sc
        )
{
    len_t j;
    const ht_t * const ht = sg->bht;
    const len_t evl = ht->evl;
    if (coprime) {
        /* the signature of the Koszul syzygy of g and h */
        sg->cnt_koszul++;
        sg_add_syzygy(sg, ix, sm);
        sg_tri_set(sg, n, i);
        return;
    }
    sg->cnt_queued++;
    sg_unpack(sm, esm, evl);
    const exp_t * const lf = ht->ev[sg->g[ix].p[OFFSET]];
    if ((uint64_t)esm[0] + lf[0] > EXP_MAX) {
        exponent_overflow();
    }
    for (j = 0; j < evl; ++j) {
        sc[j] = esm[j] + lf[j];
    }
    sg_add_pending(sg, ix, esm, sc);
}

/* the least number of earlier elements for which the S-pairs of a new
 * element are formed by several threads */
#define SG_PAIRS_PAR 1024

/* sg_new_pairs() for the S-pairs of n with the elements before it by
 * several threads: they are classified in parallel against the syzygy
 * signatures known before, then they are entered one after the other,
 * tested only against the syzygy signatures entered meanwhile, as when
 * they are formed one after the other */
static void sg_new_pairs_par(
        sg_t *sg,
        const len_t n,
        const len_t high,
        const len_t bd,
        const exp_t * const v
        )
{
    len_t i, j;
    const len_t nw  = sg->nw;
    const len_t evl = sg->bht->evl;
    /* 0: eliminated by a base divisor, 1: singular, 2: syzygy signature,
     * 3: to be entered */
    int8_t *cls  = (int8_t *)malloc((unsigned long)n * sizeof(int8_t));
    int8_t *cop  = (int8_t *)malloc((unsigned long)n * sizeof(int8_t));
    len_t *pix   = (len_t *)malloc((unsigned long)n * sizeof(len_t));
    uint64_t *pw = (uint64_t *)malloc((unsigned long)n * nw * sizeof(uint64_t));
#pragma omp parallel for num_threads(sg->nthrds) schedule(dynamic, 256)
    for (i = 0; i < n; ++i) {
        uint64_t *sm = pw + (unsigned long)i * nw;
        len_t ix;
        int coprime;
        if (sg_base_divisor_eliminates(sg, n, i, high, bd, v)) {
            cls[i] = 0;
        } else if (!sg_pair_signature(sg, n, i, &ix, sm, &coprime)) {
            cls[i] = 1;
        } else if (sg_is_syzygy_ro(sg, ix, sm)) {
            cls[i] = 2;
        } else {
            cls[i] = 3;
            cop[i] = (int8_t)coprime;
            pix[i] = ix;
        }
    }
    /* the Koszul syzygy signatures entered here, by index */
    len_t nk = 0;
    len_t *kix   = (len_t *)malloc((unsigned long)n * sizeof(len_t));
    uint64_t *kw = (uint64_t *)malloc((unsigned long)n * nw * sizeof(uint64_t));
    exp_t esm[evl], sc[evl];
    sg->cnt_pairs += n;
    for (i = 0; i < n; ++i) {
        switch (cls[i]) {
            case 0:
                sg->cnt_base++;
                sg_tri_set(sg, n, i);
                continue;
            case 1:
                sg->cnt_singular++;
                continue;
            case 2:
                sg->cnt_syz++;
                sg_tri_set(sg, n, i);
                continue;
            default:
                break;
        }
        const uint64_t * const sm = pw + (unsigned long)i * nw;
        const uint64_t nm = ~sg_mask(sm, sg->bht->nv);
        for (j = 0; j < nk; ++j) {
            if (kix[j] == pix[i]
                    && !(sg_mask(kw + (unsigned long)j * nw, sg->bht->nv) & nm)
                    && sg_divides_packed(kw + (unsigned long)j * nw, sm, nw)) {
                break;
            }
        }
        if (j < nk) {
            sg->cnt_syz++;
            sg_tri_set(sg, n, i);
            continue;
        }
        if (cop[i]) {
            kix[nk] = pix[i];
            memcpy(kw + (unsigned long)nk * nw, sm, (unsigned long)nw * sizeof(uint64_t));
            nk++;
        }
        sg_enter_pair(sg, n, i, pix[i], sm, cop[i], esm, sc);
    }
    free(kix);
    free(kw);
    free(cls);
    free(cop);
    free(pix);
    free(pw);
}

/* forms the S-pairs of the new element n with the elements before it.
 * The base divisor of n is the element of largest sig-lead ratio whose
 * signature divides the one of n. If the signature of the S-pair of the
 * base divisor and h is a syzygy signature dividing the signature of the
 * S-pair of n and h, the latter one is a syzygy signature, too. */
static void sg_new_pairs(
        sg_t *sg,
        const len_t n
        )
{
    len_t i, j, k;
    ht_t *ht        = sg->bht;
    const len_t evl = ht->evl;
    const len_t nw  = sg->nw;
    uint64_t sm[nw];
    exp_t esm[evl], sc[evl];
    len_t ix;
    int coprime;

    if (n >= sg->trisz) {
        const len_t osz = sg->trisz;
        sg->trisz = 2 * n + 16;
        sg->tri   = realloc(sg->tri, (unsigned long)sg->trisz * sizeof(uint64_t *));
        memset(sg->tri + osz, 0, (unsigned long)(sg->trisz - osz) * sizeof(uint64_t *));
    }
    sg->tri[n] = (uint64_t *)calloc((unsigned long)(n / 64 + 1), sizeof(uint64_t));

    /* base divisor by signature: of largest tau among the earlier
     * elements of the same index whose signature divides the one of n
     * (roune-2012 §3.5) */
    len_t bd = n;
    {
        const sg_el_t * const g = sg->e + n;
        const exp_t * const smg = ht->ev[g->sm];
        const sdm_t ns = ~ht->hd[g->sm].sdm;
        for (k = 0; k < sg->ixld[g->ix]; ++k) {
            const len_t c = sg->ixe[g->ix][k];
            if (c >= n) {
                continue;
            }
            const sg_el_t * const e = sg->e + c;
            if (ht->hd[e->sm].sdm & ns) {
                continue;
            }
            if (!sg_divides_exp(ht->ev[e->sm], smg, evl)) {
                continue;
            }
            if (bd == n || e->tau > sg->e[bd].tau) {
                bd = c;
            }
        }
    }
    /* base divisor by leading monomial: of largest tau among the earlier
     * elements, of any index, whose leading monomial divides the one of
     * n (roune-2012 §3.5, Theorem 7) */
    len_t high = n;
    {
        const exp_t * const eln = ht->ev[sg->e[n].p[OFFSET]];
        const sdm_t nl = ~ht->hd[sg->e[n].p[OFFSET]].sdm;
        for (k = 0; k < n; ++k) {
            const sg_el_t * const e = sg->e + k;
            if (ht->hd[e->p[OFFSET]].sdm & nl) {
                continue;
            }
            if (!sg_divides_exp(ht->ev[e->p[OFFSET]], eln, evl)) {
                continue;
            }
            if (high == n || e->tau > sg->e[high].tau) {
                high = k;
            }
        }
    }
    /* x^v of roune-2012 Theorem 8: lm(bd)*sig(n)/sig(bd), infinite where
     * the lead of n does not exceed it. Only the variables take part in
     * the divisibility test: x^v has infinite components, so its degree
     * lane is not the degree of a monomial dividing it. */
    exp_t v[evl];
    v[0] = EXP_MAX;
    if (bd != n) {
        const exp_t * const elb = ht->ev[sg->e[bd].p[OFFSET]];
        const exp_t * const esb = ht->ev[sg->e[bd].sm];
        const exp_t * const eln = ht->ev[sg->e[n].p[OFFSET]];
        const exp_t * const esn = ht->ev[sg->e[n].sm];
        for (j = 1; j < evl; ++j) {
            const exp_t pj = elb[j] + esn[j] - esb[j];
            v[j] = eln[j] <= pj ? EXP_MAX : pj;
        }
    }

    if (sg->nthrds > 1 && n >= SG_PAIRS_PAR) {
        sg_new_pairs_par(sg, n, high, bd, v);
        return;
    }
    for (i = 0; i < n; ++i) {
        sg->cnt_pairs++;
        /* the base divisors: the S-pair of the base divisor and i is
         * known to have a syzygy signature dividing the one of the S-pair
         * of n and i (roune-2012 §3.5, Theorems 7 and 8) */
        if (sg_base_divisor_eliminates(sg, n, i, high, bd, v)) {
            sg->cnt_base++;
            sg_tri_set(sg, n, i);
            continue;
        }
        if (!sg_pair_signature(sg, n, i, &ix, sm, &coprime)) {
            sg->cnt_singular++;
            continue;
        }
        if (sg_is_syzygy(sg, ix, sm)) {
            sg->cnt_syz++;
            sg_tri_set(sg, n, i);
            continue;
        }
        sg_enter_pair(sg, n, i, ix, sm, coprime, esm, sc);
    }
}

static int sg_pending_cmp(
        const void *a,
        const void *b,
        void *htp
        )
{
    const sg_sig_t *sa = (const sg_sig_t *)a;
    const sg_sig_t *sb = (const sg_sig_t *)b;
    return sg_cmp_sig(sa->ix, sa->sm, sb->ix, sb->sm, (const ht_t *)htp);
}

typedef struct sg_ctx_t sg_ctx_t;
struct sg_ctx_t
{
    sg_t *sg;
    const ht_t *ht;
};

/* rows by increasing signature, rows of equal signature by increasing
 * lead term, i.e. decreasing lead column */
static int sg_row_cmp(
        const void *a,
        const void *b,
        void *ctxp
        )
{
    const sg_row_t *ra = (const sg_row_t *)a;
    const sg_row_t *rb = (const sg_row_t *)b;
    const int c = sg_cmp_sig(ra->ix, ra->sm, rb->ix, rb->sm,
            ((sg_ctx_t *)ctxp)->ht);
    if (c != 0) {
        return c;
    }
    /* rows of equal signature keep their build order: a given row comes
     * before every reducer row of its signature (eder-2017 §6.2, Lemma
     * (singular criterion): exactly one row of a signature is handled) */
    if (ra->id != rb->id) {
        return ra->id < rb->id ? -1 : 1;
    }
    return 0;
}

/* reducer order: increasing sig-lead ratio, then fewer terms, then
 * older elements first */
static int sg_reducer_cmp(
        const void *a,
        const void *b,
        void *sgp
        )
{
    sg_t *sg = (sg_t *)sgp;
    const len_t ia = *((const len_t *)a);
    const len_t ib = *((const len_t *)b);
    const sg_el_t *ea = sg->e + ia;
    const sg_el_t *eb = sg->e + ib;
    if (ea->tau != eb->tau) {
        return ea->tau < eb->tau ? -1 : 1;
    }
    /* ties to fewer terms, then to the element made first (eder-2017
     * §7.3, Definition of ⊴_rat) */
    if (ea->p[LENGTH] != eb->p[LENGTH]) {
        return ea->p[LENGTH] < eb->p[LENGTH] ? -1 : 1;
    }
    return ia < ib ? -1 : (ia > ib ? 1 : 0);
}

static int sg_col_cmp(
        const void *a,
        const void *b,
        void *htp
        )
{
    const hi_t ma = *((const hi_t *)a);
    const hi_t mb = *((const hi_t *)b);
    return monomial_cmp_drl(mb, ma, (const ht_t *)htp);
}

/* the index of the lead terms of the elements in the order ro for
 * search_reducer(), see initialize_reducer_index() */
static void sg_reducer_index(
        red_idx_t *ri,
        const sg_t * const sg,
        const len_t * const ro
        )
{
    len_t i;
    const ht_t * const bht = sg->bht;
    const len_t n   = sg->ld;
    const len_t evl = bht->evl;
    ri->n   = n;
    ri->evl = evl;
    ri->ebl = bht->ebl;
    ri->sdm = (sdm_t *)malloc((unsigned long)(n + 1) * sizeof(sdm_t));
    ri->vm  = (uint64_t *)malloc((unsigned long)(n + 1) * sizeof(uint64_t));
    ri->pos = (len_t *)malloc((unsigned long)(n + 1) * sizeof(len_t));
    ri->ev  = (exp_t *)malloc((unsigned long)(n + 1) * evl * sizeof(exp_t));
    for (i = 0; i < n; ++i) {
        const hm_t lm = sg->e[ro[i]].p[OFFSET];
        const exp_t * const e = bht->ev[lm];
        ri->pos[i] = i;
        ri->sdm[i] = bht->hd[lm].sdm;
        ri->vm[i]  = reducer_variable_mask(e, evl, ri->ebl);
        memcpy(ri->ev + (unsigned long)i * evl, e,
                (unsigned long)evl * sizeof(exp_t));
    }
}

/* sorts the columns hcm[0], ..., hcm[n-1] by decreasing monomials, many
 * of them by keys, see sort_columns_by_keys() */
static void sg_sort_columns(
        hi_t *hcm,
        const len_t n,
        ht_t *sht
        )
{
    if (sht->ebl == 0 && sht->mo == 0 && !sht->boolean && n >= 4096) {
        sort_columns_by_keys(hcm, n, sht, 0, sg_col_cmp);
    } else {
        sort_r(hcm, (unsigned long)n, sizeof(hi_t), sg_col_cmp, sht);
    }
}

/* sorts the rows by sg_row_cmp(): many rows by the keys (rank of the
 * signature monomial, index) with a stable radix sort, which keeps the
 * build order of rows of equal signature */
static void sg_sort_rows(
        sg_row_t *rows,
        const len_t nr,
        sg_t *sg
        )
{
    len_t i;
    ht_t *rst = sg->rst;
    sg_ctx_t ctx = {sg, rst};
    const len_t nm = (len_t)(rst->eld - 1);
    uint32_t mix = 0;
    for (i = 0; i < nr; ++i) {
        mix = rows[i].ix > mix ? rows[i].ix : mix;
    }
    if (nr < 4096 || mix >= ((uint32_t)1 << 24)) {
        sort_r(rows, (unsigned long)nr, sizeof(sg_row_t), sg_row_cmp, &ctx);
        return;
    }
    /* the ranks of the signature monomials in increasing order */
    hi_t *hm = (hi_t *)malloc((unsigned long)nm * sizeof(hi_t));
    for (i = 0; i < nm; ++i) {
        hm[i] = i + 1;
    }
    sg_sort_columns(hm, nm, rst);
    uint32_t *rk = (uint32_t *)malloc((unsigned long)(nm + 1) * sizeof(uint32_t));
    for (i = 0; i < nm; ++i) {
        rk[hm[i]] = nm - 1 - i;
    }
    free(hm);
    uint64_t *k = (uint64_t *)malloc((unsigned long)nr * sizeof(uint64_t));
    uint64_t *t = (uint64_t *)malloc((unsigned long)nr * sizeof(uint64_t));
    len_t *ps   = (len_t *)malloc((unsigned long)nr * sizeof(len_t));
    len_t *pt   = (len_t *)malloc((unsigned long)nr * sizeof(len_t));
    for (i = 0; i < nr; ++i) {
        k[i]  = ((uint64_t)rk[rows[i].sm] << 24) | rows[i].ix;
        ps[i] = i;
    }
    free(rk);
    /* LSD radix sort on 16 bits at a time, stable */
    len_t *cnt = (len_t *)malloc(65537 * sizeof(len_t));
    unsigned sh;
    const unsigned bits = 24 + 64 - (unsigned)__builtin_clzll((uint64_t)nm | 1);
    for (sh = 0; sh < bits; sh += 16) {
        memset(cnt, 0, 65537 * sizeof(len_t));
        for (i = 0; i < nr; ++i) {
            cnt[((k[i] >> sh) & 0xFFFF) + 1]++;
        }
        for (i = 0; i < 65536; ++i) {
            cnt[i+1] += cnt[i];
        }
        for (i = 0; i < nr; ++i) {
            const len_t d = cnt[(k[i] >> sh) & 0xFFFF]++;
            t[d]  = k[i];
            pt[d] = ps[i];
        }
        uint64_t *x = k;
        k = t;
        t = x;
        len_t *y = ps;
        ps = pt;
        pt = y;
    }
    free(cnt);
    free(k);
    free(t);
    sg_row_t *sr = (sg_row_t *)malloc((unsigned long)nr * sizeof(sg_row_t));
    for (i = 0; i < nr; ++i) {
        sr[i] = rows[ps[i]];
    }
    memcpy(rows, sr, (unsigned long)nr * sizeof(sg_row_t));
    free(sr);
    free(ps);
    free(pt);
}

/* returns the row of the multiple of element el (or of the input
 * generator el if gen != 0) by the monomial with exponents m */
static sg_row_t sg_multiple(
        sg_t *sg,
        const sg_el_t * const el,
        const exp_t * const m,
        const int8_t reducer,
        const len_t src
        )
{
    len_t j;
    sg_row_t row;
    ht_t *bht       = sg->bht;
    ht_t *sht       = sg->sht;
    const len_t evl = bht->evl;
    exp_t em[evl];
    val_t h = 0;
    memcpy(em, m, (unsigned long)evl * sizeof(exp_t));
    for (j = 0; j < evl; ++j) {
        h += bht->rn[j] * em[j];
    }
    row.r       = multiplied_poly_to_matrix_row(sht, bht, h, em, el->p);
    row.cf      = el->cf;
    row.ix      = el->ix;
    row.reducer = reducer;
    row.gen     = 0;
    row.src     = src;
    row.mul     = sg->rec != NULL ? sg_insert(em, bht) : 0;
    row.pr      = NULL;
    row.pv      = NULL;
    exp_t e[evl];
    if ((uint64_t)em[0] + bht->ev[el->sc][0] > EXP_MAX) {
        exponent_overflow();
    }
    const exp_t * const sm = bht->ev[el->sm];
    for (j = 0; j < evl; ++j) {
        e[j] = em[j] + sm[j];
    }
    row.sm = sg_insert(e, sg->rst);
    return row;
}

/* subtracts mul times the pivot row (pr, pf) of length pl but its lead
 * entry from the dense row dr. For p < 2^16 each product is below 2^32, so
 * the entries stay below 2^63 for less than 2^31 subtractions, otherwise
 * they are kept below 2^63 by subtracting big, a multiple of p. */
static inline void sg_sub_row(
        uint64_t *dr,
        const hm_t * const pr,
        const cf32_t * const pf,
        const len_t pl,
        const uint64_t mul,
        const uint64_t big,
        const int small
        )
{
    len_t j;
    if (small) {
        for (j = 1; j < pl; ++j) {
            dr[pr[j]] += mul * pf[j];
        }
    } else {
        for (j = 1; j < pl; ++j) {
            const uint64_t t = dr[pr[j]] + mul * pf[j];
            dr[pr[j]] = t >= big ? t - big : t;
        }
    }
}

/* the least number s of random combinations with (p-1)^s >= 2^31: if s
 * random combinations of some rows reduce to zero, all rows do so with
 * probability at least 1 - 2^-31 */
static len_t sg_zero_combinations(
        const uint64_t p
        )
{
    len_t s = 1;
    long double q = (long double)(p - 1);
    while (q < 2147483648.0L) {
        q *= (long double)(p - 1);
        s++;
    }
    return s;
}

/* do s random combinations of the rows g[0], ..., g[ng-1] reduce to zero by
 * the pivots? A combination stops at its first entry in a column without
 * a pivot, which shows it nonzero. The dense row is zero afterwards. */
static int sg_zero_test(
        const len_t * const g,
        const len_t ng,
        const sg_row_t * const rows,
        hm_t **pc,
        cf32_t **pcf,
        uint64_t *dr,
        const len_t nc,
        const uint64_t p,
        const uint64_t big,
        const int small,
        const len_t s,
        uint64_t *seed,
        uint64_t *nsubs
        )
{
    len_t i, j, k, t;
    for (t = 0; t < s; ++t) {
        len_t from = nc;
        for (i = 0; i < ng; ++i) {
            const sg_row_t * const row = rows + g[i];
            const hm_t * const r = row->r + OFFSET;
            const len_t len = row->r[LENGTH];
            /* xorshift */
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            const uint64_t c = 1 + *seed % (p - 1);
            for (j = 0; j < len; ++j) {
                const uint64_t x = dr[r[j]] + c * row->cf[j];
                dr[r[j]] = x >= big ? x - big : x;
            }
            if (r[0] < from) {
                from = r[0];
            }
        }
        for (k = from; k < nc; ++k) {
            if (dr[k] == 0) {
                continue;
            }
            const uint64_t v = dr[k] % p;
            dr[k] = 0;
            if (v == 0) {
                continue;
            }
            if (pc[k] == NULL) {
                memset(dr + k + 1, 0, (unsigned long)(nc - k - 1) * sizeof(uint64_t));
                return 0;
            }
            sg_sub_row(dr, pc[k] + OFFSET, pcf[k], pc[k][LENGTH], p - v, big, small);
            *nsubs += pc[k][LENGTH];
        }
    }
    return 1;
}

#define SG_BAND 8
#define SG_NONE ((len_t)-1)

/* subtracts the multiples m[b] of the pivot row (pr, pf) of length pl from
 * the interleaved dense rows of the band in bd, nz of the multiples are
 * nonzero, one of them m[one]; returns the number of multiply-adds */
static inline uint64_t sg_band_sub(
        uint64_t *bd,
        const hm_t * const pr,
        const cf32_t * const pf,
        const len_t pl,
        const uint32_t * const m,
        const len_t nz,
        const len_t one,
        const uint64_t big,
        const int small
        )
{
    len_t j;
    /* boyer-2016 §4.2: where only one row of the band holds an
     * entry, a specialized operation reduces that row alone, and
     * the row operations run only on the nonzero multipliers */
    if (nz == 1) {
        if (small) {
            for (j = 1; j < pl; ++j) {
                bd[(unsigned long)pr[j] * SG_BAND + one] +=
                    (uint64_t)m[one] * pf[j];
            }
        } else {
            for (j = 1; j < pl; ++j) {
                uint64_t * const y = bd + (unsigned long)pr[j] * SG_BAND + one;
                const uint64_t t = *y + (uint64_t)m[one] * pf[j];
                *y = t - (t >= big ? big : 0);
            }
        }
        return pl;
    } else {
#if defined(__AVX512F__) && SG_BAND == 8
        /* one 64-bit lane for each row of the band, the products
         * of the low 32 bits of the lanes */
        const __m512i mv = _mm512_cvtepu32_epi64(
                _mm256_loadu_si256((const __m256i *)m));
        if (small) {
            for (j = 1; j < pl; ++j) {
                uint64_t * const y = bd + (unsigned long)pr[j] * SG_BAND;
                const __m512i f = _mm512_set1_epi64((long long)pf[j]);
                _mm512_storeu_si512((void *)y, _mm512_add_epi64(
                            _mm512_loadu_si512((const void *)y),
                            _mm512_mul_epu32(mv, f)));
            }
        } else {
            const __m512i bv = _mm512_set1_epi64((long long)big);
            for (j = 1; j < pl; ++j) {
                uint64_t * const y = bd + (unsigned long)pr[j] * SG_BAND;
                const __m512i f = _mm512_set1_epi64((long long)pf[j]);
                const __m512i t = _mm512_add_epi64(
                        _mm512_loadu_si512((const void *)y),
                        _mm512_mul_epu32(mv, f));
                const __mmask8 o = _mm512_cmpge_epu64_mask(t, bv);
                _mm512_storeu_si512((void *)y,
                        _mm512_mask_sub_epi64(t, o, t, bv));
            }
        }
#else
        len_t b;
        if (small) {
            for (j = 1; j < pl; ++j) {
                uint64_t * const y = bd + (unsigned long)pr[j] * SG_BAND;
                const uint32_t f   = pf[j];
#pragma omp simd
                for (b = 0; b < SG_BAND; ++b) {
                    y[b] += (uint64_t)m[b] * f;
                }
            }
        } else {
            for (j = 1; j < pl; ++j) {
                uint64_t * const y = bd + (unsigned long)pr[j] * SG_BAND;
                const uint32_t f   = pf[j];
#pragma omp simd
                for (b = 0; b < SG_BAND; ++b) {
                    const uint64_t t = y[b] + (uint64_t)m[b] * f;
                    y[b] = t - (t >= big ? big : 0);
                }
            }
        }
#endif
        return (uint64_t)pl * nz;
    }
}

/* Reduces the nb <= SG_BAND rows rows[g[0]], ..., rows[g[nb-1]] together, as
 * one after the other would be reduced: each by the pivots and by the rows
 * of the band before it. The dense rows are interleaved, bd holds the
 * entries of a column of all of them side by side, so a pivot row is read
 * once for the band. A pivot is subtracted from each dense row with its
 * entry in the pivot's column; where a row of the band takes a column
 * without pivot as its lead term, the later rows record their multiples of
 * it, which are applied whenever their entries are read at later columns
 * without pivot. The remainder of row b is written to (nc_[b], nv[b]). */
static void sg_band_reduce(
        const len_t * const g,
        const len_t nb,
        const sg_row_t * const rows,
        hm_t **pc,
        cf32_t **pcf,
        uint64_t *bd,
        const len_t nc,
        const uint64_t p,
        const uint64_t big,
        const int small,
        uint8_t *used,
        hm_t **ncol,
        cf32_t **nval,
        len_t *nnl,
        uint64_t *nsubs
        )
{
    len_t b, a, j, k;
    hm_t from = nc;
    len_t lead[SG_BAND];
    uint64_t inv[SG_BAND], rel[SG_BAND][SG_BAND], e[SG_BAND];
    memset(rel, 0, sizeof(rel));
    for (b = 0; b < nb; ++b) {
        const sg_row_t * const row = rows + g[b];
        /* the row, or its remainder modulo the reducers before it */
        const hm_t * const rh   = row->pr != NULL ? row->pr : row->r;
        const cf32_t * const rc = row->pr != NULL ? row->pv : row->cf;
        const hm_t * const r = rh + OFFSET;
        const len_t len = rh[LENGTH];
        for (j = 0; j < len; ++j) {
            bd[(unsigned long)r[j] * SG_BAND + b] = rc[j];
        }
        if (r[0] < from) {
            from = r[0];
        }
        lead[b] = nc;
        nnl[b]  = 0;
    }
    for (k = from; k < nc; ++k) {
        uint64_t * const x = bd + (unsigned long)k * SG_BAND;
        uint64_t any = 0;
        for (b = 0; b < SG_BAND; ++b) {
            any |= x[b];
        }
        if (any == 0) {
            continue;
        }
        if (pc[k] != NULL) {
            /* the multipliers and coefficients are below 2^31, their
             * products are 32 x 32 -> 64 bit products */
            uint32_t m[SG_BAND];
            len_t nz = 0, one = 0;
            for (b = 0; b < SG_BAND; ++b) {
                const uint64_t v = x[b] % p;
                m[b] = (uint32_t)(v == 0 ? 0 : p - v);
                x[b] = 0;
                if (v != 0) {
                    ++nz;
                    one = b;
                }
            }
            const hm_t * const pr   = pc[k] + OFFSET;
            const cf32_t * const pf = pcf[k];
            const len_t pl          = pc[k][LENGTH];
            *nsubs += sg_band_sub(bd, pr, pf, pl, m, nz, one, big, small);
            if (used != NULL) {
                used[k] = 1;
            }
            continue;
        }
        /* a column without pivot: the entries of the rows in turn */
        len_t owner = nb;
        for (b = 0; b < nb; ++b) {
            uint64_t v = x[b] % p;
            x[b] = 0;
            for (a = 0; a < b; ++a) {
                if (rel[b][a] != 0 && e[a] != 0) {
                    v = (v + p - (rel[b][a] * e[a]) % p) % p;
                }
            }
            if (v != 0 && owner < b) {
                /* the row is reduced by the band row taking this column */
                rel[b][owner] = (v * inv[owner]) % p;
                v = 0;
            } else if (v != 0 && lead[b] == nc) {
                lead[b]  = k;
                owner    = b;
                inv[b]   = mod_p_inverse_32((int64_t)v, (int64_t)p);
            }
            e[b] = v;
            if (v != 0) {
                ncol[b][nnl[b]]   = k;
                nval[b][nnl[b]++] = (cf32_t)v;
            }
        }
        for (; b < SG_BAND; ++b) {
            x[b] = 0;
        }
    }
}

/* the least number of rows to be reduced in a round for which they are
 * reduced by the reducer rows in parallel first, see
 * sg_reduce_by_reducers() */
#define SG_PAR_ROWS 16

/* The rows to be reduced of a round are reduced in parallel by the reducer
 * rows before them in the order of the round, each reducer row taken as
 * the pivot of its lead column if it is the first reducer row with this
 * lead column. In the reduction one row after the other, see sg_round(),
 * each such reducer row is a pivot or reduces by the pivots before it, so
 * it lies in the span of the pivots before the row to be reduced. Thus
 * the remainder of this reduction, which is (pr, pv) of the row, reduced
 * by these pivots there, gives the same result as the row itself: the
 * remainder modulo pivots with distinct lead columns is unique. The rows
 * whose reduction would exceed the multiply-adds wmax are left as they
 * are. Returns 1 if the rows are reduced so: the reductions after it do
 * not see all pivots the reduction one after the other uses. */
static int sg_reduce_by_reducers(
        sg_row_t *rows,
        const len_t nr,
        const len_t nc,
        const uint64_t p,
        const uint64_t big,
        const int small,
        uint8_t *used,
        const uint64_t wmax,
        uint64_t *nsubs,
        const int nthrds
        )
{
    len_t i, nt = 0;
    for (i = 0; i < nr; ++i) {
        nt += rows[i].reducer == 0;
    }
    if (nt < SG_PAR_ROWS) {
        return 0;
    }
    /* the first reducer row of each lead column, with its position */
    hm_t **pr     = (hm_t **)calloc((unsigned long)nc + 1, sizeof(hm_t *));
    cf32_t **pv   = (cf32_t **)calloc((unsigned long)nc + 1, sizeof(cf32_t *));
    len_t *ppos   = (len_t *)malloc((unsigned long)(nc + 1) * sizeof(len_t));
    len_t *tbr    = (len_t *)malloc((unsigned long)nt * sizeof(len_t));
    nt = 0;
    for (i = 0; i < nr; ++i) {
        const hm_t lc = rows[i].r[OFFSET];
        if (rows[i].reducer) {
            if (pr[lc] == NULL) {
                pr[lc]   = rows[i].r;
                pv[lc]   = (cf32_t *)rows[i].cf;
                ppos[lc] = i;
            }
        } else {
            tbr[nt++] = i;
        }
    }
    volatile uint64_t done = *nsubs;
    /* the rows are reduced in bands of SG_BAND consecutive rows as in
     * sg_band_reduce(), so that a pivot row is read once for the band;
     * a pivot after a row of the band is not applied to it */
    const len_t ng = (nt + SG_BAND - 1) / SG_BAND;
#pragma omp parallel num_threads(nthrds)
    {
        const size_t bsz = (size_t)(nc + 1) * SG_BAND * sizeof(uint64_t);
        uint64_t *bd = NULL;
        if (posix_memalign((void **)&bd, 64, bsz) == 0) {
            memset(bd, 0, bsz);
        } else {
            bd = (uint64_t *)calloc(bsz, 1);
        }
        hm_t *ncol[SG_BAND];
        cf32_t *nval[SG_BAND];
        len_t b;
        for (b = 0; b < SG_BAND; ++b) {
            ncol[b] = (hm_t *)malloc((unsigned long)(nc + 1) * sizeof(hm_t));
            nval[b] = (cf32_t *)malloc((unsigned long)(nc + 1) * sizeof(cf32_t));
        }
        len_t t, j, k;
#pragma omp for schedule(dynamic, 1)
        for (t = 0; t < ng; ++t) {
            if (done > wmax) {
                continue;
            }
            const len_t g0 = t * SG_BAND;
            const len_t nb = nt - g0 < SG_BAND ? nt - g0 : SG_BAND;
            len_t ri[SG_BAND], nl[SG_BAND];
            hm_t from = nc;
            for (b = 0; b < nb; ++b) {
                ri[b] = tbr[g0+b];
                const sg_row_t * const row = rows + ri[b];
                const hm_t * const r = row->r + OFFSET;
                const len_t len = row->r[LENGTH];
                for (j = 0; j < len; ++j) {
                    bd[(unsigned long)r[j] * SG_BAND + b] = row->cf[j];
                }
                if (len > 0 && r[0] < from) {
                    from = r[0];
                }
                nl[b] = 0;
            }
            uint64_t subs = 0;
            for (k = from; k < nc; ++k) {
                uint64_t * const x = bd + (unsigned long)k * SG_BAND;
                uint64_t any = 0;
                for (b = 0; b < SG_BAND; ++b) {
                    any |= x[b];
                }
                if (any == 0) {
                    continue;
                }
                uint32_t m[SG_BAND];
                len_t nz = 0, one = 0;
                for (b = 0; b < SG_BAND; ++b) {
                    const uint64_t v = x[b] % p;
                    x[b] = 0;
                    m[b] = 0;
                    if (v == 0) {
                        continue;
                    }
                    if (pr[k] == NULL || ppos[k] > ri[b]) {
                        ncol[b][nl[b]]   = k;
                        nval[b][nl[b]++] = (cf32_t)v;
                        continue;
                    }
                    m[b] = (uint32_t)(p - v);
                    ++nz;
                    one = b;
                }
                if (nz == 0) {
                    continue;
                }
                subs += sg_band_sub(bd, pr[k] + OFFSET, pv[k], pr[k][LENGTH],
                        m, nz, one, big, small);
                if (used != NULL) {
                    used[k] = 1;
                }
            }
            for (b = 0; b < nb; ++b) {
                sg_row_t *row = rows + ri[b];
                hm_t *h    = (hm_t *)malloc((unsigned long)(nl[b] + OFFSET) * sizeof(hm_t));
                cf32_t *c  = (cf32_t *)malloc((unsigned long)(nl[b] + 1) * sizeof(cf32_t));
                memcpy(h + OFFSET, ncol[b], (unsigned long)nl[b] * sizeof(hm_t));
                memcpy(c, nval[b], (unsigned long)nl[b] * sizeof(cf32_t));
                h[LENGTH]  = nl[b];
                h[PRELOOP] = nl[b] % UNROLL;
                row->pr = h;
                row->pv = c;
            }
#pragma omp atomic
            done += subs;
        }
        free(bd);
        for (b = 0; b < SG_BAND; ++b) {
            free(ncol[b]);
            free(nval[b]);
        }
    }
    *nsubs = done;
    free(pr);
    free(pv);
    free(ppos);
    free(tbr);
    return 1;
}

/* one round: returns the number of new elements, -1 if the basis holds a
 * constant */
static int64_t sg_round(
        sg_t *sg,
        md_t *md
        )
{
    len_t i, j, k;
    ht_t *bht       = sg->bht;
    ht_t *sht       = sg->sht;
    ht_t *rst       = sg->rst;
    const len_t evl = bht->evl;
    const uint64_t p = sg->fc;

    const double t0 = realtime();
    clean_hash_table(rst);
    const uint64_t nly0 = sg->nearly;
    /* select the pending signatures of least degree */
    deg_t d = sg->ps[0].deg;
    for (i = 1; i < sg->pld; ++i) {
        if (sg->ps[i].deg < d) {
            d = sg->ps[i].deg;
        }
    }
    len_t ns = 0;
    sg_sig_t *sel = (sg_sig_t *)malloc((unsigned long)sg->pld * sizeof(sg_sig_t));
    for (i = 0, j = 0; i < sg->pld; ++i) {
        if (sg->ps[i].deg == d) {
            sel[ns++] = sg->ps[i];
        } else {
            sg->ps[j++] = sg->ps[i];
        }
    }
    sg->pld = j;
    sort_r(sel, (unsigned long)ns, sizeof(sg_sig_t), sg_pending_cmp, bht);

    /* reducer order of the elements */
    len_t *ro = (len_t *)malloc((unsigned long)(sg->ld+1) * sizeof(len_t));
    for (i = 0; i < sg->ld; ++i) {
        ro[i] = i;
    }
    sort_r(ro, (unsigned long)sg->ld, sizeof(len_t), sg_reducer_cmp, sg);
    /* the lead terms in this order, the first divisor of a monomial found
     * is the one of least sig-lead ratio */
    red_idx_t ri;
    sg_reducer_index(&ri, sg, ro);

    /* rows of the selected signatures, by rewriting */
    len_t nr = 0, rsz = 2 * ns + 16;
    len_t t_dedup = 0, t_syz = 0; /* dropped in this round, for the report */
    sg_row_t *rows = (sg_row_t *)malloc((unsigned long)rsz * sizeof(sg_row_t));
    exp_t m[evl];
    len_t mix   = 0;  /* the largest signature of a row to be reduced */
    hm_t msc    = 0;
    for (i = 0; i < ns; ++i) {
        if (i > 0 && sel[i].ix == sel[i-1].ix && sel[i].sm == sel[i-1].sm) {
            t_dedup++;
            continue;
        }
        const len_t ix = sel[i].ix;
        memcpy(m, bht->ev[sel[i].sm], (unsigned long)evl * sizeof(exp_t));
        if (sg_is_syzygy_exp(sg, ix, m)) {
            t_syz++;
            continue;
        }
        /* rewriter: largest sig-lead ratio, i.e. least lead term of the
         * multiple, among the elements whose signature divides */
        const sg_el_t *rw = NULL;
        exp_t best[evl];
        const sdm_t nsm = ~bht->hd[sel[i].sm].sdm;
        for (k = 0; k < sg->ixld[ix]; ++k) {
            const sg_el_t *e = sg->e + sg->ixe[ix][k];
            if (bht->hd[e->sm].sdm & nsm) {
                continue;
            }
            const exp_t * const es = bht->ev[e->sm];
            if (!sg_divides_exp(es, m, evl)) {
                continue;
            }
            const exp_t * const el = bht->ev[e->p[OFFSET]];
            exp_t lt[evl];
            for (j = 0; j < evl; ++j) {
                lt[j] = m[j] - es[j] + el[j];
            }
            const int cl = rw == NULL ? -1 : sg_cmp_exp(lt, best, evl);
            if (rw == NULL || cl < 0
                    || (cl == 0 && sg_cmp_exp(es, bht->ev[rw->sm], evl) > 0)) {
                rw = e;
                memcpy(best, lt, (unsigned long)evl * sizeof(exp_t));
            }
        }
        /* lairez-2022 §5.7, Algorithm 8, line 94: the reductant of the
         * signature joins the given rows only if its lead term
         * L = a*lm(rw) is reducible by an element of smaller sig-lead
         * ratio than rw -- only its multiple reaching L has a smaller
         * signature. The first divisor in the reducer order has the
         * least sig-lead ratio among the elements. The multiple joins
         * the given rows only if the least sig-lead ratio of a divisor
         * of L is strictly smaller than the rewriter's. Otherwise a
         * multiple of smaller signature top-reduces it, the signature
         * is consumed without a row. */
        {
            const len_t f = search_reducer(&ri,
                    generate_short_divmask(best, bht), best);
            /* d exists always: the lead of the rewriter divides L */
            const sg_el_t *d = sg->e + ro[f-1];
            if (sg_cmp_ratio(d, rw, sg) >= 0) {
                sg->nearly++;
                continue;
            }
        }
        exp_t q[evl];
        const exp_t * const es = bht->ev[rw->sm];
        for (j = 0; j < evl; ++j) {
            q[j] = m[j] - es[j];
        }
        rows[nr] = sg_multiple(sg, rw, q, 0, (len_t)(rw - sg->e));
        rows[nr].gen = 1;
        rows[nr].id  = nr;
        /* the largest signature of the rows to be reduced, in the
         * Schreyer order of the signatures */
        if (nr == 0 || sg_cmp_sig(rows[nr].ix, rows[nr].sm, mix, msc, rst) > 0) {
            mix = rows[nr].ix;
            msc = rows[nr].sm;
        }
        nr++;
    }
    free(sel);
    const len_t nsr = nr; /* S-rows built from the selected signatures */
    if (md->info_level > 1) {
        fprintf(VERBSTREAM, "pop: selected=%u dedup=%u syzygy=%u els=%u srows=%u\n",
                (unsigned)ns, (unsigned)t_dedup, (unsigned)t_syz,
                (unsigned)sg->nearly, (unsigned)nr);
    }
    const len_t smon = (len_t)(sht->eld - 1); /* monomials of the S-rows */
    if (nr == 0) {
        free(rows);
        free(ro);
        free_reducer_index(&ri);
        return 0;
    }

    /* symbolic preprocessing: each monomial is visited once, the
     * reducers add new monomials at the end of the symbolic hash table.
     * The monomials are handled in waves: the reducers of the monomials
     * not handled yet are searched in parallel, then their rows are
     * generated in parallel, adding the monomials of the next wave. */
    for (i = 0; i < nr; ++i) {
        sht->hd[rows[i].r[OFFSET]].idx = 1;
    }
    hl_t lo = 1;
    len_t wsz = 0;
    hi_t *wm   = NULL;  /* the monomials of the wave */
    len_t *wr  = NULL;  /* their reducers in ro, plus one, 0 if none */
    while (lo < sht->eld) {
        const hl_t hi = sht->eld;
        if (wsz < hi - lo) {
            wsz = hi - lo;
            wm  = realloc(wm, (unsigned long)wsz * sizeof(hi_t));
            wr  = realloc(wr, (unsigned long)wsz * sizeof(len_t));
        }
        len_t nw = 0;
        hl_t mi;
        for (mi = lo; mi < hi; ++mi) {
            if (sht->hd[mi].idx != 2) {
                sht->hd[mi].idx = 2;
                wm[nw++] = (hi_t)mi;
            }
        }
        lo = hi;
#pragma omp parallel for num_threads(md->nthrds) \
        private(j) schedule(dynamic, 64) if (md->nthrds > 1 && nw > 256)
        for (j = 0; j < nw; ++j) {
            const hi_t m = wm[j];
            const len_t f = search_reducer(&ri, sht->hd[m].sdm, sht->ev[m]);
            wr[j] = 0;
            if (f > 0) {
                /* the reducer is only of use if its signature is less
                 * than the largest signature of a row to be reduced
                 * (eder-2017 §7.2, Algorithm RB: the reduction stays
                 * below the largest signature of the round) */
                const sg_el_t *e = sg->e + ro[f-1];
                const exp_t * const el  = bht->ev[e->p[OFFSET]];
                const exp_t * const em  = sht->ev[m];
                const exp_t * const esm = bht->ev[e->sm];
                exp_t s[evl];
                len_t l;
                for (l = 0; l < evl; ++l) {
                    s[l] = em[l] - el[l] + esm[l];
                }
                int c = sg_cmp_exp(s, rst->ev[msc], evl);
                if (c == 0) {
                    c = e->ix == mix ? 0 : (e->ix > mix ? 1 : -1);
                }
                wr[j] = c < 0 ? f : 0;
            }
        }
        len_t nn = 0;
        for (j = 0; j < nw; ++j) {
            nn += wr[j] != 0;
        }
        if (nn == 0) {
            continue;
        }
        while (nr + nn > rsz) {
            rsz  = 2 * rsz;
            rows = realloc(rows, (unsigned long)rsz * sizeof(sg_row_t));
        }
        hm_t **wrows  = (hm_t **)malloc((unsigned long)nn * sizeof(hm_t *));
        hm_t **polys  = (hm_t **)malloc((unsigned long)nn * sizeof(hm_t *));
        val_t *hs     = (val_t *)malloc((unsigned long)nn * sizeof(val_t));
        exp_t *ems    = (exp_t *)malloc((unsigned long)nn * evl * sizeof(exp_t));
        len_t *ws     = (len_t *)malloc((unsigned long)nn * sizeof(len_t));
        len_t n = 0;
        for (j = 0; j < nw; ++j) {
            if (wr[j] == 0) {
                continue;
            }
            const sg_el_t *e = sg->e + ro[wr[j]-1];
            const exp_t * const el = bht->ev[e->p[OFFSET]];
            const exp_t * const em = sht->ev[wm[j]];
            exp_t *q = ems + (unsigned long)n * evl;
            val_t h = 0;
            for (k = 0; k < evl; ++k) {
                q[k] = em[k] - el[k];
                h   += bht->rn[k] * q[k];
            }
            if ((uint64_t)q[0] + bht->ev[e->sc][0] > EXP_MAX) {
                exponent_overflow();
            }
            hs[n]    = h;
            polys[n] = e->p;
            ws[n]    = wr[j] - 1;
            n++;
        }
        multiplied_polys_to_matrix_rows(wrows, n, hs, ems, polys, sht, bht,
                md->nthrds);
        for (j = 0; j < n; ++j) {
            const sg_el_t *e = sg->e + ro[ws[j]];
            const exp_t * const q   = ems + (unsigned long)j * evl;
            const exp_t * const esm = bht->ev[e->sm];
            exp_t s[evl];
            for (k = 0; k < evl; ++k) {
                s[k] = q[k] + esm[k];
            }
            sg_row_t *row = rows + nr;
            row->r       = wrows[j];
            row->cf      = e->cf;
            row->ix      = e->ix;
            row->reducer = 1;
            row->gen     = 0;
            row->src     = ro[ws[j]];
            row->mul     = sg->rec != NULL ? sg_insert(q, bht) : 0;
            row->pr      = NULL;
            row->pv      = NULL;
            row->sm      = sg_insert(s, rst);
            row->id      = nr;
            nr++;
        }
        free(wrows);
        free(polys);
        free(hs);
        free(ems);
        free(ws);
    }
    free(wm);
    free(wr);
    free(ro);
    free_reducer_index(&ri);

    /* columns by decreasing monomials */
    const len_t nc = (len_t)(sht->eld - 1);
    hi_t *hcm = (hi_t *)malloc((unsigned long)nc * sizeof(hi_t));
    for (i = 0; i < nc; ++i) {
        hcm[i] = i+1;
    }
    sg_sort_columns(hcm, nc, sht);
    for (i = 0; i < nc; ++i) {
        sht->hd[hcm[i]].idx = i;
    }
    for (i = 0; i < nr; ++i) {
        hm_t *r = rows[i].r;
        const len_t len = r[LENGTH]+OFFSET;
        for (j = OFFSET; j < len; ++j) {
            r[j] = sht->hd[r[j]].idx;
        }
    }
    sg_sort_rows(rows, nr, sg);

    /* without random combinations ruling out the rows reducing to zero, a
     * dense matrix with many rows to be reduced is reduced faster by F4,
     * whose reduction takes several rows at once with SIMD instructions */
    if (md->laopt < 40 && sg->rec == NULL && md->sig_f4 != 2) {
        uint64_t ne = 0;
        len_t nred = 0;
        for (i = 0; i < nr; ++i) {
            ne   += rows[i].r[LENGTH];
            nred += rows[i].reducer == 0;
        }
        if (nred >= 1000 && 10 * ne >= (uint64_t)nr * nc) {
            for (i = 0; i < nr; ++i) {
                free(rows[i].r);
            }
            free(rows);
            free(hcm);
            clean_hash_table(sht);
            return -2;
        }
    }

    if (md->info_level > 1) {
        uint64_t ne = 0;
        for (i = 0; i < nr; ++i) {
            ne += rows[i].r[LENGTH];
        }
        fprintf(VERBSTREAM, "%3u %7u sigs %9u x %-9u %6.2f%%", (unsigned)d,
                (unsigned)ns, (unsigned)nr, (unsigned)nc,
                100.0 * (double)ne / ((double)nr * (double)nc));
        fflush(VERBSTREAM);
    }

    /* reduction in signature order, each row by the pivots before it */
    hm_t **pc       = (hm_t **)calloc((unsigned long)nc, sizeof(hm_t *));
    cf32_t **pcf    = (cf32_t **)calloc((unsigned long)nc, sizeof(cf32_t *));
    uint64_t *dr    = (uint64_t *)calloc((unsigned long)nc, sizeof(uint64_t));
    const uint64_t big = ((uint64_t)1 << 63) / p * p;
    hm_t *nrow      = (hm_t *)malloc((unsigned long)(nc+OFFSET) * sizeof(hm_t));
    cf32_t *ncf     = (cf32_t *)malloc((unsigned long)nc * sizeof(cf32_t));
    /* A row is reduced by a dense scan over the columns from its lead
     * column on, or by a marked scan over the columns written, one bit per
     * column: the marked scan is taken where the columns the dense scans of
     * the recent rows passed are more than twice the entries written. */
    const len_t nmw = nc / 64 + 1;
    uint64_t *mk    = NULL;
    uint64_t passed = 0, written = 0;
    len_t counted   = 0;
    const int small = p < ((uint64_t)1 << 16);
    /* rows of 64 entries and more in the mean are reduced in bands */
    uint64_t ents = 0;
    for (i = 0; i < nr; ++i) {
        ents += rows[i].r[LENGTH];
    }
    const int band_rows = ents >= 64 * (uint64_t)nr;
    len_t *bres_n   = NULL;
    hm_t **bres_c   = NULL;
    cf32_t **bres_v = NULL;
    uint64_t *bd    = NULL;
    /* with probabilistic linear algebra, after two rows of a run reduced to
     * zero one after another, random combinations test whether the rest of
     * the run, up to the next reducer row setting a pivot, reduces to zero
     * as a whole; after a failed test the run is tested in blocks of 8
     * rows, doubling up to 64 */
    const int tests = md->laopt >= 40 && p > 3;
    const len_t zs  = tests ? sg_zero_combinations(p) : 0;
    len_t streak    = 0, blk = 0;
    len_t *grp      = tests ? (len_t *)malloc((unsigned long)nr * sizeof(len_t)) : NULL;

    /* new elements of the round */
    len_t nn = 0, nsz = 16;
    sg_row_t *nw = (sg_row_t *)malloc((unsigned long)nsz * sizeof(sg_row_t));

    /* A round whose reductions take many more multiply-adds than the
     * ones of the round before, see SG_WORK_GROWTH, is stopped and the
     * run hands over to F4: the elements of the new degree are dense,
     * which F4 reduces with its dense linear algebra. */
    const uint64_t wmax = sg->psubs == 0 ? UINT64_MAX
        : (sg->psubs * SG_WORK_GROWTH > SG_WORK_MIN
                ? sg->psubs * SG_WORK_GROWTH : SG_WORK_MIN);
    int stopped = 0;
    sg->nsubs = 0;
    sg->nzero = 0;
    sg_rrd_t *rd = NULL;
    if (sg->rec != NULL) {
        sg_rec_t *rec = sg->rec;
        if (rec->nrd >= rec->szrd) {
            rec->szrd = rec->szrd > 0 ? 2 * rec->szrd : 16;
            rec->rd   = realloc(rec->rd, (unsigned long)rec->szrd * sizeof(sg_rrd_t));
        }
        rd = rec->rd + rec->nrd++;
        memset(rd, 0, sizeof(sg_rrd_t));
    }
    /* the pivot columns the reductions use, and the lead columns of the
     * recorded rows setting pivots */
    uint8_t *used = rd != NULL ? (uint8_t *)calloc((unsigned long)nc + 1, sizeof(uint8_t)) : NULL;
    len_t *kcol   = rd != NULL ? (len_t *)malloc((unsigned long)(nr + 1) * sizeof(len_t)) : NULL;
    const int byred = md->nthrds > 1
        && sg_reduce_by_reducers(rows, nr, nc, p, big, small, used, wmax,
                &sg->nsubs, md->nthrds);
    for (i = 0; i < nr; ++i) {
        if (sg->nsubs > wmax) {
            stopped = 1;
            break;
        }
        sg_row_t *row = rows + i;
        hm_t *r = row->r;
        const hm_t lc = r[OFFSET];
        if (row->reducer && pc[lc] == NULL) {
            pc[lc]  = r;
            pcf[lc] = (cf32_t *)row->cf;
            if (rd != NULL) {
                /* the record keeps the pre-run element numbering: the
                 * input generators flagged, computed elements counted
                 * from zero */
                sg_rec_add(rd, row->src < sg->ng ?
                        SG_GEN | row->src : row->src - sg->ng,
                        row->mul, 0, r);
                kcol[rd->n-1] = lc;
            }
            continue;
        }
        if (row->reducer && i > 0 && row->ix == rows[i-1].ix
                && row->sm == rows[i-1].sm) {
            /* a row of the same signature comes before */
            free(r);
            row->r = NULL;
            continue;
        }
        /* the row, or its remainder modulo the reducers before it, see
         * sg_reduce_by_reducers() */
        const hm_t * const rr   = row->pr != NULL ? row->pr : r;
        const cf32_t * const rv = row->pr != NULL ? row->pv : row->cf;
        const len_t len = rr[LENGTH];
        const hm_t lr   = len > 0 ? rr[OFFSET] : nc;
        const uint64_t subs0 = sg->nsubs;
        const int marked = counted >= 4 && passed > 2 * written;
        len_t nl = 0;
        int banded = 0;
        if (len == 0) {
            /* the row reduces to zero by the reducers before it */
        } else if (bres_n != NULL && bres_n[i] != SG_NONE) {
            /* reduced in the band of an earlier row */
            nl = bres_n[i];
            memcpy(nrow + OFFSET, bres_c[i], (unsigned long)nl * sizeof(hm_t));
            memcpy(ncf, bres_v[i], (unsigned long)nl * sizeof(cf32_t));
            free(bres_c[i]);
            free(bres_v[i]);
            bres_c[i] = NULL;
            bres_v[i] = NULL;
            bres_n[i] = SG_NONE;
            banded = 1;
        } else if (band_rows) {
            /* the band: this row and the next ones to be reduced, up to a
             * row setting a pivot or a row which may be dropped */
            len_t g[SG_BAND], nb = 1, l;
            g[0] = i;
            for (l = i + 1; l < nr && nb < SG_BAND; ++l) {
                const sg_row_t * const x = rows + l;
                if (x->r == NULL) {
                    continue;
                }
                const hm_t xc = x->r[OFFSET];
                if (pc[xc] == NULL && x->reducer) {
                    break;
                }
                if (x->pr != NULL && x->pr[LENGTH] == 0) {
                    break;
                }
                if (x->ix == rows[l-1].ix && x->sm == rows[l-1].sm) {
                    continue;
                }
                g[nb++] = l;
            }
            if (nb > 1) {
                if (bres_n == NULL) {
                    bres_n = (len_t *)malloc((unsigned long)nr * sizeof(len_t));
                    bres_c = (hm_t **)calloc((unsigned long)nr, sizeof(hm_t *));
                    bres_v = (cf32_t **)calloc((unsigned long)nr, sizeof(cf32_t *));
                    for (l = 0; l < nr; ++l) {
                        bres_n[l] = SG_NONE;
                    }
                    /* a column of the band is a cache line */
                    const size_t bsz = (size_t)(nc + 1) * SG_BAND * sizeof(uint64_t);
                    if (posix_memalign((void **)&bd, 64, bsz) == 0) {
                        memset(bd, 0, bsz);
                    } else {
                        bd = (uint64_t *)calloc(bsz, 1);
                    }
                }
                hm_t *bc[SG_BAND];
                cf32_t *bv[SG_BAND];
                len_t bn[SG_BAND];
                for (l = 0; l < nb; ++l) {
                    bc[l] = (hm_t *)malloc((unsigned long)(nc + 1) * sizeof(hm_t));
                    bv[l] = (cf32_t *)malloc((unsigned long)(nc + 1) * sizeof(cf32_t));
                }
                sg_band_reduce(g, nb, rows, pc, pcf, bd, nc, p, big, small,
                        used, bc, bv, bn, &sg->nsubs);
                nl = bn[0];
                memcpy(nrow + OFFSET, bc[0], (unsigned long)nl * sizeof(hm_t));
                memcpy(ncf, bv[0], (unsigned long)nl * sizeof(cf32_t));
                free(bc[0]);
                free(bv[0]);
                for (l = 1; l < nb; ++l) {
                    bres_n[g[l]] = bn[l];
                    bres_c[g[l]] = bc[l];
                    bres_v[g[l]] = bv[l];
                }
                banded = 1;
            }
        }
        if (banded) {
        } else if (marked) {
            if (mk == NULL) {
                mk = (uint64_t *)calloc((unsigned long)nmw, sizeof(uint64_t));
            }
            for (j = 0; j < len; ++j) {
                const hm_t c = rr[OFFSET+j];
                dr[c] = rv[j];
                mk[c / 64] |= (uint64_t)1 << (c % 64);
            }
            /* the columns written are larger than the current one, so the
             * bits are read in increasing order */
            len_t w = lr / 64;
            while (1) {
                while (w < nmw && mk[w] == 0) {
                    ++w;
                }
                if (w == nmw) {
                    break;
                }
                k = w * 64 + (len_t)__builtin_ctzll(mk[w]);
                mk[w] &= mk[w] - 1;
                const uint64_t v = dr[k] % p;
                dr[k] = 0;
                if (v == 0) {
                    continue;
                }
                if (pc[k] == NULL) {
                    nrow[OFFSET+nl] = k;
                    ncf[nl++]       = (cf32_t)v;
                    continue;
                }
                const hm_t * const pr   = pc[k] + OFFSET;
                const len_t pl          = pc[k][LENGTH];
                sg_sub_row(dr, pr, pcf[k], pl, p - v, big, small);
                if (used != NULL) {
                    used[k] = 1;
                }
                for (j = 1; j < pl; ++j) {
                    mk[pr[j] / 64] |= (uint64_t)1 << (pr[j] % 64);
                }
                sg->nsubs += pl;
            }
        } else {
            for (j = 0; j < len; ++j) {
                dr[rr[OFFSET+j]] = rv[j];
            }
            for (k = lr; k < nc; ++k) {
                if (dr[k] == 0) {
                    continue;
                }
                const uint64_t v = dr[k] % p;
                dr[k] = 0;
                if (v == 0) {
                    continue;
                }
                if (pc[k] == NULL) {
                    nrow[OFFSET+nl] = k;
                    ncf[nl++]       = (cf32_t)v;
                    continue;
                }
                const len_t pl          = pc[k][LENGTH];
                sg_sub_row(dr, pc[k] + OFFSET, pcf[k], pl, p - v, big, small);
                if (used != NULL) {
                    used[k] = 1;
                }
                sg->nsubs += pl;
            }
        }
        passed  += nc - lr;
        written += len + (sg->nsubs - subs0);
        if (++counted % 64 == 0) {
            passed  /= 2;
            written /= 2;
        }
        if (nl == 0) {
            free(r);
            row->r = NULL;
            /* reduction to zero: a syzygy signature */
            sg->nzero++;
            sg_add_syzygy_exp(sg, row->ix, rst->ev[row->sm]);
            if (tests && ++streak >= 2) {
                /* the rest of the run: the rows to be reduced up to the
                 * next reducer row setting a pivot */
                len_t ng = 0, l;
                const len_t lim = blk == 0 ? nr : blk;
                for (l = i + 1; l < nr && ng < lim; ++l) {
                    const sg_row_t * const x = rows + l;
                    const hm_t xc = x->r[OFFSET];
                    if (x->reducer && pc[xc] == NULL) {
                        break;
                    }
                    if (x->ix == rows[l-1].ix && x->sm == rows[l-1].sm) {
                        continue;
                    }
                    if (!x->reducer && pc[xc] == NULL) {
                        /* a given row of a free lead column keeps its lead
                         * term, it cannot reduce to zero */
                        continue;
                    }
                    grp[ng++] = l;
                }
                if (ng >= 2 * zs) {
                    if (sg_zero_test(grp, ng, rows, pc, pcf, dr, nc, p, big,
                                small, zs, &sg->seed, &sg->nsubs)) {
                        /* all of them reduce to zero */
                        for (l = 0; l < ng; ++l) {
                            sg_row_t *x = rows + grp[l];
                            sg->nzero++;
                            sg_add_syzygy_exp(sg, x->ix, rst->ev[x->sm]);
                            free(x->r);
                            x->r = NULL;
                        }
                        /* rows of equal signature between them go, too */
                        for (l = i + 1; l < grp[ng-1]; ++l) {
                            if (rows[l].r != NULL && !(rows[l].reducer
                                        && pc[rows[l].r[OFFSET]] == NULL)) {
                                free(rows[l].r);
                                rows[l].r = NULL;
                            }
                        }
                        sg->ntested += ng;
                        i = grp[ng-1];
                        if (blk > 0) {
                            blk = 2 * blk > 64 ? 64 : 2 * blk;
                        }
                    } else {
                        blk    = 8;
                        streak = 0;
                    }
                }
            }
            continue;
        }
        streak = 0;
        /* a new element, made monic */
        const uint64_t inv = mod_p_inverse_32((int64_t)ncf[0], (int64_t)p);
        hm_t *er    = (hm_t *)malloc((unsigned long)(nl+OFFSET) * sizeof(hm_t));
        cf32_t *ecf = (cf32_t *)malloc((unsigned long)nl * sizeof(cf32_t));
        for (j = 0; j < nl; ++j) {
            er[OFFSET+j] = nrow[OFFSET+j];
            ecf[j]       = (cf32_t)(((uint64_t)ncf[j] * inv) % p);
        }
        er[LENGTH]  = nl;
        er[PRELOOP] = nl % UNROLL;
        er[COEFFS]  = 0;
        er[MULT]    = 0;
        er[BINDEX]  = 0;
        pc[er[OFFSET]]  = er;
        pcf[er[OFFSET]] = ecf;
        if (nn >= nsz) {
            nsz = 2 * nsz;
            nw  = realloc(nw, (unsigned long)nsz * sizeof(sg_row_t));
        }
        nw[nn]    = *row;
        nw[nn].r  = er;
        nw[nn].cf = ecf;
        nn++;
        if (rd != NULL) {
            /* see the encoding comment above */
            sg_rec_add(rd, row->src < sg->ng ?
                    SG_GEN | row->src : row->src - sg->ng,
                    row->mul, 1, r);
            kcol[rd->n-1] = nc;
        }
        free(r);
        row->r = NULL;
    }

    if (stopped) {
        /* the run hands over, the new elements are not used */
        for (i = 0; i < nn; ++i) {
            free(nw[i].r);
            free((void *)nw[i].cf);
        }
        nn = 0;
    }
    if (rd != NULL) {
        /* the rows setting pivots no reduction used are not recorded;
         * after sg_reduce_by_reducers() this is not known, all of them
         * are recorded */
        len_t m = 0;
        for (i = 0; i < rd->n; ++i) {
            if (rd->kind[i] == 1 || byred || used[kcol[i]]) {
                rd->src[m]  = rd->src[i];
                rd->mul[m]  = rd->mul[i];
                rd->kind[m] = rd->kind[i];
                rd->rc[m]   = rd->rc[i];
                m++;
            } else {
                free(rd->rc[i]);
            }
        }
        rd->n  = m;
        rd->nc = nc;
        rd->lm = (hm_t *)malloc((unsigned long)(nn + 1) * sizeof(hm_t));
        rd->ec = (hm_t **)malloc((unsigned long)(nn + 1) * sizeof(hm_t *));
        rd->eb = (hm_t **)malloc((unsigned long)(nn + 1) * sizeof(hm_t *));
    }
    free(used);
    free(kcol);
    /* the new elements in the basis hash table */
    const len_t ob = sg->ld;
    int constant   = 0;
    sg->ndiv       = 0;
    for (i = 0; i < nn; ++i) {
        hm_t *r = nw[i].r;
        const len_t len = r[LENGTH]+OFFSET;
        if (rd != NULL) {
            rd->ec[i] = sg_copy_terms(r + OFFSET, r[LENGTH]);
        }
        for (j = OFFSET; j < len; ++j) {
            r[j] = sg_insert(sht->ev[hcm[r[j]]], bht);
        }
        if (rd != NULL) {
            rd->eb[i] = sg_copy_terms(r + OFFSET, r[LENGTH]);
        }
        r[DEG] = bht->hd[r[OFFSET]].deg;
        if (r[DEG] == 0) {
            constant = 1;
        }
        if (rd != NULL) {
            rd->lm[rd->nn++] = r[OFFSET];
        }
        /* the Schreyer signature: the one of the element multiplied, times
         * the multiplier of the row */
        const sg_el_t * const el = sg->e + nw[i].src;
        const exp_t * const rsm  = rst->ev[nw[i].sm];
        const exp_t * const esm  = bht->ev[el->sm];
        const exp_t * const esc  = bht->ev[el->sc];
        exp_t sce[evl];
        for (j = 0; j < evl; ++j) {
            sce[j] = rsm[j] - esm[j] + esc[j];
        }
        const hm_t sc = sg_insert(sce, bht);
        const deg_t gap = bht->hd[sc].deg - r[DEG];
        if (gap > (deg_t)sg->maxgap) {
            sg->maxgap = gap;
        }
        const sdm_t nl = ~bht->hd[r[OFFSET]].sdm;
        for (k = 0; k < ob; ++k) {
            const hm_t l = sg->e[k].p[OFFSET];
            if (!(bht->hd[l].sdm & nl)
                    && sg_divides_exp(bht->ev[l], bht->ev[r[OFFSET]], evl)) {
                sg->ndiv++;
                break;
            }
        }
        sg_add_element(sg, r, (cf32_t *)nw[i].cf, nw[i].ix,
                sg_insert(rst->ev[nw[i].sm], bht), sc);
    }

    for (i = 0; i < nr; ++i) {
        free(rows[i].r);
        free(rows[i].pr);
        free(rows[i].pv);
    }
    free(rows);
    free(mk);
    free(grp);
    if (bres_n != NULL) {
        for (i = 0; i < nr; ++i) {
            free(bres_c[i]);
            free(bres_v[i]);
        }
        free(bres_n);
        free(bres_c);
        free(bres_v);
        free(bd);
    }
    free(pc);
    free(pcf);
    free(dr);
    free(nrow);
    free(ncf);
    free(nw);
    free(hcm);
    clean_hash_table(sht);

    if (md->info_level > 1) {
        fprintf(VERBSTREAM, " %7u new %7u zero %10.2f sec %lu subs | %u srows %u reds %lu nearly %u smon\n",
                (unsigned)nn, (unsigned)sg->nzero, realtime()-t0,
                (unsigned long)sg->nsubs, (unsigned)nsr, (unsigned)(nr-nsr),
                (unsigned long)(sg->nearly-nly0), (unsigned)smon);
        fflush(VERBSTREAM);
    }
    sg->psubs = sg->nsubs;
    if (stopped) {
        return -3;
    }
    return constant ? -1 : (int64_t)nn;
}

/* does the run hand over after a round that made nn new elements? */
static int sg_hand_over(
        sg_t *sg,
        const len_t nn,
        const len_t gap,
        md_t *md
        )
{
    const char *why = NULL;
    if (md->info_level > 1) {
        fprintf(VERBSTREAM, "hand-gate: nn=%d nzero=%d ndiv=%d gap=%d ld=%d nsubs=%lu\n",
                nn, sg->nzero, sg->ndiv, gap, sg->ld, (unsigned long)sg->nsubs);
    }
    if (md->sig_f4 == 2) {
        /* the signature-based F4 is not to hand over */
        return 0;
    }
    if (nn > 0 && sg->early_done == 0 && gap >= SG_EARLY) {
        sg->early_done = 1;
        /* when a trace is learned, the signature-based run hands over
         * for more zero reductions: the time of a run handing over late
         * is lost, while F4 applies its trace about as fast */
        const uint64_t ez = md->trace_level == LEARN_TRACER
            ? SG_EARLY_ZERO_LEARN : SG_EARLY_ZERO;
        if (nn >= SG_EARLY_ROUND && (100 * (uint64_t)sg->nzero < ez * nn
                    || 10 * sg->ndiv > 6 * nn)) {
            why = "early round";
        }
    }
    if (why == NULL && nn > 0 && sg->gap_done == 0 && gap >= SG_GAP) {
        sg->gap_done = 1;
        if (2 * sg->nzero < nn || 10 * sg->ndiv > 6 * nn) {
            why = "gap round";
        }
    }
    if (why == NULL && nn >= SG_OUTGROWN_MIN && 10 * sg->ndiv >= 8 * nn) {
        why = "outgrown";
    }

    if (why == NULL && nn > 0) {
        /* the S-pairs the update would form against the work of the
         * round */
        const uint64_t ob    = sg->ld - nn;
        const uint64_t pairs = (uint64_t)nn * ob + (uint64_t)nn * (nn-1) / 2;
        const uint64_t work  = SG_PAIR_WORK_NUM * sg->nsubs / SG_PAIR_WORK_DEN;
        if (pairs >= SG_PAIRS && pairs >= work) {
            why = "pair work";
        }
    }
    if (why != NULL && md->info_level > 0) {
        fprintf(VERBSTREAM, "signature-based F4 hands over to F4 (%s)\n", why);
    }
    return why != NULL;
}

static void sg_free(
        sg_t *sg
        )
{
    len_t i;
    if (sg->rst != NULL) {
        free_hash_table(&(sg->rst));
        sg->rst = NULL;
    }
    for (i = 0; i < sg->ld; ++i) {
        /* the input generators keep their polynomials in bs, only their
         * coefficients were copied for the elements */
        if (i >= sg->ng) {
            free(sg->e[i].p);
        }
        free(sg->e[i].cf);
    }
    free(sg->e);
    free(sg->rord);
    free(sg->pw);
    free(sg->gpl);
    for (i = 0; i < sg->trisz && sg->tri != NULL; ++i) {
        free(sg->tri[i]);
    }
    free(sg->tri);
    for (i = 0; i < sg->ng; ++i) {
        free(sg->ixe[i]);
        free(sg->syz[i].m);
        free(sg->syz[i].w);
        free(sg->g[i].cf);
    }
    free(sg->ixe);
    free(sg->ixld);
    free(sg->ixsz);
    free(sg->syz);
    free(sg->syzhit);
    free(sg->g);
    free(sg->ps);
    free(sg->e0);
    free(sg->e1);
    free(sg->e2);
    free(sg->e3);
}

/* is the signature-based F4 applicable for this computation? */
static int sig_f4_applicable(
        const bs_t * const bs,
        const md_t * const md
        )
{
    return (md->sig_f4 > 0 || (md->sig_f4 < 0 && md->trace_level == NO_TRACER))
        && md->trace_level != APPLY_TRACER
        && md->fc > 2 && md->fc < ((uint64_t)1 << 31)
        && !is_gf2_ext(md->fc)
        && md->nev == 0 && bs->ht->mo == 0
        && !bs->ht->boolean
        && (md->ff_bits == 8 || md->ff_bits == 16 || md->ff_bits == 32)
        && md->ngens > 0;
}

/* the n elements (p[i], cf[i]) replace the ng input generators in bs, the
 * coefficient arrays are taken over or freed; cf holds cf32_t arrays,
 * except for ff_bits == 64 where it holds cf64_t arrays (which are taken
 * over without truncation) */
static void sg_load_basis(
        bs_t *bs,
        md_t *md,
        hm_t **p,
        void **cf,
        const len_t n,
        const len_t ng
        )
{
    len_t i, j;
    const ht_t * const bht = bs->ht;
    for (i = 0; i < ng; ++i) {
        const len_t ci = bs->hm[i][COEFFS];
        switch (md->ff_bits) {
            case 8:
                free(bs->cf_8[ci]);
                bs->cf_8[ci] = NULL;
                break;
            case 16:
                free(bs->cf_16[ci]);
                bs->cf_16[ci] = NULL;
                break;
            case 32:
                free(bs->cf_32[ci]);
                bs->cf_32[ci] = NULL;
                break;
            case 64:
                free(bs->cf_64[ci]);
                bs->cf_64[ci] = NULL;
                break;
            default:
                free(bs->cf_32[ci]);
                bs->cf_32[ci] = NULL;
                break;
        }
    }
    bs->ld = 0;
    check_enlarge_basis(bs, n, md);
    deg_t mltdeg = 0;
    for (i = 0; i < n; ++i) {
        hm_t *r = p[i];
        const len_t len = r[LENGTH];
        switch (md->ff_bits) {
            case 8:
                bs->cf_8[i] = (cf8_t *)malloc((unsigned long)len * sizeof(cf8_t));
                for (j = 0; j < len; ++j) {
                    bs->cf_8[i][j] = (cf8_t)((cf32_t *)cf[i])[j];
                }
                free(cf[i]);
                break;
            case 16:
                bs->cf_16[i] = (cf16_t *)malloc((unsigned long)len * sizeof(cf16_t));
                for (j = 0; j < len; ++j) {
                    bs->cf_16[i][j] = (cf16_t)((cf32_t *)cf[i])[j];
                }
                free(cf[i]);
                break;
            case 64:
                bs->cf_64[i] = (cf64_t *)cf[i];
                break;
            default:
                bs->cf_32[i] = (cf32_t *)cf[i];
                break;
        }
        r[COEFFS]   = i;
        bs->hm[i]   = r;
        bs->red[i]  = 0;
        bs->lmps[i] = i;
        bs->lm[i]   = bht->hd[r[OFFSET]].sdm;
        if (r[DEG] == 0) {
            bs->constant = 1;
        }
        if (r[DEG] > mltdeg) {
            mltdeg = r[DEG];
        }
    }
    bs->ld     = n;
    bs->lo     = n;
    bs->lml    = n;
    bs->mltdeg = mltdeg;
}

/* the input generators by increasing lead term, the order of the indices of
 * their signatures */
static len_t *sg_generator_order(
        const bs_t * const bs,
        const len_t ng
        )
{
    len_t i, j;
    len_t *gi = (len_t *)malloc((unsigned long)(ng + 1) * sizeof(len_t));
    for (i = 0; i < ng; ++i) {
        gi[i] = i;
    }
    for (i = 1; i < ng; ++i) {
        const len_t t = gi[i];
        for (j = i; j > 0 && monomial_cmp_drl(bs->hm[gi[j-1]][OFFSET],
                    bs->hm[t][OFFSET], bs->ht) > 0; --j) {
            gi[j] = gi[j-1];
        }
        gi[j] = t;
    }
    return gi;
}

/* Computes a Groebner basis of the input generators bs->hm[0], ...,
 * bs->hm[ngens-1] by the signature-based F4 algorithm. On success the
 * basis replaces the generators, bs->ld, bs->lmps and bs->lml are set for
 * the final reduction, and 1 is returned. If the run hands over, bs is
 * unchanged and 0 is returned. */
static int sig_f4(
        bs_t *bs,
        md_t *md,
        sg_rec_t *rec
        )
{
    len_t i, j, k;
    double rt = realtime();
    ht_t *bht = bs->ht;
    sg_t sg;
    memset(&sg, 0, sizeof(sg_t));
    sg.bht  = bht;
    sg.nthrds = md->nthrds;
    sg.sht  = md->ht;
    sg.rst  = initialize_secondary_hash_table(bht, md);
    sg.fc   = (uint32_t)md->fc;
    sg.seed = 0x9E3779B97F4A7C15ULL ^ md->fc;
    sg.rec  = rec;
    const len_t evl = bht->evl;
    sg.e0 = (exp_t *)calloc((unsigned long)evl, sizeof(exp_t));
    sg.e1 = (exp_t *)calloc((unsigned long)evl, sizeof(exp_t));
    sg.e2 = (exp_t *)calloc((unsigned long)evl, sizeof(exp_t));
    sg.e3 = (exp_t *)calloc((unsigned long)evl, sizeof(exp_t));

    /* the input generators in file order: e_i < e_j for i < j */
    const len_t ng = (len_t)md->ngens;
    len_t *gi = sg_generator_order(bs, ng);
    sg.ng   = ng;
    sg.g    = (sg_el_t *)calloc((unsigned long)ng, sizeof(sg_el_t));
    sg.ixe  = (len_t **)calloc((unsigned long)ng, sizeof(len_t *));
    sg.ixld = (len_t *)calloc((unsigned long)ng, sizeof(len_t));
    sg.ixsz = (len_t *)calloc((unsigned long)ng, sizeof(len_t));
    sg.syz  = (sg_syz_t *)calloc((unsigned long)ng, sizeof(sg_syz_t));
    sg.syzhit = (len_t *)calloc((unsigned long)ng, sizeof(len_t));
    sg.sz   = 64;
    sg.e    = (sg_el_t *)malloc((unsigned long)sg.sz * sizeof(sg_el_t));
    sg.rord = (len_t *)malloc((unsigned long)sg.sz * sizeof(len_t));
    sg.nw   = (bht->nv + SG_LPW - 1) / SG_LPW;
    sg.pw   = (uint64_t *)malloc((unsigned long)sg.sz * 2 * sg.nw * sizeof(uint64_t));
    sg.gpl  = (uint64_t *)malloc((unsigned long)ng * sg.nw * sizeof(uint64_t));
    sg.psz  = 2 * ng + 64;
    sg.ps   = (sg_sig_t *)malloc((unsigned long)sg.psz * sizeof(sg_sig_t));
    exp_t one[evl];
    memset(one, 0, (unsigned long)evl * sizeof(exp_t));
    sg_insert(one, bht);
    for (i = 0; i < ng; ++i) {
        const hm_t *b   = bs->hm[gi[i]];
        const len_t len = b[LENGTH];
        cf32_t *cf = (cf32_t *)malloc((unsigned long)len * sizeof(cf32_t));
        for (j = 0; j < len; ++j) {
            switch (md->ff_bits) {
                case 8:
                    cf[j] = bs->cf_8[b[COEFFS]][j];
                    break;
                case 16:
                    cf[j] = bs->cf_16[b[COEFFS]][j];
                    break;
                default:
                    cf[j] = bs->cf_32[b[COEFFS]][j];
                    break;
            }
        }
        if (sg_pack(bht->ev[b[OFFSET]], sg.gpl + (unsigned long)i * sg.nw,
                    evl, sg.nw) > SG_LMAX) {
            sg.wide = 1;
        }
        sg.g[i].p  = (hm_t *)b;
        sg.g[i].ix = i;
        sg.g[i].sm = b[OFFSET];
        sg.g[i].sc = b[OFFSET];
        /* the inputs enter with their lead terms as signatures (eder-2017
         * §7.2, Algorithm RB): the sig-lead ratio of a generator is 1, so
         * the signature slot sc holds the lead term */
        sg_add_element(&sg, (hm_t *)b, cf, i, b[OFFSET], b[OFFSET]);
    }
    /* the Koszul syzygies of the inputs: lm(f_i)*lm(f_j) e_i for j < i
     * (eder-2017 §7.2, Algorithm RB, line 3) */
    for (i = 1; i < ng; ++i) {
        for (j = 0; j < i; ++j) {
            const exp_t * const ei = bht->ev[sg.g[i].p[OFFSET]];
            const exp_t * const ej = bht->ev[sg.g[j].p[OFFSET]];
            for (k = 0; k < evl; ++k) {
                sg.e0[k] = ei[k] + ej[k];
            }
            if ((uint64_t)ei[0] + ej[0] > EXP_MAX) {
                continue;
            }
            sg_add_syzygy_exp(&sg, i, sg.e0);
        }
    }
    /* the S-pairs among the input generators */
    for (i = 0; i < ng; ++i) {
        sg_new_pairs(&sg, i);
    }
    if (rec != NULL) {
        rec->ng = ng;
        rec->gs = (hm_t **)malloc((unsigned long)(ng + 1) * sizeof(hm_t *));
        for (i = 0; i < ng; ++i) {
            rec->gs[i] = sg_copy_terms(bs->hm[gi[i]] + OFFSET, bs->hm[gi[i]][LENGTH]);
        }
    }
    free(gi);

    if (md->info_level > 1) {
        fprintf(VERBSTREAM, "signature-based F4\n");
    }
    int ok = sg.wide == 0;
    while (ok && sg.pld > 0) {
        const len_t ob    = sg.ld;
        sg.maxgap         = 0;
        const int64_t nn  = sg_round(&sg, md);
        if (nn == -2) {
            if (md->info_level > 0) {
                fprintf(VERBSTREAM, "signature-based F4 hands over to F4 (dense matrix)\n");
            }
            ok = 0;
            break;
        }
        if (nn == -3) {
            if (md->info_level > 0) {
                fprintf(VERBSTREAM, "signature-based F4 hands over to F4 (work)\n");
            }
            ok = 0;
            break;
        }
        if (nn < 0) {
            break;
        }
        if (sg_hand_over(&sg, (len_t)nn, sg.maxgap, md)) {
            ok = 0;
            break;
        }
        for (i = ob; i < sg.ld; ++i) {
            sg_principal_syzygies(&sg, i);
        }
        for (i = ob; i < sg.ld; ++i) {
            sg_new_pairs(&sg, i);
        }
        if (sg.wide) {
            /* a degree passes what the packed monomials can hold */
            if (md->info_level > 0) {
                fprintf(VERBSTREAM, "signature-based F4 hands over to F4 (degree)\n");
            }
            ok = 0;
            break;
        }
    }
    if (md->info_level > 1) {
        len_t ns = 0;
        for (i = 0; i < ng; ++i) {
            ns += sg.syz[i].ld;
        }
        fprintf(VERBSTREAM, "%u elements, %u syzygy signatures, %lu S-pairs: "
                "%lu singular, %lu by base divisors, %lu by syzygies, "
                "%lu Koszul, %lu queued; %lu rows dropped before and %lu "
                "rows tested zero by random combinations\n",
                (unsigned)sg.ld, (unsigned)ns, (unsigned long)sg.cnt_pairs,
                (unsigned long)sg.cnt_singular, (unsigned long)sg.cnt_base,
                (unsigned long)sg.cnt_syz, (unsigned long)sg.cnt_koszul,
                (unsigned long)sg.cnt_queued, (unsigned long)sg.nearly,
                (unsigned long)sg.ntested);
    }
    if (ok == 0) {
        sg_free(&sg);
        clean_hash_table(md->ht);
        md->sig_f4_rtime = realtime() - rt;
        return 0;
    }

    /* the signature basis replaces the generators */
    {
        hm_t **ep    = (hm_t **)malloc((unsigned long)(sg.ld + 1) * sizeof(hm_t *));
        cf32_t **ecf = (cf32_t **)malloc((unsigned long)(sg.ld + 1) * sizeof(cf32_t *));
        for (i = 0; i < sg.ld; ++i) {
            ep[i]       = sg.e[i].p;
            ecf[i]      = sg.e[i].cf;
            sg.e[i].p   = NULL;
            sg.e[i].cf  = NULL;
        }
        sg_load_basis(bs, md, ep, (void **)ecf, sg.ld, ng);
        free(ep);
        free(ecf);
    }
    sg.ld = 0;
    /* the input generators are freed above, their coefficients copied */
    sg_free(&sg);
    clean_hash_table(md->ht);
    md->sig_f4_rtime = realtime() - rt;
    return 1;
}

/* Applies the record of a signature-based run at another prime: the rows of
 * each round are formed from the elements of this prime and reduced in the
 * order of the record, each by the pivots of the rows before it, as in the
 * recorded run. A row setting a pivot must find its column free, a row
 * giving a new element must keep a nonzero remainder with the recorded
 * lead term, otherwise the prime is bad and 0 is returned. On success the
 * elements replace the input generators as after sig_f4() and 1 is
 * returned. The input generators are the same as in the recorded run,
 * here in echelon form, too.
 *
 * The core is instantiated for both supported coefficient widths: the
 * classic lazy representation for p < 2^32 and the redundant [0, 2p)
 * representation with Shoup-precomputed pivot factors for wide primes
 * p < 2^62. */

/* the input generators in the order of their signatures, coefficients
 * read from the arrays of the current field width */
static cf32_t **sg_input_coeffs_32(
        const bs_t * const bs,
        const len_t * const gi,
        const len_t ng,
        const md_t * const md
        )
{
    len_t i, j;
    cf32_t **gcf = (cf32_t **)malloc((unsigned long)(ng + 1) * sizeof(cf32_t *));
    for (i = 0; i < ng; ++i) {
        const hm_t *b   = bs->hm[gi[i]];
        const len_t len = b[LENGTH];
        gcf[i] = (cf32_t *)malloc((unsigned long)len * sizeof(cf32_t));
        for (j = 0; j < len; ++j) {
            switch (md->ff_bits) {
                case 8:
                    gcf[i][j] = bs->cf_8[b[COEFFS]][j];
                    break;
                case 16:
                    gcf[i][j] = bs->cf_16[b[COEFFS]][j];
                    break;
                default:
                    gcf[i][j] = bs->cf_32[b[COEFFS]][j];
                    break;
            }
        }
    }
    return gcf;
}

static cf64_t **sg_input_coeffs_64(
        const bs_t * const bs,
        const len_t * const gi,
        const len_t ng
        )
{
    len_t i, j;
    cf64_t **gcf = (cf64_t **)malloc((unsigned long)(ng + 1) * sizeof(cf64_t *));
    for (i = 0; i < ng; ++i) {
        const hm_t *b   = bs->hm[gi[i]];
        const len_t len = b[LENGTH];
        gcf[i] = (cf64_t *)malloc((unsigned long)len * sizeof(cf64_t));
        for (j = 0; j < len; ++j) {
            gcf[i][j] = bs->cf_64[b[COEFFS]][j];
        }
    }
    return gcf;
}

/* The rows of a round of a record giving new elements are reduced in
 * bands of up to SG_BAND rows, see sg_band_reduce(), where the rows of
 * the round have SG_BAND_MIN entries and more in the mean, as in
 * sg_round(). A band are consecutive such rows, up to the next row
 * setting a pivot. The remainders of the rows of the current band, from
 * its row i0 on, are kept for the rows in turn. */
#define SG_BAND_MIN 64

typedef struct
{
    int on;
    len_t i0;
    len_t nb;
    uint64_t *bd;
    hm_t *bc[SG_BAND];
    void *bv[SG_BAND];  /* cf32_t resp. cf64_t coefficients */
    len_t bn[SG_BAND];
#ifdef FF64_VL
    ff64_t f;           /* for the 64-bit replay, see sg_band_take64() */
    ff64_vc_t vc;
#endif
} sg_band_t;

/* whether the 64-bit replay at the prime p reduces in bands: with the
 * vector arithmetic of la_ff_64.c for p < 2^52 */
#ifdef FF64_VL
#define SG_BAND64(p) ((p) < ((uint64_t)1 << 52))
#else
#define SG_BAND64(p) 0
#endif

static void sg_band_start(
        sg_band_t *bs,
        const sg_rrd_t * const rd,
        hm_t **rows,
        const len_t nc,
        const int on
        )
{
    len_t i, b;
    uint64_t ents = 0;
    for (i = 0; i < rd->n; ++i) {
        ents += rows[i][LENGTH];
    }
    bs->on = on && ents >= SG_BAND_MIN * (uint64_t)rd->n;
    bs->nb = 0;
    bs->i0 = 0;
    bs->bd = NULL;
    for (b = 0; b < SG_BAND; ++b) {
        bs->bc[b] = NULL;
        bs->bv[b] = NULL;
    }
    if (bs->on) {
        const size_t bsz = (size_t)(nc + 1) * SG_BAND * sizeof(uint64_t);
        if (posix_memalign((void **)&(bs->bd), 64, bsz) == 0) {
            memset(bs->bd, 0, bsz);
        } else {
            bs->bd = (uint64_t *)calloc(bsz, 1);
        }
        for (b = 0; b < SG_BAND; ++b) {
            bs->bc[b] = (hm_t *)malloc((unsigned long)(nc + 1) * sizeof(hm_t));
            bs->bv[b] = malloc((unsigned long)(nc + 1) * sizeof(cf64_t));
        }
    }
}

static void sg_band_stop(
        sg_band_t *bs
        )
{
    len_t b;
    free(bs->bd);
    for (b = 0; b < SG_BAND; ++b) {
        free(bs->bc[b]);
        free(bs->bv[b]);
    }
}

/* the remainder of the row i giving a new element in (nrow, ncf), with
 * its length returned, from a band; SG_NONE where the row is not reduced
 * in a band */
static len_t sg_band_take(
        sg_band_t *bs,
        const len_t i,
        const sg_rrd_t * const rd,
        hm_t **rows,
        const cf32_t * const *rcf,
        hm_t **pc,
        cf32_t **pcf,
        const len_t nc,
        const uint64_t p,
        const uint64_t big,
        const int small,
        hm_t *nrow,
        cf32_t *ncf
        )
{
    if (!bs->on) {
        return SG_NONE;
    }
    if (bs->nb == 0 || i < bs->i0 || i >= bs->i0 + bs->nb) {
        /* a new band from row i on */
        len_t g[SG_BAND], nb = 0, l;
        sg_row_t br[SG_BAND];
        for (l = i; l < rd->n && nb < SG_BAND && rd->kind[l] == 1; ++l) {
            br[nb].r  = rows[l];
            br[nb].cf = rcf[l];
            br[nb].pr = NULL;
            br[nb].pv = NULL;
            g[nb]     = nb;
            nb++;
        }
        if (nb < 2) {
            bs->nb = 0;
            return SG_NONE;
        }
        uint64_t nsubs = 0;
        sg_band_reduce(g, nb, br, pc, pcf, bs->bd, nc, p, big, small,
                NULL, bs->bc, (cf32_t **)bs->bv, bs->bn, &nsubs);
        bs->i0 = i;
        bs->nb = nb;
    }
    const len_t b  = i - bs->i0;
    const len_t nl = bs->bn[b];
    memcpy(nrow, bs->bc[b], (unsigned long)nl * sizeof(hm_t));
    memcpy(ncf, bs->bv[b], (unsigned long)nl * sizeof(cf32_t));
    return nl;
}

#ifdef FF64_VL
/* sg_band_reduce() for the 64-bit replay at a prime p < 2^52: the
 * entries are kept in [0, p), a pivot row is subtracted from the rows of
 * the band with the vector arithmetic of la_ff_64.c, FF64_VL entries of a
 * column at once */
static void sg_band_reduce64(
        const len_t nb,
        hm_t **rows,
        const cf64_t * const *rcf,
        hm_t **pc,
        cf64_t **pcf,
        uint64_t *bd,
        const len_t nc,
        const ff64_t * const f,
        const ff64_vc_t * const vc,
        hm_t **ncol,
        cf64_t **nval,
        len_t *nnl
        )
{
    len_t b, a, j, k, l;
    const uint64_t p = f->p;
    hm_t from = nc;
    len_t lead[SG_BAND];
    uint64_t inv[SG_BAND], rel[SG_BAND][SG_BAND], e[SG_BAND];
    memset(rel, 0, sizeof(rel));
    for (b = 0; b < nb; ++b) {
        const hm_t * const r = rows[b] + OFFSET;
        const len_t len = rows[b][LENGTH];
        for (j = 0; j < len; ++j) {
            bd[(unsigned long)r[j] * SG_BAND + b] = rcf[b][j];
        }
        if (r[0] < from) {
            from = r[0];
        }
        lead[b] = nc;
        nnl[b]  = 0;
    }
    for (k = from; k < nc; ++k) {
        uint64_t * const x = bd + (unsigned long)k * SG_BAND;
        uint64_t any = 0;
        for (b = 0; b < SG_BAND; ++b) {
            any |= x[b];
        }
        if (any == 0) {
            continue;
        }
        if (pc[k] != NULL) {
            /* the entries are the multipliers of the pivot row */
            uint64_t m[SG_BAND];
            len_t nz = 0, one = 0;
            for (b = 0; b < SG_BAND; ++b) {
                m[b] = x[b];
                x[b] = 0;
                if (m[b] != 0) {
                    ++nz;
                    one = b;
                }
            }
            const hm_t * const pr   = pc[k] + OFFSET;
            const cf64_t * const pf = pcf[k];
            const len_t pl          = pc[k][LENGTH];
            if (nz == 1) {
                const uint64_t mm = ff64_mont(m[one], f);
                for (j = 1; j < pl; ++j) {
                    uint64_t * const y = bd + (unsigned long)pr[j] * SG_BAND + one;
                    *y = (uint64_t)ff64_submul((int64_t)*y, pf[j], mm, f);
                }
            } else {
                ff64_m mv[SG_BAND / FF64_VL];
                for (l = 0; l < SG_BAND / FF64_VL; ++l) {
                    mv[l] = ff64_m_from(ff64_v_load(m + l * FF64_VL), vc);
                }
                for (j = 1; j < pl; ++j) {
                    uint64_t * const y = bd + (unsigned long)pr[j] * SG_BAND;
                    const ff64_v fv = ff64_v_set1(pf[j]);
                    for (l = 0; l < SG_BAND / FF64_VL; ++l) {
                        ff64_v_store(y + l * FF64_VL, ff64_v_submul(
                                    ff64_v_load(y + l * FF64_VL), fv, mv[l], vc));
                    }
                }
            }
            continue;
        }
        /* a column without pivot: the entries of the rows in turn */
        len_t owner = nb;
        for (b = 0; b < nb; ++b) {
            uint64_t v = x[b];
            x[b] = 0;
            for (a = 0; a < b; ++a) {
                if (rel[b][a] != 0 && e[a] != 0) {
                    const uint64_t t = mul_mod_64(rel[b][a], e[a], p);
                    v = v >= t ? v - t : v + p - t;
                }
            }
            if (v != 0 && owner < b) {
                /* the row is reduced by the band row taking this column */
                rel[b][owner] = mul_mod_64(v, inv[owner], p);
                v = 0;
            } else if (v != 0 && lead[b] == nc) {
                lead[b]  = k;
                owner    = b;
                inv[b]   = mod_p_inverse_64((int64_t)v, (int64_t)p);
            }
            e[b] = v;
            if (v != 0) {
                ncol[b][nnl[b]]   = k;
                nval[b][nnl[b]++] = v;
            }
        }
        for (; b < SG_BAND; ++b) {
            x[b] = 0;
        }
    }
}
#endif

/* the arithmetic modulo p of the 64-bit replay in bands */
static void sg_band_init64(
        sg_band_t *bs,
        const uint64_t p
        )
{
#ifdef FF64_VL
    if (bs->on) {
        bs->f  = ff64_init(p);
        bs->vc = ff64_vc_init(&(bs->f));
    }
#else
    (void)bs;
    (void)p;
#endif
}

/* whether most rows of the record recp, at least half of them, lie in
 * rounds which the replay reduces in bands, see sg_band_start() */
int sig_record_banded(
        const void * const recp
        )
{
    const sg_rec_t * const rec = (const sg_rec_t *)recp;
    uint64_t all = 0, banded = 0;
    len_t r, i;
    for (r = 0; r < rec->nrd; ++r) {
        const sg_rrd_t * const rd = rec->rd + r;
        uint64_t ents = 0;
        all += rd->n;
        if (rd->rc == NULL) {
            continue;
        }
        for (i = 0; i < rd->n; ++i) {
            ents += rd->rc[i][LENGTH];
        }
        if (ents >= SG_BAND_MIN * (uint64_t)rd->n) {
            banded += rd->n;
        }
    }
    return 2 * banded >= all && all > 0;
}

/* sg_band_take() for the 64-bit replay, see sg_band_reduce64() */
static len_t sg_band_take64(
        sg_band_t *bs,
        const len_t i,
        const sg_rrd_t * const rd,
        hm_t **rows,
        const cf64_t * const *rcf,
        hm_t **pc,
        cf64_t **pcf,
        const len_t nc,
        hm_t *nrow,
        cf64_t *ncf
        )
{
#ifdef FF64_VL
    if (!bs->on) {
        return SG_NONE;
    }
    if (bs->nb == 0 || i < bs->i0 || i >= bs->i0 + bs->nb) {
        /* a new band from row i on */
        len_t nb = 0, l;
        for (l = i; l < rd->n && nb < SG_BAND && rd->kind[l] == 1; ++l) {
            nb++;
        }
        if (nb < 2) {
            bs->nb = 0;
            return SG_NONE;
        }
        sg_band_reduce64(nb, rows + i, rcf + i, pc, pcf, bs->bd, nc,
                &(bs->f), &(bs->vc), bs->bc, (cf64_t **)bs->bv, bs->bn);
        bs->i0 = i;
        bs->nb = nb;
    }
    const len_t b  = i - bs->i0;
    const len_t nl = bs->bn[b];
    memcpy(nrow, bs->bc[b], (unsigned long)nl * sizeof(hm_t));
    memcpy(ncf, bs->bv[b], (unsigned long)nl * sizeof(cf64_t));
    return nl;
#else
    (void)bs; (void)i; (void)rd; (void)rows; (void)rcf; (void)pc;
    (void)pcf; (void)nc; (void)nrow; (void)ncf;
    return SG_NONE;
#endif
}

/* coefficient-width variants of the replay operations */

#define SG_INPUT_COEFFS_32(bs, gi, ng, md) \
    sg_input_coeffs_32((bs), (gi), (ng), (md))
#define SG_INPUT_COEFFS_64(bs, gi, ng, md) \
    sg_input_coeffs_64((bs), (gi), (ng))

/* a dense entry is reduced to its canonical representative and the
 * entry is cleared: the 32-bit rows hold lazy entries below 2^63, the
 * 64-bit rows hold redundant entries in [0, 2p) */
#define SG_TAKE_32(v, dr, k, p)             \
    do {                                    \
        (v)     = (dr)[k] % (p);            \
        (dr)[k] = 0;                        \
    } while (0)
#define SG_TAKE_64(v, dr, k, p)                             \
    do {                                                    \
        (v)     = (dr)[k] >= (p) ? (dr)[k] - (p) : (dr)[k]; \
        (dr)[k] = 0;                                        \
    } while (0)

/* subtracting the pivot row (pr, pf) from the dense row: the 32-bit
 * instantiation uses the lazy big/small constants, the 64-bit
 * instantiation the Shoup product with the pre-inverse of the
 * multiplier */
#define SG_SUBROW_32(dr, pr, pf, pl, v, p, big, small) \
    sg_sub_row((dr), (pr), (pf), (pl), (p) - (v), (big), (small))
#define SG_SUBROW_64(dr, pr, pf, pl, v, p, big, small) \
    sg_sub_row_64((dr), (pr), (pf), (pl), (v), (p), &(big))

/* inverting the lead coefficient and scaling by the inverse for
 * monicization */
#define SG_INV_32(x, p) mod_p_inverse_32((int64_t)(x), (int64_t)(p))
#define SG_INV_64(x, p) mod_p_inverse_64((int64_t)(x), (int64_t)(p))
#define SG_MUL_INV_32(x, inv, invs, p) (cf32_t)(((uint64_t)(x) * (inv)) % (p))
#define SG_MUL_INV_64(x, inv, invs, p) \
    (cf64_t)shoup_mod_mul((inv), (invs), (x), (p))
/* the Shoup pre-inverse of the inverse, only used by the 64-bit
 * instantiation */
#define SG_INV_PRE_32(inv, big) 0
#define SG_INV_PRE_64(inv, big) shoup_mu_preinv(&(big), (inv))

/* local constants: the lazy representation of the 32-bit instantiation,
 * the reciprocal of p for the Shoup pre-inverses of the 64-bit one */
#define SG_BIG_SMALL_DECL_32(p) \
    const uint64_t big = ((uint64_t)1 << 63) / (p) * (p); \
    const int small = (p) < ((uint64_t)1 << 16)
#define SG_BIG_SMALL_DECL_64(p) \
    const preinv_t big = preinv_64(p)

/* the rows giving new elements reduced in bands, only by the 32-bit
 * instantiation */
#define SG_BAND_TAKE_32(bs, i, rd, rows, rcf, pc, pcf, nc, p, big, small, \
        nrow, ncf)                                                      \
    sg_band_take((bs), (i), (rd), (rows), (const cf32_t * const *)(rcf), \
            (pc), (pcf), (nc), (p), (big), (small), (nrow), (ncf))
#define SG_BAND_TAKE_64(bs, i, rd, rows, rcf, pc, pcf, nc, p, big, small, \
        nrow, ncf)                                                      \
    sg_band_take64((bs), (i), (rd), (rows), (const cf64_t * const *)(rcf), \
            (pc), (pcf), (nc), (nrow), (ncf))

/* the replay core shared by both coefficient widths: cf_t is the
 * coefficient type, the variant macros adapt input coefficient
 * extraction, dense-row invariant, pivot subtraction and monicization */
#define SG_REPLAY(name, cf_t, INPUT_COEFFS, TAKE, SUBROW, INV, MUL_INV, \
        INV_PRE, BIG_SMALL_DECL, BANDED, BAND_TAKE)                     \
static int name(                                                        \
        bs_t *bs,                                                       \
        md_t *md,                                                       \
        const sg_rec_t * const rec                                      \
        )                                                               \
{                                                                       \
    len_t i, j, k, r;                                                   \
    double rt = realtime();                                             \
    ht_t *bht       = bs->ht;                                           \
    ht_t *sht       = md->ht;                                           \
    const len_t evl = bht->evl;                                         \
    const uint64_t p = md->fc;                                          \
    BIG_SMALL_DECL(p);                                                  \
    const len_t ng  = (len_t)md->ngens;                                 \
    int ok = 1;                                                         \
                                                                        \
    /* the input generators in the order of their signatures */         \
    len_t *gi  = sg_generator_order(bs, ng);                            \
    cf_t **gcf = INPUT_COEFFS(bs, gi, ng, md);                          \
                                                                        \
    len_t nel, esz = 64;                                                \
    hm_t **ep  = (hm_t **)malloc((unsigned long)esz * sizeof(hm_t *));  \
    cf_t **ecf = (cf_t **)malloc((unsigned long)esz * sizeof(cf_t *));  \
                                                                        \
    /* elements with the supports of the recorded run: their rows take the \
     * recorded columns, no monomial is hashed */                       \
    int8_t *gcl = (int8_t *)calloc((unsigned long)(ng + 1), sizeof(int8_t)); \
    int8_t *ecl = (int8_t *)calloc((unsigned long)esz, sizeof(int8_t)); \
    for (i = 0; i < ng && rec->gs != NULL; ++i) {                       \
        const hm_t *b = bs->hm[gi[i]];                                  \
        gcl[i] = b[LENGTH] == rec->gs[i][0] && memcmp(b + OFFSET, rec->gs[i] + 1, \
                (unsigned long)b[LENGTH] * sizeof(hm_t)) == 0;          \
    }                                                                   \
    /* the generators open the replayed basis as they open the basis of \
     * the recorded run; their polynomials stay in bs, their freshly \
     * extracted coefficients are taken over by the basis loader */     \
    while (ng >= esz) {                                                 \
        esz = 2 * esz;                                                  \
    }                                                                   \
    ep  = (hm_t **)realloc(ep, (unsigned long)esz * sizeof(hm_t *));    \
    ecf = (cf_t **)realloc(ecf, (unsigned long)esz * sizeof(cf_t *));   \
    ecl = (int8_t *)realloc(ecl, (unsigned long)esz * sizeof(int8_t));  \
    for (i = 0; i < ng; ++i) {                                          \
        ep[i]  = (hm_t *)bs->hm[gi[i]];                                 \
        ecf[i] = gcf[i];                                                \
    }                                                                   \
    nel = ng;                                                           \
                                                                        \
    for (r = 0; r < rec->nrd && ok; ++r) {                              \
        if (md->deadline != NULL && md->deadline[0] > 0) {           \
            /* the share of the rows done so far */                     \
            uint64_t wd = 0, wa = 0;                                    \
            len_t r2;                                                   \
            for (r2 = 0; r2 < rec->nrd; ++r2) {                         \
                wa += rec->rd[r2].n;                                    \
                wd += r2 < r ? rec->rd[r2].n : 0;                       \
            }                                                           \
            if (apply_given_up(md, realtime(), rt,                      \
                        wa > 0 ? (double)wd / (double)wa : 0)) {        \
                /* the application is given up, see md_t->deadline */  \
                ok = 0;                                                 \
                break;                                                  \
            }                                                           \
        }                                                               \
        const sg_rrd_t * const rd = rec->rd + r;                        \
        const len_t nr = rd->n;                                         \
        if (nr == 0) {                                                  \
            continue;                                                   \
        }                                                               \
        int cached = rd->rc != NULL;                                    \
        for (i = 0; i < nr && cached; ++i) {                            \
            const len_t src = rd->src[i];                               \
            cached = (src & SG_GEN) ? gcl[src & ~SG_GEN] : ecl[src + ng];    \
        }                                                               \
    redo: ;                                                             \
        /* the rows */                                                  \
        hm_t **rows           = (hm_t **)malloc((unsigned long)nr * sizeof(hm_t *)); \
        const cf_t **rcf      = (const cf_t **)malloc((unsigned long)nr * sizeof(cf_t *)); \
        len_t nc;                                                       \
        hi_t *hcm = NULL;                                               \
        for (i = 0; i < nr; ++i) {                                      \
            const len_t src   = rd->src[i];                             \
            const hm_t *poly  = (src & SG_GEN) ? bs->hm[gi[src & ~SG_GEN]] : ep[src + ng]; \
            rcf[i]            = (src & SG_GEN) ? gcf[src & ~SG_GEN] : ecf[src + ng]; \
            if (cached) {                                               \
                /* the recorded row is read in place */                 \
                rows[i] = rd->rc[i];                                    \
                continue;                                               \
            }                                                           \
            exp_t em[evl];                                              \
            memcpy(em, bht->ev[rd->mul[i]], (unsigned long)evl * sizeof(exp_t)); \
            val_t h = 0;                                                \
            for (j = 0; j < evl; ++j) {                                 \
                h += bht->rn[j] * em[j];                                \
            }                                                           \
            rows[i] = multiplied_poly_to_matrix_row(sht, bht, h, em, poly); \
        }                                                               \
        if (cached) {                                                   \
            nc = rd->nc;                                                \
        } else {                                                        \
            /* the columns by decreasing monomials */                   \
            nc  = (len_t)(sht->eld - 1);                                \
            hcm = (hi_t *)malloc((unsigned long)(nc + 1) * sizeof(hi_t)); \
            for (i = 0; i < nc; ++i) {                                  \
                hcm[i] = i + 1;                                         \
            }                                                           \
            sg_sort_columns(hcm, nc, sht);                              \
            for (i = 0; i < nc; ++i) {                                  \
                sht->hd[hcm[i]].idx = i;                                \
            }                                                           \
            for (i = 0; i < nr; ++i) {                                  \
                hm_t *row = rows[i];                                    \
                const len_t len = row[LENGTH] + OFFSET;                 \
                for (j = OFFSET; j < len; ++j) {                        \
                    row[j] = sht->hd[row[j]].idx;                       \
                }                                                       \
            }                                                           \
        }                                                               \
        /* the reduction in the recorded order */                       \
        hm_t **pc       = (hm_t **)calloc((unsigned long)nc + 1, sizeof(hm_t *)); \
        cf_t **pcf      = (cf_t **)calloc((unsigned long)nc + 1, sizeof(cf_t *)); \
        uint64_t *dr    = (uint64_t *)calloc((unsigned long)nc + 1, sizeof(uint64_t)); \
        hm_t *nrow      = (hm_t *)malloc((unsigned long)(nc + 1) * sizeof(hm_t)); \
        cf_t *ncf       = (cf_t *)malloc((unsigned long)(nc + 1) * sizeof(cf_t)); \
        hm_t **nw       = (hm_t **)malloc((unsigned long)(rd->nn + 1) * sizeof(hm_t *)); \
        cf_t **nwc      = (cf_t **)malloc((unsigned long)(rd->nn + 1) * sizeof(cf_t *)); \
        len_t nn = 0;                                                   \
        const len_t nmw = nc / 64 + 1;                                  \
        uint64_t *mk    = NULL;                                         \
        uint64_t passed = 0, written = 0;                               \
        len_t counted   = 0;                                            \
        sg_band_t bst;                                                  \
        bst.on = 0;                                                     \
        if (BANDED) {                                                   \
            sg_band_start(&bst, rd, rows, nc,                           \
                    sizeof(cf_t) == sizeof(cf32_t) || SG_BAND64(p));    \
            if (sizeof(cf_t) == sizeof(cf64_t)) {                       \
                sg_band_init64(&bst, p);                                \
            }                                                           \
        }                                                               \
        for (i = 0; i < nr && ok; ++i) {                                \
            hm_t *row     = rows[i];                                    \
            const hm_t lc = row[OFFSET];                                \
            if (rd->kind[i] == 0) {                                     \
                if (pc[lc] != NULL) {                                   \
                    ok = 0;                                             \
                    break;                                              \
                }                                                       \
                pc[lc]  = row;                                          \
                pcf[lc] = (cf_t *)rcf[i];                               \
                continue;                                               \
            }                                                           \
            len_t nl = SG_NONE;                                         \
            if (BANDED) {                                               \
                nl = BAND_TAKE(&bst, i, rd, rows, rcf, pc, pcf, nc, p,  \
                        big, small, nrow, ncf);                         \
            }                                                           \
            if (nl == SG_NONE) {                                        \
            const len_t len = row[LENGTH];                              \
            nl = 0;                                                     \
            uint64_t wr = len;                                          \
            if (counted >= 4 && passed > 2 * written) {                 \
                /* the marked scan, see sg_round() */                   \
                if (mk == NULL) {                                       \
                    mk = (uint64_t *)calloc((unsigned long)nmw, sizeof(uint64_t)); \
                }                                                       \
                for (j = 0; j < len; ++j) {                             \
                    const hm_t c = row[OFFSET+j];                       \
                    dr[c] = rcf[i][j];                                  \
                    mk[c / 64] |= (uint64_t)1 << (c % 64);              \
                }                                                       \
                len_t w = lc / 64;                                      \
                while (1) {                                             \
                    while (w < nmw && mk[w] == 0) {                     \
                        ++w;                                            \
                    }                                                   \
                    if (w == nmw) {                                     \
                        break;                                          \
                    }                                                   \
                    k = w * 64 + (len_t)__builtin_ctzll(mk[w]);         \
                    mk[w] &= mk[w] - 1;                                 \
                    uint64_t v;                                         \
                    TAKE(v, dr, k, p);                                  \
                    if (v == 0) {                                       \
                        continue;                                       \
                    }                                                   \
                    if (pc[k] == NULL) {                                \
                        nrow[nl]  = k;                                  \
                        ncf[nl++] = (cf_t)v;                            \
                        continue;                                       \
                    }                                                   \
                    const hm_t * const pr = pc[k] + OFFSET;             \
                    const len_t pl        = pc[k][LENGTH];              \
                    SUBROW(dr, pr, pcf[k], pl, v, p, big, small);       \
                    for (j = 1; j < pl; ++j) {                          \
                        mk[pr[j] / 64] |= (uint64_t)1 << (pr[j] % 64);  \
                    }                                                   \
                    wr += pl;                                           \
                }                                                       \
            } else {                                                    \
                for (j = 0; j < len; ++j) {                             \
                    dr[row[OFFSET+j]] = rcf[i][j];                      \
                }                                                       \
                for (k = lc; k < nc; ++k) {                             \
                    if (dr[k] == 0) {                                   \
                        continue;                                       \
                    }                                                   \
                    uint64_t v;                                         \
                    TAKE(v, dr, k, p);                                  \
                    if (v == 0) {                                       \
                        continue;                                       \
                    }                                                   \
                    if (pc[k] == NULL) {                                \
                        nrow[nl]  = k;                                  \
                        ncf[nl++] = (cf_t)v;                            \
                        continue;                                       \
                    }                                                   \
                    SUBROW(dr, pc[k] + OFFSET, pcf[k], pc[k][LENGTH], v, p, big, small); \
                    wr += pc[k][LENGTH];                                \
                }                                                       \
            }                                                           \
            passed  += nc - lc;                                         \
            written += wr;                                              \
            if (++counted % 64 == 0) {                                  \
                passed  /= 2;                                           \
                written /= 2;                                           \
            }                                                           \
            }                                                           \
            /* the remainder must be nonzero with the recorded lead term */ \
            if (nl == 0 || nn >= rd->nn || (cached                      \
                        ? (rd->ec[nn][0] != nl || memcmp(rd->ec[nn] + 1, nrow, \
                                (unsigned long)nl * sizeof(hm_t)) != 0) \
                        : memcmp(sht->ev[hcm[nrow[0]]], bht->ev[rd->lm[nn]], \
                            (unsigned long)evl * sizeof(exp_t)) != 0)) { \
                ok = 0;                                                 \
                break;                                                  \
            }                                                           \
            const uint64_t inv  = INV(ncf[0], p);                       \
            const uint64_t invs = INV_PRE(inv, big);                    \
            (void)invs;                                                 \
            hm_t *er    = (hm_t *)malloc((unsigned long)(nl + OFFSET) * sizeof(hm_t)); \
            cf_t *ec    = (cf_t *)malloc((unsigned long)nl * sizeof(cf_t)); \
            for (j = 0; j < nl; ++j) {                                  \
                er[OFFSET+j] = nrow[j];                                 \
                ec[j]        = MUL_INV(ncf[j], inv, invs, p);           \
            }                                                           \
            er[LENGTH]  = nl;                                           \
            er[PRELOOP] = nl % UNROLL;                                  \
            er[COEFFS]  = 0;                                            \
            er[MULT]    = 0;                                            \
            er[BINDEX]  = 0;                                            \
            pc[er[OFFSET]]  = er;                                       \
            pcf[er[OFFSET]] = ec;                                       \
            nw[nn]  = er;                                               \
            nwc[nn] = ec;                                               \
            nn++;                                                       \
        }                                                               \
        if (ok && nn != rd->nn) {                                       \
            ok = 0;                                                     \
        }                                                               \
        /* in the cached matrix any difference is due to supports, the \
         * round is done again with hashed monomials */                 \
        const int again = cached && !ok;                                \
        if (again) {                                                    \
            ok = 1;                                                     \
        }                                                               \
        /* the new elements in the basis hash table */                  \
        for (i = 0; i < nn; ++i) {                                      \
            hm_t *er = nw[i];                                           \
            if (ok && !again) {                                         \
                const len_t len = er[LENGTH] + OFFSET;                  \
                int clean;                                              \
                if (cached) {                                           \
                    memcpy(er + OFFSET, rd->eb[i] + 1,                  \
                            (unsigned long)er[LENGTH] * sizeof(hm_t));  \
                    clean = 1;                                          \
                } else {                                                \
                    for (j = OFFSET; j < len; ++j) {                    \
                        er[j] = sg_insert(sht->ev[hcm[er[j]]], bht);    \
                    }                                                   \
                    clean = rd->eb != NULL && rd->eb[i][0] == er[LENGTH] \
                        && memcmp(rd->eb[i] + 1, er + OFFSET,           \
                                (unsigned long)er[LENGTH] * sizeof(hm_t)) == 0; \
                }                                                       \
                er[DEG] = bht->hd[er[OFFSET]].deg;                      \
                if (nel >= esz) {                                       \
                    esz = 2 * esz;                                      \
                    ep  = realloc(ep, (unsigned long)esz * sizeof(hm_t *)); \
                    ecf = realloc(ecf, (unsigned long)esz * sizeof(cf_t *)); \
                    ecl = realloc(ecl, (unsigned long)esz * sizeof(int8_t)); \
                }                                                       \
                ep[nel]  = er;                                          \
                ecf[nel] = nwc[i];                                      \
                ecl[nel] = (int8_t)clean;                               \
                nel++;                                                  \
            } else {                                                    \
                free(er);                                               \
                free(nwc[i]);                                           \
            }                                                           \
        }                                                               \
        for (i = 0; i < nr && !cached; ++i) {                           \
            free(rows[i]);                                              \
        }                                                               \
        free(rows);                                                     \
        free(rcf);                                                      \
        free(hcm);                                                      \
        free(pc);                                                       \
        free(pcf);                                                      \
        free(dr);                                                       \
        free(nrow);                                                     \
        free(ncf);                                                      \
        free(nw);                                                       \
        free(nwc);                                                      \
        free(mk);                                                       \
        if (BANDED) {                                                   \
            sg_band_stop(&bst);                                         \
        }                                                               \
        if (!cached) {                                                  \
            clean_hash_table(sht);                                      \
        }                                                               \
        if (again) {                                                    \
            cached = 0;                                                 \
            goto redo;                                                  \
        }                                                               \
    }                                                                   \
    free(gcl);                                                          \
    free(ecl);                                                          \
    /* the generator coefficients are owned by ecf[0..ng) now: taken    \
     * over or copied by the basis loader on success, freed in the      \
     * failure branch below */                                          \
    free(gcf);                                                          \
    free(gi);                                                           \
    if (ok) {                                                           \
        sg_load_basis(bs, md, ep, (void **)ecf, nel, ng);               \
    } else {                                                            \
        for (i = ng; i < nel; ++i) {                                    \
            free(ep[i]);                                                \
        }                                                               \
        for (i = 0; i < nel; ++i) {                                     \
            free(ecf[i]);                                               \
        }                                                               \
    }                                                                   \
    free(ep);                                                           \
    free(ecf);                                                          \
    md->sig_f4_rtime = realtime() - rt;                                 \
    return ok;                                                          \
}

/* whether an application of a trace which started at the time start and
 * has done the share done of its work is given up at the time now, see
 * md_t->deadline: after the deadline, unless the remaining time, as
 * estimated from the time so far, is below deadline[1] */
static int apply_given_up(
        const md_t * const md,
        const double now,
        const double start,
        const double done
        )
{
    if (md->deadline == NULL || md->deadline[0] <= 0
            || now <= md->deadline[0]) {
        return 0;
    }
    if (done <= 0) {
        return 1;
    }
    return (now - start) * (1 - done) / done >= md->deadline[1];
}

/* the classic replay for p < 2^32: byte-identical to the behavior
 * before the core was parameterized by the coefficient width */
SG_REPLAY(sig_f4_replay_32, cf32_t,
        SG_INPUT_COEFFS_32, SG_TAKE_32, SG_SUBROW_32,
        SG_INV_32, SG_MUL_INV_32, SG_INV_PRE_32,
        SG_BIG_SMALL_DECL_32, 1, SG_BAND_TAKE_32)

/* the wide replay for primes in [2^32, 2^62): coefficients in cf_64
 * arrays, dense rows in the redundant [0, 2p) representation, the Shoup
 * product with a per-application pre-inverse of the row multiplier,
 * monicization with 128-bit products */
SG_REPLAY(sig_f4_replay_64, cf64_t,
        SG_INPUT_COEFFS_64, SG_TAKE_64, SG_SUBROW_64,
        SG_INV_64, SG_MUL_INV_64, SG_INV_PRE_64,
        SG_BIG_SMALL_DECL_64, 1, SG_BAND_TAKE_64)

/* dispatches on the field width of the run */
static int sig_f4_replay(
        bs_t *bs,
        md_t *md,
        const sg_rec_t * const rec
        )
{
    if (md->ff_bits == 64) {
        return sig_f4_replay_64(bs, md, rec);
    }
    return sig_f4_replay_32(bs, md, rec);
}
