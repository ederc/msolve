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


#include "update.h" 

ps_t *initialize_pairset(
        void
        )
{
    ps_t *ps  = (ps_t *)malloc(sizeof(ps_t));
    ps->ld  = 0;
    ps->sz  = 192;
    ps->p = (spair_t *)calloc((unsigned long)ps->sz, sizeof(spair_t));
    return ps;
}

static inline void check_enlarge_pairset(
        ps_t *ps,
        len_t added
        )
{
    if (ps->ld+added >= ps->sz) {
        ps->sz  = ps->sz*2 > ps->ld+added ? ps->sz*2 : ps->ld+added;
        ps->p   = realloc(ps->p, (unsigned long)ps->sz * sizeof(spair_t));
        memset(ps->p+ps->ld, 0,
                (unsigned long)(ps->sz-ps->ld) * sizeof(spair_t));
    }
}

void free_pairset(
        ps_t **psp
        )
{
    ps_t *ps  = *psp;
    if (ps->p) {
        free(ps->p);
        ps->p   = NULL;
        ps->ld  = 0;
        ps->sz  = 0;
    }
    free(ps);
    ps  = NULL;
    *psp  = ps;
}

static spair_t *generate_new_spair_list(
        bs_t *bs,
        const len_t idx,
        const md_t *md
        )
{
    const len_t bld  = bs->ld + idx;
    const hm_t nch   = bs->hm[bld][OFFSET];
    const deg_t ndeg = bs->hm[bld][DEG];

    bs->mltdeg  = bs->mltdeg > ndeg ?
        bs->mltdeg : ndeg;

    len_t isprime    = 0;
    deg_t deg1, deg2 = 0;
    ht_t *ht         = bs->ht;
    spair_t *p       = (spair_t *)malloc((unsigned long)bld * sizeof(spair_t));

    while (ht->esz - ht->eld < bld) {
        enlarge_hash_table(ht);
    }

    for (len_t i = 0; i < bld; ++i) {
        p[i].lcm   =  get_lcm_with_primality(bs->hm[i][OFFSET], nch, ht, ht, &isprime);
        p[i].gen1  = i;
        p[i].gen2  = bld;
        if (isprime == 1) {
            p[i].deg = -1;
        } else {
            if (bs->red[i] != 0) {
                p[i].deg   =   -2;
            } else {
                /* compute total degree of pair, not trivial if block order is chosen */
                if (md->nev == 0) {
                    p[i].deg = ht->hd[p[i].lcm].deg;
                } else {
                    deg1  = ht->hd[p[i].lcm].deg - ht->hd[bs->hm[i][OFFSET]].deg + bs->hm[i][DEG];
                    deg2  = ht->hd[p[i].lcm].deg - ht->hd[nch].deg + bs->hm[bld+1][DEG];
                    p[i].deg = deg1 > deg2 ? deg1 : deg2;
                }
            }
        }
    }
    /* printf("new pairs generated\n"); */
    /* for (len_t i = 0; i < bld; ++i) { */
    /*     printf("p[%d] -> [%d,%d] -> deg %d -> ", i, p[i].gen1, p[i].gen2, p[i].deg); */
    /*     for (int ii = 0; ii < bs->ht->evl; ++ii) { */
    /*         printf("%d ", bs->ht->ev[p[i].lcm][ii]); */
    /*     } */
    /*     printf("\n"); */
    /* } */
    /* printf("###\n"); */
    return p;
}

static void remove_old_spairs_via_gebauer_moeller(
        ps_t *psl,
        spair_t **sp,
        const len_t idx,
        const bs_t *bs
        )
{
    const len_t pl     = psl->ld;
    spair_t *ops       = psl->p;
    const spair_t *nps = sp[idx];
    const hm_t nch     = bs->hm[bs->ld+idx][OFFSET];
    const ht_t * ht    = bs->ht;

    /* printf("nch[%d] = ", idx); */
    /* for (len_t i = 0; i < ht->evl; ++i) { */
    /*     printf("%d ", ht->ev[nch][i]); */
    /* } */
    /* printf("\n"); */
    len_t j, l;

    for (len_t i = 0; i < pl; ++i) {
        if (ops[i].deg >= 0) {
            j = ops[i].gen1;
            l = ops[i].gen2;
            if (nps[j].lcm != ops[i].lcm && nps[l].lcm != ops[i].lcm
                    && nps[j].deg <= ops[i].deg && nps[l].deg <= ops[i].deg
                    && check_monomial_division(ops[i].lcm, nch, ht)) {
                ops[i].deg = -2;
            }
        }
    }
    /* printf("removed?\n"); */
    /* for (int i = 0; i < psl->ld; ++i) { */
    /*     printf("ps[%d] -> [%d,%d] -> deg %d -> ", i, psl->p[i].gen1, psl->p[i].gen2, psl->p[i].deg); */
    /*     for (int ii = 0; ii < bs->ht->evl; ++ii) { */
    /*         printf("%d ", bs->ht->ev[psl->p[i].lcm][ii]); */
    /*     } */
    /*     printf("\n"); */
    /* } */
    /* printf("---\n"); */

}

static void remove_new_smaller_index_spairs_via_gebauer_moeller(
        spair_t **sp,
        const len_t idx,
        const len_t npivs,
        const bs_t *bs
        )
{
    spair_t *ops   = sp[idx];
    ht_t *ht       = bs->ht;
    const len_t pl = bs->ld + idx;

    len_t j;
    for (len_t k = idx+1; k < npivs; ++k) {
        const hm_t nch     = bs->hm[bs->ld+k][OFFSET];
        const spair_t *nps = sp[k];
        /* for the second generator we always have index pl, thus we */
        /* can precompute the corresponding lcm value */
        const hm_t lcm_idx = nps[pl].lcm;
        for (len_t i = 0; i < pl; ++i) {
            if (ops[i].deg >= 0) {
                j = ops[i].gen1;
                if (lcm_idx != ops[i].lcm && nps[j].lcm != ops[i].lcm
                        && check_monomial_division(ops[i].lcm, nch, ht)) {
                    ops[i].deg = -2;
                }
            }
        }
    }
}

static len_t remove_new_same_index_spairs_via_gebauer_moeller(
        spair_t **sp,
        bs_t *bs,
        md_t *md,
        const len_t idx,
        const len_t npivs
        )
{
    int32_t i, j;

    const len_t pl = bs->ld + idx;
    ht_t *ht   = bs->ht;
    spair_t *p = sp[idx];
    /* sort new pairs by increasing lcm, earlier polys coming first */
    sort_r(p, (unsigned long)pl, sizeof(spair_t), spair_cmp_update, ht);

    i = 0;
    while (i < pl && p[i].deg == -2) {
        ++i;
    }
    const int rp = i;
    while (i < pl && p[i].deg == -1) {
        ++i;
    }
    const int fp = i;

    /* Gebauer-Moeller: remove real multiples of new spairs */
    for (i = fp; i < pl; ++i) {
        for (j = fp; j < i; ++j) {
            if (p[j].deg >= 0
                    && p[i].deg > p[j].deg
                    && check_monomial_division(p[i].lcm, p[j].lcm, ht)) {
                p[i].deg = -2;
                break;
            }
        }
    }

    /* Gebauer-Moeller: remove same lcm spairs from the new ones */
    for (i = rp; i < fp; ++i) {
        /* try to remove all others if product criterion applies */
        for (j = fp; j < pl; ++j) {
            if (p[j].lcm == p[i].lcm) {
                p[j].deg = -2;
            }
        }
        /* try to eliminate this spair with earlier ones */
    }
    for (i = fp; i < pl; ++i) {
        if (p[i].deg == -2) {
            continue;
        }
        for (j = i-1; j >= fp; --j) {
            if (p[i].lcm == p[j].lcm) {
                p[i].deg = -2;
                break;
            }
        }
    }
    j = 0;
    for (i = fp; i < pl; ++i) {
        if (p[i].deg < 0) {
            continue;
        }
        p[j++] = p[i];
    }
    const len_t npl = j;

    const bl_t lml          = bs->lml;
    const bl_t * const lmps = bs->lmps;

    /* mark redundant elements in basis */
    const hm_t nch   = bs->hm[bs->ld+idx][OFFSET];
    const deg_t ndeg = bs->hm[bs->ld+idx][DEG];
    deg_t dd = ndeg - ht->hd[nch].deg;
    if (bs->mltdeg > ndeg) {
        for (i = 0; i < lml; ++i) {
            hm_t lm = bs->hm[lmps[i]][OFFSET];
            if (bs->red[lmps[i]] == 0
                    && check_monomial_division(lm, nch, ht)
                    && bs->hm[lmps[i]][DEG]-ht->hd[lm].deg >= dd) {
                bs->red[lmps[i]]  = 1;
#pragma omp critical
                md->num_redundant++;
            }
        }
    }

#pragma omp critical
    md->num_gb_crit +=  pl - npl;

    return npl;
}

static len_t get_old_spair_list_length(
        ps_t *ps
        )
{
    len_t i, j;
    const len_t pld = ps->ld;
    spair_t *p = ps->p;

    j = 0;
    for (i = 0; i < pld; ++i) {
        if (p[i].deg > 0) {
            p[j++] = p[i];
        }
    }
    ps->ld = j;

    return ps->ld;
}


static void insert_and_update_spairs(
        ps_t *psl,
        bs_t *bs,
        ht_t *bht,
        md_t *st
        )
{
    int i, j, l;
    deg_t deg1, deg2;

    spair_t *ps = psl->p;

    double rt = realtime();
#ifdef _OPENMP
    const int nthrds = st->nthrds;
#endif

    const int pl  = psl->ld;
    const int bl  = bs->ld;

    const hm_t nch = bs->hm[bl][OFFSET];

    deg_t ndeg  = bs->hm[bl][DEG];

    bs->mltdeg  = bs->mltdeg > ndeg ?
        bs->mltdeg : ndeg;

    spair_t *pp = ps+pl;

    while (bht->esz - bht->eld < bl) {
        enlarge_hash_table(bht);
    }
#if PARALLEL_HASHING
#pragma omp parallel for num_threads(nthrds) \
    private(i) schedule(dynamic, 50)
#endif
    for (i = 0; i < bl; ++i) {
        len_t isprime = 0;
        pp[i].lcm   =  get_lcm_with_primality(bs->hm[i][OFFSET], nch, bht, bht, &isprime);
        pp[i].gen1  = i;
        pp[i].gen2  = bl;
        if (isprime == 1) {
            pp[i].deg = -1;
        } else {
            if (bs->red[i] != 0) {
                pp[i].deg   =   -2;
            } else {
                /* compute total degree of pair, not trivial if block order is chosen */
                if (st->nev == 0) {
                    pp[i].deg = bht->hd[pp[i].lcm].deg;
                } else {
                    deg1  = bht->hd[pp[i].lcm].deg - bht->hd[bs->hm[i][OFFSET]].deg + bs->hm[i][DEG];
                    deg2  = bht->hd[pp[i].lcm].deg - bht->hd[nch].deg + bs->hm[bl][DEG];
                    pp[i].deg = deg1 > deg2 ? deg1 : deg2;
                }
            }
        }
    }

    len_t nl  = pl+bl;
    /* Gebauer-Moeller: check old pairs first */
    /* note: old pairs are sorted by the given spair order */
#pragma omp parallel for num_threads(nthrds) \
    private(i, j,  l) schedule(dynamic, 50)
    for (i = 0; i < pl; ++i) {
        j = ps[i].gen1;
        l = ps[i].gen2;
        if (pp[j].lcm != ps[i].lcm && pp[l].lcm != ps[i].lcm
                /* && pp[j].deg <= ps[i].deg && pp[l].deg <= ps[i].deg */
                && check_monomial_division(ps[i].lcm, nch, bht)) {
            ps[i].deg   =   -2;
        }
    }
    /* sort new pairs by increasing lcm, earlier polys coming first */
    sort_r(pp, (unsigned long)bl, sizeof(spair_t), spair_cmp_update, bht);

    i = 0;
    while (pp[i].deg == -2) {
        ++i;
    }
    const int sp = pl+i;
    while (pp[i].deg == -1) {
        ++i;
    }
    const int fp = pl+i;

    /* Gebauer-Moeller: remove real multiples of new spairs */
    for (i = fp; i < nl; ++i) {
        for (j = fp; j < i; ++j) {
            /* if (i == j || ps[j].deg == -1) { */
            /*     continue; */
            /* } */
            if (ps[j].deg >= 0
                    && ps[i].deg > ps[j].deg
                    && check_monomial_division(ps[i].lcm, ps[j].lcm, bht)) {
                ps[i].deg   =   -2;
                break;
            }
        }
    }


    /* Gebauer-Moeller: remove same lcm spairs from the new ones */
    for (i = sp; i < fp; ++i) {
        /* try to remove all others if product criterion applies */
        for (j = fp; j < nl; ++j) {
            if (ps[j].lcm == ps[i].lcm) {
                ps[j].deg   =   -2;
            }
        }
        /* try to eliminate this spair with earlier ones */
    }
    for (i = fp; i < nl; ++i) {
        if (ps[i].deg == -2) {
            continue;
        }
        for (j = i-1; j >= fp; --j) {
            if (ps[i].lcm == ps[j].lcm) {
                ps[i].deg   =   -2;
                break;
            }
        }
    }
    /* for (i = 0; i<nl; ++i) { */
    /*     printf("deg[%d] = %d\n", i, ps[i].deg); */
    /* } */

    /* remove useless pairs from pairset */
    j = 0;
    for (i = 0; i < pl; ++i) {
        if (ps[i].deg < 0) {
            continue;
        }
        ps[j++] = ps[i];
    }
    for (i = fp; i < nl; ++i) {
        if (ps[i].deg < 0) {
            continue;
        }
        ps[j++] = ps[i];
    }

    psl->ld =   j;
    const bl_t lml          = bs->lml;
    const bl_t * const lmps = bs->lmps;

    /* mark redundant elements in basis */
    deg_t dd = ndeg - bht->hd[nch].deg;
    if (bs->mltdeg > ndeg) {
#if PARALLEL_HASHING
#pragma omp parallel for num_threads(nthrds) \
    private(i) schedule(dynamic, 50)
#endif
        for (i = 0; i < lml; ++i) {
            hm_t lm = bs->hm[lmps[i]][OFFSET];
            if (bs->red[lmps[i]] == 0
                    && check_monomial_division(lm, nch, bht)
                    && bs->hm[lmps[i]][DEG]-bht->hd[lm].deg >= dd) {
                bs->red[lmps[i]]  = 1;
                st->num_redundant++;
            }
        }
    }

    st->num_gb_crit +=  nl - psl->ld;

    bs->ld++;

}

static void update_lm(
        bs_t *bs,
        const ht_t * const bht,
        md_t *st
        )
{
    len_t i, j, k, l;

    const bl_t * const lmps = bs->lmps;

    j = bs->lo;
nextj:
    for (; j < bs->ld; ++j) {
        k = 0;
        for (l = bs->lo; l < j; ++l) {
            if (bs->red[l]) {
                continue;
            }
            if (check_monomial_division(bs->hm[j][OFFSET], bs->hm[l][OFFSET], bht)) {
                bs->red[j]  = 1;
                st->num_redundant++;
                j++;
                goto nextj;
            }
        }
        for (i = 0; i < bs->lml; ++i) {
            if (bs->red[lmps[i]] == 0
                    && check_monomial_division(bs->hm[lmps[i]][OFFSET], bs->hm[j][OFFSET], bht)) {
                bs->red[lmps[i]]  = 1;
                st->num_redundant++;
            }
        }
        const sdm_t *lms  = bs->lm;
        for (i = 0; i < bs->lml; ++i) {
            if (bs->red[lmps[i]] == 0) {
                bs->lm[k]   = lms[i];
                bs->lmps[k] = lmps[i];
                k++;
            }
        }
        bs->lml = k;
        k = bs->lml;
        if (bs->red[j] == 0) {
            bs->lm[k]   = bht->hd[bs->hm[j][OFFSET]].sdm;
            bs->lmps[k] = j;
            k++;
        }
        bs->lml = k;
    }
    bs->lo  = bs->ld;

    st->num_redundant_old = st->num_redundant;
}

static void update_basis_f4_parallel(
        bs_t *bs,
        md_t *md
        )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    ps_t *ps          = md->ps;
    const len_t npivs = md->np;
    len_t i, k;
    len_t ctr       = 0;
    const len_t bld = bs->ld + npivs;

    if (npivs == 0) {
        return;
    }
    /* if the hash table gets too big, try to reset it */
    /* in order to remove now useless monomial data */
    if (bs->ht->esz >= pow(2,25)) {
        if (ctr == 2000) {
            printf("->%d", ctr);
            reset_hash_table_during_update(bs->ht, bs, md->ps, md, bld);
            ctr = 0;
        } else {
            ++ctr;
        }
    }

    spair_t **sp = (spair_t **)malloc((unsigned long)npivs * sizeof(spair_t *));
    len_t *lens  = (len_t *)malloc((unsigned long)npivs * sizeof(len_t));

    /* printf("before update\n"); */
    /* for (i = 0; i < ps->ld; ++i) { */
    /*     printf("ps[%d] -> [%d,%d] -> deg %d -> ", i, ps->p[i].gen1, ps->p[i].gen2, ps->p[i].deg); */
    /*     for (int ii = 0; ii < bs->ht->evl; ++ii) { */
    /*         printf("%d ", bs->ht->ev[ps->p[i].lcm][ii]); */
    /*     } */
    /*     printf("\n"); */
    /* } */
    /* printf(",,,\n"); */

    /* generate new spairs in parallel */
#pragma omp parallel for num_threads(1)
    for (i = 0; i < npivs; ++i) {
        sp[i] = generate_new_spair_list(bs, i, md);
    }

    /* remove old spairs via Gebauer-Möller */
#pragma omp parallel for num_threads(md->nthrds)
    for (i = 0; i < npivs; ++i) {
        remove_old_spairs_via_gebauer_moeller(ps, sp, i, bs);
    }

    /* remove new, smaller index spairs via Gebauer-Möller */
#pragma omp parallel for num_threads(md->nthrds)
    for (i = 0; i < npivs-1; ++i) {
        remove_new_smaller_index_spairs_via_gebauer_moeller(sp, i, npivs, bs);
    }

    /* remove new, same index spairs via Gebauer-Möller */
#pragma omp parallel for num_threads(md->nthrds)
    for (i = 0; i < npivs; ++i) {
        lens[i] = remove_new_same_index_spairs_via_gebauer_moeller(sp, bs, md, i, npivs);
    }

    unsigned long sum = 0;
    for (i = 0; i < npivs; ++i) {
        sum += lens[i];
    }
    len_t olen = get_old_spair_list_length(ps);

    sum += olen;

    ps->p  = realloc(ps->p, sum * sizeof(spair_t));
    ps->ld = sum;
    sum = olen;
    for (i = 0; i < npivs; ++i) {
        memmove(ps->p + sum, sp[i], (unsigned long)lens[i] * sizeof(spair_t));
        sum += lens[i];
    }

    /* printf("update done\n"); */
    /* for (i = 0; i < ps->ld; ++i) { */
    /*     printf("ps[%d] -> [%d,%d] -> deg %d -> ", i, ps->p[i].gen1, ps->p[i].gen2, ps->p[i].deg); */
    /*     for (int ii = 0; ii < bs->ht->evl; ++ii) { */
    /*         printf("%d ", bs->ht->ev[ps->p[i].lcm][ii]); */
    /*     } */
    /*     printf("\n"); */
    /* } */
    /* printf("...\n"); */


    const bl_t lml          = bs->lml;
    const bl_t * const lmps = bs->lmps;
    const ht_t *ht          = bs->ht;

    bs->ld  += npivs;

    k = 0;
    if (md->mo == 0 && md->num_redundant_old < md->num_redundant) {
        const sdm_t *lms  = bs->lm;
        for (i = 0; i < lml; ++i) {
            if (bs->red[lmps[i]] == 0) {
                bs->lm[k]   = lms[i];
                bs->lmps[k] = lmps[i];
                k++;
            }
        }
        bs->lml = k;
    }
    k = bs->lml;
    for (i = bs->lo; i < bs->ld; ++i) {
        if (bs->red[i] == 0) {
            bs->lm[k]   = ht->hd[bs->hm[i][OFFSET]].sdm;
            bs->lmps[k] = i;
            k++;
        }
    }
    bs->lml =  k;
    bs->lo  =  bs->ld;

    md->num_redundant_old = md->num_redundant;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    printf("[[[%.2f]]]", rt1-rt0);
    md->update_ctime  +=  ct1 - ct0;
    md->update_rtime  +=  rt1 - rt0;
}

static void update_basis_f4(
        ps_t *ps,
        bs_t *bs,
        ht_t *bht,
        md_t *st,
        const len_t npivs
        )
{
    len_t i;


    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* compute number of new pairs we need to handle at most */
    len_t np  = bs->ld * npivs;
    for (i = 1; i < npivs; ++i) {
        np  = np + i;
    }
    check_enlarge_pairset(ps, np);

    const len_t bld = bs->ld + npivs;
    len_t ctr = 0;
    for (i = 0; i < npivs; ++i) {
        if (bs->ht->esz >= pow(2,25)) {
            if (ctr == 2000) {
                printf("->%d", i);
                reset_hash_table_during_update(bs->ht, bs, st->ps, st, bld);
                ctr = 0;
            } else {
                ++ctr;
            }
        }

        insert_and_update_spairs(ps, bs, bht, st);
    }

    const bl_t lml          = bs->lml;
    const bl_t * const lmps = bs->lmps;

    len_t k = 0;

    /* Check new elements on redundancy:
     * Only elements coming from the same matrix are possible leading
     * monomial divisors, thus we only check down to bs->lo */
/* #pragma omp parallel for num_threads(st->nthrds)
    for (int l = bs->lo; l < bs->ld; ++l) {
        hm_t lm  = bs->hm[l][OFFSET];
        deg_t dd = bs->hm[l][DEG] - bht->hd[lm].deg;
        for (int m = bs->lo; m < l; ++m) {
            if (check_monomial_division(lm, bs->hm[m][OFFSET], bht) == 1
                && dd >= (bs->hm[m][DEG] - bht->hd[bs->hm[m][OFFSET]].deg)) {
                bs->red[l]  =   1;
                st->num_redundant++;
                break;
            }
        }
    } */
    if (st->mo == 0 && st->num_redundant_old < st->num_redundant) {
        const sdm_t *lms  = bs->lm;
        for (i = 0; i < lml; ++i) {
            if (bs->red[lmps[i]] == 0) {
                bs->lm[k]   = lms[i];
                bs->lmps[k] = lmps[i];
                k++;
            }
        }
        bs->lml = k;
    }
    k = bs->lml;
    for (i = bs->lo; i < bs->ld; ++i) {
        if (bs->red[i] == 0) {
            bs->lm[k]   = bht->hd[bs->hm[i][OFFSET]].sdm;
            bs->lmps[k] = i;
            k++;
        }
    }
    bs->lml = k;
    bs->lo  = bs->ld;

    st->num_redundant_old = st->num_redundant;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    printf("[[[%.2f]]]", rt1-rt0);
    st->update_ctime  +=  ct1 - ct0;
    st->update_rtime  +=  rt1 - rt0;
}

static int32_t update(
        bs_t *bs,
        md_t *md
        )
{
        update_basis_f4_parallel(bs, md);
        if (bs->constant) {
            return 1;
        } else {
            return md->ps->ld == 0;
        }
}

/* not needed right now, maybe in a later iteration of sba implementations */
#if 0
static void update_basis_sba_schreyer(
        ps_t *ps,
        bs_t *bs,
        ht_t *bht,
        ht_t *uht,
        md_t *st,
        const len_t npivs
        )
{
    len_t i;

    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* compute number of new pairs we need to handle at most */
    len_t np  = bs->ld * npivs;
    for (i = 1; i < npivs; ++i) {
        np  = np + i;
    }
    check_enlarge_pairset(ps, np);

    for (i = 0; i < npivs; ++i) {
        insert_and_update_spairs(ps, bs, bht, uht, st);
    }

    const bl_t lml          = bs->lml;
    const bl_t * const lmps = bs->lmps;

    len_t k = 0;
    if (st->mo == 0 && st->num_redundant_old < st->num_redundant) {
        const sdm_t *lms  = bs->lm;
        for (i = 0; i < lml; ++i) {
            if (bs->red[lmps[i]] == 0) {
                bs->lm[k]   = lms[i];
                bs->lmps[k] = lmps[i];
                k++;
            }
        }
        bs->lml = k;
    }
    k = bs->lml;
    for (i = bs->lo; i < bs->ld; ++i) {
        if (bs->red[i] == 0) {
            bs->lm[k]   = bht->hd[bs->hm[i][OFFSET]].sdm;
            j>lmps[k] = i;
            k++;
        }
    }
    bs->lml = k;
    bs->lo  = bs->ld;

    st->num_redundant_old = st->num_redundant;

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->update_ctime  +=  ct1 - ct0;
    st->update_rtime  +=  rt1 - rt0;
}
#endif
