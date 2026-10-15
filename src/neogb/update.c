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
    for (i = 0; i < bl; ++i) {
        pp[i].lcm   =  get_lcm(bs->hm[i][OFFSET], nch, bht, bht);
        pp[i].gen1  = i;
        pp[i].gen2  = bl;
        if (bs->red[i] != 0) {
            pp[i].deg   =   -1;
        } else {
            if (prime_monomials(bs->hm[pp[i].gen1][OFFSET], bs->hm[pp[i].gen2][OFFSET], bht)) {
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
    private(i, j,  l)
    for (i = 0; i < pl; ++i) {
        j = ps[i].gen1;
        l = ps[i].gen2;
        if (pp[j].lcm != ps[i].lcm && pp[l].lcm != ps[i].lcm
                && pp[j].deg <= ps[i].deg && pp[l].deg <= ps[i].deg
                && check_monomial_division(ps[i].lcm, nch, bht)) {
            ps[i].deg   =   -1;
        }
    }
    /* sort new pairs by increasing lcm, earlier polys coming first */
    sort_r(pp, (unsigned long)bl, sizeof(spair_t), spair_cmp_update, bht);

    /* Gebauer-Moeller: remove multiples of new spairs */
    for (i = pl+1; i < nl; ++i) {
        if (ps[i].deg < 0) {
            continue;
        }
        for (j = pl; j < i; ++j) {
            if (ps[j].deg == -1) {
                continue;
            }
            if (check_monomial_division(ps[i].lcm, ps[j].lcm, bht)) {
                ps[i].deg   =   -1;
                break;
            }
        }
    }

    /* remove useless pairs from pairset */
    j = 0;
    /* old pairs */
    for (i = 0; i < nl; ++i) {
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

    #ifdef _OPENMP
    const int nthrds = st->nthrds;
    if (npivs > 1 && nthrds > 1) {
        const len_t ld0   = bs->ld;
        const len_t pl0   = ps->ld;
        const bl_t lml    = bs->lml;
        const bl_t * const lmps = bs->lmps;
        const deg_t mltdeg0 = bs->mltdeg;

        /* pre-size bht so that no table growth is needed inside the
         * parallel region (the factor two covers slots lost to
         * concurrent insertion races) */
        while (bht->esz - bht->eld < 2ULL * (hl_t)np + 1) {
            enlarge_hash_table(bht);
        }

        hm_t *nch  = (hm_t *)malloc((unsigned long)npivs * sizeof(hm_t));
        deg_t *ndeg = (deg_t *)malloc((unsigned long)npivs * sizeof(deg_t));
        deg_t *emax = (deg_t *)malloc((unsigned long)npivs * sizeof(deg_t));
        len_t *off  = (len_t *)malloc((unsigned long)npivs * sizeof(len_t));
        len_t *bll  = (len_t *)malloc((unsigned long)npivs * sizeof(len_t));

        deg_t mx   = mltdeg0;
        len_t offk = pl0;
        for (i = 0; i < npivs; ++i) {
            nch[i]  = bs->hm[ld0+i][OFFSET];
            ndeg[i] = bs->hm[ld0+i][DEG];
            off[i]  = offk;
            bll[i]  = ld0 + i;
            offk   += bll[i];
            if (ndeg[i] > mx) {
                mx = ndeg[i];
            }
            emax[i] = mx;
        }

        /* kmark[m] = first block whose redundancy loop would mark the
         * element lmps[m] as redundant, or a sentinel if none does */
        deg_t *kmark = (deg_t *)malloc((unsigned long)lml * sizeof(deg_t));
        for (i = 0; i < (len_t)lml; ++i) {
            kmark[i] = 0x7fffffff;
        }
        int overflow = 0;

        #pragma omp parallel for num_threads(nthrds) schedule(dynamic)
        for (i = 0; i < npivs; ++i) {
            if (emax[i] > ndeg[i]) {
                const deg_t dd = ndeg[i] - bht->hd[nch[i]].deg;
                for (len_t m = 0; m < (len_t)lml; ++m) {
                    if (check_monomial_division(bs->hm[lmps[m]][OFFSET], nch[i], bht)
                            && bs->hm[lmps[m]][DEG]-bht->hd[bs->hm[lmps[m]][OFFSET]].deg >= dd) {
                        deg_t old = __atomic_load_n(&kmark[m], __ATOMIC_RELAXED);
                        while ((deg_t)i < old
                                && !__atomic_compare_exchange_n(&kmark[m], &old, (deg_t)i,
                                                                0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                        }
                    }
                }
            }
        }

        /* same information at element level, for elements older than
         * the new basis elements */
        deg_t *kmark_el = (deg_t *)malloc((unsigned long)ld0 * sizeof(deg_t));
        for (i = 0; i < ld0; ++i) {
            kmark_el[i] = 0x7fffffff;
        }
        for (i = 0; i < (len_t)lml; ++i) {
            if (kmark[i] < kmark_el[lmps[i]]) {
                kmark_el[lmps[i]] = kmark[i];
            }
        }

        /* creation-time degree and lcm of each new pair, indexed
         * relatively to the start of the pairset before this update
         * (the blocks are sorted later, so the creation-time lcm of the
         * pair (j1, ld0+k) is not found at its sorted position) */
        deg_t *cdeg = (deg_t *)malloc((unsigned long)np * sizeof(deg_t));
        hi_t  *clcm = (hi_t *)malloc((unsigned long)np * sizeof(hi_t));

        #pragma omp parallel for num_threads(nthrds) schedule(dynamic)
        for (i = 0; i < npivs; ++i) {
            spair_t *pp = ps->p + off[i];
            int cr = 0;
            for (int j = 0; j < (int)bll[i]; ++j) {
                pp[j].lcm  = get_lcm_thread_safe(bs->hm[j][OFFSET], nch[i], bht, bht, &cr);
                pp[j].gen1 = j;
                pp[j].gen2 = ld0 + i;
                if (bs->red[j] != 0 || (j < ld0 && kmark_el[j] < (deg_t)i)) {
                    pp[j].deg = -1;
                } else if (prime_monomials(bs->hm[j][OFFSET], nch[i], bht)) {
                    pp[j].deg = -2;
                } else if (st->nev == 0) {
                    pp[j].deg = bht->hd[pp[j].lcm].deg;
                } else {
                    deg_t deg1 = bht->hd[pp[j].lcm].deg - bht->hd[bs->hm[j][OFFSET]].deg + bs->hm[j][DEG];
                    deg_t deg2 = bht->hd[pp[j].lcm].deg - bht->hd[nch[i]].deg + bs->hm[ld0+i][DEG];
                    pp[j].deg  = deg1 > deg2 ? deg1 : deg2;
                }
                cdeg[off[i]-pl0+j] = pp[j].deg;
                clcm[off[i]-pl0+j] = pp[j].lcm;
                if (cr < 0) {
                    __atomic_store_n(&overflow, 1, __ATOMIC_RELAXED);
                }
            }
            sort_r(pp, (unsigned long)bll[i], sizeof(spair_t), spair_cmp_update, bht);
            /* local Gebauer-Moeller within this block's new pairs */
            for (int j = 1; j < (int)bll[i]; ++j) {
                if (pp[j].deg < 0) {
                    continue;
                }
                for (int l = 0; l < j; ++l) {
                    if (pp[l].deg == -1) {
                        continue;
                    }
                    if (check_monomial_division(pp[j].lcm, pp[l].lcm, bht)) {
                        pp[j].deg = -1;
                        break;
                    }
                }
            }
        }

        if (overflow) {
            /* pre-sizing was insufficient: discard and run the serial update */
            ps->ld = pl0;
            for (i = 0; i < npivs; ++i) {
                insert_and_update_spairs(ps, bs, bht, st);
            }
        } else {
            /* Gebauer-Moeller against new pairs of later blocks; every
             * pair is handled by exactly one thread */
            len_t *k0of = (len_t *)malloc((unsigned long)np * sizeof(len_t));
            for (i = 0; i < npivs; ++i) {
                for (len_t r = 0; r < bll[i]; ++r) {
                    k0of[off[i]-pl0+r] = i;
                }
            }

            #pragma omp parallel
            {
                #pragma omp for schedule(dynamic)
                for (i = 0; i < pl0; ++i) {
                    spair_t *p = ps->p + i;
                    const len_t j1 = p->gen1;
                    const len_t l1 = p->gen2;
                    for (len_t k = 0; k < npivs; ++k) {
                        if (cdeg[off[k]-pl0+j1] <= p->deg
                                && cdeg[off[k]-pl0+l1] <= p->deg
                                && clcm[off[k]-pl0+j1] != p->lcm
                                && clcm[off[k]-pl0+l1] != p->lcm
                                && check_monomial_division(p->lcm, nch[k], bht)) {
                            p->deg = -1;
                            break;
                        }
                    }
                }
                #pragma omp for schedule(dynamic)
                for (i = 0; i < np; ++i) {
                    const len_t k0 = k0of[i];
                    spair_t *p = ps->p + off[k0] + (i - (off[k0]-pl0));
                    if (p->deg < 0) {
                        continue;
                    }
                    for (len_t k = k0+1; k < npivs; ++k) {
                        if (cdeg[off[k]-pl0+p->gen1] <= p->deg
                                && cdeg[off[k]-pl0+p->gen2] <= p->deg
                                && clcm[off[k]-pl0+p->gen1] != p->lcm
                                && clcm[off[k]-pl0+p->gen2] != p->lcm
                                && check_monomial_division(p->lcm, nch[k], bht)) {
                            p->deg = -1;
                            break;
                        }
                    }
                }
            }

            /* mark redundant elements, as the serial version would have */
            for (i = 0; i < (len_t)lml; ++i) {
                if (kmark[i] != 0x7fffffff && bs->red[lmps[i]] == 0) {
                    bs->red[lmps[i]] = 1;
                    st->num_redundant++;
                }
            }
            bs->mltdeg = mx;

            /* single compaction of the whole pairset */
            len_t jj = 0;
            for (i = 0; i < pl0+np; ++i) {
                if (ps->p[i].deg >= 0) {
                    ps->p[jj++] = ps->p[i];
                }
            }
            ps->ld = jj;
            st->num_gb_crit += (int64_t)(pl0+np) - (int64_t)ps->ld;
            bs->ld += npivs;

            free(k0of);
        }
        free(nch);
        free(ndeg);
        free(emax);
        free(off);
        free(bll);
        free(kmark);
        free(kmark_el);
        free(cdeg);
        free(clcm);
    } else
    #endif
    {
        for (i = 0; i < npivs; ++i) {
            insert_and_update_spairs(ps, bs, bht, st);
        }
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
    st->update_ctime  +=  ct1 - ct0;
    st->update_rtime  +=  rt1 - rt0;
}

static int32_t update(
        bs_t *bs,
        md_t *md
        )
{
        update_basis_f4(md->ps, bs, bs->ht, md, md->np);
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
