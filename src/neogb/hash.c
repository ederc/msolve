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


#include "hash.h"
#include "../msolve/streams.h"

/* we have three different hash tables:
 * 1. one hash table for elements in the basis (bht)
 * 2. one hash table for the spairs during the update process (uht)
 * 3. one hash table for the multiplied elements during symbolic
 *    preprocessing (sht) */

/* The idea of the structure of the hash table is taken from an
 * implementation by Roman Pearce and Michael Monagan in Maple. */

static val_t pseudo_random_number_generator(
    uint32_t *seed
    )
{
    uint32_t rseed  = *seed;
    rseed ^=  (rseed << 13);
    rseed ^=  (rseed >> 17);
    rseed ^=  (rseed << 5);
    *seed =   rseed;
    return (val_t)rseed;
}

ht_t *initialize_basis_hash_table(
    md_t *st
    )
{
    len_t i;
    hl_t j;

    const len_t nv  = st->nvars;

    ht_t *ht  = (ht_t *)malloc(sizeof(ht_t));
    ht->nv    = nv;
    ht->mo    = st->mo;
    /* generate map */
    ht->bpv = (len_t)((CHAR_BIT * sizeof(sdm_t)) / (unsigned long)nv);
    if (ht->bpv == 0) {
        ht->bpv++;
    }
    ht->ndv = (unsigned long)nv < (CHAR_BIT * sizeof(sdm_t)) ?
        nv : (len_t)((CHAR_BIT * sizeof(sdm_t)));
    ht->dv  = (len_t *)calloc((unsigned long)ht->ndv, sizeof(len_t));

    ht->hsz   = (hl_t)pow(2, st->init_hts);
    ht->esz   = ht->hsz / 2;
    ht->hmap  = calloc(ht->hsz, sizeof(hi_t));

    if (st->nev == 0) {
        ht->evl = nv + 1; /* store also degree at first position */
        ht->ebl = 0;
        for (i = 1; i <= ht->ndv; ++i) {
            ht->dv[i-1] = i;
        }
    } else {
        ht->evl = nv + 2; /* store also degrees for both blocks, see
                           * data.h for more on exponent vector structure */
        ht->ebl = st->nev + 1; /* store also degree at first position */
        if (st->nev >= ht->ndv) {
            for (i = 1; i <= ht->ndv; ++i) {
                ht->dv[i-1] = i;
            }
        } else {
            len_t ctr = 0;
            for (i = 1; i <= st->nev; ++i) {
                ht->dv[ctr++] = i;
            }
            for (i = ht->ebl+1; i < ht->ndv+2; ++i) {
                ht->dv[ctr++] = i;
            }
        }

    }
    /* generate divmask map */
    ht->dm  = (sdm_t *)calloc(
            (unsigned long)(ht->ndv * ht->bpv), sizeof(sdm_t));

    /* generate random values */
    ht->rsd = 2463534242;
    ht->rn  = calloc((unsigned long)ht->evl, sizeof(val_t));
    for (i = ht->evl; i > 0; --i) {
        /* random values should not be zero */
        ht->rn[i-1] = pseudo_random_number_generator(&(ht->rsd)) | 1;
    }
    /* generate exponent vector */
    /* keep first entry empty for faster divisibility checks */
    ht->eld = 1;
    ht->hd  = (hd_t *)calloc(ht->esz, sizeof(hd_t));
    ht->ev  = (exp_t **)malloc(ht->esz * sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(ERRSTREAM, "Computation needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "could not initialize exponent vector for hash table,\n");
        fprintf(ERRSTREAM, "esz = %lu, segmentation fault will follow.\n", (unsigned long)ht->esz);
    }
    exp_t *tmp  = (exp_t *)malloc(
            (unsigned long)ht->evl * ht->esz * sizeof(exp_t));
    if (tmp == NULL) {
        fprintf(ERRSTREAM, "Exponent storage needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "initialization failed, esz = %lu,\n", (unsigned long)ht->esz);
        fprintf(ERRSTREAM, "segmentation fault will follow.\n");
    }
    const hl_t esz  = ht->esz;
    for (j = 0; j < esz; ++j) {
        ht->ev[j]  = tmp + (j*ht->evl);
    }
    st->max_bht_size  = ht->esz;
    return ht;
}

ht_t *copy_hash_table(
    const ht_t *bht
    )
{
    hl_t j;

    ht_t *ht  = (ht_t *)malloc(sizeof(ht_t));

    ht->nv    = bht->nv;
    ht->mo    = bht->mo;
    ht->evl   = bht->evl;
    ht->ebl   = bht->ebl;
    ht->hsz   = bht->hsz;
    ht->esz   = bht->esz;

    ht->hmap  = calloc(ht->hsz, sizeof(hi_t));
    memcpy(ht->hmap, bht->hmap, (unsigned long)ht->hsz * sizeof(hi_t));

    ht->ndv = bht->ndv;
    ht->bpv = bht->bpv;
    ht->dm  = bht->dm;
    ht->rn  = bht->rn;
    ht->dv  = bht->dv;

    /* generate exponent vector */
    /* keep first entry empty for faster divisibility checks */
    ht->hd  = (hd_t *)calloc(ht->esz, sizeof(hd_t));

    memcpy(ht->hd, bht->hd, (unsigned long)ht->esz * sizeof(hd_t));
    ht->ev  = (exp_t **)malloc(ht->esz * sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(ERRSTREAM, "Computation needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "could not initialize exponent vector for hash table,\n");
        fprintf(ERRSTREAM, "esz = %lu, segmentation fault will follow.\n", (unsigned long)ht->esz);
    }
    exp_t *tmp  = (exp_t *)malloc(
            (unsigned long)ht->evl * ht->esz * sizeof(exp_t));
    if (tmp == NULL) {
        fprintf(ERRSTREAM, "Exponent storage needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "initialization failed, esz = %lu,\n", (unsigned long)ht->esz);
        fprintf(ERRSTREAM, "segmentation fault will follow.\n");
    }
    memcpy(tmp, bht->ev[0], (unsigned long)ht->evl * ht->esz * sizeof(exp_t));
    ht->eld = bht->eld;
    const hl_t esz  = ht->esz;
    for (j = 0; j < esz; ++j) {
        ht->ev[j]  = tmp + (j*ht->evl);
    }
    return ht;
}


ht_t *initialize_secondary_hash_table(
    const ht_t * const bht,
    const md_t * const md
    )
{
    hl_t j;

    ht_t *ht  = (ht_t *)malloc(sizeof(ht_t));
    ht->nv    = bht->nv;
    ht->mo    = bht->mo;
    ht->evl   = bht->evl;
    ht->ebl   = bht->ebl;

    /* generate map */
    int32_t min = 3 > md->init_hts-5 ? 3 : md->init_hts-5;
    ht->hsz   = (hl_t)pow(2, min);
    ht->esz   = ht->hsz / 2;
    ht->hmap  = calloc(ht->hsz, sizeof(hi_t));

    /* divisor mask and random number seeds from basis hash table */
    ht->ndv = bht->ndv;
    ht->bpv = bht->bpv;
    ht->dm  = bht->dm;
    ht->rn  = bht->rn;
    ht->dv  = bht->dv;

    /* generate exponent vector */
    /* keep first entry empty for faster divisibility checks */
    ht->eld = 1;
    ht->hd  = (hd_t *)calloc(ht->esz, sizeof(hd_t));
    ht->ev  = (exp_t **)malloc(ht->esz * sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(ERRSTREAM, "Computation needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "could not initialize exponent vector for hash table,\n");
        fprintf(ERRSTREAM, "esz = %lu, segmentation fault will follow.\n", (unsigned long)ht->esz);
    }
    exp_t *tmp  = (exp_t *)malloc(
            (unsigned long)ht->evl * ht->esz * sizeof(exp_t));
    if (tmp == NULL) {
        fprintf(ERRSTREAM, "Exponent storage needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "initialization failed, esz = %lu,\n", (unsigned long)ht->esz);
        fprintf(ERRSTREAM, "segmentation fault will follow.\n");
    }
    const hl_t esz  = ht->esz;
    for (j = 0; j < esz; ++j) {
        ht->ev[j]  = tmp + (j*ht->evl);
    }
    return ht;
}

void free_shared_hash_data(
    ht_t *ht
    )
{
    if (ht != NULL) {
        if (ht->rn) {
            free(ht->rn);
            ht->rn = NULL;
        }
        if (ht->dv) {
            free(ht->dv);
            ht->dv = NULL;
        }
        if (ht->dm) {
            free(ht->dm);
            ht->dm = NULL;
        }
    }
}

void free_hash_table(
    ht_t **htp
    )
{
    ht_t *ht  = *htp;
    if (ht->hmap) {
        free(ht->hmap);
        ht->hmap = NULL;
    }
    if (ht->hd) {
        free(ht->hd);
        ht->hd  = NULL;
    }
    if (ht->ev) {
        /* note: memory is allocated as one big block,
            *       so freeing ev[0] is enough */
        free(ht->ev[0]);
        free(ht->ev);
        ht->ev  = NULL;
    }
    free(ht);
    ht    = NULL;
    *htp  = ht;
}

void full_free_hash_table(
                     ht_t **htp
                     )
{
  ht_t *ht  = *htp;
  if (ht->hmap) {
    free(ht->hmap);
    ht->hmap = NULL;
  }
  if (ht->hd) {
    free(ht->hd);
    ht->hd  = NULL;
  }
  if (ht->ev) {
    /* note: memory is allocated as one big block,
     *       so freeing ev[0] is enough */
    free(ht->ev[0]);
    free(ht->ev);
    ht->ev  = NULL;
  }
  if (ht != NULL) {
    if (ht->rn) {
      free(ht->rn);
      ht->rn = NULL;
    }
    if (ht->dv) {
      free(ht->dv);
      ht->dv = NULL;
    }
    if (ht->dm) {
      free(ht->dm);
      ht->dm = NULL;
    }
  }
  free(ht);
  ht    = NULL;
  *htp  = ht;
}

/* we just double the hash table size */
static void enlarge_hash_table(
    ht_t *ht
    )
{
    hl_t i, j;
    val_t h, k;

    ht->esz = 2 * ht->esz;
    const hl_t esz  = ht->esz;
    const hi_t eld  = ht->eld;

    ht->hd    = realloc(ht->hd, esz * sizeof(hd_t));
    memset(ht->hd+eld, 0, (esz-eld) * sizeof(hd_t));
    ht->ev    = realloc(ht->ev, esz * sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(ERRSTREAM, "Enlarging hash table failed for esz = %lu,\n", (unsigned long)esz);
        fprintf(ERRSTREAM, "segmentation fault will follow.\n");
    }
    /* note: memory is allocated as one big block, so reallocating
     *       memory from ev[0] is enough    */
    ht->ev[0] = realloc(ht->ev[0],
            esz * (unsigned long)ht->evl * sizeof(exp_t));
    if (ht->ev[0] == NULL) {
        fprintf(ERRSTREAM, "Enlarging exponent vector for hash table failed\n");
        fprintf(ERRSTREAM, "for esz = %lu, segmentation fault will follow.\n", (unsigned long)esz);
    }
    /* due to realloc we have to reset ALL ev entries,
     * memory might have been moved */
    for (i = 1; i < esz; ++i) {
        ht->ev[i] = ht->ev[0] + (i*ht->evl);
    }

    /* The hash table should be double the size of the exponent space in
     * order to never get a fill in over 50%. If the exponent size is now
     * enlarge to 2^31 elements that's the limit we can go. Thus we cannot
     * enlarge the hash table size any further and have to live with more
     * than 50% fill in. */
    if (ht->hsz < (hl_t)pow(2,32)) {
        /* remember which positions are currently published in hmap;
         * unpublished positions are lost CAS slots of the
         * thread-safe insert and must stay unreachable, otherwise
         * the rehash would revive them as duplicates of the same
         * vector */
        const hl_t old_hsz = ht->hsz;
        hi_t *pub = (hi_t *)calloc(eld > 1 ? eld : 1, sizeof(hi_t));
        for (i = 0; i < old_hsz; ++i) {
            const hi_t p = ht->hmap[i];
            if (p) {
                pub[p] = 1;
            }
        }
        ht->hsz = 2 * ht->hsz;
        const hl_t hsz  = ht->hsz;
        ht->hmap  = realloc(ht->hmap, hsz * sizeof(hi_t));
        if (ht->hmap == NULL) {
            fprintf(ERRSTREAM, "Enlarging hash table failed for hsz = %lu,\n", (unsigned long)hsz);
            fprintf(ERRSTREAM, "segmentation fault will follow.\n");
        }
        memset(ht->hmap, 0, hsz * sizeof(hi_t));
        const hi_t mod =  (hi_t )(hsz-1);

        /* reinsert known elements (published ones only) */
        for (i = 1; i < eld; ++i) {
            if (!pub[i]) {
                continue;
            }
            h = ht->hd[i].val;

            /* probing */
            k = h;
            for (j = 0; j < hsz; ++j) {
                k = (k+j) & mod;
                if (ht->hmap[k]) {
                    continue;
                }
                ht->hmap[k] = i;
                break;
            }
        }
        free(pub);
    } else {
        if (ht->hsz == (hl_t)pow(2,32)) {
          fprintf(ERRSTREAM, "Exponent space is now 2^32 elements wide, we cannot\n");
          fprintf(ERRSTREAM, "enlarge the hash table any further, thus fill in gets\n");
          fprintf(ERRSTREAM, "over 50%% and performance of hashing may get worse.\n");
        } else {
          fprintf(ERRSTREAM, "Hash table is full, we can no longer enlarge\n");
          fprintf(ERRSTREAM, "Segmentation fault will follow.\n");
          free(ht->hmap);
          ht->hmap  = NULL;
        }
    }
}

static inline sdm_t generate_short_divmask(
    const exp_t * const a,
    const ht_t *ht
    )
{
  len_t i, j;
  int32_t res = 0;
  int32_t ctr = 0;
  const len_t ndv         = ht->ndv;
  const len_t * const dv  = ht->dv;
  const len_t bpv         = ht->bpv;

  for (i = 0; i < ndv; ++i) {
    for (j = 0; j < bpv; ++j) {
      if ((sdm_t)a[dv[i]] >= ht->dm[ctr]) {
        res |= (uint32_t)(1u) << ctr;
      }
      ctr++;
    }
  }

  return res;
}

/* note: we calculate the divmask after reading in the input generators. thoseV
 * are first stored in the local hash table. thus we use the local exponents to
 * generate the divmask */
void calculate_divmask(
    ht_t *ht
    )
{
  hi_t i;
  hl_t k;
  len_t j, steps;
  int32_t ctr = 0;
  const len_t * const dv  = ht->dv;
  exp_t **ev  = ht->ev;

  deg_t *max_exp  = (deg_t *)malloc((unsigned long)ht->ndv * sizeof(deg_t));
  deg_t *min_exp  = (deg_t *)malloc((unsigned long)ht->ndv * sizeof(deg_t));

  exp_t *e  = ev[1];

  /* get initial values from first hash table entry */
  for (i = 0; i < ht->ndv; ++i) {
    max_exp[i]  = min_exp[i]  = e[dv[i]];
  }

  /* get maximal and minimal exponent element entries in hash table */
  for (i = 2; i < ht->eld; ++i) {
    e = ev[i];
    for (j = 0; j < ht->ndv; ++j) {
      if (e[dv[j]] > max_exp[j]) {
        max_exp[j]  = e[dv[j]];
        continue;
      }
      if (e[dv[j]] < min_exp[j]) {
        min_exp[j]  = e[dv[j]];
      }
    }
  }

  /* calculate average values for generating divmasks */
  for (i = 0; i < ht->ndv; ++i) {
    steps = (max_exp[i] - min_exp[i]) / ht->bpv;
    if (steps == 0)
      steps++;
    for (j = 0; j < ht->bpv; ++j) {
      ht->dm[ctr++] = (sdm_t)steps++;
    }
  }

  /* initialize divmasks for elements already added to hash table */
  for (k = 1; k < ht->eld; k++) {
    ht->hd[k].sdm = generate_short_divmask(ev[k], ht);
  }

  free(max_exp);
  free(min_exp);
}

/* returns zero if a is not divisible by b, else 1 is returned */
static inline hi_t check_monomial_division(
    const hi_t a,
    const hi_t b,
    const ht_t *ht
    )
{
  len_t i;

  /* short divisor mask check */
  if (ht->hd[b].sdm & ~ht->hd[a].sdm) {
    return 0;
  }

  const len_t evl = ht->evl;

  const exp_t *const ea = ht->ev[a];
  const exp_t *const eb = ht->ev[b];

  /* printf("! no sdm decision !\n"); */
  /* exponent check */
  for (i = 0; i < evl-1; i += 2) {
    if (ea[i] < eb[i] || ea[i+1] < eb[i+1]) {
      return 0;
    }
  }
  if (ea[evl-1] < eb[evl-1]) {
    return 0;
  }
  return 1;
}

static inline void check_monomial_division_in_update(
    hi_t *a,
    const len_t start,
    const len_t end,
    const hi_t b,
    const ht_t *ht
    )
{
    len_t i, j;
    const len_t evl = ht->evl;

    const sdm_t sb        = ht->hd[b].sdm;
    const exp_t *const eb = ht->ev[b];
    /* pairs are sorted, we only have to search entries
     * above the starting point */
        j = start+1;
restart:
    for (; j < end; ++j) {
        if (a[j] == 0) {
            continue;
        }
        /* short divisor mask check */
        if (~ht->hd[a[j]].sdm & sb) {
            continue;
        }
        const exp_t *const ea = ht->ev[a[j]];
        /* exponent check */
        for (i = 0; i < evl-1; i += 2) {
            if (ea[i] < eb[i] || ea[i+1] < eb[i+1]) {
                j++;
                goto restart;
            }
        }
        if (ea[evl-1] < eb[evl-1]) {
            continue;
        }
        a[j]  = 0;
    }
}

static inline hi_t check_lm_divisibility_and_insert_in_hash_table(
    const exp_t *a,
    ht_t *ht,
    const bs_t * const bs
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j;
    exp_t *e;
    hd_t *d;
    const len_t lml   = bs->lml;

    const sdm_t * const lms = bs->lm;
    const bl_t * const lmps = bs->lmps;

    const sdm_t nsdm  = ~generate_short_divmask(a, ht);

    val_t h = 0;
    const len_t evl = ht->evl;
    const hl_t hsz  = ht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod = (hi_t)(ht->hsz - 1);

    /* check divisibility w.r.t. current lead monomials */
    i = 0;
start:
    while (i < lml && lms[i] & nsdm) {
        i++;
    }
    if (i < lml) {
        e = ht->ev[bs->hm[lmps[i]][OFFSET]];
        for (j = 0; j < evl; ++j) {
            if (e[j] > a[j]) {
                i++;
                goto start;
            }
        }
        /* divisible by lm */
        return 0;
    }
    /* if we are here then a is not divisible by a current
     * lead monomial and we can add it to the hash table */

    /* generate hash value */
    for (j = 0; j < evl; ++j) {
        h +=  ht->rn[j] * a[j];
    }
    /* probing */
    k = h;
    i = 0;
restart:
    for (; i < hsz; ++i) {
        k = (hi_t)((k+i) & mod);
        const hi_t hm = ht->hmap[k];
        if (!hm) {
            break;
        }
        if (ht->hd[hm].val != h) {
            continue;
        }
        const exp_t * const ehm = ht->ev[hm];
        for (j = 0; j < evl-1; j += 2) {
            if (a[j] != ehm[j] || a[j+1] != ehm[j+1]) {
                i++;
                goto restart;
            }
        }
        if (a[evl-1] != ehm[evl-1]) {
            i++;
            goto restart;
        }
        return hm;
    }

    /* add element to hash table */
    ht->hmap[k]  = pos = (hi_t)ht->eld;
    e   = ht->ev[pos];
    d   = ht->hd + pos;
    memcpy(e, a, (unsigned long)evl * sizeof(exp_t));
    d->sdm  =   generate_short_divmask(e, ht);
    d->deg  =   e[0];
    d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
    d->val  =   h;

    ht->eld++;

    return pos;
}

static inline hi_t insert_multiplied_signature_in_hash_table(
    const hm_t h1,
    const hm_t h2,
    ht_t *ht
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j;
    exp_t *e;
    exp_t *a = ht->ev[0];
    hd_t *d;
    val_t h = 0;
    const len_t evl = ht->evl;
    const hl_t hsz = ht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod = (hi_t)(ht->hsz - 1);

    h   =   h1 + h2;

    /* generate exponent vector */
    for (j = 0; j < evl; ++j) {
        a[j] = ht->ev[h1][j] + ht->ev[h2][j];
    }
    /* probing */
    k = h;
    i = 0;
restart:
    for (; i < hsz; ++i) {
        k = (hi_t)((k+i) & mod);
        const hi_t hm = ht->hmap[k];
        if (!hm) {
            break;
        }
        if (ht->hd[hm].val != h) {
            continue;
        }
        const exp_t * const ehm = ht->ev[hm];
        for (j = 0; j < evl-1; j += 2) {
            if (a[j] != ehm[j] || a[j+1] != ehm[j+1]) {
                i++;
                goto restart;
            }
        }
        if (a[evl-1] != ehm[evl-1]) {
            i++;
            goto restart;
        }
        return hm;
    }

    /* add element to hash table */
    ht->hmap[k]  = pos = (hi_t)ht->eld;
    e   = ht->ev[pos];
    d   = ht->hd + pos;
    memcpy(e, a, (unsigned long)evl * sizeof(exp_t));
    d->sdm  =   generate_short_divmask(e, ht);
    d->deg  =   e[0];
    d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
    d->val  =   h;

    ht->eld++;

    return pos;
}

static inline hi_t insert_in_hash_table(
    const exp_t *a,
    ht_t *ht
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j;
    exp_t *e;
    hd_t *d;
    val_t h = 0;
    const len_t evl = ht->evl;
    const hl_t hsz = ht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod = (hi_t)(ht->hsz - 1);

    /* generate hash value */
    for (j = 0; j < evl; ++j) {
        h +=  ht->rn[j] * a[j];
    }
    /* probing */
    k = h;
    i = 0;
restart:
    for (; i < hsz; ++i) {
        k = (hi_t)((k+i) & mod);
        const hi_t hm = ht->hmap[k];
        if (!hm) {
            break;
        }
        if (ht->hd[hm].val != h) {
            continue;
        }
        const exp_t * const ehm = ht->ev[hm];
        for (j = 0; j < evl-1; j += 2) {
            if (a[j] != ehm[j] || a[j+1] != ehm[j+1]) {
                i++;
                goto restart;
            }
        }
        if (a[evl-1] != ehm[evl-1]) {
            i++;
            goto restart;
        }
        return hm;
    }

    /* add element to hash table */
    ht->hmap[k]  = pos = (hi_t)ht->eld;
    e   = ht->ev[pos];
    d   = ht->hd + pos;
    memcpy(e, a, (unsigned long)evl * sizeof(exp_t));
    d->sdm  =   generate_short_divmask(e, ht);
    d->deg  =   e[0];
    d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
    d->val  =   h;

    ht->eld++;

    return pos;
}

/* Thread-safe version of insert_in_hash_table() for use inside
 * OpenMP parallel regions. The table must be pre-enlarged such
 * that no resize (enlarge_hash_table) can occur while threads
 * are running.
 *
 * Returns the index of the exponent vector in the hash table.
 * If created is not NULL it is set to 1 if a new entry was
 * created, 0 if the vector was already contained, and -1 if the
 * table overflowed (in which case 0 is returned).
 *
 * Correctness:
 * - Writers fill ev[pos]/hd[pos] BEFORE publishing hmap[k] = pos
 *   (release). Readers acquire hmap[k] before touching hd/ev.
 * - hmap slots are never reset to 0, so an empty slot reached
 *   during probing proves that the vector is absent.
 * - If two threads insert the same vector, both probe to the same
 *   first empty slot; one wins the CAS, the other re-checks the
 *   slot and finds the winner's published entry.
 * - Lost CAS races only leave unused (hole) slots in ev/hd, which
 *   does not affect correctness, only capacity. */
static inline hi_t thread_safe_insert_in_hash_table(
    const exp_t *a,
    ht_t *ht,
    int *created
    )
{
    hl_t i;
    hi_t k;
    len_t j;
    val_t h = 0;
    const len_t evl = ht->evl;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod = (hi_t)(ht->hsz - 1);

    /* generate hash value */
    for (j = 0; j < evl; ++j) {
        h +=  ht->rn[j] * a[j];
    }
    /* probing */
    k = h;
    i = 0;
restart:
    for (; i < ht->hsz; ++i) {
        k = (hi_t)((k+i) & mod);
        const hi_t hm = __atomic_load_n(&ht->hmap[k], __ATOMIC_ACQUIRE);
        if (!hm) {
            /* a is not in the table; try to claim slot k. Reserve a
             * new position for the vector without letting eld run
             * past esz: a position that is counted but never written
             * would be rehashed with an uninitialised vector as soon
             * as the table enlarges. The CAS loop below only
             * succeeds while the position is still in bounds. */
            hl_t expected = __atomic_load_n(&ht->eld, __ATOMIC_RELAXED);
            while (expected < (hl_t)ht->esz
                    && !__atomic_compare_exchange_n(
                           &ht->eld, &expected, expected + 1, 0,
                           __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            }
            if (expected >= (hl_t)ht->esz) {
                if (created != NULL) {
                    *created = -1;
                }
                return 0;
            }
            const hi_t pos = (hi_t)expected;
            exp_t *e   = ht->ev[pos];
            hd_t *d    = ht->hd + pos;
            memcpy(e, a, (unsigned long)evl * sizeof(exp_t));
            d->sdm  =   generate_short_divmask(e, ht);
            d->deg  =   e[0];
            d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
            d->val  =   h;
            hi_t zero = 0;
            if (__atomic_compare_exchange_n(
                    &ht->hmap[k], &zero, (hi_t)pos, 0,
                    __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (created != NULL) {
                    *created = 1;
                }
                return pos;
            }
            /* lost the race: another thread published hmap[k]
             * concurrently; re-check whether it is our vector */
            {
                const hi_t hm2 = __atomic_load_n(&ht->hmap[k], __ATOMIC_ACQUIRE);
                if (ht->hd[hm2].val == h) {
                    const exp_t * const ehm = ht->ev[hm2];
                    len_t eq = 1;
                    for (j = 0; j < evl-1; j += 2) {
                        if (a[j] != ehm[j] || a[j+1] != ehm[j+1]) {
                            eq = 0;
                            break;
                        }
                    }
                    if (eq && a[evl-1] != ehm[evl-1]) {
                        eq = 0;
                    }
                    if (eq) {
                        if (created != NULL) {
                            *created = 0;
                        }
                        return hm2;
                    }
                }
            }
            i++;
            goto restart;
        }
        if (ht->hd[hm].val != h) {
            continue;
        }
        const exp_t * const ehm = ht->ev[hm];
        for (j = 0; j < evl-1; j += 2) {
            if (a[j] != ehm[j] || a[j+1] != ehm[j+1]) {
                i++;
                goto restart;
            }
        }
        if (a[evl-1] != ehm[evl-1]) {
            i++;
            goto restart;
        }
        if (created != NULL) {
            *created = 0;
        }
        return hm;
    }
    /* table is full, which must not happen if the table was
     * pre-enlarged properly */
    if (created != NULL) {
        *created = -1;
    }
    return 0;
}

static inline void reinitialize_hash_table(
    ht_t *ht,
    const hl_t size
    )
{
    hl_t i;
    /* is there still enough space in the local table? */
    if (size >= (ht->esz)) {
        while (size >= ht->esz) {
            ht->esz = 2 * ht->esz;
            ht->hsz = 2 * ht->hsz;
        }
        const hl_t esz  = ht->esz;
        const hl_t hsz  = ht->hsz;
        const len_t evl = ht->evl;
        ht->hd  = realloc(ht->hd, esz * sizeof(hd_t));
        ht->ev  = realloc(ht->ev, esz * sizeof(exp_t *));
        if (ht->ev == NULL) {
            fprintf(ERRSTREAM, "Computation needs too much memory on this machine,\n");
            fprintf(ERRSTREAM, "could not reinitialize exponent vector for hash table,\n");
            fprintf(ERRSTREAM, "esz = %lu, segmentation fault will follow.\n", (unsigned long)esz);
        }
        /* note: memory is allocated as one big block, so reallocating
         *       memory from evl[0] is enough    */
        ht->ev[0]  = realloc(ht->ev[0],
                esz * (unsigned long)evl * sizeof(exp_t));
        if (ht->ev[0] == NULL) {
            fprintf(ERRSTREAM, "Exponent storage needs too much memory on this machine,\n");
            fprintf(ERRSTREAM, "reinitialization failed, esz = %lu\n", (unsigned long)esz);
            fprintf(ERRSTREAM, "segmentation fault will follow.\n");
        }
        /* due to realloc we have to reset ALL evl entries, memory might be moved */
        for (i = 1; i < esz; ++i) {
            ht->ev[i] = ht->ev[0] + (i*evl);
        }
        ht->hmap  = realloc(ht->hmap, hsz * sizeof(hi_t));
    }
    memset(ht->hd, 0, ht->esz * sizeof(hd_t));
    memset(ht->hmap, 0, ht->hsz * sizeof(hi_t));

    ht->eld  = 1;
}

static inline void clean_hash_table(
        ht_t *ht
    )
{
    memset(ht->hd, 0, ht->esz * sizeof(hd_t));
    memset(ht->hmap, 0, ht->hsz * sizeof(hi_t));

    ht->eld  = 1;
}

static inline int prime_monomials(
    const hi_t a,
    const hi_t b,
    const ht_t *ht
    )
{
    len_t i;

    const exp_t * const ea = ht->ev[a];
    const exp_t * const eb = ht->ev[b];

    const len_t evl = ht->evl;
    const len_t ebl = ht->ebl;

    for (i = 1; i < ebl; ++i) {
        if (ea[i] != 0 && eb[i] != 0) {
            return 0;
        }
    }
    for (i = ebl+1; i < evl; ++i) {
        if (ea[i] != 0 && eb[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static inline void insert_plcms_in_basis_hash_table(
    ps_t *psl,
    spair_t *pp,
    ht_t *bht,
    const ht_t *uht,
    const bs_t * const bs,
    const hi_t * const lcms,
    const len_t start,
    const len_t end
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j, l, m;
    hd_t *d;

    spair_t *ps     = psl->p;
    const len_t evl = bht->evl;
    const hl_t hsz  = bht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod = (hi_t)(hsz - 1);
    hm_t * const * const hm = bs->hm;
    m = start;
    l = 0;
letsgo:
    for (; l < end; ++l) {
        if (lcms[l] == 0) {
            continue;
        }
        if (prime_monomials(
                    hm[pp[l].gen1][OFFSET], hm[pp[0].gen2][OFFSET], bht)) {
            continue;
        }
        ps[m] = pp[l];
        const val_t h = uht->hd[lcms[l]].val;
        memcpy(bht->ev[bht->eld], uht->ev[lcms[l]],
                (unsigned long)evl * sizeof(exp_t));
        const exp_t * const n = bht->ev[bht->eld];
        k = h;
        i = 0;
restart:
        for (; i < hsz; ++i) {
            k = (hi_t)(k+i) & mod;
            const hi_t hm = bht->hmap[k];
            if (!hm) {
                break;
            }
            if (bht->hd[hm].val != h) {
                continue;
            }
            const exp_t * const ehm = bht->ev[hm];
            for (j = 0; j < evl-1; j += 2) {
                if (n[j] != ehm[j] || n[j+1] != ehm[j+1]) {
                    i++;
                    goto restart;
                }
            }
            if (n[evl-1] != ehm[evl-1]) {
                i++;
                goto restart;
            }
            ps[m++].lcm = hm;
            l++;
            goto letsgo;
        }

        /* add element to hash table */
        bht->hmap[k] = pos = (hi_t)bht->eld;
        d = bht->hd + bht->eld;
        d->sdm  = uht->hd[lcms[l]].sdm;
        d->deg  = uht->hd[lcms[l]].deg;
        d->val  = h;

        bht->eld++;
        ps[m++].lcm =  pos;
    }
    psl->ld = m;
}

static inline void switch_hcm_data_to_basis_hash_table(
    hi_t *hcm,
    ht_t *bht,
    const mat_t *mat,
    const ht_t * const sht,
    const int nthrds
    )
{
    const len_t start = mat->ncl;
    const len_t end   = mat->nc;

    /* pre-size bht so that no table growth is needed inside the
     * parallel region (the factor two also covers slots reserved
     * but lost in concurrent insertion races). Insertions that
     * overflow anyway are retried serially on an enlarged table. */
    while (bht->esz - bht->eld < 2ULL * (hl_t)mat->ncr + 1) {
        enlarge_hash_table(bht);
    }

    int *cr = (int *)malloc((unsigned long)(end - start) * sizeof(int));
    #pragma omp parallel for num_threads(nthrds) schedule(static)
    for (len_t i = start; i < end; ++i) {
        hcm[i] = thread_safe_insert_in_hash_table(
                sht->ev[hcm[i]], bht, &cr[i-start]);
    }
    /* serial retry for the insertions that overflowed the
     * pre-enlarged table (rare; the factor two headroom only
     * fails when many concurrent races waste slots) */
    for (len_t i = start; i < end; ++i) {
        if (cr[i-start] < 0) {
            while (bht->esz - bht->eld < 2) {
                enlarge_hash_table(bht);
            }
            hcm[i] = insert_in_hash_table(sht->ev[hcm[i]], bht);
        }
    }
    free(cr);
}

static inline void insert_in_basis_hash_table_pivots(
    hm_t *row,
    ht_t *bht,
    const ht_t * const sht,
    const hi_t * const hcm,
    const md_t * const st
    )
{
    len_t l;

    /* while (bht->esz - bht->eld < row[LENGTH]) {
        enlarge_hash_table(bht);
    } */

    const len_t len = row[LENGTH]+OFFSET;
    const len_t evl = bht->evl;

    /* const hd_t * const hds    = sht->hd; */
    exp_t * const * const evs = sht->ev;

    exp_t *evt  = (exp_t *)malloc((unsigned long)evl * sizeof(exp_t));
    for (l = OFFSET; l < len; ++l) {
        memcpy(evt, evs[hcm[row[l]]],
                (unsigned long)evl * sizeof(exp_t));
        row[l] = insert_in_hash_table(evt, bht);
    }
    free(evt);
}

static inline void insert_poly_in_hash_table(
    hm_t *row,
    const hm_t * const b,
    const ht_t * const ht1,
    ht_t *ht2
    )
{
    len_t l;

    const len_t len = b[LENGTH]+OFFSET;

    l = OFFSET;

    for (; l < len; ++l) {
        row[l] = insert_in_hash_table(ht1->ev[b[l]], ht2);
    }
}

static inline void insert_multiplied_poly_in_hash_table(
    hm_t *row,
    const val_t h1,
    const exp_t * const ea,
    const hm_t * const b,
    const ht_t * const ht1,
    ht_t *ht2
    )
{
    len_t j, l;
    exp_t *n;

    const len_t len = b[LENGTH]+OFFSET;
    const len_t evl = ht1->evl;

    exp_t * const *ev1      = ht1->ev;

    exp_t **ev2     = ht2->ev;

    l = OFFSET;

    for (; l < len; ++l) {
        const exp_t * const eb = ev1[b[l]];

        n = ev2[ht2->eld];
        for (j = 0; j < evl; ++j) {
            n[j]  = (exp_t)(ea[j] + eb[j]);
        }

        row[l] = insert_in_hash_table(n, ht2);
    }
}

static inline void reinsert_in_hash_table(
    hm_t *row,
    exp_t * const *oev,
    ht_t *ht
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j, l;
    exp_t *e;
    hd_t *d;
    val_t h;

    const len_t len = row[LENGTH]+OFFSET;
    const len_t evl = ht->evl;
    const hi_t hsz  = ht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod  = (hi_t)(hsz - 1);
    l = OFFSET;
letsgo:
    for (; l < len; ++l) {
        const exp_t * const n = oev[row[l]];
        /* generate hash value */
        h = 0;
        for (j = 0; j < evl; ++j) {
            h +=  ht->rn[j] * n[j];
        }
        k = h;
        i = 0;
restart:
        for (; i < hsz; ++i) {
            k = (hi_t)(k+i) & mod;
            const hi_t hm  = ht->hmap[k];
            if (!hm) {
                break;
            }
            if (ht->hd[hm].val != h) {
                continue;
            }
            const exp_t * const ehm = ht->ev[hm];
            for (j = 0; j < evl-1; j += 2) {
                if (n[j] != ehm[j] || n[j+1] != ehm[j+1]) {
                    i++;
                    goto restart;
                }
            }
            if (n[evl-1] != ehm[evl-1]) {
                i++;
                goto restart;
            }
            row[l] = hm;
            l++;
            goto letsgo;
        }

        /* add element to hash table */
        ht->hmap[k] = pos = (hi_t)ht->eld;
        e = ht->ev[ht->eld];
        d = ht->hd + ht->eld;
        memcpy(e, n, (unsigned long)evl * sizeof(exp_t));
        d->sdm  =   generate_short_divmask(e, ht);
        d->deg  =   e[0];
        d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
        d->val  =   h;

        ht->eld++;
        row[l] =  pos;
    }
}

void reset_hash_table_indices(
        ht_t *ht,
        const hi_t * const hcm,
        const len_t len
        )
{
    for (len_t i = 0; i < len; ++i) {
        ht->hd[hcm[i]].idx = 0;
    }
}

static void reset_hash_table(
    ht_t *ht,
    bs_t *bs,
    ps_t *psl,
    md_t *st
    )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    len_t i;
    hi_t k;
    exp_t *e;

    spair_t *ps = psl->p;
    exp_t **oev  = ht->ev;

    const len_t evl = ht->evl;
    const hl_t esz  = ht->esz;
    const bl_t bld  = bs->ld;
    const len_t pld = psl->ld;

    ht->ev  = calloc(esz, sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(ERRSTREAM, "Computation needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "cannot reset ht->ev, esz = %lu\n", (unsigned long)esz);
        fprintf(ERRSTREAM, "segmentation fault will follow.\n");
    }
    exp_t *tmp  = (exp_t *)malloc(
            (unsigned long)evl * esz * sizeof(exp_t));
    if (tmp == NULL) {
        fprintf(ERRSTREAM, "Computation needs too much memory on this machine,\n");
        fprintf(ERRSTREAM, "resetting table failed, esz = %lu\n", (unsigned long)esz);
        fprintf(ERRSTREAM, "segmentation fault will follow.\n");
    }
    for (k = 0; k < esz; ++k) {
        ht->ev[k]  = tmp + k*evl;
    }
    ht->eld = 1;
    memset(ht->hmap, 0, ht->hsz * sizeof(hi_t));
    memset(ht->hd, 0, esz * sizeof(hd_t));

    /* reinsert known elements */
    for (i = 0; i < bld; ++i) {
      if (bs->red[i] < 2) {
        reinsert_in_hash_table(bs->hm[i], oev, ht);
      }
    }
    for (i = 0; i < pld; ++i) {
        e = oev[ps[i].lcm];
        ps[i].lcm = insert_in_hash_table(e, ht);
    }
    /* note: all memory is allocated as a big block, so it is
     *       enough to free oev[0].       */
    free(oev[0]);
    free(oev);

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->rht_ctime  +=  ct1 - ct0;
    st->rht_rtime  +=  rt1 - rt0;
}

/* computes lcm of a and b from ht1 and inserts it in ht2 */
static inline hi_t get_lcm(
    const hi_t a,
    const hi_t b,
    const ht_t *ht1,
    ht_t *ht2
    )
{
    len_t i;

    /* exponents of basis elements, thus from basis hash table */
    const exp_t * const ea = ht1->ev[a];
    const exp_t * const eb = ht1->ev[b];
    exp_t etmp[ht1->evl];
    const len_t evl = ht1->evl;
    const len_t ebl = ht1->ebl;

    /* set degree(s), if ebl == 0, i.e. we do not have an elimination block
     * order then the second for loop is just not executed and the third one
     * computes correctly the full degree of the lcm. */
    for (i = 1; i < evl; ++i) {
        etmp[i]  = ea[i] < eb[i] ? eb[i] : ea[i];
    }
    /* reset degree entries */
    etmp[0]   = 0;
    etmp[ebl] = 0;
    for (i = 1; i < ebl; ++i) {
        etmp[0]  += etmp[i];
    }
    for (i = ebl+1; i < evl; ++i) {
        etmp[ebl] += etmp[i];
    }
    /* printf("lcm -> ");
      * for (int ii = 0; ii < evl; ++ii) {
      *     printf("%d ", etmp[ii]);
      * }
      * printf("\n"); */
    return insert_in_hash_table(etmp, ht2);
}

/* computes lcm of a and b from ht1 and inserts it in ht2 in a
 * thread-safe way; created (if not NULL) receives 1 if a new
 * entry was created, 0 if the lcm was already present, -1 on
 * table overflow. ht2 must be pre-enlarged. */
static inline hi_t get_lcm_thread_safe(
    const hi_t a,
    const hi_t b,
    const ht_t *ht1,
    ht_t *ht2,
    int *created
    )
{
    len_t i;

    /* exponents of basis elements, thus from basis hash table */
    const exp_t * const ea = ht1->ev[a];
    const exp_t * const eb = ht1->ev[b];
    exp_t etmp[ht1->evl];
    const len_t evl = ht1->evl;
    const len_t ebl = ht1->ebl;

    /* set degree(s), if ebl == 0, i.e. we do not have an elimination block
     * order then the second for loop is just not executed and the third one
     * computes correctly the full degree of the lcm. */
    for (i = 1; i < evl; ++i) {
        etmp[i]  = ea[i] < eb[i] ? eb[i] : ea[i];
    }
    /* reset degree entries */
    etmp[0]   = 0;
    etmp[ebl] = 0;
    for (i = 1; i < ebl; ++i) {
        etmp[0]  += etmp[i];
    }
    for (i = ebl+1; i < evl; ++i) {
        etmp[ebl] += etmp[i];
    }
    return thread_safe_insert_in_hash_table(etmp, ht2, created);
}

static inline hm_t *poly_to_matrix_row(
    ht_t *sht,
    const ht_t *bht,
    const hm_t *poly
    )
{
  hm_t *row = (hm_t *)malloc((unsigned long)(poly[LENGTH]+OFFSET) * sizeof(hm_t));
  row[COEFFS]   = poly[COEFFS];
  row[PRELOOP]  = poly[PRELOOP];
  row[LENGTH]   = poly[LENGTH];
  /* hash table product insertions appear only here:
   * we check for hash table enlargements first and then do the insertions
   * without further elargment checks there */
  while (sht->eld+poly[LENGTH] >= sht->esz) {
    enlarge_hash_table(sht);
  }
  insert_poly_in_hash_table(row, poly, bht, sht);

  return row;
}
static inline hm_t *multiplied_poly_to_matrix_row(
    ht_t *sht,
    const ht_t *bht,
    const val_t hm,
    const exp_t * const em,
    const hm_t *poly
    )
{
  hm_t *row = (hm_t *)malloc((uint64_t)(poly[LENGTH]+OFFSET) * sizeof(hm_t));
  row[COEFFS]   = poly[COEFFS];
  row[PRELOOP]  = poly[PRELOOP];
  row[LENGTH]   = poly[LENGTH];
  /* hash table product insertions appear only here:
   * we check for hash table enlargements first and then do the insertions
   * without further elargment checks there */
  while (sht->eld+poly[LENGTH] >= sht->esz) {
    enlarge_hash_table(sht);
  }
  insert_multiplied_poly_in_hash_table(row, hm, em, poly, bht, sht);

  return row;
}

/* Thread-safe version of multiplied_poly_to_matrix_row() for use inside
 * OpenMP parallel regions. The table sht must be pre-enlarged so that
 * no resize can occur; if the table overflows, *overflow is set and the
 * row is left incomplete (the caller must discard it). If q and q_tail
 * are not NULL, every newly created sht entry is appended to the queue
 * q at position *q_tail (which is incremented atomically). */
static inline hm_t *ts_multiplied_poly_to_matrix_row(
    ht_t *sht,
    const ht_t *bht,
    const exp_t * const em,
    const hm_t *poly,
    qent_t *q,
    hl_t *q_tail,
    int *overflow
    )
{
    hm_t *row = (hm_t *)malloc((uint64_t)(poly[LENGTH]+OFFSET) * sizeof(hm_t));
    row[COEFFS]   = poly[COEFFS];
    row[PRELOOP]  = poly[PRELOOP];
    row[LENGTH]   = poly[LENGTH];

    const len_t len = poly[LENGTH]+OFFSET;
    const len_t evl = bht->evl;

    exp_t *n = (exp_t *)malloc((unsigned long)evl * sizeof(exp_t));

    for (len_t l = OFFSET; l < len; ++l) {
        const exp_t * const eb = bht->ev[poly[l]];
        for (len_t j = 0; j < evl; ++j) {
            n[j]  = (exp_t)(em[j] + eb[j]);
        }
        int cr = 0;
        const hi_t pos = thread_safe_insert_in_hash_table(n, sht, &cr);
        row[l] = pos;
        if (cr < 0) {
            *overflow = 1;
            break;
        }
        if (cr > 0 && q != NULL) {
            const hl_t slot = __atomic_fetch_add(q_tail, 1,
                    __ATOMIC_RELAXED);
            /* publish the value first, then the ready flag: the
             * consumer of this slot spins on the flag before
             * reading the value */
            q[slot].v = pos;
            __atomic_store_n(&q[slot].flag, 1, __ATOMIC_RELEASE);
        }
    }
    free(n);

    return row;
}
