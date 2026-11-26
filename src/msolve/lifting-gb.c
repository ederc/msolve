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

#include<flint/fmpz.h>
#include "streams.h"

#define NBCHECK 2
/* number of bits of slack kept in rational reconstruction bounds */
#define RR_MARGIN 16
/* witnesses are reconstructed every rr primes, rr being the largest
   power of 2 <= nprimes / RR_PERIOD_DIV */
#define RR_PERIOD_DIV 128
/* the time of CRT in the lifting of the coefficients is taken for one
   coefficient in LIFT_TIME_STEP */
#define LIFT_TIME_STEP 64
/* a trace of a signature-based run is replaced by a trace learned by F4
   once its applications took SIG_RELEARN times its learning */
#define SIG_RELEARN 2

/* lifting of the polynomials whose witness coefficients are lifted, see
   the option --gb-lifting: 0 by multi-modular CRT and rational
   reconstruction of each coefficient, 1 by the mixed radix digits of the
   coefficients times their common denominator */
int32_t msolve_gb_lifting = 0;
/* number of products of the first primes kept by this lifting (their
   memory grows with the square of their number) */
#define MR_PRODUCTS 2048
/* number of primes from which on this lifting uses a product tree */
#define MR_COMB 256
#ifndef MIN
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#endif
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

typedef struct{
  uint32_t len; /* length of the encoded polynomial */
  /* Coefficients modulo several primes, stored as one contiguous block of
   * len * alloc entries in prime-major order: the coefficient of index j
   * modulo the k-th prime is cf_32[k * len + j] (use MODCF below).
   * Adding primes only grows the block at its end, and each new prime
   * fills one contiguous row.
   * Warning: this is free-ed dynamically (and set to NULL) as soon as the
   * polynomial has been lifted and checked. */
  uint32_t *cf_32;
  /* 64-bit residues, same layout as cf_32; only used when the lifting walks
   * 61-bit application primes (modgbs->wide). When it is allocated the
   * former cf_32 images have been migrated into it (and cf_32 freed). */
  uint64_t *cf_64;
  /* len integer numerators of the lifted coefficients; they all share the
   * common denominator lm (hence the polynomial is lm * LM + sum cf_qq[j]*m_j
   * up to the order of the terms). */
  mpz_t *cf_qq;
  mpz_t lm; /* stores the leading coefficient (hence all numerators of cf_qq should be divided by lm) */
} modpolys_struct;

typedef modpolys_struct modpolys_t[1];

/* coefficient of index j of polynomial p modulo the k-th prime */
#define MODCF(p, j, k) ((p)->cf_32[(size_t)(k) * (p)->len + (j)])
/* same coefficient when the images are stored wide (64-bit primes) */
#define MODCF64(p, j, k) ((p)->cf_64[(size_t)(k) * (p)->len + (j)])

typedef struct {
  uint32_t alloc; /* alloc -> max number of primes */
  uint32_t nprimes; /* number of primes */
  mp_limb_t *primes; /* array of prime numbers encoded with uint64_t to ensure
                       compatibility with flint */
  mp_limb_t *cf_64; /* array of length equal to number of primes which will be used
                    to copy coefficients (hence ensuring compatibility with
                    flint) */
  /* whether modular images are stored wide (64-bit primes): the per-poly
   * residues then live in modpolys[i]->cf_64 instead of cf_32 */
  int wide;
  uint32_t ld; /* number of polynomials */
  int nv; /* number of variables */
  int32_t *ldm; /* lead monomials */
  ht_t *bht; /* hash table */
  hm_t **hm; /* hashed monomials representing exponents */
  bl_t *lmps; /* position of non-redundant lead monomials in basis */
  modpolys_t *modpolys; /* array of polynomials modulo primes */
} gb_modpoly_array_struct;

typedef gb_modpoly_array_struct gb_modpoly_t[1];

#define NEWGBLIFT 1


typedef struct{
  int32_t npol; /* number of polynomials to be lifted */
  int32_t rr; /* number of primes before activating rational reconstruction */
  int32_t nsteps; /* number of steps for lifting GB (per degree) */
  int32_t *steps; /* array of length nsteps ; the sum of the entries should
                     equal npol */
  int32_t cstep; /* current step lifting GB */
  /* liftings are performed on ranges of polynomials (depending on their degrees) */
  int32_t lstart; /* index of first polynomial to be lifted */
  int32_t lend; /* index of last polynomial to be lifted */

  int crt_mult; /* indicates if multi-mod flint structures need to be
                initialized */
  mpz_t *crt; /* stores current CRT */
  int recon; /* equals 1 when some rational number can be lifted, else 0 */
  int32_t *coef; /* array of indices to lift */
  mpz_t *num; /* lifted numerator */
  mpz_t *den; /* lifted denominator */
  mpz_t gden; /* guessed denominator */
  mpz_t tmp;
  int32_t start; /* smallest index of poly whose witness coef has been lifted but not checked*/
  int32_t end; /* largest index of poly whose witness coef has been lifted but not checked*/
  int *check1; /* tells whether lifted data are ok with one more prime */
  int *check2; /* tells whether lifted data are ok with two more primes */
  int32_t S;
  int32_t *coef2; /* second witness coefficient */
  mpz_t *crt2;
  mpz_t *num2;
  mpz_t *den2;
  mpz_t *cden; /* lcm of the witness denominators */
  mpz_t pgden; /* denominator guessed from the previous polynomial */
  int32_t *coef3; /* third witness coefficient */
  mpz_t *crt3;
  mpz_t *num3;
  mpz_t *den3;
} data_lift_struct;

typedef data_lift_struct data_lift_t[1];


static inline void data_lift_init(data_lift_t dl,
                                  int32_t npol,
                                  int32_t *steps, int32_t nsteps){
  dl->npol = npol;
  dl->rr = 1;
  dl->lstart = 0;
  dl->nsteps = nsteps;
  dl->S = 0;
  mpz_init_set_ui(dl->pgden, 1);
  int32_t i;

  dl->steps = (int32_t *)calloc(nsteps, sizeof(int32_t));
  for(i = 0; i < nsteps; i++){
    dl->steps[i] = steps[i];
  }
  dl->cstep = 0;
  dl->lend = npol;
  dl->crt_mult = 0;
  dl->crt = (mpz_t *)malloc(sizeof(mpz_t) * dl->npol);
  for(int32_t i = 0; i < dl->npol; i++){
    mpz_init(dl->crt[i]);
  }
  dl->recon = 0;
  dl->coef =(int32_t *) calloc(npol, sizeof(int32_t) );

  dl->num = malloc(sizeof(mpz_t) * npol);
  for(i = 0; i < npol; i++){
    mpz_init(dl->num[i]);
  }
  dl->den = malloc(sizeof(mpz_t) * npol);
  for(i = 0; i < npol; i++){
    mpz_init(dl->den[i]);
  }
  mpz_init_set_ui(dl->gden, 1);
  mpz_init(dl->tmp);
  dl->start = 0;
  dl->end = npol;
  dl->check1 = calloc(npol, sizeof(int));
  dl->check2 = calloc(npol, sizeof(int));
  dl->coef2 = calloc(npol, sizeof(int32_t));
  dl->crt2 = malloc(sizeof(mpz_t) * npol);
  dl->num2 = malloc(sizeof(mpz_t) * npol);
  dl->den2 = malloc(sizeof(mpz_t) * npol);
  dl->cden = malloc(sizeof(mpz_t) * npol);
  dl->coef3 = calloc(npol, sizeof(int32_t));
  dl->crt3 = malloc(sizeof(mpz_t) * npol);
  dl->num3 = malloc(sizeof(mpz_t) * npol);
  dl->den3 = malloc(sizeof(mpz_t) * npol);
  for(i = 0; i < npol; i++){
    mpz_init(dl->crt2[i]);
    mpz_init(dl->num2[i]);
    mpz_init(dl->den2[i]);
    mpz_init(dl->cden[i]);
    mpz_init(dl->crt3[i]);
    mpz_init(dl->num3[i]);
    mpz_init(dl->den3[i]);
  }

}

static inline void data_lift_clear(data_lift_t dl){
  for(int32_t i = 0; i < dl->npol; i++){
    mpz_clear(dl->crt[i]);
  }
  free(dl->crt);

  free(dl->steps);
  free(dl->coef);

  for(int32_t i = 0; i < dl->npol; i++){
    mpz_clear(dl->num[i]);
  }
  free(dl->num);

  for(int32_t i = 0; i < dl->npol; i++){
    mpz_clear(dl->den[i]);
  }
  free(dl->den);

  mpz_clear(dl->gden);
  mpz_clear(dl->tmp);
  mpz_clear(dl->pgden);
  free(dl->check1);
  free(dl->check2);
  for(int32_t i = 0; i < dl->npol; i++){
    mpz_clear(dl->crt2[i]);
    mpz_clear(dl->num2[i]);
    mpz_clear(dl->den2[i]);
    mpz_clear(dl->cden[i]);
    mpz_clear(dl->crt3[i]);
    mpz_clear(dl->num3[i]);
    mpz_clear(dl->den3[i]);
  }
  free(dl->crt2);
  free(dl->num2);
  free(dl->den2);
  free(dl->cden);
  free(dl->coef2);
  free(dl->crt3);
  free(dl->num3);
  free(dl->den3);
  free(dl->coef3);

}


static inline void gb_modpoly_init(gb_modpoly_t modgbs,
                                   uint32_t alloc, int32_t *lens,
                                   bs_t *bs,
                                   int nv, uint32_t ld,
                                   int32_t *lm,
                                   md_t *st){
  modgbs->alloc = alloc;
  modgbs->nprimes = 0;
  modgbs->wide = 0;
  modgbs->primes = (mp_limb_t *)calloc(alloc, sizeof(mp_limb_t));
  modgbs->cf_64 = (mp_limb_t *)calloc(alloc, sizeof(mp_limb_t));
  modgbs->ld = ld;
  modgbs->nv = nv;
  modgbs->modpolys = (modpolys_t *)malloc(sizeof(modpolys_t) * ld);

  modgbs->ldm = (int32_t *)calloc(nv*ld, sizeof(int32_t));
  modgbs->bht = bs->ht;
  modgbs->hm = (hm_t **)malloc(modgbs->ld * sizeof(hm_t *));
  modgbs->lmps = (bl_t *)calloc(modgbs->ld, sizeof(bl_t));

  len_t idx;
  for(len_t i = 0; i < modgbs->ld; i++){
      idx = bs->lmps[i];
      modgbs->hm[i] = malloc((1+lens[i] + OFFSET)*sizeof(hm_t));
      for(len_t j = 0; j<lens[i]+1; j++){
          modgbs->hm[i][j+OFFSET] = bs->hm[idx][j+OFFSET];
      }
  }
  for(bl_t i = 0; i < modgbs->ld; i++){
      modgbs->lmps[i] = bs->lmps[i];
  }
  for(int32_t i = 0; i < ld; i++){
    for(int j = 0; j < nv; j++){
      modgbs->ldm[i*nv+j] = lm[i*nv+j];
    }
  }
  for(uint32_t i = 0; i < ld; i++){
    modgbs->modpolys[i]->len = lens[i];
    if(lens[i] > 0){
      modgbs->modpolys[i]->cf_32 = (uint32_t *)calloc((size_t)lens[i] * alloc,
                                                      sizeof(uint32_t));
      modgbs->modpolys[i]->cf_64 = NULL;
      modgbs->modpolys[i]->cf_qq = (mpz_t *)malloc(sizeof(mpz_t)*lens[i]);
    }
    else{
      modgbs->modpolys[i]->cf_32 = NULL;
      modgbs->modpolys[i]->cf_64 = NULL;
      modgbs->modpolys[i]->cf_qq = NULL;
    }
    for(uint32_t j = 0; j < lens[i]; j++){
      mpz_init(modgbs->modpolys[i]->cf_qq[j]);
    }
    mpz_init(modgbs->modpolys[i]->lm);
    mpz_set_ui(modgbs->modpolys[i]->lm, 1);
  }
}


/* Makes room for newalloc more primes. Polynomials whose modular data have
 * already been free-ed (cf_32 == NULL) are skipped. */
static inline void gb_modpoly_realloc(gb_modpoly_t modgbs,
                                      uint32_t newalloc){

  uint32_t oldalloc = modgbs->alloc;
  modgbs->alloc += newalloc;

  mp_limb_t *newprimes = (mp_limb_t *)realloc(modgbs->primes,
                                            modgbs->alloc * sizeof(mp_limb_t));

  if(newprimes == NULL){
    fprintf(ERRSTREAM, "Problem when reallocating modgbs (primes)\n");
    exit(1);
  }
  modgbs->primes = newprimes;
  for(uint32_t i = oldalloc; i < modgbs->alloc; i++){
    modgbs->primes[i] = 0;
  }

  mp_limb_t *ncf_64 = (mp_limb_t *)realloc(modgbs->cf_64,
                                         modgbs->alloc * sizeof(mp_limb_t));
  if(ncf_64 == NULL){
    fprintf(ERRSTREAM, "Problem when reallocating modgbs (cfs)\n");
    exit(1);
  }
  modgbs->cf_64 = ncf_64;
  for(uint32_t i = oldalloc; i < modgbs->alloc; i++){
    modgbs->cf_64[i] = 0;
  }
  for(uint32_t i = 0; i < modgbs->ld; i++){
    const size_t len = modgbs->modpolys[i]->len;
    if(modgbs->wide){
      /* images of already lifted polys are gone (cf_64 == NULL) */
      if(modgbs->modpolys[i]->cf_64 == NULL || len == 0){
        continue;
      }
      uint64_t *newcfs_pol = (uint64_t *)realloc(modgbs->modpolys[i]->cf_64,
                                                 len * modgbs->alloc * sizeof(uint64_t));
      if(newcfs_pol == NULL){
        fprintf(ERRSTREAM, "Problem when reallocating modgbs (cfs_pol)\n");
        exit(1);
      }
      /* prime-major storage: the new primes are appended at the end */
      memset(newcfs_pol + len * oldalloc, 0,
             len * (modgbs->alloc - oldalloc) * sizeof(uint64_t));
      modgbs->modpolys[i]->cf_64 = newcfs_pol;
    }
    else{
      if(modgbs->modpolys[i]->cf_32 == NULL || len == 0){
        continue;
      }
      uint32_t *newcfs_pol = (uint32_t *)realloc(modgbs->modpolys[i]->cf_32,
                                                 len * modgbs->alloc * sizeof(uint32_t));
      if(newcfs_pol == NULL){
        fprintf(ERRSTREAM, "Problem when reallocating modgbs (cfs_pol)\n");
        exit(1);
      }
      /* prime-major storage: the new primes are appended at the end */
      memset(newcfs_pol + len * oldalloc, 0,
             len * (modgbs->alloc - oldalloc) * sizeof(uint32_t));
      modgbs->modpolys[i]->cf_32 = newcfs_pol;
    }
  }
}


static inline void display_gbmodpoly_cf_32(FILE *file,
                                     gb_modpoly_t modgbs){
  fprintf(file, "alloc = %d\n", modgbs->alloc);
  fprintf(file, "nprimes = %d\n", modgbs->nprimes);
  fprintf(file, "primes = [");
  for(uint32_t i = 0; i < modgbs->alloc-1; i++){
    fprintf(file, "%lu, ", (unsigned long)modgbs->primes[i]);
  }
  fprintf(file, "%lu]\n", (unsigned long)modgbs->primes[modgbs->alloc -1]);
  fprintf(file, "numpolys = %d\n", modgbs->ld);
  fprintf(file, "[\n");
  for(uint32_t i = 0; i < modgbs->ld; i++){
    uint32_t len = modgbs->modpolys[i]->len;
    fprintf(file, "[%d, ", len);
    if((modgbs->wide ?
        (const void *)modgbs->modpolys[i]->cf_64 :
        (const void *)modgbs->modpolys[i]->cf_32) == NULL){
      fprintf(file, "(lifted)],\n");
      continue;
    }
    for(uint32_t j = 0; j < len; j++){
      fprintf(file, "[");
      for(uint32_t k = 0; k < modgbs->alloc-1; k++){
        fprintf(file, "%lu, ", (unsigned long)(modgbs->wide ?
                    MODCF64(modgbs->modpolys[i], j, k) :
                    (uint64_t)MODCF(modgbs->modpolys[i], j, k)));
      }
      if(j < len - 1){
        fprintf(file, "%lu], ", (unsigned long)(modgbs->wide ?
                    MODCF64(modgbs->modpolys[i], j, modgbs->alloc-1) :
                    (uint64_t)MODCF(modgbs->modpolys[i], j, modgbs->alloc-1)));
      }
      else{
        fprintf(file, "%lu]\n", (unsigned long)(modgbs->wide ?
                    MODCF64(modgbs->modpolys[i], j, modgbs->alloc-1) :
                    (uint64_t)MODCF(modgbs->modpolys[i], j, modgbs->alloc-1)));
      }
    }
    fprintf(file, "],\n");
  }
  fprintf(file, "]:\n");
}

static inline void display_modpoly(FILE *file,
                                   gb_modpoly_t modgbs,
                                   int32_t pos,
                                   data_gens_ff_t *gens){

  if(modgbs->modpolys[pos]->len == 0){
    display_monomial_single(file, gens, pos, &modgbs->ldm);
    return;
  }
  if(mpz_cmp_ui(modgbs->modpolys[pos]->lm, 1) != 0){
    mpz_out_str_10(file, modgbs->modpolys[pos]->lm);
    fprintf(file, "*");
  }
  display_monomial_single(file, gens, pos, &modgbs->ldm);

  len_t i, k;
  hm_t *hm    = NULL;
  /* modgbs->hm[pos] is the copy of the monomials of the pos-th element
   * (see gb_modpoly_init), hence it is indexed by pos, not by lmps[pos] */
  hm  = modgbs->hm[pos]+OFFSET;

  ht_t *ht = modgbs->bht;
  const len_t nv  = ht->nv;
  const len_t ebl = ht->ebl;
  const len_t evl = ht->evl;

  int *evi    =   (int *)malloc((unsigned long)(ht->nv + 1) * sizeof(int));
  if (ebl == 0) {
    for (i = 1; i < evl; ++i) {
      evi[i-1]    =   i;
    }
  } else {
    for (i = 1; i < ebl; ++i) {
      evi[i-1]    =   i;
    }
    for (i = ebl+1; i < evl; ++i) {
      evi[i-2]    =   i;
    }
  }
  len_t len = modgbs->modpolys[pos]->len + 1;
  /* once lifted, all coefficients are integers (common denominator is
   * stored in lm), cf_qq[i] is the coefficient of index i; the terms are
   * written through a buffer */
  mpz_t *cf = modgbs->modpolys[pos]->cf_qq;
  obuf_t ob = {file, (char *)malloc(1 << 16), 0, 1 << 16};
  for(i = modgbs->modpolys[pos]->len; i > 0 ; i--){
    const len_t c = i - 1;
    const int sg = mpz_sgn(cf[c]);
    const exp_t * const e = ht->ev[hm[len-i]];
    int deg = 0;
    for (k = 0; k < nv; ++k) {
      deg += e[evi[k]];
    }
    if(sg > 0){
      ob_char(&ob, '+');
    }
    if(c == 0 && deg == 0){
      /* the constant term */
      ob_mpz(&ob, cf[c]);
      continue;
    }
    const int one = mpz_cmpabs_ui(cf[c], 1) == 0;
    //coef is -1
    if(one && sg < 0){
      ob_char(&ob, '-');
    }
    //coef is neither 1 nor -1
    if(!one){
      ob_mpz(&ob, cf[c]);
      ob_char(&ob, '*');
    }
    if(sg != 0){
      int star = 0;
      for (k = 0; k < nv; ++k) {
        if (e[evi[k]] == 0) {
          continue;
        }
        if(star == 1){
          ob_char(&ob, '*');
        }
        ob_put(&ob, gens->vnames[k], strlen(gens->vnames[k]));
        if (e[evi[k]] > 1) {
          ob_char(&ob, '^');
          ob_u64(&ob, e[evi[k]]);
        }
        star = 1;
      }
    }
  }
  ob_flush(&ob);
  free(ob.b);

  free(evi);
}

static inline void display_gbmodpoly_cf_qq(FILE *file,
                                           gb_modpoly_t modgbs,
                                           data_gens_ff_t *gens){
  int32_t p = modgbs->ld ;
  if(p==0){
    fprintf(file, "[0]:\n");
    return;
  }
  fprintf(file, "[");
  for(int i = 0; i < p-1; i++){
    display_modpoly(file, modgbs, i, gens);
    fprintf(file, ", \n");
  }
  display_modpoly(file, modgbs, p-1, gens);
  fprintf(file, "]:\n");
}

static inline void display_lm_gbmodpoly_cf_qq(FILE *file,
                                              gb_modpoly_t modgbs,
                                              data_gens_ff_t *gens){
  int32_t p = modgbs->ld ;

  if(p==0){
    fprintf(file, "[0]:\n");
    return;
  }

  fprintf(file, "[");
  for(int i = 0; i < p-1; i++){
    display_monomial_single(file, gens, i, &modgbs->ldm);
    fprintf(file, ", \n");
  }
  display_monomial_single(file, gens, p-1, &modgbs->ldm);
  fprintf(file, "]:\n");
}

/* frees everything but the hash table modgbs->bht */
static inline void gb_modpoly_without_hash_table_clear(gb_modpoly_t modgbs){
  free(modgbs->primes);
  modgbs->primes = NULL;
  free(modgbs->cf_64);
  modgbs->cf_64 = NULL;
  free(modgbs->ldm);
  modgbs->ldm = NULL;
  for(uint32_t i = 0; i < modgbs->ld; i++){
    /* cf_32 is NULL if already free-ed after lifting */
    free(modgbs->modpolys[i]->cf_32);
    modgbs->modpolys[i]->cf_32 = NULL;
    /* same for the wide images (only allocated when wide, cf_32 is
     * already free-ed then) */
    free(modgbs->modpolys[i]->cf_64);
    modgbs->modpolys[i]->cf_64 = NULL;
    for(uint32_t j = 0; j < modgbs->modpolys[i]->len; j++){
      mpz_clear(modgbs->modpolys[i]->cf_qq[j]);
    }
    mpz_clear(modgbs->modpolys[i]->lm);
    free(modgbs->modpolys[i]->cf_qq);
    modgbs->modpolys[i]->cf_qq = NULL;
    free(modgbs->hm[i]);
  }
  free(modgbs->lmps);
  modgbs->lmps = NULL;
  free(modgbs->hm);
  modgbs->hm = NULL;
  free(modgbs->modpolys);
  modgbs->modpolys = NULL;
}

static inline void gb_modpoly_clear(gb_modpoly_t modgbs){
  gb_modpoly_without_hash_table_clear(modgbs);
  full_free_hash_table(&modgbs->bht);
}

static inline int32_t degree(int32_t *mon, int nv){
  int32_t deg = 0;
  for(int i = 0; i < nv; i++){
    deg += mon[i];
  }
  return deg;
}

/*
  bexp_lm is a list of monomials involving nv variables of increasing degrees
  return an array of length *nb containing at the i-th position the number of
  monomials of the i-th degree (hence nb is the number of different degrees).
 */
static inline int32_t *array_nbdegrees(int32_t *bexp_lm, int len,
                                     int nv, int * nb){
  *nb = 1;
  int32_t deg = degree(bexp_lm, nv);

  for(int32_t i = 1; i < len; i++){
    int32_t newdeg = degree(bexp_lm + i * nv, nv);

    if(deg != newdeg){
      (*nb)++;
      deg = newdeg;
    }
  }
  int32_t *ldeg = calloc(*nb, sizeof(int32_t));
  deg = degree(bexp_lm, nv);
  ldeg[0] = 1;
  int32_t i = 0, j = 1;
  while(j < len){
    int32_t newdeg = degree(bexp_lm + j * nv, nv);
    if(deg == newdeg){
      ldeg[i]++;
    }
    else{
      i++;
      ldeg[i] = 1;
      deg = newdeg;
    }
    j++;
  }
  return ldeg;
}

static inline int grevlex_is_less_than(int nv, int32_t* m1, int32_t *m2){
  int32_t deg1 = 0, deg2 = 0;
  for(int i = 0; i < nv; i++){
    deg1 += m1[i];
  }
  for(int i = 0; i < nv; i++){
    deg2 += m2[i];
  }
  if(deg1 < deg2){
    return 1;
  }
  if(deg1 > deg2){
    return 0;
  }
  for(int i = 0; i < nv; i++){
    if(m1[i] < m2[i]){
      return 0;
    }
  }
  return 1;
}

static inline int32_t compute_length(int32_t *mon, int nv,
                                     int32_t *basis, int dquot){

  for(int i = dquot - 1; i >= 0; i--){
    if(!grevlex_is_less_than(nv, mon, basis + i * nv)){
      return i + 1;
    }
  }
  return 0;
}

/*
 * bexp_lm is an array of len  monomials with nv variables
 * basis is an array of dquot monomials with nv variables
 * (basis of some quotient defined with bexp_lm, all with grevlex order)
 * basis is sorted increasingly, as for bexp_lm
 *
 * returns an array of length len containing the maximum possible lengths
 * of polynomials with leading terms in bexp_lm (length excluding the
 * leading term).
 */
static inline int32_t *array_of_lengths(int32_t *bexp_lm, len_t len,
        const bs_t * const bs, int nv){
  int32_t *lens = calloc(len, sizeof(int32_t));
  for(len_t i = 0; i < len; i++){
    len_t idx = bs->lmps[i];
    lens[i] = bs->hm[idx][LENGTH]-1;
  }
  return lens;
}

/* returns 0 in case of failure else returns 1 */
static inline int modpgbs_set(gb_modpoly_t modgbs,
                              const bs_t *bs, const ht_t * const ht,
                              const uint64_t fc,
                              int32_t start, const long elim){
  if(modgbs->nprimes >= modgbs->alloc-1){
    fprintf(ERRSTREAM, "Not enough space in modgbs\n");
    exit(1);
  }
  modgbs->primes[modgbs->nprimes] = fc;

  if(fc > UINT32_MAX && !modgbs->wide){
    /* first image at a 61-bit prime: the residues read so far (the image
     * modulo the learning prime) are widened once, cf_32 is freed and all
     * further images are stored (and read) wide */
    modgbs->wide = 1;
    for(uint32_t i = 0; i < modgbs->ld; i++){
      if(modgbs->modpolys[i]->cf_32 == NULL){
        /* already lifted, or no terms: nothing to migrate */
        continue;
      }
      const size_t len = modgbs->modpolys[i]->len;
      uint64_t *cfs64 = (uint64_t *)malloc(len * modgbs->alloc
                                           * sizeof(uint64_t));
      if(cfs64 == NULL){
        fprintf(ERRSTREAM, "Problem when allocating modgbs (wide cfs)\n");
        exit(1);
      }
      for(uint32_t j = 0; j < modgbs->alloc; j++){
        for(size_t k = 0; k < len; k++){
          cfs64[(size_t)j * len + k] =
            (uint64_t)modgbs->modpolys[i]->cf_32[(size_t)j * len + k];
        }
      }
      free(modgbs->modpolys[i]->cf_32);
      modgbs->modpolys[i]->cf_32 = NULL;
      modgbs->modpolys[i]->cf_64 = cfs64;
    }
  }

  len_t i, j, idx;

  /*****************************************************
   * It should be checked if modbgs->ht needs to be updated as well as
   * modgbs->lmps
   * *****************************************************/

  len_t len   = 0;
  /* hm_t *hm    = NULL; */

  for(i = start; i < modgbs->ld; i++){
    idx = bs->lmps[i];
    if (bs->hm[idx] == NULL) {
      fprintf(ERRSTREAM, " poly is 0\n");
      exit(1);
    } else {
      /* hm  = bs->hm[idx]+OFFSET; */
      len = bs->hm[idx][LENGTH];
    }
    if(modgbs->wide){
      if(modgbs->modpolys[i]->cf_64 == NULL){
        /* already lifted, modular data are no longer needed */
        continue;
      }
      int32_t bc = modgbs->modpolys[i]->len - 1;
      if(fc > UINT32_MAX){
        const cf64_t * const cfs = bs->cf_64[bs->hm[idx][COEFFS]];
        for (j = 1; j < len; ++j) {
          MODCF64(modgbs->modpolys[i], bc, modgbs->nprimes) = cfs[j];
          bc--;
        }
      }
      else{
        /* an image at a ~31-bit prime after images at 61-bit primes */
        const uint32_t * const cfs = bs->cf_32[bs->hm[idx][COEFFS]];
        for (j = 1; j < len; ++j) {
          MODCF64(modgbs->modpolys[i], bc, modgbs->nprimes) = cfs[j];
          bc--;
        }
      }
    }
    else{
      if(modgbs->modpolys[i]->cf_32 == NULL){
        /* already lifted, modular data are no longer needed */
        continue;
      }
      int32_t bc = modgbs->modpolys[i]->len - 1;
      const uint32_t * const cfs = bs->cf_32[bs->hm[idx][COEFFS]];
      for (j = 1; j < len; ++j) {
        MODCF(modgbs->modpolys[i], bc, modgbs->nprimes) = cfs[j];
        bc--;
      }
   }
  }

  modgbs->nprimes++;

  return 1;
}

static inline int32_t maxbitsize_gens(data_gens_ff_t *gens, len_t ngens){
  if(gens->field_char != 0){
    return -1;
  }
  const int32_t *lens = gens->lens;
  mpz_t **cfs = gens->mpz_cfs;
  int32_t off = 0;
  int32_t mbs = 0;
  for(int32_t i = 0; i < ngens; i++){
    for(int32_t j = off; j < off + lens[i]; ++j){
      mbs = MAX(mbs,
                mpz_sizeinbase(*(cfs[2*j]), 2) + mpz_sizeinbase(*(cfs[2*j+1]), 2));
    }
    off += lens[i];
  }
  return mbs;
}

static inline int32_t maxbitsize_generators(bs_t *bs){
    int32_t max = 0;
    int32_t ngens = bs->ld;
    for(int32_t i = 0; i < ngens; i++){
        int32_t len = bs->hm[i][LENGTH];
        mpz_t *cfs = bs->cf_qq[bs->hm[i][COEFFS]];
        for(int32_t j = 0; j < len; j++){
            max = MAX(max,
                    mpz_sizeinbase(cfs[j], 2)
                    );
        }
    }
    return max;
}

/* returns the index of the largest polynomial in the basis of the elimination
 * ideal */
static inline int32_t compute_num_gb(int32_t *bexp_lm, int32_t len, int nv, int nev){
  if(nev){
    for(int32_t i = 0; i < len; i++){
      for(int32_t j = 0; j < nev; j++){
        if(bexp_lm[i*nv+j]){
          return i;
        }
      }
    }
    return len;
  }
  return len;
}

/* whether the traces over the rationals are learned guided by a
   signature-based run, see the option --signature-f4 3 and
   guided_trace_learning() */
int32_t msolve_guided_learning = 0;

/* the number of primes below 2^31 at which a trace is applied at once,
   see core_f4_multi() */
#define MULTI_PRIMES 8
/* whether traces over the rationals are applied at several primes at
   once */
int32_t msolve_multi_primes = 1;

/* returns 1 if the reduced Groebner bases a and b over the same prime
   field, with the same basis hash table, are equal */
static int same_reduced_basis(const bs_t * const a, const bs_t * const b){
  if(a->lml != b->lml || a->constant != b->constant){
    return 0;
  }
  for(len_t i = 0; i < a->lml; i++){
    const hm_t *ha = a->hm[a->lmps[i]];
    const hm_t *hb = b->hm[b->lmps[i]];
    const len_t len = ha[LENGTH];
    if(hb[LENGTH] != len
       || memcmp(ha + OFFSET, hb + OFFSET, (size_t)len * sizeof(hm_t)) != 0
       || memcmp(a->cf_32[ha[COEFFS]], b->cf_32[hb[COEFFS]],
                 (size_t)len * sizeof(cf32_t)) != 0){
      return 0;
    }
  }
  return 1;
}

/*
  Learns the trace modulo fc guided by a signature-based run: the
  signature-based F4 computes the reduced Groebner basis G modulo fc and
  shows which S-pairs do not reduce to zero. F4 then learns its trace,
  selecting the S-pairs with the lcms of these first. The other S-pairs of
  a degree d are dropped once each lead term of G of degree at most d is
  divisible by a lead term of the basis of F4: the basis of F4 lies in the
  ideal, so all elements of the ideal with lead terms of degree at most d
  reduce to zero by it. F4 stops once all lead terms of G are, then its
  basis is a Groebner basis. Thus F4 reduces few rows to zero. As a check
  the reduced basis of F4 must be G, otherwise, or if the signature-based
  run hands over to F4, the trace is learned by F4 as usual. The trace is
  applied as any trace learned by F4, which is cheaper than the
  application of the record of the signature-based run. Where the
  signature-based run is expensive this costs more than learning the
  trace by F4, so it is optional.
 */
static bs_t *guided_trace_learning(bs_t *bs_qq, md_t *st, const int32_t fc,
                                   const int info_level, int32_t *errp){
  neogb_sig_f4 = 1;
  bs_t *ref = core_gba(bs_qq, st, errp, fc);
  neogb_sig_f4 = -1;
  if(*errp == 0 && ref != NULL && (st->tr == NULL || st->tr->sgr == NULL)){
    /* the signature-based run has handed over to F4, which has learned
       the trace */
    return ref;
  }
  bs_t *bs = NULL;
  if(*errp == 0 && ref != NULL && !ref->constant){
    st->glcm = sig_record_lead_terms(st->tr->sgr, bs_qq->ht, &(st->nglcm));
    st->ngref = ref->lml;
    st->gref  = (hm_t *)malloc((size_t)(ref->lml + 1) * sizeof(hm_t));
    st->gcov  = (int8_t *)calloc((size_t)ref->lml + 1, sizeof(int8_t));
    st->gchk  = 0;
    for(len_t i = 0; i < ref->lml; i++){
      st->gref[i] = ref->hm[ref->lmps[i]][OFFSET];
    }
    free_trace(&(st->tr));
    st->trace_level = NO_TRACER;
    bs = core_gba(bs_qq, st, errp, fc);
    free(st->glcm);
    free(st->gref);
    free(st->gcov);
    st->glcm  = NULL;
    st->nglcm = 0;
    st->gref  = NULL;
    st->ngref = 0;
    st->gcov  = NULL;
    if(*errp == 0 && bs != NULL && same_reduced_basis(bs, ref)){
      free_basis_without_hash_table(&ref);
      return bs;
    }
  }
  if(info_level){
    fprintf(VERBSTREAM, "trace learned without guide\n");
  }
  if(bs != NULL){
    free_basis_without_hash_table(&bs);
  }
  if(ref != NULL){
    free_basis_without_hash_table(&ref);
  }
  if(st->tr != NULL){
    free_trace(&(st->tr));
  }
  st->trace_level = NO_TRACER;
  *errp = 0;
  return core_gba(bs_qq, st, errp, fc);
}

/*
  The trace of a signature-based run is applied at each prime more slowly
  than a trace learned by F4, but is learned faster. Where many primes are
  needed, see SIG_RELEARN, a trace is learned by F4 at the prime fc and
  replaces it from then on. The new trace is taken only if the leading
  monomials of its basis are those of the basis learned first, nlm
  exponent vectors of nv variables in lmo. Returns 1 if it is taken.
 */
static int relearn_by_f4(bs_t *bs_qq, md_t *st, const uint32_t fc,
                         const int32_t *lmo, const int32_t nlm){
  trace_t *old = st->tr;
  const int32_t osig  = neogb_sig_f4;
  const int ofr       = st->f4_qq_round;
  const int32_t oinfo = st->info_level;
  st->tr          = NULL;
  st->trace_level = NO_TRACER;
  st->f4_qq_round = 1;
  st->info_level  = 0;
  neogb_sig_f4    = 0;
  int32_t err = 0;
  bs_t *bs = core_gba(bs_qq, st, &err, fc);
  neogb_sig_f4    = osig;
  st->f4_qq_round = ofr;
  st->info_level  = oinfo;
  int ok = err == 0 && bs != NULL && st->tr != NULL && st->tr->sgr == NULL
    && (int32_t)bs->lml == nlm;
  if(ok){
    int32_t *lm = get_lm_from_bs(bs, bs->ht);
    ok = memcmp(lm, lmo, (size_t)nlm * bs->ht->nv * sizeof(int32_t)) == 0;
    free(lm);
  }
  if(bs != NULL){
    free_basis_without_hash_table(&bs);
  }
  if(ok){
    free_trace(&old);
  }
  else{
    if(st->tr != NULL){
      free_trace(&(st->tr));
    }
    st->tr = old;
  }
  st->trace_level = APPLY_TRACER;
  return ok;
}

static int32_t gb_modular_trace_learning(gb_modpoly_t modgbs,
                                           int32_t *mgb,
                                           int32_t *num_gb,
                                           int32_t **leadmons,
                                           trace_t *trace,
                                           bs_t *bs_qq,
                                           md_t *st,
                                           const int32_t fc,
                                           int info_level,
                                           int print_gb,
                                           int truncate_lifting,
                                           int32_t start,
                                           int32_t maxbitsize,
                                           int *success)
{
    double ca0/*, rt*/;
    ca0 = realtime();

    bs_t *bs = NULL;
    int32_t err     = 0;
    st->f4_qq_round = 1;
    if (msolve_guided_learning && neogb_sig_f4 < 0
        && st->trace_level == NO_TRACER
        && st->laopt < 40 && st->reduce_gb == 1 && st->nev == 0
        && st->mo == 0 && !bs_qq->ht->boolean
        && fc > ((uint32_t)1 << 16) && fc < ((uint32_t)1 << 31)){
      bs = guided_trace_learning(bs_qq, st, fc, info_level, &err);
    }
    else{
      bs = core_gba(bs_qq, st, &err, fc);
    }
    if (err) {
      fprintf(ERRSTREAM,"Problem with F4, stopped computation.\n");
      exit(1);
    }

    /* rt = realtime()-ca0; */
    st->learning_rtime = realtime()-ca0;

    ht_t *bht = bs->ht;

    /* Leading monomials from Grobner basis */
    int32_t *bexp_lm = get_lm_from_bs(bs, bht);
    /* may be non NULL if the computation has been restarted */
    free(leadmons[0]);
    leadmons[0] = bexp_lm;

    int32_t len = bs->lml;
    num_gb[0] = compute_num_gb(bexp_lm, len, bht->nv, st->nev);
    int32_t *bexp_lm2 = NULL;
    if(st->nev){
      bexp_lm2 = calloc(num_gb[0]*(bht->nv - st->nev), sizeof(int32_t));
      for(int32_t i = 0; i < num_gb[0]; i++){
        for(int j = 0; j < bht->nv - st->nev; j++){
          bexp_lm2[i*(bht->nv - st->nev) + j] = bexp_lm[i*bht->nv + st->nev + j];
        }
      }
      leadmons[0] = bexp_lm2;
    }
    /************************************************/

    /**
    * int32_t *lmb;
    * if(st->nev){
    *   lmb = NULL; //monomial_basis_enlarged(num_gb[0], bht->nv - st->nev,  bexp_lm2, &dquot);
    * }
    * else{
    *   lmb = NULL; //monomial_basis_enlarged(num_gb[0], bht->nv,  bexp_lm, &dquot);
    * }
    */

    /************************************************/

    int32_t *lens = array_of_lengths(leadmons[0], num_gb[0], bs, bht->nv - st->nev);

    if(truncate_lifting != 0 && truncate_lifting < num_gb[0]){
      gb_modpoly_init(modgbs, 2, lens, bs, bht->nv - st->nev, truncate_lifting, leadmons[0], st);
    }
    else{
      gb_modpoly_init(modgbs, 2, lens, bs, bht->nv - st->nev, num_gb[0], leadmons[0], st);
    }
    free(lens);
    modpgbs_set(modgbs, bs, bht, fc, start, st->nev);
    int is_empty = 0;
    if(bs->lml == 1){
        if(info_level){
            fprintf(VERBSTREAM, "Grobner basis has a single element\n");
        }
        is_empty = 1;
        for(int i = 0; i < bht->nv; i++){
            if(bexp_lm[i]!=0){
                is_empty = 0;
            }
        }
        if(is_empty){
            if(bexp_lm2 != NULL){
              free(bexp_lm);
            }
            free_basis_without_hash_table(&(bs));
            return is_empty;
        }
    }

    /**************************************************/

    if(bexp_lm2 != NULL){
      /* leadmons[0] is bexp_lm2, bexp_lm is no longer needed */
      free(bexp_lm);
    }
    free_basis_without_hash_table(&(bs));
    return is_empty;
}


static void gb_modular_trace_application(gb_modpoly_t modgbs,
                                         int32_t *mgb,
                                         int32_t *num_gb,
                                         int32_t **leadmons_ori,
                                         int32_t **leadmons_current,
                                         trace_t **btrace,
                                         bs_t *bs_qq,
                                         md_t *st,
                                         int info_level,
                                         uint64_t fc,
                                         int32_t start,
                                         double *stf4,
                                         int *bad_primes,
                                         const int use_pre,
                                         bs_t *pre)
{

  double rt = realtime();
  st->info_level  = 0;
  st->f4_qq_round = 2;
  /* the primes are handled one after the other, so they all work on the
   * hash table of bs_qq, which already holds nearly all monomials */
  st->share_ht = 1;
  /* tracing phase */

  /* F4 and FGLM are run using a single thread */
  /* st->nthrds is reset to its original value afterwards */
  /*at the moment multi-threading is not supprted here*/
  memset(bad_primes, 0, (unsigned long)st->nprimes * sizeof(int));

  /* current load of the hash table */
  /* hl_t eld = bs_qq->ht->eld; */
  bs_t *bs = NULL;
  int32_t error = 0;
  if(use_pre){
    /* the basis is computed already together with other primes, see
       core_f4_multi(), NULL if the prime is bad */
    bs = pre;
  }
  else{
    bs = core_gba(bs_qq, st, &error, fc);
  }
  *stf4 = realtime()-rt;
  /* core_gba returns NULL (and has already free-ed its data) when the
   * prime has been detected as bad while applying the tracer */
  if (bs == NULL || error) {
      bad_primes[0] = 1;
      if (bs != NULL) {
          free_basis_and_only_local_hash_table_data(&bs);
      }
      return;
  }
  ht_t **bht = &(bs->ht);
  for(len_t i = 0; i < modgbs->ld; i++){
        if(modgbs->modpolys[i]->len < bs->hm[bs->lmps[i]][LENGTH]-1){
            bad_primes[0] = 2;
            free_basis_and_only_local_hash_table_data(&bs);
            return;
        }
        if(modgbs->modpolys[i]->len > bs->hm[bs->lmps[i]][LENGTH]-1){
            bad_primes[0]=1;
            free_basis_and_only_local_hash_table_data(&bs);
            return;
        }
  }
  int32_t lml = bs->lml;
  if (st->nev > 0) {
      int32_t j = 0;
      for (len_t i = 0; i < bs->lml; ++i) {
          if ((*bht)->ev[bs->hm[bs->lmps[i]][OFFSET]][0] == 0) {
              bs->lm[j]   = bs->lm[i];
              bs->lmps[j] = bs->lmps[i];
              ++j;
          }
      }
      lml = j;
  }
  if (lml != num_gb[0]) {
      if (bs != NULL) {
          bad_primes[0] = 2;
          free_basis_and_only_local_hash_table_data(&bs);
      }
      return;
  }

  if(st->nev){
    get_lm_from_bs_trace_elim(bs, bht[0], leadmons_current[0], num_gb[0]);
  }
  else{
    get_lm_from_bs_trace(bs, bht[0], leadmons_current[0]);
  }

  if(!equal_staircase(leadmons_current[0], leadmons_ori[0],
                      num_gb[0], num_gb[0], bht[0]->nv - st->nev)){
    bad_primes[0] = 1;
  }

  if(!bad_primes[0] && bs != NULL){
    /* copy of data for multi-mod computation */
    modpgbs_set(modgbs, bs, bht[0], fc, start, st->nev);
  }

  if (bs != NULL) {
    free_basis_and_only_local_hash_table_data(&bs);
  }

}

/* three witness coefficients are chosen per polynomial: the last nonzero
   one and the nonzero ones found scanning downwards from 2*len/3 and
   len/2 */
static inline void choose_coef_to_lift(gb_modpoly_t modgbs, data_lift_t dlift){
  uint32_t ld = modgbs->ld;
  for(int32_t i = 0; i < ld; i++){
    len_t len = modgbs->modpolys[i]->len;
    len_t d = len - 1;
    while(d > 0 && len > 0){
      if(modgbs->wide ?
         MODCF64(modgbs->modpolys[i], d, 0) :
         MODCF(modgbs->modpolys[i], d, 0)){
        dlift->coef[i] = d;
        break;
      }
      else{
        d--;
      }
    }
    len_t d2 = 2 * len / 3;
    while(d2 > 0 && !(modgbs->wide ?
                      MODCF64(modgbs->modpolys[i], d2, 0) :
                      MODCF(modgbs->modpolys[i], d2, 0))){
      d2--;
    }
    dlift->coef2[i] = d2;
    len_t d3 = len / 2;
    while(d3 > 0 && !(modgbs->wide ?
                      MODCF64(modgbs->modpolys[i], d3, 0) :
                      MODCF(modgbs->modpolys[i], d3, 0))){
      d3--;
    }
    dlift->coef3[i] = d3;
  }
}


/* Incremental CRT (called once FLINT multi_CRT has been called) */
/* mod is the current modulus */
static inline void incremental_dlift_crt(gb_modpoly_t modgbs, data_lift_t dlift,
                                         int32_t *coef, mpz_t *mod_p, mpz_t *prod_p,
                                         int thrds){

  /* all primes are assumed to be good primes */
  mpz_mul_ui(prod_p[0], mod_p[0], modgbs->primes[modgbs->nprimes - 1 ]);
  for(int32_t k = dlift->lstart; k <= dlift->lend; k++){
      if(modgbs->modpolys[k]->len){
        uint64_t c = modgbs->wide ?
          MODCF64(modgbs->modpolys[k], coef[k], modgbs->nprimes  - 1) :
          MODCF(modgbs->modpolys[k], coef[k], modgbs->nprimes  - 1);
        mpz_CRT_ui(dlift->crt[k], dlift->crt[k], mod_p[0],
                   c, modgbs->primes[modgbs->nprimes - 1 ],
                   prod_p[0], dlift->tmp, 1);
      }
  }
  mpz_set(mod_p[0], prod_p[0]);
}


/* Incremental CRT on the whole array of witness coefficients */
/* mod is the current modulus */
static inline void incremental_dlift_crt_full(gb_modpoly_t modgbs, data_lift_t dl,
                                              int32_t *coef, mpz_t mod_p, mpz_t prod_p,
                                              int thrds){

  mp_limb_t newprime = modgbs->primes[modgbs->nprimes - 1 ];
  /* all primes are assumed to be good primes */
  mpz_mul_ui(prod_p, mod_p, newprime);
  for(int32_t k = dl->S; k < dl->end; k++){
      if(modgbs->modpolys[k]->len){
        uint64_t c = modgbs->wide ?
          MODCF64(modgbs->modpolys[k], coef[k], modgbs->nprimes  - 1) :
          MODCF(modgbs->modpolys[k], coef[k], modgbs->nprimes  - 1);
        mpz_CRT_ui(dl->crt[k], dl->crt[k], mod_p,
                   c, newprime, prod_p, dl->tmp, 1);
        c = modgbs->wide ?
          MODCF64(modgbs->modpolys[k], dl->coef2[k], modgbs->nprimes  - 1) :
          MODCF(modgbs->modpolys[k], dl->coef2[k], modgbs->nprimes  - 1);
        mpz_CRT_ui(dl->crt2[k], dl->crt2[k], mod_p,
                   c, newprime, prod_p, dl->tmp, 1);
        c = modgbs->wide ?
          MODCF64(modgbs->modpolys[k], dl->coef3[k], modgbs->nprimes  - 1) :
          MODCF(modgbs->modpolys[k], dl->coef3[k], modgbs->nprimes  - 1);
        mpz_CRT_ui(dl->crt3[k], dl->crt3[k], mod_p,
                   c, newprime, prod_p, dl->tmp, 1);
      }
  }
  mpz_set(mod_p, prod_p);
}


/* balanced bounds, used when no denominator is guessed; RR_MARGIN bits of
   slack are kept (N*D <= mod_p / 2^(1+RR_MARGIN)) so that spurious
   reconstructions become unlikely */
static inline void set_recdata(rrec_data_t rd, mpz_t mod_p){
  mpz_fdiv_q_2exp(rd->N, mod_p, 1 + RR_MARGIN);
  mpz_sqrt(rd->N, rd->N);
  mpz_set(rd->D, rd->N);
}

/* bounds used to reconstruct u * gden, where gden is a guessed denominator:
   the numerator is then expected to be about gden times the remaining
   denominator, so that with M = mod_p / 2^(1+RR_MARGIN), the bounds are
   D = sqrt(M / gden) and N = M / D */
static inline void set_recdata_guess(rrec_data_t rd, const mpz_t mod_p,
                                     const mpz_t gden){
  mpz_fdiv_q_2exp(rd->N, mod_p, 1 + RR_MARGIN);
  mpz_fdiv_q(rd->D, rd->N, gden);
  mpz_sqrt(rd->D, rd->D);
  if(mpz_sgn(rd->D) == 0){
    mpz_set_ui(rd->D, 1);
  }
  mpz_fdiv_q(rd->N, rd->N, rd->D);
}

/*
  For the polynomials of index in [start, end) whose witness coefficient has
  been lifted and checked (check1 > 1), lifts all coefficients by CRT
  followed by rational reconstruction, and stores the integer numerators in
  cf_qq and the common denominator in lm.

  The CRT images are only temporaries: each coefficient is lifted with
  multi-CRT into a single scratch integer which is immediately rationally
  reconstructed, so that no array of CRT images is kept alive in memory.
  Denominators are also kept in a scratch array, the size of the longest
  polynomial handled here.

  returns the index of the poly if some of its coeff could not be lifted
  else returns end
 */
static inline int crt_ratrecon_lift_modgbs(gb_modpoly_t modgbs, data_lift_t dl,
                                           int32_t start, int32_t end,
                                           mpz_t mod_p, rrec_data_t rd1, rrec_data_t rd2,
                                           double *st_crt, double *st_rrec){

  modpolys_t *polys = modgbs->modpolys;

  /* scratch data */
  uint32_t maxlen = 0;
  for(int32_t k = start; k < end; k++){
    if(dl->check1[k] > 1 && polys[k]->len > maxlen){
      maxlen = polys[k]->len;
    }
  }
  mpz_t rden, zz;
  mpz_init(rden);
  mpz_init(zz);
  mpz_t *dens = NULL;
  if(maxlen > 0){
    dens = (mpz_t *)malloc(sizeof(mpz_t) * maxlen);
    for(uint32_t l = 0; l < maxlen; l++){
      mpz_init(dens[l]);
    }
  }

  /* Data needed by multi CRT functions */
  fmpz_comb_t comb;
  fmpz_comb_temp_t comb_temp;
  fmpz_t y;
  if(maxlen > 0){
    fmpz_comb_init(comb, modgbs->primes, modgbs->nprimes);
    fmpz_comb_temp_init(comb_temp, comb);
    fmpz_init(y);
  }

  set_recdata(rd2, mod_p);

  int32_t ret = end;
  double rt = 0;

  for(int32_t k = start; k < end; k++){
    if(dl->check1[k] > 1 && polys[k]->len > 0){
      /* dl->tmp is the lcm of the denominators found so far, starting
         from the witness ones; it is used as a guess for the next
         coefficient, so that only the missing factor is reconstructed */
      mpz_set(dl->tmp, dl->cden[k]);
      set_recdata_guess(rd1, mod_p, dl->tmp);
      int ok = 1;
      /* reading the clock for each coefficient costs as much as a good
       * part of their lifting: the time of the polynomial is split
       * between CRT and rational reconstruction by the time of CRT for
       * every LIFT_TIME_STEP-th coefficient */
      double tcrt = 0;
      int32_t ncrt = 0, l;
      const double rp = realtime();
      for(l = 0; l < polys[k]->len; l++){
        if(l % LIFT_TIME_STEP == 0){
          rt = realtime();
        }
        for(uint32_t i = 0; i < modgbs->nprimes-1; i++){
          modgbs->cf_64[i] = modgbs->wide ?
            (mp_limb_t)MODCF64(polys[k], l, i) :
            (mp_limb_t)MODCF(polys[k], l, i);
        }
        fmpz_multi_CRT_ui(y, modgbs->cf_64,
                          comb, comb_temp, 1);
        fmpz_get_mpz(zz, y);
        if(l % LIFT_TIME_STEP == 0){
          tcrt += realtime() - rt;
          ncrt++;
        }
        if(lift_ratreconwden(polys[k]->cf_qq[l], rden, zz, mod_p, dl->tmp, rd1)){
          mpz_mul(dens[l], rden, dl->tmp);
          if(mpz_cmp_ui(rden, 1)){
            mpz_mul(dl->tmp, dl->tmp, rden);
            set_recdata_guess(rd1, mod_p, dl->tmp);
          }
        }
        else{
          /* the witnesses of this poly are still valid: their common
             denominator remains the best guess for reconstructing them */
          mpz_set(dl->gden, dl->cden[k]);
          for(int32_t i = k; i < modgbs->ld; i++){
            dl->check2[i] = 0;
          }
          ok = 0;
          break;
        }
      }
      {
        const double tp = realtime() - rp;
        const double tc = ncrt > 0 ? tcrt * (l + (l < polys[k]->len)) / ncrt : 0;
        *st_crt  += tc < tp ? tc : tp;
        *st_rrec += tc < tp ? tp - tc : 0;
      }
      if(!ok){
        ret = k;
        break;
      }
      mpz_set(polys[k]->lm, dl->tmp);
      for(int32_t l = 0; l < polys[k]->len; l++){
        /* num * (lcm / den) rather than (num * lcm) / den: same value, but
         * the limb array of the (long-lived) result is not over-allocated
         * by the size of lcm (mpz_divexact does not shrink its output) */
        mpz_divexact(dens[l], dl->tmp, dens[l]);
        mpz_mul(polys[k]->cf_qq[l], polys[k]->cf_qq[l], dens[l]);
      }
      dl->check2[k] += 1;
    }
    if(dl->check1[k] < 1 && polys[k]->len > 0){
      ret = k;
      break;
    }
    if(polys[k]->len == 0){
        dl->check2[k] += 1;
    }
  }

  if(maxlen > 0){
    fmpz_clear(y);
    fmpz_comb_temp_clear(comb_temp);
    fmpz_comb_clear(comb);
    for(uint32_t l = 0; l < maxlen; l++){
      mpz_clear(dens[l]);
    }
    free(dens);
  }
  mpz_clear(zz);
  mpz_clear(rden);

  return ret;
}


/*
  Same as crt_ratrecon_lift_modgbs, the coefficients being lifted from
  their mixed radix digits (Garner's algorithm). The coefficients of a
  polynomial share their denominator, so with d the denominator found so
  far, x = c * d is an integer for most coefficients c. Its digits w.r.t.
  the primes p_0, p_1, ... are computed one after another from the
  residues of c * d, and x is known as soon as its digits become 0
  (x >= 0) resp. p_i - 1 (x < 0): lifting c costs as many primes as x has
  bits, not the product of all primes. At least 60 bits of digits are
  required to stop, i.e. one digit at a 61-bit prime, two at a 31-bit
  one. Where x is not found with all primes, i.e. where the denominator
  of c has a new factor, c * d is lifted by rational reconstruction, which
  extends d.

  The primes are the ones of mod_p, i.e. all but the learning one and the
  last one, which is left for the verification of the lifted polynomials.
 */
static inline int crt_ratrecon_lift_modgbs_mr(gb_modpoly_t modgbs, data_lift_t dl,
                                              int32_t start, int32_t end,
                                              mpz_t mod_p, rrec_data_t rd1,
                                              double *st_crt, double *st_rrec){

  modpolys_t *polys = modgbs->modpolys;
  /* the primes of the slots 1, ..., np */
  const uint32_t np = modgbs->nprimes - 2;
  const mp_limb_t *pr = modgbs->primes + 1;

  uint32_t maxlen = 0;
  for(int32_t k = start; k < end; k++){
    if(dl->check1[k] > 1 && polys[k]->len > maxlen){
      maxlen = polys[k]->len;
    }
  }
  mpz_t x, P, rden, one;
  mpz_inits(x, P, rden, one, NULL);
  mpz_set_ui(one, 1);
  mpz_set_ui(P, 1);
  /* pinv[i] is the inverse of p_0 ... p_{i-1} modulo p_i, dmod[i] the
     residue of d modulo p_i, both with their Shoup pre-inverses in pinvs
     and dmods */
  uint64_t *pinv = (uint64_t *)malloc(sizeof(uint64_t) * 4 * (np + 1));
  uint64_t *pinvs = pinv + np + 1;
  uint64_t *dmod = pinvs + np + 1;
  uint64_t *dmods = dmod + np + 1;
  /* the products p_0 ... p_{i-1} are kept for the first primes */
  const uint32_t npp = np < MR_PRODUCTS ? np + 1 : MR_PRODUCTS;
  mpz_t *pp = (mpz_t *)malloc(sizeof(mpz_t) * npp);
  for(uint32_t i = 0; i < np; i++){
    if(i < npp){
      mpz_init_set(pp[i], P);
    }
    pinv[i] = mod_p_inverse_64((int64_t)mpz_fdiv_ui(P, pr[i]), (int64_t)pr[i]);
    pinvs[i] = shoup_mu(pr[i], pinv[i]);
    mpz_mul_ui(P, P, pr[i]);
  }
  if(np < npp){
    mpz_init_set(pp[np], P);
  }
  /* for many primes the quadratic cost of the digits exceeds the one of
     a product tree */
  const int comb_init = np >= MR_COMB;
  uint64_t *res = NULL;
  fmpz_comb_t comb;
  fmpz_comb_temp_t comb_temp;
  fmpz_t y;
  if(comb_init){
    res = (uint64_t *)malloc(sizeof(uint64_t) * np);
    fmpz_comb_init(comb, (mp_limb_t *)pr, np);
    fmpz_comb_temp_init(comb_temp, comb);
    fmpz_init(y);
  }
  /* the denominators by which the coefficients have been lifted: the
     coefficient of index l is cf_qq[l] / eden[ep[l]] */
  uint32_t *ep = (uint32_t *)malloc(sizeof(uint32_t) * (maxlen + 1));
  uint32_t nealloc = 0;
  mpz_t *eden = NULL;

  int32_t ret = end;
  double rt;

  for(int32_t k = start; k < end; k++){
    if(dl->check1[k] > 1 && polys[k]->len > 0){
      modpolys_t *pol = polys + k;
      rt = realtime();
      /* the lcm of the witness denominators is the first guess */
      mpz_set(dl->tmp, dl->cden[k]);
      for(uint32_t i = 0; i < np; i++){
        dmod[i] = mpz_fdiv_ui(dl->tmp, pr[i]);
        dmods[i] = shoup_mu(pr[i], dmod[i]);
      }
      set_recdata_guess(rd1, mod_p, dl->tmp);
      uint32_t ne = 0;
      if(nealloc == 0){
        eden = (mpz_t *)malloc(sizeof(mpz_t) * 4);
        for(uint32_t e = 0; e < 4; e++){
          mpz_init(eden[e]);
        }
        nealloc = 4;
      }
      mpz_set(eden[ne++], dl->tmp);
      int ok = 1;
      for(int32_t l = 0; l < pol[0]->len; l++){
        uint32_t i = np;
        if(comb_init){
          /* x from all primes by a product tree, in balanced form */
          for(uint32_t j = 0; j < np; j++){
            const uint64_t c = modgbs->wide ?
              MODCF64(pol[0], l, j + 1) : (uint64_t)MODCF(pol[0], l, j + 1);
            res[j] = shoup_mod_mul(dmod[j], dmods[j], c, pr[j]);
          }
          fmpz_multi_CRT_ui(y, res, comb, comb_temp, 1);
          fmpz_get_mpz(x, y);
          if(mpz_cmpabs(x, rd1->N) <= 0){
            i = 0;
          }
        }
        else{
          mpz_set_ui(x, 0);
          /* sign of the run of digits ending x (0: digits 0, 1: digits
             p_i - 1) and its number of bits */
          int sgn = -1, bits = 0;
          for(i = 0; i < np; i++){
            const uint64_t p = pr[i];
            const uint64_t c = modgbs->wide ?
              MODCF64(pol[0], l, i + 1) : (uint64_t)MODCF(pol[0], l, i + 1);
            const uint64_t r = shoup_mod_mul(dmod[i], dmods[i], c, p);
            const uint64_t xm = mpz_fdiv_ui(x, p);
            uint64_t v = r >= xm ? r - xm : r + (p - xm);
            v = shoup_mod_mul(pinv[i], pinvs[i], v, p);
            if(i > 0 && (v == 0 || v == p - 1)){
              const int s = v != 0;
              bits = s == sgn ? bits + (p >> 32 ? 61 : 30) : (p >> 32 ? 61 : 30);
              sgn = s;
            }
            else{
              sgn = -1;
            }
            if(i < npp - 1){
              if(v != 0){
                mpz_addmul_ui(x, pp[i], v);
              }
            }
            else{
              if(i == npp - 1){
                mpz_set(P, pp[i]);
              }
              if(v != 0){
                mpz_addmul_ui(x, P, v);
              }
              mpz_mul_ui(P, P, p);
            }
            if(sgn >= 0 && bits >= 60){
              /* the digits from the start of the run on are p_j - 1:
                 x is the current value minus p_0 ... p_i */
              if(sgn == 1){
                mpz_sub(x, x, i < npp - 1 ? pp[i + 1] : P);
              }
              break;
            }
          }
        }
        if(i < np){
          mpz_swap(pol[0]->cf_qq[l], x);
          ep[l] = ne - 1;
          continue;
        }
        /* x is the residue of c * d modulo mod_p */
        double rt2 = realtime();
        if(!lift_ratreconwden(pol[0]->cf_qq[l], rden, x, mod_p, one, rd1)){
          *st_rrec += realtime() - rt2;
          /* the witnesses of this poly are still valid: their common
             denominator remains the best guess for reconstructing them */
          mpz_set(dl->gden, dl->cden[k]);
          for(int32_t i = k; i < modgbs->ld; i++){
            dl->check2[i] = 0;
          }
          ok = 0;
          break;
        }
        if(mpz_cmp_ui(rden, 1)){
          mpz_mul(dl->tmp, dl->tmp, rden);
          for(uint32_t i = 0; i < np; i++){
            dmod[i] = mpz_fdiv_ui(dl->tmp, pr[i]);
            dmods[i] = shoup_mu(pr[i], dmod[i]);
          }
          set_recdata_guess(rd1, mod_p, dl->tmp);
          if(ne == nealloc){
            eden = (mpz_t *)realloc(eden, sizeof(mpz_t) * 2 * nealloc);
            for(uint32_t e = nealloc; e < 2 * nealloc; e++){
              mpz_init(eden[e]);
            }
            nealloc *= 2;
          }
          mpz_set(eden[ne++], dl->tmp);
        }
        ep[l] = ne - 1;
        *st_rrec += realtime() - rt2;
        rt += realtime() - rt2;
      }
      *st_crt += realtime() - rt;
      if(!ok){
        ret = k;
        break;
      }
      mpz_set(pol[0]->lm, dl->tmp);
      if(ne > 1){
        /* numerators w.r.t. the final denominator */
        for(uint32_t e = 0; e < ne - 1; e++){
          mpz_divexact(eden[e], dl->tmp, eden[e]);
        }
        for(int32_t l = 0; l < pol[0]->len; l++){
          if(ep[l] < ne - 1){
            mpz_mul(pol[0]->cf_qq[l], pol[0]->cf_qq[l], eden[ep[l]]);
          }
        }
      }
      dl->check2[k] += 1;
    }
    if(dl->check1[k] < 1 && polys[k]->len > 0){
      ret = k;
      break;
    }
    if(polys[k]->len == 0){
        dl->check2[k] += 1;
    }
  }

  for(uint32_t e = 0; e < nealloc; e++){
    mpz_clear(eden[e]);
  }
  free(eden);
  free(ep);
  if(comb_init){
    fmpz_clear(y);
    fmpz_comb_temp_clear(comb_temp);
    fmpz_comb_clear(comb);
    free(res);
  }
  for(uint32_t i = 0; i < npp; i++){
    mpz_clear(pp[i]);
  }
  free(pp);
  free(pinv);
  mpz_clears(x, P, rden, one, NULL);

  return ret;
}


/* returns (coef == num / den mod prime) */
static inline int verif_coef(mpz_t num, mpz_t den, uint64_t prime, uint64_t coef){
  uint64_t lc = mpz_fdiv_ui(den, prime);
  lc = mod_p_inverse_64((int64_t)lc, (int64_t)prime);

  uint64_t c = mpz_fdiv_ui(num, prime);
  c = mul_mod_64(c, lc, prime);

  return (c==coef);
}


static inline int verif_lifted_basis(gb_modpoly_t modgbs, data_lift_t dl,
                                     int thrds){
  for(int32_t k = 0; k < modgbs->ld; k++){
    if(dl->check1[k]>=1 && dl->check2[k]>0 && dl->check2[k] < NBCHECK ){
      modpolys_t *pol = modgbs->modpolys + k;
      if(pol[0]->len == 0){
        dl->check2[k]++;
      }
      else{
        for(int i = 0; i < thrds; i++){
          const uint32_t np = modgbs->nprimes - (thrds - i);
          const mp_limb_t prime = modgbs->primes[np];
          /* all coefficients share the denominator lm */
          uint64_t ilc = mpz_fdiv_ui(pol[0]->lm, prime);
          ilc = mod_p_inverse_64((int64_t)ilc, (int64_t)prime);
          const uint64_t ilcs = shoup_mu(prime, ilc);
          for(int32_t c = 0; c < pol[0]->len; c++){
            uint64_t v = mpz_fdiv_ui(pol[0]->cf_qq[c], prime);
            v = shoup_mod_mul(ilc, ilcs, v, prime);
            if(v != (modgbs->wide ?
                     MODCF64(pol[0], c, np) :
                     (uint64_t)MODCF(pol[0], c, np))){
              dl->check2[k] = 0;
              return 0;
            }
          }
          dl->check2[k]++;
          if(dl->check2[k] >= NBCHECK){
            dl->S = k+1;
            dl->lstart = dl->S;
            /* modular images are no longer needed */
            if(modgbs->wide){
              free(pol[0]->cf_64);
              pol[0]->cf_64 = NULL;
            }
            else{
              free(pol[0]->cf_32);
              pol[0]->cf_32 = NULL;
            }
            break;
          }
        }
      }
    }
  }
  return 1;
}

static inline int verif_lifted_rational_wcoef(gb_modpoly_t modgbs, data_lift_t dl,
                                              int thrds){
  for(int32_t k = dl->lstart; k < dl->lend; k++){
    if(dl->check1[k]==0){
      /* too early to perform the verification */
      return k;
    }
    if(modgbs->modpolys[k]->len == 0){
        dl->start = MIN(dl->start + 1, dl->lend);
        dl->check1[k]++;
    }
    else{
      for(int i = 0; i < thrds; i++){

        mp_limb_t prime = modgbs->primes[modgbs->nprimes - (thrds - i) ];
        uint64_t coef = modgbs->wide ?
          MODCF64(modgbs->modpolys[k], dl->coef[k], modgbs->nprimes  - (thrds - i)) :
          MODCF(modgbs->modpolys[k], dl->coef[k], modgbs->nprimes  - (thrds - i));
        int boo = verif_coef(dl->num[k], dl->den[k], prime, coef);
        if(boo){
          coef = modgbs->wide ?
            MODCF64(modgbs->modpolys[k], dl->coef2[k], modgbs->nprimes  - (thrds - i)) :
            MODCF(modgbs->modpolys[k], dl->coef2[k], modgbs->nprimes  - (thrds - i));
          boo = verif_coef(dl->num2[k], dl->den2[k], prime, coef);
        }
        if(boo){
          coef = modgbs->wide ?
            MODCF64(modgbs->modpolys[k], dl->coef3[k], modgbs->nprimes  - (thrds - i)) :
            MODCF(modgbs->modpolys[k], dl->coef3[k], modgbs->nprimes  - (thrds - i));
          boo = verif_coef(dl->num3[k], dl->den3[k], prime, coef);
        }

        if(!boo){
          for(int32_t kk = k; kk < dl->end; kk++){
            dl->check1[kk] = 0;
          }
          dl->start = k;
          return k;
        }
        dl->start = MIN(dl->start + 1, dl->lend);
        dl->check1[k]++;
      }
    }
  }
  return -1;
}

/* returns 0 iff rational number could not be lifted */
static inline int reconstructcoeff(data_lift_t dl, int32_t i, mpz_t mod_p,
                                   rrec_data_t recdata1, rrec_data_t recdata2){

  set_recdata_guess(recdata1, mod_p, dl->gden);
  int b = lift_ratreconwden(dl->num[i], dl->den[i],
                            dl->crt[i], mod_p, dl->gden, recdata1);
  if(b){
    mpz_mul(dl->den[i], dl->den[i], dl->gden);
    mpz_gcd(dl->tmp, dl->den[i], dl->num[i]);

    mpz_divexact(dl->num[i], dl->num[i], dl->tmp);
    mpz_divexact(dl->den[i], dl->den[i], dl->tmp);

    mpz_set(dl->gden, dl->den[i]);
  }
  else{

    b = lift_ratrecon(dl->num[i], dl->den[i],
                      dl->crt[i], mod_p, recdata2);
    if(b){
      mpz_set(dl->gden, dl->den[i]);
    }
  }
  return b;
}

/* reconstructs the second witness using as a guess the lcm of the
   denominator of the first one and of the one guessed from the previous
   polynomial; cden[i] is set to the lcm of both witness denominators */
static inline int reconstructcoeff2(data_lift_t dl, int32_t i, mpz_t mod_p,
                                    rrec_data_t recdata1, rrec_data_t recdata2){
  mpz_lcm(dl->cden[i], dl->den[i], dl->pgden);
  set_recdata_guess(recdata1, mod_p, dl->cden[i]);
  if(!lift_ratreconwden(dl->num2[i], dl->den2[i],
                        dl->crt2[i], mod_p, dl->cden[i], recdata1)){
    /* same fallback as for the first witness: balanced bounds, no guess */
    if(!lift_ratrecon(dl->num2[i], dl->den2[i], dl->crt2[i], mod_p, recdata2)){
      return 0;
    }
    mpz_lcm(dl->cden[i], dl->den[i], dl->den2[i]);
    mpz_set(dl->gden, dl->cden[i]);
    return 1;
  }
  mpz_mul(dl->den2[i], dl->den2[i], dl->cden[i]);
  mpz_gcd(dl->tmp, dl->den2[i], dl->num2[i]);
  mpz_divexact(dl->num2[i], dl->num2[i], dl->tmp);
  mpz_divexact(dl->den2[i], dl->den2[i], dl->tmp);
  mpz_lcm(dl->cden[i], dl->den[i], dl->den2[i]);
  mpz_set(dl->gden, dl->cden[i]);
  return 1;
}

/* reconstructs the third witness using the lcm of the denominators of the
   first two witnesses as a guess; cden[i] is updated accordingly */
static inline int reconstructcoeff3(data_lift_t dl, int32_t i, mpz_t mod_p,
                                    rrec_data_t recdata1, rrec_data_t recdata2){
  set_recdata_guess(recdata1, mod_p, dl->cden[i]);
  if(!lift_ratreconwden(dl->num3[i], dl->den3[i],
                        dl->crt3[i], mod_p, dl->cden[i], recdata1)){
    /* same fallback as for the first witness: balanced bounds, no guess */
    if(!lift_ratrecon(dl->num3[i], dl->den3[i], dl->crt3[i], mod_p, recdata2)){
      return 0;
    }
    mpz_lcm(dl->cden[i], dl->cden[i], dl->den3[i]);
    mpz_set(dl->gden, dl->cden[i]);
    return 1;
  }
  mpz_mul(dl->den3[i], dl->den3[i], dl->cden[i]);
  mpz_gcd(dl->tmp, dl->den3[i], dl->num3[i]);
  mpz_divexact(dl->num3[i], dl->num3[i], dl->tmp);
  mpz_divexact(dl->den3[i], dl->den3[i], dl->tmp);
  mpz_lcm(dl->cden[i], dl->cden[i], dl->den3[i]);
  mpz_set(dl->gden, dl->cden[i]);
  return 1;
}

static void ratrecon_gb(gb_modpoly_t modgbs, data_lift_t dl,
                        mpz_t mod_p, mpz_t prod_p,
                        rrec_data_t recdata1, rrec_data_t recdata2,
                        int thrds, double *st_crt, double *st_rrec, double *st_wit){

  double st = realtime();
  verif_lifted_rational_wcoef(modgbs, dl, thrds);
  int  b = verif_lifted_basis(modgbs, dl, thrds);
  if(b == 0){
    for(int32_t i = 0; i < modgbs->ld; i++){
      if(dl->check2[i] == 0){
        dl->lstart = i;
        break;
      }
    }
  }
  int stop = 1;
  for(int32_t i = 0; i < modgbs->ld; i++){
    if(dl->check2[i] < NBCHECK){
      stop = 0;
      break;
    }
  }
  if(stop){
    return;
  }
  *st_rrec += realtime()-st;
  if(dl->lstart != dl->start){
    if(msolve_gb_lifting == 1){
      dl->start = crt_ratrecon_lift_modgbs_mr(modgbs, dl, dl->lstart, dl->start,
                                              mod_p, recdata1,
                                              st_crt, st_rrec);
    }
    else{
      dl->start = crt_ratrecon_lift_modgbs(modgbs, dl, dl->lstart, dl->start,
                                           mod_p, recdata1, recdata2,
                                           st_crt, st_rrec);
    }
  }
  dl->lstart = dl->start;

  st = realtime();
  incremental_dlift_crt_full(modgbs, dl,
                             dl->coef, mod_p, prod_p,
                             thrds);
  *st_crt += realtime() - st;

  /********************************************************/
  /*            RATRECON WITNESS COEFFS                   */
  /********************************************************/

  if(dl->lstart == 0){
    mpz_set_ui(dl->gden, 1);
  }

  set_recdata(recdata2, mod_p);

  st = realtime();
  if(modgbs->nprimes % dl->rr == 0){
    for(int32_t i = dl->lstart; i < dl->lend; i++){
      if(modgbs->modpolys[i]->len==0){
          dl->check1[i]++;
      }
      else{
        /* guess the denominator from the previous polynomial (its lifted
           denominator, or the one of its witnesses once they are
           verified), so that a spurious reconstruction never becomes the
           guess */
        if(i > 0){
          if(dl->check2[i-1] > 0 && modgbs->modpolys[i-1]->len > 0){
            mpz_set(dl->gden, modgbs->modpolys[i-1]->lm);
          }
          else if(dl->check1[i-1] > 1 && modgbs->modpolys[i-1]->len > 0){
            mpz_set(dl->gden, dl->cden[i-1]);
          }
        }
        mpz_set(dl->pgden, dl->gden);
        int b = reconstructcoeff(dl, i, mod_p,
                               recdata1, recdata2);
        if(b){
          b = reconstructcoeff2(dl, i, mod_p, recdata1, recdata2);
        }
        if(b){
          b = reconstructcoeff3(dl, i, mod_p, recdata1, recdata2);
        }
        if(!b){
          dl->check1[i] = 0;
          break;
        }
        else{
          dl->check1[i]++;
        }
      }
    }
  }
  *st_rrec += realtime()-st;
  *st_wit += realtime()-st;

}

long max_bit_size_gb(gb_modpoly_t modgbs){
  long nb = 0;
  for(uint32_t i = 0; i < modgbs->ld; i++){
    for(uint32_t j = 0; j < modgbs->modpolys[i]->len; j++){
      nb = MAX(nb, mpz_sizeinbase(modgbs->modpolys[i]->cf_qq[j], 2));
    }
    nb = MAX(nb, mpz_sizeinbase(modgbs->modpolys[i]->lm, 2));
  }
  return nb;
}


/*

  - returns 0 if the computation went ok

  - returns 1 in case of failure

  - returns -3 if meta data were not ok

  - returns -4 if there are too many bad primes
*/

/* Frees the data which are (re)built by a learning phase, so that a new
 * learning phase can start from scratch after a failed attempt. */
static void reset_learning_data(mstrace_t msd, md_t *st){
  for(int i = 0; i < st->nthrds; i++){
    free(msd->leadmons_current[i]);
    msd->leadmons_current[i] = NULL;
    if(i > 0){
      free(msd->leadmons_ori[i]);
      msd->leadmons_ori[i] = NULL;
      if(msd->btrace[i] != NULL){
        free_trace(&(msd->btrace[i]));
      }
    }
  }
  /* the tracer learnt with the previous prime must not be applied when
   * learning again */
  if(st->tr != NULL){
    free_trace(&(st->tr));
  }
  st->trace_level = NO_TRACER;
  /* the same holds for the DRL Groebner basis computed first for
   * elimination orders */
  free_elim_drl_data(st);
}

/* smallest prime greater than n; n is expected to be far below 2^63 so
 * that the result fits in an unsigned long */
static uint64_t next_prime_64(uint64_t n){
  mpz_t p;
  mpz_init_set_ui(p, n);
  mpz_nextprime(p, p);
  const uint64_t ret = mpz_get_ui(p);
  mpz_clear(p);
  return ret;
}

/* returns 1 if some coefficient of some input generator vanishes modulo
 * fc; same as is_lucky_prime_ui, but for 64-bit primes */
static int is_lucky_prime_u64(uint64_t fc, const bs_t *bs){
  mpz_t p;
  mpz_init_set_ui(p, fc);
  const int ret = is_lucky_prime(p, bs);
  mpz_clear(p);
  return ret;
}

gb_modpoly_t *core_groebner_qq(
        gb_modpoly_t *modgbsp,
        bs_t *bs,
        mstrace_t msd,
        md_t *st,
        int32_t *errp,
        const len_t fc,
        const int print_gb
        )
{

restart:

  *errp = 0;

  int32_t info_level = st->info_level;
  int32_t truncate_lifting = st->truncate_lifting;

  if(fc == 0){
    /*generate lucky prime numbers */
    generate_lucky_primes(msd->lp, bs, st->prime_start, st->nthrds);
  }
  else{
    msd->lp->old = 0;
    msd->lp->ld = 1;
    msd->lp->p = calloc(1, sizeof(uint32_t));
    normalize_initial_basis(bs, st->gfc);
  }

  uint32_t prime = 0;
  uint32_t primeinit = 0;
  uint32_t lprime = 1303905299;
  /* application prime when the signature record is replayed at 61-bit
   * primes (only used then) */
  uint64_t prime64 = 0;

  prime = next_prime(rand() % (1303905301 - (1<<30) + 1) + (1<<30));
  while(fc == 0 && is_lucky_prime_ui(prime, bs)){
    prime = next_prime(rand() % (1303905301 - (1<<30) + 1) + (1<<30));
  }

  primeinit = prime;
  msd->lp->p[0] = primeinit;
  if(fc){
    msd->lp->p[0] = fc;
    primeinit = fc;
  }
  if(info_level){
      fprintf(VERBSTREAM, "Initial prime = %d\n", msd->lp->p[0]);
  }

  int success = 1;

  int32_t maxbitsize = maxbitsize_generators(bs);

  int learn = 1, apply = 1, nprimes = 0;
  double stf4 = 0;

  rrec_data_t recdata1, recdata2;
  initialize_rrec_data(recdata1);
  initialize_rrec_data(recdata2);

  data_lift_t dlift;

  /* indicates that dlift has been already initialized */
  int dlinit = 0;

  double st_crt = 0;
  double st_rrec = 0;
  double st_wit = 0;

  uint32_t nbadprimes = 0;

  while(learn){

    int32_t is_empty = gb_modular_trace_learning((*modgbsp),
                                                 msd->mgb,
                                                 msd->num_gb, msd->leadmons_ori,
                                                 msd->btrace[0],
                                                 bs, st,
                                                 msd->lp->p[0],
                                                 info_level,
                                                 print_gb,
                                                 truncate_lifting,
                                                 0,
                                                 maxbitsize,
                                                 &success);
    if(is_empty == 1 || print_gb == 1 || (*modgbsp)->ld == 0){
      if(dlinit){
        data_lift_clear(dlift);
      }
      free_rrec_data(recdata1);
      free_rrec_data(recdata2);

      return modgbsp;
    }

    apply = 1;

    gb_modpoly_realloc((*modgbsp), 1);

#ifdef DEBUGGBLIFT
    display_gbmodpoly_cf_32(ERRSTREAM, (*modgbsp));
#endif

    if(!dlinit){
      int nb = 0;
      int32_t *ldeg = array_nbdegrees((*msd->leadmons_ori), msd->num_gb[0],
                                      bs->ht->nv - st->nev, &nb);
      data_lift_init(dlift, (*modgbsp)->ld, ldeg, nb);
      choose_coef_to_lift((*modgbsp), dlift);
      free(ldeg);
      dlinit = 1;
    }

    if(info_level){
      fprintf(VERBSTREAM,"\n---------- COMPUTATIONAL DATA -----------\n");
      int s= 0;
      for(int i = 0; i < dlift->nsteps; i++){
        fprintf(VERBSTREAM, "[%d]", dlift->steps[i]);
        fflush(VERBSTREAM);
        s+=dlift->steps[i];
      }
      fprintf(VERBSTREAM, "\n");
      fprintf(VERBSTREAM,
	      "#polynomials to lift %14lu\n",
	      (unsigned long) s);
      fprintf(VERBSTREAM, "-----------------------------------------\n");
    }

    if(is_empty || success == 0 || fc) {

      apply = 0;
      if(dlinit){
        data_lift_clear(dlift);
      }

      gb_modpoly_without_hash_table_clear((*modgbsp));
      reset_learning_data(msd, st);

      free_rrec_data(recdata1);
      free_rrec_data(recdata2);

      fprintf(VERBSTREAM, "Something went wrong in the learning phase, msolve restarts.");
      /* return core_groebner_qq(modgbsp, bs, msd, st, errp, fc, print_gb); */
      goto restart;
    }
    /* duplicate data for multi-threaded multi-mod computation */
    duplicate_data_mthread_gbtrace(st->nthrds, bs, st, msd->num_gb,
                                   msd->leadmons_ori, msd->leadmons_current,
                                   msd->btrace);


    learn = 0;
    /* The trace, of F4 or of a signature-based run, is applied at two
     * ~31-bit primes, the first one warming up the data, and then at a
     * 61-bit prime, and the walk goes on with the primes of the size
     * which gives more bits of the coefficients per second: a 61-bit
     * prime gives about twice the bits, but the linear algebra at 61-bit
     * primes is slower for denser matrices. The time at the 61-bit prime
     * includes the widening of the images stored so far, which keeps the
     * decision on the side of the ~31-bit primes. */
    int wide_walk = 0;
    /* 0, 1, 2: the probe applications of the trace, 3: decided */
    int probe = 0;
    double tprobe[2] = {0, 0};
    {
      /* seed a random prime in [2^61, 2^62) from two rand() calls */
      uint64_t r = ((uint64_t)rand() << 31) | (uint64_t)rand();
      prime64 = ((uint64_t)1 << 61) + r % ((uint64_t)1 << 61);
      prime64 = next_prime_64(prime64);
    }
    prime = next_prime(rand() % (1303905301 - (1<<30) + 1) + (1<<30));
    if(info_level){
        if(wide_walk){
          printf("application phase at 61-bit primes\n");
          fflush(stdout);
          fprintf(VERBSTREAM, "New prime = %lu\n", (unsigned long)prime64);
        }
        else{
          fprintf(VERBSTREAM, "New prime = %d\n", prime);
        }
    }
    /* the bases of the next primes, computed together in one application
       of the trace, see core_f4_multi() */
    bs_t *mbs[MULTI_PRIMES];
    uint64_t mpr[MULTI_PRIMES];
    int mld = 0, mnx = 0;
    /* the time of such a pass, -1 before the first one, whether passes
       are not possible, and the primes used so far, those of 61 bits
       counted twice */
    double tpass = -1;
    int multi_off = !msolve_multi_primes;
    double r31 = 0;
    /* the time spent applying a trace of a signature-based run, and
       whether it was replaced by a trace learned by F4 */
    double tsig = 0;
    int relearned = st->nev > 0 || st->tr == NULL || st->tr->sgr == NULL;
    const double tlearn = st->learning_rtime;
    while(apply){

      int nthrds = 1; /* mono-threaded mult-mid comp */

      if(!relearned && mnx >= mld && probe >= 3
         && tsig >= SIG_RELEARN * tlearn){
        relearned = 1;
        if(relearn_by_f4(bs, st, msd->lp->p[0], msd->leadmons_ori[0],
                         msd->num_gb[0])){
          if(info_level > 1){
            fprintf(VERBSTREAM, "(F4 trace)");
          }
          /* the probes are done again for the new trace */
          probe     = 0;
          tprobe[0] = tprobe[1] = 0;
          tpass     = -1;
        }
      }
      if(probe < 3){
        wide_walk = probe == 2;
      }
      /* is the basis of the next prime computed already? */
      const int queued = mnx < mld;
      /* A pass applying the trace at MULTI_PRIMES primes costs about the
         same for any number of primes needed, so it is only done where
         it is clearly cheaper per prime than applying the trace at single
         primes of the size chosen by the probe, and where at least as many
         primes were used already as the pass costs: the number of primes
         needed roughly doubles, so then most of the primes of the pass are
         needed. Before the first pass its time is estimated. */
      int multi_now = 0;
      if(!queued && probe >= 3 && !multi_off && tprobe[0] > 0){
        const double unit = tprobe[1] > 0
          ? MIN(tprobe[0], tprobe[1] / 2) : tprobe[0];
        const double tp = tpass >= 0 ? tpass
          : 0.6 * MULTI_PRIMES * tprobe[0];
        multi_now = tp < 0.9 * MULTI_PRIMES * unit && tp <= r31 * unit;
      }
      /* the tests set NEOGB_MULTI_FORCE to apply the trace at several
         primes at once wherever this is possible */
      if(!queued && probe >= 3 && !multi_off
         && getenv("NEOGB_MULTI_FORCE") != NULL){
        multi_now = 1;
      }
      if(queued){
        /* the prime is the next one of the queue */
      }
      else if(wide_walk && !multi_now){
        /* walk the primes in [2^61, 2^62), wrapping around at 2^62 */
        prime64 = next_prime_64(prime64);
        if(prime64 >= (uint64_t)1 << 62){
          prime64 = next_prime_64((uint64_t)1 << 61);
        }
        /* generate lucky prime numbers; the learning prime is < 2^31 so
         * no primeinit duplicate check is needed here */
        while(is_lucky_prime_u64(prime64, bs)){
          prime64 = next_prime_64(prime64);
          if(prime64 >= (uint64_t)1 << 62){
            prime64 = next_prime_64((uint64_t)1 << 61);
          }
        }
      }
      else{
        prime = next_prime(prime);
        if(prime >= lprime){
          prime = next_prime(1<<30);
        }
        /* generate lucky prime numbers */
        msd->lp->p[0] = prime;
        while(is_lucky_prime_ui(prime, bs) || prime==primeinit){
          prime = next_prime(prime);
          if(prime >= lprime){
            prime = next_prime(1<<30);
          }
          msd->lp->p[0] = prime;
        }
        for(len_t i = 1; i < nthrds/* st->nthrds */; i++){
          prime = next_prime(prime);
          if(prime >= lprime){
            prime = next_prime(1<<30);
          }
          msd->lp->p[i] = prime;
          while(is_lucky_prime_ui(prime, bs) || prime==primeinit){
            prime = next_prime(prime);
            if(prime >= lprime){
              prime = next_prime(1<<30);
            }
            msd->lp->p[i] = prime;
          }
        }
        prime = msd->lp->p[nthrds /* st->nthrds */ - 1];
        /* the trace is applied at this prime and the next lucky ones in
           one pass if all its matrices are stored */
        if(multi_now){
          mpr[0] = prime;
          uint32_t q = prime;
          for(int l = 1; l < MULTI_PRIMES; l++){
            q = next_prime(q);
            if(q >= lprime){
              q = next_prime(1<<30);
            }
            while(is_lucky_prime_ui(q, bs) || q==primeinit){
              q = next_prime(q);
              if(q >= lprime){
                q = next_prime(1<<30);
              }
            }
            mpr[l] = q;
          }
          const double t0 = realtime();
          if(core_f4_multi(bs, st, mpr, MULTI_PRIMES, mbs)){
            if(info_level > 1){
              fprintf(VERBSTREAM, "(x%d)", MULTI_PRIMES);
            }
            tpass = realtime() - t0;
            if(!relearned){
              tsig += tpass;
            }
            mld = MULTI_PRIMES;
            mnx = 0;
            prime = (uint32_t)mpr[MULTI_PRIMES-1];
          }
          else{
            multi_off = 1;
          }
        }
      }

      if((*modgbsp)->alloc <= nprimes + 2){
        /* Room for the images modulo the next primes. The multi-modular
         * computation is mono-threaded (one prime per step), so this must
         * not scale with st->nthrds. The increment is kept small (at most
         * ~1/16 of the current number of primes, but at least 4) since the
         * unused slots are allocated for every term of the basis which is
         * not lifted yet, i.e. for most of the basis at the end of the
         * computation, when memory consumption peaks. */
        gb_modpoly_realloc((*modgbsp), MAX(4, (uint32_t)nprimes / 16));
      }

      const double tapp = realtime();
      const int pre = mnx < mld;
      const uint64_t afc = pre ? mpr[mnx]
        : (wide_walk && !multi_now ? prime64 : (uint64_t)prime);
      r31 += afc > UINT32_MAX ? 2 : 1;
      gb_modular_trace_application((*modgbsp), msd->mgb,
                                   msd->num_gb,
                                   msd->leadmons_ori,
                                   msd->leadmons_current,
                                   msd->btrace,
                                   bs,  st,
                                   0, /* info_level, */
                                   afc,
                                   dlift->S, &stf4, msd->bad_primes,
                                   pre, pre ? mbs[mnx] : NULL);
      if(pre){
        mbs[mnx++] = NULL;
      }
      else if(!relearned){
        tsig += realtime() - tapp;
      }
      if(probe < 3){
        if(probe > 0){
          tprobe[probe-1] = realtime() - tapp;
        }
        if(msd->bad_primes[0] != 0){
          /* the time of a bad prime says nothing, keep ~31-bit primes */
          probe = 3;
          wide_walk = 0;
        }
        else if(++probe == 3){
          wide_walk = tprobe[1] < 1.9 * tprobe[0];
          if(info_level > 1){
            fprintf(VERBSTREAM, "(%s-bit primes: %.3f s at 31 bits, %.3f s at 61 bits)",
                    wide_walk ? "61" : "31", tprobe[0], tprobe[1]);
          }
        }
      }

      /* nprimes += st->nthrds; */
      nprimes += 1; /* at the moment, multi-mod comp is mono-threaded */
      if(nprimes == 1){
        /* if(info_level>1){ */
        /*   fprintf(ERRSTREAM, "Application phase %.2f Gops/sec\n", */
        /*           (st->application_nr_add+st->application_nr_mult)/1000.0/1000.0/(stf4)); */
        /*   fprintf(ERRSTREAM, "Elapsed time: %.2f\n", stf4); */
        /* } */
	  if(info_level){
	    fprintf(VERBSTREAM,
		    "\n---------------- TIMINGS ----------------\n");
	    fprintf(VERBSTREAM,
		    "multi-mod overall(elapsed) %9.2f sec\n",
		    stf4);
	    /* no trace is learned with probabilistic linear algebra */
	    if (info_level > 1 && st->tr != NULL){
	      /* multiply-adds by the known pivots, recorded in the trace,
	       * for the prime of this step */
	      fprintf(VERBSTREAM,
		      "application phase          %9.2f Gops/sec\n",
		      2.0*st->tr->nmadd/1000.0/1000.0/1000.0/(stf4));
	    }
	    fprintf(VERBSTREAM,
		    "-----------------------------------------\n");
        }
      if (info_level) {
	      fprintf(VERBSTREAM,
		      "\nmulti-modular steps\n");
	      fprintf(VERBSTREAM, "-------------------------------------------------\
-----------------------------------------------------\n");
        }
      }
      int bad = 0;
      for(int i = 0; i < nthrds/* st->nthrds */; i++){
        if(msd->bad_primes[i] == 1){
            bad = 1;
            if(info_level > 1){
                fprintf(VERBSTREAM, "[!]");
            }
            nbadprimes++;
            msd->bad_primes[i] = 0;
        }
        if(msd->bad_primes[i] == 2){
            bad = 1;
            if(info_level > 1){
                fprintf(VERBSTREAM, "[!!]");
            }
            /* nbadprimes = nprimes; */
            nbadprimes++;
            msd->bad_primes[i] = 0;
        }
      }

      if(nbadprimes >= nprimes){
        if(info_level){
          fprintf(VERBSTREAM, "\nToo many bad primes, computation will restart\n");
          fprintf(VERBSTREAM, "-------------------------------------------------\
-----------------------------------------------------\n");
        }
        if(dlinit){
          data_lift_clear(dlift);
        }

        gb_modpoly_without_hash_table_clear((*modgbsp));
        reset_learning_data(msd, st);
        free_rrec_data(recdata1);
        free_rrec_data(recdata2);
        /* reset info_level to previous setting */
        st->info_level = info_level;
        for(; mnx < mld; mnx++){
          if(mbs[mnx] != NULL){
            free_basis_and_only_local_hash_table_data(&mbs[mnx]);
          }
        }
        /* return core_groebner_qq(modgbsp, bs, msd, st, errp, fc, print_gb); */
        goto restart;

      }

      int lstart = dlift->lstart;
      st_wit = 0;

      if(!bad){
        ratrecon_gb((*modgbsp), dlift, msd->mod_p, msd->prod_p, recdata1, recdata2,
                    nthrds/* st->nthrds */, &st_crt, &st_rrec, &st_wit);
      }
      /* the period depends only on the number of primes (not on timings)
         so that the number of primes used is reproducible */
      int32_t rr = 1;
      while(2 * rr <= nprimes / RR_PERIOD_DIV){
        rr *= 2;
      }
      if(rr != dlift->rr){
        dlift->rr = rr;
        if(info_level){
          fprintf(VERBSTREAM, "(->%d)", dlift->rr);
	        fflush(VERBSTREAM);
        }
      }
      if(info_level){
        if(!(nprimes & (nprimes - 1))){
          fprintf(VERBSTREAM, "{%d}", nprimes);
	      fflush(VERBSTREAM);
        }
      }
      apply = 0;
      for(len_t i = 0; i < (*modgbsp)->ld; i++){
        if(dlift->check2[i] < NBCHECK){
          apply = 1;
          break;
        }
      }
      if(!bad && print_gb == 1){
          apply = 0;
      }
      if(dlift->lstart != lstart){
        if(info_level){
          fprintf(VERBSTREAM, "<%.2f%%>", 100* (float)MIN((dlift->lstart + 1), (*modgbsp)->ld)/(*modgbsp)->ld);
	      fflush(VERBSTREAM);
        }
        lstart = dlift->lstart;
      }
      /* this is where learn could be reset to 1 */
      /* but then duplicated datas and others should be free-ed */
    }
    for(; mnx < mld; mnx++){
      if(mbs[mnx] != NULL){
        free_basis_and_only_local_hash_table_data(&mbs[mnx]);
      }
    }
    if (info_level){
      fprintf(VERBSTREAM, " \n-------------------------------------------------\
-----------------------------------------------------\n");
    }
  }
  if(info_level){
    long nbits = max_bit_size_gb((*modgbsp));
    fprintf(VERBSTREAM,"\n\n---------- COMPUTATIONAL DATA -----------\n");
    fprintf(VERBSTREAM, "Max coeff. bitsize %16lu\n", (unsigned long) nbits);
    fprintf(VERBSTREAM, "#primes            %16lu\n", (unsigned long) nprimes);
    fprintf(VERBSTREAM, "#bad primes        %16lu\n", (unsigned long) nbadprimes);
    fprintf(VERBSTREAM, "-----------------------------------------\n");
    fprintf(VERBSTREAM, "\n---------------- TIMINGS ----------------\n");
    fprintf(VERBSTREAM, "CRT     (elapsed)         %10.2f sec\n", st_crt);
    fprintf(VERBSTREAM, "ratrecon(elapsed)         %10.2f sec\n", st_rrec);
    fprintf(VERBSTREAM, "-----------------------------------------\n");
  }

  if(dlinit){
    data_lift_clear(dlift);
  }

  free_rrec_data(recdata1);
  free_rrec_data(recdata2);

  return modgbsp;
}

uint64_t export_results_from_groebner_qq(
        /* return values */
        int32_t *bld,   /* basis load */
        int32_t **blen, /* length of each poly in basis */
        int32_t **bexp, /* basis exponent vectors */
        void **bcf,     /* coefficients of basis elements */
        void *(*mallocp) (size_t),
        const int32_t elim_block_len,
        const int32_t print_gb, /* if this is 1, only leading monomials are exported */
        gb_modpoly_t gb
        )
{
    int64_t nelts = gb->ld;
    /* Over QQ, if eliminating, these variables are no longer included in */
    /* gb->nv, thus we need to help us with adding elim_block_len */
    /* correspondingly when exporting the basis. */
    int32_t nv    = gb->nv;
    int32_t nve   = gb->bht->nv;

    *bld  = nelts;

    int32_t *lens = (int32_t *)(*mallocp)(((uint64_t)nelts) * sizeof(int32_t));

    int64_t nterms = 0;

    for(int32_t i = 0; i < nelts; i++){
        int32_t nlen = 0;
        if (print_gb == 2) {
            int32_t len = gb->modpolys[i]->len;

            for(int32_t j = len-1; j >= 0; j--){
                if (mpz_cmp_ui(gb->modpolys[i]->cf_qq[j], 0) != 0) {
                    nlen++;
                }
            }
        }
        nlen++; /* to take into account the lm */
        lens[i] = nlen;
        nterms += nlen;
    }

    int32_t *exp = (int32_t *)(*mallocp)(
            ((uint64_t)nterms) * ((uint64_t) nve) * sizeof(int32_t));

    memset(exp, 0, (unsigned long)nterms * nve * sizeof(int32_t));

    mpz_t *cf_qq = (mpz_t *)(*mallocp)(
            nterms *sizeof(mpz_t));
    for(int64_t i = 0; i < nterms; i++){
        mpz_init(cf_qq[i]);
    }

    hm_t *hm = NULL;
    ht_t *ht = gb->bht;
    int *evi = NULL;
    if (print_gb == 2) {
        const len_t ebl = ht->ebl;
        const len_t evl = ht->evl;
        evi = (int *)malloc((unsigned long)(ht->nv + 1) * sizeof(int));
        if (ebl == 0) {
            for (len_t i = 1; i <= evl; ++i) {
                evi[i-1]    =   i;
            }
        } else {
            for (len_t i = 1; i < ebl; ++i) {
                evi[i-1]    =   i;
            }
            for (len_t i = ebl + 1; i < evl; ++i) {
                evi[i - 2] = i;
            }
        }
    }

    int64_t term = 0;
    for(int64_t p = 0; p < nelts; p++){
        hm  = gb->hm[p]+OFFSET;
        int32_t l = gb->modpolys[p]->len;
        for(int32_t n = 0; n < nv; n++){
            exp[term * nve + n + elim_block_len] = gb->ldm[p * nv + n];
        }
        mpz_set(cf_qq[term], gb->modpolys[p]->lm);

        term++;
        if (print_gb == 2) {
            for(int32_t i = l-1; i >= 0; i--) {
                if(mpz_cmp_ui(gb->modpolys[p]->cf_qq[i], 0) != 0) {
                  for (int32_t n = 0; n < nve; n++) {
                    exp[term * nve + n] = ht->ev[hm[l - i]][evi[n]];
                  }
                  mpz_set(cf_qq[term], gb->modpolys[p]->cf_qq[i]);

                  term++;
                }
            }
        }
    }

    *blen = lens;
    *bexp = exp;
    *bcf  = cf_qq;

    free(evi);
    return nterms;
}

gb_modpoly_t *groebner_qq(
        data_gens_ff_t *gens,
        msflags_t flags)
{
  int32_t ht_size = flags->ht_size;
  int32_t nr_threads = flags->nr_threads;
  int32_t max_nr_pairs = flags->max_nr_pairs;
  int32_t elim_block_len = flags->elim_block_len;
  int32_t reset_ht = flags->reset_ht;
  int32_t la_option = flags->la_option;
  const int32_t use_signatures = flags->use_signatures;
  int32_t info_level = flags->info_level;
  int32_t pbm_file = flags->pbm_file;
  int32_t print_gb = flags->print_gb;
  int32_t truncate_lifting = flags->truncate_lifting;
  int mon_order = 0;

  /* input data */
  uint32_t field_char = gens->field_char;
  const void *cfs = gens->mpz_cfs;
  if(gens->field_char){
    cfs = gens->cfs;
  }
  else{
    cfs = gens->mpz_cfs;
  }
  int32_t *exps = gens->exps;
  int32_t *lens = gens->lens;

  int32_t nr_vars = gens->nvars;
  int32_t nr_gens = gens->ngens;
  int reduce_gb = 1;

  /* timings */
  double ct0, ct1, rt0, rt1;
  ct0 = cputime();
  rt0 = realtime();

  /* data structures for basis, hash table and statistics */
  bs_t *bs  = NULL;
  ht_t *bht = NULL;
  md_t *md  = NULL; /* meta data */

  int success = 0;

  /*msolve trace data */
  mstrace_t msd;
  success = initialize_gba_input_data(&bs, &bht, &md,
          lens, exps, cfs, field_char, mon_order, elim_block_len,
          nr_vars, nr_gens, 0 /* # normal forms */, ht_size,
          nr_threads, max_nr_pairs, reset_ht, la_option, use_signatures,
          reduce_gb, pbm_file, truncate_lifting, info_level);

  /* all input generators are invalid */
  if (success == -1) {
      return NULL;
  }
  if (success == 0) {
      fprintf(ERRSTREAM,"Bad input data, stopped computation.\n");
      exit(1);
  }

  md->elim_drl = gens->elim_drl;
  initialize_mstrace(msd, md, bs);
  int err = 0;

  gb_modpoly_t *modgbsp = malloc(sizeof(gb_modpoly_t));
  modgbsp = core_groebner_qq(modgbsp, bs, msd, md, &err, field_char, print_gb);
  if (err) {
      fprintf(ERRSTREAM,"Problem with groebner_qq, stopped computation (%d).\n", err);
      exit(1);
  }
  free_elim_drl_data(md);

  /* timings */
  ct1 = cputime();
  rt1 = realtime();
  md->f4_ctime = ct1 - ct0;
  md->f4_rtime = rt1 - rt0;

  get_and_print_final_statistics(VERBSTREAM, md, bs);

  /* free and clean up */
  free_mstrace(msd, md);
  free_basis_without_hash_table(&bs);
  free_trace(&md->tr);
  free(md);
  md    = NULL;

  return modgbsp;

}

/*
 Function which is called by core_msolve
 */

void print_msolve_gbtrace_qq(data_gens_ff_t *gens,
                            msflags_t flags)
{
  gb_modpoly_t *modgbsp;

  modgbsp = groebner_qq(gens, flags);

  output_timer_t ot;
  output_timer_start(&ot);
  FILE *ofile;
  if (flags->files->out_file != NULL) {
    ofile = fopen(flags->files->out_file, "wb+");
  } else {
    ofile = OUTSTREAM;
  }
  if (flags->print_gb == 1) {
    fprintf(ofile, "#Leading ideal data\n");
  } else {
    if (flags->print_gb > 1) {
      if(flags->truncate_lifting>0){
        fprintf(ofile, "#Truncated reduced Groebner basis data\n");
      }
      else{
        fprintf(ofile, "#Reduced Groebner basis data\n");
      }
    }
  }
  fprintf(ofile, "#---\n");
  fprintf(ofile, "#field characteristic: 0\n");
  fprintf(ofile, "#variable order:       ");
  for (int i = gens->elim; i < gens->nvars-1; ++i) {
    fprintf(ofile, "%s, ", gens->vnames[i]);
  }
  fprintf(ofile, "%s\n", gens->vnames[gens->nvars-1]);
  fprintf(ofile, "#monomial order:       graded reverse lexicographical\n");
  if ((*modgbsp)->ld == 1) {
    fprintf(ofile, "#length of basis:      1 element\n");
  } else {
    fprintf(ofile, "#length of basis:      %u elements sorted by increasing leading monomials\n", (*modgbsp)->ld);
  }
  fprintf(ofile, "#---\n");
  if (flags->files->out_file != NULL) {
    fclose(ofile);
  }

  if(flags->print_gb > 1){

    if(flags->files->out_file != NULL){
      FILE *ofile = fopen(flags->files->out_file, "ab+");
      display_gbmodpoly_cf_qq(ofile, (*modgbsp), gens);
      fclose(ofile);
    }
    else{
      display_gbmodpoly_cf_qq(OUTSTREAM, (*modgbsp), gens);
    }
  }
  if(flags->print_gb == 1){
    if(flags->files->out_file != NULL){
      FILE *ofile = fopen(flags->files->out_file, "ab+");
      display_lm_gbmodpoly_cf_qq(ofile, (*modgbsp), gens);
      fclose(ofile);
    }
    else{
      display_lm_gbmodpoly_cf_qq(OUTSTREAM, (*modgbsp), gens);
    }
  }
  output_timer_stop(flags->files, &ot);
  gb_modpoly_clear((*modgbsp));
  free(modgbsp);
}

/* we get from julia the generators as three arrays:
 * 0.  a pointer to an int32_t array for returning the basis to julia
 * 1.  an array of the lengths of each generator
 * 2.  an array of all coefficients of all generators in the order:
 *     first all coefficients of generator 1, then all of generator 2, ...
 * 3.  an array of all exponents of all generators in the order:
 *     first all exponents of generator 1, then all of generator 2, ...
 *
 *  RETURNs the length of the jl_basis array */
int64_t export_groebner_qq(
                void *(*mallocp) (size_t),
        /* return values */
        int32_t *bld,   /* basis load */
        int32_t **blen, /* length of each poly in basis */
        int32_t **bexp, /* basis exponent vectors */
        void **bcf,     /* coefficients of basis elements */
        /* input values */
        const int32_t *lens,
        const int32_t *exps,
        const void *cfs,
        const uint32_t field_char,
        const int32_t mon_order,
        const int32_t elim_block_len,
        const int32_t nr_vars,
        const int32_t nr_gens,
        const int32_t ht_size,
        const int32_t nr_threads,
        const int32_t max_nr_pairs,
        const int32_t reset_ht,
        const int32_t la_option,
        const int32_t print_gb,
        const int32_t reduce_gb,
        const int32_t pbm_file,
        const int32_t truncate_lifting,
        const int32_t info_level
        )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    /* data structures for basis, hash table and statistics */
    bs_t *bs  = NULL;
    ht_t *bht = NULL;
    md_t *md  = NULL; /* meta data */

    int success = 0;

    /*msolve trace data*/
    mstrace_t msd;
    success = initialize_gba_input_data(&bs, &bht, &md,
            lens, exps, cfs, field_char, mon_order, elim_block_len,
            nr_vars, nr_gens, 0 /* # normal forms */, ht_size,
            nr_threads, max_nr_pairs, reset_ht, la_option, 0 /*use_signatures*/,
            reduce_gb, pbm_file, truncate_lifting, info_level);

    /* all input generators are invalid */
    if (success == -1) {
        return_zero(bld, blen, bexp, bcf, nr_vars, field_char, mallocp);
        return 1;
    }
    if (success == 0) {
        fprintf(ERRSTREAM, "Bad input data, stopped computation.\n");
        exit(1);
    }

    initialize_mstrace(msd, md, bs);
    int err = 0;

    gb_modpoly_t *modgbsp = malloc(sizeof(gb_modpoly_t));
    modgbsp =
    core_groebner_qq(modgbsp, bs, msd, md, &err, field_char,
                     print_gb);
    if (err) {
        fprintf(ERRSTREAM, "Problem with groebner_qq, stopped computation.\n");
        exit(1);
    }

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    md->f4_ctime = ct1 - ct0;
    md->f4_rtime = rt1 - rt0;

    get_and_print_final_statistics(VERBSTREAM, md, bs);

    int64_t nterms  = export_results_from_groebner_qq(bld, blen, bexp,
                                                      bcf, mallocp, elim_block_len, print_gb, (*modgbsp));

    /* free and clean up */
    free_mstrace(msd, md);
    free_basis_without_hash_table(&bs);
    free_trace(&md->tr);
    free(md);
    md = NULL;
    gb_modpoly_clear((*modgbsp));
    free(modgbsp);

    return nterms;

}
