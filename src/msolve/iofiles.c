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

#include "getdelim.h"

#include "msolve-data.h"
#include "streams.h"

#include <errno.h>
#include <limits.h>

/* writes x in base 10 to file through FLINT's fmpz_get_str, which uses
 * FLINT's radix conversion for small and large integers (below 40 and
 * from 1400 limbs in builds with fft_small), faster than GMP's
 * mpz_out_str there, and GMP's conversion in between */
static inline void mpz_out_str_10(FILE *file, const mpz_t x){
  fmpz_t f;
  fmpz_init_set_readonly(f, x);
  char *str = fmpz_get_str(NULL, 10, f);
  fputs(str, file);
  flint_free(str);
  fmpz_clear_readonly(f);
}

/* start of an interval of time spent writing the output */
typedef struct{
  double rt;
  double ct;
} output_timer_t;

static inline void output_timer_start(output_timer_t *t){
  t->rt = realtime();
  t->ct = cputime();
}

/* adds the time elapsed since output_timer_start(t) to the time spent
 * writing the output, accumulated in files so that msolve can report the
 * computation time apart from it (see print_overall_times) */
static inline void output_timer_stop(files_gb *files, const output_timer_t *t){
  files->output_rtime += realtime() - t->rt;
  files->output_ctime += cputime() - t->ct;
}

/* prints the computation time, the time spent writing the output (stored
 * in files) and the overall time since rt0 (elapsed) and st0 (cpu) */
static inline void print_overall_times(const files_gb *files,
                                       const double rt0, const double st0){
  const double st1 = cputime();
  const double rt1 = realtime();
  fprintf(VERBSTREAM, "msolve computation time  %9.2f sec (elapsed) / %5.2f sec (cpu)\n",
          rt1 - rt0 - files->output_rtime, st1 - st0 - files->output_ctime);
  fprintf(VERBSTREAM, "msolve output time       %9.2f sec (elapsed) / %5.2f sec (cpu)\n",
          files->output_rtime, files->output_ctime);
  fprintf(VERBSTREAM, "msolve overall time  %13.2f sec (elapsed) / %5.2f sec (cpu)\n",
          rt1 - rt0, st1 - st0);
}

/* over GF(16) and GF(256) coefficients may be given in hexadecimal
 * notation, e.g. 0x1b, returns the base of the integer str and sets
 * *digits to its first digit */
static inline int coefficient_base(const char *str, int64_t field_char,
                                   const char **digits)
{
    const char *s = str;
    while (*s == ' ' || *s == '+' || *s == '-') {
        ++s;
    }
    *digits = s;
    if (is_gf2_ext(field_char) && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        *digits = s+2;
        return 16;
    }
    return 10;
}

/* returns the coefficient at the beginning of term over GF(16) resp.
 * GF(256): a hexadecimal coefficient like 0x1b is the field element with
 * this byte representation, bit i being the coefficient of t^i, an
 * integer n stands for n*1, i.e. for n modulo 2, a missing coefficient
 * for 1. Signs do not matter in characteristic 2. Returns -1 if a
 * hexadecimal coefficient is not an element of the field. */
static inline int64_t coefficient_gf2x(const char *term, int64_t field_char)
{
    const char *digits;
    const int base = coefficient_base(term, field_char, &digits);
    if (base == 16) {
        const int64_t v = (int64_t)strtoll(digits, NULL, 16);
        return v < field_char ? v : -1;
    }
    /* the parity of an integer is given by its last digit */
    const char *e = digits;
    while (*e >= '0' && *e <= '9') {
        ++e;
    }
    return e == digits ? 1 : (e[-1] - '0') & 1;
}

/* returns the integer coefficient at the beginning of term modulo the
 * prime field_char, in [0, field_char): the digits are reduced while they
 * are read, so integers of any size are taken correctly, and a missing
 * coefficient stands for 1 resp. -1 */
static inline int64_t coefficient_mod_p(const char *term, int64_t field_char)
{
    const char *s  = term;
    int neg        = 0;
    while (*s == ' ' || *s == '+' || *s == '-') {
        if (*s == '-') {
            neg ^= 1;
        }
        ++s;
    }
    const unsigned __int128 p = (unsigned __int128)field_char;
    unsigned __int128 r       = 0;
    const char *digits        = s;
    while (*s >= '0' && *s <= '9') {
        r = (r * 10 + (unsigned)(*s - '0')) % p;
        ++s;
    }
    if (s == digits) {
        r = 1 % p;
    }
    if (neg && r != 0) {
        r = p - r;
    }
    return (int64_t)r;
}

static inline int store_exponent(const char *term, data_gens_ff_t *gens, int64_t pos)
{
    len_t i, j, k;

    len_t op = 0;
    len_t index = 0;

    int res = 0;
    mpz_t tmp_z;
    mpq_t tmp_q;
    if (gens->field_char > 0) {
        mpz_init(tmp_z);
    } else {
        mpq_init(tmp_q);
    }

    char *var = NULL;
    char *ev  = NULL;
    char *end = NULL;
    const len_t tl = strlen(term);
    for (i = 0; i < tl+1; ++i) {
        if (term[i] == '*' || i == tl) {
            j = op;
            while (j < i && term[j] != '^') {
                ++j;
            }
            while(term[op] == ' ' || term[op] == '+' || term[op] == '-') {
                ++op;
            }
            if (op >= j) {
                if (index == 0 && term[op] == '\0') {
                    fprintf(stderr, "Error when parsing term %s (empty term)\n", term);
                } else {
                    fprintf(stderr, "Error when parsing term %s (missing operand)\n", term);
                }
                res = 1;
                break;
            }
            var = realloc(var, sizeof(char)*(j-op+1));
            memcpy(var, term+op, j-op);
            var[j-op] = '\0';
            int has_exponent = (j < i && term[j] == '^');
            if (has_exponent) {
                if (j + 1 >= i) {
                    fprintf(stderr, "Error when parsing term %s (missing exponent)\n", term);
                    res = 1;
                    break;
                }
                ev = realloc(ev, (sizeof(char)*(i-j)));
                ev = memcpy(ev, term+(j+1), i-j-1);
                ev[i-j-1] = '\0';
            } else {
                ev = realloc(ev, (sizeof(char)*2));
                ev[0] = '1';
                ev[1] ='\0';
            }
            int is_var = 0;
            for (k = 0; k < gens->nvars; ++k) {
                if (strcmp(gens->vnames[k], var) == 0) {
                    is_var = 1;
                    if (((gens->exps) + pos)[k] != 0) {
                        fprintf(stderr, "Error when parsing term %s (variable appears multiple times)\n", term);
                        res = 1;
                        break;
                    }
                    errno = 0;
                    long val = strtol(ev, &end, 10);
                    if (ev == end || *end != '\0') {
                        fprintf(stderr, "Error when parsing term %s (invalid exponent)\n", term);
                        res = 1;
                        break;
                    }
                    if (errno == ERANGE || val > INT_MAX) {
                        fprintf(stderr, "Error when parsing term %s (exponent out of range)\n", term);
                        res = 1;
                        break;
                    }
                    if (val <= 0) {
                        fprintf(stderr, "Error when parsing term %s (exponent must be positive)\n", term);
                        res = 1;
                        break;
                    }
                    ((gens->exps) + pos)[k] = val;
                    break;
                }
            }
            if (res == 1) {
                break;
            }
            if (is_var == 0) {
                if (gens->field_char > 0) {
                    const char *digits;
                    const int base = coefficient_base(var, gens->field_char, &digits);
                    if (mpz_set_str(tmp_z, base == 10 ? var : digits, base) != 0) {
                        fprintf(stderr, "Error when parsing term %s (unknown variable or invalid coefficient for finite field: %s)\n", term, var);
                        res = 1;
                        break;
                    }
                } else {
                    if (mpq_set_str(tmp_q, var, 10) != 0) {
                        fprintf(stderr, "Error when parsing term %s (unknown variable or invalid coefficient: %s)\n", term, var);
                        res = 1;
                        break;
                    }
                }
                if (has_exponent) {
                    fprintf(stderr, "Error when parsing term %s (coefficient cannot have an exponent)\n", term);
                    res = 1;
                    break;
                }
                if (index > 0) {
                    fprintf(stderr, "Error when parsing term %s (coefficient must be the first factor)\n", term);
                    res = 1;
                    break;
                }
            }
            op = i+1;
            index++;
        }
    }

    free(var);
    free(ev);
    if (gens->field_char > 0) {
        mpz_clear(tmp_z);
    } else {
        mpq_clear(tmp_q);
    }
    return res;
}



static inline void display_monomial(FILE *file, data_gens_ff_t *gens, int64_t pos,
                                    int32_t **bexp){
  int32_t exp = 0;
  for(int k = 0; k < gens->nvars - gens->elim; k++){
    exp = (*bexp)[(pos)*(gens->nvars - gens->elim) + k];
    if(exp > 0){
      break;
    }
  }
  if(exp == 0){
    fprintf(file, "1");
    return;
  }
  for(int k = 0; k < (gens->nvars - gens->elim); k++){
    exp = (*bexp)[(pos)*(gens->nvars - gens->elim) + k];
    if(exp > 0){
      fprintf(file, "*");
      if(exp==1){
        fprintf(file, "%s", gens->vnames[k + gens->elim]);
      }
      else{
        fprintf(file, "%s^%d",gens->vnames[k + gens->elim], exp);
      }
    }
  }
}

static inline int32_t display_monomial_single(FILE *file, data_gens_ff_t *gens,
                                              int64_t pos, int32_t **bexp){
  int32_t exp = 0;
  int32_t b = 0;
  for(int k = 0; k < gens->nvars - gens->elim; k++){
    exp = (*bexp)[(pos)*(gens->nvars - gens->elim) + k];
    if(exp > 0){
      break;
    }
  }
  if(exp == 0){
    fprintf(file, "1");
    return 0;
  }
  for(int k = 0; k < gens->nvars - gens->elim; k++){
    exp = (*bexp)[(pos)*(gens->nvars - gens->elim) + k];
    if(exp > 0){
      if(exp==1){
        if(b){
          fprintf(file, "*%s", gens->vnames[k + gens->elim]);
        }
        else{
          fprintf(file, "%s", gens->vnames[k + gens->elim]);
        }
      }
      else{
        if(b){
          fprintf(file, "*%s^%d",gens->vnames[k + gens->elim], exp);
        }
        else{
          fprintf(file, "%s^%d",gens->vnames[k + gens->elim], exp);
        }
      }
      b = 1;
    }
  }
  return b;
}


static inline int32_t display_monomial_full(FILE *file, const int nv,
                                            char **vnames,
                                            int64_t pos, int32_t *bexp){
  int32_t exp, b = 0;
  for(int k = 0; k < nv; k++){
    exp = (bexp)[(pos)*nv + k];
    if(exp > 0){
      if(exp==1){
        if(b){
          //          fprintf(file, "*%s", vnames[k]);
          fprintf(file, "*x%d", k+1);
        }
        else{
          //          fprintf(file, "%s", vnames[k]);
          fprintf(file, "x%d", k+1);
        }
      }
      else{
        if(b){
          fprintf(file, "*x%d^%d",k+1, exp);
        }
        else{
          fprintf(file, "x%d^%d",k+1, exp);
        }
      }
      b = 1;
    }
  }
  if(b==0){
    fprintf(file, "1");
  }
  return b;
}

/* output through a buffer: the polynomials of a basis are written term by
 * term, formatting each coefficient and exponent by fprintf took a large
 * part of the time of small computations */
typedef struct {
    FILE *f;
    char *b;
    size_t n;
    size_t sz;
} obuf_t;

static inline void ob_flush(
        obuf_t *o
        )
{
    if (o->n > 0) {
        fwrite(o->b, 1, o->n, o->f);
        o->n = 0;
    }
}

static inline void ob_put(
        obuf_t *o,
        const char *s,
        const size_t len
        )
{
    if (o->n + len > o->sz) {
        ob_flush(o);
        if (len > o->sz) {
            fwrite(s, 1, len, o->f);
            return;
        }
    }
    memcpy(o->b + o->n, s, len);
    o->n += len;
}

static inline void ob_char(
        obuf_t *o,
        const char c
        )
{
    if (o->n == o->sz) {
        ob_flush(o);
    }
    o->b[o->n++] = c;
}

/* the decimal digits of v */
static inline void ob_u64(
        obuf_t *o,
        uint64_t v
        )
{
    char t[20];
    int i = 20;
    do {
        t[--i] = (char)('0' + v % 10);
        v /= 10;
    } while (v != 0);
    ob_put(o, t + i, (size_t)(20 - i));
}

/* the hexadecimal digits of v after 0x */
static inline void ob_hex(
        obuf_t *o,
        uint64_t v
        )
{
    char t[18];
    int i = 18;
    do {
        t[--i] = "0123456789abcdef"[v & 15];
        v >>= 4;
    } while (v != 0);
    t[--i] = 'x';
    t[--i] = '0';
    ob_put(o, t + i, (size_t)(18 - i));
}

static void print_msolve_polynomials_ff(
        FILE *file,
        const bi_t from,
        const bi_t to,
        const bs_t * const bs,
        const ht_t * const ht,
        const md_t *st,
        char **vnames,
        const int lead_ideal_only,
        const int is_nf
        )
{
    len_t i, j, k, idx;

    len_t len   = 0;
    hm_t *hm    = NULL;

    /* over GF(16) and GF(256) the coefficients are printed as bytes in
     * hexadecimal notation, as they are read */
    const int hex = is_gf2_ext(st->fc);

    const len_t nv  = ht->nv;
    const len_t ebl = ht->ebl;
    const len_t evl = ht->evl;

    /* state context if full basis is printed */
    if (is_nf == 0 && from == 0 && to == bs->lml) {
        if (lead_ideal_only != 0) {
            fprintf(file, "#Leading ideal data\n");
        } else {
            fprintf(file, "#Reduced Groebner basis data\n");
        }
        fprintf(file, "#---\n");
        if (is_gf2_ext(st->fc)) {
            fprintf(file, "#field characteristic: 2^%u -> GF(%" PRIu64 ")",
                    st->fc == 16 ? 4 : 8, st->fc);
            if (hex) {
                fprintf(file, ", coefficient byte encoding via: %s\n",
                        st->fc == 16 ? "@^4 + @ + 1"
                        : "@^8 + @^4 + @^3 + @ + 1");
            } else {
                fprintf(file, "\n");
            }
        } else {
            /* gfc only has 32 bits */
            fprintf(file, "#field characteristic: %" PRIu64 "\n",
                    st->ff_bits == 64 ? st->fc : (uint64_t)st->gfc);
        }
        if (st->boolean) {
            fprintf(file, "#boolean ring:         x^2 = x for all variables x\n");
        }
        fprintf(file, "#variable order:       ");
        for (i = 0; i < nv-1; ++i) {
            fprintf(file, "%s, ", vnames[i]);
        }
        fprintf(file, "%s\n", vnames[nv-1]);
        if (st->nev == 0) {
            fprintf(file, "#monomial order:       graded reverse lexicographical\n");
        } else {
            if (st->nev == 1) {
                fprintf(file, "#monomial order:       eliminating first variable, blocks: graded reverse lexicographical\n");
            } else {
                fprintf(file, "#monomial order:       eliminating first %d variables, blocks: graded reverse lexicographical\n", st->nev);
            }
        }
        if (bs->lml == 1) {
            fprintf(file, "#length of basis:      1 element\n");
        } else {
            fprintf(file, "#length of basis:      %u elements sorted by increasing leading monomials\n", bs->lml);
        }
        fprintf(file, "#---\n");
    }

    int *evi    =   (int *)malloc((unsigned long)ht->nv * sizeof(int));
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

    if (lead_ideal_only != 0) {
        int ctr = 0;
        fprintf(file, "[");
        for (i = from; i < to; ++i) {
            idx = bs->lmps[i];
            if (bs->hm[idx] == NULL) {
                fprintf(file, "0,\n");
            } else {
                hm  = bs->hm[idx]+OFFSET;
                len = bs->hm[idx][LENGTH];
                ctr = 0;
                k = 0;
                while (ctr == 0 && k < nv) {
                    if (ht->ev[hm[0]][evi[k]] > 0) {
                        fprintf(file, "%s^%u",vnames[k], ht->ev[hm[0]][evi[k]]);
                        ctr++;
                    }
                    k++;
                }
                for (;k < nv; ++k) {
                    if (ht->ev[hm[0]][evi[k]] > 0) {
                        fprintf(file, "*%s^%u",vnames[k], ht->ev[hm[0]][evi[k]]);
                        ctr++;
                    }
                }
                if (ctr == 0) {
                    fprintf(file, "1");
                }
                if (i < to-1) {
                    fprintf(file, ",\n");
                } else {
                    fprintf(file, "]:\n");
                }
            }
        }
    } else {
        /* the strings "*name^" of the variables */
        size_t *vl  = (size_t *)malloc((unsigned long)nv * sizeof(size_t));
        char **vp   = (char **)malloc((unsigned long)nv * sizeof(char *));
        for (k = 0; k < nv; ++k) {
            vl[k] = strlen(vnames[k]) + 2;
            vp[k] = (char *)malloc(vl[k] + 1);
            vp[k][0] = '*';
            memcpy(vp[k] + 1, vnames[k], vl[k] - 2);
            vp[k][vl[k] - 1] = '^';
            vp[k][vl[k]] = '\0';
        }
        fflush(file);
        obuf_t ob = {file, (char *)malloc(1 << 16), 0, 1 << 16};
        ob_char(&ob, '[');
        for (i = from; i < to; ++i) {
            idx = bs->lmps[i];
            if (bs->hm[idx] == NULL) {
                ob_char(&ob, '0');
            } else {
                hm  = bs->hm[idx]+OFFSET;
                len = bs->hm[idx][LENGTH];
                const len_t ci = bs->hm[idx][COEFFS];
                for (j = 0; j < len; ++j) {
                    if (j > 0) {
                        ob_char(&ob, '+');
                    }
                    switch (st->ff_bits) {
                    case 0:
                        ob_u64(&ob, bs->cf_32[ci][j]);
                        break;
                    case 8:
                        if (hex) {
                            ob_hex(&ob, bs->cf_8[ci][j]);
                        } else {
                            ob_u64(&ob, bs->cf_8[ci][j]);
                        }
                        break;
                    case 16:
                        ob_u64(&ob, bs->cf_16[ci][j]);
                        break;
                    case 32:
                        ob_u64(&ob, bs->cf_32[ci][j]);
                        break;
                    case 64:
                        ob_u64(&ob, bs->cf_64[ci][j]);
                        break;
                    }
                    const exp_t * const e = ht->ev[hm[j]];
                    for (k = 0; k < nv; ++k) {
                        if (e[evi[k]] > 0) {
                            ob_put(&ob, vp[k], vl[k]);
                            ob_u64(&ob, e[evi[k]]);
                        }
                    }
                }
            }
            if (i < to-1) {
                ob_put(&ob, ",\n", 2);
            } else {
                ob_put(&ob, "]:\n", 3);
            }
        }
        ob_flush(&ob);
        free(ob.b);
        for (k = 0; k < nv; ++k) {
            free(vp[k]);
        }
        free(vp);
        free(vl);
    }
    free(evi);
}

static void print_ff_nf_data(
        const char *fn,
        const char *mode,
        const bi_t from,
        const bi_t to,
        const bs_t *bs,
        const ht_t *ht,
        const md_t *st,
        const data_gens_ff_t *gens,
        const int32_t print_gb)
{
    if (print_gb > 0) {
        if(fn != NULL){
            FILE *ofile = fopen(fn, mode);
            print_msolve_polynomials_ff(ofile, from, to, bs, ht,
                    st, gens->vnames, 2-print_gb, 1);
            fclose(ofile);
        }
        else{
            print_msolve_polynomials_ff(OUTSTREAM, from, to, bs, ht,
                    st, gens->vnames, 2-print_gb, 1);
        }
    }
}

static void print_ff_basis_data(
        const char *fn,
        const char *mode,
        const bs_t *bs,
        const ht_t *ht,
        const md_t *st,
        const data_gens_ff_t *gens,
        const int32_t print_gb)
{
    if (print_gb > 0) {
        if(fn != NULL){
            FILE *ofile = fopen(fn, mode);
            print_msolve_polynomials_ff(ofile, 0, bs->lml, bs, ht,
                    st, gens->vnames, 2-print_gb, 0);
            fclose(ofile);
        }
        else{
            print_msolve_polynomials_ff(OUTSTREAM, 0, bs->lml, bs, ht,
                    st, gens->vnames, 2-print_gb, 0);
        }
    }
}

static int32_t get_nvars(const char *fn)
{
    FILE * fh = fopen(fn, "r");
    char * line = NULL;
    char * line2 = NULL;
    size_t len;
    nvars_t nvars = -1;

    /* number of variables is read from first line, it is 1 + (number of commata) */
    if (getline(&line, &len, fh) != -1)
    {
        nvars = 1;
        line2 = strchr(line, ',');
        while (line2 != NULL)
        {
            nvars++;
            // line points to a comma, which must be followed by one or more characters
            // --> line+1 is valid
            line2 = strchr(line2+1, ',');
        }
    }

    free(line);
    free(line2);
    fclose(fh);

    return nvars;
}

/**
 * Checks if a null-or-delim-terminated string is just empty
 * ("only whitespaces" counts as empty)
 *
 * \return 1 if the line is empty, else 0
 */
static inline int is_line_empty(const char *line, const char delim) {
    while (*line != '\0' && *line != delim) {
        if (!isspace(*line))
            return 0;
        line++;
    }
    return 1;
}

static void trim_polynomial(char *poly);

static int32_t get_ngenerators(char *fn)
{
    int32_t ngens = 0;
    char *line  = NULL;
    size_t len;
    FILE *fh  = fopen(fn,"r");

    /* 1st and 2nd lines are ignored; still check all went fine */
    if (   getline(&line, &len, fh) == -1
        || getline(&line, &len, fh) == -1)
        ngens = -1;

    /* go through subsequent lines, not counting empty or zero ones */
    else {
        while(getdelim(&line, &len, ',', fh) != -1) {
            if (is_line_empty(line, ',')) {
                continue;
            }
            trim_polynomial(line);
            if (strcmp(line, "0") != 0) {
                ngens++;
            }
        }
    }

    free(line);
    fclose(fh);

    return ngens;
}


static inline char *get_variable_name(const char *line, char **prev_pos)
{
  const char comma_splicer  = ',';

  char *tmp_var;
  char *curr_pos  = strchr(*prev_pos, comma_splicer);
  if (curr_pos != NULL) {
    size_t var_diff   = (size_t)(curr_pos - *prev_pos);
    tmp_var = (char *)malloc((var_diff + 1) * sizeof(char));
    memcpy(tmp_var, *prev_pos, var_diff);
    tmp_var[var_diff] = '\0';
    *prev_pos = curr_pos+1;
  } else { /** we are at the last variable */
    int prev_idx    = (int)(*prev_pos - line);
    int curr_idx    = (int)(strlen(line)+1);
    size_t var_diff = (size_t)(curr_idx - prev_idx);
    tmp_var = (char *)malloc((var_diff + 1) * sizeof(char));
    memcpy(tmp_var, *prev_pos, var_diff);
    tmp_var[var_diff] = '\0';
  }

  /** trim variable, remove blank spaces */
  char *tmp_var_begin, *tmp_var_end;
  tmp_var_begin = tmp_var_end =  tmp_var;
  while (isspace(*tmp_var_begin))
    tmp_var_begin++;
  if (*tmp_var_begin != 0) {
    while (*tmp_var_end)
      tmp_var_end++;
    do {
      tmp_var_end--;
    } while (isspace(*tmp_var_end));
  }
  int fin_len  = (int)(tmp_var_end - tmp_var_begin + 1);
  size_t final_length = 0;
  if (fin_len > 0)
    final_length  = (size_t)fin_len;
  char *final_var   = (char *)malloc((final_length+1) * sizeof(char));
  memcpy(final_var, tmp_var_begin, final_length);
  final_var[final_length] = '\0';

  free(tmp_var);
  return final_var;
}

static inline nelts_t get_number_of_terms(const char *line)
{
  const char add_splicer        = '+';
  const char minus_splicer      = '-';
  const char whitespace_splicer = ' ';
  const char *tmp = NULL;
  nelts_t nterms  = 1;
  /** remove useless whitespaces at the beginning */
  int i = 0;
  while (strncmp(&whitespace_splicer, line+i, 1) == 0)
    i++;
  /** check if first non-whitespace char is "-", set term counter -1 in this case */
  if (strncmp(&minus_splicer, line+i, 1) == 0)
    nterms--;
  /** now count terms */
  tmp = strchr(line, add_splicer);
  while (tmp != NULL) {
    nterms++;
    tmp = strchr(tmp+1, add_splicer);
  }
  tmp = strchr(line, minus_splicer);
  while (tmp != NULL) {
    nterms++;
    tmp = strchr(tmp+1, minus_splicer);
  }
  //MS que faire si dans la ligne on a 0*x+0*y ?
  return nterms;
}


static void get_variables(FILE *fh, char * line, int max_line_size,
                          int32_t *nr_vars, data_gens_ff_t *gens, char **vnames){
  long i;
  char *tmp = NULL;
  if (fgets(line, max_line_size, fh) != NULL) {
    tmp = line;
  } else {
    fprintf(ERRSTREAM,"Bad file format (variable names).\n");
    /* free(vnames);
     * free(line); */
    fclose(fh);
    exit(1);
  }
  for (i=0; i<(*nr_vars); ++i) {
    vnames[i]  = get_variable_name(line, &tmp);
  }
  gens->vnames = vnames;

}

static void get_characteristic(FILE *fh, char * line, int max_line_size,
                               int64_t *field_char, int32_t *boolean,
                               char **vnames){
  *field_char = 0;
  *boolean    = 0;

  if(fgets(line, max_line_size, fh) != NULL){
    /* the Boolean ring over GF(2) is given as boolean */
    const char *b = line;
    while (*b == ' ') {
      ++b;
    }
    if (strncmp(b, "boolean", 7) == 0) {
      *field_char = 2;
      *boolean    = 1;
      return;
    }
    char *end = NULL;
    int64_t tmp_mod = strtoll(line, &end, 10);
    /* the binary extension fields are given as 2^4 resp. 2^8 */
    while (*end == ' ') {
      ++end;
    }
    if (*end == '^') {
      const int64_t e = strtoll(end+1, NULL, 10);
      if (tmp_mod != 2 || (e != 1 && e != 4 && e != 8)) {
        fprintf(ERRSTREAM, "Bad file format (only 2^4 and 2^8 are supported as fields of non-prime order)\n");
        free(vnames);
        free(line);
        fclose(fh);
        exit(1);
      }
      tmp_mod = (int64_t)1 << e;
    }
    if(tmp_mod >= 0){
      *field_char = tmp_mod;
      if(tmp_mod >= ((int64_t)1 << 52)){
        fprintf(ERRSTREAM, "Warning: characteristic must be < 2^52\n");
        free(vnames);
        free(line);
        fclose(fh);
        exit(1);
      }
    }
    else{
      fprintf(ERRSTREAM, "Bad file format (characteristic)\n");
      free(vnames);
      free(line);
      fclose(fh);
      exit(1);
    }
  }

}

static void trim_polynomial(char *poly) {
    /*
     * WW: The function trims the input, supposedly a polynomial,
     * in a possibly too trivial way, but is left as is for
     * backward compatibility.
     *
     * For example, trim_polynomial converts "x 1, pi/2" to "x1",
     * even when the former is not a valid polynomial.
     * This actually mimics the behavior of msolve v0.7.5,
     * for which the following is a valid input:
     *   x1
     *   0
     *   x 1, pi/2
     *
     * Ideally, in a future version, we should throw an error if the
     * input is not a valid polynomial.
     */
    size_t k = 0;
    for (size_t j = 0; poly[j] != '\0'; j++) {
        if (poly[j] == ',') {
            break;
        }
        if (!isspace(poly[j])) {
            poly[k++] = poly[j];
        }
    }
    poly[k] = '\0';
}

static void remove_newlines(char *line, ssize_t *len) {
    while (*len && line[*len - 1] == '\0') {
        (*len)--;
    }
    while (*len && isspace(line[*len - 1])) {
        line[--(*len)] = '\0';
    }
    for (size_t i = 0; i < *len; i++) {
        if (line[i] == '\n' || line[i] == '\r') {
            line[i] = ' ';
        }
    }
}

static void remove_trailing_delim(char *line, ssize_t *len, char delim) {
    if (*len > 0 && line[*len - 1] == delim) {
        line[--(*len)] = '\0';
    }
}

static void get_nterms_and_all_nterms(FILE *fh,
                                      int max_line_size, data_gens_ff_t *gens,
                                      int32_t *nr_gens, nelts_t *nterms, nelts_t *all_nterms){

    char *line  = NULL;
    size_t size;
    ssize_t len;
    for (int32_t i = 0; i < *nr_gens; i++) {
        do {
            len = getdelim(&line, &size, ',', fh);
        } while (len != -1 && is_line_empty(line, ','));
        if (i < *nr_gens - 1) {
            remove_trailing_delim(line, &len, ',');
        }
        remove_newlines(line, &len);
        trim_polynomial(line);
        if (strcmp(line, "0") == 0) {
            i--;
            continue;
        }
        *nterms  = get_number_of_terms(line);
        gens->lens[i] = *nterms;
        *all_nterms += *nterms;
    }
    free(line);
    gens->nterms = *all_nterms;
}


static inline void get_term(const char *line, char **prev_pos,
                            char **term, size_t *term_size)
{
  /* the term ends before the next "+" or "-", a "-" at the beginning of
   * the line is the sign of the first term; the line is scanned only up
   * to this position, otherwise reading a polynomial without "-" would
   * take quadratic time */
  const char *start_pos = *prev_pos != line ? *prev_pos - 1 : *prev_pos;
  const char *curr_pos  = *prev_pos;
  if (curr_pos == line && *curr_pos == '-') {
    ++curr_pos;
  }
  curr_pos  +=  strcspn(curr_pos, "+-");

  /* the last term is copied with its terminating null character */
  const size_t term_diff  = (size_t)(curr_pos - start_pos)
    + (*curr_pos == '\0' ? 1 : 0);
  if (term_diff >= *term_size) {
    fprintf(ERRSTREAM, "Too large input integers, exit...\n");
    exit(1);
  }
  memcpy(*term, start_pos, term_diff);
  (*term)[term_diff]  = '\0';
  if (*curr_pos != '\0') {
    *prev_pos = (char *)curr_pos + 1;
  }
}


/*assumes that coeffs in file fit in word size */
static int get_coefficient_ff_and_term_from_line(char *line, int32_t nterms,
                                          int64_t field_char,
                                          data_gens_ff_t *gens, int64_t pos){
  char *prev_pos = NULL;
  size_t term_size = 50000;
  char *term  = (char *)malloc(term_size * sizeof(char));
  int64_t cf_tmp  = 0; /** temp for coefficient value, possibly coeff is negative. */

  prev_pos = line;
  get_term(line, &prev_pos, &term, &term_size);
  if(term != NULL){
    int64_t iv_tmp;
    if (is_gf2_ext(field_char)) {
      /* zero coefficients are possible, e.g. for even integers, neogb
       * removes these terms */
      iv_tmp = coefficient_gf2x(term, field_char);
      if (iv_tmp < 0) {
        fprintf(stderr, "Error when parsing term %s (coefficient is not an element of GF(%" PRId64 ")).\n", term, field_char);
        free(term);
        return 1;
      }
    } else {
    iv_tmp  = coefficient_mod_p(term, field_char);
    if (iv_tmp == 0) {
      fprintf(stderr, "Error when parsing term %s (coefficient cannot be 0 modulo %" PRId64 ").\n", term, field_char);
      free(term);
      return 1;
    }
    }
    if (gens->cfs_64 != NULL) {
      gens->cfs_64[pos] = iv_tmp;
    } else {
      gens->cfs[pos]  = (int32_t)iv_tmp;
    }
    if (store_exponent(term, gens, pos * (int64_t)gens->nvars)) {
      free(term);
      return 1;
    }
    for(int64_t j = 1; j < nterms; j++){
      get_term(line, &prev_pos, &term, &term_size);
      if (term != NULL) {
        if (is_gf2_ext(field_char)) {
          cf_tmp = coefficient_gf2x(term, field_char);
          if (cf_tmp < 0) {
            fprintf(stderr, "Error when parsing term %s (coefficient is not an element of GF(%" PRId64 ")).\n", term, field_char);
            free(term);
            return 1;
          }
        } else {
        cf_tmp  = coefficient_mod_p(term, field_char);
        if (cf_tmp == 0) {
          fprintf(stderr, "Error when parsing term %s (coefficient cannot be 0 modulo %" PRId64 ").\n", term, field_char);
          free(term);
          return 1;
        }
        }
        if (gens->cfs_64 != NULL) {
          gens->cfs_64[pos+j] = cf_tmp;
        } else {
          gens->cfs[pos+j] = (int32_t)cf_tmp;
        }
        if (store_exponent(term, gens, (pos + j) * (int64_t)gens->nvars)) {
          free(term);
          return 1;
        }
      }
      //      store_exponent(term, basis, ht);
    }
    free(term);
    return 0;
  }
  free(term);
  return 1;
}

static void beginning_strterm_to_mpz(char *str, mpz_t *num, mpz_t *den){
  /* fprintf(ERRSTREAM, "TERM = %s\n", str); */
  mpq_t tmp;
  mpq_init(tmp);
  mpz_set_ui(*num, 1);
  mpz_set_ui(*den, 1);
  size_t m = 0;
  size_t len = strlen(str);
  for(m = 0; m < len; m++){
    if(str[m]=='*' || str[m]==',') break;
  }
  const size_t sz = m;
  char *buf = (char *)malloc((sz+1) * sizeof(char));
  for(long i = 0; i < sz; i++){
    buf[i]=str[i];
  }
  buf[sz]='\0';

  int b = mpq_set_str(tmp, buf, 10);
  mpq_get_num(*num, tmp);
  mpq_get_den(*den, tmp);
  if(b!=0) {
    if (buf[0] == '-') {
      mpz_set_si(*num, -1);
      mpz_set_si(*den, 1);
    } else {
      mpz_set_si(*num, 1);
      mpz_set_si(*den, 1);
    }
  }
  free(buf);
  mpq_clear(tmp);
}

static void inner_strterm_to_mpz(char *str, mpz_t *num, mpz_t *den){
  mpq_t tmp;
  mpq_init(tmp);
  mpz_set_ui(*num, 1);
  mpz_set_ui(*den, 1);
  size_t m = 0;
  size_t len = strlen(str);
  for(m = 0; m < len; m++){
    if(str[m]=='*' || str[m]==',') break;
  }
  const size_t sz = m;
  char *buf = (char *)malloc((sz+1) * sizeof(char));
  for(long i = 1; i < sz; i++){
    buf[i-1]=str[i];
  }
  buf[sz-1]='\0';

  int b = mpq_set_str(tmp, buf, 10);
  mpq_get_num(*num, tmp);
  mpq_get_den(*den, tmp);
  if(b!=0) {
    mpz_set_si(*num, 1);
    mpz_set_si(*den, 1);
  }
  if(str[0]=='-'){
    mpz_neg(*num, *num);
  }
  free(buf);
  mpq_clear(tmp);
}

static int get_coefficient_mpz_and_term_from_line(char *line, int32_t nterms,
                                          int32_t field_char,
                                          data_gens_ff_t *gens, int64_t pos){
  char *prev_pos = NULL;
  size_t term_size = 50000;
  char *term  = (char *)malloc(term_size * sizeof(char));

  prev_pos = line;
  get_term(line, &prev_pos, &term, &term_size);
  if(term != NULL){

    beginning_strterm_to_mpz(term, gens->mpz_cfs[pos], gens->mpz_cfs[pos+1]);
    if (store_exponent(term, gens, (pos / 2) * (int64_t)gens->nvars)) {
      free(term);
      return 1;
    }
    for(int64_t j = 2; j < 2*(int64_t)nterms; j+=2){
      get_term(line, &prev_pos, &term, &term_size);
      inner_strterm_to_mpz(term, gens->mpz_cfs[pos+j], gens->mpz_cfs[pos+j+1]);
      if (store_exponent(term, gens, ((pos + j) / 2) * (int64_t)gens->nvars)) {
        free(term);
        return 1;
      }
    }
    free(term);
    return 0;
  }
  free(term);

  return 1;
}

static void get_coeffs_and_exponents_ff32(FILE *fh, nelts_t all_nterms,
        int32_t *nr_gens, data_gens_ff_t *gens){
    int64_t pos = 0;
    size_t size;
    ssize_t len;

    char *line  = NULL;
    if(getline(&line, &size, fh) !=-1){
    }
    if(getline(&line, &size, fh) !=-1){
    }

    if (gens->field_char > INT32_MAX) {
        gens->cfs_64 = (int64_t *)(malloc(sizeof(int64_t) * (size_t)all_nterms));
    } else {
        gens->cfs = (int32_t *)(malloc(sizeof(int32_t) * (size_t)all_nterms));
    }
    gens->exps = (int32_t *)calloc((size_t)all_nterms * (size_t)gens->nvars, sizeof(int32_t));
    for (int32_t i = 0; i < *nr_gens; i++) {
        do {
            len = getdelim(&line, &size, ',', fh);
        } while (len != -1 && is_line_empty(line, ','));
        if (i < *nr_gens - 1) {
            remove_trailing_delim(line, &len, ',');
        }
        remove_newlines(line, &len);
        trim_polynomial(line);
        if (strcmp(line, "0") == 0) {
            i--;
            continue;
        }
        if(get_coefficient_ff_and_term_from_line(line, gens->lens[i], gens->field_char,
                    gens, pos)){
            fprintf(ERRSTREAM, "Error when reading file (exit but things need to be free-ed)\n");
            free(line);
            fclose(fh);
            exit(1);
        }
        pos += gens->lens[i];
    }
    free(line);
}


static void get_coeffs_and_exponents_mpz(FILE *fh, nelts_t all_nterms,
        int32_t *nr_gens, data_gens_ff_t *gens){
    int64_t pos = 0;
    size_t size;
    ssize_t len;

    char *line  = NULL;
    if(getline(&line, &size, fh) !=-1){
    }
    if(getline(&line, &size, fh) !=-1){
    }

    gens->cfs = (int32_t*)(malloc(sizeof(int32_t) * (size_t)all_nterms));

    gens->mpz_cfs = (mpz_t **)(malloc(sizeof(mpz_t *) * 2 * (size_t)all_nterms));
    for(long i = 0; i < 2 * (long)all_nterms; i++){
      gens->mpz_cfs[i]  = (mpz_t *)malloc(sizeof(mpz_t));
      mpz_init(*(gens->mpz_cfs[i]));
    }

    gens->exps = (int32_t *)calloc((size_t)all_nterms * (size_t)gens->nvars, sizeof(int32_t));
    for (int32_t i = 0; i < *nr_gens; i++) {
        do {
            len = getdelim(&line, &size, ',', fh);
        } while (len != -1 && is_line_empty(line, ','));
        if (i < *nr_gens - 1) {
            remove_trailing_delim(line, &len, ',');
        }
        remove_newlines(line, &len);
        trim_polynomial(line);
        if (strcmp(line, "0") == 0) {
            i--;
            continue;
        }
        if(get_coefficient_mpz_and_term_from_line(line, gens->lens[i], gens->field_char,
                    gens, pos)){
            fprintf(ERRSTREAM, "Error when reading file (exit but things need to be free-ed)\n");
            free(line);
            fclose(fh);
            exit(1);
        }
        pos += 2 * gens->lens[i];
    }
    free(line);
}



static inline void initialize_data_gens(int32_t nvars, int32_t ngens, int64_t field_char, data_gens_ff_t *gens){
  gens->nvars = nvars;
  gens->ngens = ngens;
  gens->field_char = field_char;
  gens->change_var_order  = -1;
  gens->linear_form_base_coef = 0;
  gens->rand_linear = 0;
  gens->lens = (int32_t *)malloc(sizeof(int32_t) * ngens);
}

static int duplicate_vnames(char **vnames, int32_t nvars) {
  int32_t i, j;

  for (i = 1; i < nvars; ++i) {
    for (j = 0; j < i; ++j) {
      if (strcmp(vnames[i], vnames[j]) == 0) {
        fprintf(ERRSTREAM, "Duplicate variable name %s in input file.\n", vnames[i]);
        return 1;
      }
    }
  }

  return 0;
}

//nr_gens is a pointer to the number of generators
static inline void get_data_from_file(char *fn, int32_t *nr_vars,
                                      int64_t *field_char,
                                      int32_t *nr_gens, data_gens_ff_t *gens){
  *nr_vars = get_nvars(fn);
  if (*nr_vars == -1)
    fprintf(ERRSTREAM,"Bad file format (first line).\n");

  *nr_gens = get_ngenerators(fn);
  if (*nr_gens == -1)
    fprintf(ERRSTREAM,"Bad file format (generators).\n");

  const int max_line_size  = 1073741824;
  char *line  = (char *)malloc((nelts_t)max_line_size * sizeof(char));

  FILE *fh  = fopen(fn,"r");

  /** allocate memory for storing variable names */
  char **vnames = (char **)malloc((*nr_vars) * sizeof(char *));
  get_variables(fh, line, max_line_size, nr_vars, gens, vnames);
  if (duplicate_vnames(vnames, *nr_vars) == 1) {
      for(int32_t i = 0; i < *nr_vars; i++){
          free(vnames[i]);
      }
      free(vnames);
      free(line);
      exit(1);
  }
  int32_t boolean = 0;
  get_characteristic(fh, line, max_line_size, field_char, &boolean, vnames);

  initialize_data_gens(*nr_vars, *nr_gens, *field_char, gens);
  gens->boolean = boolean;

  nelts_t nterms, all_nterms = 0;
  get_nterms_and_all_nterms(fh, max_line_size, gens, nr_gens,
                            &nterms, &all_nterms);

  fclose(fh);
  fh = fopen(fn, "r");

  if(gens->field_char){
    get_coeffs_and_exponents_ff32(fh, all_nterms, nr_gens, gens);
  }
  else{
    get_coeffs_and_exponents_mpz(fh, all_nterms, nr_gens, gens);
  }

  free(line);
  fclose(fh);

  return;
}

static inline void display_gens_ff(FILE *fh, data_gens_ff_t *gens){
  int64_t pos = 0;
  int64_t c;
  for(long i = 0; i < gens->ngens; i++){
    for(long j = 0; j < gens->lens[i]-1; j++){
      c = gens->cfs_64 != NULL ? gens->cfs_64[pos+j] : gens->cfs[pos+j];
      if(c != 1){
        fprintf(fh, "%" PRId64, c);
        display_monomial(fh, gens, pos+j, &gens->exps);
      }
      else{
        display_monomial_single(fh, gens, pos+j, &gens->exps);
      }
      fprintf(fh, "+");
    }
    c = gens->cfs_64 != NULL ? gens->cfs_64[pos+gens->lens[i]-1]
        : gens->cfs[pos+gens->lens[i]-1];
    if(c!= 1){
      fprintf(fh, "%" PRId64 "*", c);
      display_monomial(fh, gens, pos+gens->lens[i]-1, &gens->exps);
    }
    else{
      int b = display_monomial_single(fh, gens, pos+gens->lens[i]-1, &gens->exps);
      if(b==0){
        fprintf(fh, "1");
      }
    }
    pos+=gens->lens[i];
    if(i < gens->ngens-1){
      fprintf(fh,",\n");
    }
    else{
      fprintf(fh,"\n");
    }
  }
}

static void display_gens_mpz(FILE *fh, data_gens_ff_t *gens){
  long pos = 0, posex = 0;
  for(long i = 0; i < gens->ngens; i++){
    for(long j = 0; j < gens->lens[i]-1; j++){
      if(mpz_cmp_ui(*(gens->mpz_cfs[pos+2*j]), 1) != 0){
        mpz_out_str(fh, 10, *(gens->mpz_cfs[pos+2*j]));
        display_monomial(fh, gens, posex+j, &gens->exps);
      }
      else{
        display_monomial_single(fh, gens, posex+j, &gens->exps);
      }
      if(mpz_cmp_ui(*(gens->mpz_cfs[pos+2*(j+1)]), 0) > 0){
        fprintf(fh, "+");
      }
    }
    if(mpz_cmp_ui(*(gens->mpz_cfs[pos+ 2*(gens->lens[i] -1)]), 1) != 0){
      mpz_out_str(fh, 10, *(gens->mpz_cfs[pos+2*(gens->lens[i]-1)]));
      display_monomial(fh, gens, posex+(gens->lens[i]-1), &gens->exps);
    }
    else{
      int b = display_monomial_single(fh, gens, posex+gens->lens[i]-1, &gens->exps);
      if(b==0){
        fprintf(fh, "1");
      }
    }
    pos+=2*gens->lens[i];
    posex+=gens->lens[i];
    if(i < gens->ngens-1){
      fprintf(fh,",\n");
    }
    else{
      fprintf(fh,"\n");
    }
  }
}

static inline void display_gens(FILE *fh, data_gens_ff_t *gens){
  for(int i = 0; i < gens->nvars-1; i++){
    fprintf(fh, "%s, ", gens->vnames[i]);
  }
  fprintf(fh, "%s\n", gens->vnames[gens->nvars-1]);
  fprintf(fh, "%" PRId64 "\n", gens->field_char);
  if(gens->field_char > 0){
    display_gens_ff(fh, gens);
  }
  else{
    display_gens_mpz(fh, gens);
  }
}
