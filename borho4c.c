/*
 * borho4c.c — crible rapide (C + GMP) pour borho4.py
 * ==================================================
 * Pour une paire amicale donnee par ses deux factorisations ("3^2*5*7*..."), enumere les couples
 * de graines (a1, a2) exactement comme graines_paire() de borho4.py (meme ordre) et refait le
 * crible de theoreme1() :
 *   a1 = a2, D <= 0, condition necessaire du theoreme 2, congruences sur d1 modulo 2D + CRT,
 *   filtre N = c^2 (mod L), cas p = 2 (« extras »), enumeration des diviseurs d1 = c (mod L)
 *   par rencontre au milieu, d1 <= d2, classe de d2, integralite des quatre p_ij, conditions
 *   p_i1 != p_i2, p >= 2, p ne divise pas a_i, primalite (probable) des quatre p_ij.
 * Un couple n'est renvoye a Python (indices de bases i1, i2) que s'il possede un d1 qui passe
 * TOUT cela, ou s'il a trop de diviseurs : Python le refait alors entierement avec theoreme1()
 * (r_i, verification complete du cycle, statut exact). C'est un FILTRE : il ne fabrique aucun
 * cycle et ne peut pas en perdre (un test probabiliste ne declare jamais compose un premier).
 * Facteurs de la paire au-dela de 64 bits : test de Miller-Rabin en base 2 seulement (5 fois moins cher) ;
 * le test complet n'est refait que si la paire signale un couple. Un facteur compose qui tromperait la
 * base 2 (jamais observe par hasard a ces tailles) ferait cribler la paire telle quelle, sans cycle
 * possible a la sortie, au lieu de la renvoyer a Python pour etre depliee.
 *
 * Compilation (faite automatiquement par borho4.py au premier lancement) :
 *   gcc -O2 -shared -fPIC -o libborho4c.so borho4c.c -lgmp -lm
 *
 * int borho_paire(const char *s1, const char *s2, int retire, int total, int sauf, long max_div,
 *                 int *stats, int *out, int max_out)
 *   total : si > 0, couples avec k1 + k2 <= total seulement ; sauf : si > 0, saute les couples
 *           k1 <= sauf et k2 <= sauf (deja faits par un run --retire sauf)
 *   stats[0..5] (ajoutes) : couples, "a1 = a2", "D <= 0", "theoreme 2", "aucun d1",
 *                           "ok" (couples entierement cribles ici, sans candidat)
 *   out : paires (i1, i2) d'indices de bases des couples a refaire en Python
 *   retour : nombre de couples a refaire (>= 0),
 *            -1 facteur non premier / chaine illisible / trop de premiers (Python reprend la paire),
 *            -2 trop de couples ou de bases (idem).
 */
#include <gmp.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAXP 64            /* premiers distincts par membre (masques sur 64 bits) */
#define MAXB 4096          /* bases par membre */

typedef struct { mpz_t p; unsigned long e; } fac_t;

/* ------------------------------------------------------------------ lecture "p^e*q*..." */
static int lire(const char *s, fac_t *f, int *nf) {
    int n = 0;                     /* *nf suit n : en cas d'erreur, les facteurs lus sont liberes */
    char buf[8192];
    *nf = 0;
    while (*s) {
        while (*s == ' ' || *s == '\t' || *s == '*' || *s == '\r' || *s == '\n') s++;
        if (!*s) break;
        int k = 0;
        while (*s >= '0' && *s <= '9') { if (k >= 8191) return -1; buf[k++] = *s++; }
        if (k == 0) return -1;
        buf[k] = 0;
        while (*s == ' ') s++;
        unsigned long e = 1;
        if (*s == '^') {
            s++;
            while (*s == ' ') s++;
            if (!(*s >= '0' && *s <= '9')) return -1;
            e = 0;
            while (*s >= '0' && *s <= '9') { e = e * 10 + (unsigned long)(*s - '0'); if (e > 100000UL) return -1; s++; }
        }
        while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
        if (*s && *s != '*') return -1;
        mpz_t p;
        mpz_init_set_str(p, buf, 10);
        int j;
        for (j = 0; j < n; j++)
            if (mpz_cmp(f[j].p, p) == 0) { f[j].e += e; break; }   /* p repete : exposants cumules */
        if (j < n) { mpz_clear(p); continue; }
        if (n >= MAXP) { mpz_clear(p); return -1; }
        mpz_init_set(f[n].p, p);
        f[n].e = e;
        mpz_clear(p);
        *nf = ++n;
    }
    return 0;
}

static void liberer(fac_t *f, int n) {
    for (int i = 0; i < n; i++) mpz_clear(f[i].p);
}

/* p^e et sigma(p^e) */
static void puiss(mpz_t q, mpz_t sq, mpz_srcptr p, unsigned long e) {
    if (e == 1) { mpz_set(q, p); mpz_add_ui(sq, p, 1); return; }
    mpz_pow_ui(q, p, e);
    mpz_mul(sq, q, p);
    mpz_sub_ui(sq, sq, 1);
    mpz_t pm;
    mpz_init(pm);
    mpz_sub_ui(pm, p, 1);
    mpz_divexact(sq, sq, pm);
    mpz_clear(pm);
}

static int premier(mpz_srcptr p) { return mpz_probab_prime_p(p, 2) != 0; }

/* Test de Miller-Rabin en base 2 (n impair > 2) : une seule exponentiation modulaire, ~5 fois moins
   cher que mpz_probab_prime_p (qui ajoute un test de Lucas fort). Ne declare jamais compose un premier. */
static int sprp2(mpz_srcptr n) {
    if (mpz_even_p(n)) return 0;
    mpz_t d, x, nm;
    mpz_inits(d, x, nm, NULL);
    mpz_sub_ui(nm, n, 1);
    mp_bitcnt_t s = mpz_scan1(nm, 0);
    mpz_tdiv_q_2exp(d, nm, s);
    mpz_set_ui(x, 2);
    mpz_powm(x, x, d, n);
    int ok = mpz_cmp_ui(x, 1) == 0 || mpz_cmp(x, nm) == 0;
    for (mp_bitcnt_t r = 1; !ok && r < s; r++) {
        mpz_mul(x, x, x);
        mpz_mod(x, x, n);
        if (mpz_cmp(x, nm) == 0) ok = 1;
        else if (mpz_cmp_ui(x, 1) <= 0) break;
    }
    mpz_clears(d, x, nm, NULL);
    return ok;
}

/* Facteurs de la paire : test complet si p tient sur 64 bits (peu couteux), sinon base 2 seulement et
   *gros = 1 ; borho_paire refait alors le test complet avant de signaler un couple a Python. */
static int premier_vite(mpz_srcptr p, int *gros) {
    if (mpz_sizeinbase(p, 2) <= 64) return premier(p);
    *gros = 1;
    return sprp2(p);
}

static int grand_sprp2(mpz_srcptr p) {             /* 1 si p <= 2^64 (laisse au test complet) */
    return mpz_sizeinbase(p, 2) <= 64 || sprp2(p);
}

static double lg2(mpz_srcptr x) {                  /* log2(x), x > 0, sans debordement */
    long e;
    double d = mpz_get_d_2exp(&e, x);
    return log2(d) + (double)e;
}

/* Petit facteur premier (3..101) ? Deux restes machine au lieu d'une exponentiation modulaire : sert a
   ecarter les quadruplets (p11, p12, p21, p22) dont UN membre est visiblement compose avant de lancer le
   premier test de primalite. Seulement pour p > 2^64 (un petit p peut etre lui-meme un de ces premiers). */
#if ULONG_MAX >= 0xFFFFFFFFFFFFFFFFUL
#define PP1 16294579238595022365UL                              /* 3*5*7*...*53 */
#define PP2 (59UL * 61 * 67 * 71 * 73 * 79 * 83 * 89 * 97 * 101)
static int pgcd1(unsigned long a, unsigned long b) {            /* pgcd(a, b) == 1 ? */
    while (b) { unsigned long r = a % b; a = b; b = r; }
    return a == 1;
}
static int petit_facteur(mpz_srcptr p, unsigned long pp) {
    return mpz_size(p) > 1 && !pgcd1(pp, mpz_fdiv_ui(p, pp));
}
#else
#define PP1 0UL
#define PP2 0UL
static int petit_facteur(mpz_srcptr p, unsigned long pp) { (void)p; (void)pp; return 0; }
#endif

/* ------------------------------------------------------------------ bases (comme _bases) */
typedef struct { int k; uint64_t retires; mpz_t v, s; } base_t;     /* retires : indices dans f */

static const fac_t *tri_f;
static int cmp_idx(const void *a, const void *b) {
    return mpz_cmp(tri_f[*(const int *)a].p, tri_f[*(const int *)b].p);
}

/* n, s du membre entier ; m = indices (dans f) des premiers propres, tries par valeur ;
   combinaisons lexicographiques de k = 1..min(retire, |m|) indices. Renvoie le nombre de bases,
   ou -1 - (bases deja creees) si MAXB est depasse. */
static int bases(const fac_t *f, int nf, const int *m, int nm, int retire, base_t *out) {
    mpz_t n, s, q, sq, t1, t2;
    mpz_inits(n, s, q, sq, t1, t2, NULL);
    mpz_set_ui(n, 1);
    mpz_set_ui(s, 1);
    for (int i = 0; i < nf; i++) {
        puiss(q, sq, f[i].p, f[i].e);
        mpz_mul(n, n, q);
        mpz_mul(s, s, sq);
    }
    int nb = 0, kmax = retire < nm ? retire : nm;
    int idx[MAXP];
    mpz_t qm[MAXP], sm[MAXP];                         /* p^e et sigma(p^e) des premiers propres : une fois */
    for (int i = 0; i < nm; i++) {
        mpz_init(qm[i]); mpz_init(sm[i]);
        puiss(qm[i], sm[i], f[m[i]].p, f[m[i]].e);
    }
    for (int k = 1; k <= kmax; k++) {
        for (int i = 0; i < k; i++) idx[i] = i;
        for (;;) {
            if (nb >= MAXB) { nb = -1 - nb; goto fin; }
            mpz_set_ui(t1, 1);
            mpz_set_ui(t2, 1);
            uint64_t r = 0;
            for (int i = 0; i < k; i++) {
                mpz_mul(t1, t1, qm[idx[i]]);
                mpz_mul(t2, t2, sm[idx[i]]);
                r |= (uint64_t)1 << m[idx[i]];
            }
            out[nb].k = k;
            out[nb].retires = r;
            mpz_init(out[nb].v);
            mpz_init(out[nb].s);
            mpz_divexact(out[nb].v, n, t1);
            mpz_divexact(out[nb].s, s, t2);
            nb++;
            int i = k - 1;
            while (i >= 0 && idx[i] == nm - k + i) i--;
            if (i < 0) break;
            idx[i]++;
            for (int j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
        }
    }
fin:
    for (int i = 0; i < nm; i++) { mpz_clear(qm[i]); mpz_clear(sm[i]); }
    mpz_clears(n, s, q, sq, t1, t2, NULL);
    return nb;
}

static void liberer_bases(base_t *b, int n) {
    for (int i = 0; i < n; i++) { mpz_clear(b[i].v); mpz_clear(b[i].s); }
}

/* ------------------------------------------------------------------ arithmetique (comme Python) */
/* a*x = b (mod D) : x = c (mod M) ; 0 si aucune solution (_congruence) */
static int congruence(mpz_t c, mpz_t M, mpz_srcptr a, mpz_srcptr b, mpz_srcptr D) {
    mpz_t g, r;
    mpz_inits(g, r, NULL);
    mpz_gcd(g, a, D);
    int ok = mpz_divisible_p(b, g);
    if (ok) {
        mpz_divexact(M, D, g);
        if (mpz_cmp_ui(M, 1) > 0) {
            mpz_divexact(r, a, g);
            mpz_invert(r, r, M);                    /* pgcd(a/g, D/g) = 1 */
            mpz_divexact(c, b, g);
            mpz_mul(c, c, r);
            mpz_mod(c, c, M);
        } else
            mpz_set_ui(c, 0);
    }
    mpz_clears(g, r, NULL);
    return ok;
}

/* restes chinois (_crt) : resultat dans (c1, m1) ; 0 si incompatible */
static int crt(mpz_t c1, mpz_t m1, mpz_srcptr c2, mpz_srcptr m2) {
    mpz_t g, d, m2g, k;
    mpz_inits(g, d, m2g, k, NULL);
    mpz_gcd(g, m1, m2);
    mpz_sub(d, c2, c1);
    int ok = mpz_divisible_p(d, g);
    if (ok) {
        mpz_divexact(m2g, m2, g);
        if (mpz_cmp_ui(m2g, 1) > 0) {
            mpz_divexact(k, m1, g);
            mpz_invert(k, k, m2g);
            mpz_divexact(d, d, g);
            mpz_mul(k, k, d);
            mpz_mod(k, k, m2g);
        } else
            mpz_set_ui(k, 0);
        mpz_addmul(c1, m1, k);
        mpz_mul(m1, m1, m2g);                        /* l = m1/g*m2 = m1*(m2/g) */
        mpz_mod(c1, c1, m1);
    }
    mpz_clears(g, d, m2g, k, NULL);
    return ok;
}

/* ------------------------------------------------------------------ crible d'un couple */
typedef struct {
    mpz_t a1, a2, N, D, A1, A2, s1, s2, c, L, rac;   /* A1 = t2*s1, A2 = t1*s2, rac = isqrt(N) */
    mpz_t ex[4];
    int nex, avec_cl;
    double lrac;
    mpz_t d1, d2, n1, n2, p11, p12, p21, p22, r;     /* temporaires */
} ctx_t;

static void ctx_init(ctx_t *C) {
    mpz_inits(C->a1, C->a2, C->N, C->D, C->A1, C->A2, C->s1, C->s2, C->c, C->L, C->rac,
              C->ex[0], C->ex[1], C->ex[2], C->ex[3],
              C->d1, C->d2, C->n1, C->n2, C->p11, C->p12, C->p21, C->p22, C->r, NULL);
}

static void ctx_clear(ctx_t *C) {
    mpz_clears(C->a1, C->a2, C->N, C->D, C->A1, C->A2, C->s1, C->s2, C->c, C->L, C->rac,
               C->ex[0], C->ex[1], C->ex[2], C->ex[3],
               C->d1, C->d2, C->n1, C->n2, C->p11, C->p12, C->p21, C->p22, C->r, NULL);
}

/* La boucle de theoreme1 pour le d1 contenu dans C->d1 : 1 si les quatre p_ij sont premiers. */
static int tester(ctx_t *C, int extra) {
    if (mpz_cmp(C->d1, C->rac) > 0) return 0;          /* d1 > isqrt(N)  =>  d1 > d2 */
    mpz_divexact(C->d2, C->N, C->d1);
    if (mpz_cmp(C->d1, C->d2) > 0) return 0;
    for (int i = 0; !extra && i < C->nex; i++)
        if (mpz_cmp(C->d1, C->ex[i]) == 0) extra = 1;
    if (C->avec_cl && !extra) {
        mpz_mod(C->r, C->d2, C->L);
        if (mpz_cmp(C->r, C->c) != 0) return 0;
    }
    mpz_set(C->n1, C->A1); mpz_addmul(C->n1, C->d1, C->s2);
    mpz_set(C->n2, C->A1); mpz_addmul(C->n2, C->d2, C->s2);
    if (!mpz_divisible_p(C->n1, C->D) || !mpz_divisible_p(C->n2, C->D)) return 0;
    mpz_divexact(C->p11, C->n1, C->D);
    mpz_divexact(C->p12, C->n2, C->D);
    mpz_set(C->n1, C->A2); mpz_addmul(C->n1, C->d1, C->s1);
    mpz_set(C->n2, C->A2); mpz_addmul(C->n2, C->d2, C->s1);
    if (!mpz_divisible_p(C->n1, C->D) || !mpz_divisible_p(C->n2, C->D)) return 0;
    mpz_divexact(C->p21, C->n1, C->D);
    mpz_divexact(C->p22, C->n2, C->D);
    if (mpz_cmp(C->p11, C->p12) == 0 || mpz_cmp(C->p21, C->p22) == 0) return 0;
    if (mpz_cmp_ui(C->p11, 2) < 0 || mpz_cmp_ui(C->p12, 2) < 0 ||
        mpz_cmp_ui(C->p21, 2) < 0 || mpz_cmp_ui(C->p22, 2) < 0) return 0;
    if (mpz_divisible_p(C->a1, C->p11) || mpz_divisible_p(C->a1, C->p12) ||
        mpz_divisible_p(C->a2, C->p21) || mpz_divisible_p(C->a2, C->p22)) return 0;
    if (petit_facteur(C->p11, PP1) || petit_facteur(C->p12, PP1) || petit_facteur(C->p21, PP1) || petit_facteur(C->p22, PP1) ||
        petit_facteur(C->p11, PP2) || petit_facteur(C->p12, PP2) || petit_facteur(C->p21, PP2) || petit_facteur(C->p22, PP2))
        return 0;
    /* base 2 d'abord sur les quatre (une exponentiation chacun) : le test complet, ~5 fois plus cher des que
       p est premier, n'est lance que si les quatre passent. Meme verdict : il contient le test en base 2. */
    if (!(grand_sprp2(C->p11) && grand_sprp2(C->p12) && grand_sprp2(C->p21) && grand_sprp2(C->p22))) return 0;
    if (!(premier(C->p11) && premier(C->p12) && premier(C->p21) && premier(C->p22))) return 0;
    /* r1 = tau(a1 p11 p12) / a1 et r2 = tau(a2 p21 p22) / a2 : entiers, premiers, ne divisant pas a1, a2
       (memes tests que theoreme1 : un couple signale a Python est alors presque toujours un cycle) */
    for (int w = 0; w < 2; w++) {
        mpz_srcptr a = w ? C->a2 : C->a1, s = w ? C->s2 : C->s1;
        mpz_srcptr p = w ? C->p21 : C->p11, q = w ? C->p22 : C->p12;
        mpz_add_ui(C->n1, p, 1);
        mpz_add_ui(C->n2, q, 1);
        mpz_mul(C->n1, C->n1, C->n2);
        mpz_mul(C->n1, C->n1, s);                     /* sigma(a p q) */
        mpz_mul(C->n2, a, p);
        mpz_mul(C->n2, C->n2, q);
        mpz_sub(C->n1, C->n1, C->n2);                 /* tau(a p q) */
        if (!mpz_divisible_p(C->n1, a)) return 0;
        mpz_divexact(C->n1, C->n1, a);
        if (mpz_cmp_ui(C->n1, 2) < 0 || mpz_divisible_p(a, C->n1) || !premier(C->n1)) return 0;
    }
    return 1;
}

/* diviseurs d'une factorisation : tableau alloue de n valeurs, et leurs log2 dans *lg */
static const double *tri_lg;
static int cmp_lg(const void *a, const void *b) {
    double x = tri_lg[*(const long *)a], y = tri_lg[*(const long *)b];
    return (x > y) - (x < y);
}

static mpz_t *diviseurs(mpz_srcptr *p, const unsigned long *e, int np, long n, double **lg) {
    mpz_t *d = malloc(sizeof(mpz_t) * n);
    *lg = malloc(sizeof(double) * n);
    long k = 1;
    mpz_init_set_ui(d[0], 1);
    for (int i = 0; i < np; i++) {
        long k0 = k;
        for (unsigned long j = 1; j <= e[i]; j++)
            for (long t = 0; t < k0; t++) {
                mpz_init(d[k]);
                mpz_mul(d[k], d[k - k0], p[i]);
                k++;
            }
    }
    for (long t = 0; t < n; t++) (*lg)[t] = lg2(d[t]);
    /* tri par log2 croissant : les boucles de congrus() s'arretent (break) des que d1 > sqrt(N) */
    long *ix = malloc(sizeof(long) * n);
    for (long t = 0; t < n; t++) ix[t] = t;
    tri_lg = *lg;
    qsort(ix, n, sizeof(long), cmp_lg);
    mpz_t *d2 = malloc(sizeof(mpz_t) * n);
    double *l2 = malloc(sizeof(double) * n);
    for (long t = 0; t < n; t++) { d2[t][0] = d[ix[t]][0]; l2[t] = (*lg)[ix[t]]; }   /* deplacement */
    free(d); free(*lg); free(ix);
    *lg = l2;
    return d2;
}

static void liberer_tab(mpz_t *d, long n) {
    for (long t = 0; t < n; t++) mpz_clear(d[t]);
    free(d);
}

/* Diviseurs d1 = c (mod L) de N (comme diviseurs_congrus) ; tester() sur chacun.
   Renvoie 1 (candidat trouve), 0 (aucun), 2 (trop de diviseurs : OverflowError en Python),
   3 (plus de MAX_ATTENDUS*max_cote diviseurs attendus dans la classe : OverflowError aussi). */
#define MAX_ATTENDUS 2
static int congrus(ctx_t *C, mpz_srcptr *P, const unsigned long *E, int np, long max_cote) {
    /* mauvais : p | L ; bons : les autres, tries par (e+1) decroissant (stable), repartis en A, B */
    mpz_srcptr bonp[2 * MAXP]; unsigned long bone[2 * MAXP]; int nbon = 0;
    int mau[2 * MAXP], nmau = 0;
    for (int i = 0; i < np; i++) {
        if (mpz_divisible_p(C->L, P[i])) mau[nmau++] = i;
        else { bonp[nbon] = P[i]; bone[nbon] = E[i]; nbon++; }
    }
    for (int i = 1; i < nbon; i++) {
        mpz_srcptr xp = bonp[i]; unsigned long xe = bone[i]; int j = i - 1;
        while (j >= 0 && bone[j] < xe) { bonp[j + 1] = bonp[j]; bone[j + 1] = bone[j]; j--; }
        bonp[j + 1] = xp; bone[j + 1] = xe;
    }
    mpz_srcptr Ap[2 * MAXP], Bp[2 * MAXP]; unsigned long Ae[2 * MAXP], Be[2 * MAXP];
    int na = 0, nbb = 0;
    double ta = 1, tb = 1;
    for (int i = 0; i < nbon; i++) {
        if (ta <= tb) { Ap[na] = bonp[i]; Ae[na++] = bone[i]; ta *= bone[i] + 1; }
        else { Bp[nbb] = bonp[i]; Be[nbb++] = bone[i]; tb *= bone[i] + 1; }
    }
    /* Valuations permises pour chaque premier p de L (p^k || L). d1 = c (mod p^k) fixe la valuation v de
       d1 : v = v_p(c mod p^k) =: w si c n'est pas divisible par p^k, sinon v >= k. Mais d2 = N / d1 doit
       etre dans la MEME classe (tester() l'exige) : e - v obeit a la meme regle. Donc v dans [k, e - k]
       (vide si e < 2k), ou v = w avec e = 2w (vide sinon). En particulier un premier de L qui divise N
       exactement une fois vide le couple. On n'enumere plus que les b dont le cofacteur est permis. */
    mpz_t pk, cm, gC, MpC, xC;
    mpz_inits(pk, cm, gC, MpC, xC, NULL);
    unsigned long vmin[2 * MAXP], vmax[2 * MAXP], kval[2 * MAXP];
    double nb = 1;
    int vide = 0;
    mpz_set_ui(gC, 1);
    for (int t = 0; t < nmau; t++) {
        mpz_srcptr p = P[mau[t]];
        unsigned long e = E[mau[t]], k = mpz_remove(cm, C->L, p);
        kval[t] = k;
        mpz_pow_ui(pk, p, k);
        mpz_mod(cm, C->c, pk);
        if (mpz_sgn(cm) == 0) {
            if (e < 2 * k) { vide = 1; break; }
            vmin[t] = k; vmax[t] = e - k;
        } else {
            unsigned long w = mpz_remove(cm, cm, p);
            if (e != 2 * w) { vide = 1; break; }
            vmin[t] = vmax[t] = w;
        }
        nb *= (double)(vmax[t] - vmin[t] + 1);
        mpz_pow_ui(pk, p, vmin[t] < k ? vmin[t] : k);
        mpz_mul(gC, gC, pk);
    }
    /* Pour tout b = prod p^v permis, g = pgcd(b, L) = prod p^min(v, k) est le MEME (v >= k, ou v = w fixe),
       il divise c, et le cofacteur N_L / b a le meme pgcd avec L. Avec Mp = L / g et d1 = b*x*y :
         d1 = c (mod L)  <=>  x*y = (c/g) * (b/g)^-1                    (mod Mp)
         d2 = c (mod L)  <=>  x*y = (N/g^2) * (b/g)^-1 * (c/g)^-1       (mod Mp)
       Les deux classes coincident ssi (c/g)^2 = N / g^2 (mod Mp), condition independante de b (plus forte
       que N = c^2 mod L) : testee UNE fois ; si elle echoue, ou si c/g n'est pas inversible mod Mp, le
       couple n'a aucun d1. Sinon tout d1 de la classe a son d2 dans la classe. */
    if (!vide && !mpz_divisible_p(C->c, gC)) vide = 1;       /* n'arrive pas (voir ci-dessus) ; garde-fou */
    if (!vide) {
        mpz_divexact(MpC, C->L, gC);
        mpz_divexact(xC, C->c, gC);
        if (mpz_cmp_ui(MpC, 1) > 0) {
            mpz_gcd(cm, xC, MpC);
            if (mpz_cmp_ui(cm, 1) != 0) vide = 1;
            else {
                mpz_mul(pk, gC, gC);
                mpz_divexact(pk, C->N, pk);               /* g^2 | N : e >= 2 min(v, k) pour chaque p */
                mpz_mul(cm, xC, xC);
                mpz_sub(cm, cm, pk);
                if (!mpz_divisible_p(cm, MpC)) vide = 1;
            }
        }
    }
    mpz_clears(pk, cm, NULL);
    if (vide) { mpz_clears(gC, MpC, xC, NULL); return 0; }
    if ((ta * nb > tb ? ta * nb : tb) > (double)max_cote) { mpz_clears(gC, MpC, xC, NULL); return 2; }
    /* nombre de couples (x, y) attendus dans la bonne classe : ta*tb/Mp pour chaque b. Si L est petit
       (D petit), presque toutes les combinaisons passent : travail en ta*tb, intraitable. */
    if (nb * (ta * tb / mpz_get_d(MpC)) > MAX_ATTENDUS * (double)max_cote) { mpz_clears(gC, MpC, xC, NULL); return 3; }
    /* puissances p^v et p^(v - min(v, k)) en table */
    mpz_t *pw[2 * MAXP], *pz[2 * MAXP];
    for (int t = 0; t < nmau; t++) {
        unsigned long n_v = vmax[t] - vmin[t] + 1;
        pw[t] = malloc(sizeof(mpz_t) * n_v);
        pz[t] = malloc(sizeof(mpz_t) * n_v);
        for (unsigned long v = vmin[t]; v <= vmax[t]; v++) {
            unsigned long m = v < kval[t] ? v : kval[t];
            mpz_init(pw[t][v - vmin[t]]); mpz_pow_ui(pw[t][v - vmin[t]], P[mau[t]], v);
            mpz_init(pz[t][v - vmin[t]]); mpz_pow_ui(pz[t][v - vmin[t]], P[mau[t]], v - m);
        }
    }

    double *lga, *lgb;
    long la_n = (long)ta, lb_n = (long)tb;
    mpz_t *la = diviseurs(Ap, Ae, na, la_n, &lga);
    mpz_t *lb = diviseurs(Bp, Be, nbb, lb_n, &lgb);
    mpz_t *inv = NULL;
    long H = 1;
    while (H < 2 * lb_n) H <<= 1;
    long *tete = malloc(sizeof(long) * H), *suiv = malloc(sizeof(long) * lb_n);
    mpz_t *res = malloc(sizeof(mpz_t) * lb_n);
    for (long t = 0; t < lb_n; t++) mpz_init(res[t]);
    mpz_t Mp_tab, b, g, Mp, z, x, bx;
    mpz_inits(Mp_tab, b, g, Mp, z, x, bx, NULL);    /* Mp_tab = 0 : table pas encore construite */
    unsigned long cur[2 * MAXP];
    for (int t = 0; t < nmau; t++) cur[t] = vmin[t];
    int trouve = 0;
    for (;;) {                                        /* chaque b = produit des p^v permis */
        mpz_set_ui(b, 1);
        for (int t = 0; t < nmau; t++) mpz_mul(b, b, pw[t][cur[t] - vmin[t]]);
        double lgb_b = lg2(b);
        {
            mpz_set(Mp, MpC);
            if (mpz_cmp_ui(Mp, 1) == 0) {
                for (long i = 0; i < la_n && !trouve; i++) {
                    if (lgb_b + lga[i] + lgb[0] > C->lrac + 1e-6) break;          /* la, lb tries par log2 */
                    mpz_mul(bx, b, la[i]);
                    for (long j = 0; j < lb_n; j++) {
                        if (lgb_b + lga[i] + lgb[j] > C->lrac + 1e-6) break;       /* d1 > sqrt(N) */
                        mpz_mul(C->d1, bx, lb[j]);
                        if (tester(C, 0)) { trouve = 1; break; }
                    }
                }
            } else {
                mpz_set_ui(z, 1);                      /* z = b / g */
                for (int t = 0; t < nmau; t++) mpz_mul(z, z, pz[t][cur[t] - vmin[t]]);
                mpz_invert(z, z, Mp);
                mpz_mul(z, z, xC);
                mpz_mod(z, z, Mp);
                {                                      /* z inversible mod Mp (c/g l'est : teste plus haut) */
                    if (mpz_cmp(Mp_tab, Mp) != 0) {   /* table des y mod Mp */
                        mpz_set(Mp_tab, Mp);
                        for (long h = 0; h < H; h++) tete[h] = -1;
                        for (long j = lb_n - 1; j >= 0; j--) {     /* chaines par log2 croissant */
                            mpz_mod(res[j], lb[j], Mp);
                            long h = (long)(mpz_getlimbn(res[j], 0) & (mp_limb_t)(H - 1));
                            suiv[j] = tete[h];
                            tete[h] = j;
                        }
                    }
                    if (!inv) {
                        inv = malloc(sizeof(mpz_t) * la_n);
                        for (long i = 0; i < la_n; i++) { mpz_init(inv[i]); mpz_invert(inv[i], la[i], C->L); }
                    }
                    for (long i = 0; i < la_n && !trouve; i++) {
                        if (lgb_b + lga[i] + lgb[0] > C->lrac + 1e-6) break;
                        mpz_mul(x, z, inv[i]);
                        mpz_mod(x, x, Mp);
                        long h = (long)(mpz_getlimbn(x, 0) & (mp_limb_t)(H - 1));
                        int bx_ok = 0;
                        for (long j = tete[h]; j >= 0; j = suiv[j]) {
                            if (lgb_b + lga[i] + lgb[j] > C->lrac + 1e-6) break;   /* chaine croissante */
                            if (mpz_cmp(res[j], x) != 0) continue;
                            if (!bx_ok) { mpz_mul(bx, b, la[i]); bx_ok = 1; }
                            mpz_mul(C->d1, bx, lb[j]);
                            if (tester(C, 0)) { trouve = 1; break; }
                        }
                    }
                }
            }
        }
        if (trouve) break;
        int t = 0;                                    /* b suivant (odometre) */
        while (t < nmau && cur[t] == vmax[t]) { cur[t] = vmin[t]; t++; }
        if (t == nmau) break;
        cur[t]++;
    }
    mpz_clears(Mp_tab, b, g, Mp, z, x, bx, NULL);
    for (int t = 0; t < nmau; t++) {
        for (unsigned long v = 0; v <= vmax[t] - vmin[t]; v++) { mpz_clear(pw[t][v]); mpz_clear(pz[t][v]); }
        free(pw[t]); free(pz[t]);
    }
    mpz_clears(gC, MpC, xC, NULL);
    liberer_tab(res, lb_n);
    free(tete); free(suiv);
    if (inv) liberer_tab(inv, la_n);
    liberer_tab(la, la_n); liberer_tab(lb, lb_n);
    free(lga); free(lgb);
    return trouve;
}

/* 0 : "ok" sans candidat, 1 : "aucun d1", 2 : a refaire en Python (candidat),
   3 : trop de diviseurs (borho_paire le fait refaire en Python, croise_tranche le compte).
   C->D est deja calcule (et > 0). */
static int couple(ctx_t *C, const base_t *x, const base_t *y, mpz_srcptr t1, mpz_srcptr t2,
                  const fac_t *f1, int n1, const fac_t *f2, int n2, long max_div) {
    mpz_t D2, num, q, c3, M3, rhs;
    mpz_inits(D2, num, q, c3, M3, rhs, NULL);
    mpz_set(C->a1, x->v); mpz_set(C->s1, x->s);
    mpz_set(C->a2, y->v); mpz_set(C->s2, y->s);
    mpz_mul(C->N, C->a1, C->a2);
    mpz_mul_2exp(D2, C->D, 1);
    mpz_mul(C->A1, t2, C->s1);
    mpz_mul(C->A2, t1, C->s2);
    /* p11 et p21 impairs : deux congruences sur d1 modulo 2D, combinees par CRT */
    mpz_sub(rhs, C->D, C->A1);
    int ok1 = congruence(C->c, C->L, C->s2, rhs, D2);
    mpz_sub(rhs, C->D, C->A2);
    int ok3 = congruence(c3, M3, C->s1, rhs, D2);
    C->avec_cl = ok1 && ok3 && crt(C->c, C->L, c3, M3);
    /* extras : un des premiers egal a 2 */
    C->nex = 0;
    for (int w = 0; w < 2; w++) {
        mpz_sub(num, D2, w == 0 ? C->A1 : C->A2);
        mpz_srcptr den = w == 0 ? C->s2 : C->s1;
        if (mpz_sgn(num) > 0 && mpz_divisible_p(num, den)) {
            mpz_divexact(q, num, den);
            if (mpz_divisible_p(C->N, q)) {
                mpz_set(C->ex[C->nex++], q);
                mpz_divexact(C->ex[C->nex++], C->N, q);
            }
        }
    }
    if (C->avec_cl) {                                 /* N = c^2 (mod L) */
        mpz_mul(q, C->c, C->c);
        mpz_sub(q, C->N, q);
        if (!mpz_divisible_p(q, C->L)) C->avec_cl = 0;
    }
    int r = 0;
    if (!C->avec_cl && C->nex == 0) { r = 1; goto fin; }
    mpz_sqrt(C->rac, C->N);
    C->lrac = lg2(C->N) / 2;
    for (int i = 0; i < C->nex; i++) {                /* les extras (sans condition de classe) */
        mpz_set(C->d1, C->ex[i]);
        if (tester(C, 1)) { r = 2; goto fin; }
    }
    if (C->avec_cl) {
        /* factorisation de N = a1*a2 : fusion des deux bases */
        mpz_srcptr P[2 * MAXP]; unsigned long E[2 * MAXP]; int np = 0;
        for (int i = 0; i < n1; i++)
            if (!(x->retires >> i & 1)) { P[np] = f1[i].p; E[np++] = f1[i].e; }
        for (int j = 0; j < n2; j++) {
            if (y->retires >> j & 1) continue;
            int t;
            for (t = 0; t < np; t++)
                if (mpz_cmp(P[t], f2[j].p) == 0) { E[t] += f2[j].e; break; }
            if (t == np) { P[np] = f2[j].p; E[np++] = f2[j].e; }
        }
        int k = congrus(C, P, E, np, max_div);
        if (k == 1) r = 2;
        else if (k >= 2) r = 3;
    }
fin:
    mpz_clears(D2, num, q, c3, M3, rhs, NULL);
    return r;
}

/* ------------------------------------------------------------------ point d'entree */
/* --max-graine : couples dont les deux graines ont au plus k chiffres (k <= 0 : pas de limite) */
static mpz_t LIMG;
static int limg_ok = 0, limg_init = 0;
void borho_set_max_graine(int k) {
    if (!limg_init) { mpz_init(LIMG); limg_init = 1; }
    limg_ok = k > 0;
    if (limg_ok) mpz_ui_pow_ui(LIMG, 10, (unsigned long)k);   /* graine < 10^k  <=>  au plus k chiffres */
}

int borho_paire(const char *s1, const char *s2, int retire, int total, int sauf, long max_div,
                int *stats, int *out, int max_out) {
    fac_t *f1 = malloc(sizeof(fac_t) * MAXP), *f2 = malloc(sizeof(fac_t) * MAXP);
    base_t *b1 = malloc(sizeof(base_t) * MAXB), *b2 = malloc(sizeof(base_t) * MAXB);
    int m1[MAXP], m2[MAXP];
    int n1 = 0, n2 = 0, nb1 = 0, nb2 = 0, ret = 0;
    if (lire(s1, f1, &n1) || lire(s2, f2, &n2)) { ret = -1; goto fin; }
    /* la construction suppose des facteurs premiers : sinon Python reprend (deplier) */
    int gros = 0;
    for (int i = 0; i < n1; i++) if (mpz_cmp_ui(f1[i].p, 2) < 0 || !premier_vite(f1[i].p, &gros)) { ret = -1; goto fin; }
    for (int i = 0; i < n2; i++) if (mpz_cmp_ui(f2[i].p, 2) < 0 || !premier_vite(f2[i].p, &gros)) { ret = -1; goto fin; }
    /* partie commune G : meme premier ET meme exposant ; m1, m2 : le reste, tries par valeur */
    int k1 = 0, k2 = 0;
    for (int i = 0; i < n1; i++) {
        int commun = 0;
        for (int j = 0; j < n2; j++)
            if (mpz_cmp(f1[i].p, f2[j].p) == 0) { commun = (f1[i].e == f2[j].e); break; }
        if (!commun) m1[k1++] = i;
    }
    for (int j = 0; j < n2; j++) {
        int commun = 0;
        for (int i = 0; i < n1; i++)
            if (mpz_cmp(f1[i].p, f2[j].p) == 0) { commun = (f1[i].e == f2[j].e); break; }
        if (!commun) m2[k2++] = j;
    }
    tri_f = f1; qsort(m1, k1, sizeof(int), cmp_idx);  /* combinations(sorted(m), k) */
    tri_f = f2; qsort(m2, k2, sizeof(int), cmp_idx);
    nb1 = bases(f1, n1, m1, k1, retire, b1);
    if (nb1 < 0) { nb1 = -1 - nb1; ret = -2; goto fin; }
    nb2 = bases(f2, n2, m2, k2, retire, b2);
    if (nb2 < 0) { nb2 = -1 - nb2; ret = -2; goto fin; }

    ctx_t C;
    ctx_init(&C);
    mpz_t t1, t2, x, g, h;
    mpz_inits(t1, t2, x, g, h, NULL);
    for (int i = 0; i < nb1 && ret >= 0; i++) {
        if (limg_ok && mpz_cmp(b1[i].v, LIMG) >= 0) continue;
        mpz_sub(t1, b1[i].s, b1[i].v);               /* tau(a1) */
        mpz_mul(h, b1[i].s, b1[i].v);                /* s1 * a1 : une fois par base (pre-test du theoreme 2) */
        for (int j = 0; j < nb2; j++) {
            if (limg_ok && mpz_cmp(b2[j].v, LIMG) >= 0) continue;
            if (total && b1[i].k + b2[j].k > total) continue;
            if (sauf && b1[i].k <= sauf && b2[j].k <= sauf) continue;
            stats[0]++;
            if (mpz_cmp(b1[i].v, b2[j].v) == 0) { stats[1]++; continue; }
            mpz_sub(t2, b2[j].s, b2[j].v);
            mpz_mul(C.D, b1[i].v, b2[j].v);
            mpz_submul(C.D, t1, t2);
            if (mpz_sgn(C.D) <= 0) { stats[2]++; continue; }
            /* theoreme 2 : (a1 - a2) * pgcd(s1, s2) * pgcd(a1, a2) = 0 (mod D) ; pre-test sans pgcd :
               pgcd(s1, s2) * pgcd(a1, a2) divise s1 * a1, donc D doit diviser (a1 - a2) * s1 * a1 */
            mpz_sub(x, b1[i].v, b2[j].v);
            mpz_mul(g, x, h);
            if (!mpz_divisible_p(g, C.D)) { stats[3]++; continue; }
            mpz_gcd(g, b1[i].s, b2[j].s);
            mpz_mul(x, x, g);
            mpz_gcd(g, b1[i].v, b2[j].v);
            mpz_mul(x, x, g);
            if (!mpz_divisible_p(x, C.D)) { stats[3]++; continue; }
            int r = couple(&C, &b1[i], &b2[j], t1, t2, f1, n1, f2, n2, max_div);
            if (r == 1) { stats[4]++; continue; }
            if (r == 0) { stats[5]++; continue; }
            if (ret >= max_out) { ret = -2; break; }
            out[2 * ret] = i;
            out[2 * ret + 1] = j;
            ret++;
        }
    }
    mpz_clears(t1, t2, x, g, h, NULL);
    ctx_clear(&C);
    /* un couple va etre signale : les grands facteurs n'ont passe que le test en base 2, on refait ici le
       test complet (meme garantie qu'avant pour tout cycle ecrit) ; en cas d'echec, Python reprend la paire */
    if (ret > 0 && gros) {
        for (int i = 0; i < n1 && ret > 0; i++) if (!premier(f1[i].p)) ret = -1;
        for (int i = 0; i < n2 && ret > 0; i++) if (!premier(f2[i].p)) ret = -1;
    }
fin:
    liberer_bases(b1, nb1);
    liberer_bases(b2, nb2);
    liberer(f1, n1);
    liberer(f2, n2);
    free(f1); free(f2); free(b1); free(b2);
    return ret;
}

/* ---------------------------------------------------------------------------------------------
 * Croisement de graines tirees de paires DIFFERENTES (croise.py)
 *   int croise_init(int nb, const char **facs, const int *paires)
 *       charge les graines d'un groupe : factorisation (« p^e*q*... ») et numero de paire d'origine
 *   int croise_tranche(int i0, int i1, long max_div, int *stats, int *out, int max_out)
 *       couples (i, j), i0 <= i < i1, i < j, paires differentes : theoreme 2 puis crible complet de
 *       theoreme1 (couple()) ; out : couples (i, j) a refaire en Python (candidat ou trop de
 *       diviseurs). stats[0..5] comme borho_paire, stats[6] : couples abandonnes (trop de diviseurs). Retour : nombre de couples signales, -2 si
 *       max_out est depasse, -1 si croise_init n'a pas ete appele.
 * ------------------------------------------------------------------------------------------- */
typedef struct { fac_t *f; int n; base_t b; int paire; } graine_t;
static graine_t *GR = NULL;
static int NGR = 0;
static ctx_t CTXC;
static int ctxc_ok = 0;

static void croise_liberer(void) {
    for (int i = 0; i < NGR; i++) {
        liberer(GR[i].f, GR[i].n);
        free(GR[i].f);
        mpz_clear(GR[i].b.v); mpz_clear(GR[i].b.s);
    }
    free(GR); GR = NULL; NGR = 0;
}

int croise_init(int nb, const char **facs, const int *paires) {
    croise_liberer();
    GR = calloc(nb, sizeof(graine_t));
    fac_t tmp[MAXP];
    mpz_t q, sq;
    mpz_inits(q, sq, NULL);
    for (int i = 0; i < nb; i++) {
        int n = 0;
        if (lire(facs[i], tmp, &n)) { liberer(tmp, n); NGR = i; mpz_clears(q, sq, NULL); croise_liberer(); return -1; }
        GR[i].f = malloc(sizeof(fac_t) * (n ? n : 1));
        memcpy(GR[i].f, tmp, sizeof(fac_t) * n);       /* les mpz sont transferes (pas de double liberation) */
        GR[i].n = n;
        GR[i].paire = paires[i];
        GR[i].b.k = 0; GR[i].b.retires = 0;
        mpz_init_set_ui(GR[i].b.v, 1); mpz_init_set_ui(GR[i].b.s, 1);
        for (int t = 0; t < n; t++) {
            puiss(q, sq, GR[i].f[t].p, GR[i].f[t].e);
            mpz_mul(GR[i].b.v, GR[i].b.v, q);
            mpz_mul(GR[i].b.s, GR[i].b.s, sq);
        }
        NGR = i + 1;
    }
    mpz_clears(q, sq, NULL);
    return 0;
}

static int JMIN = 0;           /* croise.py --mixte : seules les graines j >= JMIN (paires anciennes) */
void croise_jmin(int j) { JMIN = j > 0 ? j : 0; }

int croise_tranche(int i0, int i1, long max_div, int *stats, int *out, int max_out) {
    if (!GR) return -1;
    if (!ctxc_ok) { ctx_init(&CTXC); ctxc_ok = 1; }
    if (i1 > NGR) i1 = NGR;
    mpz_t t1, t2, x, g, h;
    mpz_inits(t1, t2, x, g, h, NULL);
    int ret = 0;
    for (int i = i0; i < i1 && ret >= 0; i++) {
        graine_t *A = &GR[i];
        mpz_sub(t1, A->b.s, A->b.v);
        mpz_mul(h, A->b.s, A->b.v);
        for (int j = (i + 1 > JMIN ? i + 1 : JMIN); j < NGR; j++) {
            graine_t *Bq = &GR[j];
            if (Bq->paire == A->paire) continue;       /* meme paire : c'est le travail de borho4 */
            stats[0]++;
            if (mpz_cmp(A->b.v, Bq->b.v) == 0) { stats[1]++; continue; }
            mpz_sub(t2, Bq->b.s, Bq->b.v);
            mpz_mul(CTXC.D, A->b.v, Bq->b.v);
            mpz_submul(CTXC.D, t1, t2);
            if (mpz_sgn(CTXC.D) <= 0) { stats[2]++; continue; }
            /* pre-test sans pgcd : f*g divise s1*a1, donc D | (a1-a2)*f*g  =>  D | (a1-a2)*s1*a1 */
            mpz_sub(x, A->b.v, Bq->b.v);
            mpz_mul(g, x, h);
            if (!mpz_divisible_p(g, CTXC.D)) { stats[3]++; continue; }
            mpz_gcd(g, A->b.s, Bq->b.s);
            mpz_mul(x, x, g);
            mpz_gcd(g, A->b.v, Bq->b.v);
            mpz_mul(x, x, g);
            if (!mpz_divisible_p(x, CTXC.D)) { stats[3]++; continue; }
            int r = couple(&CTXC, &A->b, &Bq->b, t1, t2, A->f, A->n, Bq->f, Bq->n, max_div);
            if (r == 1) { stats[4]++; continue; }
            if (r == 3) { stats[6]++; continue; }      /* trop de diviseurs : abandonne (compte) */
            if (r == 0) { stats[5]++; continue; }
            if (ret >= max_out) { ret = -2; break; }
            out[2 * ret] = i; out[2 * ret + 1] = j;
            ret++;
        }
    }
    mpz_clears(t1, t2, x, g, h, NULL);
    return ret;
}

/* -------------------------------------------------------------------------------------------
 * apparie_couples2 (apparie_graines.py) : couples de graines a1 = G*u0*p, a2 = G*u2, p grand premier
 *   inconnu. E = alpha*p + beta (D = S*E), alpha = G*s0*u2 + G*s2*u0 - S*s0*s2, beta = s0*(G*u2 - S*s2) ;
 *   pour p de pmin a pmax (pmax <= pmin + maxp, E <= Emax) : E | M0 (M0 = |u0*beta + alpha*u2|*G^2*
 *   pgcd(u0,u2)*s2, test par un reste machine), theoreme 2 exact, p premier ne divisant pas G*u0*u2 ;
 *   stats[0..3] += p testes, th2, premiers, couples hors 64 bits. Chaque p retenu passe aussi le
 * theoreme 1 complet en C (couple(), le meme que pour borho4 et croise : congruences, diviseurs,
 * premiers p_ij et r_i) ; seuls les couples avec un candidat complet sont rendus (a refaire en Python
 * pour ecrire le cycle). Gfac : factorisation de G (« 3^4*11^2 »). stats[4..7] += aucun d1, ok sans
 * candidat, trop de diviseurs, candidats rendus.
 * ------------------------------------------------------------------------------------------- */
static ctx_t CTXA;
static int ctxa_ok = 0;

static int fac_petit(long u, fac_t *f, int *n) {     /* ajoute les facteurs de u a f[*n..] ; -1 si plein */
    for (long d = 2; d * d <= u; d += (d == 2 ? 1 : 2)) {   /* *n compte toujours les entrees initialisees */
        if (u % d) continue;
        unsigned long e = 0;
        while (u % d == 0) { u /= d; e++; }
        if (*n >= MAXP) return -1;
        mpz_init_set_si(f[*n].p, d); f[*n].e = e; (*n)++;
    }
    if (u > 1) {
        if (*n >= MAXP) return -1;
        mpz_init_set_si(f[*n].p, u); f[*n].e = 1; (*n)++;
    }
    return 0;
}

static int ac_impl(const char *Gfac, const char *Ss, int n, const long *u0, const long *s0, const long *u2,
                   const long *s2, unsigned long P0, unsigned long maxp, const char *Emaxs, long max_div,
                   int *out_k, unsigned long *out_p, int max_out, long *stats, const unsigned long *p0k);

int apparie_couples2(const char *Gfac, const char *Ss, int n, const long *u0, const long *s0, const long *u2,
                     const long *s2, unsigned long P0, unsigned long maxp, const char *Emaxs, long max_div,
                     int *out_k, unsigned long *out_p, int max_out, long *stats) {
    return ac_impl(Gfac, Ss, n, u0, s0, u2, s2, P0, maxp, Emaxs, max_div, out_k, out_p, max_out, stats, NULL);
}

/* apparie_couples_p : meme chose, mais un seul p par couple, p0k[k] (survivants du crible GPU,
   apparie_gpu.cu) : pmin = max(borne, p0k[k]), maxp = 0. */
int apparie_couples_p(const char *Gfac, const char *Ss, int n, const long *u0, const long *s0, const long *u2,
                      const long *s2, const unsigned long *p0k, const char *Emaxs, long max_div,
                      int *out_k, unsigned long *out_p, int max_out, long *stats) {
    return ac_impl(Gfac, Ss, n, u0, s0, u2, s2, 0, 0, Emaxs, max_div, out_k, out_p, max_out, stats, p0k);
}

/* E | M0 ?  (M0 = m[0..n-1], E > 0 sur un mot machine.) Test de divisibilite exacte SANS division, en ligne :
   inverse de E modulo 2^64 par Newton (4 multiplications), puis deux multiplications par mot de M0
   (meme methode que mpn_modexact_1_odd de GMP, sans les appels de fonction). C'est le coeur de la boucle
   sur p : elle l'execute des milliards de fois. */
#if GMP_LIMB_BITS == 64 && ULONG_MAX == 0xFFFFFFFFFFFFFFFFUL && defined(__SIZEOF_INT128__)
#define AC_EN_LIGNE 1
static inline int ac_divise(const mp_limb_t *m, int n, unsigned long E) {
    if (n == 0) return 1;                                /* M0 = 0 */
    if (!(E & 1)) {                                      /* E pair : M0 doit avoir au moins autant de facteurs 2 */
        if (m[0] & ((E & -E) - 1)) return 0;
        E >>= __builtin_ctzl(E);
    }
    /* inverse de E mod 2^64 (variante de Dumas : y = 1 - E*inv eleve au carre en parallele des produits,
       chaine de dependances plus courte que Newton, meme nombre de multiplications) */
    unsigned long inv = (3 * E) ^ 2, y = 1 - E * inv;   /* 5 bits exacts, doubles a chaque pas */
    inv *= 1 + y; y *= y; inv *= 1 + y; y *= y; inv *= 1 + y; y *= y; inv *= 1 + y;
    unsigned long c = 0;
    for (int i = 0; i < n; i++) {
        unsigned long x = m[i], l = x - c;
        c = (unsigned long)(((unsigned __int128)(l * inv) * E) >> 64) + (x < c);
    }
    return c == 0 || c == E;
}

/* E | M0 pour 2^64 <= E < 2^128 (E = eh*2^64 + el ; gros G, --depuis-paires), sans GMP : division exacte de
   Hensel par un diviseur de deux mots. Apres retrait de la puissance de 2 (E1 impair), chaque pas annule le
   mot de poids faible de r = M0 - Q*E1 (q = r[i] / el1 mod 2^64) ; apres n - 2 pas, r = R * 2^(64(n-2)) avec
   -E1 < R < 2^128, et E1 | M0 <=> E1 | R (E1 impair) : un seul reste 128 bits a la fin. Verifie contre
   mpz_divisible_p (test_ac_divise2.c). Renvoie -1 si M0 est trop long (le test GMP reprend). */
#define AC2_MAX 32
static inline int ac_divise2(const mp_limb_t *m, int n, unsigned long el, unsigned long eh) {
    if (n == 0) return 1;                                /* M0 = 0 */
    if (n > AC2_MAX) return -1;
    unsigned __int128 E1 = ((unsigned __int128)eh << 64) | el;
    if (!(el & 1)) {                                     /* 2^k | E : 2^k | M0 exige, puis E1 = E / 2^k */
        int k = el ? __builtin_ctzl(el) : 64 + __builtin_ctzl(eh);
        if (k < 64) { if (m[0] & ((1UL << k) - 1)) return 0; }
        else {
            if (m[0] || n < 2) return 0;
            if (k > 64 && (m[1] & ((1UL << (k - 64)) - 1))) return 0;
        }
        E1 >>= k;
    }
    unsigned long e0 = (unsigned long)E1, e1 = (unsigned long)(E1 >> 64);
    if (e1 == 0) return ac_divise(m, n, e0);             /* E1 tient sur un mot */
    if (n < 2) return 0;                                 /* 0 < M0 < 2^64 <= E1 */
    unsigned long inv = (3 * e0) ^ 2, y = 1 - e0 * inv;  /* 1 / e0 mod 2^64 (Dumas) */
    inv *= 1 + y; y *= y; inv *= 1 + y; y *= y; inv *= 1 + y; y *= y; inv *= 1 + y;
    unsigned long r[AC2_MAX];
    for (int i = 0; i < n; i++) r[i] = m[i];
    for (int i = 0; i + 2 < n; i++) {
        unsigned long q = r[i] * inv;                    /* q*e0 = r[i] (mod 2^64) : le mot i s'annule */
        unsigned __int128 a = (unsigned __int128)q * e0, b = (unsigned __int128)q * e1;
        unsigned __int128 s = (a >> 64) + b;             /* a soustraire a partir du mot i+1 (< 2^128) */
        unsigned long s0 = (unsigned long)s, s1 = (unsigned long)(s >> 64);
        unsigned long t = r[i + 1];
        r[i + 1] = t - s0;
        unsigned long br = t < s0;
        t = r[i + 2];
        unsigned long d = t - s1, br2 = t < s1;
        r[i + 2] = d - br;
        br2 |= d < br;
        for (int j = i + 3; br2 && j < n; j++) { br2 = r[j] == 0; r[j]--; }
        if (br2) return 0;                               /* r < 0 : R < 0 et R > -E1, donc E1 ne divise pas */
    }
    unsigned __int128 R = ((unsigned __int128)r[n - 1] << 64) | r[n - 2];
    return R % E1 == 0;
}
#else
#define AC_EN_LIGNE 0
#endif

static int ac_impl(const char *Gfac, const char *Ss, int n, const long *u0, const long *s0, const long *u2,
                   const long *s2, unsigned long P0, unsigned long maxp, const char *Emaxs, long max_div,
                   int *out_k, unsigned long *out_p, int max_out, long *stats, const unsigned long *p0k) {
    fac_t fG[MAXP]; int nG = 0;
    if (lire(Gfac, fG, &nG)) { liberer(fG, nG); return -1; }
    if (!ctxa_ok) { ctx_init(&CTXA); ctxa_ok = 1; }
    mpz_t G, S, Emax, al, be, t, M0, E, a1, a2, sa1, sa2, x, g, pz, G2, t1, t2;
    mpz_inits(G, S, Emax, al, be, t, M0, E, a1, a2, sa1, sa2, x, g, pz, G2, t1, t2, NULL);
    mpz_set_ui(G, 1);
    for (int i = 0; i < nG; i++) { mpz_pow_ui(t, fG[i].p, fG[i].e); mpz_mul(G, G, t); }
    mpz_set_str(S, Ss, 10); mpz_set_str(Emax, Emaxs, 10);
    mpz_mul(G2, G, G);
    fac_t f1[MAXP], f2[MAXP];
    base_t bx, by;
    mpz_inits(bx.v, bx.s, by.v, by.s, NULL);
    bx.k = by.k = 0; bx.retires = by.retires = 0;
    int ret = 0;
    for (int k = 0; k < n && ret >= 0; k++) {
        const unsigned long U0 = (unsigned long)u0[k], S0 = (unsigned long)s0[k];
        const unsigned long U2 = (unsigned long)u2[k], S2 = (unsigned long)s2[k];
        /* alpha = G*(s0*u2 + s2*u0) - S*(s0*s2), beta = s0*(G*u2 - S*s2) : 5 operations GMP au lieu de 11
           quand les produits des petits facteurs tiennent sur 64 bits (u, sigma(u) < 2^31 en pratique) */
        unsigned long A_, B_, c1_, c2_;
        if (!__builtin_mul_overflow(S0, U2, &c1_) && !__builtin_mul_overflow(S2, U0, &c2_)
            && !__builtin_add_overflow(c1_, c2_, &A_) && !__builtin_mul_overflow(S0, S2, &B_)) {
            mpz_mul_ui(al, G, A_); mpz_submul_ui(al, S, B_);
        } else {
            mpz_mul_ui(al, G, S0); mpz_mul_ui(al, al, U2);
            mpz_mul_ui(t, G, S2); mpz_mul_ui(t, t, U0); mpz_add(al, al, t);
            mpz_mul_ui(t, S, S0); mpz_mul_ui(t, t, S2); mpz_sub(al, al, t);
        }
        if (mpz_sgn(al) <= 0) continue;
        mpz_mul_ui(be, G, U2); mpz_submul_ui(be, S, S2);
        if (mpz_sgn(be) >= 0) continue;
        mpz_mul_ui(be, be, S0);
        if (p0k) { P0 = p0k[k]; maxp = 0; }
        /* pmin = max(floor(-beta/alpha) + 1, P0), et E0 = alpha*pmin + beta : avec -beta = q*alpha + r,
           E0 = alpha - r si pmin = q + 1. Le rejet E0 > Emax (la plupart des couples) se fait ainsi apres UNE
           division, sans la seconde (pmax) */
        mpz_neg(t, be); mpz_fdiv_qr(t, x, t, al); mpz_add_ui(t, t, 1);
        if (mpz_cmp_ui(t, P0) < 0) {
            mpz_set_ui(t, P0);
            mpz_mul_ui(E, al, P0); mpz_add(E, E, be);
        } else mpz_sub(E, al, x);
        if (!mpz_fits_ulong_p(t)) { stats[3]++; continue; }
        if (mpz_cmp(E, Emax) > 0) continue;               /* aucun p : E croit avec p */
        unsigned long pmin = mpz_get_ui(t);
        /* pmax = pmin + min(maxp, floor((Emax - E0)/alpha)) */
        unsigned long pmax;
        mpz_sub(x, Emax, E);
        mpz_mul_ui(t, al, maxp);
        if (mpz_cmp(t, x) <= 0) pmax = pmin + maxp;
        else {
            mpz_fdiv_q(x, x, al);
            pmax = pmin + mpz_get_ui(x);                  /* < maxp ici */
        }
        mpz_mul_si(M0, be, u0[k]); mpz_mul_si(t, al, u2[k]); mpz_add(M0, M0, t); mpz_abs(M0, M0);
        mpz_mul(M0, M0, G2);
        long g0 = u0[k], h0 = u2[k];
        while (h0) { long r = g0 % h0; g0 = h0; h0 = r; }
        mpz_mul_si(M0, M0, g0); mpz_mul_si(M0, M0, s2[k]);
#if AC_EN_LIGNE
        /* E et alpha suivis sur 128 bits tant qu'ils y tiennent (E > 0, alpha > 0 ici) */
        int machine = mpz_sizeinbase(E, 2) <= 128 && mpz_sizeinbase(al, 2) <= 128;
        unsigned __int128 Eu = 0, alu = 0;
        if (machine) {
            Eu = ((unsigned __int128)mpz_getlimbn(E, 1) << 64) | mpz_getlimbn(E, 0);
            alu = ((unsigned __int128)mpz_getlimbn(al, 1) << 64) | mpz_getlimbn(al, 0);
        }
        const mp_limb_t *m0l = mpz_limbs_read(M0);
        const int m0n = (int)mpz_size(M0);
#else
        int machine = mpz_fits_ulong_p(E) && mpz_fits_ulong_p(al);
        unsigned long Eu = machine ? mpz_get_ui(E) : 0, alu = machine ? mpz_get_ui(al) : 0;
#endif
        stats[0] += (long)(pmax - pmin + 1);              /* p testes (compte une fois par couple) */
        for (unsigned long p = pmin; ; p++) {
            int divise;
            if (machine) {
                /* divisibilite exacte, sans division (le reste de M0 par E etait le cout de la boucle) */
#if AC_EN_LIGNE
                unsigned long eh = (unsigned long)(Eu >> 64), el = (unsigned long)Eu;
                if (eh == 0) divise = el != 0 && ac_divise(m0l, m0n, el);
                else if ((divise = ac_divise2(m0l, m0n, el, eh)) < 0) {
                    mpz_set_ui(E, eh); mpz_mul_2exp(E, E, 64); mpz_add_ui(E, E, el);
                    divise = mpz_divisible_p(M0, E);
                }
                if (divise) { mpz_set_ui(E, eh); mpz_mul_2exp(E, E, 64); mpz_add_ui(E, E, el); }
#else
                divise = Eu != 0 && mpz_divisible_ui_p(M0, Eu);
                if (divise) mpz_set_ui(E, Eu);
#endif
            } else divise = mpz_sgn(E) > 0 && mpz_divisible_p(M0, E);
            if (divise) {
                mpz_set_ui(pz, p);
                mpz_mul_si(a1, G, u0[k]); mpz_mul(a1, a1, pz);
                mpz_mul_si(a2, G, u2[k]);
                mpz_add_ui(t, pz, 1); mpz_mul_si(sa1, S, s0[k]); mpz_mul(sa1, sa1, t);
                mpz_mul_si(sa2, S, s2[k]);
                mpz_mul(CTXA.D, S, E);
                mpz_sub(x, a1, a2);
                mpz_gcd(g, sa1, sa2); mpz_mul(x, x, g);
                mpz_gcd(g, a1, a2); mpz_mul(x, x, g);
                if (mpz_divisible_p(x, CTXA.D)) {
                    stats[1]++;
                    mpz_mul_si(t, G, u0[k]); mpz_mul_si(t, t, u2[k]);
                    if (mpz_probab_prime_p(pz, 25) && !mpz_divisible_ui_p(t, p)) {
                        stats[2]++;
                        /* theoreme 1 en C : graines a1 = G*u0*p, a2 = G*u2, factorisees */
                        int n1 = 0, n2 = 0, okf = 1;
                        for (int i = 0; i < nG; i++) { mpz_init_set(f1[n1].p, fG[i].p); f1[n1++].e = fG[i].e; }
                        if (fac_petit(u0[k], f1, &n1) || n1 >= MAXP) okf = 0;
                        else { mpz_init_set_ui(f1[n1].p, p); f1[n1++].e = 1; }
                        for (int i = 0; i < nG; i++) { mpz_init_set(f2[n2].p, fG[i].p); f2[n2++].e = fG[i].e; }
                        if (fac_petit(u2[k], f2, &n2)) okf = 0;
                        int r = 2;                                  /* par prudence : a refaire en Python */
                        if (okf) {
                            mpz_set(bx.v, a1); mpz_set(bx.s, sa1); mpz_set(by.v, a2); mpz_set(by.s, sa2);
                            mpz_sub(t1, sa1, a1); mpz_sub(t2, sa2, a2);
                            r = couple(&CTXA, &bx, &by, t1, t2, f1, n1, f2, n2, max_div);
                        }
                        liberer(f1, n1);
                        liberer(f2, n2);
                        if (r == 1) stats[4]++;
                        else if (r == 0) stats[5]++;
                        else if (r == 3) stats[6]++;
                        if (r == 2 || r == 3) {                    /* candidat, ou trop de diviseurs : Python decide */
                            stats[7]++;
                            if (ret >= max_out) { ret = -2; break; }
                            out_k[ret] = k; out_p[ret] = p; ret++;
                        }
                    }
                }
            }
            if (p >= pmax) break;
            if (machine) {
#if AC_EN_LIGNE
                if (Eu > ~(unsigned __int128)0 - alu) {        /* E + alpha >= 2^128 : suite en GMP */
                    mpz_set_ui(E, (unsigned long)(Eu >> 64)); mpz_mul_2exp(E, E, 64);
                    mpz_add_ui(E, E, (unsigned long)Eu); mpz_add(E, E, al); machine = 0;
                } else Eu += alu;
#else
                if (Eu > (unsigned long)-1 - alu) { mpz_set_ui(E, Eu); mpz_add(E, E, al); machine = 0; }
                else Eu += alu;
#endif
            } else mpz_add(E, E, al);
        }
    }
    mpz_clears(bx.v, bx.s, by.v, by.s, NULL);
    mpz_clears(G, S, Emax, al, be, t, M0, E, a1, a2, sa1, sa2, x, g, pz, G2, t1, t2, NULL);
    liberer(fG, nG);
    return ret;
}
