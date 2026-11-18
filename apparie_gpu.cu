// apparie_gpu.cu — boucle des couples (u0, u2) et des p de apparie_graines.py sur GPU.
//
// Meme calcul que apparie_couples2 (borho4c.c), partie « crible » :
//   alpha = G*(s0*u2 + s2*u0) - S*s0*s2 > 0,  beta = s0*(G*u2 - S*s2) < 0,
//   pmin = max(floor(-beta/alpha) + 1, P0),  pmax = min(floor((Emax - beta)/alpha), pmin + maxp),
//   pour p de pmin a pmax : E = alpha*p + beta ; garder p si E | M0,
//   M0 = |u0*beta + alpha*u2| * G^2 * pgcd(u0, u2) * s2   (teste modulo E sans former M0 et sans division :
//   valuation 2-adique + reductions de Montgomery modulo la partie impaire de E).
// Theoreme 2 exact et primalite de p sont aussi faits ici ; les p gardes (rares) sont rendus a Python,
// qui refait CHAQUE couple (u0, u2, p) en C (apparie_couples_p : theoreme 2, primalite, theoreme 1).
//
// Domaine (controle par apparie_graines.py, sinon chemin C habituel) : G < 2^44, S < 2^46 et, avec bG, bS,
// bu, bs les tailles en bits de G, S, max u, max sigma(u) :
//   bu + bs + max(bG+bu, bS+bs) <= 125,  bG + bs + 2*bu + 1 <= 125,  bu <= 31,  bs <= 33
// (tout tient alors en entiers 128 bits ; u jusqu'a ~2^30 pour les petits G).
// E < 2^63 est teste en 64 bits. Mode hybride (ag_emax_vrai) : quand K*G^2/S >= 2^63 (G au-dela de ~1,7e12),
// les p avec 2^63 <= E <= K*G^2/S < 2^127 sont testes en 128 bits et rendus avec le bit 63 leve.
// Un couple hors domaine (alpha >= 2^96, P0 >= 2^30, p >= 2^62) est rendu avec p = 0 : Python le fait en C.
//
// GPU :  nvcc -O3 -arch=sm_120 -maxrregcount=64 -Xcompiler -fPIC -shared -o libapparie_gpu.so apparie_gpu.cu
//         (-maxrregcount=64 : 96 registres par fil sinon, occupation 33 % ; mesure RTX 50 : 1,32 fois plus rapide)
// CPU :  g++ -O3 -fopenmp -fPIC -shared -x c++ -o libapparie_cpu.so apparie_gpu.cu   (controle)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(_OPENMP) && !defined(__CUDACC__)
#include <omp.h>
#endif

typedef __int128 i128;
typedef unsigned __int128 u128;

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif

struct Sortie { int32_t i, j; uint64_t p; };

HD uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) { return (uint64_t)(((u128)a * b) % m); }

HD uint64_t pgcd(uint64_t a, uint64_t b) { while (b) { uint64_t r = a % b; a = b; b = r; } return a; }

HD uint64_t powmod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m; b %= m;
    while (e) { if (e & 1) r = mulmod(r, b, m); b = mulmod(b, b, m); e >>= 1; }
    return r;
}

HD bool premier64(uint64_t n) {                    // Miller-Rabin deterministe pour n < 2^64
    if (n < 2) return false;
    const uint64_t pp[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) { if (n % pp[i] == 0) return n == pp[i]; }
    uint64_t d = n - 1; int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    for (int i = 0; i < 12; i++) {
        uint64_t x = powmod(pp[i], d, n);
        if (x == 1 || x == n - 1) continue;
        bool ok = false;
        for (int k = 1; k < r; k++) { x = mulmod(x, x, n); if (x == n - 1) { ok = true; break; } }
        if (!ok) return false;
    }
    return true;
}

HD int ctz64(uint64_t x) {                         // x != 0
#ifdef __CUDA_ARCH__                               // code GPU (defini seulement a la compilation du noyau)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}

HD uint64_t ninv64(uint64_t n) {                   // -1/n mod 2^64, n impair (Newton, sans division)
    // variante de Dumas : y = 1 - n*x est eleve au carre en parallele des produits x*(1 + y), d'ou une
    // chaine de dependances plus courte que Newton (meme nombre de multiplications)
    uint64_t x = (3 * n) ^ 2;                      // 5 bits justes
    uint64_t y = 1 - n * x;
    x *= 1 + y; y *= y;                            // 10 bits
    x *= 1 + y; y *= y;                            // 20 bits
    x *= 1 + y; y *= y;                            // 40 bits
    x *= 1 + y;                                    // 80 bits
    return 0 - x;
}

// REDC : T * 2^-64 mod n, pour n impair < 2^63 et T < n * 2^64 ; q = ninv64(n)
HD uint64_t redc(u128 T, uint64_t n, uint64_t q) {
    uint64_t m = (uint64_t)T * q;
    uint64_t t = (uint64_t)((T + (u128)m * n) >> 64);
    return t >= n ? t - n : t;
}

// theoreme 2 exact (comme apparie_couples2) : D = S*E divise (a1 - a2)*pgcd(sa1, sa2)*pgcd(a1, a2), avec
// a1 = G*u0*p, a2 = G*u2, sa1 = S*s0*(p+1), sa2 = S*s2, soit  E | G^2 * (u0*p - u2) * g1 * g2,
// g1 = pgcd(s0*(p+1), s2), g2 = pgcd(u0*p, u2).
HD bool theoreme2(uint64_t G, uint64_t u0, uint64_t s0, uint64_t u2, uint64_t s2, uint64_t p, uint64_t E) {
    u128 a = (u128)u0 * p;
    uint64_t d = (uint64_t)(a >= u2 ? (a - u2) % E : (E - (uint64_t)(((u128)u2 - a) % E)) % E);
    if (d == 0) return true;
    uint64_t gm = G % E;
    uint64_t r = mulmod(mulmod(gm, gm, E), d, E);
    if (r == 0) return true;
    uint64_t g1 = pgcd(s2, (uint64_t)(((u128)s0 * (p + 1)) % s2));
    uint64_t g2 = pgcd(u2, (uint64_t)(a % u2));
    r = mulmod(mulmod(r, g1 % E, E), g2 % E, E);
    return r == 0;
}

// ---- suite 128 bits (mode hybride) : E entre 2^63 et Evrai = K*G^2/S < 2^127 ----
// redc128 : (T2:T1:T0) * 2^-64 mod n, n impair < 2^127, T < n * 2^64 (ou T < 2^127, cas de |X| d'un i128 : resultat < 2^63 + n, a reduire par l'appelant) ;
// q = ninv64(poids faible de n). Un seul pas de Montgomery : le facteur 2^-64 est inversible modulo n,
// donc le test « = 0 » reste exact, comme dans la version 64 bits.
HD u128 redc128(uint64_t T0, uint64_t T1, uint64_t T2, u128 n, uint64_t q) {
    uint64_t m = T0 * q;
    u128 x = (u128)m * (uint64_t)n;
    u128 y = (u128)m * (uint64_t)(n >> 64) + (uint64_t)(x >> 64);
    u128 s1 = (u128)T1 + (uint64_t)y + (T0 ? 1 : 0);              // T0 + poids faible de m*n = 0 ou 2^64
    u128 s2 = (u128)T2 + (uint64_t)(y >> 64) + (uint64_t)(s1 >> 64);
    u128 t = (s2 << 64) | (uint64_t)s1;
    return t >= n ? t - n : t;
}
HD u128 mr128(u128 a, uint64_t b, u128 n, uint64_t q) {           // a * b * 2^-64 mod n, a < n
    u128 lo = (u128)(uint64_t)a * b;
    u128 hi = (u128)(uint64_t)(a >> 64) * b + (uint64_t)(lo >> 64);
    return redc128((uint64_t)lo, (uint64_t)hi, (uint64_t)(hi >> 64), n, q);
}
#define P128 ((uint64_t)1 << 63)                 // marque d'un p rendu par la suite 128 bits

// Quotient et reste de n / a quand le quotient est petit (< 2^20 : > 99,8 % des couples, mesure) :
// estimation en float (24 bits, erreur relative < 3 * 2^-24, donc erreur absolue < 0,2) puis correction
// EXACTE par le reste. Remplace la division entiere, de loin l'operation la plus chere par couple sur GPU
// (pas de diviseur 64 bits materiel ; la division 128 bits est une boucle logicielle, CPU compris).
// Renvoie false si le quotient estime est >= 2^20 : l'appelant fait alors la division entiere.
HD float f128(u128 x) { return (float)(uint64_t)(x >> 64) * 18446744073709551616.0f + (float)(uint64_t)x; }
HD bool div_petit(u128 n, u128 a, uint64_t *q, u128 *r) {         // a > 0, n < 2^127, a < 2^96
    float qf = f128(n) / f128(a);
    if (!(qf < 1048576.0f)) return false;
    uint64_t q_ = (uint64_t)qf;                                    // q_ * a < 2^117 : pas de debordement
    i128 rr = (i128)n - (i128)(a * q_);
    while (rr < 0) { q_--; rr += (i128)a; }
    while (rr >= (i128)a) { q_++; rr -= (i128)a; }
    *q = q_; *r = (u128)rr;
    return true;
}
HD bool div_petit64(uint64_t n, uint64_t a, uint64_t *q, uint64_t *r) {
    float qf = (float)n / (float)a;
    if (!(qf < 1048576.0f)) return false;
    uint64_t q_ = (uint64_t)qf;
    i128 rr = (i128)n - (i128)((u128)q_ * a);
    while (rr < 0) { q_--; rr += a; }
    while (rr >= (i128)a) { q_++; rr -= a; }
    *q = q_; *r = (uint64_t)rr;
    return true;
}

// ---- un couple = un etat + des pas ----
// init_couple prepare le couple (rejet rapide, premier p, premier E) ; pas_couple teste UN p et avance.
// Ce decoupage sert au GPU : chaque fil enchaine « prendre un couple » et « un pas », sans attendre les
// 31 autres fils de son warp (avant : la boucle des p d'un seul fil immobilisait tout le warp).
// mode : 0 = pas de couple en cours, 1 = E < 2^63 (64 bits), 2 = 2^63 <= E <= Evrai (128 bits, hybride).
struct Etat { int mode, v2X, v2L; uint64_t a, p, c, xl, xh, g0; u128 E, al; };

HD u128 mul64x32(uint64_t a, uint32_t b) {         // a * b exact (< 2^96), en deux produits 32 x 32 -> 64
    uint64_t lo = (uint64_t)(uint32_t)a * b;
    uint64_t hi = (a >> 32) * (uint64_t)b;
    return ((u128)hi << 32) + lo;
}

template <typename F>
HD void init_couple(int64_t u0, int64_t s0, int64_t u2, uint64_t G, uint64_t s2, uint64_t P0, uint64_t Emax,
                    u128 Evrai, i128 A, i128 B, Etat &e, F emit) {
    // A = G*u2 - S*s2 et B = G*s2 ne dependent que de u2 : calcules une fois par fenetre par l'appelant
    // (beta = s0*A, alpha = beta + u0*B).
    e.mode = 0;
    i128 be, al;
    if (A < 0 && !((u128)(-A) >> 64) && B >= 0 && !((u128)B >> 64) && !((uint64_t)s0 >> 32) && !((uint64_t)u0 >> 32)) {
        // cas courant (petits G) : |A|, B sur 64 bits et s0, u0 sur 32 bits -> deux produits 64 x 32 bits
        // en mots de 32 bits (2 multiplications machine chacun sur GPU, au lieu d'un produit 128 x 64 emule)
        u128 x = mul64x32((uint64_t)(-A), (uint32_t)s0);
        u128 y = mul64x32((uint64_t)B, (uint32_t)u0);
        be = -(i128)x;
        al = (i128)y - (i128)x;
    } else {
        be = A * s0;
        al = be + B * u0;
    }
    if (al <= 0 || be >= 0) return;
    u128 nb = (u128)(-be);
    uint64_t a, pmin;
    i128 E0;
    if (al >> 64) {
        // alpha >= 2^64 (gros G) : E augmente de alpha > 2^63 a chaque p ; sans mode hybride un seul p est
        // possible, p = max(floor(-beta/alpha) + 1, P0).
        if ((al >> 96) || (P0 >> 30)) { emit(0); return; }           // hors domaine : couple entier en C
        uint64_t qg; u128 rg;
        if ((u128)al * P0 > nb) { pmin = P0; E0 = (i128)((u128)al * P0) - (i128)nb; }   // sans division
        else if (div_petit(nb, (u128)al, &qg, &rg)) {              // t = qg + 1 >= P0 ici
            pmin = qg + 1; E0 = al - (i128)rg;                     // E0 = alpha*pmin - nb = alpha - reste
        } else {
            u128 t = nb / (u128)al + 1;
            pmin = t < P0 ? P0 : (uint64_t)t;                      // t < 2^63 (nb < 2^127, alpha >= 2^64)
            E0 = (i128)((u128)al * pmin) - (i128)nb;               // > 0 ; pas de debordement (pmin < 2^30 ou = t)
        }
        a = ~(uint64_t)0;                                          // force la sortie du mode 64 bits apres ce p
    } else {
        a = (uint64_t)al;
        // pmin = max(floor(nb/a) + 1, P0) : P0 si a*P0 > nb (un produit suffit), sinon division
        if ((u128)a * P0 > nb) { pmin = P0; E0 = (i128)((u128)a * P0) - (i128)nb; }
        else if ((nb >> 64) == 0) {
            // pmin = nb/a + 1 (> P0 ici) et E0 = a*pmin - nb = a - reste (rejet E0 > Emax sans multiplication)
            uint64_t n64 = (uint64_t)nb, q, r;
#if defined(__CUDA_ARCH__) || defined(AG_F32)
            if (!div_petit64(n64, a, &q, &r)) { q = n64 / a; r = n64 - q * a; }
#else
            double qd = (double)n64 / (double)a;
            if (qd < 1125899906842624.0) {                         // 2^50 : erreur de l'estimation < 1
                q = (uint64_t)qd;
                i128 rr = (i128)n64 - (i128)((u128)q * a);
                while (rr < 0) { q--; rr += a; }
                while (rr >= (i128)a) { q++; rr -= a; }
                r = (uint64_t)rr;
            } else { q = n64 / a; r = n64 - q * a; }
#endif
            pmin = q + 1; E0 = (i128)(a - r);
        } else {
            u128 t = nb / a + 1;
            if (t >> 64) return;                                   // C : stats[3] (hors 64 bits), saute
            pmin = (uint64_t)t;
            E0 = (i128)a * pmin + be;                              // > 0
        }
    }
    if (E0 > (i128)Emax) {                                         // aucun p avec E < 2^63
        if ((u128)E0 > Evrai) return;                              // (Evrai = 0 hors mode hybride)
        if (pmin >> 62) { emit(0); return; }                       // hors domaine : couple entier en C
        e.mode = 2;
    } else e.mode = 1;
    e.E = (u128)E0; e.al = (u128)al; e.a = a; e.p = pmin; e.c = 0; e.g0 = 0;
    // E | M0 = Xa * G^2 * g0 * s2 (g0 = pgcd(u0, u2)), sans division : E = 2^k * E1 (E1 impair) ;
    //   2^k | M0  <=>  k <= v2(M0) (calcule une fois par couple) ;
    //   E1 | M0   <=>  la chaine de reductions de Montgomery modulo E1 finit a 0
    //                  (chaque REDC multiplie par 2^-64, inversible modulo E1 : le test « = 0 » est exact).
    // Le pgcd g0 n'est calcule que si le test passe avec u2 a sa place (u2 est un multiple de g0).
    i128 X = (i128)u0 * be + al * u2;
    u128 Xa = (u128)(X < 0 ? -X : X);
    e.xh = (uint64_t)(Xa >> 64); e.xl = (uint64_t)Xa;
    e.v2X = (Xa == 0 ? 1000 : (e.xl ? ctz64(e.xl) : 64 + ctz64(e.xh))) + 2 * ctz64(G) + ctz64(s2);
    e.v2L = e.v2X + ctz64((uint64_t)u2);
}

// teste le p courant (e.p, rang e.c) puis passe au suivant, ou termine le couple (e.mode = 0)
template <typename F>
HD void pas_couple(Etat &e, uint64_t G, uint64_t u0, uint64_t s0, uint64_t u2, uint64_t s2, uint64_t maxp,
                   uint64_t Emax, u128 Evrai, uint64_t *nth2, F emit) {
    if (e.mode == 1) {
        uint64_t E = (uint64_t)e.E;
        int k = ctz64(E);
        if (k <= e.v2L) {
            uint64_t E1 = E >> k;
            uint64_t q = ninv64(E1);
            uint64_t r = redc((u128)e.xl, E1, q) + (e.xh < E1 ? e.xh : e.xh % E1);   // = Xa * 2^-64 mod E1
            if (r >= E1) r -= E1;
            if (r) r = redc((u128)r * G, E1, q);
            if (r) r = redc((u128)r * G, E1, q);
            if (r) r = redc((u128)r * s2, E1, q);
            if (r == 0 || redc((u128)r * u2, E1, q) == 0) {
                if (!e.g0) e.g0 = pgcd(u0, u2);
                uint64_t p = e.p;
                if (k <= e.v2X + ctz64(e.g0) && (r == 0 || redc((u128)r * e.g0, E1, q) == 0)
                    && theoreme2(G, u0, s0, u2, s2, p, E)) {
                    (*nth2)++;
                    if (G % p && u0 % p && u2 % p && premier64(p)) emit(p);
                }
            }
        }
        if (e.c >= maxp) { e.mode = 0; return; }
        if (e.a > Emax - E) {                                      // p suivant : E + alpha > Emax
            u128 En = (u128)E + e.al;                              // mode hybride : la suite en 128 bits
            if (Evrai && En <= Evrai) {
                if ((e.p + 1) >> 62) { emit(0); e.mode = 0; }      // hors domaine : couple entier en C
                else { e.mode = 2; e.E = En; e.p++; e.c++; }
            } else e.mode = 0;
            return;
        }
        e.E = E + e.a; e.p++; e.c++;
    } else {
        // 2^63 <= E <= Evrai < 2^127 : meme test E | M0 en 128 bits. Le theoreme 2 n'est pas fait ici : le p
        // (premier, ne divisant pas G*u0*u2) est rendu avec le bit 63 leve et le C (apparie_couples_p, GMP)
        // fait le theoreme 2 puis le theoreme 1.
        u128 E = e.E;
        uint64_t el = (uint64_t)E;
        int k = el ? ctz64(el) : 64 + ctz64((uint64_t)(E >> 64));
        if (k <= e.v2L) {
            u128 E1 = E >> k;
            uint64_t q = ninv64((uint64_t)E1);
            u128 r = redc128(e.xl, e.xh, 0, E1, q);                // = Xa * 2^-64 mod E1
            if (r >= E1) r %= E1;                                  // seulement si E1 < 2^63
            if (r) r = mr128(r, G, E1, q);
            if (r) r = mr128(r, G, E1, q);
            if (r) r = mr128(r, s2, E1, q);
            if (r == 0 || mr128(r, u2, E1, q) == 0) {
                if (!e.g0) e.g0 = pgcd(u0, u2);
                uint64_t p = e.p;
                if (k <= e.v2X + ctz64(e.g0) && (r == 0 || mr128(r, e.g0, E1, q) == 0)
                    && G % p && u0 % p && u2 % p && premier64(p)) emit(p | P128);
            }
        }
        if (e.c >= maxp) { e.mode = 0; return; }
        e.E = E + e.al; e.p++; e.c++;                              // E <= Evrai < 2^127, alpha < 2^96
        if (e.E > Evrai) e.mode = 0;
    }
}

// ---- un fil (« voie ») d'un warp de 32 ----
// Les 32 voies d'un warp se partagent une tuile de couples consecutifs [suiv, fin). A chaque tour, les voies
// sans couple en cours (masque m, le meme pour toutes) prennent les couples suivants, dans l'ordre des
// voies ; puis chaque voie qui a un couple fait UN pas. Les champs b, suiv, fin, epuise evoluent de la
// meme facon dans les 32 voies (ils ne dependent que de m) : aucune memoire partagee.
// Le meme code tourne sur GPU (m = __ballot_sync) et sur CPU avec -DAG_SIMU (32 voies simulees), ce qui
// permet de controler cette logique sans GPU.
struct Voie {
    int voie, w, wA, tw_ok, i, j, epuise;
    int64_t b, suiv, fin, saut, t1, base, u0, s0, u2, s2;
    i128 A, B;
    Etat e;
    uint64_t loc, loc2;
};
HD int popc32(unsigned x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}
HD int fenetre(const int64_t *cs, int nw, int64_t t);
HD void voie_debut(Voie &v, int voie, int64_t b, int64_t saut, int64_t tuile, int64_t t1) {
    v.voie = voie; v.w = 0; v.wA = -1; v.tw_ok = 0; v.i = v.j = 0; v.base = 0;
    v.b = b; v.saut = saut; v.t1 = t1; v.loc = v.loc2 = 0; v.e.mode = 0;
    v.epuise = b >= t1;
    v.suiv = v.epuise ? 0 : b;
    v.fin = v.epuise ? 0 : (b + tuile < t1 ? b + tuile : t1);
    v.u0 = v.s0 = v.u2 = v.s2 = 0; v.A = v.B = 0;
}
template <typename F, typename TU>
HD void voie_tour(Voie &v, unsigned m, int64_t tuile, const TU *U, const TU *Sg, const int32_t *jx,
                  const int32_t *lo, const int64_t *cs, int nw, int64_t G, int64_t S, uint64_t P0,
                  uint64_t maxp, uint64_t Emax, u128 Evrai, F emit) {
    int64_t dispo = v.fin - v.suiv;
    int pris = popc32(m);
    if ((int64_t)pris > dispo) pris = (int)dispo;
    if (v.e.mode == 0) {
        int r = popc32(m & ((1u << v.voie) - 1));                  // rang parmi les voies qui demandent
        if ((int64_t)r < dispo) {
            int64_t t = v.suiv + r;
            if (!v.tw_ok) {                                        // nouvelle tuile : fenetre cherchee une fois
                v.w = fenetre(cs, nw, t); v.base = v.w ? cs[v.w - 1] : 0; v.tw_ok = 1;
            }
            while (t >= cs[v.w]) v.base = cs[v.w++];
            v.i = lo[v.w] + (int)(t - v.base); v.j = jx[v.w];
            if (v.w != v.wA) {
                v.wA = v.w; v.u2 = U[v.j]; v.s2 = Sg[v.j];
                v.A = (i128)G * v.u2 - (i128)S * v.s2; v.B = (i128)G * v.s2;
            }
            v.u0 = U[v.i]; v.s0 = Sg[v.i];
            init_couple(v.u0, v.s0, v.u2, (uint64_t)G, (uint64_t)v.s2, P0, Emax, Evrai, v.A, v.B, v.e, emit);
        }
    }
    v.suiv += pris;
    if (!v.epuise && v.suiv >= v.fin) {                            // tuile finie : la suivante de ce warp
        v.b += v.saut; v.tw_ok = 0;
        if (v.b < v.t1) { v.suiv = v.b; v.fin = v.b + tuile < v.t1 ? v.b + tuile : v.t1; }
        else { v.epuise = 1; v.suiv = v.fin = 0; }
    }
    if (v.e.mode) {
        v.loc++;
        pas_couple(v.e, (uint64_t)G, (uint64_t)v.u0, (uint64_t)v.s0, (uint64_t)v.u2, (uint64_t)v.s2, maxp, Emax,
                   Evrai, &v.loc2, emit);
    }
}

// couple numero t (0 <= t < tot) : fenetre w telle que cs[w-1] <= t < cs[w] ; u2 = jx[w], u0 = lo[w] + t - cs[w-1]
HD int fenetre(const int64_t *cs, int nw, int64_t t) {
    int lo_ = 0, hi = nw - 1;
    while (lo_ < hi) { int m = (lo_ + hi) / 2; if (cs[m] > t) hi = m; else lo_ = m + 1; }
    return lo_;
}

static int64_t hG, hS; static uint64_t hP0, hmaxp, hEmax; static int hnw;
static u128 hEvrai = 0;
// vraie borne de E quand Emax a ete ramenee a 2^63 - 1 (0 sinon) ; a appeler apres ag_charger
extern "C" void ag_emax_vrai(uint64_t hi, uint64_t lo) { hEvrai = ((u128)hi << 64) | lo; }

#ifdef __CUDACC__

#define TUILE ((int64_t)32 * 64)                 // couples par tuile de warp (64 par fil)

static int64_t *dU, *dS, *dcs; static int32_t *djx, *dlo; static Sortie *dout; static int *dno;
static int32_t *dU32, *dS32; static int h32 = 0;     // u et sigma(u) sur 32 bits quand ils y tiennent : trafic memoire / 2
static unsigned long long *dnt, *dn2; static int dcap = 0;

template <typename TU>
__global__ void k_scan(int64_t t0, int64_t t1, const TU *U, const TU *Sg, const int32_t *jx,
                       const int32_t *lo, const int64_t *cs, int nw, int64_t G, int64_t S, uint64_t P0,
                       uint64_t maxp, uint64_t Emax, Sortie *out, int cap, int *no, unsigned long long *nt,
                       unsigned long long *n2, uint64_t Evhi, uint64_t Evlo) {
    const u128 Evrai = ((u128)Evhi << 64) | Evlo;
    // Chaque warp traite des tuiles de TUILE couples consecutifs (voir struct Voie).
    const int voie = threadIdx.x & 31;
    const int64_t warp = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    const int64_t nwarps = ((int64_t)gridDim.x * blockDim.x) >> 5;
    Voie v;
    voie_debut(v, voie, t0 + warp * TUILE, nwarps * TUILE, TUILE, t1);
    auto emit = [&](uint64_t p) {
        int k = atomicAdd(no, 1);
        if (k < cap) { out[k].i = v.i; out[k].j = v.j; out[k].p = p; }
    };
    for (;;) {
        unsigned m = __ballot_sync(0xffffffffu, v.e.mode == 0);
        voie_tour(v, m, TUILE, U, Sg, jx, lo, cs, nw, G, S, P0, maxp, Emax, Evrai, emit);
        if (__ballot_sync(0xffffffffu, v.e.mode != 0) == 0 && v.epuise) break;
    }
    uint64_t loc = v.loc, loc2 = v.loc2;
    atomicAdd(nt, (unsigned long long)loc);
    atomicAdd(n2, (unsigned long long)loc2);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA : %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

// libere tout ce qui a ete alloue (pointeurs remis a NULL : appel sans risque a tout moment). Appele au debut
// de ag_charger : si une allocation avait echoue au G precedent (ag_charger rend -1 sans ag_liberer), la memoire
// de la carte n'est plus perdue
static void liberer_tout(void) {
    void **pp[] = {(void **)&dU, (void **)&dS, (void **)&dU32, (void **)&dS32, (void **)&djx, (void **)&dlo,
                   (void **)&dcs, (void **)&dout, (void **)&dno, (void **)&dnt, (void **)&dn2};
    for (unsigned k = 0; k < sizeof(pp) / sizeof(pp[0]); k++)
        if (*pp[k]) { cudaFree(*pp[k]); *pp[k] = NULL; }
}

extern "C" int ag_charger(const int64_t *U, const int64_t *Sg, int nu, const int32_t *jx, const int32_t *lo,
                          const int64_t *cs, int nw, int64_t G, int64_t S, uint64_t P0, uint64_t maxp,
                          uint64_t Emax, int cap) {
    hG = G; hS = S; hP0 = P0; hmaxp = maxp; hEmax = Emax; hnw = nw; hEvrai = 0; (void)nu; dcap = cap;
    liberer_tout();
    // u et sigma(u) sur 32 bits s'ils y tiennent tous (u <= 2^31 : toujours vrai pour --u <= 2e9 et
    // sigma(u) < 2^31 pour u <= ~3e8) : le noyau lit alors 8 octets par couple au lieu de 16
    h32 = 1;
    for (int k = 0; k < nu && h32; k++)
        if (U[k] < 0 || U[k] > 2147483647LL || Sg[k] < 0 || Sg[k] > 2147483647LL) h32 = 0;
    if (h32) {
        int32_t *t = (int32_t *)malloc((size_t)(nu ? nu : 1) * 4);
        if (!t) return -1;
        CK(cudaMalloc(&dU32, (size_t)nu * 4)); CK(cudaMalloc(&dS32, (size_t)nu * 4));
        for (int k = 0; k < nu; k++) t[k] = (int32_t)U[k];
        CK(cudaMemcpy(dU32, t, (size_t)nu * 4, cudaMemcpyHostToDevice));
        for (int k = 0; k < nu; k++) t[k] = (int32_t)Sg[k];
        CK(cudaMemcpy(dS32, t, (size_t)nu * 4, cudaMemcpyHostToDevice));
        free(t);
    } else {
        CK(cudaMalloc(&dU, (size_t)nu * 8)); CK(cudaMalloc(&dS, (size_t)nu * 8));
        CK(cudaMemcpy(dU, U, (size_t)nu * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dS, Sg, (size_t)nu * 8, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&djx, (size_t)nw * 4)); CK(cudaMalloc(&dlo, (size_t)nw * 4)); CK(cudaMalloc(&dcs, (size_t)nw * 8));
    CK(cudaMalloc(&dout, (size_t)cap * sizeof(Sortie))); CK(cudaMalloc(&dno, sizeof(int)));
    CK(cudaMalloc(&dnt, sizeof(unsigned long long))); CK(cudaMalloc(&dn2, sizeof(unsigned long long)));
    CK(cudaMemcpy(djx, jx, (size_t)nw * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlo, lo, (size_t)nw * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcs, cs, (size_t)nw * 8, cudaMemcpyHostToDevice));
    return 0;
}

// couples t0 <= t < t1 ; renvoie le nombre de sorties (-2 : tampon trop petit, refaire en plus petit)
extern "C" int ag_scan(int64_t t0, int64_t t1, int32_t *oi, int32_t *oj, uint64_t *op, unsigned long long *ntest,
                       unsigned long long *nth2) {
    CK(cudaMemset(dno, 0, sizeof(int))); CK(cudaMemset(dnt, 0, sizeof(unsigned long long)));
    CK(cudaMemset(dn2, 0, sizeof(unsigned long long)));
    int64_t m = t1 - t0;
    // blocs de 256 fils = 8 warps = 8 tuiles par tour de grille
    int64_t nb64 = (m + 8 * TUILE - 1) / (8 * TUILE);
    int nb = nb64 > 65536 ? 65536 : (nb64 < 1 ? 1 : (int)nb64);
    if (h32)
        k_scan<<<nb, 256>>>(t0, t1, (const int32_t *)dU32, (const int32_t *)dS32, djx, dlo, dcs, hnw, hG, hS, hP0, hmaxp,
                             hEmax, dout, dcap, dno, dnt, dn2, (uint64_t)(hEvrai >> 64), (uint64_t)hEvrai);
    else
        k_scan<<<nb, 256>>>(t0, t1, (const int64_t *)dU, (const int64_t *)dS, djx, dlo, dcs, hnw, hG, hS, hP0, hmaxp,
                             hEmax, dout, dcap, dno, dnt, dn2, (uint64_t)(hEvrai >> 64), (uint64_t)hEvrai);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int no; unsigned long long nt;
    CK(cudaMemcpy(&no, dno, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&nt, dnt, sizeof(nt), cudaMemcpyDeviceToHost));
    *ntest = nt;
    CK(cudaMemcpy(&nt, dn2, sizeof(nt), cudaMemcpyDeviceToHost));
    *nth2 = nt;
    if (no > dcap) return -2;
    Sortie *h = (Sortie *)malloc((size_t)(no ? no : 1) * sizeof(Sortie));
    CK(cudaMemcpy(h, dout, (size_t)no * sizeof(Sortie), cudaMemcpyDeviceToHost));
    for (int k = 0; k < no; k++) { oi[k] = h[k].i; oj[k] = h[k].j; op[k] = h[k].p; }
    free(h);
    return no;
}

extern "C" void ag_liberer(void) { liberer_tout(); }

#else   // ---- version CPU (OpenMP) : meme un_couple(), pour controle sans GPU ----

static const int64_t *cU, *cS, *ccs; static const int32_t *cjx, *clo; static int ccap;

extern "C" int ag_charger(const int64_t *U, const int64_t *Sg, int nu, const int32_t *jx, const int32_t *lo,
                          const int64_t *cs, int nw, int64_t G, int64_t S, uint64_t P0, uint64_t maxp,
                          uint64_t Emax, int cap) {
    hG = G; hS = S; hP0 = P0; hmaxp = maxp; hEmax = Emax; hnw = nw; hEvrai = 0; (void)nu;
    cU = U; cS = Sg; cjx = jx; clo = lo; ccs = cs; ccap = cap;   // tableaux numpy gardes par Python
    return 0;
}

extern "C" int ag_scan(int64_t t0, int64_t t1, int32_t *oi, int32_t *oj, uint64_t *op, unsigned long long *ntest,
                       unsigned long long *nth2) {
    int no = 0;
    unsigned long long nt = 0, n2 = 0;
#ifdef AG_SIMU
    // controle de la logique des warps du GPU : 32 voies simulees par « warp », memes fonctions
    const int64_t TU = 2048, NWARP = 64;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:nt, n2)
    for (int64_t wp = 0; wp < NWARP; wp++) {
        Voie V[32];
        for (int l = 0; l < 32; l++) voie_debut(V[l], l, t0 + wp * TU, NWARP * TU, TU, t1);
        for (;;) {
            unsigned m = 0;
            for (int l = 0; l < 32; l++) m |= (unsigned)(V[l].e.mode == 0) << l;
            unsigned actifs = 0;
            for (int l = 0; l < 32; l++) {
                Voie &v = V[l];
                voie_tour(v, m, TU, cU, cS, cjx, clo, ccs, hnw, hG, hS, hP0, hmaxp, hEmax, hEvrai,
                          [&](uint64_t p) {
                    int k;
                    #pragma omp atomic capture
                    k = no++;
                    if (k < ccap) { oi[k] = v.i; oj[k] = v.j; op[k] = p; }
                });
                actifs |= (unsigned)(v.e.mode != 0) << l;
            }
            if (!actifs && V[0].epuise) break;
        }
        for (int l = 0; l < 32; l++) { nt += V[l].loc; n2 += V[l].loc2; }
    }
#else
    // blocs de BLOC couples consecutifs : la fenetre (u2) n'est cherchee qu'une fois par bloc, puis avancee
    // lineairement (une fenetre compte en moyenne des centaines de couples)
    const int64_t BLOC = 4096;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:nt, n2)
    for (int64_t b = t0; b < t1; b += BLOC) {
        int64_t fin = b + BLOC < t1 ? b + BLOC : t1;
        int w = fenetre(ccs, hnw, b);
        int64_t base = w ? ccs[w - 1] : 0;
        int wA = -1; int64_t u2 = 0, s2 = 0; i128 A = 0, B = 0;
        for (int64_t t = b; t < fin; t++) {
            while (t >= ccs[w]) base = ccs[w++];
            int i = clo[w] + (int)(t - base), j = cjx[w];
            if (w != wA) { wA = w; u2 = cU[j]; s2 = cS[j]; A = (i128)hG * u2 - (i128)hS * s2; B = (i128)hG * s2; }
            uint64_t l2 = 0;
            auto emit = [&](uint64_t p) {
                int k;
                #pragma omp atomic capture
                k = no++;
                if (k < ccap) { oi[k] = i; oj[k] = j; op[k] = p; }
            };
            Etat e;
            init_couple(cU[i], cS[i], u2, (uint64_t)hG, (uint64_t)s2, hP0, hEmax, hEvrai, A, B, e, emit);
            while (e.mode) {
                nt++;
                pas_couple(e, (uint64_t)hG, (uint64_t)cU[i], (uint64_t)cS[i], (uint64_t)u2, (uint64_t)s2, hmaxp,
                           hEmax, hEvrai, &l2, emit);
            }
            n2 += l2;
        }
    }
#endif
    *ntest = nt;
    *nth2 = n2;
    return no > ccap ? -2 : no;
}

extern "C" void ag_liberer(void) {}

// nombre de fils OpenMP de ce processus (apparie_graines.py : coeurs / --jobs, sans quoi chaque processus
// prendrait tous les coeurs)
extern "C" void ag_fils(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

#endif
