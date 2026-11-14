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
// Domaine (controle par apparie_graines.py, sinon chemin C habituel) : G < 2^44, S < 2^46,
// u < 2^24, sigma(u) < 2^26, Emax < 2^63 ; tout tient alors en entiers 128 bits.
// Un couple dont alpha >= 2^64 n'a qu'un seul p possible : il est traite ici (alpha < 2^96, P0 < 2^30 ;
// au-dela, rendu avec p = 0 et Python le fait entierement en C).
//
// GPU :  nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libapparie_gpu.so apparie_gpu.cu
// CPU :  g++ -O3 -fopenmp -fPIC -shared -x c++ -o libapparie_cpu.so apparie_gpu.cu   (controle)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    uint64_t x = (3 * n) ^ 2;                      // 5 bits justes
    x *= 2 - n * x; x *= 2 - n * x; x *= 2 - n * x; x *= 2 - n * x;
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

// cherche le couple k ; ecrit via emit les p qui passent E | M0, le theoreme 2 exact et la primalite
// (ne divisant pas G*u0*u2) ; renvoie le nombre de p testes, *nth2 += nombre de p passant le theoreme 2
template <typename F>
HD uint64_t un_couple(int64_t G, int64_t S, int64_t u0, int64_t s0, int64_t u2, int64_t s2,
                      uint64_t P0, uint64_t maxp, uint64_t Emax, uint64_t *nth2, F emit) {
    i128 al = (i128)G * ((i128)s0 * u2 + (i128)s2 * u0) - (i128)S * s0 * s2;
    i128 be = (i128)s0 * ((i128)G * u2 - (i128)S * s2);
    if (al <= 0 || be >= 0) return 0;
    u128 nb = (u128)(-be);
    uint64_t a, pmin;
    i128 E0;
    if (al >> 64) {
        // alpha >= 2^64 (gros G) : E augmente de alpha > Emax a chaque p, donc un seul p est possible,
        // p = max(floor(-beta/alpha) + 1, P0). Traite ici (une division 128 bits par couple) au lieu
        // de rendre le couple a Python : sinon les gros G saturent le tampon et finissent sur CPU.
        if ((al >> 96) || (P0 >> 30)) { emit(0); return 0; }         // hors domaine : couple entier en C
        u128 t = nb / (u128)al + 1;
        pmin = t < P0 ? P0 : (uint64_t)t;                          // t < 2^63 (nb < 2^127, alpha >= 2^64)
        E0 = (i128)((u128)al * pmin) - (i128)nb;                   // > 0 ; pas de debordement (pmin < 2^30 ou = t)
        a = ~(uint64_t)0;                                          // force l'arret apres ce p
    } else {
        a = (uint64_t)al;
        // pmin = max(floor(nb/a) + 1, P0) : le plus souvent P0 (un produit suffit), sinon division 64 bits si possible
        if ((u128)a * P0 > nb) pmin = P0;
        else if ((nb >> 64) == 0) pmin = (uint64_t)nb / a + 1;     // > P0 ici ; pas de depassement (nb >= a*P0)
        else {
            u128 t = nb / a + 1;
            if (t >> 64) return 0;                                 // C : stats[3] (hors 64 bits), saute
            pmin = (uint64_t)t;
        }
        E0 = (i128)a * pmin + be;                                  // > 0
    }
    if (E0 > (i128)Emax) return 0;                                 // aucun p : E <= Emax < 2^63 impossible
    uint64_t E = (uint64_t)E0;
    i128 X = (i128)u0 * be + (i128)al * u2;
    u128 Xa = (u128)(X < 0 ? -X : X);
    // E | M0 = Xa * G^2 * g0 * s2 (g0 = pgcd(u0, u2)), sans division : E = 2^k * E1 (E1 impair) ;
    //   2^k | M0  <=>  k <= v2(M0) (calcule une fois par couple) ;
    //   E1 | M0   <=>  la chaine de reductions de Montgomery modulo E1 finit a 0
    //                  (chaque REDC multiplie par 2^-64, inversible modulo E1 : le test « = 0 » est exact).
    // Le pgcd g0 n'est calcule que si le test passe avec u2 a sa place (u2 est un multiple de g0).
    uint64_t xh = (uint64_t)(Xa >> 64), xl = (uint64_t)Xa;
    int v2X = Xa == 0 ? 1000 : (xl ? ctz64(xl) : 64 + ctz64(xh));
    v2X += 2 * ctz64((uint64_t)G) + ctz64((uint64_t)s2);
    int v2L = v2X + ctz64((uint64_t)u2);
    uint64_t g0 = 0;
    uint64_t n = 0;
    for (uint64_t c = 0;; c++) {
        n++;
        int k = ctz64(E);
        if (k <= v2L) {
            uint64_t E1 = E >> k;
            uint64_t q = ninv64(E1);
            uint64_t r = redc((u128)xl, E1, q) + (xh < E1 ? xh : xh % E1);   // = Xa * 2^-64 mod E1
            if (r >= E1) r -= E1;
            if (r) r = redc((u128)r * (uint64_t)G, E1, q);
            if (r) r = redc((u128)r * (uint64_t)G, E1, q);
            if (r) r = redc((u128)r * (uint64_t)s2, E1, q);
            if (r == 0 || redc((u128)r * (uint64_t)u2, E1, q) == 0) {
                if (!g0) g0 = pgcd((uint64_t)u0, (uint64_t)u2);
                uint64_t p = pmin + c;
                if (k <= v2X + ctz64(g0) && (r == 0 || redc((u128)r * g0, E1, q) == 0)
                    && theoreme2((uint64_t)G, (uint64_t)u0, (uint64_t)s0, (uint64_t)u2, (uint64_t)s2, p, E)) {
                    (*nth2)++;
                    if ((uint64_t)G % p && (uint64_t)u0 % p && (uint64_t)u2 % p && premier64(p)) emit(p);
                }
            }
        }
        if (c >= maxp || a > Emax - E) break;                      // p suivant : E + a > Emax
        E += a;
    }
    return n;
}

// couple numero t (0 <= t < tot) : fenetre w telle que cs[w-1] <= t < cs[w] ; u2 = jx[w], u0 = lo[w] + t - cs[w-1]
HD int fenetre(const int64_t *cs, int nw, int64_t t) {
    int lo_ = 0, hi = nw - 1;
    while (lo_ < hi) { int m = (lo_ + hi) / 2; if (cs[m] > t) hi = m; else lo_ = m + 1; }
    return lo_;
}

static int64_t hG, hS; static uint64_t hP0, hmaxp, hEmax; static int hnw;

#ifdef __CUDACC__

static int64_t *dU, *dS, *dcs; static int32_t *djx, *dlo; static Sortie *dout; static int *dno;
static unsigned long long *dnt, *dn2; static int dcap = 0;

__global__ void k_scan(int64_t t0, int64_t t1, const int64_t *U, const int64_t *Sg, const int32_t *jx,
                       const int32_t *lo, const int64_t *cs, int nw, int64_t G, int64_t S, uint64_t P0,
                       uint64_t maxp, uint64_t Emax, Sortie *out, int cap, int *no, unsigned long long *nt,
                       unsigned long long *n2) {
    uint64_t loc = 0, loc2 = 0;
    for (int64_t t = t0 + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < t1;
         t += (int64_t)gridDim.x * blockDim.x) {
        int w = fenetre(cs, nw, t);
        int64_t base = w ? cs[w - 1] : 0;
        int i = lo[w] + (int)(t - base), j = jx[w];
        loc += un_couple(G, S, U[i], Sg[i], U[j], Sg[j], P0, maxp, Emax, &loc2, [&](uint64_t p) {
            int k = atomicAdd(no, 1);
            if (k < cap) { out[k].i = i; out[k].j = j; out[k].p = p; }
        });
    }
    atomicAdd(nt, (unsigned long long)loc);
    atomicAdd(n2, (unsigned long long)loc2);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA : %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

extern "C" int ag_charger(const int64_t *U, const int64_t *Sg, int nu, const int32_t *jx, const int32_t *lo,
                          const int64_t *cs, int nw, int64_t G, int64_t S, uint64_t P0, uint64_t maxp,
                          uint64_t Emax, int cap) {
    hG = G; hS = S; hP0 = P0; hmaxp = maxp; hEmax = Emax; hnw = nw; (void)nu; dcap = cap;
    CK(cudaMalloc(&dU, (size_t)nu * 8)); CK(cudaMalloc(&dS, (size_t)nu * 8));
    CK(cudaMalloc(&djx, (size_t)nw * 4)); CK(cudaMalloc(&dlo, (size_t)nw * 4)); CK(cudaMalloc(&dcs, (size_t)nw * 8));
    CK(cudaMalloc(&dout, (size_t)cap * sizeof(Sortie))); CK(cudaMalloc(&dno, sizeof(int)));
    CK(cudaMalloc(&dnt, sizeof(unsigned long long))); CK(cudaMalloc(&dn2, sizeof(unsigned long long)));
    CK(cudaMemcpy(dU, U, (size_t)nu * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS, Sg, (size_t)nu * 8, cudaMemcpyHostToDevice));
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
    int nb = (int)((m + 255) / 256); if (nb > 65536) nb = 65536; if (nb < 1) nb = 1;
    k_scan<<<nb, 256>>>(t0, t1, dU, dS, djx, dlo, dcs, hnw, hG, hS, hP0, hmaxp, hEmax, dout, dcap, dno, dnt, dn2);
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

extern "C" void ag_liberer(void) {
    cudaFree(dU); cudaFree(dS); cudaFree(djx); cudaFree(dlo); cudaFree(dcs); cudaFree(dout); cudaFree(dno);
    cudaFree(dnt); cudaFree(dn2);
}

#else   // ---- version CPU (OpenMP) : meme un_couple(), pour controle sans GPU ----

static const int64_t *cU, *cS, *ccs; static const int32_t *cjx, *clo; static int ccap;

extern "C" int ag_charger(const int64_t *U, const int64_t *Sg, int nu, const int32_t *jx, const int32_t *lo,
                          const int64_t *cs, int nw, int64_t G, int64_t S, uint64_t P0, uint64_t maxp,
                          uint64_t Emax, int cap) {
    hG = G; hS = S; hP0 = P0; hmaxp = maxp; hEmax = Emax; hnw = nw; (void)nu;
    cU = U; cS = Sg; cjx = jx; clo = lo; ccs = cs; ccap = cap;   // tableaux numpy gardes par Python
    return 0;
}

extern "C" int ag_scan(int64_t t0, int64_t t1, int32_t *oi, int32_t *oj, uint64_t *op, unsigned long long *ntest,
                       unsigned long long *nth2) {
    int no = 0;
    unsigned long long nt = 0, n2 = 0;
    #pragma omp parallel for schedule(dynamic, 4096) reduction(+:nt, n2)
    for (int64_t t = t0; t < t1; t++) {
        int w = fenetre(ccs, hnw, t);
        int64_t base = w ? ccs[w - 1] : 0;
        int i = clo[w] + (int)(t - base), j = cjx[w];
        uint64_t l2 = 0;
        nt += un_couple(hG, hS, cU[i], cS[i], cU[j], cS[j], hP0, hmaxp, hEmax, &l2, [&](uint64_t p) {
            int k;
            #pragma omp atomic capture
            k = no++;
            if (k < ccap) { oi[k] = i; oj[k] = j; op[k] = p; }
        });
        n2 += l2;
    }
    *ntest = nt;
    *nth2 = n2;
    return no > ccap ? -2 : no;
}

extern "C" void ag_liberer(void) {}

#endif
