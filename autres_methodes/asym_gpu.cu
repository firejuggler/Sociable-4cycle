// asym_gpu.cu -- appariement au milieu de la recherche asymetrique (search_asym_gpu.py), sur GPU ou CPU.
//
// Meme source pour les deux :
//   GPU : nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libasym_gpu.so asym_gpu.cu
//   CPU : g++ -O3 -fopenmp -fPIC -shared -x c++ -o libasym_cpu.so asym_gpu.cu
//
// Pour chaque couple A = (grand cofacteur, petit cofacteur), on cherche dans la moitie B (couples de petits
// cofacteurs, tries par log f) ceux qui donnent log(1 - eps) <= sum log f < 0, puis on calcule le C reduit
// (meme calcul que la fonction run() de search_asym.py) et on garde les quadruplets avec C <= cmax
// qui passent le pre-filtre local modulo 3^K, 5^K, 7^K, 2^K (local_ok).
// Le nombre de quadruplets gardes est renvoye meme s'il depasse cap : l'appelant coupe alors la tranche en deux.

#include <cstdint>
#include <cmath>
#include <cstring>

typedef long long i64;
typedef unsigned long long u64;

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

struct Tables {
    int ns, nb, nB;                 // petits cofacteurs, grands cofacteurs, couples de la moitie B
    const i64 *X, *Y, *Z;           // f(u) = X/Y (reduit), Z = sigma(G) sigma(u) reduit
    const double *lf, *fv, *ly, *f1p, *GU;
    const double *Rs;               // moitie B : sommes log f triees
    const int *RI, *RJ;             // moitie B : indices des deux petits cofacteurs
    const i64 *U, *SU;              // cofacteurs u et sigma(u)
    i64 G, SG;                      // G et sigma(G)
    i64 fmax;                       // pre-filtre local : modules l^K <= fmax (0 = pas de pre-filtre)
};

HD static inline int ctz64(u64 x) {
#ifdef __CUDA_ARCH__
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}

HD static inline u64 gcd64(u64 a, u64 b) {      // pgcd binaire (pas de division 64 bits, lente sur GPU)
    if (a == 0) return b;
    if (b == 0) return a;
    int k = ctz64(a | b);
    a >>= ctz64(a);
    do {
        b >>= ctz64(b);
        if (a > b) { u64 t = a; a = b; b = t; }
        b -= a;
    } while (b);
    return a << k;
}

HD static inline int lower_bound_d(const double *R, int n, double v) {   // = numpy.searchsorted(R, v)
    int lo = 0, hi = n;
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (R[m] < v) lo = m + 1; else hi = m;
    }
    return lo;
}

HD static inline i64 mmod(i64 a, i64 m) { a %= m; return a < 0 ? a + m : a; }

HD static inline i64 invmod(i64 a, i64 m) {      // a inversible modulo m
    i64 t = 0, nt = 1, r = m, nr = mmod(a, m);
    while (nr) { i64 q = r / nr, x = t - q * nt; t = nt; nt = x; x = r - q * nr; r = nr; nr = x; }
    return mmod(t, m);
}

#ifdef ASYM_TEST_BRUTE   // version exhaustive O(M^2), gardee pour verifier local_ok (g++ -DASYM_TEST_BRUTE)
// Pre-filtre local (meme logique que rho() de resid.py, mais sans C, D, E) : existe-t-il des residus unites
// p, q modulo M = l^K tels que la chaine n1 = G u1 p q -> G u2 P2 -> G u3 P3 -> G u4 P4 -> n1 soit possible
// modulo les puissances de l ? Si non, le quadruplet n'a aucune solution. Perte de precision = accepte.
// Plus permissif que rho() (q n'est pas impose par l'equation bilineaire) : il ne rejette jamais un vrai cycle.
HD static inline bool local_ok_brute(i64 l, const i64 *u, const i64 *su, i64 G, i64 SG, i64 fmax) {
    i64 pw[4], us[4]; int v = 0;
    for (int i = 0; i < 4; i++) {
        i64 x = G * u[i]; pw[i] = 1;
        while (x % l == 0) { x /= l; pw[i] *= l; v++; }
        us[i] = x;
    }
    int K = v + 1 > 2 ? v + 1 : 2;
    i64 M = 1;
    for (int k = 0; k < K; k++) { M *= l; if (M > fmax) return true; }
    i64 a1 = mmod(G * u[0], M), b1 = mmod(SG * su[0], M);
    for (i64 p = 1; p < M; p++) {
        if (p % l == 0) continue;
        for (i64 q = 1; q < M; q++) {
            if (q % l == 0) continue;
            i64 n1 = a1 * (p * q % M) % M;
            i64 n = mmod(b1 * ((p + 1) * (q + 1) % M) % M - n1, M);
            i64 Mc = M; bool ok = true, unk = false;
            for (int i = 1; i < 4; i++) {
                if (Mc <= pw[i]) { unk = true; break; }
                if (n % pw[i] != 0) { ok = false; break; }
                i64 Mc2 = Mc / pw[i];
                i64 x = (n / pw[i]) % Mc2;
                if (x % l == 0) { ok = false; break; }
                i64 P = x * invmod(us[i] % Mc2, Mc2) % Mc2;
                Mc = Mc2;
                n = mmod(mmod(SG * su[i], Mc) * ((P + 1) % Mc) % Mc - mmod(G * u[i], Mc) * P % Mc, Mc);
            }
            if (unk) return true;
            if (ok && mmod(n - n1, Mc) == 0) return true;
        }
    }
    return false;
}

#endif

// Pre-filtre local (meme logique que rho() de resid.py, mais sans C, D, E) : existe-t-il des residus unites
// p, q modulo M = l^K tels que la chaine soit possible modulo les puissances de l ? Si non, aucun cycle.
// Plus permissif que rho() : il ne rejette jamais un vrai cycle. Calcul en O(M) au lieu de O(M^2) : pour p fixe, tout est affine en q. On ecrit
// q = Q0 + Qm*t (t entier libre) et chaque quantite comme a*t + b modulo Mc. Une divisibilite par pw restreint t
// a une classe (t = t0 + s*t'), ce qui garde les formes affines. A la fin, la fermeture n4' = n1 est une
// congruence lineaire en t ; il reste a verifier que q et les quotients x_i peuvent etre des unites mod l.
struct Aff { i64 a, b; };                                  // a*t + b

struct Prep { i64 l, M, A1, B1, pw[4], us[4], cs[4], cg[4]; };

// Prepare le test pour le premier l : renvoie false si le module l^K depasse fmax (test non fait = accepte).
HD static inline bool prep_local(Prep &R, i64 l, const i64 *u, const i64 *su, i64 G, i64 SG, i64 fmax) {
    int v = 0;
    R.l = l;
    for (int i = 0; i < 4; i++) {
        i64 x = G * u[i]; R.pw[i] = 1;
        while (x % l == 0) { x /= l; R.pw[i] *= l; v++; }
        R.us[i] = x;
        R.cs[i] = SG * su[i]; R.cg[i] = G * u[i];
    }
    int K = v + 1 > 2 ? v + 1 : 2;
    R.M = 1;
    for (int k = 0; k < K; k++) { R.M *= l; if (R.M > fmax) return false; }
    R.A1 = mmod(G * u[0], R.M); R.B1 = mmod(SG * su[0], R.M);
    return true;
}

// Un residu p (unite mod l) : true s'il existe q qui rend la chaine possible (ou si la precision manque).
HD static inline bool test_p(const Prep &R, i64 p) {
    const i64 l = R.l, M = R.M, A1 = R.A1, B1 = R.B1;
    const i64 *pw = R.pw, *us = R.us, *cs = R.cs, *cg = R.cg;
    // formes en t, modulo M au depart ; q = t
    Aff q = {1, 0};
    Aff cons[4]; int nc = 0;                            // formes qui doivent etre des unites mod l
    // n1 = A1 p q ; n2 = B1 (p+1)(q+1) - n1
    i64 c1 = A1 * p % M;
    Aff n = {mmod(B1 * ((p + 1) % M) % M - c1, M), B1 * ((p + 1) % M) % M};
    i64 Mc = M; bool ok = true, unk = false;
    for (int i = 1; i < 4 && ok; i++) {
        if (Mc <= pw[i]) { unk = true; break; }
        if (pw[i] > 1) {                                // a t + b = 0 (mod pw)  ->  t = t0 + s t'
            i64 a = mmod(n.a, pw[i]), b = mmod(n.b, pw[i]);
            i64 g = 1;
            while (g < pw[i] && a % (g * l) == 0) g *= l;   // g = pgcd(a, pw) (puissance de l)
            if (a == 0) g = pw[i];
            if (b % g) { ok = false; break; }
            i64 s = pw[i] / g;
            i64 t0 = (s == 1) ? 0 : mmod((-b / g) % s * invmod((a / g) % s, s), s);
            // substitution t = t0 + s t' dans toutes les formes
            n = {mmod(n.a * s, Mc), mmod(n.a * t0 + n.b, Mc)};
            q = {mmod(q.a * s, M), mmod(q.a * t0 + q.b, M)};
            for (int k = 0; k < nc; k++) cons[k] = {mmod(cons[k].a * s, l), mmod(cons[k].a * t0 + cons[k].b, l)};
        }
        i64 Mc2 = Mc / pw[i];
        Aff x = {mmod(n.a / pw[i], Mc2), mmod(n.b / pw[i], Mc2)};   // exact : pw divise a et b
        cons[nc++] = {x.a % l, x.b % l};
        i64 iv = invmod(us[i] % Mc2, Mc2);
        Aff P = {x.a * iv % Mc2, x.b * iv % Mc2};
        Mc = Mc2;
        i64 e1 = mmod(cs[i], Mc), e2 = mmod(cg[i], Mc);
        // n = SG s_i (P+1) - G u_i P
        n = {mmod(e1 * P.a % Mc - e2 * P.a % Mc, Mc), mmod(e1 * ((P.b + 1) % Mc) % Mc - e2 * P.b % Mc, Mc)};
    }
    if (unk) return true;
    if (!ok) return false;
    // fermeture : n = A1 p q (mod Mc)
    Aff n1 = {mmod(c1 * q.a, Mc), mmod(c1 * q.b, Mc)};
    i64 ca = mmod(n.a - n1.a, Mc), cb = mmod(n1.b - n.b, Mc);  // ca t = cb (mod Mc)
    i64 g = 1;
    while (g < Mc && ca % (g * l) == 0) g *= l;
    if (ca == 0) g = Mc;
    if (cb % g) return false;
    i64 s = Mc / g;
    cons[nc++] = {q.a % l, q.b % l};
    if (s == 1) {                                       // t libre : existe-t-il t mod l convenable ?
        for (i64 t = 0; t < l; t++) {
            bool bon = true;
            for (int k = 0; k < nc && bon; k++) if ((cons[k].a * t + cons[k].b) % l == 0) bon = false;
            if (bon) return true;
        }
    } else {                                            // t = t1 (mod s), s >= l : t mod l fixe
        i64 t1 = mmod((cb / g) % s * invmod((ca / g) % s, s), s) % l;
        bool bon = true;
        for (int k = 0; k < nc && bon; k++) if ((cons[k].a * t1 + cons[k].b) % l == 0) bon = false;
        if (bon) return true;
    }
    return false;
}

HD static inline bool local_ok(i64 l, const i64 *u, const i64 *su, i64 G, i64 SG, i64 fmax) {
    Prep R;
    if (!prep_local(R, l, u, su, G, SG, fmax)) return true;
    for (i64 p = 1; p < R.M; p++)
        if (p % l != 0 && test_p(R, p)) return true;
    return false;
}

HD static inline bool prefiltre(const Tables &T, int i1, int i2, int i3, int i4) {
    if (T.fmax <= 0) return true;
    i64 u[4] = {T.U[i1], T.U[i2], T.U[i3], T.U[i4]}, su[4] = {T.SU[i1], T.SU[i2], T.SU[i3], T.SU[i4]};
    const i64 L[4] = {3, 5, 7, 2};
    for (int k = 0; k < 4; k++)
        if (!local_ok(L[k], u, su, T.G, T.SG, T.fmax)) return false;
    return true;
}

// Traite le couple A d'indice t ; renvoie le nombre de quadruplets examines (nq), compte les "trop grands".
// emit(i1, i2, i3, i4, C) est appele pour chaque quadruplet garde.
template <class Emit>
HD static inline i64 un_couple(const Tables &T, int pos, i64 t, double lo_eps, i64 cmax, double LYmax, double nmin,
                               i64 &nbig, i64 &nfilt, bool filtrer, Emit emit) {
    int ib = T.ns + (int)(t / T.ns), isml = (int)(t % T.ns);
    int a0, a1;
    if (pos == 2 || pos == 4) { a0 = isml; a1 = ib; } else { a0 = ib; a1 = isml; }
    double sA = T.lf[a0] + T.lf[a1];
    int lo = lower_bound_d(T.Rs, T.nB, lo_eps - sA), hi = lower_bound_d(T.Rs, T.nB, -sA);
    i64 nq = 0;
    for (int r = lo; r < hi; r++) {
        int b0 = T.RI[r], b1 = T.RJ[r];
        nq++;
        int i1, i2, i3, i4;
        if (pos <= 2) { i1 = a0; i2 = a1; i3 = b0; i4 = b1; }
        else          { i1 = b0; i2 = b1; i3 = a0; i4 = a1; }
        double pf = T.fv[i1] * T.fv[i2] * T.fv[i3] * T.fv[i4];
        if (pf >= 1.0) continue;
        double rr = T.f1p[i1] * T.fv[i2] * T.fv[i3] * T.fv[i4] / (1.0 - pf);
        if (rr * rr * rr * T.GU[i1] < nmin) continue;
        if (T.ly[i1] + T.ly[i2] + T.ly[i3] + T.ly[i4] > LYmax) { nbig++; continue; }
        // C = prod Y - prod X, exact modulo 2^64 ; vrai C < 2^63 garanti par le test LYmax
        u64 C = (u64)T.Y[i1] * (u64)T.Y[i2] * (u64)T.Y[i3] * (u64)T.Y[i4]
              - (u64)T.X[i1] * (u64)T.X[i2] * (u64)T.X[i3] * (u64)T.X[i4];
        if ((i64)C <= 0) continue;
        u64 vs[5] = {(u64)T.Y[i1], (u64)T.Z[i1], (u64)T.X[i2], (u64)T.X[i3], (u64)T.X[i4]};
        for (int k = 0; k < 5; k++) {
            u64 g = gcd64(C, vs[k]);
            if (g > 1) C /= g;
            if ((i64)C <= cmax) break;
        }
        if ((i64)C <= cmax) {
            if (!filtrer || prefiltre(T, i1, i2, i3, i4)) emit(i1, i2, i3, i4, (i64)C);
            else nfilt++;
        }
    }
    return nq;
}

#ifdef __CUDACC__
// ------------------------------------------------------------------------------------------------ GPU
#include <cuda_runtime.h>
#include <cstdlib>

static Tables D;                          // pointeurs sur la carte
static void *dbuf[16];
static i64 *d_out = nullptr; static i64 d_cap = 0;
static u64 *d_cnt = nullptr;
struct Cand { int i[4]; i64 C; };
static Cand *d_c1 = nullptr; static i64 d_cap1 = 0;     // candidats de la phase 1 (avant pre-filtre)

// Phase 1 : appariement seul. Chaque fil traite des couples A ; les quadruplets de C <= cmax vont dans c1.
// Le pre-filtre n'est PAS fait ici : seul ~1 % des quadruplets y arrive, et un fil qui filtre ferait attendre
// les 31 autres de son warp.
__global__ void noyau_app(Tables T, int pos, i64 t0, i64 t1, double lo_eps, i64 cmax, double LYmax, double nmin,
                          Cand *c1, i64 cap1, u64 *cnt) {
    i64 nq = 0, nbig = 0, nfilt = 0;
    for (i64 t = t0 + blockIdx.x * (i64)blockDim.x + threadIdx.x; t < t1; t += (i64)gridDim.x * blockDim.x) {
        nq += un_couple(T, pos, t, lo_eps, cmax, LYmax, nmin, nbig, nfilt, false,
            [&](int i1, int i2, int i3, int i4, i64 C) {
                u64 k = atomicAdd(&cnt[0], 1ULL);
                if ((i64)k < cap1) { Cand &o = c1[k]; o.i[0] = i1; o.i[1] = i2; o.i[2] = i3; o.i[3] = i4; o.C = C; }
            });
    }
    atomicAdd(&cnt[1], (u64)nq);
    atomicAdd(&cnt[2], (u64)nbig);
}

// Phase 2 : pre-filtre local, un WARP par candidat. Les 32 fils se partagent les residus p (p = 1 + lane,
// 1 + lane + 32, ...) ; __any_sync arrete des qu'un fil trouve un p possible. Meme verdict que local_ok().
__global__ void noyau_filtre(Tables T, const Cand *c1, i64 n1, i64 *out, i64 cap, u64 *cnt) {
    const unsigned FULL = 0xffffffffu;
    int lane = threadIdx.x & 31;
    i64 w = (blockIdx.x * (i64)blockDim.x + threadIdx.x) >> 5, nw = ((i64)gridDim.x * blockDim.x) >> 5;
    const i64 L[4] = {3, 5, 7, 2};
    for (; w < n1; w += nw) {
        Cand c = c1[w];
        i64 u[4], su[4];
        for (int k = 0; k < 4; k++) { u[k] = T.U[c.i[k]]; su[k] = T.SU[c.i[k]]; }
        bool garde = true;
        for (int k = 0; k < 4 && garde; k++) {
            Prep R;
            if (T.fmax <= 0 || !prep_local(R, L[k], u, su, T.G, T.SG, T.fmax)) continue;   // non teste = accepte
            bool trouve = false;
            for (i64 base = 1; base < R.M; base += 32) {
                i64 p = base + lane;
                bool moi = p < R.M && p % R.l != 0 && test_p(R, p);
                if (__any_sync(FULL, moi)) { trouve = true; break; }
            }
            garde = trouve;
        }
        if (lane == 0) {
            if (garde) {
                u64 k = atomicAdd(&cnt[0], 1ULL);
                if ((i64)k < cap) {
                    i64 *o = out + 5 * k;
                    o[0] = c.i[0]; o[1] = c.i[1]; o[2] = c.i[2]; o[3] = c.i[3]; o[4] = c.C;
                }
            } else {
                atomicAdd(&cnt[3], 1ULL);
            }
        }
    }
}

template <class T> static T *copie(const T *h, size_t n, int k) {
    void *d = nullptr;
    if (cudaMalloc(&d, n * sizeof(T) + 8) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice);
    dbuf[k] = d;
    return (T *)d;
}

extern "C" void asym_free() {
    for (int k = 0; k < 16; k++) if (dbuf[k]) { cudaFree(dbuf[k]); dbuf[k] = nullptr; }
    if (d_out) { cudaFree(d_out); d_out = nullptr; d_cap = 0; }
    if (d_cnt) { cudaFree(d_cnt); d_cnt = nullptr; }
    if (d_c1) { cudaFree(d_c1); d_c1 = nullptr; d_cap1 = 0; }
}

extern "C" int asym_init(int ns, int nb, int nB, const i64 *X, const i64 *Y, const i64 *Z, const double *lf,
                         const double *fv, const double *ly, const double *f1p, const double *GU,
                         const double *Rs, const int *RI, const int *RJ, const i64 *U, const i64 *SU,
                         i64 G, i64 SG, i64 fmax) {
    asym_free();
    size_t n = (size_t)ns + nb;
    D.ns = ns; D.nb = nb; D.nB = nB;
    D.X = copie(X, n, 0); D.Y = copie(Y, n, 1); D.Z = copie(Z, n, 2);
    D.lf = copie(lf, n, 3); D.fv = copie(fv, n, 4); D.ly = copie(ly, n, 5); D.f1p = copie(f1p, n, 6);
    D.GU = copie(GU, n, 7);
    D.Rs = copie(Rs, (size_t)nB, 8); D.RI = copie(RI, (size_t)nB, 9); D.RJ = copie(RJ, (size_t)nB, 10);
    D.U = copie(U, n, 11); D.SU = copie(SU, n, 12); D.G = G; D.SG = SG; D.fmax = fmax;
    if (cudaMalloc(&d_cnt, 4 * sizeof(u64)) != cudaSuccess) return -1;
    for (int k = 0; k <= 12; k++) if (!dbuf[k]) return -1;
    return 0;
}

extern "C" i64 asym_run(int pos, i64 t0, i64 t1, double lo_eps, i64 cmax, double LYmax, double nmin,
                        i64 *out, i64 cap, i64 *stats) {
    if (cap > d_cap) {
        if (d_out) cudaFree(d_out);
        if (cudaMalloc(&d_out, (size_t)cap * 5 * sizeof(i64)) != cudaSuccess) { d_out = nullptr; d_cap = 0; return -1; }
        d_cap = cap;
    }
    // tampon de la phase 1 : ASYM_CAP1 candidats (defaut 3e7, 24 octets chacun = 720 Mo), au moins 4 * cap
    i64 cap1 = 30000000;
    if (const char *e = getenv("ASYM_CAP1")) cap1 = atoll(e);
    if (cap1 < 4 * cap) cap1 = 4 * cap;
    if (cap1 > d_cap1) {
        if (d_c1) cudaFree(d_c1);
        if (cudaMalloc(&d_c1, (size_t)cap1 * sizeof(Cand)) != cudaSuccess) { d_c1 = nullptr; d_cap1 = 0; return -1; }
        d_cap1 = cap1;
    }
    cudaMemset(d_cnt, 0, 4 * sizeof(u64));
    int bloc = 256;
    i64 nth = t1 - t0;
    int grille = (int)((nth + bloc - 1) / bloc);
    if (grille > 65535 * 8) grille = 65535 * 8;
    if (grille < 1) grille = 1;
    noyau_app<<<grille, bloc>>>(D, pos, t0, t1, lo_eps, cmax, LYmax, nmin, d_c1, cap1, d_cnt);
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    u64 c[4];
    cudaMemcpy(c, d_cnt, sizeof(c), cudaMemcpyDeviceToHost);
    stats[0] = (i64)c[1]; stats[1] = (i64)c[2]; stats[2] = 0;
    i64 n1 = (i64)c[0];
    if (n1 > cap1) return cap + 1;                    // tampon de phase 1 plein : l'appelant coupe la tranche
    u64 zero = 0;
    cudaMemcpy(&d_cnt[0], &zero, sizeof(u64), cudaMemcpyHostToDevice);
    if (n1 > 0) {
        i64 g2 = (n1 * 32 + bloc - 1) / bloc;
        if (g2 > 65535 * 8) g2 = 65535 * 8;
        noyau_filtre<<<(int)g2, bloc>>>(D, d_c1, n1, d_out, cap, d_cnt);
        if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    }
    cudaMemcpy(c, d_cnt, sizeof(c), cudaMemcpyDeviceToHost);
    stats[2] = (i64)c[3];
    i64 k = (i64)c[0] < cap ? (i64)c[0] : cap;
    if (k > 0) cudaMemcpy(out, d_out, (size_t)k * 5 * sizeof(i64), cudaMemcpyDeviceToHost);
    return (i64)c[0];
}

#else
// ------------------------------------------------------------------------------------------------ CPU (OpenMP)
#include <omp.h>

static Tables H;

extern "C" void asym_free() {}

extern "C" int asym_init(int ns, int nb, int nB, const i64 *X, const i64 *Y, const i64 *Z, const double *lf,
                         const double *fv, const double *ly, const double *f1p, const double *GU,
                         const double *Rs, const int *RI, const int *RJ, const i64 *U, const i64 *SU,
                         i64 G, i64 SG, i64 fmax) {
    // les tableaux restent ceux de numpy : l'appelant les garde vivants jusqu'a asym_free
    H.ns = ns; H.nb = nb; H.nB = nB; H.X = X; H.Y = Y; H.Z = Z; H.lf = lf; H.fv = fv; H.ly = ly; H.f1p = f1p;
    H.GU = GU; H.Rs = Rs; H.RI = RI; H.RJ = RJ; H.U = U; H.SU = SU; H.G = G; H.SG = SG; H.fmax = fmax;
    return 0;
}

extern "C" i64 asym_run(int pos, i64 t0, i64 t1, double lo_eps, i64 cmax, double LYmax, double nmin,
                        i64 *out, i64 cap, i64 *stats) {
    i64 cnt = 0, NQ = 0, NBIG = 0, NFILT = 0;
    #pragma omp parallel for schedule(dynamic, 256) reduction(+:NQ, NBIG, NFILT)
    for (i64 t = t0; t < t1; t++) {
        i64 nbig = 0, nfilt = 0;
        NQ += un_couple(H, pos, t, lo_eps, cmax, LYmax, nmin, nbig, nfilt, true,
            [&](int i1, int i2, int i3, int i4, i64 C) {
                i64 k;
                #pragma omp atomic capture
                k = cnt++;
                if (k < cap) {
                    i64 *o = out + 5 * k;
                    o[0] = i1; o[1] = i2; o[2] = i3; o[3] = i4; o[4] = C;
                }
            });
        NBIG += nbig; NFILT += nfilt;
    }
    stats[0] = NQ; stats[1] = NBIG; stats[2] = NFILT;
    return cnt;
}
#endif
