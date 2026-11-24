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

// Pre-filtre local (meme logique que rho() de resid.py, mais sans C, D, E) : existe-t-il des residus unites
// p, q modulo M = l^K tels que la chaine n1 = G u1 p q -> G u2 P2 -> G u3 P3 -> G u4 P4 -> n1 soit possible
// modulo les puissances de l ? Si non, le quadruplet n'a aucune solution. Perte de precision = accepte.
// Plus permissif que rho() (q n'est pas impose par l'equation bilineaire) : il ne rejette jamais un vrai cycle.
HD static inline bool local_ok(i64 l, const i64 *u, const i64 *su, i64 G, i64 SG, i64 fmax) {
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
                               i64 &nbig, i64 &nfilt, Emit emit) {
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
            if (prefiltre(T, i1, i2, i3, i4)) emit(i1, i2, i3, i4, (i64)C);
            else nfilt++;
        }
    }
    return nq;
}

#ifdef __CUDACC__
// ------------------------------------------------------------------------------------------------ GPU
#include <cuda_runtime.h>

static Tables D;                          // pointeurs sur la carte
static void *dbuf[16];
static i64 *d_out = nullptr; static i64 d_cap = 0;
static u64 *d_cnt = nullptr;

__global__ void noyau(Tables T, int pos, i64 t0, i64 t1, double lo_eps, i64 cmax, double LYmax, double nmin,
                      i64 *out, i64 cap, u64 *cnt) {
    i64 nq = 0, nbig = 0, nfilt = 0;
    for (i64 t = t0 + blockIdx.x * (i64)blockDim.x + threadIdx.x; t < t1; t += (i64)gridDim.x * blockDim.x) {
        nq += un_couple(T, pos, t, lo_eps, cmax, LYmax, nmin, nbig, nfilt,
            [&](int i1, int i2, int i3, int i4, i64 C) {
                u64 k = atomicAdd(&cnt[0], 1ULL);
                if ((i64)k < cap) {
                    i64 *o = out + 5 * k;
                    o[0] = i1; o[1] = i2; o[2] = i3; o[3] = i4; o[4] = C;
                }
            });
    }
    atomicAdd(&cnt[1], (u64)nq);
    atomicAdd(&cnt[2], (u64)nbig);
    atomicAdd(&cnt[3], (u64)nfilt);
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
    cudaMemset(d_cnt, 0, 4 * sizeof(u64));
    int bloc = 256;
    i64 nth = t1 - t0;
    int grille = (int)((nth + bloc - 1) / bloc);
    if (grille > 65535 * 8) grille = 65535 * 8;
    if (grille < 1) grille = 1;
    noyau<<<grille, bloc>>>(D, pos, t0, t1, lo_eps, cmax, LYmax, nmin, d_out, cap, d_cnt);
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    u64 c[4];
    cudaMemcpy(c, d_cnt, sizeof(c), cudaMemcpyDeviceToHost);
    i64 k = (i64)c[0] < cap ? (i64)c[0] : cap;
    if (k > 0) cudaMemcpy(out, d_out, (size_t)k * 5 * sizeof(i64), cudaMemcpyDeviceToHost);
    stats[0] = (i64)c[1]; stats[1] = (i64)c[2]; stats[2] = (i64)c[3];
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
        NQ += un_couple(H, pos, t, lo_eps, cmax, LYmax, nmin, nbig, nfilt,
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
