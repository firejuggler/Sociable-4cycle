// Fonctions du noyau (apparie_gpu.cu, version CPU) exposees pour tests_croises.py
#include "../apparie_gpu.cu"
#include <vector>
static u128 mk(uint64_t h, uint64_t l) { return ((u128)h << 64) | l; }
extern "C" {
void t_redc128(uint64_t T0, uint64_t T1, uint64_t T2, uint64_t nh, uint64_t nl, uint64_t *rh, uint64_t *rl) {
    u128 n = mk(nh, nl); u128 r = redc128(T0, T1, T2, n, ninv64((uint64_t)n)); *rh = (uint64_t)(r >> 64); *rl = (uint64_t)r;
}
void t_mr128(uint64_t ah, uint64_t al, uint64_t b, uint64_t nh, uint64_t nl, uint64_t *rh, uint64_t *rl) {
    u128 n = mk(nh, nl); u128 r = mr128(mk(ah, al), b, n, ninv64((uint64_t)n)); *rh = (uint64_t)(r >> 64); *rl = (uint64_t)r;
}
uint64_t t_redc(uint64_t Th, uint64_t Tl, uint64_t n) { return redc(mk(Th, Tl), n, ninv64(n)); }
int t_div_petit(uint64_t nh, uint64_t nl, uint64_t ah, uint64_t al, uint64_t *q, uint64_t *rh, uint64_t *rl) {
    u128 r; int ok = div_petit(mk(nh, nl), mk(ah, al), q, &r); *rh = (uint64_t)(r >> 64); *rl = (uint64_t)r; return ok;
}
int t_div_petit64(uint64_t n, uint64_t a, uint64_t *q, uint64_t *r) { return div_petit64(n, a, q, r); }
void t_mul64x32(uint64_t a, uint32_t b, uint64_t *rh, uint64_t *rl) { u128 r = mul64x32(a, b); *rh = (uint64_t)(r >> 64); *rl = (uint64_t)r; }
int t_premier64(uint64_t n) { return premier64(n); }
// un couple complet (init_couple + pas_couple) : p emis (bit 63 = suite 128 bits ; 0 = rendu au C)
int t_couple(int64_t G, int64_t S, int64_t u0, int64_t s0, int64_t u2, int64_t s2, uint64_t P0, uint64_t maxp,
             uint64_t Emax, uint64_t Evh, uint64_t Evl, uint64_t *out, int cap, uint64_t *ntest) {
    u128 Evrai = mk(Evh, Evl);
    i128 A = (i128)G * u2 - (i128)S * s2, B = (i128)G * s2;
    int n = 0; uint64_t nt = 0, n2 = 0;
    auto emit = [&](uint64_t p) { if (n < cap) out[n] = p; n++; };
    Etat e;
    init_couple(u0, s0, u2, (uint64_t)G, (uint64_t)s2, P0, Emax, Evrai, A, B, e, emit);
    while (e.mode) { nt++; pas_couple(e, (uint64_t)G, (uint64_t)u0, (uint64_t)s0, (uint64_t)u2, (uint64_t)s2, maxp, Emax, Evrai, &n2, emit); }
    *ntest = nt;
    return n;
}
}
