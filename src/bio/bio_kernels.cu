// bio_kernels.cu - extern "C" __global__ wrappers over bio_core.h, compiled at
// run time by NVRTC as the concatenation bio_types.h + bio_core.h + this file.
// No includes. Integer only. The world struct is a __grid_constant__ kernel
// parameter, so &w points at the parameter itself (no local copy).

#define BIO_TID ((int32_t)(blockIdx.x * blockDim.x + threadIdx.x))
#define BIO_KW const __grid_constant__ BioWorld w

static __device__ __forceinline__ long long bio_warp_sum(long long v) {
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
    return v;
}
/* accumulator indices written by k_env and by k_metab */
static __device__ __forceinline__ int bio_env_acc(int k) {
    return k == BACC_SED_C || k == BACC_SED_S || k == BACC_SED_FE || (k >= BACC_VENT_C && k <= BACC_WEATH_P) ||
           (k >= BACC_ATMD && k < BACC_ATMD + BIO_ATMG);
}
static __device__ __forceinline__ int bio_metab_acc(int k) {
    return k < BACC_PATH + BP_N || k == BACC_FIX || k == BACC_ASSIM || k == BACC_RESP || k == BACC_DEATH_C || k == BACC_SED_FE ||
           k == BACC_WANT;
}

extern "C" __global__ void k_globals(BIO_KW) {
    if (blockIdx.x == 0 && threadIdx.x == 0) bio_globals_update(&w);
}

// cells: diffusion + chemistry; block sums of the ledgers go to G->acc
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_env(BIO_KW) {
    __shared__ long long sacc[BACC_N];
    for (int k = threadIdx.x; k < BACC_N; k += blockDim.x) sacc[k] = 0;
    __syncthreads();
    int64_t loc[BACC_N];
    BIO_UNROLL
    for (int k = 0; k < BACC_N; ++k) loc[k] = 0;
    const int32_t c = BIO_TID;
    if (c < w.n_cells) bio_cell_env(&w, c, loc);
    BIO_UNROLL
    for (int k = 0; k < BACC_N; ++k) {
        if (!bio_env_acc(k)) continue;
        const long long v = bio_warp_sum(loc[k]);
        if ((threadIdx.x & 31) == 0 && v) atomicAdd((unsigned long long*)&sacc[k], (unsigned long long)v);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < BACC_N; k += blockDim.x)
        if (sacc[k]) atomicAdd((unsigned long long*)&w.G->acc[k], (unsigned long long)sacc[k]);
}

extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_summary(BIO_KW) {
    const int32_t i = BIO_TID;
    if (i < (int32_t)w.n_slots) bio_agent_summary(&w, i);
}

// list of agents due to think (order does not matter: each touches only itself)
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_think_list(BIO_KW) {
    __shared__ uint32_t wcnt[BIO_BLOCK / 32];
    __shared__ uint32_t base;
    const int32_t i = BIO_TID;
    const int due = i < (int32_t)w.n_slots && bio_think_due(&w, i);
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t bal = __ballot_sync(0xFFFFFFFFu, due);
    if (lane == 0) wcnt[warp] = __popc(bal);
    __syncthreads();
    if (threadIdx.x == 0) {     /* one global atomic per block */
        uint32_t t = 0;
        for (int k = 0; k < BIO_BLOCK / 32; ++k) { const uint32_t c = wcnt[k]; wcnt[k] = t; t += c; }
        base = t ? atomicAdd(&w.G->think_n, t) : 0u;
    }
    __syncthreads();
    if (due) w.pre_a[base + wcnt[warp] + __popc(bal & ((1u << lane) - 1u))] = (uint32_t)i;
}
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_think(BIO_KW) {
    const uint32_t n = w.G->think_n;
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x)
        bio_think_run(&w, (int32_t)w.pre_a[k]);
}

extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_pre(BIO_KW) {
    const int32_t i = BIO_TID;
    if (i < (int32_t)w.n_slots) bio_agent_pre(&w, i);
}

extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_supply(BIO_KW) {
    const int32_t c = BIO_TID;
    if (c < w.n_cells) bio_cell_supply(&w, c);
}

extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_metab(BIO_KW) {
    __shared__ long long sacc[BACC_N];
    for (int k = threadIdx.x; k < BACC_N; k += blockDim.x) sacc[k] = 0;
    __syncthreads();
    int64_t loc[BACC_N];
    BIO_UNROLL
    for (int k = 0; k < BACC_N; ++k) loc[k] = 0;
    const int32_t i = BIO_TID;
    if (i < (int32_t)w.n_slots) bio_agent_metab(&w, i, loc);
    BIO_UNROLL
    for (int k = 0; k < BACC_N; ++k) {
        if (!bio_metab_acc(k)) continue;
        const long long v = bio_warp_sum(loc[k]);
        if ((threadIdx.x & 31) == 0 && v) atomicAdd((unsigned long long*)&sacc[k], (unsigned long long)v);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < BACC_N; k += blockDim.x)
        if (sacc[k]) atomicAdd((unsigned long long*)&w.G->acc[k], (unsigned long long)sacc[k]);
}

// block-local exclusive scan of 1024 items. mode 0: births | deaths << 16
// (pre_bd), mode 1: alive (pre_a). Block totals go to boff[blk * 3 + lane].
extern "C" __global__ void __launch_bounds__(BIO_SCAN_BLOCK) k_scan_a(BIO_KW, int mode) {
    __shared__ uint32_t wsum[32];
    const int32_t i = BIO_TID;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t v = 0;
    if (i < (int32_t)w.n_slots) v = mode == 0 ? bio_scan_bits(w.flags[i]) : ((w.flags[i] & BA_ALIVE) ? 1u : 0u);
    uint32_t x = v;
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31) wsum[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint32_t s = wsum[lane];
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, s, d);
            if (lane >= d) s += y;
        }
        wsum[lane] = s;
    }
    __syncthreads();
    const uint32_t incl = x + (warp > 0 ? wsum[warp - 1] : 0u);
    const uint32_t excl = incl - v;
    if (i < (int32_t)w.n_slots) {
        if (mode == 0) w.pre_bd[i] = excl; else w.pre_a[i] = excl;
    }
    if (threadIdx.x == blockDim.x - 1) {
        const uint32_t blk = blockIdx.x;
        if (mode == 0) { w.boff[blk * 3 + 0] = incl & 0xFFFFu; w.boff[blk * 3 + 1] = incl >> 16; }
        else w.boff[blk * 3 + 2] = incl;
    }
}

// one block: exclusive scan of the block totals; grand totals at boff[nb * 3 + lane]
extern "C" __global__ void __launch_bounds__(BIO_SCAN_BLOCK) k_scan_b(BIO_KW, int mode, int nb) {
    __shared__ uint32_t wsum[32];
    __shared__ uint32_t carry[2];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int l0 = mode == 0 ? 0 : 2, l1 = mode == 0 ? 1 : 2;
    for (int ln = l0; ln <= l1; ++ln) {
        if (threadIdx.x == 0) carry[0] = 0;
        __syncthreads();
        for (int base = 0; base < nb; base += BIO_SCAN_BLOCK) {
            const int b = base + (int)threadIdx.x;
            const uint32_t v = b < nb ? w.boff[b * 3 + ln] : 0u;
            uint32_t x = v;
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, d);
                if (lane >= d) x += y;
            }
            if (lane == 31) wsum[warp] = x;
            __syncthreads();
            if (warp == 0) {
                uint32_t s = wsum[lane];
                for (int d = 1; d < 32; d <<= 1) {
                    const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, s, d);
                    if (lane >= d) s += y;
                }
                wsum[lane] = s;
            }
            __syncthreads();
            const uint32_t incl = x + (warp > 0 ? wsum[warp - 1] : 0u) + carry[0];
            if (b < nb) w.boff[b * 3 + ln] = incl - v;
            __syncthreads();
            if (threadIdx.x == blockDim.x - 1) carry[0] = incl;
            __syncthreads();
        }
        if (threadIdx.x == 0) w.boff[nb * 3 + ln] = carry[0];
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_birth(BIO_KW) {
    const int32_t i = BIO_TID;
    if (i < (int32_t)w.n_slots) bio_agent_birth(&w, i);
}

extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_free(BIO_KW) {
    const int32_t i = BIO_TID;
    if (i < (int32_t)w.n_slots) bio_agent_free(&w, i);
}

// compaction: move alive agents to their prefix position in the other buffer
extern "C" __global__ void __launch_bounds__(BIO_BLOCK)
k_gather32(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, const uint32_t* __restrict__ flags,
           const uint32_t* __restrict__ pre_a, const uint32_t* __restrict__ boff, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !(flags[i] & BA_ALIVE)) return;
    dst[pre_a[i] + boff[(i / BIO_SCAN_BLOCK) * 3 + 2]] = src[i];
}
extern "C" __global__ void __launch_bounds__(BIO_BLOCK)
k_gather64(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst, const uint32_t* __restrict__ flags,
           const uint32_t* __restrict__ pre_a, const uint32_t* __restrict__ boff, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !(flags[i] & BA_ALIVE)) return;
    dst[pre_a[i] + boff[(i / BIO_SCAN_BLOCK) * 3 + 2]] = src[i];
}

// statistics: block sums in shared memory, then one atomic per value into part[0..]
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_stats(BIO_KW, int cells) {
    __shared__ long long st[BIO_STAT_N];
    for (int k = threadIdx.x; k < BIO_STAT_N; k += blockDim.x) st[k] = 0;
    __syncthreads();
    const int32_t i = BIO_TID;
    if (cells) {
        int64_t loc[BS_CELL_N];
        BIO_UNROLL
        for (int k = 0; k < BS_CELL_N; ++k) loc[k] = 0;
        if (i < w.n_cells) bio_cell_stats(&w, i, loc);
        BIO_UNROLL
        for (int k = 0; k < BS_CELL_N; ++k) {
            const long long v = bio_warp_sum(loc[k]);
            if ((threadIdx.x & 31) == 0 && v) atomicAdd((unsigned long long*)&st[BS_FLD + k], (unsigned long long)v);
        }
    } else {
        int64_t loc[BS_AGENT_N];
        BIO_UNROLL
        for (int k = 0; k < BS_AGENT_N; ++k) loc[k] = 0;
        const int ok = i < (int32_t)w.n_slots;
        if (ok) bio_agent_stats(&w, i, loc);
        BIO_UNROLL
        for (int k = 0; k < BS_AGENT_N; ++k) {
            const long long v = bio_warp_sum(loc[k]);
            if ((threadIdx.x & 31) == 0 && v) atomicAdd((unsigned long long*)&st[k], (unsigned long long)v);
        }
        /* gene sums: the 64 body bytes loaded once as words; two genes per 32-bit
         * warp sum (32 x 255 fits in 16 bits) */
        const int alive = ok && (w.flags[i] & BA_ALIVE);
        uint32_t gw[16];
        if (alive) {
            const uint32_t* p = (const uint32_t*)w.gen[w.gslot[i]].body;
            BIO_UNROLL
            for (int k = 0; k < 16; ++k) gw[k] = p[k];
        } else {
            BIO_UNROLL
            for (int k = 0; k < 16; ++k) gw[k] = 0u;
        }
        BIO_UNROLL
        for (int k = 0; k < 32; ++k) {
            const uint32_t wv = gw[k >> 1] >> ((k & 1) * 16);
            uint32_t v = (wv & 0xFFu) | (((wv >> 8) & 0xFFu) << 16);
            for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
            if ((threadIdx.x & 31) == 0) {
                if (v & 0xFFFFu) atomicAdd((unsigned long long*)&st[BS_GENE + 2 * k], (unsigned long long)(v & 0xFFFFu));
                if (v >> 16) atomicAdd((unsigned long long*)&st[BS_GENE + 2 * k + 1], (unsigned long long)(v >> 16));
            }
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < BIO_STAT_N; k += blockDim.x)
        if (st[k]) atomicAdd((unsigned long long*)&w.part[k], (unsigned long long)st[k]);
}

// ---- sort by cell (stable): count, scan over cells, scatter, per-cell order fix ----
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_sort_count(BIO_KW) {
    const int32_t i = BIO_TID;
    if (i < (int32_t)w.n_slots && (w.flags[i] & BA_ALIVE)) atomicAdd(&w.cnt[w.cell[i]], 1u);
}
// exclusive scan of cnt[] into lai[0..n) (offsets); block sums in cscan[]
extern "C" __global__ void __launch_bounds__(BIO_SCAN_BLOCK) k_cscan_a(BIO_KW) {
    __shared__ uint32_t wsum[32];
    const int32_t c = BIO_TID;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t v = c < w.n_cells ? w.cnt[c] : 0u;
    uint32_t x = v;
    for (int d = 1; d < 32; d <<= 1) { const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, d); if (lane >= d) x += y; }
    if (lane == 31) wsum[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint32_t s = wsum[lane];
        for (int d = 1; d < 32; d <<= 1) { const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, s, d); if (lane >= d) s += y; }
        wsum[lane] = s;
    }
    __syncthreads();
    const uint32_t incl = x + (warp > 0 ? wsum[warp - 1] : 0u);
    if (c < w.n_cells) w.lai[c] = incl - v;
    if (threadIdx.x == blockDim.x - 1) w.cscan[blockIdx.x] = incl;
}
extern "C" __global__ void __launch_bounds__(BIO_SCAN_BLOCK) k_cscan_b(BIO_KW, int nb) {
    __shared__ uint32_t wsum[32];
    __shared__ uint32_t carry;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < nb; base += BIO_SCAN_BLOCK) {
        const int b = base + (int)threadIdx.x;
        const uint32_t v = b < nb ? w.cscan[b] : 0u;
        uint32_t x = v;
        for (int d = 1; d < 32; d <<= 1) { const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, d); if (lane >= d) x += y; }
        if (lane == 31) wsum[warp] = x;
        __syncthreads();
        if (warp == 0) {
            uint32_t s = wsum[lane];
            for (int d = 1; d < 32; d <<= 1) { const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, s, d); if (lane >= d) s += y; }
            wsum[lane] = s;
        }
        __syncthreads();
        const uint32_t incl = x + (warp > 0 ? wsum[warp - 1] : 0u) + carry;
        if (b < nb) w.cscan[b] = incl - v;
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) carry = incl;
        __syncthreads();
    }
}
// add block offsets; zero the per-cell cursors (lai[n..2n))
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_cscan_c(BIO_KW) {
    const int32_t c = BIO_TID;
    if (c >= w.n_cells) return;
    w.lai[c] += w.cscan[c / BIO_SCAN_BLOCK];
    w.lai[w.n_cells + c] = 0u;
}
// perm (in req) gets agent indices, grouped by cell in any order
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_sort_scatter(BIO_KW) {
    const int32_t i = BIO_TID;
    if (i >= (int32_t)w.n_slots || !(w.flags[i] & BA_ALIVE)) return;
    const uint32_t c = w.cell[i];
    const uint32_t r = atomicAdd(&w.lai[w.n_cells + c], 1u);
    ((uint32_t*)w.req)[w.lai[c] + r] = (uint32_t)i;
}
// each cell's slice sorted by agent index -> the stable order, as on the CPU
extern "C" __global__ void __launch_bounds__(BIO_BLOCK) k_sort_fix(BIO_KW) {
    const int32_t c = BIO_TID;
    if (c >= w.n_cells) return;
    const uint32_t n = w.cnt[c];
    if (n < 2) return;
    uint32_t* p = (uint32_t*)w.req + w.lai[c];
    for (uint32_t a = 1; a < n; ++a) {
        const uint32_t v = p[a];
        uint32_t b = a;
        while (b > 0 && p[b - 1] > v) { p[b] = p[b - 1]; --b; }
        p[b] = v;
    }
}
extern "C" __global__ void __launch_bounds__(BIO_BLOCK)
k_permute32(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, const uint32_t* __restrict__ perm, uint32_t n) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) dst[j] = src[perm[j]];
}
extern "C" __global__ void __launch_bounds__(BIO_BLOCK)
k_permute64(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst, const uint32_t* __restrict__ perm, uint32_t n) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) dst[j] = src[perm[j]];
}
