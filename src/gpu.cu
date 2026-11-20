// gpu.cu - CUDA backend.
//
// Per batch, on its own stream:
//   0. H2D tile job list + k_col_tiles : smooth-surface tiles (density.h), one
//                   block per chunk column, written straight into their record
//   1. H2D job list (pinned)
//   2. k_generate : one thread per voxel column, writes the batch buffer and
//                   reduces per-chunk min/max material + solid count
//   3. k_commit   : non-uniform chunks are copied into their device pool slot
//                   AND streamed straight into mapped pinned host memory, so
//                   only non-empty data ever crosses PCIe
//   4. D2H meta, record event
// Two batches alternate, so generation of one overlaps transfer of the other.
#include "gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLS_PER_BLOCK   256
#define BLOCKS_PER_CHUNK ((GME_CHUNK_DIM * GME_CHUNK_DIM) / COLS_PER_BLOCK)
#define CHUNK_U4         (GME_CHUNK_VOXELS / 16)
#define TILE_THREADS     256
#define TILE_U4          (sizeof(GmeColTile) / 16)

static char g_err[512];

static int ck(cudaError_t e, const char* what, int line) {
    if (e == cudaSuccess) return 0;
    snprintf(g_err, sizeof g_err, "gpu.cu:%d %s: %s", line, what, cudaGetErrorString(e));
    return 1;
}
#define CK(x) if (ck((x), #x, __LINE__)) goto fail

extern "C" const char* gpu_last_error(void) { return g_err; }

struct Buf {
    cudaStream_t  stream;
    cudaEvent_t   done;
    GmeGenJob*    d_jobs;
    GmeChunkMeta* d_meta;
    uint8_t*      d_vox;     // batch scratch
    GmeGenJob*    h_jobs;    // pinned
    GmeChunkMeta* h_meta;    // pinned
    uint8_t*      h_vox;     // pinned + mapped (GPU writes it directly)
    uint8_t*      h_vox_dev; // device alias of h_vox
    GmeTileJob*   d_tjobs;
    GmeTileJob*   h_tjobs;   // pinned (cached: the world reads them back in finish_batch)
};

struct GpuWorld {
    uint32_t slots, max_batch;
    uint8_t* d_pool;
    Buf      b[GPU_BUFFERS];
    size_t   device_bytes, pinned_bytes;
    uint32_t* d_bio;         // biomap cells (vegetation), NULL = none
    size_t    bio_cells;
    GmeColTile* d_tiles;     // column tiles, tile_count records (toroidal page_n^2 table)
    uint32_t    tile_count;
    cudaEvent_t fence;       // M4: the renderer's frame in flight (gpu_world_set_fence), NULL = none
};

// ---------------------------------------------------------------------------
__global__ void k_meta_init(GmeChunkMeta* m, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { m[i].mat_min = 0xFFFFFFFFu; m[i].mat_max = 0; m[i].non_air = 0; m[i].pad = 0; }
}

// One column of one chunk. kLife = false is plain geology (gme_material).
template <bool kLife>
__device__ __forceinline__ void gen_column(int32_t wx, int32_t wy0, int32_t wz, int32_t h, uint32_t seed,
                                           const GmePlanet& planet, const GmeColumn& fc, uint8_t* dst,
                                           uint32_t& mn, uint32_t& mx, uint32_t& solid) {
    // voxel index = col + y*1024 -> each warp writes 32 contiguous bytes per layer
    #pragma unroll 4
    for (int ly = 0; ly < GME_CHUNK_DIM; ++ly) {
        const uint32_t m = kLife ? gme_voxel(wx, wy0 + ly, wz, &fc, seed, &planet)
                                 : gme_material(wx, wy0 + ly, wz, h, seed, &planet);
        dst[ly * (GME_CHUNK_DIM * GME_CHUNK_DIM)] = (uint8_t)m;
        mn = min(mn, m);
        mx = max(mx, m);
        solid += (m != MAT_AIR);
    }
}

// bio: vegetation view, by value, with .cells pointing at device memory
__global__ void __launch_bounds__(COLS_PER_BLOCK)
k_generate(const GmeGenJob* __restrict__ jobs, uint8_t* __restrict__ out,
           GmeChunkMeta* __restrict__ meta, uint32_t seed, GmePlanet planet, GmeBioView bio) {
    const uint32_t chunk = blockIdx.x / BLOCKS_PER_CHUNK;
    const uint32_t col   = (blockIdx.x % BLOCKS_PER_CHUNK) * COLS_PER_BLOCK + threadIdx.x; // 0..1023
    const GmeGenJob j = jobs[chunk];
    const int32_t wx  = j.cx * GME_CHUNK_DIM + (int32_t)(col & 31u);
    const int32_t wz  = j.cz * GME_CHUNK_DIM + (int32_t)(col >> 5);
    const int32_t wy0 = j.cy * GME_CHUNK_DIM;
    const int32_t h   = gme_height(wx, wz, seed, &planet);

    uint8_t* dst = out + (size_t)chunk * GME_CHUNK_VOXELS + col;
    uint32_t mn = 0xFFu, mx = 0u, solid = 0u;
    GmeColumn fc;
    if (bio.cells || bio.fallback_on) {   // same for the whole launch: no divergence
        gme_column_span(wx, wz, h, wy0, wy0 + GME_CHUNK_DIM - 1, seed, &planet, &bio, &fc);
        gen_column<true>(wx, wy0, wz, h, seed, planet, fc, dst, mn, mx, solid);
    } else {
        gme_column_bare(h, &fc);
        gen_column<false>(wx, wy0, wz, h, seed, planet, fc, dst, mn, mx, solid);
    }
    for (int o = 16; o > 0; o >>= 1) {
        mn = min(mn, __shfl_xor_sync(0xFFFFFFFFu, mn, o));
        mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
        solid += __shfl_xor_sync(0xFFFFFFFFu, solid, o);
    }
    if ((threadIdx.x & 31u) == 0) {
        atomicMin(&meta[chunk].mat_min, mn);
        atomicMax(&meta[chunk].mat_max, mx);
        atomicAdd(&meta[chunk].non_air, solid);
    }
}

__global__ void __launch_bounds__(256)
k_commit(const GmeGenJob* __restrict__ jobs, const GmeChunkMeta* __restrict__ meta,
         const uint4* __restrict__ src, uint4* __restrict__ pool, uint4* __restrict__ host) {
    const uint32_t c = blockIdx.x;
    if (meta[c].mat_min == meta[c].mat_max) return; // uniform: no storage, no transfer
    const uint4* s = src  + (size_t)c * CHUNK_U4;
    uint4*       d = pool + (size_t)jobs[c].slot * CHUNK_U4;
    uint4*       h = host + (size_t)c * CHUNK_U4;
    for (uint32_t i = threadIdx.x; i < CHUNK_U4; i += blockDim.x) {
        const uint4 v = s[i];
        d[i] = v;
        h[i] = v;
    }
}

// One block per tile job: the 36 x 36 columns (h, dev) in parallel, then the
// same steps as gme_col_tile_finish (density.h) split over the threads: h range
// by warp shuffles, packing by every thread, the 16 blocks (slot + bounds) by 16
// threads and the 81 lod4 lattice points (M3) by 81 others, the header and the
// lod bounds by one. All integer, so the bytes equal the CPU's
// gme_col_tile_build. direct != 0 writes tile k to out[k] instead of out[index].
__global__ void __launch_bounds__(TILE_THREADS)
k_col_tiles(const GmeTileJob* __restrict__ jobs, GmeColTile* __restrict__ out, int direct, uint32_t seed,
            GmePlanet planet, GmeBioView bio) {
    __shared__ int32_t s_h[GME_TILE_COLS];
    __shared__ int32_t s_dev[GME_TILE_COLS];
    __shared__ __align__(16) GmeColTile s_t;
    __shared__ int32_t s_mn[TILE_THREADS / 32], s_mx[TILE_THREADS / 32];
    __shared__ uint32_t s_fl[16];
    __shared__ int s_sat;
    const GmeTileJob j = jobs[blockIdx.x];
    const int32_t x0 = j.cx * GME_CHUNK_DIM - GME_TILE_APRON, z0 = j.cz * GME_CHUNK_DIM - GME_TILE_APRON;

    uint32_t* tw = (uint32_t*)&s_t;                      // gme_col_tile_init, in parallel
    for (uint32_t i = threadIdx.x; i < sizeof(GmeColTile) / 4; i += TILE_THREADS) tw[i] = 0u;
    int32_t mn = 0x7FFFFFFF, mx = -0x7FFFFFFF - 1;
    for (int i = threadIdx.x; i < GME_TILE_COLS; i += TILE_THREADS) {
        int32_t h, dev;
        gme_col_tile_build_column(x0 + i % GME_TILE_N, z0 + i / GME_TILE_N, seed, &planet, &h, &dev);
        s_h[i] = h;
        s_dev[i] = dev;
        mn = min(mn, h);
        mx = max(mx, h);
    }
    for (int o = 16; o > 0; o >>= 1) {
        mn = min(mn, __shfl_xor_sync(0xFFFFFFFFu, mn, o));
        mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
    }
    if ((threadIdx.x & 31u) == 0) { s_mn[threadIdx.x >> 5] = mn; s_mx[threadIdx.x >> 5] = mx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t a = s_mn[0], b = s_mx[0];
        for (int w = 1; w < TILE_THREADS / 32; ++w) { a = min(a, s_mn[w]); b = max(b, s_mx[w]); }
        s_t.tag = gme_page_tag(j.cx, j.cz);
        s_t.cx = j.cx;
        s_t.cz = j.cz;
        s_t.h_base = (int16_t)a;
        s_sat = (b - a) > 255;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < GME_TILE_COLS; i += TILE_THREADS) gme_tile_pack_col(&s_t, i, s_h[i], s_dev[i]);
    if (threadIdx.x < 16) s_fl[threadIdx.x] = gme_tile_block(&s_t, (int)threadIdx.x, s_h, s_dev, seed, &planet, &bio);
    if (threadIdx.x >= 32 && threadIdx.x < 32 + GME_LOD_N * GME_LOD_N) {   // M3: the 4 m LOD lattice, one point per thread
        const int k = (int)threadIdx.x - 32;
        s_t.lod4[k] = (int16_t)gme_tile_lod4_at(s_h, s_dev, k % GME_LOD_N, k / GME_LOD_N);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        gme_tile_header(&s_t, s_fl, s_sat);
        gme_tile_lod_bounds(&s_t);
    }
    __syncthreads();
    const uint4* src = (const uint4*)&s_t;
    uint4* dst = (uint4*)(out + (direct ? blockIdx.x : j.index));
    for (uint32_t i = threadIdx.x; i < TILE_U4; i += TILE_THREADS) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
extern "C" int gpu_query(GpuInfo* out) {
    memset(out, 0, sizeof *out);
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        cudaGetLastError();
        snprintf(g_err, sizeof g_err, "no CUDA device");
        return -1;
    }
    cudaDeviceProp p;
    if (ck(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties", __LINE__)) return -1;
    if (ck(cudaSetDevice(0), "cudaSetDevice", __LINE__)) return -1;
    size_t fr = 0, tot = 0;
    if (ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo", __LINE__)) return -1;
    snprintf(out->name, sizeof out->name, "%s", p.name);
    out->cc_major  = p.major;
    out->cc_minor  = p.minor;
    out->sm_count  = p.multiProcessorCount;
    out->total_mem = tot;
    out->free_mem  = fr;
    out->available = 1;
    return 0;
}

extern "C" void gpu_world_destroy(GpuWorld* g) {
    if (!g) return;
    for (int i = 0; i < GPU_BUFFERS; ++i) {
        Buf* b = &g->b[i];
        if (b->stream) cudaStreamSynchronize(b->stream);
        cudaFree(b->d_jobs); cudaFree(b->d_meta); cudaFree(b->d_vox); cudaFree(b->d_tjobs);
        cudaFreeHost(b->h_jobs); cudaFreeHost(b->h_meta); cudaFreeHost(b->h_vox); cudaFreeHost(b->h_tjobs);
        if (b->done) cudaEventDestroy(b->done);
        if (b->stream) cudaStreamDestroy(b->stream);
    }
    cudaFree(g->d_pool);
    cudaFree(g->d_bio);
    cudaFree(g->d_tiles);
    free(g);
}

extern "C" GpuWorld* gpu_world_create(uint32_t slots, uint32_t max_batch, uint32_t tile_count) {
    GpuWorld* g = (GpuWorld*)calloc(1, sizeof(GpuWorld));
    if (!g) return NULL;
    g->slots = slots;
    g->max_batch = max_batch;
    g->tile_count = tile_count;
    const size_t pool  = (size_t)slots * GME_CHUNK_VOXELS;
    const size_t bvox  = (size_t)max_batch * GME_CHUNK_VOXELS;
    const size_t bmeta = (size_t)max_batch * sizeof(GmeChunkMeta);
    const size_t bjobs = (size_t)max_batch * sizeof(GmeGenJob);
    const size_t btile = (size_t)max_batch * sizeof(GmeTileJob);
    const size_t tiles = (size_t)tile_count * sizeof(GmeColTile);
    int prio_least = 0, prio_greatest = 0;   // declared before the first CK: goto fail must not skip it

    CK(cudaSetDevice(0));
    // Mapped pinned memory is implicit with UVA (all 64-bit targets); the flag
    // may be rejected once the context exists, which is harmless.
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaGetLastError();
    CK(cudaMalloc(&g->d_pool, pool));
    g->device_bytes = pool;
    if (tile_count) {
        CK(cudaMalloc(&g->d_tiles, tiles));
        CK(cudaMemset(g->d_tiles, 0, tiles));
        g->device_bytes += tiles;
    }
    // generation streams get the least priority: the renderer's stream (greatest)
    // then wins the SMs whenever both have work queued (spec 6.1)
    CK(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest));
    for (int i = 0; i < GPU_BUFFERS; ++i) {
        Buf* b = &g->b[i];
        CK(cudaStreamCreateWithPriority(&b->stream, cudaStreamNonBlocking, prio_least));
        CK(cudaEventCreateWithFlags(&b->done, cudaEventDisableTiming));
        CK(cudaMalloc(&b->d_jobs, bjobs));
        CK(cudaMalloc(&b->d_meta, bmeta));
        CK(cudaMalloc(&b->d_vox, bvox));
        CK(cudaMalloc(&b->d_tjobs, btile));
        CK(cudaHostAlloc(&b->h_jobs, bjobs, cudaHostAllocWriteCombined)); // host writes, GPU reads
        CK(cudaHostAlloc(&b->h_meta, bmeta, cudaHostAllocDefault));
        CK(cudaHostAlloc(&b->h_vox, bvox, cudaHostAllocMapped));
        CK(cudaHostAlloc(&b->h_tjobs, btile, cudaHostAllocDefault));
        CK(cudaHostGetDevicePointer((void**)&b->h_vox_dev, b->h_vox, 0));
        g->device_bytes += bjobs + bmeta + bvox + btile;
        g->pinned_bytes += bjobs + bmeta + bvox + btile;
    }
    return g;
fail:
    cudaGetLastError();
    gpu_world_destroy(g);
    return NULL;
}

extern "C" size_t gpu_world_device_bytes(const GpuWorld* g) { return g->device_bytes; }
extern "C" size_t gpu_world_pinned_bytes(const GpuWorld* g) { return g->pinned_bytes; }
extern "C" const uint8_t* gpu_world_pool(const GpuWorld* g) { return g->d_pool; }
extern "C" const GmeColTile* gpu_world_tiles(const GpuWorld* g) { return g->d_tiles; }
extern "C" GmeGenJob*    gpu_staging_jobs(GpuWorld* g, int i)   { return g->b[i].h_jobs; }
extern "C" GmeChunkMeta* gpu_staging_meta(GpuWorld* g, int i)   { return g->b[i].h_meta; }
extern "C" uint8_t*      gpu_staging_voxels(GpuWorld* g, int i) { return g->b[i].h_vox; }
extern "C" GmeTileJob*   gpu_staging_tiles(GpuWorld* g, int i)  { return g->b[i].h_tjobs; }

extern "C" int gpu_world_set_biomap(GpuWorld* g, const uint32_t* cells, size_t n) {
    // caller guarantees no batch is generating (world_create / after world_flush)
    cudaFree(g->d_bio);
    g->device_bytes -= g->bio_cells * sizeof(uint32_t);
    g->d_bio = NULL;
    g->bio_cells = 0;
    if (!cells || !n) return 0;
    CK(cudaMalloc(&g->d_bio, n * sizeof(uint32_t)));
    CK(cudaMemcpy(g->d_bio, cells, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    g->bio_cells = n;
    g->device_bytes += n * sizeof(uint32_t);
    return 0;
fail:
    cudaFree(g->d_bio);
    g->d_bio = NULL;
    return -1;
}

extern "C" const uint32_t* gpu_world_biomap(const GpuWorld* g) { return g->d_bio; }

extern "C" int gpu_submit(GpuWorld* g, int i, uint32_t n, uint32_t ntiles, uint32_t seed, const GmePlanet* planet,
                          const GmeBioView* view) {
    Buf* b = &g->b[i];
    if (n > g->max_batch || ntiles > g->max_batch || (n == 0 && ntiles == 0) || (ntiles && !g->d_tiles)) {
        snprintf(g_err, sizeof g_err, "bad batch size %u chunks + %u tiles", n, ntiles);
        return -1;
    }
    GmeBioView bv;
    if (view) bv = *view; else memset(&bv, 0, sizeof bv);
    if (bv.cells) {  // same cells, device copy
        if (!g->d_bio) { snprintf(g_err, sizeof g_err, "biomap view without device cells (gpu_world_set_biomap)"); return -1; }
        bv.cells = g->d_bio;
    }
    // M4: the renderer reads slots and tiles of the frame it queued last; a slot or tile cell
    // freed since may be rewritten only after that frame (the GPU waits, the CPU does not)
    if (g->fence) CK(cudaStreamWaitEvent(b->stream, g->fence, 0));
    if (ntiles) {
        CK(cudaMemcpyAsync(b->d_tjobs, b->h_tjobs, ntiles * sizeof(GmeTileJob), cudaMemcpyHostToDevice, b->stream));
        k_col_tiles<<<ntiles, TILE_THREADS, 0, b->stream>>>(b->d_tjobs, g->d_tiles, 0, seed, *planet, bv);
    }
    if (n) {
        CK(cudaMemcpyAsync(b->d_jobs, b->h_jobs, n * sizeof(GmeGenJob), cudaMemcpyHostToDevice, b->stream));
        k_meta_init<<<(n + 255) / 256, 256, 0, b->stream>>>(b->d_meta, n);
        k_generate<<<n * BLOCKS_PER_CHUNK, COLS_PER_BLOCK, 0, b->stream>>>(b->d_jobs, b->d_vox, b->d_meta, seed, *planet, bv);
        k_commit<<<n, 256, 0, b->stream>>>(b->d_jobs, b->d_meta, (const uint4*)b->d_vox,
                                           (uint4*)g->d_pool, (uint4*)b->h_vox_dev);
    }
    CK(cudaGetLastError());
    if (n) CK(cudaMemcpyAsync(b->h_meta, b->d_meta, n * sizeof(GmeChunkMeta), cudaMemcpyDeviceToHost, b->stream));
    CK(cudaEventRecord(b->done, b->stream));
    return 0;
fail:
    return -1;
}

extern "C" void gpu_world_set_fence(GpuWorld* g, void* ev) { g->fence = (cudaEvent_t)ev; }

extern "C" int gpu_poll(GpuWorld* g, int i) {
    cudaError_t e = cudaEventQuery(g->b[i].done);
    if (e == cudaSuccess) return 1;
    if (e == cudaErrorNotReady) return 0;
    ck(e, "cudaEventQuery", __LINE__);
    return -1;
}

extern "C" int gpu_wait(GpuWorld* g, int i) {
    return ck(cudaEventSynchronize(g->b[i].done), "cudaEventSynchronize", __LINE__) ? -1 : 0;
}

extern "C" int gpu_read_slot(GpuWorld* g, uint32_t slot, uint8_t* dst) {
    if (slot >= g->slots) return -1;
    return ck(cudaMemcpy(dst, g->d_pool + (size_t)slot * GME_CHUNK_VOXELS, GME_CHUNK_VOXELS,
                         cudaMemcpyDeviceToHost), "cudaMemcpy slot", __LINE__) ? -1 : 0;
}

extern "C" int gpu_read_tile(GpuWorld* g, uint32_t index, GmeColTile* dst) {
    if (index >= g->tile_count) return -1;
    return ck(cudaMemcpy(dst, g->d_tiles + index, sizeof(GmeColTile), cudaMemcpyDeviceToHost),
              "cudaMemcpy tile", __LINE__) ? -1 : 0;
}

extern "C" int gpu_build_tiles(const GmeTileJob* jobs, uint32_t n, uint32_t seed, const GmePlanet* planet,
                               const GmeBioView* view, size_t ncells, GmeColTile* out) {
    GmeTileJob* d_jobs = NULL;
    GmeColTile* d_out = NULL;
    uint32_t* d_cells = NULL;
    int rc = -1;
    GmeBioView bv;
    if (view) bv = *view; else memset(&bv, 0, sizeof bv);
    if (!n) return 0;
    CK(cudaSetDevice(0));
    if (bv.cells && ncells) {
        CK(cudaMalloc(&d_cells, ncells * sizeof(uint32_t)));
        CK(cudaMemcpy(d_cells, bv.cells, ncells * sizeof(uint32_t), cudaMemcpyHostToDevice));
        bv.cells = d_cells;
    } else {
        bv.cells = NULL;
    }
    CK(cudaMalloc(&d_jobs, n * sizeof(GmeTileJob)));
    CK(cudaMalloc(&d_out, (size_t)n * sizeof(GmeColTile)));
    CK(cudaMemcpy(d_jobs, jobs, n * sizeof(GmeTileJob), cudaMemcpyHostToDevice));
    k_col_tiles<<<n, TILE_THREADS>>>(d_jobs, d_out, 1, seed, *planet, bv);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out, d_out, (size_t)n * sizeof(GmeColTile), cudaMemcpyDeviceToHost));
    rc = 0;
fail:
    cudaFree(d_jobs);
    cudaFree(d_out);
    cudaFree(d_cells);
    return rc;
}
