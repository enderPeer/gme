/* world.c - chunk residency, streaming and budgets.
 *
 * Memory model
 *   slot   : 32 KB of voxel storage, same index on GPU pool and host mirror.
 *            Only non-uniform chunks hold a slot; all-air / all-stone /
 *            all-water chunks are stored as a single material in the record.
 *   record : 24 B per known chunk (coords, state, slot or uniform material),
 *            found via an open-addressing hash (linear probing, backward-shift
 *            delete, so no tombstones ever accumulate).
 *
 * Streaming
 *   A precomputed table of column offsets sorted by distance gives the load
 *   order (nearest first). Each update scans a window of columns in parallel
 *   for missing chunks, fills a batch and hands it to the GPU. The view
 *   radius adapts: it grows while storage is left and shrinks under pressure,
 *   so the world always fills exactly the memory budget of this machine.
 *
 * Column tiles (GPU mode, density.h)
 *   One 3,456 B GmeColTile per chunk column holds the smooth surface T the
 *   renderer draws. Tile records live in a toroidal page_n^2 device table
 *   counted inside the VRAM budget. A column whose published tile_tag is not
 *   its key gets one tile job in the next batch (tile_job_tag marks the cell
 *   busy, so at most one job is in flight per cell); finish_batch publishes it
 *   when the column is still within radius + 1, and eviction unpublishes it.
 *   The renderer reads only published tiles, exactly like page entries.
 *
 * Renderer upload (M4, spec 6.1)
 *   Every page entry and tile tag that changes is also appended to a scatter
 *   list (capacity 1/8 of its table); world_get_scene hands the lists to the
 *   renderer, which scatters only those entries. An overflowing list (or a new
 *   table) makes the renderer upload the whole table once. The generation
 *   streams wait for the renderer's last queued frame (world_set_render_fence)
 *   before writing chunk slots or tile records, which that frame may still be
 *   reading now that the readback is asynchronous.
 */
#include "world.h"
#include "jobs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK_BYTES   ((size_t)GME_CHUNK_VOXELS)
#define NONE          0xFFFFFFFFu
#define SCAN_WINDOW   4096u
#define KEY_BIAS      (1 << 20)          /* chunk coords valid in [-2^20, 2^20) */
#define COORD_LIMIT   ((1 << 20) - 2)
#define UNIFORM_NONE  0xFFu

enum { REC_FREE = 0, REC_PENDING = 1, REC_READY = 2 };

typedef struct ChunkRec {
    int32_t  cx, cy, cz;
    uint32_t slot;
    uint32_t non_air;
    uint8_t  state, uniform, pad[2];
} ChunkRec;

typedef struct ColOffset { int16_t dx, dz; uint32_t d2; } ColOffset;

typedef struct Batch {
    GmeGenJob*    jobs;
    uint32_t*     recs;
    GmeChunkMeta* meta;   /* CPU mode only */
    uint32_t      count;
    GmeTileJob*   tiles;  /* GPU mode only (pinned staging) */
    uint32_t      ntiles;
    int           inflight;
} Batch;

struct World {
    WorldConfig cfg;
    uint32_t    layers;
    int         use_gpu;
    GpuWorld*   gpu;
    int         nbuf;
    Batch       batch[GPU_BUFFERS];

    uint32_t  slot_count;
    uint8_t*  host_pool;
    size_t    host_pool_bytes;
    uint32_t* slot_free;
    uint32_t  slot_free_top;

    uint32_t  rec_cap;
    ChunkRec* recs;
    uint32_t* rec_free;
    uint32_t  rec_free_top;

    uint64_t* hkeys;
    uint32_t* hvals;
    uint32_t  hmask;

    ColOffset* cols;
    uint32_t   col_total;
    int32_t    radius_max, radius;

    uint32_t* missing;
    uint8_t*  evict_mark;
    uint32_t  cursor;
    int32_t   vcx, vcy, vcz;
    int       have_viewer;
    int32_t   ref_cx, ref_cz;   /* first viewer column: anchor of the ref_hash block */
    uint32_t  layer_order[32];
    int       evict_needed, pressure;

    GmePageEntry* page;     /* renderer lookup table (toroidal) */
    uint32_t      page_n;
    int           page_dirty;
    uint32_t*     pdirty;        /* M4 scatter list: page entries changed since world_get_scene */
    uint32_t      pdirty_n, pdirty_cap;
    int           pdirty_full;   /* the list overflowed (or the table is new): upload all of it */
    uint32_t*     tdirty;        /* ... tile tags */
    uint32_t      tdirty_n, tdirty_cap;
    int           tdirty_full;

    /* column tiles, GPU mode only (toroidal page_n^2, like page) */
    uint32_t*     tile_tag;      /* published tile per cell: gme_tile_key, 0 = none */
    uint32_t*     tile_job_tag;  /* tile job in flight per cell: gme_tile_key, 0 = none */
    uint8_t*      tile_missing;  /* scan output, per SCAN_WINDOW column */
    int           tile_dirty;    /* tile_tag changed since world_get_scene */
    int           tile_deferred; /* a tile job waited for a busy cell: rescan */
    uint32_t      tiles_ready;
    uint64_t      tiles_built;
    size_t        tile_bytes;

    uint32_t n_ready, n_pending, n_uniform_ready, slots_used;
    uint64_t generated, generated_uniform, evicted, pressure_events;
};

/* ---- hash ---------------------------------------------------------------- */
static inline uint64_t chunk_key(int32_t cx, int32_t cy, int32_t cz) {
    return (1ull << 63)
         | ((uint64_t)((uint32_t)(cx + KEY_BIAS) & 0x1FFFFFu) << 42)
         | ((uint64_t)((uint32_t)(cy + KEY_BIAS) & 0x1FFFFFu) << 21)
         |  (uint64_t)((uint32_t)(cz + KEY_BIAS) & 0x1FFFFFu);
}

static inline uint32_t key_hash(uint64_t k) {
    k ^= k >> 30; k *= 0xBF58476D1CE4E5B9ull;
    k ^= k >> 27; k *= 0x94D049BB133111EBull;
    k ^= k >> 31;
    return (uint32_t)k;
}

static uint32_t h_find(const World* w, uint64_t key) {
    uint32_t i = key_hash(key) & w->hmask;
    for (;;) {
        const uint64_t k = w->hkeys[i];
        if (k == key) return w->hvals[i];
        if (!k) return NONE;
        i = (i + 1) & w->hmask;
    }
}

static void h_insert(World* w, uint64_t key, uint32_t val) {
    uint32_t i = key_hash(key) & w->hmask;
    while (w->hkeys[i]) i = (i + 1) & w->hmask;
    w->hkeys[i] = key;
    w->hvals[i] = val;
}

static void h_remove(World* w, uint64_t key) {
    uint32_t i = key_hash(key) & w->hmask;
    while (w->hkeys[i] != key) {
        if (!w->hkeys[i]) return;
        i = (i + 1) & w->hmask;
    }
    for (uint32_t j = i;;) {
        j = (j + 1) & w->hmask;
        if (!w->hkeys[j]) break;
        const uint32_t home = key_hash(w->hkeys[j]) & w->hmask;
        /* entry j may stay if its home lies cyclically in (i, j] */
        const int stays = (i <= j) ? (i < home && home <= j) : (i < home || home <= j);
        if (stays) continue;
        w->hkeys[i] = w->hkeys[j];
        w->hvals[i] = w->hvals[j];
        i = j;
    }
    w->hkeys[i] = 0;
}

/* ---- helpers ------------------------------------------------------------ */
static int col_cmp(const void* a, const void* b) {
    const ColOffset* x = (const ColOffset*)a;
    const ColOffset* y = (const ColOffset*)b;
    if (x->d2 != y->d2) return x->d2 < y->d2 ? -1 : 1;
    if (x->dz != y->dz) return x->dz < y->dz ? -1 : 1;
    return (x->dx > y->dx) - (x->dx < y->dx);
}

/* number of columns within radius r (prefix of the sorted table) */
static uint32_t col_count(const World* w, int32_t r) {
    const uint32_t r2 = (uint32_t)(r * r);
    uint32_t lo = 0, hi = w->col_total;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (w->cols[mid].d2 <= r2) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static int any_inflight(const World* w) {
    for (int b = 0; b < w->nbuf; ++b) if (w->batch[b].inflight) return 1;
    return 0;
}

static int fully_loaded(const World* w) {
    return w->have_viewer && w->cursor >= col_count(w, w->radius) && !any_inflight(w);
}

/* M4 scatter lists: an entry of the page table / tile tags changed */
static inline void page_changed(World* w, uint32_t idx) {
    w->page_dirty = 1;
    if (w->pdirty_full) return;
    if (w->pdirty_n < w->pdirty_cap) w->pdirty[w->pdirty_n++] = idx;
    else w->pdirty_full = 1;
}
static inline void tile_changed(World* w, uint32_t idx) {
    w->tile_dirty = 1;
    if (w->tdirty_full) return;
    if (w->tdirty_n < w->tdirty_cap) w->tdirty[w->tdirty_n++] = idx;
    else w->tdirty_full = 1;
}

static void slot_release(World* w, uint32_t s) {
    w->slot_free[w->slot_free_top++] = s;
    w->slots_used--;
}

void worldpos_normalize(WorldPos* p) {
    float* l[3] = { &p->lx, &p->ly, &p->lz };
    int32_t* c[3] = { &p->cx, &p->cy, &p->cz };
    for (int i = 0; i < 3; ++i) {
        const float f = floorf(*l[i] / (float)GME_CHUNK_DIM);
        *c[i] += (int32_t)f;
        *l[i] -= f * (float)GME_CHUNK_DIM;
        if (*l[i] >= (float)GME_CHUNK_DIM) { *l[i] -= (float)GME_CHUNK_DIM; ++*c[i]; }
        if (*l[i] < 0.0f) *l[i] = 0.0f;
    }
}

/* ---- CPU generator (fallback + validation reference) -------------------- */
/* Same per-column / per-voxel functions as k_generate (flora.h), so the CPU
 * and GPU worlds are bit-identical with or without vegetation. The 4 x 4
 * vegetation slots of the chunk are computed once each, on first use (the
 * GPU computes them per column; the result is the same). flora_cols
 * (optional) receives the number of columns whose biomap kind is not 0. */
#define SLOTS_PER_ROW (GME_CHUNK_DIM >> GME_FLORA_SLOT_SHIFT)

static void cpu_gen_chunk(int32_t cx, int32_t cy, int32_t cz, uint32_t seed, const GmePlanet* pl,
                          const GmeBioView* bio, uint8_t* out, GmeChunkMeta* meta, uint32_t* flora_cols) {
    uint32_t mn = 0xFF, mx = 0, solid = 0, fcols = 0;
    const int32_t wy0 = cy * GME_CHUNK_DIM, wy1 = wy0 + GME_CHUNK_DIM - 1;
    GmeSlot slot[SLOTS_PER_ROW * SLOTS_PER_ROW];
    uint8_t have_slot[SLOTS_PER_ROW * SLOTS_PER_ROW] = { 0 };
    for (int lz = 0; lz < GME_CHUNK_DIM; ++lz) {
        for (int lx = 0; lx < GME_CHUNK_DIM; ++lx) {
            const int32_t wx = cx * GME_CHUNK_DIM + lx, wz = cz * GME_CHUNK_DIM + lz;
            const int32_t h = gme_height(wx, wz, seed, pl);
            GmeColumn fc;
            if (gme_flora_in_span(h, wy0, wy1, pl, bio)) {
                const int si = (lz >> GME_FLORA_SLOT_SHIFT) * SLOTS_PER_ROW + (lx >> GME_FLORA_SLOT_SHIFT);
                if (!have_slot[si]) {
                    gme_slot(wx >> GME_FLORA_SLOT_SHIFT, wz >> GME_FLORA_SLOT_SHIFT, seed, pl, bio, &slot[si]);
                    have_slot[si] = 1;
                }
                gme_column_in_slot(wx, wz, h, seed, &slot[si], &fc);
            } else {
                gme_column_bare(h, &fc);
            }
            fcols += fc.cell != 0;
            uint8_t* dst = out + GME_VOXEL_INDEX(lx, 0, lz);
            const size_t ys = GME_CHUNK_DIM * GME_CHUNK_DIM;   /* one XZ layer */
            if (!fc.cell) {   /* bare column: plain geology (gme_voxel would return the same) */
                for (int ly = 0; ly < GME_CHUNK_DIM; ++ly) dst[ly * ys] = gme_material(wx, wy0 + ly, wz, h, seed, pl);
            } else {
                for (int ly = 0; ly < GME_CHUNK_DIM; ++ly) dst[ly * ys] = gme_voxel(wx, wy0 + ly, wz, &fc, seed, pl);
            }
            for (int ly = 0; ly < GME_CHUNK_DIM; ++ly) {
                const uint8_t m = dst[ly * ys];
                if (m < mn) mn = m;
                if (m > mx) mx = m;
                solid += (m != MAT_AIR);
            }
        }
    }
    meta->mat_min = mn; meta->mat_max = mx; meta->non_air = solid; meta->pad = 0;
    if (flora_cols) *flora_cols = fcols;
}

/* cells a view's regions address (the size of the array behind view->cells) */
static size_t bio_view_cells(const GmeBioView* b) {
    size_t n = 0;
    for (int r = 0; r < b->nreg && r < BIOMAP_MAX_REGIONS; ++r) {
        const size_t e = (size_t)b->reg[r].offset + (size_t)b->reg[r].w * (size_t)b->reg[r].h;
        if (e > n) n = e;
    }
    return n;
}

/* records for a pool of `slots`: room for ~4 chunks per slot (uniform chunks need no slot) */
static uint32_t rec_cap_for(uint64_t slots, uint32_t batch) {
    uint64_t rc = slots * 4 + (uint64_t)GPU_BUFFERS * batch;
    if (rc > (1u << 24)) rc = 1u << 24;
    return (uint32_t)rc;
}

/* max radius: all columns' records fit, minus in-flight + hysteresis margin */
static int32_t radius_max_for(uint32_t rec_cap, uint32_t batch, uint32_t layers) {
    const double cols_budget = 0.85 * (double)(rec_cap - GPU_BUFFERS * batch) / layers;
    int32_t rmax = (int32_t)floor(sqrt(cols_budget / 3.14159265358979));
    if (rmax > 2000) rmax = 2000;
    if (rmax < 2) rmax = 2;
    return rmax;
}

/* ---- create / destroy --------------------------------------------------- */
World* world_create(const WorldConfig* cfg, const SysInfo* si, const GpuInfo* gi) {
    World* w = (World*)calloc(1, sizeof *w);
    if (!w) return NULL;
    w->cfg = *cfg;
    if (cfg->max_cy <= cfg->min_cy || cfg->max_cy - cfg->min_cy > 32) {
        fprintf(stderr, "world: vertical range must be 1..32 chunk layers\n");
        goto fail;
    }
    w->layers = (uint32_t)(cfg->max_cy - cfg->min_cy);
    w->use_gpu = !cfg->force_cpu && gi && gi->available;

    uint32_t batch = cfg->max_batch ? cfg->max_batch : 2048;
    if (!w->use_gpu) batch = (uint32_t)jobs_thread_count() * 4u;
    w->cfg.max_batch = batch;
    w->nbuf = w->use_gpu ? GPU_BUFFERS : 1;

    /* --- storage budget: limited by both VRAM and RAM --- */
    uint64_t slots = (uint64_t)((double)si->avail_ram * cfg->ram_fraction) / CHUNK_BYTES;
    if (w->use_gpu) {
        double budget = (double)gi->free_mem * cfg->vram_fraction;
        budget -= (double)GPU_BUFFERS * batch * (double)(CHUNK_BYTES + sizeof(GmeChunkMeta) + sizeof(GmeGenJob));
        uint64_t gslots = budget > 0 ? (uint64_t)(budget / (double)CHUNK_BYTES) : 0;
        /* column tiles are counted inside the budget: page_n^2 records for the
         * page_n this pool would get (a smaller final page_n over-reserves
         * slightly), plus their job staging */
        {
            const uint64_t s1 = gslots < slots ? gslots : slots;
            const uint32_t n1 = (uint32_t)(2 * radius_max_for(rec_cap_for(s1, batch), batch, w->layers) + 8);
            const double tb = (double)n1 * n1 * sizeof(GmeColTile) + (double)GPU_BUFFERS * batch * 2.0 * sizeof(GmeTileJob);
            budget -= tb;
            gslots = budget > 0 ? (uint64_t)(budget / (double)CHUNK_BYTES) : 0;
        }
        if (gslots < slots) slots = gslots;
    }
    if (slots > 0x3FFFFFFFull) slots = 0x3FFFFFFFull;
    if (slots < (uint64_t)batch * 4) {
        fprintf(stderr, "world: not enough memory for a chunk pool (%llu slots)\n", (unsigned long long)slots);
        goto fail;
    }
    w->slot_count = (uint32_t)slots;
    w->rec_cap = rec_cap_for(slots, batch);
    w->radius_max = radius_max_for(w->rec_cap, batch, w->layers);
    {
        const int32_t r = w->radius_max;
        w->cols = (ColOffset*)malloc(sizeof(ColOffset) * (size_t)(2 * r + 1) * (size_t)(2 * r + 1));
        if (!w->cols) goto fail;
        for (int32_t dz = -r; dz <= r; ++dz)
            for (int32_t dx = -r; dx <= r; ++dx) {
                const uint32_t d2 = (uint32_t)(dx * dx + dz * dz);
                if (d2 <= (uint32_t)(r * r)) {
                    ColOffset c = { (int16_t)dx, (int16_t)dz, d2 };
                    w->cols[w->col_total++] = c;
                }
            }
        qsort(w->cols, w->col_total, sizeof(ColOffset), col_cmp);
    }
    w->radius = cfg->radius_override > 0 ? cfg->radius_override : 4;
    if (w->radius > w->radius_max) w->radius = w->radius_max;

    {
        uint32_t hcap = 1;
        while (hcap < w->rec_cap * 2u) hcap <<= 1;
        w->hmask = hcap - 1;
        w->hkeys = (uint64_t*)calloc(hcap, sizeof(uint64_t));
        w->hvals = (uint32_t*)calloc(hcap, sizeof(uint32_t));
    }
    w->host_pool_bytes = (size_t)w->slot_count * CHUNK_BYTES;
    w->host_pool  = (uint8_t*)os_alloc(w->host_pool_bytes);
    w->slot_free  = (uint32_t*)malloc(sizeof(uint32_t) * w->slot_count);
    w->recs       = (ChunkRec*)calloc(w->rec_cap, sizeof(ChunkRec));
    w->rec_free   = (uint32_t*)malloc(sizeof(uint32_t) * w->rec_cap);
    w->missing    = (uint32_t*)malloc(sizeof(uint32_t) * SCAN_WINDOW);
    w->evict_mark = (uint8_t*)malloc(w->rec_cap);
    /* table wider than the resident diameter so live chunks never collide */
    w->page_n     = (uint32_t)(2 * w->radius_max + 8);
    w->page       = (GmePageEntry*)malloc(sizeof(GmePageEntry) * (size_t)w->page_n * w->page_n * w->layers);
    w->pdirty_cap = (uint32_t)((size_t)w->page_n * w->page_n * w->layers / 8 + 1);
    w->pdirty     = (uint32_t*)malloc(sizeof(uint32_t) * w->pdirty_cap);
    w->tdirty_cap = (uint32_t)((size_t)w->page_n * w->page_n / 8 + 1);
    w->tdirty     = (uint32_t*)malloc(sizeof(uint32_t) * w->tdirty_cap);
    w->pdirty_full = w->tdirty_full = 1;          /* the first upload is the whole table */
    if (!w->hkeys || !w->hvals || !w->host_pool || !w->slot_free || !w->recs ||
        !w->rec_free || !w->missing || !w->evict_mark || !w->page || !w->pdirty || !w->tdirty) {
        fprintf(stderr, "world: out of host memory\n");
        goto fail;
    }
    if (w->use_gpu) {
        const size_t nt = (size_t)w->page_n * w->page_n;
        w->tile_tag     = (uint32_t*)calloc(nt, sizeof(uint32_t));
        w->tile_job_tag = (uint32_t*)calloc(nt, sizeof(uint32_t));
        w->tile_missing = (uint8_t*)calloc(SCAN_WINDOW, 1);
        w->tile_bytes   = nt * sizeof(GmeColTile);
        if (!w->tile_tag || !w->tile_job_tag || !w->tile_missing) {
            fprintf(stderr, "world: out of host memory\n");
            goto fail;
        }
        w->tile_dirty = 1;
    }
    for (uint32_t i = 0; i < w->slot_count; ++i) w->slot_free[i] = w->slot_count - 1 - i;
    w->slot_free_top = w->slot_count;
    for (uint32_t i = 0; i < w->rec_cap; ++i) w->rec_free[i] = w->rec_cap - 1 - i;
    w->rec_free_top = w->rec_cap;
    for (size_t i = 0; i < (size_t)w->page_n * w->page_n * w->layers; ++i) {
        w->page[i].tag = 0;
        w->page[i].val = GME_PAGE_UNLOADED;
    }
    w->page_dirty = 1;

    if (w->use_gpu) {
        w->gpu = gpu_world_create(w->slot_count, batch, w->page_n * w->page_n);
        if (!w->gpu) { fprintf(stderr, "world: GPU init failed: %s\n", gpu_last_error()); goto fail; }
        if (cfg->bio.cells && gpu_world_set_biomap(w->gpu, cfg->bio.cells, bio_view_cells(&cfg->bio))) {
            fprintf(stderr, "world: biomap upload failed: %s\n", gpu_last_error());
            goto fail;
        }
        /* version 2: the morphology words, parallel to the cells */
        if (cfg->bio.cells && cfg->bio.morph && gpu_world_set_biomorph(w->gpu, cfg->bio.morph, bio_view_cells(&cfg->bio))) {
            fprintf(stderr, "world: biomap morphology upload failed: %s\n", gpu_last_error());
            goto fail;
        }
    }
    for (int b = 0; b < w->nbuf; ++b) {
        Batch* B = &w->batch[b];
        B->recs = (uint32_t*)malloc(sizeof(uint32_t) * batch);
        if (w->use_gpu) {
            B->jobs = gpu_staging_jobs(w->gpu, b);
            B->tiles = gpu_staging_tiles(w->gpu, b);
        } else {
            B->jobs = (GmeGenJob*)malloc(sizeof(GmeGenJob) * batch);
            B->meta = (GmeChunkMeta*)malloc(sizeof(GmeChunkMeta) * batch);
            if (!B->jobs || !B->meta) goto fail;
        }
        if (!B->recs) goto fail;
    }
    return w;
fail:
    world_destroy(w);
    return NULL;
}

void world_destroy(World* w) {
    if (!w) return;
    if (w->gpu) { world_flush(w); gpu_world_destroy(w->gpu); }
    for (int b = 0; b < GPU_BUFFERS; ++b) {
        free(w->batch[b].recs);
        if (!w->use_gpu) { free(w->batch[b].jobs); free(w->batch[b].meta); }
    }
    os_free(w->host_pool, w->host_pool_bytes);
    free(w->cols); free(w->hkeys); free(w->hvals); free(w->slot_free);
    free(w->recs); free(w->rec_free); free(w->missing); free(w->evict_mark);
    free(w->page);
    free(w->pdirty); free(w->tdirty);
    free(w->tile_tag); free(w->tile_job_tag); free(w->tile_missing);
    free(w);
}

/* ---- batch completion ---------------------------------------------------- */
typedef struct CopyCtx { World* w; const Batch* B; const GmeChunkMeta* meta; const uint8_t* vox; } CopyCtx;

static void copy_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    const CopyCtx* c = (const CopyCtx*)p;
    for (uint32_t i = b; i < e; ++i) {
        if (c->meta[i].mat_min == c->meta[i].mat_max) continue;
        memcpy(c->w->host_pool + (size_t)c->B->jobs[i].slot * CHUNK_BYTES,
               c->vox + (size_t)i * CHUNK_BYTES, CHUNK_BYTES);
    }
}

static void finish_batch(World* w, Batch* B, const GmeChunkMeta* meta, const uint8_t* vox) {
    if (vox) { /* GPU path: drain pinned staging into the host mirror on all cores */
        CopyCtx c = { w, B, meta, vox };
        jobs_parallel_for(B->count, 4, copy_fn, &c);
    }
    for (uint32_t i = 0; i < B->count; ++i) {
        ChunkRec* r = &w->recs[B->recs[i]];
        r->state = REC_READY;
        r->non_air = meta[i].non_air;
        if (meta[i].mat_min == meta[i].mat_max) {
            r->uniform = (uint8_t)meta[i].mat_min;
            slot_release(w, r->slot);
            r->slot = NONE;
            w->n_uniform_ready++;
            w->generated_uniform++;
        } else {
            r->uniform = UNIFORM_NONE;
        }
        const uint32_t pi = gme_page_index(r->cx, r->cy - w->cfg.min_cy, r->cz, w->page_n, w->layers);
        GmePageEntry* pe = &w->page[pi];
        pe->tag = gme_page_tag(r->cx, r->cz);
        pe->val = r->slot != NONE ? (int32_t)r->slot : -2 - (int32_t)r->uniform;
        page_changed(w, pi);
        w->n_pending--;
        w->n_ready++;
        w->generated++;
    }
    /* tiles: free the cell, publish if the column is still within radius + 1 */
    const int64_t r1 = (int64_t)w->radius + 1;
    for (uint32_t k = 0; k < B->ntiles; ++k) {
        const GmeTileJob tj = B->tiles[k];
        const uint32_t key = gme_tile_key(tj.cx, tj.cz);
        if (w->tile_job_tag[tj.index] == key) w->tile_job_tag[tj.index] = 0;
        const int64_t dx = (int64_t)tj.cx - w->vcx, dz = (int64_t)tj.cz - w->vcz;
        if (dx * dx + dz * dz <= r1 * r1) {
            if (w->tile_tag[tj.index] != key) w->tiles_ready += w->tile_tag[tj.index] == 0;
            w->tile_tag[tj.index] = key;
            tile_changed(w, tj.index);
        }
        w->tiles_built++;
    }
    B->count = 0;
    B->ntiles = 0;
    B->inflight = 0;
}

/* ---- eviction ------------------------------------------------------------ */
static void evict_mark_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    World* w = (World*)p;
    const int64_t r = w->radius + 1; /* 1 chunk hysteresis */
    const int64_t r2 = r * r;
    for (uint32_t i = b; i < e; ++i) {
        const ChunkRec* c = &w->recs[i];
        uint8_t m = 0;
        if (c->state != REC_FREE) {
            const int64_t dx = (int64_t)c->cx - w->vcx, dz = (int64_t)c->cz - w->vcz;
            if (dx * dx + dz * dz > r2) m = (c->state == REC_READY) ? 1 : 2;
        }
        w->evict_mark[i] = m;
    }
}

static void evict(World* w) {
    jobs_parallel_for(w->rec_cap, 8192, evict_mark_fn, w);
    int deferred = 0;
    for (uint32_t i = 0; i < w->rec_cap; ++i) {
        const uint8_t m = w->evict_mark[i];
        if (!m) continue;
        if (m == 2) { deferred = 1; continue; } /* still generating: retry later */
        ChunkRec* r = &w->recs[i];
        h_remove(w, chunk_key(r->cx, r->cy, r->cz));
        const uint32_t pi = gme_page_index(r->cx, r->cy - w->cfg.min_cy, r->cz, w->page_n, w->layers);
        GmePageEntry* pe = &w->page[pi];
        if (pe->tag == gme_page_tag(r->cx, r->cz)) { pe->val = GME_PAGE_UNLOADED; page_changed(w, pi); }
        if (w->tile_tag) {   /* the column leaves the view: unpublish its tile */
            const uint32_t ti = gme_page_index(r->cx, 0, r->cz, w->page_n, 1);
            if (w->tile_tag[ti] == gme_tile_key(r->cx, r->cz)) { w->tile_tag[ti] = 0; w->tiles_ready--; tile_changed(w, ti); }
        }
        if (r->slot != NONE) slot_release(w, r->slot);
        else w->n_uniform_ready--;
        r->state = REC_FREE;
        r->slot = NONE;
        w->rec_free[w->rec_free_top++] = i;
        w->n_ready--;
        w->evicted++;
    }
    w->evict_needed = deferred;
}

/* ---- adaptive radius ----------------------------------------------------- */
static void try_grow(World* w) {
    if (w->cfg.radius_override > 0 || !fully_loaded(w)) return;
    /* measured storage cost per column (non-uniform chunks) */
    double per_col = (double)w->layers;
    if (w->n_ready >= w->layers * 64u) per_col = (double)w->slots_used * w->layers / (double)w->n_ready;
    int64_t free_slots = (int64_t)w->slot_free_top - 2ll * GPU_BUFFERS * w->cfg.max_batch;
    int64_t free_recs  = (int64_t)w->rec_free_top - 2ll * GPU_BUFFERS * w->cfg.max_batch;
    while (w->radius < w->radius_max) {
        const uint32_t ring = col_count(w, w->radius + 1) - col_count(w, w->radius);
        /* next ring plus the hysteresis ring that eviction keeps behind us */
        const double need = 1.25 * per_col * (double)ring * 2.0 + 1.0;
        const int64_t need_recs = 2ll * ring * w->layers;
        if ((double)free_slots < need || free_recs < need_recs) break;
        free_slots -= (int64_t)(need * 0.5);
        free_recs  -= need_recs / 2;
        w->radius++;
    }
}

/* ---- missing-chunk scan + batch build ------------------------------------- */
typedef struct ScanCtx { World* w; uint32_t base; } ScanCtx;

static void scan_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    const ScanCtx* s = (const ScanCtx*)p;
    const World* w = s->w;
    for (uint32_t k = b; k < e; ++k) {
        const ColOffset c = w->cols[s->base + k];
        const int32_t cx = w->vcx + c.dx, cz = w->vcz + c.dz;
        uint32_t mask = 0;
        uint8_t tmiss = 0;
        if (cx > -COORD_LIMIT && cx < COORD_LIMIT && cz > -COORD_LIMIT && cz < COORD_LIMIT) {
            for (uint32_t L = 0; L < w->layers; ++L)
                if (h_find(w, chunk_key(cx, w->cfg.min_cy + (int32_t)L, cz)) == NONE) mask |= 1u << L;
            if (w->tile_tag) tmiss = w->tile_tag[gme_page_index(cx, 0, cz, w->page_n, 1)] != gme_tile_key(cx, cz);
        }
        w->missing[k] = mask;
        if (w->tile_missing) w->tile_missing[k] = tmiss;
    }
}

static uint32_t build_batch(World* w, Batch* B) {
    const uint32_t ncols = col_count(w, w->radius);
    const uint32_t max = w->cfg.max_batch;
    uint32_t n = 0, nt = 0;
    while (n < max && nt < max && w->cursor < ncols) {
        const uint32_t base = w->cursor;
        const uint32_t end = (base + SCAN_WINDOW < ncols) ? base + SCAN_WINDOW : ncols;
        ScanCtx sc = { w, base };
        jobs_parallel_for(end - base, 64, scan_fn, &sc);
        for (uint32_t k = 0; k < end - base; ++k) {
            uint32_t mask = w->missing[k];
            const ColOffset c = w->cols[base + k];
            if (w->tile_missing && w->tile_missing[k]) {
                /* one tile job per column; a tile-only column consumes no slot */
                const int32_t tcx = w->vcx + c.dx, tcz = w->vcz + c.dz;
                const uint32_t ti = gme_page_index(tcx, 0, tcz, w->page_n, 1), key = gme_tile_key(tcx, tcz);
                if (w->tile_job_tag[ti] == 0) {
                    if (nt == max) goto out;
                    if (w->tile_tag[ti]) {        /* another column's stale tile: unpublish before overwriting */
                        w->tile_tag[ti] = 0;
                        w->tiles_ready--;
                        tile_changed(w, ti);
                    }
                    GmeTileJob tj = { tcx, tcz, ti, 0 };
                    B->tiles[nt++] = tj;
                    w->tile_job_tag[ti] = key;
                } else if (w->tile_job_tag[ti] != key) {
                    w->tile_deferred = 1;         /* cell busy with another column's job: rescan later */
                }
                w->tile_missing[k] = 0;
            }
            for (uint32_t li = 0; li < w->layers && mask; ++li) {
                const uint32_t L = w->layer_order[li], bit = 1u << L;
                if (!(mask & bit)) continue;
                if (n == max) goto out;
                if (!w->slot_free_top || !w->rec_free_top) { w->pressure = 1; goto out; }
                const uint32_t slot = w->slot_free[--w->slot_free_top];
                const uint32_t ri = w->rec_free[--w->rec_free_top];
                w->slots_used++;
                ChunkRec* r = &w->recs[ri];
                r->cx = w->vcx + c.dx;
                r->cy = w->cfg.min_cy + (int32_t)L;
                r->cz = w->vcz + c.dz;
                r->slot = slot;
                r->non_air = 0;
                r->state = REC_PENDING;
                r->uniform = UNIFORM_NONE;
                h_insert(w, chunk_key(r->cx, r->cy, r->cz), ri);
                w->n_pending++;
                GmeGenJob j = { r->cx, r->cy, r->cz, slot };
                B->jobs[n] = j;
                B->recs[n] = ri;
                ++n;
                mask &= ~bit;
            }
            w->cursor = base + k + 1; /* column fully present or pending */
        }
    }
out:
    B->count = n;
    B->ntiles = nt;
    return n + nt;
}

typedef struct GenCtx { World* w; Batch* B; } GenCtx;

static void cpu_gen_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    GenCtx* g = (GenCtx*)p;
    for (uint32_t i = b; i < e; ++i) {
        const GmeGenJob j = g->B->jobs[i];
        cpu_gen_chunk(j.cx, j.cy, j.cz, g->w->cfg.seed, &g->w->cfg.planet, &g->w->cfg.bio,
                      g->w->host_pool + (size_t)j.slot * CHUNK_BYTES, &g->B->meta[i], NULL);
    }
}

/* ---- update ------------------------------------------------------------- */
int world_update(World* w, const WorldPos* viewer) {
    /* 1. drain finished GPU batches */
    if (w->use_gpu) {
        for (int b = 0; b < w->nbuf; ++b) {
            if (!w->batch[b].inflight) continue;
            const int st = gpu_poll(w->gpu, b);
            if (st < 0) { fprintf(stderr, "world: %s\n", gpu_last_error()); return -1; }
            if (st == 1) finish_batch(w, &w->batch[b], gpu_staging_meta(w->gpu, b), gpu_staging_voxels(w->gpu, b));
        }
    }

    /* 2. viewer moved to another column / layer? */
    int32_t vcx = viewer->cx, vcz = viewer->cz, vcy = viewer->cy;
    if (vcx < -COORD_LIMIT + 1) vcx = -COORD_LIMIT + 1;
    if (vcx >  COORD_LIMIT - 1) vcx =  COORD_LIMIT - 1;
    if (vcz < -COORD_LIMIT + 1) vcz = -COORD_LIMIT + 1;
    if (vcz >  COORD_LIMIT - 1) vcz =  COORD_LIMIT - 1;
    if (vcy < w->cfg.min_cy) vcy = w->cfg.min_cy;
    if (vcy >= w->cfg.max_cy) vcy = w->cfg.max_cy - 1;
    if (!w->have_viewer) { w->ref_cx = vcx; w->ref_cz = vcz; }
    if (!w->have_viewer || vcx != w->vcx || vcz != w->vcz) {
        w->vcx = vcx; w->vcz = vcz;
        w->cursor = 0;
        w->evict_needed = 1;
    }
    if (w->tile_deferred) {   /* a tile job waited for a busy cell: rescan for it */
        w->tile_deferred = 0;
        w->cursor = 0;
    }
    if (!w->have_viewer || vcy != w->vcy) {
        /* load layers nearest the viewer's altitude first */
        w->vcy = vcy;
        const int32_t rel = vcy - w->cfg.min_cy;
        uint32_t n = 0;
        w->layer_order[n++] = (uint32_t)rel;
        for (int32_t d = 1; n < w->layers; ++d) {
            if (rel - d >= 0) w->layer_order[n++] = (uint32_t)(rel - d);
            if (rel + d < (int32_t)w->layers && n < w->layers) w->layer_order[n++] = (uint32_t)(rel + d);
        }
    }
    w->have_viewer = 1;

    /* 3. drop chunks that left the view */
    if (w->evict_needed) evict(w);

    /* 4. widen the view if memory allows */
    try_grow(w);

    /* 5. feed idle batch buffers */
    w->pressure = 0;
    for (int b = 0; b < w->nbuf; ++b) {
        Batch* B = &w->batch[b];
        if (B->inflight) continue;
        if (!build_batch(w, B)) break;
        if (w->use_gpu) {
            if (gpu_submit(w->gpu, b, B->count, B->ntiles, w->cfg.seed, &w->cfg.planet, &w->cfg.bio)) {
                fprintf(stderr, "world: %s\n", gpu_last_error());
                return -1;
            }
            B->inflight = 1;
        } else {
            GenCtx g = { w, B };
            jobs_parallel_for(B->count, 1, cpu_gen_fn, &g);
            finish_batch(w, B, B->meta, NULL);
        }
    }

    /* 6. out of storage: shrink the view. Pending chunks hold slots only
     * transiently (uniform ones are returned on completion), so judge by the
     * storage held by finished chunks: if those alone leave less than one
     * batch pair of headroom, the radius is too large for this terrain. */
    const uint32_t ready_slots = w->slots_used - w->n_pending;
    const uint32_t headroom = GPU_BUFFERS * w->cfg.max_batch;
    if (w->pressure && w->cfg.radius_override <= 0 && w->radius > 2 &&
        (!any_inflight(w) || ready_slots + headroom >= w->slot_count)) {
        w->radius--;
        const uint32_t nc = col_count(w, w->radius);
        if (w->cursor > nc) w->cursor = nc;
        w->evict_needed = 1;
        w->pressure_events++;
    }
    return 0;
}

int world_flush(World* w) {
    if (!w->use_gpu) return 0;
    for (int b = 0; b < w->nbuf; ++b) {
        if (!w->batch[b].inflight) continue;
        if (gpu_wait(w->gpu, b)) { fprintf(stderr, "world: %s\n", gpu_last_error()); return -1; }
        finish_batch(w, &w->batch[b], gpu_staging_meta(w->gpu, b), gpu_staging_voxels(w->gpu, b));
    }
    return 0;
}

void world_get_stats(const World* w, WorldStats* s) {
    memset(s, 0, sizeof *s);
    s->use_gpu = w->use_gpu;
    s->radius = w->radius;
    s->radius_max = w->radius_max;
    s->layers = w->layers;
    s->slots_total = w->slot_count;
    s->slots_used = w->slots_used;
    s->recs_total = w->rec_cap;
    s->recs_used = w->rec_cap - w->rec_free_top;
    s->ready = w->n_ready;
    s->pending = w->n_pending;
    s->uniform_ready = w->n_uniform_ready;
    s->generated = w->generated;
    s->generated_uniform = w->generated_uniform;
    s->evicted = w->evicted;
    s->pressure_events = w->pressure_events;
    s->fully_loaded = fully_loaded(w);
    s->device_bytes = w->gpu ? gpu_world_device_bytes(w->gpu) : 0;
    s->pinned_bytes = w->gpu ? gpu_world_pinned_bytes(w->gpu) : 0;
    s->tiles_total = w->tile_tag ? w->page_n * w->page_n : 0;
    s->tiles_ready = w->tiles_ready;
    s->tiles_built = w->tiles_built;
    s->tile_bytes = w->tile_bytes;
    s->host_reserved_bytes = w->host_pool_bytes;
    s->host_meta_bytes = (size_t)w->rec_cap * (sizeof(ChunkRec) + sizeof(uint32_t) + 1)
                       + (size_t)(w->hmask + 1) * (sizeof(uint64_t) + sizeof(uint32_t))
                       + (size_t)w->slot_count * sizeof(uint32_t)
                       + (size_t)w->col_total * sizeof(ColOffset);
}

void world_get_scene(World* w, RenderScene* s) {
    s->page = w->page;
    s->page_n = w->page_n;
    s->layers = w->layers;
    s->min_cy = w->cfg.min_cy;
    s->page_dirty = w->page_dirty;
    s->page_dirty_idx = w->pdirty_full ? NULL : w->pdirty;   /* M4 scatter list (valid until the next world_update) */
    s->page_dirty_n = w->pdirty_full ? 0 : w->pdirty_n;
    s->pool = w->gpu ? gpu_world_pool(w->gpu) : NULL;
    s->max_dist = (float)((w->radius > 1 ? w->radius - 1 : 1) * GME_CHUNK_DIM);
    w->page_dirty = 0;
    w->pdirty_n = 0;
    w->pdirty_full = 0;
    /* vegetation tint: device cells + regions + palette (sRGB -> linear, gamma 2
     * like the renderer's output encoding) */
    s->bio_cells = (w->gpu && w->cfg.bio.cells) ? gpu_world_biomap(w->gpu) : NULL;
    s->bio_nreg = s->bio_cells ? w->cfg.bio.nreg : 0;
    s->bio_shift = w->cfg.bio.shift;
    memcpy(s->bio_reg, w->cfg.bio.reg, sizeof s->bio_reg);
    for (int i = 0; i < BIOMAP_PALETTE; ++i)
        for (int k = 0; k < 3; ++k) {
            const float v = (float)w->cfg.bio_palette[i][k] / 255.0f;
            s->bio_palette[i][k] = v * v;
        }
    /* smooth surface: published column tiles + what the analytic fallbacks need */
    s->tiles = w->gpu ? gpu_world_tiles(w->gpu) : NULL;
    s->tile_tag = w->tile_tag;
    s->tile_dirty = w->tile_dirty;
    s->tile_dirty_idx = w->tdirty_full ? NULL : w->tdirty;
    s->tile_dirty_n = w->tdirty_full ? 0 : w->tdirty_n;
    w->tile_dirty = 0;
    w->tdirty_n = 0;
    w->tdirty_full = 0;
    s->planet = w->cfg.planet;
    s->seed = w->cfg.seed;
    s->bio = w->cfg.bio;
    s->bio.cells = s->bio_cells;        /* device cells (NULL without life or without GPU) */
    if (!s->bio.cells) s->bio.nreg = 0;
    /* version 2: the device morphology words (NULL: version 1 shapes, for the fallback classes too) */
    s->bio.morph = (w->gpu && w->cfg.bio.cells && w->cfg.bio.morph) ? gpu_world_biomorph(w->gpu) : NULL;
}

void world_set_render_fence(World* w, void* ev) {
    if (w->gpu) gpu_world_set_fence(w->gpu, ev);
}

int world_surface_q8(const World* w, int32_t x, int32_t z, int32_t* t_q8) {
    *t_q8 = gme_surface_q8(x, z, w->cfg.seed, &w->cfg.planet);
    return 0;
}

/* ---- queries ------------------------------------------------------------ */
uint8_t world_get_voxel(const World* w, int32_t x, int32_t y, int32_t z, int* loaded) {
    const uint32_t ri = h_find(w, chunk_key(x >> GME_CHUNK_SHIFT, y >> GME_CHUNK_SHIFT, z >> GME_CHUNK_SHIFT));
    if (ri == NONE || w->recs[ri].state != REC_READY) { if (loaded) *loaded = 0; return MAT_AIR; }
    if (loaded) *loaded = 1;
    const ChunkRec* r = &w->recs[ri];
    if (r->uniform != UNIFORM_NONE) return r->uniform;
    return w->host_pool[(size_t)r->slot * CHUNK_BYTES + GME_VOXEL_INDEX(x & 31, y & 31, z & 31)];
}

/* ---- validation ----------------------------------------------------------- */
#define FNV64_INIT  0xCBF29CE484222325ull
#define FNV64_PRIME 0x100000001B3ull

static uint64_t fnv1a64(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= FNV64_PRIME; }
    return h;
}

/* Reference block for ref_hash: chunk i of WORLD_REF_COLS^2 * layers, in
 * canonical order (x fastest, then z, then layer), around the anchor column. */
typedef struct RefCtx { const World* w; uint8_t* scratch; uint64_t* hash; } RefCtx;

static void ref_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    RefCtx* r = (RefCtx*)p;
    const World* w = r->w;
    const uint32_t n = WORLD_REF_COLS;
    uint8_t* buf = r->scratch + (size_t)wk * CHUNK_BYTES;
    for (uint32_t i = b; i < e; ++i) {
        const int32_t cx = w->ref_cx + (int32_t)(i % n) - (int32_t)(n / 2);
        const int32_t cz = w->ref_cz + (int32_t)((i / n) % n) - (int32_t)(n / 2);
        const int32_t cy = w->cfg.min_cy + (int32_t)(i / (n * n));
        GmeChunkMeta m;
        cpu_gen_chunk(cx, cy, cz, w->cfg.seed, &w->cfg.planet, &w->cfg.bio, buf, &m, NULL);
        r->hash[i] = fnv1a64(FNV64_INIT, buf, CHUNK_BYTES);
    }
}

static int ref_hash(const World* w, uint64_t* out, uint32_t* nchunks) {
    const uint32_t n = WORLD_REF_COLS * WORLD_REF_COLS * w->layers;
    RefCtx r;
    r.w = w;
    r.scratch = (uint8_t*)malloc((size_t)jobs_thread_count() * CHUNK_BYTES);
    r.hash = (uint64_t*)malloc(sizeof(uint64_t) * n);
    if (!r.scratch || !r.hash) { free(r.scratch); free(r.hash); return -1; }
    jobs_parallel_for(n, 1, ref_fn, &r);
    uint64_t h = FNV64_INIT;
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t le[8];
        for (int k = 0; k < 8; ++k) le[k] = (uint8_t)(r.hash[i] >> (8 * k));
        h = fnv1a64(h, le, 8);
    }
    free(r.scratch); free(r.hash);
    *out = h;
    *nchunks = n;
    return 0;
}

/* ---- density validation: column tiles ------------------------------------- */
typedef struct TileVal {
    uint32_t h_bad, coast_bad, sat, clamped, cols;
    double   dsum, dsq, dmax;
} TileVal;

typedef struct TileCtx {
    const World*   w;
    const int32_t* cxz;    /* 2 per sample */
    GmeColTile*    cpu;    /* per sample */
    TileVal*       val;    /* per sample, NULL = build only */
} TileCtx;

/* CPU tile of each sample, checked against the voxel truth: decoded h ==
 * gme_height and wet/dry == voxel wet/dry at all 36 x 36 columns; deviation
 * statistics and clamp share over the tile's own 32 x 32 columns. */
static void tile_val_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    TileCtx* c = (TileCtx*)p;
    const World* w = c->w;
    const GmePlanet* pl = &w->cfg.planet;
    const uint32_t seed = w->cfg.seed;
    for (uint32_t s = b; s < e; ++s) {
        const int32_t cx = c->cxz[2 * s], cz = c->cxz[2 * s + 1];
        GmeColTile* t = &c->cpu[s];
        gme_col_tile_build(t, cx, cz, seed, pl, &w->cfg.bio);
        if (!c->val) continue;
        TileVal v;
        memset(&v, 0, sizeof v);
        v.sat = (t->flags & GME_TILE_SATURATED) != 0;
        const int32_t x0 = cx * GME_CHUNK_DIM - GME_TILE_APRON, z0 = cz * GME_CHUNK_DIM - GME_TILE_APRON;
        for (int j = 0; j < GME_TILE_N; ++j)
            for (int i = 0; i < GME_TILE_N; ++i) {
                const int k = j * GME_TILE_N + i;
                const int32_t x = x0 + i, z = z0 + j;
                const int32_t h = gme_height(x, z, seed, pl);
                v.h_bad += gme_tile_h(t, k) != h;
                v.coast_bad += (gme_tile_t_q8(t, k) < 256 * (pl->sea_level + 1)) != (h < pl->sea_level);
                if (i >= GME_TILE_APRON && i < GME_TILE_APRON + GME_CHUNK_DIM &&
                    j >= GME_TILE_APRON && j < GME_TILE_APRON + GME_CHUNK_DIM) {
                    int32_t hh, traw;
                    gme_surface_pair(x, z, seed, pl, &hh, &traw);
                    const int32_t dev = gme_tile_dev(t, k);
                    v.clamped += ((traw - 256 * (hh + 1) + 1) >> 1) != dev;
                    const double d = (double)dev / GME_DEV_UNIT;
                    v.dsum += d;
                    v.dsq += d * d;
                    if (fabs(d) > v.dmax) v.dmax = fabs(d);
                    v.cols++;
                }
            }
        c->val[s] = v;
    }
}

/* Up to `want` columns of the loaded area, half nearest the viewer and half at
 * the edge; GPU mode takes only columns with a published tile. */
static uint32_t pick_tile_columns(const World* w, uint32_t want, int32_t* cxz) {
    const uint32_t nc = col_count(w, w->radius);
    uint32_t n = 0, i = 0;
    for (; i < nc && n < want / 2; ++i) {
        const int32_t cx = w->vcx + w->cols[i].dx, cz = w->vcz + w->cols[i].dz;
        if (w->tile_tag && w->tile_tag[gme_page_index(cx, 0, cz, w->page_n, 1)] != gme_tile_key(cx, cz)) continue;
        cxz[2 * n] = cx; cxz[2 * n + 1] = cz; ++n;
    }
    for (uint32_t k = nc; k-- > i && n < want;) {
        const int32_t cx = w->vcx + w->cols[k].dx, cz = w->vcz + w->cols[k].dz;
        if (w->tile_tag && w->tile_tag[gme_page_index(cx, 0, cz, w->page_n, 1)] != gme_tile_key(cx, cz)) continue;
        cxz[2 * n] = cx; cxz[2 * n + 1] = cz; ++n;
    }
    return n;
}

static int validate_density(World* w, uint32_t want, ValidateReport* rep) {
    const uint32_t nref = WORLD_REF_COLS * WORLD_REF_COLS;
    int32_t* cxz = (int32_t*)malloc(sizeof(int32_t) * 2 * (want + nref));
    GmeColTile* cpu = (GmeColTile*)malloc(sizeof(GmeColTile) * (want + nref));
    TileVal* val = (TileVal*)calloc(want + 1, sizeof(TileVal));
    if (!cxz || !cpu || !val) { free(cxz); free(cpu); free(val); return -1; }

    /* sampled tiles vs the voxel truth (and vs the GPU's records) */
    const uint32_t np = pick_tile_columns(w, want, cxz);
    TileCtx c = { w, cxz, cpu, val };
    jobs_parallel_for(np, 1, tile_val_fn, &c);
    double dsum = 0, dsq = 0, dmax = 0;
    uint64_t cols = 0, clamped = 0;
    for (uint32_t s = 0; s < np; ++s) {
        rep->tile_h_bad += val[s].h_bad;
        rep->tile_coast_bad += val[s].coast_bad;
        rep->tile_sat += val[s].sat;
        dsum += val[s].dsum; dsq += val[s].dsq;
        if (val[s].dmax > dmax) dmax = val[s].dmax;
        cols += val[s].cols; clamped += val[s].clamped;
        if (w->use_gpu) {
            GmeColTile g;
            const uint32_t ti = gme_page_index(cxz[2 * s], 0, cxz[2 * s + 1], w->page_n, 1);
            if (gpu_read_tile(w->gpu, ti, &g)) { rep->tile_mismatch += sizeof g; continue; }
            const uint8_t* a = (const uint8_t*)&g;
            const uint8_t* b = (const uint8_t*)&cpu[s];
            for (size_t k = 0; k < sizeof g; ++k) rep->tile_mismatch += a[k] != b[k];
            rep->tiles_checked++;
        }
    }
    rep->tiles_cpu = np;
    rep->dev_mean = cols ? dsum / (double)cols : 0;
    rep->dev_rms = cols ? sqrt(dsq / (double)cols) : 0;
    rep->dev_max = dmax;
    rep->clamp_share = cols ? (double)clamped / (double)cols : 0;

    /* dens_hash: the CPU tiles of the reference block, x fastest */
    for (uint32_t i = 0; i < nref; ++i) {
        cxz[2 * i] = w->ref_cx + (int32_t)(i % WORLD_REF_COLS) - (int32_t)(WORLD_REF_COLS / 2);
        cxz[2 * i + 1] = w->ref_cz + (int32_t)(i / WORLD_REF_COLS) - (int32_t)(WORLD_REF_COLS / 2);
    }
    TileCtx r = { w, cxz, cpu, NULL };
    jobs_parallel_for(nref, 1, tile_val_fn, &r);
    uint64_t h = FNV64_INIT;
    for (uint32_t i = 0; i < nref; ++i) h = fnv1a64(h, (const uint8_t*)&cpu[i], sizeof(GmeColTile));
    rep->dens_hash = h;
    free(cxz); free(cpu); free(val);
    return 0;
}

typedef struct ValCtx {
    World* w;
    const uint32_t* picks;
    uint8_t* ref;          /* picks * CHUNK_BYTES */
    uint64_t* host_bad;    /* per pick */
    uint8_t* meta_bad;     /* per pick */
    uint32_t* flora_cols;  /* per pick: columns with vegetation */
    uint32_t* flora_vox;   /* per pick: flora voxels */
} ValCtx;

static void val_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    ValCtx* v = (ValCtx*)p;
    for (uint32_t i = b; i < e; ++i) {
        const ChunkRec* r = &v->w->recs[v->picks[i]];
        uint8_t* ref = v->ref + (size_t)i * CHUNK_BYTES;
        GmeChunkMeta m;
        cpu_gen_chunk(r->cx, r->cy, r->cz, v->w->cfg.seed, &v->w->cfg.planet, &v->w->cfg.bio, ref, &m,
                      &v->flora_cols[i]);
        uint32_t fv = 0;
        for (size_t k = 0; k < CHUNK_BYTES; ++k) fv += (uint32_t)gme_is_flora(ref[k]);
        v->flora_vox[i] = fv;
        uint64_t bad = 0;
        if (r->uniform != UNIFORM_NONE) {
            for (size_t k = 0; k < CHUNK_BYTES; ++k) bad += ref[k] != r->uniform;
        } else {
            const uint8_t* h = v->w->host_pool + (size_t)r->slot * CHUNK_BYTES;
            for (size_t k = 0; k < CHUNK_BYTES; ++k) bad += ref[k] != h[k];
        }
        v->host_bad[i] = bad;
        v->meta_bad[i] = (uint8_t)(((m.mat_min == m.mat_max) != (r->uniform != UNIFORM_NONE)) ||
                                   m.non_air != r->non_air);
    }
}

int world_validate(World* w, uint32_t samples, ValidateReport* rep) {
    memset(rep, 0, sizeof *rep);
    if (world_flush(w)) return -1;

    /* pick half uniform, half non-uniform, spread across the record table */
    const uint32_t want_u = samples / 2, want_n = samples - want_u;
    uint32_t nu = w->n_uniform_ready, nn = w->n_ready - w->n_uniform_ready;
    const uint32_t step_u = nu > want_u ? nu / want_u : 1, step_n = nn > want_n ? nn / want_n : 1;
    uint32_t* picks = (uint32_t*)malloc(sizeof(uint32_t) * (samples + 1));
    uint32_t np = 0, cu = 0, cn = 0, tu = 0, tn = 0;
    for (uint32_t i = 0; picks && i < w->rec_cap && np < samples; ++i) {
        const ChunkRec* r = &w->recs[i];
        if (r->state != REC_READY) continue;
        if (r->uniform != UNIFORM_NONE) { if (cu++ % step_u == 0 && tu < want_u) { picks[np++] = i; ++tu; } }
        else                            { if (cn++ % step_n == 0 && tn < want_n) { picks[np++] = i; ++tn; } }
    }

    ValCtx v;
    v.w = w;
    v.picks = picks;
    v.ref = (uint8_t*)malloc((size_t)np * CHUNK_BYTES + 1);
    v.host_bad = (uint64_t*)calloc(np + 1, sizeof(uint64_t));
    v.meta_bad = (uint8_t*)calloc(np + 1, 1);
    v.flora_cols = (uint32_t*)calloc(np + 1, sizeof(uint32_t));
    v.flora_vox = (uint32_t*)calloc(np + 1, sizeof(uint32_t));
    uint8_t* dev = (uint8_t*)malloc(CHUNK_BYTES);
    if (!picks || !v.ref || !v.host_bad || !v.meta_bad || !v.flora_cols || !v.flora_vox || !dev) {
        fprintf(stderr, "world: out of memory for validation\n");
        free(dev); free(v.ref); free(v.host_bad); free(v.meta_bad); free(v.flora_cols); free(v.flora_vox); free(picks);
        return -1;
    }
    jobs_parallel_for(np, 1, val_fn, &v);

    for (uint32_t i = 0; i < np; ++i) {
        const ChunkRec* r = &w->recs[picks[i]];
        rep->chunks_checked++;
        rep->voxels_checked += CHUNK_BYTES;
        rep->host_mismatch += v.host_bad[i];
        rep->meta_mismatch += v.meta_bad[i];
        rep->flora_checked += v.flora_cols[i];
        rep->flora_voxels += v.flora_vox[i];
        if (r->uniform != UNIFORM_NONE) { rep->uniform_checked++; continue; }
        if (w->use_gpu) {
            if (gpu_read_slot(w->gpu, r->slot, dev)) { rep->device_mismatch += CHUNK_BYTES; continue; }
            const uint8_t* ref = v.ref + (size_t)i * CHUNK_BYTES;
            for (size_t k = 0; k < CHUNK_BYTES; ++k) rep->device_mismatch += dev[k] != ref[k];
        }
    }

    /* gameplay-style query: surface found through world_get_voxel must equal
     * the analytic terrain height for columns inside the loaded area */
    const int32_t span = (w->radius > 2 ? w->radius - 2 : 1) * GME_CHUNK_DIM;
    const int32_t ymax = w->cfg.max_cy * GME_CHUNK_DIM - 1, ymin = w->cfg.min_cy * GME_CHUNK_DIM;
    uint32_t rng = 0x12345678u;
    for (int i = 0; i < 4096; ++i) {
        rng = rng * 1664525u + 1013904223u; const int32_t ox = (int32_t)(rng >> 8) % (2 * span) - span;
        rng = rng * 1664525u + 1013904223u; const int32_t oz = (int32_t)(rng >> 8) % (2 * span) - span;
        if ((int64_t)ox * ox + (int64_t)oz * oz > (int64_t)span * span) continue;
        const int32_t x = w->vcx * GME_CHUNK_DIM + ox, z = w->vcz * GME_CHUNK_DIM + oz;
        int32_t top = ymin - 1, ok = 1;
        for (int32_t y = ymax; y >= ymin; --y) {
            int loaded;
            const uint8_t m = world_get_voxel(w, x, y, z, &loaded);
            if (!loaded) { ok = 0; break; }
            /* standing flora (leaves, trunks, kelp) is above the ground; grass
             * and mats replace the ground voxel, so the surface stays at h */
            if (m != MAT_AIR && m != w->cfg.planet.ocean_mat && m != w->cfg.planet.ocean_top_mat &&
                !gme_is_standing_flora(m)) { top = y; break; }
        }
        if (!ok) continue;
        rep->columns_checked++;
        if (top != gme_height(x, z, w->cfg.seed, &w->cfg.planet)) rep->column_mismatch++;
    }

    free(dev); free(v.ref); free(v.host_bad); free(v.meta_bad); free(v.flora_cols); free(v.flora_vox); free(picks);
    const int ref_bad = ref_hash(w, &rep->ref_hash, &rep->ref_chunks);
    const int vacuous = w->cfg.bio.cells && rep->flora_checked == 0;   /* a biomap but no vegetation tested */
    /* smooth-surface tiles: 256 sampled (half near, half far) + dens_hash */
    const int dens_bad = validate_density(w, 256, rep) != 0 || rep->tile_mismatch || rep->tile_h_bad ||
                         rep->tile_coast_bad || rep->tile_sat || rep->tiles_cpu == 0 ||
                         (w->use_gpu && rep->tiles_checked == 0);
    return (rep->host_mismatch || rep->device_mismatch || rep->meta_mismatch || rep->column_mismatch ||
            rep->chunks_checked == 0 || ref_bad || vacuous || dens_bad) ? 1 : 0;
}
