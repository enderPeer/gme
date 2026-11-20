/* gpu.h - CUDA backend: device-resident chunk pool + batched generation. */
#ifndef GME_GPU_H
#define GME_GPU_H

#include <stddef.h>
#include <stdint.h>
#include "worldgen.h"
#include "flora.h"
#include "density.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GPU_BUFFERS 2  /* batches in flight (one generating, one draining) */

typedef struct GpuInfo {
    int    available;
    char   name[256];
    int    cc_major, cc_minor;
    int    sm_count;
    size_t total_mem, free_mem;
} GpuInfo;

typedef struct GpuWorld GpuWorld;

int         gpu_query(GpuInfo* out);
const char* gpu_last_error(void);

/* slot_count chunks of device storage + GPU_BUFFERS staging batches +
 * tile_count column tiles (density.h; the world's toroidal page_n^2 table). */
GpuWorld*     gpu_world_create(uint32_t slot_count, uint32_t max_batch, uint32_t tile_count);
void          gpu_world_destroy(GpuWorld* g);
size_t        gpu_world_device_bytes(const GpuWorld* g);
size_t        gpu_world_pinned_bytes(const GpuWorld* g);
const uint8_t* gpu_world_pool(const GpuWorld* g); /* device pointer, slot * 32 KB */
const GmeColTile* gpu_world_tiles(const GpuWorld* g); /* device pointer, tile_count records */

/* Biomap cells for generation and rendering: copies n packed cells to the
 * device (replacing any previous ones; cells == NULL or n == 0 frees them). 0 ok. */
int             gpu_world_set_biomap(GpuWorld* g, const uint32_t* cells, size_t n);
const uint32_t* gpu_world_biomap(const GpuWorld* g); /* device pointer or NULL */

/* Pinned host staging for batch `buf`. Write jobs, then gpu_submit. After
 * completion, meta[i] is filled and voxels[i] holds chunk i if it is not
 * uniform (uniform chunks are not transferred). */
GmeGenJob*    gpu_staging_jobs(GpuWorld* g, int buf);
GmeChunkMeta* gpu_staging_meta(GpuWorld* g, int buf);
uint8_t*      gpu_staging_voxels(GpuWorld* g, int buf);
GmeTileJob*   gpu_staging_tiles(GpuWorld* g, int buf);   /* max_batch tile jobs */

/* Generates `count` chunk jobs and `ntiles` tile jobs (k_col_tiles writes
 * tile job k to record jobs[k].index; it runs first on the batch stream).
 * bio: host view of the vegetation (NULL = none). Its cells are replaced by the
 * ones uploaded with gpu_world_set_biomap, which must match. 0 ok. */
int gpu_submit(GpuWorld* g, int buf, uint32_t count, uint32_t ntiles, uint32_t seed, const GmePlanet* planet,
               const GmeBioView* bio);
int gpu_poll(GpuWorld* g, int buf);   /* 1 done, 0 running, -1 error */
/* M4: a cudaEvent_t that every later batch waits for on the GPU before it writes
 * the pool or the tile table (the renderer's frame in flight); NULL = none */
void gpu_world_set_fence(GpuWorld* g, void* ev);
int gpu_wait(GpuWorld* g, int buf);   /* 0 ok */

/* Synchronous copy of one device slot (validation / debugging). */
int gpu_read_slot(GpuWorld* g, uint32_t slot, uint8_t* dst);
/* Synchronous copy of one tile record (validation). 0 ok. */
int gpu_read_tile(GpuWorld* g, uint32_t index, GmeColTile* dst);

/* Self-test D1: builds n tiles with k_col_tiles outside any world (jobs[k].index
 * is ignored; tile k goes to out[k]). bio: host view, its ncells cells are
 * uploaded for the call. 0 ok. */
int gpu_build_tiles(const GmeTileJob* jobs, uint32_t n, uint32_t seed, const GmePlanet* planet,
                    const GmeBioView* bio, size_t ncells, GmeColTile* out);

#ifdef __cplusplus
}
#endif
#endif
