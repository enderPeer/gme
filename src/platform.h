/* platform.h - OS abstraction: hardware query, time, memory, atomics. */
#ifndef GME_PLATFORM_H
#define GME_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

typedef struct SysInfo {
    char     cpu_name[128];
    int      logical_cpus;   /* hardware threads */
    int      physical_cores;
    int      perf_cores;     /* hybrid CPUs: P-cores (0 if not hybrid) */
    int      eff_cores;      /* hybrid CPUs: E-cores */
    uint64_t total_ram;
    uint64_t avail_ram;
} SysInfo;

void   platform_init(void);
void   platform_shutdown(void);
void   sys_query(SysInfo* out);
double time_now(void);               /* seconds, monotonic */
void   sleep_ms(int ms);

/* Large page-backed allocation. Zero-filled; physical pages are only
 * consumed when first touched, so reserving a big pool is cheap. */
void*  os_alloc(size_t bytes);
void   os_free(void* p, size_t bytes);

/* 32-bit atomics (sequentially consistent). */
#if defined(_MSC_VER)
#include <intrin.h>
static inline int32_t atomic_fetch_add_i32(volatile int32_t* p, int32_t v) {
    return (int32_t)_InterlockedExchangeAdd((volatile long*)p, (long)v);
}
#else
static inline int32_t atomic_fetch_add_i32(volatile int32_t* p, int32_t v) {
    return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}
#endif

#endif
