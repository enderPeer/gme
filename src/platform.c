/* platform.c */
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

static double g_inv_freq;

void platform_init(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_inv_freq = 1.0 / (double)f.QuadPart;
    timeBeginPeriod(1); /* 1 ms Sleep granularity for frame pacing */
}

void platform_shutdown(void) { timeEndPeriod(1); }

double time_now(void) {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * g_inv_freq;
}

void sleep_ms(int ms) { Sleep((DWORD)ms); }

void* os_alloc(size_t bytes) {
    return VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void os_free(void* p, size_t bytes) {
    (void)bytes;
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

static void cpu_brand(char* out, size_t n) {
    int r[4];
    char b[49];
    memset(b, 0, sizeof b);
    __cpuid(r, (int)0x80000000);
    if ((unsigned)r[0] >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) {
            __cpuid(r, (int)(0x80000002u + (unsigned)i));
            memcpy(b + 16 * i, r, 16);
        }
    }
    const char* s = b;
    while (*s == ' ') ++s;
    snprintf(out, n, "%s", *s ? s : "unknown CPU");
}

void sys_query(SysInfo* s) {
    memset(s, 0, sizeof *s);
    cpu_brand(s->cpu_name, sizeof s->cpu_name);
    s->logical_cpus = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    uint8_t* buf = (uint8_t*)malloc(len);
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        int max_class = 0, cores = 0;
        for (DWORD off = 0; off < len;) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX it = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
            if (it->Processor.EfficiencyClass > max_class) max_class = it->Processor.EfficiencyClass;
            ++cores;
            off += it->Size;
        }
        s->physical_cores = cores;
        if (max_class > 0) { /* hybrid: highest class = performance cores */
            for (DWORD off = 0; off < len;) {
                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX it = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
                if (it->Processor.EfficiencyClass == max_class) ++s->perf_cores; else ++s->eff_cores;
                off += it->Size;
            }
        }
    }
    free(buf);
    if (!s->physical_cores) s->physical_cores = s->logical_cpus;

    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        s->total_ram = ms.ullTotalPhys;
        s->avail_ram = ms.ullAvailPhys;
    }
}

#else /* POSIX */
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

void platform_init(void) {}
void platform_shutdown(void) {}

double time_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

void sleep_ms(int ms) {
    struct timespec t = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&t, NULL);
}

void* os_alloc(size_t bytes) {
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

void os_free(void* p, size_t bytes) { if (p) munmap(p, bytes); }

void sys_query(SysInfo* s) {
    memset(s, 0, sizeof *s);
    snprintf(s->cpu_name, sizeof s->cpu_name, "unknown CPU");
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "model name", 10)) {
                char* c = strchr(line, ':');
                if (c) { c += 2; c[strcspn(c, "\n")] = 0; snprintf(s->cpu_name, sizeof s->cpu_name, "%s", c); }
                break;
            }
        }
        fclose(f);
    }
    s->logical_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    s->physical_cores = s->logical_cpus;
    long pg = sysconf(_SC_PAGESIZE);
    s->total_ram = (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)pg;
    s->avail_ram = (uint64_t)sysconf(_SC_AVPHYS_PAGES) * (uint64_t)pg;
}
#endif
