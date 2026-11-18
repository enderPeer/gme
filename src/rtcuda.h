/* rtcuda.h - run-time CUDA: loads the driver (libcuda / nvcuda.dll) and NVRTC
 * dynamically and compiles kernels from source at start-up. No CUDA toolkit
 * or headers are needed to build or run, so the same C code runs on the
 * cluster rigs (which have the driver + NVRTC but no nvcc). */
#ifndef GME_RTCUDA_H
#define GME_RTCUDA_H

#include <stddef.h>
#include <stdint.h>

typedef unsigned long long rt_dptr;  /* CUdeviceptr */
typedef struct RtModule RtModule;
typedef void* RtFunc;

typedef struct RtDevice {
    int  ordinal;
    char name[128];
    int  cc_major, cc_minor, sm_count;
    size_t total_mem;
} RtDevice;

int         rt_init(int device, RtDevice* out);        /* 0 ok */
const char* rt_error(void);
int         rt_device_count(void);

RtModule*   rt_compile(const char* source, const char* name);   /* NULL + rt_error() on failure */
RtFunc      rt_func(RtModule* m, const char* kernel);
void        rt_module_free(RtModule* m);

int  rt_alloc(rt_dptr* p, size_t bytes);
void rt_free(rt_dptr p);
int  rt_upload(rt_dptr dst, const void* src, size_t bytes);
int  rt_download(void* dst, rt_dptr src, size_t bytes);
int  rt_memset32(rt_dptr dst, uint32_t value, size_t count);
int  rt_launch(RtFunc f, unsigned grid, unsigned block, unsigned shared, void** args);
int  rt_sync(void);
int  rt_copy(rt_dptr dst, rt_dptr src, size_t bytes);          /* device to device */
int  rt_mem_info(size_t* free_bytes, size_t* total_bytes);     /* cuMemGetInfo of the current context */
/* cuFuncGetAttribute: 0 max threads/block, 1 shared bytes, 3 local bytes, 4 registers */
int  rt_func_attr(RtFunc f, int attr, int* value);

#endif
