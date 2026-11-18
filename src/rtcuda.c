/* rtcuda.c */
#include "rtcuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE lib_t;
static lib_t lib_open(const char* n) { return LoadLibraryA(n); }
static void* lib_sym(lib_t l, const char* s) { return (void*)GetProcAddress(l, s); }
#define CUAPI __stdcall
#else
#include <dlfcn.h>
typedef void* lib_t;
static lib_t lib_open(const char* n) { return dlopen(n, RTLD_NOW | RTLD_GLOBAL); }
static void* lib_sym(lib_t l, const char* s) { return dlsym(l, s); }
#define CUAPI
#endif

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef int nvrtcResult;
typedef void* nvrtcProgram;

static struct {
    lib_t cu, rtc;
    CUcontext ctx;
    CUdevice dev;
    CUresult (CUAPI *Init)(unsigned);
    CUresult (CUAPI *DeviceGetCount)(int*);
    CUresult (CUAPI *DeviceGet)(CUdevice*, int);
    CUresult (CUAPI *DeviceGetName)(char*, int, CUdevice);
    CUresult (CUAPI *DeviceGetAttribute)(int*, int, CUdevice);
    CUresult (CUAPI *DeviceTotalMem)(size_t*, CUdevice);
    CUresult (CUAPI *PrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (CUAPI *CtxSetCurrent)(CUcontext);
    CUresult (CUAPI *CtxSynchronize)(void);
    CUresult (CUAPI *ModuleLoadData)(CUmodule*, const void*);
    CUresult (CUAPI *ModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (CUAPI *ModuleUnload)(CUmodule);
    CUresult (CUAPI *MemAlloc)(rt_dptr*, size_t);
    CUresult (CUAPI *MemFree)(rt_dptr);
    CUresult (CUAPI *MemcpyHtoD)(rt_dptr, const void*, size_t);
    CUresult (CUAPI *MemcpyDtoH)(void*, rt_dptr, size_t);
    CUresult (CUAPI *MemsetD32)(rt_dptr, unsigned, size_t);
    CUresult (CUAPI *MemGetInfo)(size_t*, size_t*);
    CUresult (CUAPI *MemcpyDtoD)(rt_dptr, rt_dptr, size_t);
    CUresult (CUAPI *FuncGetAttribute)(int*, int, CUfunction);
    CUresult (CUAPI *LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   unsigned, void*, void**, void**);
    CUresult (CUAPI *GetErrorString)(CUresult, const char**);
    nvrtcResult (*CreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
    nvrtcResult (*CompileProgram)(nvrtcProgram, int, const char* const*);
    nvrtcResult (*GetProgramLogSize)(nvrtcProgram, size_t*);
    nvrtcResult (*GetProgramLog)(nvrtcProgram, char*);
    nvrtcResult (*GetCUBINSize)(nvrtcProgram, size_t*);
    nvrtcResult (*GetCUBIN)(nvrtcProgram, char*);
    nvrtcResult (*DestroyProgram)(nvrtcProgram*);
    const char* (*GetErrorStringRtc)(nvrtcResult);
    int cc_major, cc_minor;
} R;

struct RtModule { CUmodule mod; };

static char g_err[1024];
const char* rt_error(void) { return g_err; }

static int cu_check(CUresult r, const char* what) {
    if (r == 0) return 0;
    const char* s = "?";
    if (R.GetErrorString) R.GetErrorString(r, &s);
    snprintf(g_err, sizeof g_err, "%s failed: %s (%d)", what, s, r);
    return -1;
}

static void* sym2(lib_t l, const char* a, const char* b) {
    void* p = lib_sym(l, a);
    return p ? p : (b ? lib_sym(l, b) : NULL);
}

static lib_t open_nvrtc(void) {
    const char* env = getenv("GME_NVRTC");
    if (env) { lib_t l = lib_open(env); if (l) return l; }
#ifdef _WIN32
    static const char* names[] = { "nvrtc64_130_0.dll", "nvrtc64_120_0.dll" };
    for (int i = 0; i < 2; ++i) { lib_t l = lib_open(names[i]); if (l) return l; }
    const char* cp = getenv("CUDA_PATH");
    if (cp) {
        char p[1024];
        for (int i = 0; i < 2; ++i) {
            snprintf(p, sizeof p, "%s\\bin\\x64\\%s", cp, names[i]);
            lib_t l = lib_open(p); if (l) return l;
            snprintf(p, sizeof p, "%s\\bin\\%s", cp, names[i]);
            l = lib_open(p); if (l) return l;
        }
    }
#else
    static const char* names[] = { "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so" };
    for (int i = 0; i < 3; ++i) { lib_t l = lib_open(names[i]); if (l) return l; }
#endif
    return NULL;
}

#define LOAD(lib, field, n1, n2) do { *(void**)&R.field = sym2(lib, n1, n2); \
    if (!R.field) { snprintf(g_err, sizeof g_err, "missing symbol %s", n1); return -1; } } while (0)

static int load_libs(void) {
    if (R.cu) return 0;
#ifdef _WIN32
    R.cu = lib_open("nvcuda.dll");
#else
    R.cu = lib_open("libcuda.so.1");
#endif
    if (!R.cu) { snprintf(g_err, sizeof g_err, "NVIDIA driver library not found"); return -1; }
    R.rtc = open_nvrtc();
    if (!R.rtc) { snprintf(g_err, sizeof g_err, "NVRTC not found (set GME_NVRTC to libnvrtc's path)"); return -1; }
    LOAD(R.cu, Init, "cuInit", NULL);
    LOAD(R.cu, DeviceGetCount, "cuDeviceGetCount", NULL);
    LOAD(R.cu, DeviceGet, "cuDeviceGet", NULL);
    LOAD(R.cu, DeviceGetName, "cuDeviceGetName", NULL);
    LOAD(R.cu, DeviceGetAttribute, "cuDeviceGetAttribute", NULL);
    LOAD(R.cu, DeviceTotalMem, "cuDeviceTotalMem_v2", "cuDeviceTotalMem");
    LOAD(R.cu, PrimaryCtxRetain, "cuDevicePrimaryCtxRetain", NULL);
    LOAD(R.cu, CtxSetCurrent, "cuCtxSetCurrent", NULL);
    LOAD(R.cu, CtxSynchronize, "cuCtxSynchronize", "cuCtxSynchronize_v2");
    LOAD(R.cu, ModuleLoadData, "cuModuleLoadData", NULL);
    LOAD(R.cu, ModuleGetFunction, "cuModuleGetFunction", NULL);
    LOAD(R.cu, ModuleUnload, "cuModuleUnload", NULL);
    LOAD(R.cu, MemAlloc, "cuMemAlloc_v2", "cuMemAlloc");
    LOAD(R.cu, MemFree, "cuMemFree_v2", "cuMemFree");
    LOAD(R.cu, MemcpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    LOAD(R.cu, MemcpyDtoH, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    LOAD(R.cu, MemsetD32, "cuMemsetD32_v2", "cuMemsetD32");
    LOAD(R.cu, MemGetInfo, "cuMemGetInfo_v2", "cuMemGetInfo");
    LOAD(R.cu, MemcpyDtoD, "cuMemcpyDtoD_v2", "cuMemcpyDtoD");
    LOAD(R.cu, FuncGetAttribute, "cuFuncGetAttribute", NULL);
    LOAD(R.cu, LaunchKernel, "cuLaunchKernel", NULL);
    LOAD(R.cu, GetErrorString, "cuGetErrorString", NULL);
    LOAD(R.rtc, CreateProgram, "nvrtcCreateProgram", NULL);
    LOAD(R.rtc, CompileProgram, "nvrtcCompileProgram", NULL);
    LOAD(R.rtc, GetProgramLogSize, "nvrtcGetProgramLogSize", NULL);
    LOAD(R.rtc, GetProgramLog, "nvrtcGetProgramLog", NULL);
    LOAD(R.rtc, GetCUBINSize, "nvrtcGetCUBINSize", NULL);
    LOAD(R.rtc, GetCUBIN, "nvrtcGetCUBIN", NULL);
    LOAD(R.rtc, DestroyProgram, "nvrtcDestroyProgram", NULL);
    LOAD(R.rtc, GetErrorStringRtc, "nvrtcGetErrorString", NULL);
    return 0;
}

int rt_device_count(void) {
    int n = 0;
    if (load_libs() || cu_check(R.Init(0), "cuInit")) return 0;
    if (R.DeviceGetCount(&n)) return 0;
    return n;
}

int rt_init(int device, RtDevice* out) {
    if (load_libs()) return -1;
    if (cu_check(R.Init(0), "cuInit")) return -1;
    if (cu_check(R.DeviceGet(&R.dev, device), "cuDeviceGet")) return -1;
    if (cu_check(R.PrimaryCtxRetain(&R.ctx, R.dev), "cuDevicePrimaryCtxRetain")) return -1;
    if (cu_check(R.CtxSetCurrent(R.ctx), "cuCtxSetCurrent")) return -1;
    R.DeviceGetAttribute(&R.cc_major, 75, R.dev); /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
    R.DeviceGetAttribute(&R.cc_minor, 76, R.dev);
    if (out) {
        memset(out, 0, sizeof *out);
        out->ordinal = device;
        R.DeviceGetName(out->name, (int)sizeof out->name, R.dev);
        out->cc_major = R.cc_major;
        out->cc_minor = R.cc_minor;
        R.DeviceGetAttribute(&out->sm_count, 16, R.dev); /* MULTIPROCESSOR_COUNT */
        R.DeviceTotalMem(&out->total_mem, R.dev);
    }
    return 0;
}

RtModule* rt_compile(const char* source, const char* name) {
    nvrtcProgram prog;
    if (R.CreateProgram(&prog, source, name, 0, NULL, NULL)) { snprintf(g_err, sizeof g_err, "nvrtcCreateProgram failed"); return NULL; }
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", R.cc_major, R.cc_minor);
    const char* opts[] = { arch, "-std=c++17", "--use_fast_math", "-default-device" };
    const nvrtcResult cr = R.CompileProgram(prog, 4, opts);
    if (cr) {
        size_t n = 0;
        R.GetProgramLogSize(prog, &n);
        char* log = (char*)malloc(n + 1);
        if (log) { R.GetProgramLog(prog, log); log[n] = 0; }
        snprintf(g_err, sizeof g_err, "NVRTC compile of %s failed: %s\n%s", name, R.GetErrorStringRtc(cr), log ? log : "");
        free(log);
        R.DestroyProgram(&prog);
        return NULL;
    }
    size_t n = 0;
    R.GetCUBINSize(prog, &n);
    char* bin = (char*)malloc(n);
    R.GetCUBIN(prog, bin);
    R.DestroyProgram(&prog);
    RtModule* m = (RtModule*)calloc(1, sizeof *m);
    const CUresult lr = R.ModuleLoadData(&m->mod, bin);
    free(bin);
    if (cu_check(lr, "cuModuleLoadData")) { free(m); return NULL; }
    return m;
}

RtFunc rt_func(RtModule* m, const char* kernel) {
    CUfunction f = NULL;
    if (cu_check(R.ModuleGetFunction(&f, m->mod, kernel), kernel)) return NULL;
    return f;
}

void rt_module_free(RtModule* m) { if (m) { R.ModuleUnload(m->mod); free(m); } }
int  rt_alloc(rt_dptr* p, size_t bytes) { return cu_check(R.MemAlloc(p, bytes), "cuMemAlloc"); }
void rt_free(rt_dptr p) { if (p) R.MemFree(p); }
int  rt_upload(rt_dptr d, const void* s, size_t n) { return cu_check(R.MemcpyHtoD(d, s, n), "cuMemcpyHtoD"); }
int  rt_download(void* d, rt_dptr s, size_t n) { return cu_check(R.MemcpyDtoH(d, s, n), "cuMemcpyDtoH"); }
int  rt_memset32(rt_dptr d, uint32_t v, size_t n) { return cu_check(R.MemsetD32(d, v, n), "cuMemsetD32"); }
int  rt_sync(void) { return cu_check(R.CtxSynchronize(), "cuCtxSynchronize"); }
int  rt_copy(rt_dptr d, rt_dptr s, size_t n) { return cu_check(R.MemcpyDtoD(d, s, n), "cuMemcpyDtoD"); }
int  rt_func_attr(RtFunc f, int attr, int* value) { return cu_check(R.FuncGetAttribute(value, attr, (CUfunction)f), "cuFuncGetAttribute"); }
int  rt_mem_info(size_t* free_b, size_t* total_b) {
    size_t f = 0, t = 0;
    if (cu_check(R.MemGetInfo(&f, &t), "cuMemGetInfo")) return -1;
    if (free_b) *free_b = f;
    if (total_b) *total_b = t;
    return 0;
}

int rt_launch(RtFunc f, unsigned grid, unsigned block, unsigned shared, void** args) {
    return cu_check(R.LaunchKernel((CUfunction)f, grid, 1, 1, block, 1, 1, shared, NULL, args, NULL), "cuLaunchKernel");
}
