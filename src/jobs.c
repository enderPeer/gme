/* jobs.c */
#include "jobs.h"
#include "platform.h"
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef SRWLOCK            lock_t;
typedef CONDITION_VARIABLE cond_t;
typedef HANDLE             thread_t;
#define LOCK(l)      AcquireSRWLockExclusive(l)
#define UNLOCK(l)    ReleaseSRWLockExclusive(l)
#define WAIT(c, l)   SleepConditionVariableSRW(c, l, INFINITE, 0)
#define WAKE_ALL(c)  WakeAllConditionVariable(c)
#define THREAD_FN    DWORD WINAPI
#else
#include <pthread.h>
typedef pthread_mutex_t lock_t;
typedef pthread_cond_t  cond_t;
typedef pthread_t       thread_t;
#define LOCK(l)      pthread_mutex_lock(l)
#define UNLOCK(l)    pthread_mutex_unlock(l)
#define WAIT(c, l)   pthread_cond_wait(c, l)
#define WAKE_ALL(c)  pthread_cond_broadcast(c)
#define THREAD_FN    void*
#endif

static struct {
    lock_t   lock;
    cond_t   wake, done;
    thread_t* threads;
    int      nthreads;
    int      quit;
    uint32_t generation;
    int      pending;           /* workers still inside the current task */
    JobFn    fn;
    void*    ctx;
    uint32_t count, grain;
    volatile int32_t next;
} J;

static void run_task(uint32_t worker) {
    for (;;) {
        int32_t b = atomic_fetch_add_i32(&J.next, (int32_t)J.grain);
        if ((uint32_t)b >= J.count) break;
        uint32_t e = (uint32_t)b + J.grain;
        if (e > J.count) e = J.count;
        J.fn(J.ctx, (uint32_t)b, e, worker);
    }
}

static THREAD_FN worker_main(void* arg) {
    uint32_t id = (uint32_t)(uintptr_t)arg;
    uint32_t seen = 0;
    for (;;) {
        LOCK(&J.lock);
        while (J.generation == seen && !J.quit) WAIT(&J.wake, &J.lock);
        if (J.quit) { UNLOCK(&J.lock); break; }
        seen = J.generation;
        UNLOCK(&J.lock);

        run_task(id);

        LOCK(&J.lock);
        if (--J.pending == 0) WAKE_ALL(&J.done);
        UNLOCK(&J.lock);
    }
    return 0;
}

int jobs_init(int worker_threads) {
    if (worker_threads < 0) worker_threads = 0;
#ifdef _WIN32
    InitializeSRWLock(&J.lock);
    InitializeConditionVariable(&J.wake);
    InitializeConditionVariable(&J.done);
#else
    pthread_mutex_init(&J.lock, NULL);
    pthread_cond_init(&J.wake, NULL);
    pthread_cond_init(&J.done, NULL);
#endif
    J.threads = (thread_t*)calloc((size_t)worker_threads + 1, sizeof(thread_t));
    if (!J.threads) return -1;
    for (int i = 0; i < worker_threads; ++i) {
        void* arg = (void*)(uintptr_t)(i + 1);
#ifdef _WIN32
        J.threads[i] = CreateThread(NULL, 0, worker_main, arg, 0, NULL);
        if (!J.threads[i]) break;
#else
        if (pthread_create(&J.threads[i], NULL, worker_main, arg)) break;
#endif
        J.nthreads = i + 1;
    }
    return J.nthreads;
}

void jobs_shutdown(void) {
    LOCK(&J.lock);
    J.quit = 1;
    WAKE_ALL(&J.wake);
    UNLOCK(&J.lock);
    for (int i = 0; i < J.nthreads; ++i) {
#ifdef _WIN32
        WaitForSingleObject(J.threads[i], INFINITE);
        CloseHandle(J.threads[i]);
#else
        pthread_join(J.threads[i], NULL);
#endif
    }
    free(J.threads);
    J.threads = NULL;
    J.nthreads = 0;
}

int jobs_thread_count(void) { return J.nthreads + 1; }

void jobs_parallel_for(uint32_t count, uint32_t grain, JobFn fn, void* ctx) {
    if (!count) return;
    if (!grain) {
        grain = count / ((uint32_t)jobs_thread_count() * 8u);
        if (!grain) grain = 1;
    }
    if (J.nthreads == 0 || count <= grain) { fn(ctx, 0, count, 0); return; }

    LOCK(&J.lock);
    J.fn = fn;
    J.ctx = ctx;
    J.count = count;
    J.grain = grain;
    J.next = 0;
    J.pending = J.nthreads;
    J.generation++;
    WAKE_ALL(&J.wake);
    UNLOCK(&J.lock);

    run_task(0);

    LOCK(&J.lock);
    while (J.pending) WAIT(&J.done, &J.lock);
    UNLOCK(&J.lock);
}
