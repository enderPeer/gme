/* jobs.h - fork/join worker pool. The calling (main) thread participates,
 * so jobs_parallel_for uses every hardware thread. */
#ifndef GME_JOBS_H
#define GME_JOBS_H

#include <stdint.h>

/* Processes indices [begin, end). worker is 0 (main) .. jobs_thread_count()-1. */
typedef void (*JobFn)(void* ctx, uint32_t begin, uint32_t end, uint32_t worker);

int  jobs_init(int worker_threads);   /* extra threads besides main */
void jobs_shutdown(void);
int  jobs_thread_count(void);         /* workers + main */

/* grain = indices per work item; 0 picks one automatically. */
void jobs_parallel_for(uint32_t count, uint32_t grain, JobFn fn, void* ctx);

#endif
