/* bio_test_main.c - gme_biotest: runs the life-simulation test suite.
 *   gme_biotest [--device K | --cpu] [--quiet]   (exit code = failed tests) */
#include "bio_api.h"
#include "../jobs.h"
#include "../platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    int device = 0, verbose = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--cpu")) device = -1;
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) verbose = 0;
    }
    platform_init();
    SysInfo si;
    sys_query(&si);
    jobs_init(si.logical_cpus > 1 ? si.logical_cpus - 1 : 0);
    const int failed = bio_run_tests(device, verbose);
    printf("%s: %d test(s) failed\n", failed ? "FAIL" : "OK", failed);
    jobs_shutdown();
    platform_shutdown();
    return failed;
}
