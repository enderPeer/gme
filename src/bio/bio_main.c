/* bio_main.c - gme_genesis life stages (docs/biosphere-spec.md 1, 6, 7.2, 8).
 *
 *   env --scout                      scout 6 islands -> life/regions.txt (+ regions.png)
 *   env --island K [--run DIR]       island K's static environment -> DIR/env.bin, env.png
 *   env --info                       planet hash, seed and sea level of the planet file
 *   life --island K [--run DIR] ...  run or resume one island (checkpoints, metrics, maps)
 *   life-test [--device D | --cpu]   the life-simulation test suite (bio_test.c)
 *   bioexport --run DIR [--out F]    one island -> biomap part (biopart_K.bin)
 *   bioexport --merge P... [--out life/home.biomap]   parts -> the engine's biomap
 *   biostat --run DIR [--hash]       statistics of the newest checkpoint
 *   biomap-test [--out F]            T11 checkerboard of every vegetation kind around the engine spawn
 *   life-hub --dir D --islands ...   one pass of the island hub: planet-wide atmosphere, migrants
 *   life-seedbank --runs DIR         the best genomes of every island -> a seed bank
 *   life-rebase --bank F --island K  a new run on this planet founded from a seed bank
 *   life-inoculate --from A --to B   evolved genomes of one guild from run A into run B
 *
 * genesis_main.c hands these stages to bio_main() before its own option
 * parser; bio_main returns -1 for any other stage name. */
#include "bio_internal.h"     /* read-only use: accumulators, raw stats and agent download for metrics and maps */
#include "bio_env.h"
#include "../biomap.h"
#include "../fauna.h"
#include "../flora.h"
#include "../image.h"
#include "../jobs.h"
#include "../planetfile.h"
#include "../platform.h"
#include "../spectrum_file.h"
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

#define DEFAULT_PLANET  "genesis/home.planet"
#define DEFAULT_REGIONS "life/regions.txt"
#define DEFAULT_BIOMAP  "life/home.biomap"
#define DEFAULT_SPECTRUM "life/spectrum.bin"
#define DEFAULT_BANK    "life/seedbank.bin"
#define DEFAULT_HUB     "life/hub"
#define HUB_MIGRANTS    256          /* genomes per habitat an island sends each epoch (spec 6.4) */
#define STEPS_DAY       24u
#define STEPS_YEAR      8760u
#define KEEP_CKPTS      3

/* ---- options --------------------------------------------------------------------- */
typedef struct Opt {
    const char* stage;
    const char* planet;
    const char* regions;
    const char* run;
    const char* out;
    int      island, device, device_set, threads;
    int      resume, scout, info, crc_only, hash, quiet, no_fallback, xz_set;
    double   duty, years, minutes, ckpt_min, barrier;
    uint32_t days;                 /* --days: total sim-days (exact; overrides --years) */
    uint32_t cap, salt, seed_n, maps_every, metrics_every;
    int32_t  x, z;
    const char* merge[64];
    int      nmerge;
    /* features, light, hub, seed bank */
    const char* features;          /* --features LIST: of a new run (default all) */
    const char* enable;            /* --enable LIST: switched on when resuming (an intervention) */
    const char* spectrum;          /* --spectrum FILE: light in bands (default life/spectrum.bin when present) */
    int      no_spectrum, no_hub, species, fauna_set;
    const char* bank;              /* --bank FILE */
    const char* from;              /* --from RUN (life-inoculate) */
    const char* to;                /* --to RUN */
    const char* guild;             /* --guild NAME[@HABITAT] */
    const char* runs;              /* --runs DIR[,DIR...] */
    const char* dir;               /* --dir DIR (life-hub) */
    const char* islands;           /* --islands 0,1,... */
    const char* fauna;             /* --fauna FILE */
    uint32_t n;                    /* --n N */
    int      stop_at_start;        /* internal (life-rebase): create, checkpoint, stop */
} Opt;

static void usage(void) {
    printf("life stages of gme_genesis (docs/biosphere-spec.md):\n"
           "  env --scout [--out life/regions.txt]            scout %d islands on the planet\n"
           "  env --island K [--run DIR] [--crc-only]         island K's environment -> DIR/env.bin, env.png\n"
           "  env --info                                      planet hash, seed, sea level\n"
           "  life --island K [--run DIR] [--device D | --cpu] [--resume] [--duty X] [--cap N]\n"
           "       [--years Y | --days D] [--minutes M] [--salt S] [--checkpoint-min M] [--seed-protocells N]\n"
           "       [--maps-every Y] [--metrics-every D] [--barrier X] [--hash]\n"
           "       [--features LIST] (new run: all | none | animals,sex,learn,morph,hub,spectral; default all)\n"
           "       [--enable LIST] (with --resume: switch features on in a running world; logged as an intervention)\n"
           "       [--spectrum FILE | --no-spectrum] (light in wavelength bands; default %s when present)\n"
           "       [--bank FILE] (new run: found it from a seed bank instead of protocells) [--no-hub]\n"
           "  life-test [--device D | --cpu] [--quiet]        test suite T1-T10, T12-T19\n"
           "  bioexport --run DIR [--out FILE] [--fauna FILE] [--device D]   island -> biomap part (DIR/biopart_K.bin)\n"
           "                                                  and its creatures (DIR/fauna_K.bin)\n"
           "  bioexport --merge PATH... [--out %s] [--fauna life/home.fauna] [--no-fallback]\n"
           "  biostat --run DIR [--hash] [--species] [--device D]   statistics of the newest checkpoint\n"
           "  biomap-test [--out life/test.biomap] [--x X --z Z]   T11 pattern around the engine spawn\n"
           "  life-hub [--dir %s] --islands 0,1,... [--runs DIR,DIR,...]   one pass of the hub: the planet-wide\n"
           "       atmosphere from the islands' gas uptake, migrants around the ring. --runs: the islands' run folders\n"
           "       on this machine (outboxes are collected and inboxes filled; rigs: cluster/bio_hub.ps1 does that)\n"
           "  life-seedbank --runs DIR [--out %s] [--n 2000]   best genomes per guild x habitat of every\n"
           "       island_* run under DIR (or of DIR itself)\n"
           "  life-rebase --bank FILE --island K [--run DIR] [--n 20000]   new run on this planet founded from the bank\n"
           "  life-inoculate --from RUN --to RUN --guild G [--n 500]   evolved genomes of guild G (producer, chemotroph,\n"
           "       decomposer, consumer, any; with @marine, @coastal or @land; land_plant, land_animal) from one run's\n"
           "       newest checkpoint into the other's (stop that island first)\n"
           "  common: [--planet %s] [--regions %s] [--threads N]\n"
           "  DIR defaults to runs/bio_<planet hash>/island_K\n",
           BIO_ISLANDS, DEFAULT_SPECTRUM, DEFAULT_BIOMAP, DEFAULT_HUB, DEFAULT_BANK, DEFAULT_PLANET, DEFAULT_REGIONS);
}

static int parse(int argc, char** argv, Opt* o) {
    memset(o, 0, sizeof *o);
    o->stage = argv[1];
    o->planet = DEFAULT_PLANET; o->regions = DEFAULT_REGIONS;
    o->island = -1; o->device = 0;
    o->duty = -1; o->years = 0; o->minutes = 0; o->ckpt_min = 30; o->barrier = 0;
    o->seed_n = 20000; o->maps_every = 1; o->metrics_every = 1;
    for (int i = 2; i < argc; ++i) {
        const char* k = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        int used = 1;   /* value consumed */
        if      (!strcmp(k, "--scout"))       { o->scout = 1; used = 0; }
        else if (!strcmp(k, "--info"))        { o->info = 1; used = 0; }
        else if (!strcmp(k, "--crc-only"))    { o->crc_only = 1; used = 0; }
        else if (!strcmp(k, "--resume"))      { o->resume = 1; used = 0; }
        else if (!strcmp(k, "--hash"))        { o->hash = 1; used = 0; }
        else if (!strcmp(k, "--quiet"))       { o->quiet = 1; used = 0; }
        else if (!strcmp(k, "--no-fallback")) { o->no_fallback = 1; used = 0; }
        else if (!strcmp(k, "--no-spectrum")) { o->no_spectrum = 1; used = 0; }
        else if (!strcmp(k, "--no-hub"))      { o->no_hub = 1; used = 0; }
        else if (!strcmp(k, "--species"))     { o->species = 1; used = 0; }
        else if (!strcmp(k, "--cpu"))         { o->device = -1; o->device_set = 1; used = 0; }
        else if (!strcmp(k, "--merge")) {
            used = 0;
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) && o->nmerge < 64) o->merge[o->nmerge++] = argv[++i];
            if (!o->nmerge) { fprintf(stderr, "--merge needs at least one part file or directory\n"); return -1; }
        }
        else if (!v) {
            static const char* vals[] = { "--planet", "--regions", "--run", "--out", "--island", "--device", "--threads",
                                          "--duty", "--years", "--days", "--minutes", "--checkpoint-min", "--barrier", "--cap",
                                          "--salt", "--seed-protocells", "--maps-every", "--metrics-every", "--x", "--z",
                                          "--features", "--enable", "--spectrum", "--bank", "--from", "--to", "--guild", "--runs",
                                          "--dir", "--islands", "--fauna", "--n" };
            int known = 0;
            for (size_t q = 0; q < sizeof vals / sizeof vals[0]; ++q) known |= !strcmp(k, vals[q]);
            fprintf(stderr, known ? "%s needs a value\n" : "unknown option %s\n", k);
            return -1;
        }
        else if (!strcmp(k, "--planet"))  o->planet = v;
        else if (!strcmp(k, "--regions")) o->regions = v;
        else if (!strcmp(k, "--run"))     o->run = v;
        else if (!strcmp(k, "--out"))     o->out = v;
        else if (!strcmp(k, "--island"))  o->island = atoi(v);
        else if (!strcmp(k, "--device"))  { o->device = atoi(v); o->device_set = 1; }
        else if (!strcmp(k, "--threads")) o->threads = atoi(v);
        else if (!strcmp(k, "--duty"))    o->duty = atof(v);
        else if (!strcmp(k, "--years"))   o->years = atof(v);
        else if (!strcmp(k, "--days"))    o->days = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--minutes")) o->minutes = atof(v);
        else if (!strcmp(k, "--checkpoint-min")) o->ckpt_min = atof(v);
        else if (!strcmp(k, "--barrier")) o->barrier = atof(v);
        else if (!strcmp(k, "--cap"))     o->cap = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--salt"))    o->salt = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--seed-protocells")) o->seed_n = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--maps-every"))    o->maps_every = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--metrics-every")) o->metrics_every = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--features")) o->features = v;
        else if (!strcmp(k, "--enable"))  o->enable = v;
        else if (!strcmp(k, "--spectrum")) o->spectrum = v;
        else if (!strcmp(k, "--bank"))    o->bank = v;
        else if (!strcmp(k, "--from"))    o->from = v;
        else if (!strcmp(k, "--to"))      o->to = v;
        else if (!strcmp(k, "--guild"))   o->guild = v;
        else if (!strcmp(k, "--runs"))    o->runs = v;
        else if (!strcmp(k, "--dir"))     o->dir = v;
        else if (!strcmp(k, "--islands")) o->islands = v;
        else if (!strcmp(k, "--fauna"))   { o->fauna = v; o->fauna_set = 1; }
        else if (!strcmp(k, "--n"))       o->n = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--x"))       { o->x = atoi(v); o->xz_set |= 1; }
        else if (!strcmp(k, "--z"))       { o->z = atoi(v); o->xz_set |= 2; }
        else { fprintf(stderr, "unknown option %s\n", k); return -1; }
        i += used;
    }
    if (!o->metrics_every) o->metrics_every = 1;
    return 0;
}

/* ---- small helpers -------------------------------------------------------------------- */
static int load_planet(const char* path, HomePlanet* hp) {
    if (planet_load(path, hp)) { fprintf(stderr, "cannot read planet file %s\n", path); return -1; }
    return 0;
}

static void default_run(char* buf, size_t n, uint64_t ph, int island) {
    snprintf(buf, n, "runs/bio_%016llx/island_%d", (unsigned long long)ph, island);
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

/* parent directory of a file path (mkdir -p) */
static int mkdirs_for(const char* file) {
    char d[1024];
    snprintf(d, sizeof d, "%s", file);
    char* s = strrchr(d, '/');
    char* b = strrchr(d, '\\');
    if (b && (!s || b > s)) s = b;
    if (!s) return 0;
    *s = 0;
    return d[0] ? bio_mkdirs(d) : 0;
}

static const char* fmt_n(double v, char* buf, size_t n) {   /* 1.23M, 41.0k, 512 */
    if (v >= 1e9) snprintf(buf, n, "%.2fG", v / 1e9);
    else if (v >= 1e6) snprintf(buf, n, "%.2fM", v / 1e6);
    else if (v >= 1e3) snprintf(buf, n, "%.1fk", v / 1e3);
    else snprintf(buf, n, "%.0f", v);
    return buf;
}

/* "animals,sex,learn" -> BIO_F_*; -1 on an unknown name */
static const struct { const char* name; uint32_t bit; } FEATS[] = {
    { "animals", BIO_F_ANIMALS }, { "sex", BIO_F_SEX }, { "learn", BIO_F_LEARN }, { "morph", BIO_F_MORPH }, { "hub", BIO_F_HUB },
    { "spectral", BIO_F_SPECTRAL },
};
static int64_t parse_features(const char* list) {
    if (!list || !*list) return 0;
    uint32_t m = 0;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", list);
    for (char* t = strtok(buf, ",+ "); t; t = strtok(NULL, ",+ ")) {
        if (!strcmp(t, "all")) { m |= BIO_F_ALL | BIO_F_SPECTRAL; continue; }
        if (!strcmp(t, "none")) continue;
        int found = 0;
        for (size_t k = 0; k < sizeof FEATS / sizeof FEATS[0]; ++k) if (!strcmp(t, FEATS[k].name)) { m |= FEATS[k].bit; found = 1; }
        if (!found) { fprintf(stderr, "unknown feature '%s' (animals, sex, learn, morph, hub, spectral, all, none)\n", t); return -1; }
    }
    return m;
}
static const char* features_str(uint32_t m, char* buf, size_t n) {
    size_t w = 0;
    buf[0] = 0;
    for (size_t k = 0; k < sizeof FEATS / sizeof FEATS[0]; ++k)
        if (m & FEATS[k].bit) w += (size_t)snprintf(buf + w, n - w, "%s%s", w ? "," : "", FEATS[k].name);
    if (!w) snprintf(buf, n, "none");
    return buf;
}

/* The spectrum file of a run: --spectrum, else life/spectrum.bin when it exists.
 * Returns 1 loaded, 0 none (scalar light), -1 given but unusable. */
static int load_spectrum(const Opt* o, uint64_t ph, SpectrumFile* sf) {
    if (o->no_spectrum) return 0;
    const char* path = o->spectrum ? o->spectrum : DEFAULT_SPECTRUM;
    const int rc = spectrum_load(path, sf);
    if (rc == -1 && !o->spectrum) return 0;                       /* no default file: scalar light */
    if (rc) { fprintf(stderr, "%s: %s\n", path, rc == -1 ? "cannot read it" : rc == -3 ? "CRC mismatch (corrupt)" : "not a GMESPEC1 file"); return -1; }
    if (sf->planet_hash != ph) {
        fprintf(stderr, "%s was made for planet %016llx; this planet is %016llx (gme --export-spectrum %s)\n", path,
                (unsigned long long)sf->planet_hash, (unsigned long long)ph, path);
        return o->spectrum ? -1 : 0;                              /* a stale default file is ignored, a named one is an error */
    }
    return 1;
}

static void gates_str(uint32_t g, char* buf, size_t n) {
    size_t w = 0;
    buf[0] = 0;
    for (int k = 0; k < 6; ++k)
        if (g & (1u << k)) w += (size_t)snprintf(buf + w, n - w, "%sG%d", w ? " " : "", k);
    if (!w) snprintf(buf, n, "none");
}
static const char* const GATE_NAME[6] = { "established", "marine photosynthesis", "oxygenic", "land plants",
                                          "land animals", "oxidation" };

/* files of a directory whose names start with prefix and end with suffix */
typedef void (*ListFn)(void* ctx, const char* name);
static void list_dir(const char* dir, const char* prefix, const char* suffix, ListFn fn, void* ctx) {
    const size_t lp = strlen(prefix), ls = strlen(suffix);
#ifdef _WIN32
    char pat[1100];
    snprintf(pat, sizeof pat, "%s\\*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        const char* nm = fd.cFileName;
        const size_t l = strlen(nm);
        if (l >= lp + ls && !strncmp(nm, prefix, lp) && !strcmp(nm + l - ls, suffix)) fn(ctx, nm);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* d = opendir(dir);
    if (!d) return;
    struct dirent* de;
    while ((de = readdir(d)) != NULL) {
        const char* nm = de->d_name;
        const size_t l = strlen(nm);
        if (l >= lp + ls && !strncmp(nm, prefix, lp) && !strcmp(nm + l - ls, suffix)) fn(ctx, nm);
    }
    closedir(d);
#endif
}

/* ---- checkpoints ------------------------------------------------------------------------ */
typedef struct CkList { uint64_t step[256]; int n; } CkList;
static void ck_add(void* ctx, const char* name) {
    CkList* L = (CkList*)ctx;
    const char* p = name + 5;                     /* "ckpt_" */
    char* end = NULL;
    const unsigned long long v = strtoull(p, &end, 10);
    if (end == p || strcmp(end, ".bin") || L->n >= 256) return;
    L->step[L->n++] = v;
}
static int cmp_u64_desc(const void* a, const void* b) {
    const uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x > y ? -1 : (x < y ? 1 : 0);
}
static void ck_scan(const char* dir, CkList* L) {
    L->n = 0;
    list_dir(dir, "ckpt_", ".bin", ck_add, L);
    qsort(L->step, (size_t)L->n, sizeof(uint64_t), cmp_u64_desc);
}
static void ck_path(char* buf, size_t n, const char* dir, uint64_t step) {
    snprintf(buf, n, "%s/ckpt_%012llu.bin", dir, (unsigned long long)step);
}
static void tmp_remove(void* ctx, const char* name) {
    char p[1100];
    snprintf(p, sizeof p, "%s/%s", (const char*)ctx, name);
    remove(p);
}

/* newest checkpoint that loads (CRCs, planet and env checked by bio_sim_load) */
static BioSim* ck_load_newest(const char* dir, const BioEnv* e, const BioConfig* c, uint64_t* step, int verbose) {
    CkList L;
    ck_scan(dir, &L);
    for (int i = 0; i < L.n; ++i) {
        char p[1100];
        ck_path(p, sizeof p, dir, L.step[i]);
        BioSim* s = bio_sim_load(p, e, c);
        if (s) { *step = L.step[i]; return s; }
        if (verbose) fprintf(stderr, "checkpoint %s rejected: %s\n", p, bio_error());
    }
    return NULL;
}

static int ck_save(BioSim* s, const char* dir, uint64_t step, double* secs, double* mb) {
    char p[1100];
    ck_path(p, sizeof p, dir, step);
    const double t0 = time_now();
    if (bio_sim_save(s, p)) { fprintf(stderr, "checkpoint %s failed: %s\n", p, bio_error()); return -1; }
    *secs = time_now() - t0;
    FILE* f = fopen(p, "rb");
    *mb = 0;
    if (f) { fseek(f, 0, SEEK_END); *mb = (double)ftell(f) / 1048576.0; fclose(f); }
    CkList L;
    ck_scan(dir, &L);
    for (int i = KEEP_CKPTS; i < L.n; ++i) { ck_path(p, sizeof p, dir, L.step[i]); remove(p); }
    return 0;
}

/* drop lines written after the resumed step (metrics rows, GATE lines), so a
 * resumed run continues the files without duplicates */
static void prune_after(const char* path, uint64_t step, int is_csv) {
    FILE* f = fopen(path, "rb");
    if (!f) return;
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* g = fopen(tmp, "wb");
    if (!g) { fclose(f); return; }
    char line[8192];
    int dropped = 0, first = 1;
    while (fgets(line, sizeof line, f)) {
        unsigned long long st = 0;
        int drop = 0;
        if (is_csv) drop = !first && sscanf(line, "%llu,", &st) == 1 && st > step;
        else drop = strstr(line, " GATE ") && sscanf(line, "step=%llu", &st) == 1 && st > step;
        first = 0;
        if (drop) { ++dropped; continue; }
        fputs(line, g);
    }
    fclose(f);
    const int bad = fclose(g);
    if (!dropped || bad) { remove(tmp); return; }
    remove(path);
    if (rename(tmp, path)) remove(tmp);
}

static int count_lines_with(const char* path, const char* needle) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    char line[4096];
    int n = 0;
    while (fgets(line, sizeof line, f)) n += strstr(line, needle) != NULL;
    fclose(f);
    return n;
}

/* Opens a CSV for appending. A file written with other columns (an older
 * build) is kept as <name>_old<k>.csv and a new one started, so every file has
 * one header and rows that match it. */
static FILE* open_table(const char* dir, const char* name, const char* header) {
    char p[1100], q[1200];
    snprintf(p, sizeof p, "%s/%s.csv", dir, name);
    FILE* f = fopen(p, "rb");
    if (f) {
        static char line[8192];
        const int same = fgets(line, sizeof line, f) && !strcmp(line, header);
        fclose(f);
        if (!same) {
            for (int k = 1; k < 100; ++k) {
                snprintf(q, sizeof q, "%s/%s_old%d.csv", dir, name, k);
                if (!file_exists(q)) break;
            }
            if (rename(p, q)) remove(p);
            printf("life: %s.csv had other columns: kept as %s, starting a new table\n", name, q);
        }
    }
    const int fresh = !file_exists(p);
    f = fopen(p, "ab");
    if (f && fresh) fputs(header, f);
    return f;
}

static void stage_log(const char* dir, const char* fmt, ...) {
    char p[1100];
    snprintf(p, sizeof p, "%s/stages.log", dir);
    FILE* f = fopen(p, "ab");
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}

/* ---- signals: checkpoint and exit on Ctrl+C / SIGTERM (a second one exits at once) -------- */
static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) {
    if (g_stop) _exit(130);
    g_stop = sig;
    signal(sig, on_signal);
}
static void install_signals(void) {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
#ifdef SIGHUP
    signal(SIGHUP, on_signal);
#endif
#ifdef SIGBREAK
    signal(SIGBREAK, on_signal);
#endif
}

/* ---- env.bin of a run --------------------------------------------------------------------- */
/* Loads DIR/env.bin (crc checked) and makes sure it belongs to this planet and
 * island; builds it from the regions file when missing. 0 ok. */
static int run_env(const Opt* o, const HomePlanet* hp, const char* dir, int island, BioEnv* e, int build_missing) {
    char path[1100];
    snprintf(path, sizeof path, "%s/env.bin", dir);
    const uint64_t ph = planet_hash(hp);
    uint32_t q = 0, fc = 0;
    const int rc = bio_env_load(path, e, &q, &fc);
    if (rc == 0) {
        if (e->planet_hash != ph) {
            fprintf(stderr, "%s was built for planet %016llx; this planet (%s) is %016llx\n", path,
                    (unsigned long long)e->planet_hash, o->planet, (unsigned long long)ph);
            bio_env_free(e);
            return -1;
        }
        if (island >= 0 && e->island != island) {
            fprintf(stderr, "%s belongs to island %d, not %d\n", path, e->island, island);
            bio_env_free(e);
            return -1;
        }
        return 0;
    }
    if (rc != -1 || !build_missing) {
        fprintf(stderr, "%s: %s\n", path, rc == -1 ? "cannot read it (run gme_genesis env --island K first)"
                                         : rc == -3 ? "CRC mismatch (corrupt)" : "not a valid env file");
        return -1;
    }
    if (island < 0) { fprintf(stderr, "%s missing and no --island given\n", path); return -1; }
    BioRegions rg;
    if (bio_regions_load(o->regions, &rg)) { fprintf(stderr, "%s missing; cannot read %s either\n", path, o->regions); return -1; }
    if (rg.planet_hash != ph) {
        fprintf(stderr, "%s was scouted for planet %016llx; this planet is %016llx (run env --scout)\n", o->regions,
                (unsigned long long)rg.planet_hash, (unsigned long long)ph);
        return -1;
    }
    printf("life: %s missing, building island %d's environment\n", path, island);
    if (bio_env_build(e, hp, &rg, island)) return -1;
    if (bio_mkdirs(dir) || bio_env_save(path, e, rg.vent_q10, &fc)) { fprintf(stderr, "cannot write %s\n", path); bio_env_free(e); return -1; }
    snprintf(path, sizeof path, "%s/env.png", dir);
    bio_env_png(path, e);
    return 0;
}

/* ======================================================================================
 * env
 * ====================================================================================== */
static void env_summary(const BioEnv* e) {
    const int64_t n = (int64_t)e->w * e->h;
    int64_t water = 0, marine = 0, tidal = 0, land = 0, deep = 0, shelf = 0, vents = 0, sinks = 0, coast1 = 0;
    int32_t emin = 1 << 20, emax = -(1 << 20);
    for (int64_t c = 0; c < n; ++c) {
        const int32_t d = e->depth[c];
        water += d >= 1;
        if (e->wet[2][c] == 0) land++;
        else if (e->wet[0][c] >= 255 && d >= 1) marine++;
        else tidal++;
        deep += d > BIO_VENT_DEPTH;
        shelf += d >= 1 && d <= 60;
        vents += e->vent[c] > 0;
        sinks += d < 1 && e->flow_dir[c] == 255;
        coast1 += d < 1 && e->coast[c] <= 64;
        if (e->elev[c] < emin) emin = e->elev[c];
        if (e->elev[c] > emax) emax = e->elev[c];
    }
    printf("env: island %d at x %d..%d, z %d..%d (%dx%d cells of %d m), sea level %d\n", e->island, e->x0,
           e->x0 + (e->w << e->shift), e->z0, e->z0 + (e->h << e->shift), e->w, e->h, 1 << e->shift, e->sea_level);
    printf("env: water %.1f%% (marine %.1f%%, tidal %.1f%%), dry land %.1f%%, shelf 1-60 m %.1f%%, deeper than %d m %.1f%%\n",
           100.0 * water / n, 100.0 * marine / n, 100.0 * tidal / n, 100.0 * land / n, 100.0 * shelf / n, BIO_VENT_DEPTH,
           100.0 * deep / n);
    printf("env: elevation %d..%d m, %lld vents, land within 1 km of the sea %.1f%%, land sinks %lld\n", emin, emax,
           (long long)vents, 100.0 * coast1 / n, (long long)sinks);
    printf("env: S_top %d W/m^2, T_surf %.2f K, gravity %.3f g, organics %d ppm, atmosphere ppm H2S %d CH4 %d NH3 %d CO2 %d O2 %d N2 %d H2 %d\n",
           e->s_top_wm2, e->t_surf_ck / 100.0, e->gravity_q8 / 256.0, e->organic_ppm, e->atm_ppm[0], e->atm_ppm[1],
           e->atm_ppm[2], e->atm_ppm[3], e->atm_ppm[4], e->atm_ppm[5], e->atm_ppm[6]);
}

static int stage_env(const Opt* o) {
    HomePlanet hp;
    if (load_planet(o->planet, &hp)) return 1;
    const uint64_t ph = planet_hash(&hp);
    if (o->info) {
        printf("planet_hash %016llx\nseed %u\nsea_level %d\natm_known %d\n", (unsigned long long)ph, hp.system_seed,
               hp.gen.sea_level, hp.atm_known);
        return 0;
    }
    if (o->scout) {
        const char* out = o->out ? o->out : DEFAULT_REGIONS;
        if (mkdirs_for(out)) { fprintf(stderr, "cannot create the folder of %s\n", out); return 1; }
        char png[1100];
        snprintf(png, sizeof png, "%s", out);
        char* dot = strrchr(png, '.');
        if (dot && !strchr(dot, '/') && !strchr(dot, '\\')) *dot = 0;
        strncat(png, ".png", sizeof png - strlen(png) - 1);
        BioRegions rg;
        const int rc = bio_env_scout(&hp, &rg, png);
        if (rg.n == 0) return 1;
        if (bio_regions_save(out, &rg, &hp)) { fprintf(stderr, "cannot write %s\n", out); return 1; }
        printf("wrote %s (planet %016llx, vent threshold %u/1024)\n", out, (unsigned long long)ph, rg.vent_q10);
        return rc ? 1 : 0;
    }
    if (o->island < 0) { usage(); return 2; }
    BioRegions rg;
    if (bio_regions_load(o->regions, &rg)) { fprintf(stderr, "cannot read %s (run gme_genesis env --scout)\n", o->regions); return 1; }
    if (rg.planet_hash != ph) {
        fprintf(stderr, "%s was scouted for planet %016llx; %s is planet %016llx - run env --scout again\n", o->regions,
                (unsigned long long)rg.planet_hash, o->planet, (unsigned long long)ph);
        return 1;
    }
    const double t0 = time_now();
    BioEnv e;
    if (bio_env_build(&e, &hp, &rg, o->island)) return 1;
    const double tb = time_now() - t0;
    env_summary(&e);
    int expect = -1;
    for (int k = 0; k < rg.n; ++k) if (rg.r[k].k == o->island && rg.r[k].vents) expect = rg.r[k].vents;
    if (expect >= 0 && (uint32_t)expect != bio_env_vents(&e))
        printf("env: NOTE the scout counted %d vents here, the full build %u\n", expect, bio_env_vents(&e));
    if (o->crc_only) {
        printf("island %d planet %016llx env crc %08x (built in %.1f s)\n", o->island, (unsigned long long)ph, e.crc, tb);
        bio_env_free(&e);
        return 0;
    }
    char dir[1024], path[1100];
    if (o->run) snprintf(dir, sizeof dir, "%s", o->run);
    else default_run(dir, sizeof dir, ph, o->island);
    if (bio_mkdirs(dir)) { fprintf(stderr, "cannot create %s\n", dir); bio_env_free(&e); return 1; }
    snprintf(path, sizeof path, "%s/env.bin", dir);
    BioEnv old;
    uint32_t oq = 0, ofc = 0, fc = 0;
    const int same = bio_env_load(path, &old, &oq, &ofc) == 0 && old.crc == e.crc && oq == rg.vent_q10 &&
                     old.planet_hash == e.planet_hash && old.island == e.island && old.x0 == e.x0 && old.z0 == e.z0 &&
                     old.s_top_wm2 == e.s_top_wm2 && old.t_surf_ck == e.t_surf_ck &&
                     !memcmp(old.atm_ppm, e.atm_ppm, sizeof e.atm_ppm);
    const int had_old = old.elev != NULL;
    if (had_old) bio_env_free(&old);
    CkList L;
    ck_scan(dir, &L);
    if (same) {
        fc = ofc;
        printf("env: %s unchanged\n", path);
    } else if (L.n && (had_old || file_exists(path))) {
        fprintf(stderr, "env: %s differs from the new environment but %s holds %d checkpoint(s) of a run on it; "
                        "not overwritten (use another --run, or move that run away)\n", path, dir, L.n);
        bio_env_free(&e);
        return 1;
    } else {
        if (bio_env_save(path, &e, rg.vent_q10, &fc)) { fprintf(stderr, "cannot write %s\n", path); bio_env_free(&e); return 1; }
        printf("env: wrote %s\n", path);
    }
    snprintf(path, sizeof path, "%s/env.png", dir);
    if (!bio_env_png(path, &e)) printf("env: wrote %s\n", path);
    printf("island %d planet %016llx env crc %08x file crc %08x (built in %.1f s)\n", o->island, (unsigned long long)ph,
           e.crc, fc, tb);
    bio_env_free(&e);
    return 0;
}

/* ======================================================================================
 * life: statistics, console block, metrics, maps
 * ====================================================================================== */
typedef struct Snap {           /* everything we report at one moment */
    BioStats st;
    int64_t  acc[BACC_N];
    int64_t  acc2[BACC2_N];
    int64_t  raw[BIO_STAT_N];
    double   wall;
} Snap;

static int snap(BioSim* s, Snap* x) {
    if (bio_sim_stats(s, &x->st)) return -1;
    if (bio_stats_raw(s, x->raw) < 0) return -1;      /* also refreshes s->Gh */
    memcpy(x->acc, s->Gh.acc, sizeof x->acc);
    memcpy(x->acc2, s->Gh.acc2, sizeof x->acc2);
    x->wall = time_now();
    return 0;
}

typedef struct Run {
    const Opt* o;
    BioSim*  s;
    const BioEnv* e;
    char     dir[1024];
    int      island;
    double   vol_total;          /* m^3 of all cells' active volumes */
    double   duty;
    int      rollbacks;
    FILE*    metrics;
    FILE*    census;
    uint64_t census_step;        /* step of the last census row */
    uint32_t* base;              /* dim terrain colours for the maps */
} Run;

static double conc_uM(const Run* R, const Snap* x, int f) {
    return R->vol_total > 0 ? (double)x->raw[BS_FLD + f] / R->vol_total : 0.0;   /* mmol / m^3 = uM */
}

static double o2_max_uM(Run* R) {
    const BioWorld* w = &R->s->w;
    const size_t n = (size_t)w->n_cells;
    int64_t* buf = (int64_t*)malloc(n * 8);
    if (!buf) return -1;
    double m = 0;
    if (!bio_copy_out(R->s, buf, w->fld + (size_t)BF_O2 * n, n * 8)) {
        for (size_t c = 0; c < n; ++c) {
            const int32_t d = R->e->depth[c];
            const double V = d >= 1 ? 256.0 * (d < 1 ? 1 : (d > 10 ? 10 : d)) : 128.0;
            const double v = (double)buf[c] / V;
            if (v > m) m = v;
        }
    }
    free(buf);
    return m;
}

static void print_block(Run* R, const Snap* x, const Snap* p, double days, double sps) {
    const BioStats* st = &x->st;
    char a[32], b[32], c[32], g[64];
    const double dd = days > 0 ? days : 1;
    int64_t atot = 0;
    for (int k = 0; k < BIO_ATM_N; ++k) atot += st->atm[k] > 0 ? st->atm[k] : 0;
    const double af = atot > 0 ? 100.0 / (double)atot : 0;
    gates_str(st->gates, g, sizeof g);
    const double gpp = (double)(x->acc[BACC_FIX] - p->acc[BACC_FIX]) / dd / 1000.0;
    const double resp = (double)(x->acc[BACC_RESP] - p->acc[BACC_RESP]) / dd / 1000.0;
    const double bur = (double)(x->acc[BACC_SED_C] - p->acc[BACC_SED_C]) / dd / 1000.0;
    printf("isl%d y=%.2f live=%s (motile %s, sessile photo %s) C[mar %.2e | coast %.2e | land %.2e mol]\n", R->island,
           st->sim_years, fmt_n((double)st->live, a, sizeof a), fmt_n((double)st->motile, b, sizeof b),
           fmt_n((double)st->sessile_photo, c, sizeof c), st->carbon_marine / 1000.0, st->carbon_coastal / 1000.0,
           st->carbon_land / 1000.0);
    const double pd = dd * 1000.0;     /* mmol -> mol per day */
    printf("  mol/d: GPP %.2e R %.2e NEP %+.2e burial %.2e | METH %.1e ANOX %.1e OXY %.1e FERM %.1e SULF %.1e AERO %.1e MOX %.1e FEOX %.1e\n",
           gpp, resp, gpp - resp, bur, (st->path_flux[0] - p->st.path_flux[0]) / pd, (st->path_flux[1] - p->st.path_flux[1]) / pd,
           (st->path_flux[2] - p->st.path_flux[2]) / pd, (st->path_flux[3] - p->st.path_flux[3]) / pd,
           (st->path_flux[4] - p->st.path_flux[4]) / pd, (st->path_flux[5] - p->st.path_flux[5]) / pd,
           (st->path_flux[6] - p->st.path_flux[6]) / pd, (st->path_flux[7] - p->st.path_flux[7]) / pd);
    printf("  atm H2S %.1f%% CH4 %.1f%% NH3 %.1f%% CO2 %.3f%% O2 %.5f%%  O2aq max %.3guM mean %.3guM  H2S aq %.3g mM  DIC %.3g mM\n",
           st->atm[0] * af, st->atm[1] * af, st->atm[2] * af, st->atm[3] * af, st->atm[4] * af, o2_max_uM(R),
           conc_uM(R, x, BF_O2), conc_uM(R, x, BF_H2S) / 1000.0, conc_uM(R, x, BF_DIC) / 1000.0);
    const uint32_t* mg = st->mean_gene;
    printf("  genes METH %u ANOX %u OXY %u FERM %u SULF %u AERO %u | PIG %u HUE %u HOLD %u STRU %u CUT %u ROOT %u UVS %u MOTOR %u SIZE %u\n",
           mg[BG_E_METH], mg[BG_E_ANOX], mg[BG_E_OXY], mg[BG_E_FERM], mg[BG_E_SULFRED], mg[BG_E_AERO], mg[BG_PIGMENT],
           mg[BG_HUE], mg[BG_HOLDFAST], mg[BG_STRUCT], mg[BG_CUTICLE], mg[BG_ROOT], mg[BG_UVSHIELD], mg[BG_MOTOR], mg[BG_SIZE]);
    int64_t resid = 0;
    for (int k = 0; k < 4; ++k) resid |= st->total[k] - st->ledger[k];
    printf("  think %.0f%% of agents  dW|mean| %.2f  rollbacks %d  %.0f st/s (duty %.2f)  gates %s  land veg cells %u  resid %s\n",
           100.0 * st->think_share, st->mean_abs_dw, R->rollbacks, sps, R->duty, g, st->land_cells_vegetated,
           resid ? "NONZERO" : "0");
    if (st->features & (BIO_F_ANIMALS | BIO_F_SEX))
    {
        char cr[32];
        printf("  animals: %s creatures, %s land animals; %s agents live mostly on prey (trophic level %.2f); "
               "per day %.3g kills, %.3g bites, %.3g mol C eaten | sexual %s, %.3g births/d with two parents\n",
               fmt_n((double)st->creatures, cr, sizeof cr), fmt_n((double)st->land_animals, b, sizeof b),
               fmt_n((double)st->consumers, a, sizeof a), st->mean_troph, (double)(st->kills - p->st.kills) / dd,
               (double)(st->bites - p->st.bites) / dd, (double)(st->pred_carbon - p->st.pred_carbon) / pd,
               fmt_n((double)st->sexual, c, sizeof c), (double)(st->sexual_births - p->st.sexual_births) / dd);
    }
    fflush(stdout);
}

static const char* METRICS_HEADER =
    "step,day,year,wall_s,steps_per_s,live,motile,sessile_photo,births,deaths,"
    "C_marine,C_coastal,C_land,"
    "METH,ANOX,OXY,FERM,SULFRED,AERO,METHOX,FEOX,"
    "GPP,assim,resp,death_C,burial_C,vent_C,weath_C,import_C,"
    "atm_H2S,atm_CH4,atm_NH3,atm_CO2,atm_O2,atm_N2,atm_H2,"
    "aq_O2_uM,aq_H2S_uM,aq_CH4_uM,aq_H2_uM,aq_DIC_uM,aq_DOC_uM,"
    "total_C,total_N,total_S,total_Fe,resid_C,resid_N,resid_S,resid_Fe,neg_agents,neg_cells,"
    "g_E_METH,g_E_ANOX,g_E_OXY,g_E_FERM,g_E_SULFRED,g_E_AERO,g_E_METHOX,g_E_FEOX,"
    "g_size,g_struct,g_holdfast,g_cuticle,g_root,g_buoy,g_motor,g_pigment,g_hue,g_uvshield,g_disperse,g_mut,g_think,"
    "think_period,mean_abs_dw,think_share,land_cells_veg,o2_cells,"
    "consumers,creatures,mean_troph,land_animals,kills,bites,pred_C,sexual,sexual_births,gut_C,features,gates\n";

/* one metrics row; fluxes are per day, averaged over the row interval */
static void metrics_row(Run* R, const Snap* x, const Snap* p, double t_start) {
    FILE* f = R->metrics;
    if (!f) return;
    const BioStats* s = &x->st;
    const BioStats* q = &p->st;
    const int64_t days = (int64_t)((s->step - q->step) / STEPS_DAY) > 0 ? (int64_t)((s->step - q->step) / STEPS_DAY) : 1;
    const double sps = x->wall > p->wall ? (double)(s->step - q->step) / (x->wall - p->wall) : 0.0;
    fprintf(f, "%llu,%llu,%.4f,%.1f,%.1f,%llu,%llu,%llu,%lld,%lld,%lld,%lld,%lld",
            (unsigned long long)s->step, (unsigned long long)(s->step / STEPS_DAY), s->sim_years, x->wall - t_start, sps,
            (unsigned long long)s->live, (unsigned long long)s->motile, (unsigned long long)s->sessile_photo,
            (long long)((int64_t)(s->births - q->births) / days), (long long)((int64_t)(s->deaths - q->deaths) / days),
            (long long)s->carbon_marine, (long long)s->carbon_coastal, (long long)s->carbon_land);
    for (int k = 0; k < BIO_PATHWAYS; ++k) fprintf(f, ",%lld", (long long)((s->path_flux[k] - q->path_flux[k]) / days));
    static const int accs[8] = { BACC_FIX, BACC_ASSIM, BACC_RESP, BACC_DEATH_C, BACC_SED_C, BACC_VENT_C, BACC_WEATH_C, BACC_IMPORT_C };
    for (int k = 0; k < 7; ++k) fprintf(f, ",%lld", (long long)((x->acc[accs[k]] - p->acc[accs[k]]) / days));
    fprintf(f, ",%lld", (long long)x->acc[BACC_IMPORT_C]);              /* cumulative */
    for (int k = 0; k < BIO_ATM_N; ++k) fprintf(f, ",%lld", (long long)s->atm[k]);
    static const int fl[6] = { BF_O2, BF_H2S, BF_CH4, BF_H2, BF_DIC, BF_DOC };
    for (int k = 0; k < 6; ++k) fprintf(f, ",%.6g", conc_uM(R, x, fl[k]));
    for (int k = 0; k < 4; ++k) fprintf(f, ",%lld", (long long)s->total[k]);
    for (int k = 0; k < 4; ++k) fprintf(f, ",%lld", (long long)(s->total[k] - s->ledger[k]));
    fprintf(f, ",%lld,%lld", (long long)x->raw[BS_NEGA], (long long)x->raw[BS_NEGF]);
    for (int k = 0; k < 8; ++k) fprintf(f, ",%u", s->mean_gene[k]);
    static const int gn[13] = { BG_SIZE, BG_STRUCT, BG_HOLDFAST, BG_CUTICLE, BG_ROOT, BG_BUOY, BG_MOTOR, BG_PIGMENT, BG_HUE,
                                BG_UVSHIELD, BG_DISPERSE, BG_MUT, BG_THINK };
    for (int k = 0; k < 13; ++k) fprintf(f, ",%u", s->mean_gene[gn[k]]);
    fprintf(f, ",%.2f,%.4f,%.4f,%u,%lld", 1.0 + s->mean_gene[BG_THINK] / 4.0, s->mean_abs_dw, s->think_share,
            s->land_cells_vegetated, (long long)x->raw[BS_O2CELLS]);
    /* animals and sex: counts now, events per day over the row interval */
    fprintf(f, ",%llu,%llu,%.3f,%llu,%lld,%lld,%lld,%llu,%lld,%lld,%u,%u\n", (unsigned long long)s->consumers,
            (unsigned long long)s->creatures, s->mean_troph, (unsigned long long)s->land_animals, (long long)((s->kills - q->kills) / days), (long long)((s->bites - q->bites) / days),
            (long long)((s->pred_carbon - q->pred_carbon) / days), (unsigned long long)s->sexual,
            (long long)((s->sexual_births - q->sexual_births) / days),
            (long long)((x->acc2[BACC2_GUT_C] - p->acc2[BACC2_GUT_C]) / days), s->features, s->gates);
    fflush(f);
}

/* ---- yearly outputs: census of guilds x realms, maps ------------------------------------------ */
/* Guilds by capability (genes), not by realised flux, which is not kept per agent:
 * producers can photosynthesise (pigment and ANOX or OXY >= 32, the test of
 * BioStats.sessile_photo and of the biomap export); chemotrophs lean to
 * lithotrophy (METH / FEOX / METHOX at least as strong as FERM / SULFRED / AERO);
 * the other heterotrophs are decomposers, or consumers when motile. Most agents
 * are mixotrophs, so the split is coarse. */
enum { GUILD_PROD = 0, GUILD_CHEMO, GUILD_DECOMP, GUILD_CONS, GUILD_N };
static int guild_of(const uint8_t* g) {
    if (g[BG_PIGMENT] > 0 && (g[BG_E_ANOX] >= 32 || g[BG_E_OXY] >= 32)) return GUILD_PROD;
    int chemo = g[BG_E_METH];
    if (g[BG_E_FEOX] > chemo) chemo = g[BG_E_FEOX];
    if (g[BG_E_METHOX] > chemo) chemo = g[BG_E_METHOX];
    int het = g[BG_E_FERM];
    if (g[BG_E_SULFRED] > het) het = g[BG_E_SULFRED];
    if (g[BG_E_AERO] > het) het = g[BG_E_AERO];
    if (chemo >= het && chemo > 0) return GUILD_CHEMO;
    return g[BG_MOTOR] > 0 ? GUILD_CONS : GUILD_DECOMP;
}
static int realm_of(const BioEnv* e, int64_t c) {
    if (e->wet[2][c] == 0) return 2;
    if (e->wet[0][c] >= 255 && e->depth[c] >= 1) return 0;
    return 1;
}
static int popc8(uint32_t v) { int n = 0; for (v &= 255u; v; v &= v - 1) ++n; return n; }
static int cmp_u64(const void* a, const void* b) {
    const uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static uint32_t rgb(double r, double g, double b) {
    const int R = r < 0 ? 0 : (r > 255 ? 255 : (int)r), G = g < 0 ? 0 : (g > 255 ? 255 : (int)g), B = b < 0 ? 0 : (b > 255 ? 255 : (int)b);
    return 0xFF000000u | ((uint32_t)R << 16) | ((uint32_t)G << 8) | (uint32_t)B;
}
static void unrgb(uint32_t c, double* r, double* g, double* b) { *r = (c >> 16) & 255u; *g = (c >> 8) & 255u; *b = c & 255u; }

static void make_base(Run* R) {
    const BioEnv* e = R->e;
    const int64_t n = (int64_t)e->w * e->h;
    R->base = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)n);
    if (!R->base) return;
    for (int64_t c = 0; c < n; ++c) {
        const int32_t d = e->depth[c];
        if (e->wet[0][c] >= 255 && d >= 1) { const double t = sqrt((d > 120 ? 120 : d) / 120.0); R->base[c] = rgb(22 - 12 * t, 44 - 26 * t, 70 - 30 * t); }
        else if (e->wet[2][c] > 0) R->base[c] = rgb(70, 66, 52);
        else { const double u = (e->elev[c] - e->sea_level) / 220.0; R->base[c] = rgb(52 + 40 * u, 52 + 30 * u, 46 + 40 * u); }
    }
}

/* average base colour of a block of cells */
static void base_block(const Run* R, int32_t i0, int32_t j0, int32_t s, double* r, double* g, double* b) {
    double sr = 0, sg = 0, sb = 0;
    for (int32_t j = j0; j < j0 + s; ++j)
        for (int32_t i = i0; i < i0 + s; ++i) {
            double x, y, z;
            unrgb(R->base[(int64_t)j * R->e->w + i], &x, &y, &z);
            sr += x; sg += y; sb += z;
        }
    *r = sr / (s * s); *g = sg / (s * s); *b = sb / (s * s);
}

static void write_map(const Run* R, const char* name, uint32_t year, const uint32_t* px, int w, int h, int last) {
    char p[1200];
    if (last) snprintf(p, sizeof p, "%s/maps/%s_last.png", R->dir, name);
    else snprintf(p, sizeof p, "%s/maps/%s_y%05u.png", R->dir, name, year);
    if (write_png(p, px, w, h)) fprintf(stderr, "cannot write %s\n", p);
}

static int epoch_outputs(Run* R, int last) {
    BioSim* s = R->s;
    const BioEnv* e = R->e;
    const int32_t W = e->w, H = e->h;
    const int64_t n_cells = (int64_t)W * H;
    BioStats st;
    if (bio_sim_stats(s, &st)) return -1;
    const uint32_t year = (uint32_t)(st.step / STEPS_YEAR);
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) { fprintf(stderr, "epoch: %s\n", bio_error()); return -1; }
    /* census */
    int64_t cg[3][GUILD_N], ng[3][GUILD_N], nreal[3] = { 0, 0, 0 };
    memset(cg, 0, sizeof cg); memset(ng, 0, sizeof ng);
    int64_t neur = 0, plast = 0, motile = 0, ph_n = 0, ph_h = 0;
    int32_t hmax = 0;
    const int32_t PW = W / 4, PH = H / 4;                         /* 64 m pixels */
    double* gm = (double*)calloc((size_t)PW * PH * 3, sizeof(double));
    uint64_t* gh = (uint64_t*)malloc(sizeof(uint64_t) * (n ? n : 1));
    for (uint32_t i = 0; i < n; ++i) {
        const BioAgentRec* r = &a[i];
        const uint8_t* g = r->g.body;
        const int32_t c = bio_cell_at(&s->w, r->x, r->z);
        const int rl = realm_of(e, c), gd = guild_of(g);
        const int64_t body = (int64_t)r->C + r->S + r->gut;
        cg[rl][gd] += body; ng[rl][gd]++; nreal[rl]++;
        neur += popc8(g[BG_NEURONS]); plast += popc8(g[BG_PLASTIC]);
        motile += g[BG_MOTOR] > 0;
        const int32_t hq4 = bio_height_q4(r->S);
        if (hq4 > hmax) hmax = hq4;
        if (gd == GUILD_PROD && g[BG_HOLDFAST] > 64) { ph_n++; ph_h += hq4; }
        if (gm) {
            const int32_t px = (c % W) / 4, py = (c / W) / 4;
            const int ch = gd == GUILD_PROD ? 1 : (gd == GUILD_CHEMO ? 0 : 2);
            gm[((size_t)py * PW + px) * 3 + ch] += (double)body;
        }
        if (gh) gh[i] = bio_fnv64(BIO_FNV_INIT, &r->g, sizeof r->g);
    }
    uint64_t distinct = 0;
    if (gh && n) {
        qsort(gh, n, sizeof(uint64_t), cmp_u64);
        distinct = 1;
        for (uint32_t i = 1; i < n; ++i) distinct += gh[i] != gh[i - 1];
    }
    free(gh);
    free(a);
    BioSpecies sp;                       /* clusters of similar genomes in a sample (spec 8.3) */
    memset(&sp, 0, sizeof sp);
    if (bio_sim_species(s, 4096, &sp)) fprintf(stderr, "epoch: species: %s\n", bio_error());
    if (R->census && R->census_step != st.step) {
        R->census_step = st.step;
        fprintf(R->census, "%llu,%.3f,%u", (unsigned long long)st.step, st.sim_years, n);
        for (int rl = 0; rl < 3; ++rl) fprintf(R->census, ",%lld", (long long)nreal[rl]);
        for (int rl = 0; rl < 3; ++rl) for (int gd = 0; gd < GUILD_N; ++gd) fprintf(R->census, ",%lld", (long long)cg[rl][gd]);
        for (int rl = 0; rl < 3; ++rl) for (int gd = 0; gd < GUILD_N; ++gd) fprintf(R->census, ",%lld", (long long)ng[rl][gd]);
        fprintf(R->census, ",%lld,%.3f,%.3f,%llu,%.2f,%.3f", (long long)motile, n ? (double)neur / n : 0.0,
                n ? (double)plast / n : 0.0, (unsigned long long)distinct, hmax / 16.0, ph_n ? ph_h / 16.0 / ph_n : 0.0);
        fprintf(R->census, ",%u,%u,%u,%.4f,%.4f,%.4f\n", sp.sample, sp.richness, sp.clusters2, sp.shannon, sp.simpson, sp.top_share);
        fflush(R->census);
    }
    printf("  species y=%.2f: %u clusters in a sample of %u genomes (%u with 2+ members), Shannon H %.2f, Simpson %.3f, largest %.0f%%\n",
           st.sim_years, sp.richness, sp.sample, sp.clusters2, sp.shannon, sp.simpson, 100.0 * sp.top_share);
    printf("  census y=%.2f: C mol [marine prod %.2e chemo %.2e decomp %.2e cons %.2e] [coastal prod %.2e chemo %.2e decomp %.2e cons %.2e]\n"
           "                [land prod %.2e chemo %.2e decomp %.2e cons %.2e]  %llu distinct genomes, tallest %.2f m, sessile producers %.3f m mean\n",
           st.sim_years, cg[0][0] / 1e3, cg[0][1] / 1e3, cg[0][2] / 1e3, cg[0][3] / 1e3, cg[1][0] / 1e3, cg[1][1] / 1e3,
           cg[1][2] / 1e3, cg[1][3] / 1e3, cg[2][0] / 1e3, cg[2][1] / 1e3, cg[2][2] / 1e3, cg[2][3] / 1e3,
           (unsigned long long)distinct, hmax / 16.0, ph_n ? ph_h / 16.0 / ph_n : 0.0);
    if (!R->base) { free(gm); return 0; }
    char dir[1200];
    snprintf(dir, sizeof dir, "%s/maps", R->dir);
    bio_mkdirs(dir);
    /* biomass by guild, 64 m pixels: red chemotrophs, green producers, blue heterotrophs */
    if (gm) {
        uint32_t* px = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)PW * PH);
        if (px) {
            const double ref = log10(1.0 + 5000.0);                /* 5000 mmol C / m^2 = full brightness */
            for (int32_t y = 0; y < PH; ++y)
                for (int32_t x = 0; x < PW; ++x) {
                    double br, bg, bb;
                    base_block(R, 4 * x, 4 * y, 4, &br, &bg, &bb);
                    const double* v = &gm[((size_t)y * PW + x) * 3];
                    const double cr = log10(1.0 + v[0] / 4096.0) / ref, cgr = log10(1.0 + v[1] / 4096.0) / ref,
                                 cb = log10(1.0 + v[2] / 4096.0) / ref;
                    px[(size_t)y * PW + x] = rgb(br + 255 * cr * 1.4, bg + 255 * cgr * 1.4, bb + 255 * cb * 1.4);
                }
            write_map(R, "biomass", year, px, PW, PH, last);
            free(px);
        }
        free(gm);
    }
    /* vegetation as exported to the biomap: dominant kind and pigment hue, 32 m pixels */
    uint32_t* cells = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)n_cells);
    if (cells) {
        uint32_t fb[64];
        uint8_t pal[16][3];
        if (!bio_sim_vegetation(s, cells, fb, pal)) {
            const int32_t VW = W / 2, VH = H / 2;
            uint32_t* pk = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)VW * VH);
            uint32_t* ph = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)VW * VH);
            static const double kc[6][3] = { { 0, 0, 0 }, { 165, 95, 175 }, { 40, 175, 140 }, { 160, 215, 90 }, { 70, 165, 60 }, { 25, 105, 35 } };
            int64_t nk[6] = { 0 };
            for (int64_t c = 0; c < n_cells; ++c) { const uint32_t k = BIOCELL_KIND(cells[c]); if (k < 6) nk[k]++; }
            for (int32_t y = 0; pk && ph && y < VH; ++y)
                for (int32_t x = 0; x < VW; ++x) {
                    uint32_t best = 0;
                    for (int b = 0; b < 4; ++b) {
                        const uint32_t v = cells[(int64_t)(2 * y + (b >> 1)) * W + 2 * x + (b & 1)];
                        if (BIOCELL_KIND(v) && (!best || BIOCELL_COVER(v) > BIOCELL_COVER(best))) best = v;
                    }
                    double br, bg, bb;
                    base_block(R, 2 * x, 2 * y, 2, &br, &bg, &bb);
                    const uint32_t k = BIOCELL_KIND(best);
                    const double t = best ? 0.35 + 0.65 * BIOCELL_COVER(best) / 255.0 : 0.0;
                    if (k >= 1 && k <= 5) pk[(size_t)y * VW + x] = rgb(br * (1 - t) + kc[k][0] * t, bg * (1 - t) + kc[k][1] * t, bb * (1 - t) + kc[k][2] * t);
                    else pk[(size_t)y * VW + x] = rgb(br, bg, bb);
                    const uint32_t pi = BIOCELL_PAL(best);
                    if (best && pi) ph[(size_t)y * VW + x] = rgb(br * (1 - t) + pal[pi][0] * t, bg * (1 - t) + pal[pi][1] * t, bb * (1 - t) + pal[pi][2] * t);
                    else ph[(size_t)y * VW + x] = rgb(br, bg, bb);
                }
            if (pk && ph) { write_map(R, "kind", year, pk, VW, VH, last); write_map(R, "hue", year, ph, VW, VH, last); }
            free(pk); free(ph);
            printf("  vegetation: mat %.2f%% kelp %.2f%% cover %.2f%% shrub %.2f%% tree %.2f%% of cells\n", 100.0 * nk[1] / n_cells,
                   100.0 * nk[2] / n_cells, 100.0 * nk[3] / n_cells, 100.0 * nk[4] / n_cells, 100.0 * nk[5] / n_cells);
        }
        free(cells);
    }
    /* dissolved O2 and H2S, 64 m pixels, log scale */
    int64_t* fbuf = (int64_t*)malloc(sizeof(int64_t) * (size_t)n_cells);
    uint32_t* px = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)PW * PH);
    for (int which = 0; fbuf && px && which < 2; ++which) {
        const int f = which ? BF_H2S : BF_O2;
        if (bio_copy_out(s, fbuf, s->w.fld + (size_t)f * (size_t)n_cells, (size_t)n_cells * 8)) break;
        for (int32_t y = 0; y < PH; ++y)
            for (int32_t x = 0; x < PW; ++x) {
                double sum = 0, vol = 0;
                for (int b = 0; b < 16; ++b) {
                    const int64_t c = (int64_t)(4 * y + (b >> 2)) * W + 4 * x + (b & 3);
                    const int32_t d = e->depth[c];
                    sum += (double)fbuf[c];
                    vol += d >= 1 ? 256.0 * (d > 10 ? 10 : d) : 128.0;
                }
                const double uM = sum / vol;
                double br, bg, bb;
                base_block(R, 4 * x, 4 * y, 4, &br, &bg, &bb);
                const double t = which ? log10(1.0 + uM) / log10(1.0 + 3000.0) : log10(1.0 + 100.0 * uM) / log10(1.0 + 100.0 * 300.0);
                const double tt = t < 0 ? 0 : (t > 1 ? 1 : t);
                px[(size_t)y * PW + x] = which ? rgb(br + 200 * tt, bg + 40 * tt, bb + 220 * tt) : rgb(br + 240 * tt, bg + 220 * tt, bb + 40 * tt);
            }
        write_map(R, which ? "h2s" : "o2", year, px, PW, PH, last);
    }
    free(fbuf); free(px);
    return 0;
}

/* ======================================================================================
 * life
 * ====================================================================================== */
static int stage_life(const Opt* o) {
    if (o->island < 0 && !o->run) { fprintf(stderr, "life needs --island K (and optionally --run DIR)\n"); return 2; }
    HomePlanet hp;
    if (load_planet(o->planet, &hp)) return 1;
    const uint64_t ph = planet_hash(&hp);
    Run R;
    memset(&R, 0, sizeof R);
    R.o = o;
    if (o->run) snprintf(R.dir, sizeof R.dir, "%s", o->run);
    else default_run(R.dir, sizeof R.dir, ph, o->island);
    if (bio_mkdirs(R.dir)) { fprintf(stderr, "cannot create %s\n", R.dir); return 1; }
    BioEnv e;
    if (run_env(o, &hp, R.dir, o->island, &e, 1)) return 1;
    R.e = &e;
    R.island = e.island;
    for (int64_t c = 0; c < (int64_t)e.w * e.h; ++c) {
        const int32_t d = e.depth[c];
        R.vol_total += d >= 1 ? 256.0 * (d > 10 ? 10 : d) : 128.0;
    }
    list_dir(R.dir, "ckpt_", ".tmp", tmp_remove, R.dir);           /* half-written checkpoints of a crash */

    BioConfig cfg;
    bio_config_default(&cfg);
    cfg.device = o->device;
    cfg.cap = o->cap;
    cfg.salt = o->salt;
    if (o->duty > 0) cfg.duty = o->duty > 1 ? 1.0 : o->duty;
    static const int32_t barrier_q8[5] = { 179, 218, 256, 294, 333 };   /* islands 0-4: x0.7 .. x1.3 (spec 4) */
    cfg.barrier_q8 = o->barrier > 0 ? (int32_t)lround(o->barrier * 256.0) : (R.island >= 0 && R.island < 5 ? barrier_q8[R.island] : 256);
    R.duty = cfg.duty;
    /* features of a new run (default: all), what a resumed run switches on, and the light file */
    static SpectrumFile sf;
    const int64_t fm = o->features ? parse_features(o->features) : (int64_t)(BIO_F_ALL | BIO_F_SPECTRAL);
    const int64_t en = parse_features(o->enable);
    const int have_sf = (fm < 0 || en < 0) ? -1 : load_spectrum(o, ph, &sf);
    if (fm < 0 || en < 0 || have_sf < 0) { bio_env_free(&e); return 2; }
    if (!have_sf && ((o->features && (fm & BIO_F_SPECTRAL)) || (en & BIO_F_SPECTRAL) || o->spectrum)) {
        fprintf(stderr, "light in bands needs a spectrum file: gme --export-spectrum %s (or --spectrum FILE)\n", DEFAULT_SPECTRUM);
        bio_env_free(&e);
        return 1;
    }
    cfg.features = (uint32_t)fm & BIO_F_ALL;
    cfg.enable = (uint32_t)en;
    cfg.spectrum = have_sf ? &sf : NULL;       /* a resumed run takes it when its checkpoint (or --enable) says so */

    BioSim* s = NULL;
    uint64_t step0 = 0;
    CkList L;
    ck_scan(R.dir, &L);
    printf("life: island %d, %s, planet %016llx, env crc %08x, %u vents\n", R.island, R.dir, (unsigned long long)ph, e.crc,
           bio_env_vents(&e));
    const double tc = time_now();
    if (L.n && !o->resume) {
        fprintf(stderr, "%s already holds %d checkpoint(s); add --resume to continue that run, or use another --run\n", R.dir, L.n);
        bio_env_free(&e);
        return 1;
    }
    char ckname[64] = "";
    if (o->resume && L.n) {
        s = ck_load_newest(R.dir, &e, &cfg, &step0, 1);
        if (!s) { fprintf(stderr, "no valid checkpoint in %s\n", R.dir); bio_env_free(&e); return 1; }
        snprintf(ckname, sizeof ckname, "ckpt_%012llu.bin", (unsigned long long)step0);
    }
    {   /* rows written after the starting point (a crash, an extinction) go; a fresh start
         * keeps an earlier attempt's tables as *_prev.csv */
        static const char* tabs[2] = { "metrics", "census" };
        char mp[1100], pp[1100];
        for (int k = 0; k < 2; ++k) {
            snprintf(mp, sizeof mp, "%s/%s.csv", R.dir, tabs[k]);
            if (s) prune_after(mp, step0, 1);
            else if (file_exists(mp)) {
                snprintf(pp, sizeof pp, "%s/%s_prev.csv", R.dir, tabs[k]);
                remove(pp);
                rename(mp, pp);
            }
        }
        snprintf(mp, sizeof mp, "%s/stages.log", R.dir);
        prune_after(mp, step0, 0);
    }
    int founders = 0;
    uint32_t bank_n = 0;
    if (!s) {
        if (!(fm & BIO_F_SPECTRAL)) cfg.spectrum = NULL;       /* a new run with scalar light */
        s = bio_sim_create(&e, &cfg);
        if (!s) { fprintf(stderr, "life: %s\n", bio_error()); bio_env_free(&e); return 1; }
        if (o->bank) {      /* founded from a seed bank: evolved genomes into their habitats (spec 6.6) */
            BioBankRec* br = NULL;
            uint64_t bph = 0;
            if (bio_bank_load(o->bank, &br, &bank_n, &bph)) { fprintf(stderr, "life: %s\n", bio_error()); bio_sim_destroy(s); bio_env_free(&e); return 1; }
            founders = bio_sim_seed_bank(s, br, bank_n, o->seed_n, -1, -1);
            free(br);
            if (founders <= 0) { fprintf(stderr, "life: the seed bank %s placed no founders: %s\n", o->bank, founders < 0 ? bio_error() : "no matching habitat"); bio_sim_destroy(s); bio_env_free(&e); return 1; }
            printf("life: founded from %s (%u genomes of planet %016llx): %d founders\n", o->bank, bank_n, (unsigned long long)bph, founders);
        } else if (bio_sim_seed(s, o->seed_n)) { fprintf(stderr, "life: seeding failed: %s\n", bio_error()); bio_sim_destroy(s); bio_env_free(&e); return 1; }
    }
    R.s = s;
    {
        char sl[1100];
        snprintf(sl, sizeof sl, "%s/stages.log", R.dir);
        R.rollbacks = count_lines_with(sl, "INTERVENTION rollback");
    }
    if (s->gpu) printf("life: GPU %d: %s (sm_%d%d, %d SMs)", cfg.device, s->dev.name, s->dev.cc_major, s->dev.cc_minor, s->dev.sm_count);
    else printf("life: CPU reference (%d threads)", jobs_thread_count());
    printf(", cap %u agents, duty %.2f, salt %u, barrier x%.2f, atm scale %d ppm, ready in %.1f s\n", s->cfg.cap, cfg.duty,
           s->w.salt, s->w.P.barrier_q8 / 256.0, s->w.P.atm_scale_ppm, time_now() - tc);
    char fs[96], fs0[96];
    const uint32_t feat = bio_sim_features(s);
    features_str(feat, fs, sizeof fs);
    printf("life: features %s%s\n", fs, (feat & BIO_F_SPECTRAL) ? "" : (have_sf ? "" : " (scalar light: no spectrum file)"));
    if (feat & BIO_F_SPECTRAL)
        printf("life: light in 9 bands from %s (crc %08x): harvest %.3f / %.3f eu per umol photons s (oxygenic / anoxygenic)\n",
               o->spectrum ? o->spectrum : DEFAULT_SPECTRUM, (uint32_t)s->w.P.spectrum_crc, s->w.P.eff_sp_oxy_q16 / 65536.0,
               s->w.P.eff_sp_anox_q16 / 65536.0);
    if (ckname[0]) {
        printf("life: resumed %s at step %llu (year %.3f)\n", ckname, (unsigned long long)step0, step0 / (double)STEPS_YEAR);
        stage_log(R.dir, "step=%llu year=%.3f RESUME from %s salt %u features %s", (unsigned long long)step0, step0 / (double)STEPS_YEAR,
                  ckname, s->w.salt, fs);
        if (feat != s->feat_loaded) {      /* switched on in a living world: an intervention, on the record */
            features_str(feat & ~s->feat_loaded, fs0, sizeof fs0);
            printf("life: INTERVENTION: switched on %s at step %llu\n", fs0, (unsigned long long)step0);
            stage_log(R.dir, "step=%llu year=%.3f INTERVENTION enable %s (features now %s)", (unsigned long long)step0,
                      step0 / (double)STEPS_YEAR, fs0, fs);
        }
    } else if (o->bank) {
        printf("life: new run from a seed bank, %d founders, salt %u\n", founders, s->w.salt);
        stage_log(R.dir, "step=0 year=0.000 START island %d, rebase: %d founders of %u banked genomes (%s), salt %u, cap %u, barrier x%.2f, features %s",
                  R.island, founders, bank_n, o->bank, s->w.salt, s->cfg.cap, s->w.P.barrier_q8 / 256.0, fs);
    } else {
        printf("life: new run, %u protocells, salt %u\n", o->seed_n, s->w.salt);
        stage_log(R.dir, "step=0 year=0.000 START island %d, %u protocells, salt %u, cap %u, barrier x%.2f, features %s", R.island,
                  o->seed_n, s->w.salt, s->cfg.cap, s->w.P.barrier_q8 / 256.0, fs);
    }
    /* output files */
    {
        static char ch[2048];
        size_t cw = (size_t)snprintf(ch, sizeof ch, "step,year,live,n_marine,n_coastal,n_land");
        static const char* rn[3] = { "mar", "coast", "land" }, * gn[4] = { "prod", "chemo", "decomp", "cons" };
        for (int r = 0; r < 3; ++r) for (int g = 0; g < 4; ++g) cw += (size_t)snprintf(ch + cw, sizeof ch - cw, ",C_%s_%s", rn[r], gn[g]);
        for (int r = 0; r < 3; ++r) for (int g = 0; g < 4; ++g) cw += (size_t)snprintf(ch + cw, sizeof ch - cw, ",n_%s_%s", rn[r], gn[g]);
        snprintf(ch + cw, sizeof ch - cw, ",motile,neurons,plastic_rows,distinct_genomes,tallest_m,producer_height_m"
                                          ",sp_sample,species,species_2plus,shannon,simpson,top_share\n");
        R.metrics = open_table(R.dir, "metrics", METRICS_HEADER);
        R.census = open_table(R.dir, "census", ch);
        if (!R.metrics || !R.census) fprintf(stderr, "life: cannot open the metrics files in %s\n", R.dir);
    }
    make_base(&R);
    R.census_step = UINT64_MAX;
    install_signals();

    const uint64_t target = o->stop_at_start ? step0 : o->days ? (uint64_t)o->days * STEPS_DAY
                          : (o->years > 0 ? (uint64_t)ceil(o->years * STEPS_YEAR / STEPS_DAY - 1e-9) * STEPS_DAY : UINT64_MAX);
    const uint64_t epoch = (uint64_t)STEPS_YEAR * (o->maps_every ? o->maps_every : 1);
    const double t_start = time_now();
    double t_ckpt = t_start;
    uint64_t last_ckpt = ckname[0] ? step0 : UINT64_MAX;
    Snap prev, cur, blk;
    if (snap(s, &prev)) { fprintf(stderr, "life: stats: %s\n", bio_error()); return 1; }
    blk = prev;
    Snap mrow = prev;
    uint32_t gates = prev.st.gates;
    printf("\n");
    print_block(&R, &prev, &prev, 1, 0);
    if (!ckname[0] && o->maps_every) epoch_outputs(&R, 0);           /* year 0: the seed */
    int rc = 0, extinct = 0;
    const char* why = "done";
    const int hub = (bio_sim_features(s) & BIO_F_HUB) && !o->no_hub;
    while (1) {
        if (g_stop) { why = "signal"; break; }
        if (s->w.step >= target) { why = "years reached"; break; }
        if (o->minutes > 0 && time_now() - t_start >= o->minutes * 60.0) { why = "time limit"; break; }
        const uint64_t nstep = STEPS_DAY - s->w.step % STEPS_DAY;
        if (bio_sim_run(s, nstep)) { fprintf(stderr, "life: step %llu failed: %s\n", (unsigned long long)s->w.step, bio_error()); rc = 1; why = "error"; break; }
        if (hub && s->w.step % STEPS_YEAR == 0 && s->w.step / STEPS_YEAR > bio_sim_hub_epoch(s)) {
            /* the island hub: this year's figures and emigrants out, what arrived in (spec 6.4) */
            const uint64_t ep = s->w.step / STEPS_YEAR;
            int nm = 0, na = 0;
            if (bio_hub_export(s, R.dir, ep, HUB_MIGRANTS) || bio_hub_import(s, R.dir, ep, &nm, &na))
                fprintf(stderr, "life: hub at year %llu: %s\n", (unsigned long long)ep, bio_error());
            else if (nm || na) {
                stage_log(R.dir, "step=%llu year=%.3f HUB %d migrants imported%s", (unsigned long long)s->w.step, (double)ep, nm,
                          na ? ", atmosphere moved a quarter toward the planet's" : "");
                printf("life: hub at year %llu: %d migrants imported%s\n", (unsigned long long)ep, nm, na ? ", atmosphere relaxed toward the planet's" : "");
            }
        }
        if (snap(s, &cur)) { fprintf(stderr, "life: stats: %s\n", bio_error()); rc = 1; why = "error"; break; }
        const uint64_t step = cur.st.step, day = step / STEPS_DAY;
        if (day % o->metrics_every == 0) { metrics_row(&R, &cur, &mrow, t_start); mrow = cur; }
        if (cur.st.gates != gates) {
            char g[64];
            gates_str(cur.st.gates, g, sizeof g);
            for (int k = 0; k < 6; ++k)
                if ((cur.st.gates & ~gates) & (1u << k)) {
                    stage_log(R.dir, "step=%llu year=%.3f GATE G%d %s reached (gates %s)", (unsigned long long)step,
                              cur.st.sim_years, k, GATE_NAME[k], g);
                    printf("life: step %llu (year %.2f): gate G%d %s reached\n", (unsigned long long)step, cur.st.sim_years, k, GATE_NAME[k]);
                }
            gates = cur.st.gates;
        }
        if (cur.st.live == 0) {
            extinct = 1; rc = 3; why = "extinct";
            stage_log(R.dir, "step=%llu year=%.3f EXTINCT (live 0)", (unsigned long long)step, cur.st.sim_years);
            print_block(&R, &cur, &blk, (double)(step - blk.st.step) / STEPS_DAY, 0);
            break;
        }
        if (day % 10 == 0) {
            const double sps = cur.wall > blk.wall ? (double)(step - blk.st.step) / (cur.wall - blk.wall) : 0;
            print_block(&R, &cur, &blk, (double)(step - blk.st.step) / STEPS_DAY, sps);
            blk = cur;
        }
        if (o->maps_every && step % epoch == 0) epoch_outputs(&R, 0);
        if (o->ckpt_min > 0 && time_now() - t_ckpt >= o->ckpt_min * 60.0) {
            double secs, mb;
            if (!ck_save(s, R.dir, step, &secs, &mb)) {
                last_ckpt = step;
                printf("life: checkpoint ckpt_%012llu.bin (%.0f MB, %.1f s)\n", (unsigned long long)step, mb, secs);
            }
            t_ckpt = time_now();
        }
        prev = cur;
    }
    if (rc == 0) {
        if (o->maps_every && !g_stop) epoch_outputs(&R, 1);     /* on a signal the checkpoint comes first */
        if (last_ckpt != s->w.step) {
            double secs, mb;
            if (ck_save(s, R.dir, s->w.step, &secs, &mb)) rc = 1;
            else printf("life: checkpoint ckpt_%012llu.bin (%.0f MB, %.1f s)\n", (unsigned long long)s->w.step, mb, secs);
        }
        stage_log(R.dir, "step=%llu year=%.3f STOP (%s)", (unsigned long long)s->w.step, s->w.step / (double)STEPS_YEAR, why);
    }
    const double wall = time_now() - t_start;
    printf("life: %s at step %llu (year %.3f) after %.1f min wall, %.0f steps/s average%s\n", why,
           (unsigned long long)s->w.step, s->w.step / (double)STEPS_YEAR, wall / 60.0,
           wall > 0 ? (double)(s->w.step - step0) / wall : 0.0, extinct ? " - EXTINCT (exit 3)" : "");
    if (o->hash) printf("life: state hash %016llx at step %llu\n", (unsigned long long)bio_sim_hash(s), (unsigned long long)s->w.step);
    if (R.metrics) fclose(R.metrics);
    if (R.census) fclose(R.census);
    free(R.base);
    bio_sim_destroy(s);
    bio_env_free(&e);
    return rc;
}

/* ======================================================================================
 * biostat
 * ====================================================================================== */
static int open_run_sim(const Opt* o, const HomePlanet* hp, BioEnv* e, BioSim** s, uint64_t* step) {
    if (!o->run) { fprintf(stderr, "--run DIR is required\n"); return 2; }
    if (run_env(o, hp, o->run, -1, e, 0)) return 1;
    BioConfig cfg;
    bio_config_default(&cfg);
    cfg.device = o->device_set ? o->device : -1;     /* CPU by default: never competes with a running island */
    cfg.duty = 1.0;
    static SpectrumFile sf;                          /* a run with light in bands cannot be loaded without its light file */
    const int have_sf = load_spectrum(o, planet_hash(hp), &sf);
    if (have_sf < 0) { bio_env_free(e); return 1; }
    cfg.spectrum = have_sf ? &sf : NULL;
    *s = ck_load_newest(o->run, e, &cfg, step, 1);
    if (!*s) { fprintf(stderr, "no valid checkpoint in %s\n", o->run); bio_env_free(e); return 1; }
    return 0;
}

static int stage_biostat(const Opt* o) {
    HomePlanet hp;
    if (load_planet(o->planet, &hp)) return 1;
    BioEnv e;
    BioSim* s = NULL;
    uint64_t step = 0;
    const int r = open_run_sim(o, &hp, &e, &s, &step);
    if (r) return r;
    Run R;
    memset(&R, 0, sizeof R);
    R.o = o; R.s = s; R.e = &e; R.island = e.island;
    snprintf(R.dir, sizeof R.dir, "%s", o->run);
    for (int64_t c = 0; c < (int64_t)e.w * e.h; ++c) { const int32_t d = e.depth[c]; R.vol_total += d >= 1 ? 256.0 * (d > 10 ? 10 : d) : 128.0; }
    char sl[1100];
    snprintf(sl, sizeof sl, "%s/stages.log", o->run);
    R.rollbacks = count_lines_with(sl, "INTERVENTION rollback");
    R.duty = s->cfg.duty;
    Snap x;
    if (snap(s, &x)) { fprintf(stderr, "biostat: %s\n", bio_error()); bio_sim_destroy(s); bio_env_free(&e); return 1; }
    printf("biostat: %s/ckpt_%012llu.bin, island %d, salt %u, cap %u, births %llu, deaths %llu\n", o->run,
           (unsigned long long)step, e.island, s->w.salt, s->cfg.cap, (unsigned long long)x.st.births, (unsigned long long)x.st.deaths);
    print_block(&R, &x, &x, 1, 0);
    printf("  cumulative pathway flux (mol): METH %.3e ANOX %.3e OXY %.3e FERM %.3e SULFRED %.3e AERO %.3e METHOX %.3e FEOX %.3e\n",
           x.st.path_flux[0] / 1e3, x.st.path_flux[1] / 1e3, x.st.path_flux[2] / 1e3, x.st.path_flux[3] / 1e3,
           x.st.path_flux[4] / 1e3, x.st.path_flux[5] / 1e3, x.st.path_flux[6] / 1e3, x.st.path_flux[7] / 1e3);
    printf("  totals C %lld N %lld S %lld Fe %lld mmol; total - ledger: %lld %lld %lld %lld\n", (long long)x.st.total[0],
           (long long)x.st.total[1], (long long)x.st.total[2], (long long)x.st.total[3], (long long)(x.st.total[0] - x.st.ledger[0]),
           (long long)(x.st.total[1] - x.st.ledger[1]), (long long)(x.st.total[2] - x.st.ledger[2]),
           (long long)(x.st.total[3] - x.st.ledger[3]));
    printf("  mean genes:");
    for (int k = 0; k < 48; ++k) printf(" %u", x.st.mean_gene[k]);
    printf("\n");
    {
        char fs[96];
        printf("  features %s; cumulative: %lld kills, %lld bites, %.3e mol C eaten, %lld births with two parents\n",
               features_str(x.st.features, fs, sizeof fs), (long long)x.st.kills, (long long)x.st.bites, x.st.pred_carbon / 1e3,
               (long long)x.st.sexual_births);
    }
    if (o->species) {
        BioSpecies sp;
        if (bio_sim_species(s, 4096, &sp)) fprintf(stderr, "biostat: species: %s\n", bio_error());
        else printf("  species: %u clusters in a sample of %u genomes (%u with 2+ members), Shannon H %.2f, Simpson %.3f, largest %.0f%%\n",
                    sp.richness, sp.sample, sp.clusters2, sp.shannon, sp.simpson, 100.0 * sp.top_share);
    }
    if (o->hash) printf("biostat: state hash %016llx\n", (unsigned long long)bio_sim_hash(s));
    bio_sim_destroy(s);
    bio_env_free(&e);
    return 0;
}

/* ======================================================================================
 * bioexport
 * ====================================================================================== */
static void print_kinds(const char* tag, const uint32_t* c, size_t n) {
    uint64_t k[6] = { 0 }, cov = 0, nveg = 0;
    uint32_t hmax = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t kd = BIOCELL_KIND(c[i]);
        if (kd < 6) k[kd]++;
        if (kd) { nveg++; cov += BIOCELL_COVER(c[i]); if (BIOCELL_HEIGHT(c[i]) > hmax) hmax = BIOCELL_HEIGHT(c[i]); }
    }
    printf("%s: %.2fM cells: bare %.2f%% mat %.2f%% kelp %.2f%% cover %.2f%% shrub %.2f%% tree %.2f%%; mean cover %.0f/255, tallest %u m\n",
           tag, n / 1e6, 100.0 * k[0] / n, 100.0 * k[1] / n, 100.0 * k[2] / n, 100.0 * k[3] / n, 100.0 * k[4] / n,
           100.0 * k[5] / n, nveg ? (double)cov / nveg : 0.0, hmax);
}

/* crown classes among shrubs and trees with an evolved shape word */
static void print_shapes(const char* tag, const uint32_t* c, const uint32_t* m, size_t n) {
    uint64_t cc[BIO_CROWN_N] = { 0 }, woody = 0, shaped = 0, asp = 0;
    for (size_t i = 0; i < n; ++i) {
        if (BIOCELL_KIND(c[i]) < BIO_KIND_SHRUB) continue;
        ++woody;
        if (!(m[i] & BIOMORPH_VALID)) continue;
        ++shaped;
        if (BIOMORPH_CROWN(m[i]) < BIO_CROWN_N) cc[BIOMORPH_CROWN(m[i])]++;
        asp += BIOMORPH_ASPECT_Q8(BIOMORPH_ASPECT(m[i]));
    }
    const double d = shaped ? (double)shaped : 1.0;
    printf("%s: shapes of %llu shrub and tree cells (%llu with evolved shape): ellipsoid %.1f%% cone %.1f%% umbrella %.1f%% column %.1f%% "
           "dome %.1f%%; mean height : width %.2f\n", tag, (unsigned long long)woody, (unsigned long long)shaped, 100.0 * cc[0] / d,
           100.0 * cc[1] / d, 100.0 * cc[2] / d, 100.0 * cc[3] / d, 100.0 * cc[4] / d, asp / d / 256.0);
}

static void print_fallback(const uint32_t fb[64], uint32_t on) {
    int n = 0;
    for (int b = 0; b < 64; ++b) n += fb[b] != 0;
    printf("  fallback %s: %d of 64 environment classes vegetated", on ? "on" : "off", n);
    static const char* kn[6] = { "-", "mat", "kelp", "cover", "shrub", "tree" };
    int shown = 0;
    for (int b = 0; b < 64 && shown < 8; ++b)
        if (fb[b]) { printf("%s bin %d %s/%u", shown ? "," : ":", b, kn[BIOCELL_KIND(fb[b]) < 6 ? BIOCELL_KIND(fb[b]) : 0], BIOCELL_COVER(fb[b])); ++shown; }
    printf("%s\n", n > shown ? ", ..." : "");
}

static int export_one(const Opt* o, const HomePlanet* hp) {
    BioEnv e;
    BioSim* s = NULL;
    uint64_t step = 0;
    const int r = open_run_sim(o, hp, &e, &s, &step);
    if (r) return r;
    BioStats st;
    Biomap b;
    memset(&b, 0, sizeof b);
    int rc = 1;
    if (bio_sim_stats(s, &st)) { fprintf(stderr, "bioexport: %s\n", bio_error()); goto out; }
    b.planet_hash = planet_hash(hp);
    b.seed = hp->system_seed;
    b.sim_year = (uint32_t)(st.step / STEPS_YEAR);
    b.gates = st.gates;
    b.nreg = 1;
    b.shift = e.shift;
    b.reg[0].x0 = e.x0; b.reg[0].z0 = e.z0; b.reg[0].w = e.w; b.reg[0].h = e.h; b.reg[0].offset = 0;
    b.ncells = (size_t)e.w * e.h;
    b.cells = (uint32_t*)malloc(sizeof(uint32_t) * b.ncells);
    if (!b.cells) { fprintf(stderr, "bioexport: out of memory\n"); goto out; }
    /* a run with any feature writes version 2: evolved shapes (zeros while that feature is off) and the
     * atmosphere life has made; a first-release run keeps writing version 1, byte for byte */
    const uint32_t feat = bio_sim_features(s);
    if (feat) {
        b.morph = (uint32_t*)calloc(b.ncells, sizeof(uint32_t));
        if (!b.morph) { fprintf(stderr, "bioexport: out of memory\n"); goto out; }
        int64_t tot = 0;
        for (int g = 0; g < BIO_ATM_N; ++g) tot += st.atm[g] > 0 ? st.atm[g] : 0;
        const int64_t per_ppm = tot / 1000000 > 0 ? tot / 1000000 : 1;      /* divide first: the box is ~1e13 mmol */
        for (int g = 0; g < BIO_ATM_N && tot > 0; ++g) {
            const int64_t ppm = (st.atm[g] > 0 ? st.atm[g] : 0) / per_ppm;
            b.atm_ppm[g] = (int32_t)(ppm > 1000000 ? 1000000 : ppm);
        }
        b.atm_known = tot > 0;
    }
    if (bio_sim_vegetation2(s, b.cells, b.morph, b.fallback, b.morph ? b.fallback_morph : NULL, b.palette)) { fprintf(stderr, "bioexport: %s\n", bio_error()); goto out; }
    for (int k = 0; k < 64; ++k) b.fallback_on |= b.fallback[k] != 0;
    char path[1100];
    if (o->out) snprintf(path, sizeof path, "%s", o->out);
    else snprintf(path, sizeof path, "%s/biopart_%d.bin", o->run, e.island);
    if (mkdirs_for(path) || biomap_save(path, &b)) { fprintf(stderr, "bioexport: cannot write %s\n", path); goto out; }
    {   /* its creatures next to the part (fauna_<K>.bin), or where --fauna says */
        Fauna fa;
        char fp[1200];
        if (o->fauna) snprintf(fp, sizeof fp, "%s", o->fauna);
        else {
            snprintf(fp, sizeof fp, "%s", path);
            char* sl = strrchr(fp, '/');
            char* bs = strrchr(fp, '\\');
            if (bs && (!sl || bs > sl)) sl = bs;
            snprintf(sl ? sl + 1 : fp, sizeof fp - (size_t)(sl ? sl + 1 - fp : 0), "fauna_%d.bin", e.island);
        }
        if (bio_sim_fauna(s, &fa, FAUNA_MAX)) fprintf(stderr, "bioexport: creatures: %s\n", bio_error());
        else {
            if (mkdirs_for(fp) || fauna_save(fp, &fa)) fprintf(stderr, "bioexport: cannot write %s\n", fp);
            else printf("bioexport: wrote %s: %u of %u creatures (largest %.2f m)\n", fp, fa.count, fa.total, fa.count ? fa.rec[0].size_q4 / 16.0 : 0.0);
            fauna_free(&fa);
        }
        uint32_t mc[6];
        if (!bio_sim_motor_census(s, mc))
            printf("bioexport: %u agents carry a motor gene: %u creatures; %u plants with a holdfast, drawn as vegetation (%u of them live mostly "
                   "on prey that comes to them, %u could move 1 m per hour); %u free but slower than that and not living on prey\n",
                   mc[0], mc[1], mc[2], mc[3], mc[4], mc[5]);
    }
    printf("bioexport: island %d, checkpoint step %llu (year %.2f), %llu live agents, %llu sessile phototrophs, gates %02x\n",
           e.island, (unsigned long long)step, st.sim_years, (unsigned long long)st.live, (unsigned long long)st.sessile_photo, st.gates);
    print_kinds("bioexport", b.cells, b.ncells);
    print_fallback(b.fallback, b.fallback_on);
    if (b.morph) print_shapes("bioexport", b.cells, b.morph, b.ncells);
    printf("bioexport: wrote %s (%s; region x %d z %d, %dx%d cells of %d m)\n", path, b.morph ? "version 2" : "version 1", e.x0, e.z0,
           e.w, e.h, 1 << e.shift);
    rc = 0;
out:
    free(b.cells);
    free(b.morph);
    bio_sim_destroy(s);
    bio_env_free(&e);
    return rc;
}

/* part files: a file itself, or biopart_*.bin in a directory and its subdirectories (one level) */
typedef struct PartList { char p[64][1100]; int n; const char* dir; } PartList;
static void part_add(void* ctx, const char* name) {
    PartList* L = (PartList*)ctx;
    if (L->n < 64) snprintf(L->p[L->n++], sizeof L->p[0], "%s/%s", L->dir, name);
}
static void sub_add(void* ctx, const char* name) {
    PartList* L = (PartList*)ctx;
    if (name[0] == '.') return;
    char sub[1100];
    snprintf(sub, sizeof sub, "%s/%s", L->dir, name);
    const char* keep = L->dir;
    L->dir = sub;
    list_dir(sub, "biopart_", ".bin", part_add, L);
    L->dir = keep;
}
static int cmp_str(const void* a, const void* b) { return strcmp((const char*)a, (const char*)b); }
static int fauna_size_cmp(const void* a, const void* b) {       /* largest bodies first; position breaks ties (a stable file) */
    const FaunaRec* x = (const FaunaRec*)a; const FaunaRec* y = (const FaunaRec*)b;
    if (x->size_q4 != y->size_q4) return x->size_q4 > y->size_q4 ? -1 : 1;
    if (x->x_q4 != y->x_q4) return x->x_q4 < y->x_q4 ? -1 : 1;
    return x->z_q4 < y->z_q4 ? -1 : (x->z_q4 > y->z_q4 ? 1 : 0);
}

typedef struct BinCtx { const Biomap* m; const GmePlanet* p; uint32_t seed; uint8_t* bins; } BinCtx;
static void bins_rows(void* ctx, uint32_t b, uint32_t e_, uint32_t wk) {
    (void)wk;
    BinCtx* B = (BinCtx*)ctx;
    for (uint32_t row = b; row < e_; ++row) {       /* row = global row index over all regions */
        uint32_t r = 0, base = 0;
        while (r + 1 < (uint32_t)B->m->nreg && row >= base + (uint32_t)B->m->reg[r].h) { base += (uint32_t)B->m->reg[r].h; ++r; }
        const GmeBioRegion* R = &B->m->reg[r];
        const int32_t j = (int32_t)(row - base);
        for (int32_t i = 0; i < R->w; ++i)
            B->bins[(size_t)R->offset + (size_t)j * R->w + i] =
                (uint8_t)bio_env_bin_at(R->x0 + (i << B->m->shift), R->z0 + (j << B->m->shift), B->m->shift, B->seed, B->p);
    }
}

static int export_merge(const Opt* o, const HomePlanet* hp) {
    PartList* L = (PartList*)calloc(1, sizeof(PartList));
    if (!L) return 1;
    for (int k = 0; k < o->nmerge; ++k) {
        const char* a = o->merge[k];
        const size_t l = strlen(a);
        if (l > 4 && !strcmp(a + l - 4, ".bin") && file_exists(a)) { if (L->n < 64) snprintf(L->p[L->n++], sizeof L->p[0], "%s", a); continue; }
        L->dir = a;
        list_dir(a, "biopart_", ".bin", part_add, L);
        list_dir(a, "", "", sub_add, L);
    }
    qsort(L->p, (size_t)L->n, sizeof L->p[0], cmp_str);
    const uint64_t ph = planet_hash(hp);
    Biomap out;
    memset(&out, 0, sizeof out);
    Biomap parts[BIOMAP_MAX_REGIONS];
    static char part_path[BIOMAP_MAX_REGIONS][1100];
    int np = 0, np_paths = 0;
    size_t total = 0;
    for (int k = 0; k < L->n; ++k) {
        Biomap m;
        const int rc = biomap_load(L->p[k], &m);
        if (rc) { fprintf(stderr, "merge: skipping %s (%s)\n", L->p[k], rc == -3 ? "CRC mismatch" : "not a valid biomap"); continue; }
        if (m.planet_hash != ph || m.seed != hp->system_seed) {
            fprintf(stderr, "merge: skipping %s: made for planet %016llx seed %u, this planet is %016llx seed %u\n", L->p[k],
                    (unsigned long long)m.planet_hash, m.seed, (unsigned long long)ph, hp->system_seed);
            biomap_free(&m);
            continue;
        }
        if (m.nreg < 1 || np >= BIOMAP_MAX_REGIONS) {
            fprintf(stderr, "merge: skipping %s: %s\n", L->p[k], m.nreg < 1 ? "no regions" : "too many parts");
            biomap_free(&m);
            continue;
        }
        if (np && m.shift != parts[0].shift) { fprintf(stderr, "merge: skipping %s: cell size differs\n", L->p[k]); biomap_free(&m); continue; }
        if (out.nreg + m.nreg > BIOMAP_MAX_REGIONS) { fprintf(stderr, "merge: skipping %s: more than %d regions\n", L->p[k], BIOMAP_MAX_REGIONS); biomap_free(&m); continue; }
        printf("merge: %s: %d region(s), sim year %u, gates %02x\n", L->p[k], m.nreg, m.sim_year, m.gates);
        for (int r = 0; r < m.nreg; ++r) {
            out.reg[out.nreg] = m.reg[r];
            out.reg[out.nreg].offset = (uint32_t)(total + m.reg[r].offset);
            out.nreg++;
        }
        total += m.ncells;
        if (m.sim_year > out.sim_year) out.sim_year = m.sim_year;
        out.gates |= m.gates;
        snprintf(part_path[np_paths++], sizeof part_path[0], "%s", L->p[k]);
        parts[np++] = m;
    }
    free(L);
    if (!np) { fprintf(stderr, "merge: no usable biomap parts\n"); return 1; }
    out.planet_hash = ph;
    out.seed = hp->system_seed;
    out.shift = parts[0].shift;
    out.ncells = total;
    out.cells = (uint32_t*)malloc(sizeof(uint32_t) * (total ? total : 1));
    uint8_t* bins = (uint8_t*)malloc(total ? total : 1);
    int rc = 1;
    int any_morph = 0;
    for (int k = 0; k < np; ++k) any_morph |= parts[k].morph != NULL;
    if (any_morph) out.morph = (uint32_t*)calloc(total ? total : 1, sizeof(uint32_t));     /* version 2: parts without shapes stay 0 */
    if (!out.cells || !bins || (any_morph && !out.morph)) { fprintf(stderr, "merge: out of memory\n"); goto done; }
    memset(bins, 0xFF, total ? total : 1);       /* cells outside every region keep 255: not counted */
    {
        size_t off = 0;
        int64_t atm[BIOMAP_ATM] = { 0 };
        int natm = 0;
        for (int k = 0; k < np; ++k) {
            memcpy(out.cells + off, parts[k].cells, sizeof(uint32_t) * parts[k].ncells);
            if (out.morph && parts[k].morph) memcpy(out.morph + off, parts[k].morph, sizeof(uint32_t) * parts[k].ncells);
            if (parts[k].atm_known) { for (int g = 0; g < BIOMAP_ATM; ++g) atm[g] += parts[k].atm_ppm[g]; ++natm; }
            off += parts[k].ncells;
        }
        /* the sky: the mean of the islands' atmospheres (with the hub they are one) */
        for (int g = 0; g < BIOMAP_ATM && natm; ++g) out.atm_ppm[g] = (int32_t)(atm[g] / natm);
        out.atm_known = natm > 0;
    }
    /* combined fallback: the environment class of every region cell (as bio_env_build
     * stores it), then the same statistics as bio_sim_vegetation over all regions */
    {
        BinCtx B = { &out, &hp->gen, hp->system_seed, bins };
        uint32_t rows = 0;
        for (int r = 0; r < out.nreg; ++r) rows += (uint32_t)out.reg[r].h;
        /* rows are laid out region after region; offsets of merged parts are contiguous in that order */
        jobs_parallel_for(rows, 4, bins_rows, &B);
        static int64_t ncell[64], kcnt[64][6], kcov[64][6], khgt[64][6], kpal[64][6][16];
        memset(ncell, 0, sizeof ncell); memset(kcnt, 0, sizeof kcnt); memset(kcov, 0, sizeof kcov);
        memset(khgt, 0, sizeof khgt); memset(kpal, 0, sizeof kpal);
        for (size_t c = 0; c < total; ++c) {
            if (bins[c] > 63) continue;
            const int b = bins[c];
            ncell[b]++;
            const uint32_t v = out.cells[c], kd = BIOCELL_KIND(v);
            if (!kd || kd > 5) continue;
            kcnt[b][kd]++; kcov[b][kd] += BIOCELL_COVER(v); khgt[b][kd] += BIOCELL_HEIGHT(v); kpal[b][kd][BIOCELL_PAL(v)]++;
        }
        for (int b = 0; b < 64; ++b) {
            int best = 0;
            for (int q = 1; q <= 5; ++q) if (kcnt[b][q] > kcnt[b][best]) best = q;
            if (!best || !ncell[b]) { out.fallback[b] = 0; continue; }
            uint32_t pi = 0;
            for (uint32_t q = 1; q < 16; ++q) if (kpal[b][best][q] > kpal[b][best][pi]) pi = q;
            const int64_t cov = kcov[b][best] / ncell[b], hgt = khgt[b][best] / kcnt[b][best];
            out.fallback[b] = cov > 0 ? BIOCELL((uint32_t)best, pi, (uint32_t)cov, (uint32_t)hgt, 0u) : 0u;
        }
        if (out.morph) {      /* the commonest shape per environment class, among its cells of the fallback kind */
            static int64_t mc[64][5], ma[64][8], ml[64][4], mp[64][4], mq[64][16], mpig[64], mn[64];
            memset(mc, 0, sizeof mc); memset(ma, 0, sizeof ma); memset(ml, 0, sizeof ml); memset(mp, 0, sizeof mp);
            memset(mq, 0, sizeof mq); memset(mpig, 0, sizeof mpig); memset(mn, 0, sizeof mn);
            for (size_t c = 0; c < total; ++c) {
                const int b = bins[c];
                const uint32_t mw = out.morph[c];
                if (b > 63 || !(mw & BIOMORPH_VALID) || !out.fallback[b] || BIOCELL_KIND(out.cells[c]) != BIOCELL_KIND(out.fallback[b])) continue;
                if (BIOMORPH_CROWN(mw) < 5) mc[b][BIOMORPH_CROWN(mw)]++;
                ma[b][BIOMORPH_ASPECT(mw)]++; ml[b][BIOMORPH_LEAFANG(mw)]++; mp[b][BIOMORPH_POROSITY(mw)]++; mq[b][BIOMORPH_QWIDTH(mw)]++;
                mpig[b] += BIOMORPH_PIG(mw); mn[b]++;
            }
            for (int b = 0; b < 64; ++b) {
                if (!mn[b]) continue;
                uint32_t bc = 0, ba = 0, bl = 0, bp = 0, bq = 0;
                for (uint32_t q = 1; q < 5; ++q) if (mc[b][q] > mc[b][bc]) bc = q;
                for (uint32_t q = 1; q < 8; ++q) if (ma[b][q] > ma[b][ba]) ba = q;
                for (uint32_t q = 1; q < 4; ++q) if (ml[b][q] > ml[b][bl]) bl = q;
                for (uint32_t q = 1; q < 4; ++q) if (mp[b][q] > mp[b][bp]) bp = q;
                for (uint32_t q = 1; q < 16; ++q) if (mq[b][q] > mq[b][bq]) bq = q;
                out.fallback_morph[b] = BIOMORPH(bc, ba, bl, bp, (uint32_t)(mpig[b] / mn[b]), bq);
            }
        }
        if (np == 1) {
            int same = 1;
            for (int b = 0; b < 64; ++b) same &= out.fallback[b] == parts[0].fallback[b];
            printf("merge: fallback recomputed from the planet %s the part's own table\n", same ? "equals" : "DIFFERS FROM");
        }
    }
    for (int b = 0; b < 64; ++b) out.fallback_on |= out.fallback[b] != 0;
    if (o->no_fallback) out.fallback_on = 0;
    biomap_default_palette(out.palette);
    {
        const char* path = o->out ? o->out : DEFAULT_BIOMAP;
        if (mkdirs_for(path) || biomap_save(path, &out)) { fprintf(stderr, "merge: cannot write %s\n", path); goto done; }
        print_kinds("merge", out.cells, out.ncells);
        print_fallback(out.fallback, out.fallback_on);
        if (out.morph) print_shapes("merge", out.cells, out.morph, out.ncells);
        printf("merge: wrote %s (%s): %d region(s), sim year %u, gates %02x, planet %016llx seed %u\n", path,
               out.morph ? "version 2" : "version 1", out.nreg, out.sim_year, out.gates, (unsigned long long)ph, out.seed);
        if (out.atm_known)
            printf("merge: atmosphere for the sky: H2S %.2f%% CH4 %.2f%% NH3 %.2f%% CO2 %.3f%% O2 %.3f%% N2 %.2f%% H2 %.3f%%\n", out.atm_ppm[0] / 1e4,
                   out.atm_ppm[1] / 1e4, out.atm_ppm[2] / 1e4, out.atm_ppm[3] / 1e4, out.atm_ppm[4] / 1e4, out.atm_ppm[5] / 1e4, out.atm_ppm[6] / 1e4);
        /* creatures: fauna_<K>.bin beside each part -> one file beside the biomap (home.fauna) */
        Fauna all;
        memset(&all, 0, sizeof all);
        all.rec = (FaunaRec*)malloc(sizeof(FaunaRec) * (size_t)FAUNA_MAX * (size_t)(np > 0 ? np : 1));
        uint32_t nf = 0, nfiles = 0;
        for (int k = 0; all.rec && k < np_paths; ++k) {
            char fp[1300];
            snprintf(fp, sizeof fp, "%s", part_path[k]);
            char* sl = strrchr(fp, '/');
            char* bs = strrchr(fp, '\\');
            if (bs && (!sl || bs > sl)) sl = bs;
            const char* base = sl ? sl + 1 : fp;
            int isl = -1;
            if (sscanf(base, "biopart_%d.bin", &isl) != 1) continue;
            snprintf(sl ? sl + 1 : fp, sizeof fp - (size_t)(sl ? sl + 1 - fp : 0), "fauna_%d.bin", isl);
            Fauna f1;
            if (fauna_load(fp, &f1)) continue;
            if (f1.planet_hash == ph) {
                memcpy(all.rec + nf, f1.rec, sizeof(FaunaRec) * f1.count);
                nf += f1.count; all.total += f1.total; ++nfiles;
                memcpy(all.palette, f1.palette, sizeof all.palette);
            }
            fauna_free(&f1);
        }
        if (all.rec && nfiles) {
            qsort(all.rec, nf, sizeof(FaunaRec), fauna_size_cmp);
            all.count = nf > FAUNA_MAX ? FAUNA_MAX : nf;
            all.planet_hash = ph; all.seed = hp->system_seed; all.sim_year = out.sim_year;
            char fo[1300];
            if (o->fauna) snprintf(fo, sizeof fo, "%s", o->fauna);
            else {
                snprintf(fo, sizeof fo, "%s", path);
                char* dot = strrchr(fo, '.');
                char* sl = strrchr(fo, '/');
                if (dot && (!sl || dot > sl)) *dot = 0;
                strncat(fo, ".fauna", sizeof fo - strlen(fo) - 1);
            }
            if (mkdirs_for(fo) || fauna_save(fo, &all)) fprintf(stderr, "merge: cannot write %s\n", fo);
            else printf("merge: wrote %s: %u of %u creatures from %u island file(s)\n", fo, all.count, all.total, nfiles);
        }
        free(all.rec);
    }
    rc = 0;
done:
    for (int k = 0; k < np; ++k) biomap_free(&parts[k]);
    free(out.cells);
    free(out.morph);
    free(bins);
    return rc;
}

static int stage_bioexport(const Opt* o) {
    HomePlanet hp;
    if (load_planet(o->planet, &hp)) return 1;
    if (o->nmerge) return export_merge(o, &hp);
    return export_one(o, &hp);
}

/* ======================================================================================
 * biomap-test, life-test
 * ====================================================================================== */
/* the engine's spawn without life (main.c spawn_default): spiral out from the
 * origin on a 64 m lattice until the ground is 40 m above the sea */
static void engine_spawn(uint32_t seed, const GmePlanet* pl, int32_t* x, int32_t* z) {
    int32_t sx = 16, sz = 16;
    const int32_t want = (pl->sea_level > -200 ? pl->sea_level : 0) + 40;
    for (int ring = 0, found = 0; ring < 128 && !found; ++ring)
        for (int k = -ring; k <= ring && !found; ++k) {
            const int32_t c[4][2] = { { k, -ring }, { k, ring }, { -ring, k }, { ring, k } };
            for (int e = 0; e < 4 && !found; ++e) {
                const int32_t px = c[e][0] * 64, pz = c[e][1] * 64;
                if (gme_height(px, pz, seed, pl) > want) { sx = px; sz = pz; found = 1; }
            }
        }
    *x = sx; *z = sz;
}

static int stage_biomaptest(const Opt* o) {
    HomePlanet hp;
    if (load_planet(o->planet, &hp)) return 1;
    int32_t cx, cz;
    engine_spawn(hp.system_seed, &hp.gen, &cx, &cz);
    if (o->xz_set & 1) cx = o->x;
    if (o->xz_set & 2) cz = o->z;
    Biomap b;
    if (biomap_make_test(&b, planet_hash(&hp), hp.system_seed, cx, cz)) { fprintf(stderr, "biomap-test: out of memory\n"); return 1; }
    const char* path = o->out ? o->out : "life/test.biomap";
    const int rc = mkdirs_for(path) || biomap_save(path, &b);
    if (rc) fprintf(stderr, "biomap-test: cannot write %s\n", path);
    else printf("biomap-test: wrote %s: test pattern centred on (%d, %d), region x %d z %d (%dx%d cells of %d m), planet %016llx seed %u\n",
                path, cx, cz, b.reg[0].x0, b.reg[0].z0, b.reg[0].w, b.reg[0].h, 1 << b.shift,
                (unsigned long long)b.planet_hash, b.seed);
    biomap_free(&b);
    return rc ? 1 : 0;
}

static int stage_lifetest(const Opt* o) {
    const int device = o->device_set ? o->device : 0;
    const int failed = bio_run_tests(device, !o->quiet);
    printf("%s: %d test(s) failed\n", failed ? "FAIL" : "OK", failed);
    return failed > 100 ? 100 : failed;
}

/* ======================================================================================
 * hub, seed bank, rebase, inoculation
 * ====================================================================================== */
static int parse_int_list(const char* list, int* out, int max) {
    int n = 0;
    char buf[512];
    snprintf(buf, sizeof buf, "%s", list ? list : "");
    for (char* t = strtok(buf, ", "); t && n < max; t = strtok(NULL, ", ")) out[n++] = atoi(t);
    return n;
}
static int split_list(const char* list, char out[][1024], int max) {
    int n = 0;
    const char* p = list ? list : "";
    while (*p && n < max) {
        const char* e = strchr(p, ',');
        const size_t l = e ? (size_t)(e - p) : strlen(p);
        if (l && l < 1024) { memcpy(out[n], p, l); out[n][l] = 0; ++n; }
        if (!e) break;
        p = e + 1;
    }
    return n;
}

/* One pass of the island hub (bio_hub.c). With --runs the islands' run folders are
 * on this machine and the hub carries the files itself; for rigs cluster/bio_hub.ps1
 * copies outboxes into DIR/in before, and DIR/out/<k> into the inboxes after. */
static int stage_hub(const Opt* o) {
    HomePlanet hp;
    if (load_planet(o->planet, &hp)) return 1;
    int isl[16];
    const int n = parse_int_list(o->islands, isl, 16);
    if (n < 1) { fprintf(stderr, "life-hub needs --islands 0,1,...\n"); return 2; }
    const char* dir = o->dir ? o->dir : DEFAULT_HUB;
    static char runs[16][1024];
    const int nr = split_list(o->runs, runs, 16);
    if (nr && nr != n) { fprintf(stderr, "life-hub: --runs needs one folder per island of --islands (%d given for %d islands)\n", nr, n); return 2; }
    char a[1200], b[1200];
    snprintf(b, sizeof b, "%s/in", dir);
    for (int k = 0; k < nr; ++k) {             /* in/ holds, per island, what its outbox holds now */
        char pre[2][32];
        snprintf(a, sizeof a, "%s/outbox", runs[k]);
        snprintf(pre[0], sizeof pre[0], "atm_%d_", isl[k]);
        snprintf(pre[1], sizeof pre[1], "mig_%d_", isl[k]);
        for (int q = 0; q < 2; ++q)
            if (bio_dir_mirror(a, b, pre[q]) < 0) { fprintf(stderr, "life-hub: cannot copy %s to %s\n", a, b); return 1; }
    }
    static char report[4096];
    const int32_t ocean_ppm = (int32_t)lround(hp.ocean_cover * 1e6);
    const int rc = bio_hub_run(dir, isl, n, ocean_ppm, report, sizeof report);
    if (rc < 0) { fprintf(stderr, "life-hub: %s\n", bio_error()); return 1; }
    fputs(report, stdout);
    printf("hub: planet ocean cover %.1f%%; state in %s/hub_state.txt, atmosphere in %s/atm_global.txt\n", ocean_ppm / 1e4, dir, dir);
    for (int k = 0; k < nr; ++k) {
        snprintf(a, sizeof a, "%s/out/%d", dir, isl[k]);
        snprintf(b, sizeof b, "%s/inbox", runs[k]);
        const int c = bio_dir_copy(a, b, "");
        if (c < 0) { fprintf(stderr, "life-hub: cannot copy %s to %s\n", a, b); return 1; }
        bio_dir_clear(a);
        printf("hub: island %d: %d file(s) into %s\n", isl[k], c, b);
    }
    return 0;
}

/* A run folder opened on its own authority: env.bin says which planet and island it
 * is, no planet file needed (the run may be of another planet). CPU unless --device. */
static int open_run_any(const Opt* o, const char* dir, BioEnv* e, BioSim** s, uint64_t* step) {
    char path[1100];
    snprintf(path, sizeof path, "%s/env.bin", dir);
    uint32_t q = 0, fc = 0;
    const int rc = bio_env_load(path, e, &q, &fc);
    if (rc) { fprintf(stderr, "%s: %s\n", path, rc == -1 ? "cannot read it" : rc == -3 ? "CRC mismatch (corrupt)" : "not a valid env file"); return 1; }
    BioConfig cfg;
    bio_config_default(&cfg);
    cfg.device = o->device_set ? o->device : -1;
    cfg.duty = 1.0;
    static SpectrumFile sf;                      /* needed only when the run uses light in bands */
    if (!o->no_spectrum && !spectrum_load(o->spectrum ? o->spectrum : DEFAULT_SPECTRUM, &sf)) cfg.spectrum = &sf;
    *s = ck_load_newest(dir, e, &cfg, step, 1);
    if (!*s) { fprintf(stderr, "no valid checkpoint in %s\n", dir); bio_env_free(e); return 1; }
    return 0;
}

typedef struct DirList { char name[64][160]; int n; } DirList;
static void dir_add(void* ctx, const char* name) {
    DirList* L = (DirList*)ctx;
    if (L->n < 64 && strlen(name) < 160) snprintf(L->name[L->n++], 160, "%s", name);
}

static void bank_table(const BioBankRec* r, uint32_t n) {
    int cls[BIO_GUILD_N][BIO_HAB_N];
    memset(cls, 0, sizeof cls);
    for (uint32_t k = 0; k < n; ++k) cls[r[k].guild][r[k].habitat]++;
    printf("  genomes        marine  coastal     land\n");
    for (int g = 0; g < BIO_GUILD_N; ++g)
        printf("  %-12s %8d %8d %8d\n", bio_guild_name(g), cls[g][BIO_HAB_MARINE], cls[g][BIO_HAB_COASTAL], cls[g][BIO_HAB_LAND]);
}

/* life-seedbank: the best genomes per guild x habitat of every island under --runs */
static int stage_seedbank(const Opt* o) {
    if (!o->runs) { fprintf(stderr, "life-seedbank needs --runs DIR (a folder of island_* runs, or one run)\n"); return 2; }
    static DirList L;
    L.n = 0;
    char probe[1100];
    snprintf(probe, sizeof probe, "%s/env.bin", o->runs);
    if (file_exists(probe)) dir_add(&L, ".");
    else list_dir(o->runs, "island_", "", dir_add, &L);
    if (!L.n) { fprintf(stderr, "life-seedbank: no run (env.bin) in %s or in its island_* folders\n", o->runs); return 1; }
    qsort(L.name, (size_t)L.n, sizeof L.name[0], cmp_str);
    const uint32_t per = o->n ? o->n : 2000;
    BioBankRec* all = NULL;
    uint32_t na = 0;
    uint64_t ph = 0;
    int used = 0;
    for (int k = 0; k < L.n; ++k) {
        char dir[1300];
        if (!strcmp(L.name[k], ".")) snprintf(dir, sizeof dir, "%s", o->runs);
        else snprintf(dir, sizeof dir, "%s/%s", o->runs, L.name[k]);
        snprintf(probe, sizeof probe, "%s/env.bin", dir);
        if (!file_exists(probe)) continue;
        BioEnv e;
        BioSim* s = NULL;
        uint64_t step = 0;
        if (open_run_any(o, dir, &e, &s, &step)) continue;
        if (used && e.planet_hash != ph) {
            fprintf(stderr, "life-seedbank: %s is of planet %016llx, the others of %016llx: skipped\n", dir, (unsigned long long)e.planet_hash, (unsigned long long)ph);
            bio_sim_destroy(s); bio_env_free(&e);
            continue;
        }
        ph = e.planet_hash;
        BioBankRec* r = NULL;
        uint32_t n = 0;
        if (bio_sim_bank(s, &r, &n)) fprintf(stderr, "life-seedbank: %s: %s\n", dir, bio_error());
        else {
            const uint32_t kept = bio_bank_top(r, n, per);      /* per island first: bounds the memory */
            BioBankRec* grown = (BioBankRec*)realloc(all, sizeof(BioBankRec) * ((size_t)na + kept + 1));
            if (grown) { all = grown; memcpy(all + na, r, sizeof(BioBankRec) * kept); na += kept; ++used; }
            printf("seedbank: %s: island %d at year %.1f, %u live agents, %u genome x habitat records, %u kept\n", dir, e.island,
                   step / (double)STEPS_YEAR, s->n_live, n, kept);
        }
        free(r);
        bio_sim_destroy(s); bio_env_free(&e);
    }
    if (!used) { fprintf(stderr, "life-seedbank: no run could be read\n"); free(all); return 1; }
    na = bio_bank_top(all, na, per);
    const char* out = o->out ? o->out : DEFAULT_BANK;
    if (mkdirs_for(out) || bio_bank_save(out, all, na, ph)) { fprintf(stderr, "life-seedbank: cannot write %s: %s\n", out, bio_error()); free(all); return 1; }
    printf("seedbank: wrote %s: %u genomes of %d island(s) of planet %016llx, the best %u per guild x habitat\n", out, na, used, (unsigned long long)ph, per);
    bank_table(all, na);
    free(all);
    return 0;
}

/* life-rebase: a new run on this planet founded from a seed bank; writes its
 * first checkpoint and stops (life --island K --resume takes it from there) */
static int stage_life(const Opt* o);
static int stage_rebase(const Opt* o) {
    if (!o->bank || o->island < 0) { fprintf(stderr, "life-rebase needs --bank FILE and --island K\n"); return 2; }
    Opt o2 = *o;
    o2.stage = "life";
    o2.resume = 0;
    o2.stop_at_start = 1;
    o2.seed_n = o->n ? o->n : 20000;
    return stage_life(&o2);
}

/* "producer@land", "consumer", "land_plant", ... -> guild, habitat (-1 = any); 0 ok */
static int parse_guild(const char* spec, int* guild, int* habitat) {
    *guild = -1; *habitat = -1;
    if (!spec || !*spec) return -1;
    if (!strcmp(spec, "land_plant")) { *guild = BIO_GUILD_PRODUCER; *habitat = BIO_HAB_LAND; return 0; }
    if (!strcmp(spec, "land_animal")) { *guild = BIO_GUILD_CONSUMER; *habitat = BIO_HAB_LAND; return 0; }
    char buf[96];
    snprintf(buf, sizeof buf, "%s", spec);
    char* at = strchr(buf, '@');
    if (at) *at++ = 0;
    if (strcmp(buf, "any") && strcmp(buf, "all")) {
        for (int g = 0; g < BIO_GUILD_N; ++g) if (!strcmp(buf, bio_guild_name(g))) *guild = g;
        if (*guild < 0) return -1;
    }
    if (at && strcmp(at, "any")) {
        for (int h = 0; h < BIO_HAB_N; ++h) if (!strcmp(at, bio_habitat_name(h))) *habitat = h;
        if (*habitat < 0) return -1;
    }
    return 0;
}

/* life-inoculate: evolved genomes of one guild from run A's newest checkpoint into
 * run B's, into cells of their habitat. Genomes only, never behaviour; logged. */
static int stage_inoculate(const Opt* o) {
    int guild, habitat;
    if (!o->from || !o->to || parse_guild(o->guild, &guild, &habitat)) {
        fprintf(stderr, "life-inoculate needs --from RUN --to RUN --guild G, G = producer | chemotroph | decomposer | consumer | any, "
                        "optionally @marine | @coastal | @land, or land_plant, land_animal\n");
        return 2;
    }
    const uint32_t want = o->n ? o->n : 500;
    BioEnv ea, eb;
    BioSim* a = NULL; BioSim* b = NULL;
    uint64_t sa = 0, sb = 0;
    if (open_run_any(o, o->from, &ea, &a, &sa)) return 1;
    BioBankRec* r = NULL;
    uint32_t n = 0;
    int rc = 1;
    if (bio_sim_bank(a, &r, &n)) { fprintf(stderr, "life-inoculate: %s\n", bio_error()); bio_sim_destroy(a); bio_env_free(&ea); return 1; }
    n = bio_bank_top(r, n, 2000);
    const uint64_t pha = ea.planet_hash;
    const int isla = ea.island;
    bio_sim_destroy(a); bio_env_free(&ea);
    if (open_run_any(o, o->to, &eb, &b, &sb)) { free(r); return 1; }
    const uint32_t before = b->n_live;
    const int placed = bio_sim_seed_bank(b, r, n, want, guild, habitat);
    if (placed <= 0) {
        fprintf(stderr, "life-inoculate: nothing placed: %s\n", placed < 0 ? bio_error() : "the source has no such genomes, or the target island no such habitat");
        goto out;
    }
    {   /* the checkpoint it replaces is kept aside */
        char keep[1200], pre[64], kept[1300];
        snprintf(keep, sizeof keep, "%s/before_inoculate", o->to);
        snprintf(pre, sizeof pre, "ckpt_%012llu", (unsigned long long)sb);
        snprintf(kept, sizeof kept, "%s/%s.bin", keep, pre);
        FILE* kf = fopen(kept, "rb");
        if (kf) fclose(kf);      /* an earlier inoculation at this step: that copy is the untouched checkpoint and stays */
        else if (bio_dir_copy(o->to, keep, pre) < 1) {
            fprintf(stderr, "life-inoculate: cannot keep a copy of the checkpoint in %s: nothing was changed\n", keep);
            goto out;
        }
        double secs, mb;
        if (ck_save(b, o->to, sb, &secs, &mb)) goto out;
        stage_log(o->to, "step=%llu year=%.3f INTERVENTION inoculate %d founders of guild %s@%s from %s (island %d of planet %016llx, year %.1f); live %u -> %u",
                  (unsigned long long)sb, sb / (double)STEPS_YEAR, placed, bio_guild_name(guild), bio_habitat_name(habitat), o->from, isla,
                  (unsigned long long)pha, sa / (double)STEPS_YEAR, before, b->n_live);
        printf("inoculate: %d founders of guild %s@%s from %s (island %d, planet %016llx) into %s (island %d): live %u -> %u\n", placed,
               bio_guild_name(guild), bio_habitat_name(habitat), o->from, isla, (unsigned long long)pha, o->to, eb.island, before, b->n_live);
        printf("inoculate: wrote ckpt_%012llu.bin (%.0f MB; the one before is in %s) and logged the intervention in stages.log\n",
               (unsigned long long)sb, mb, keep);
        rc = 0;
    }
out:
    free(r);
    bio_sim_destroy(b); bio_env_free(&eb);
    return rc;
}

/* ======================================================================================
 * dispatch
 * ====================================================================================== */
int bio_main(int argc, char** argv) {
    if (argc < 2) return -1;
    static const struct { const char* name; int (*fn)(const Opt*); } T[] = {
        { "env", stage_env }, { "life", stage_life }, { "life-test", stage_lifetest }, { "bioexport", stage_bioexport },
        { "biostat", stage_biostat }, { "biomap-test", stage_biomaptest },
        { "life-hub", stage_hub }, { "life-seedbank", stage_seedbank }, { "life-rebase", stage_rebase },
        { "life-inoculate", stage_inoculate },
    };
    int (*fn)(const Opt*) = NULL;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; ++i) if (!strcmp(argv[1], T[i].name)) fn = T[i].fn;
    if (!fn) return -1;
    Opt o;
    if (parse(argc, argv, &o)) { usage(); return 2; }
    platform_init();
    SysInfo si;
    sys_query(&si);
    jobs_init(o.threads > 0 ? o.threads - 1 : (si.logical_cpus > 1 ? si.logical_cpus - 1 : 0));
#ifndef _WIN32
    setvbuf(stdout, NULL, _IOLBF, 1 << 14);   /* line by line into life.log on the rigs (MSVC has no line buffering) */
#endif
    const int rc = fn(&o);
    fflush(stdout);
    jobs_shutdown();
    platform_shutdown();
    return rc;
}
