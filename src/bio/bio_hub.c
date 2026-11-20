/* bio_hub.c - the island hub (docs/biosphere-spec.md 2.5.7, 6.4): one
 * planet-wide atmosphere and migrants between islands. Integer arithmetic for
 * everything that reaches a simulation, plain files, no network: a script
 * (cluster/bio_hub.ps1) only carries files between the rigs and the PC.
 *
 * Island side, once per epoch (a sim-year), in its run folder:
 *   outbox/mig_<k>_<epoch>.bin     its most successful genomes per guild and
 *                                  habitat (GMEMIG01, bio_sim_export_emigrants)
 *   outbox/atm_<k>_<epoch>.txt     its box and its cumulative gas uptake by realm
 *   inbox/mig_<j>_<e>.bin          migrants from other islands: imported
 *   inbox/atm_global_<n>.txt       the planet's atmosphere: the box relaxes a
 *                                  quarter of the way to it
 *   imported/e<epoch>_<name>       what was applied at which epoch, and
 *   imported/last_live.txt         the last epoch at which the inbox was read.
 *                                  An island resumed from an older checkpoint
 *                                  applies the same files at the same epochs
 *                                  again (and nothing where nothing was
 *                                  applied), so its history repeats exactly;
 *                                  what arrived meanwhile waits in inbox/.
 * Hub side (PC), over a folder holding in/ (copies of the outboxes):
 *   hub_state.txt                  the global column, the last file seen per island
 *   atm_global.txt                 the same, to read
 *   out/<k>/...                    what island k's inbox gets next
 * The global column is kept per m^2 of planet at the islands' box scale, in
 * umol. It moves by the islands' uptake per m^2 of ocean and of land, weighted
 * by the planet's ocean cover (the islands are coasts: far more sea than the
 * planet has), and by the same photochemistry as an island box. Islands run at
 * different speeds; the hub's clock is the mean of their epochs. */
#include "bio_internal.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ---- small file helpers ------------------------------------------------------------- */
int bio_mkdir_p(const char* path) {
    char d[1100];
    snprintf(d, sizeof d, "%s", path);
    for (char* p = d + 1; ; ++p) {
        if (*p == '/' || *p == '\\' || *p == 0) {
            const char keep = *p;
            *p = 0;
            if (!(p > d && p[-1] == ':')) {
#ifdef _WIN32
                _mkdir(d);
#else
                mkdir(d, 0775);
#endif
            }
            *p = keep;
            if (!keep) break;
        }
    }
#ifdef _WIN32
    const DWORD at = GetFileAttributesA(path);
    return (at != INVALID_FILE_ATTRIBUTES && (at & FILE_ATTRIBUTE_DIRECTORY)) ? 0 : -1;
#else
    struct stat st;
    return (!stat(path, &st) && S_ISDIR(st.st_mode)) ? 0 : -1;
#endif
}

typedef struct NameList { char (*name)[160]; int n, cap; } NameList;
static void names_free(NameList* L) { free(L->name); L->name = NULL; L->n = L->cap = 0; }
static int name_cmp(const void* a, const void* b) { return strcmp((const char*)a, (const char*)b); }
/* file names of dir that start with prefix and end with suffix, sorted */
static int names_list(const char* dir, const char* prefix, const char* suffix, NameList* L) {
    L->name = NULL; L->n = 0; L->cap = 0;
    const size_t lp = strlen(prefix), ls = strlen(suffix);
#ifdef _WIN32
    char pat[1200];
    snprintf(pat, sizeof pat, "%s\\*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        const char* nm = fd.cFileName;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
#else
    DIR* d = opendir(dir);
    if (!d) return 0;
    struct dirent* de;
    while ((de = readdir(d)) != NULL) {
        const char* nm = de->d_name;
        if (nm[0] == '.') continue;
        {   /* plain files only, as on Windows (fopen opens a directory here) */
            char full[1500];
            struct stat st;
            snprintf(full, sizeof full, "%s/%s", dir, nm);
            if (stat(full, &st) || !S_ISREG(st.st_mode)) continue;
        }
#endif
        const size_t l = strlen(nm);
        if (l >= 160 || l < lp + ls || strncmp(nm, prefix, lp) || strcmp(nm + l - ls, suffix)) continue;
        if (L->n == L->cap) {
            const int nc = L->cap ? L->cap * 2 : 64;
            void* p = realloc(L->name, (size_t)nc * 160);
            if (!p) break;
            L->name = (char (*)[160])p; L->cap = nc;
        }
        snprintf(L->name[L->n++], 160, "%s", nm);
#ifdef _WIN32
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    }
    closedir(d);
#endif
    if (L->n) qsort(L->name, (size_t)L->n, 160, name_cmp);
    return L->n;
}

static int file_move(const char* from, const char* to) {
#ifdef _WIN32
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
#else
    return rename(from, to);
#endif
}
static int file_copy(const char* from, const char* to) {
    FILE* f = fopen(from, "rb");
    if (!f) return -1;
    char tmp[1300];
    snprintf(tmp, sizeof tmp, "%s.tmp", to);
    FILE* g = fopen(tmp, "wb");
    if (!g) { fclose(f); return -1; }
    char buf[65536];
    size_t n;
    int ok = 1;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) if (fwrite(buf, 1, n, g) != n) { ok = 0; break; }
    fclose(f);
    if (fclose(g)) ok = 0;
    if (ok) { remove(to); ok = rename(tmp, to) == 0; }
    if (!ok) remove(tmp);
    return ok ? 0 : -1;
}
/* "key v0 v1 ..." lines of a small text file */
static int kv_read(const char* path, const char* key, int64_t* v, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char line[1024];
    const size_t lk = strlen(key);
    int got = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, lk) || (line[lk] != ' ' && line[lk] != '\t')) continue;
        const char* p = line + lk;
        got = 0;
        for (int k = 0; k < n; ++k) {
            char* end = NULL;
            const long long x = strtoll(p, &end, 10);
            if (end == p) break;
            v[k] = x; p = end; ++got;
        }
        break;
    }
    fclose(f);
    return got == n ? 0 : -1;
}

/* "key <hex>" (the planet hash); 0 when the file has no such line */
static uint64_t kv_hex(const char* path, const char* key) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    char line[1024];
    const size_t lk = strlen(key);
    uint64_t v = 0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, lk) && (line[lk] == ' ' || line[lk] == '\t')) { v = strtoull(line + lk, NULL, 16); break; }
    fclose(f);
    return v;
}

/* ---- island side ---------------------------------------------------------------------- */
#define HUB_KEEP_EPOCHS 1024      /* imported/ keeps the records of this many epochs back (replay of the kept checkpoints) */
uint64_t bio_sim_hub_epoch(const BioSim* s) { return s ? s->hub_epoch : 0; }

int bio_hub_export(BioSim* s, const char* run_dir, uint64_t epoch, int n_migrants) {
    if (!s || !run_dir) return -1;
    char dir[1100], path[1300], tmp[1340];
    snprintf(dir, sizeof dir, "%s/outbox", run_dir);
    if (bio_mkdir_p(dir)) { bio_set_error("cannot create %s", dir); return -1; }
    const int k = s->env->island;
    BioStats st;
    if (bio_sim_stats(s, &st)) return -1;
    int64_t ocean = 0, land = 0;
    for (int64_t c = 0; c < (int64_t)s->env->w * s->env->h; ++c) { if (s->env->depth[c] >= 1) ++ocean; else ++land; }
    snprintf(path, sizeof path, "%s/atm_%d_%08llu.txt", dir, k, (unsigned long long)epoch);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) { bio_set_error("cannot write %s", tmp); return -1; }
    fprintf(f, "GMEATM1\nisland %d\nepoch %llu\nstep %llu\nplanet %016llx\nscale_ppm %d\ncells_ocean %lld\ncells_land %lld\ncell_m2 %lld\n",
            k, (unsigned long long)epoch, (unsigned long long)st.step, (unsigned long long)s->planet_hash, s->w.P.atm_scale_ppm,
            (long long)ocean, (long long)land, (long long)((int64_t)1 << (2 * s->env->shift)));
    fprintf(f, "box");
    for (int g = 0; g < BIO_ATM_N; ++g) fprintf(f, " %lld", (long long)st.atm[g]);
    fprintf(f, "\nxch_all");
    for (int g = 0; g < BIO_ATM_N; ++g) fprintf(f, " %lld", (long long)st.xch_all[g]);
    fprintf(f, "\nxch_land");
    for (int g = 0; g < BIO_ATM_N; ++g) fprintf(f, " %lld", (long long)st.xch_land[g]);
    fprintf(f, "\n");
    if (fclose(f)) { remove(tmp); bio_set_error("cannot write %s", tmp); return -1; }
    remove(path);
    if (rename(tmp, path)) { remove(tmp); bio_set_error("cannot write %s", path); return -1; }
    if (n_migrants > 0) {
        snprintf(path, sizeof path, "%s/mig_%d_%08llu.bin", dir, k, (unsigned long long)epoch);
        if (bio_sim_export_emigrants(s, path, n_migrants) < 0) return -1;
    }
    /* an outbox is a mailbox, not an archive: keep the 4 newest of each kind */
    static const char* const kinds[2][2] = { { "atm_", ".txt" }, { "mig_", ".bin" } };
    for (int q = 0; q < 2; ++q) {
        NameList L;
        names_list(dir, kinds[q][0], kinds[q][1], &L);
        for (int i = 0; i + 4 < L.n; ++i) { snprintf(path, sizeof path, "%s/%s", dir, L.name[i]); remove(path); }
        names_free(&L);
    }
    return 0;
}

/* the planet's atmosphere file -> the island's box: a quarter of the way there */
static int apply_global(BioSim* s, const char* path) {
    int64_t g_umol[BIO_ATM_N], scale = 0;
    if (kv_read(path, "umol_m2", g_umol, BIO_ATM_N) || kv_read(path, "scale_ppm", &scale, 1) || scale <= 0) {
        bio_set_error("%s: not a global atmosphere file", path); return -1;
    }
    const uint64_t planet = kv_hex(path, "planet");
    if (planet && s->planet_hash && planet != s->planet_hash) { bio_set_error("%s: the air of another planet", path); return -1; }
    int64_t atm[BIO_ATM_N];
    if (bio_sim_get_atm(s, atm)) return -1;
    const int64_t area = (int64_t)s->env->w * s->env->h * ((int64_t)1 << (2 * s->env->shift));     /* m^2 */
    for (int g = 0; g < BIO_ATM_N; ++g) {
        /* umol / m^2 at the file's scale -> mmol in this island's box */
        int64_t target = bio_mulshr((uint64_t)(g_umol[g] > 0 ? g_umol[g] : 0), (uint64_t)area, 0) / 1000;
        if (scale != s->w.P.atm_scale_ppm) target = target / scale * s->w.P.atm_scale_ppm;
        atm[g] += (target - atm[g]) / 4;
        if (atm[g] < 0) atm[g] = 0;
    }
    return bio_sim_set_atm(s, atm);
}

/* Applies this epoch's inputs. An epoch up to the last one at which the inbox
 * was read (imported/last_live.txt) is a replay: exactly the files recorded for
 * it in imported/ are applied again, possibly none, and inbox/ is left alone.
 * A later epoch reads inbox/, moves what it applied to imported/ and becomes
 * the last live epoch. */
int bio_hub_import(BioSim* s, const char* run_dir, uint64_t epoch, int* n_migrants, int* atm_applied) {
    if (!s || !run_dir) return -1;
    if (n_migrants) *n_migrants = 0;
    if (atm_applied) *atm_applied = 0;
    char inbox[1100], done[1100], path[1400], to[1500], pre[32];
    snprintf(inbox, sizeof inbox, "%s/inbox", run_dir);
    snprintf(done, sizeof done, "%s/imported", run_dir);
    snprintf(pre, sizeof pre, "e%08llu_", (unsigned long long)epoch);
    const size_t lp = strlen(pre);
    NameList L;
    char livep[1200];
    snprintf(livep, sizeof livep, "%s/last_live.txt", done);
    int64_t last_live = 0;
    if (kv_read(livep, "epoch", &last_live, 1)) last_live = 0;
    const int replay = (int64_t)epoch <= last_live;
    if (replay) names_list(done, pre, "", &L);
    else names_list(inbox, "", "", &L);
    char own[32];
    snprintf(own, sizeof own, "mig_%d_", s->env->island);
    /* of several global atmosphere files only the newest counts */
    int last_atm = -1;
    for (int i = 0; i < L.n; ++i) if (!strncmp(L.name[i] + (replay ? lp : 0), "atm_global_", 11)) last_atm = i;
    int rc = 0;
    for (int i = 0; i < L.n && !rc; ++i) {
        const char* nm = L.name[i] + (replay ? lp : 0);
        snprintf(path, sizeof path, "%s/%s", replay ? done : inbox, L.name[i]);
        const size_t l = strlen(nm);
        int used = 0;
        if (!strncmp(nm, "mig_", 4) && l > 4 && !strcmp(nm + l - 4, ".bin")) {
            if (!strncmp(nm, own, strlen(own))) { if (!replay) remove(path); continue; }     /* its own emigrants, back by the ring */
            if (!replay) {         /* a backlog (the island was away, or did not read): only a sender's two newest files count */
                const char* u = strchr(nm + 4, '_');
                const size_t ls = u ? (size_t)(u - nm) + 1 : l;
                int later = 0;
                for (int q = i + 1; q < L.n; ++q) if (!strncmp(L.name[q], nm, ls)) ++later;
                if (later >= 2) { remove(path); continue; }
            }
            const int m = bio_sim_import_migrants(s, path);
            if (m < 0) { if (replay) { rc = -1; break; } remove(path); continue; }              /* a foreign or broken file: dropped */
            if (n_migrants) *n_migrants += m;
            used = 1;
        } else if (!strncmp(nm, "atm_global_", 11)) {
            if (i != last_atm) { if (!replay) remove(path); continue; }
            if (apply_global(s, path)) { if (replay) { rc = -1; break; } remove(path); continue; }
            if (atm_applied) *atm_applied = 1;
            used = 1;
        } else continue;                                                                       /* anything else stays where it is */
        if (used && !replay) {
            if (bio_mkdir_p(done)) { bio_set_error("cannot create %s", done); rc = -1; break; }
            snprintf(to, sizeof to, "%s/%s%s", done, pre, nm);
            if (file_move(path, to)) { bio_set_error("cannot move %s to %s", path, to); rc = -1; break; }
        }
    }
    names_free(&L);
    if (!rc && !replay) {                      /* this epoch read the inbox, whatever it found */
        char tmp[1300];
        snprintf(tmp, sizeof tmp, "%s.tmp", livep);
        FILE* f = bio_mkdir_p(done) ? NULL : fopen(tmp, "wb");
        if (!f) { bio_set_error("cannot write %s", tmp); rc = -1; }
        else {
            fprintf(f, "epoch %llu\n", (unsigned long long)epoch);
            if (fclose(f)) { remove(tmp); bio_set_error("cannot write %s", tmp); rc = -1; }
            else { remove(livep); if (rename(tmp, livep)) { remove(tmp); bio_set_error("cannot write %s", livep); rc = -1; } }
        }
        /* imported/ is a log for replays, not an archive: records older than HUB_KEEP_EPOCHS go */
        NameList O;
        names_list(done, "e", "", &O);
        for (int i = 0; i < O.n; ++i) {
            char* end = NULL;
            const unsigned long long ep = strtoull(O.name[i] + 1, &end, 10);
            if (end == O.name[i] + 9 && *end == '_' && ep + HUB_KEEP_EPOCHS < epoch) { snprintf(path, sizeof path, "%s/%s", done, O.name[i]); remove(path); }
        }
        names_free(&O);
    }
    if (!rc) s->hub_epoch = epoch;
    return rc;
}

/* ---- hub side --------------------------------------------------------------------------- */
#define HUB_MAX 16
typedef struct HubIsland {
    int      k, seen;                 /* seen: a file of this island has been taken into the state */
    int64_t  epoch, xall[BIO_ATM_N], xland[BIO_ATM_N], mig_epoch;
} HubIsland;
typedef struct HubState {
    int64_t updates, clock_q10;       /* hub clock in 1/1024 sim-years */
    int64_t g[BIO_ATM_N];             /* umol per m^2 at scale_ppm */
    int64_t scale_ppm;
    uint64_t planet;
    int      n;
    HubIsland isl[HUB_MAX];
} HubState;

static HubIsland* hub_island(HubState* S, int k) {
    for (int i = 0; i < S->n; ++i) if (S->isl[i].k == k) return &S->isl[i];
    if (S->n >= HUB_MAX) return NULL;
    HubIsland* h = &S->isl[S->n++];
    memset(h, 0, sizeof *h);
    h->k = k; h->mig_epoch = -1;
    return h;
}

static void hub_load(const char* path, HubState* S) {
    memset(S, 0, sizeof *S);
    FILE* f = fopen(path, "rb");
    if (!f) return;
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        long long a, b, c;
        unsigned long long ph;
        if (sscanf(line, "planet %llx", &ph) == 1) S->planet = ph;
        else if (sscanf(line, "updates %lld", &a) == 1) S->updates = a;
        else if (sscanf(line, "clock_q10 %lld", &a) == 1) S->clock_q10 = a;
        else if (sscanf(line, "scale_ppm %lld", &a) == 1) S->scale_ppm = a;
        else if (!strncmp(line, "global ", 7)) {
            const char* p = line + 7;
            for (int g = 0; g < BIO_ATM_N; ++g) { char* e = NULL; S->g[g] = strtoll(p, &e, 10); p = e; }
        } else if (sscanf(line, "island %lld epoch %lld mig %lld", &a, &b, &c) == 3) {
            HubIsland* h = hub_island(S, (int)a);
            if (!h) continue;
            h->seen = 1; h->epoch = b; h->mig_epoch = c;
            const char* p = strstr(line, " all ");
            if (p) { p += 5; for (int g = 0; g < BIO_ATM_N; ++g) { char* e = NULL; h->xall[g] = strtoll(p, &e, 10); p = e; } }
            p = strstr(line, " land ");
            if (p) { p += 6; for (int g = 0; g < BIO_ATM_N; ++g) { char* e = NULL; h->xland[g] = strtoll(p, &e, 10); p = e; } }
        }
    }
    fclose(f);
}

static int hub_save(const char* path, const HubState* S) {
    char tmp[1340];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    fprintf(f, "GMEHUB1\nplanet %016llx\nupdates %lld\nclock_q10 %lld\nscale_ppm %lld\nglobal", (unsigned long long)S->planet,
            (long long)S->updates, (long long)S->clock_q10, (long long)S->scale_ppm);
    for (int g = 0; g < BIO_ATM_N; ++g) fprintf(f, " %lld", (long long)S->g[g]);
    fprintf(f, "\n");
    for (int i = 0; i < S->n; ++i) {
        const HubIsland* h = &S->isl[i];
        if (!h->seen) continue;
        fprintf(f, "island %d epoch %lld mig %lld all", h->k, (long long)h->epoch, (long long)h->mig_epoch);
        for (int g = 0; g < BIO_ATM_N; ++g) fprintf(f, " %lld", (long long)h->xall[g]);
        fprintf(f, " land");
        for (int g = 0; g < BIO_ATM_N; ++g) fprintf(f, " %lld", (long long)h->xland[g]);
        fprintf(f, "\n");
    }
    if (fclose(f)) { remove(tmp); return -1; }
    remove(path);
    return rename(tmp, path);
}

/* the photochemistry of an island box (bio_globals_update), on the global column */
static void hub_photochem(int64_t* a, int64_t steps) {
    BioParams P;
    bio_params_default(&P);
    for (int64_t t = 0; t < steps && a[BIO_ATM_O2] > 0; ++t) {
        int64_t r = bio_min64(bio_min64(a[BIO_ATM_O2] / 2, a[BIO_ATM_CH4]), bio_mulshr((uint64_t)a[BIO_ATM_CH4], (uint64_t)P.photo_ch4_q16, 16));
        if (r > 0) { a[BIO_ATM_O2] -= 2 * r; a[BIO_ATM_CH4] -= r; a[BIO_ATM_CO2] += r; }
        const int64_t r2 = bio_min64(bio_min64(a[BIO_ATM_O2] / 2, a[BIO_ATM_H2S]), bio_mulshr((uint64_t)a[BIO_ATM_H2S], (uint64_t)P.photo_h2s_q16, 16));
        if (r2 > 0) { a[BIO_ATM_O2] -= 2 * r2; a[BIO_ATM_H2S] -= r2; }
        if (r <= 0 && r2 <= 0) break;                 /* nothing left that reacts */
    }
}

static int rep_add(char* rep, size_t n, size_t* used, const char* fmt, ...) {
    if (!rep || *used >= n) return 0;
    va_list ap;
    va_start(ap, fmt);
    const int w = vsnprintf(rep + *used, n - *used, fmt, ap);
    va_end(ap);
    if (w > 0) *used += (size_t)w < n - *used ? (size_t)w : n - *used - 1;
    return w;
}

/* One pass of the hub over hub_dir (in/, out/<k>/, hub_state.txt, atm_global.txt).
 * islands: the islands taking part, in ring order. ocean_cover_ppm: the planet's
 * ocean share. report gets a few lines for the operator. Returns the number of
 * islands whose new figures were taken in, -1 on error. */
int bio_hub_run(const char* hub_dir, const int* islands, int n_islands, int32_t ocean_cover_ppm, char* report, size_t report_n) {
    if (!hub_dir || !islands || n_islands < 1 || n_islands > HUB_MAX) { bio_set_error("hub: bad arguments"); return -1; }
    char in[1100], path[1400], to[1500], statep[1200];
    size_t ru = 0;
    if (report && report_n) report[0] = 0;
    snprintf(in, sizeof in, "%s/in", hub_dir);
    snprintf(statep, sizeof statep, "%s/hub_state.txt", hub_dir);
    if (bio_mkdir_p(in)) { bio_set_error("cannot create %s", in); return -1; }
    HubState S;
    hub_load(statep, &S);
    if (ocean_cover_ppm < 0) ocean_cover_ppm = 0;
    if (ocean_cover_ppm > 1000000) ocean_cover_ppm = 1000000;

    /* newest atmosphere file of each island */
    int64_t box[HUB_MAX][BIO_ATM_N], xall[HUB_MAX][BIO_ATM_N], xland[HUB_MAX][BIO_ATM_N], area_o[HUB_MAX], area_l[HUB_MAX], ep[HUB_MAX];
    int have[HUB_MAX];
    int64_t scale = S.scale_ppm;
    for (int i = 0; i < n_islands; ++i) {
        have[i] = 0;
        char pre[32];
        snprintf(pre, sizeof pre, "atm_%d_", islands[i]);
        NameList L;
        if (names_list(in, pre, ".txt", &L) > 0) {
            snprintf(path, sizeof path, "%s/%s", in, L.name[L.n - 1]);
            int64_t co = 0, cl = 0, cm = 0, sc = 0, e0 = 0, pk = 0;
            if (!kv_read(path, "island", &pk, 1) && pk == islands[i] && !kv_read(path, "epoch", &e0, 1) &&
                !kv_read(path, "cells_ocean", &co, 1) && !kv_read(path, "cells_land", &cl, 1) && !kv_read(path, "cell_m2", &cm, 1) &&
                !kv_read(path, "scale_ppm", &sc, 1) && !kv_read(path, "box", box[i], BIO_ATM_N) &&
                !kv_read(path, "xch_all", xall[i], BIO_ATM_N) && !kv_read(path, "xch_land", xland[i], BIO_ATM_N) && cm > 0 && sc > 0) {
                have[i] = 1; ep[i] = e0; area_o[i] = co * cm; area_l[i] = cl * cm;
                if (!scale) scale = sc;
                if (sc != scale) { have[i] = 0; rep_add(report, report_n, &ru, "hub: island %d runs at box scale %lld ppm, the hub at %lld: left out\n", islands[i], (long long)sc, (long long)scale); }
                /* one hub folder serves one planet: its column must never reach another planet's islands */
                const uint64_t ph = kv_hex(path, "planet");
                if (!S.planet) S.planet = ph;
                else if (ph && ph != S.planet) {
                    names_free(&L);
                    bio_set_error("hub: %s is of planet %016llx, but this hub folder serves planet %016llx: use another folder (--dir) or clear this one",
                                  path, (unsigned long long)ph, (unsigned long long)S.planet);
                    return -1;
                }
            }
        }
        names_free(&L);
    }
    S.scale_ppm = scale;

    /* first pass over a planet: the global column starts as the islands' mean column */
    int started = 0;
    for (int g = 0; g < BIO_ATM_N; ++g) started |= S.g[g] != 0;
    if (!started) {
        int64_t area = 0, sum[BIO_ATM_N] = { 0 };
        for (int i = 0; i < n_islands; ++i)
            if (have[i]) { area += area_o[i] + area_l[i]; for (int g = 0; g < BIO_ATM_N; ++g) sum[g] += box[i][g]; }
        if (area > 0) for (int g = 0; g < BIO_ATM_N; ++g) S.g[g] = bio_mulshr((uint64_t)(sum[g] > 0 ? sum[g] : 0), 1000, 0) / area;
    }

    /* Uptake per m^2 of the islands that moved on since the last pass, each over
     * its own span. The planet is the mean of the islands taking part, so every
     * island-year counts for 1 / n of a planet-year whenever it is reported: the
     * column and the clock do not depend on how the reports fall into passes.
     * An island whose epoch went backwards was founded anew (its old files are
     * gone from in/): it is taken in again from where it now stands. */
    int64_t so[BIO_ATM_N] = { 0 }, sl[BIO_ATM_N] = { 0 }, dts = 0;
    int nup = 0, npart = 0;
    for (int i = 0; i < n_islands; ++i) {
        HubIsland* h = hub_island(&S, islands[i]);
        if (!h) continue;
        if (have[i] && h->seen && ep[i] < h->epoch) {
            rep_add(report, report_n, &ru, "hub: island %d is at epoch %lld after %lld: founded anew, taken in from there\n", islands[i],
                    (long long)ep[i], (long long)h->epoch);
            h->seen = 0; h->mig_epoch = -1;
        }
        if (have[i] && h->seen && ep[i] > h->epoch) {
            int any = 0;
            if (area_o[i] > 0) { for (int g = 0; g < BIO_ATM_N; ++g) so[g] += ((xall[i][g] - xland[i][g]) - (h->xall[g] - h->xland[g])) * 1000 / area_o[i]; any = 1; }
            if (area_l[i] > 0) { for (int g = 0; g < BIO_ATM_N; ++g) sl[g] += (xland[i][g] - h->xland[g]) * 1000 / area_l[i]; any = 1; }
            if (any) { dts += ep[i] - h->epoch; ++nup; }
        }
        if (have[i] && (!h->seen || ep[i] > h->epoch)) {
            h->seen = 1; h->epoch = ep[i];
            memcpy(h->xall, xall[i], sizeof h->xall); memcpy(h->xland, xland[i], sizeof h->xland);
        }
        if (h->seen) ++npart;
    }
    int64_t rate[BIO_ATM_N] = { 0 };       /* umol per m^2 of planet per sim-year, into the atmosphere */
    if (nup && npart) {
        const int64_t dq10 = dts * 1024 / npart;                       /* planet-years this pass, q10 */
        for (int g = 0; g < BIO_ATM_N; ++g) {
            /* uptake by the planet's surface over those planet-years, umol per m^2 */
            const int64_t up = (so[g] / npart * ocean_cover_ppm + sl[g] / npart * (1000000 - ocean_cover_ppm)) / 1000000;
            rate[g] = dq10 > 0 ? -up * 1024 / dq10 : 0;
            S.g[g] -= up;
            if (S.g[g] < 0) S.g[g] = 0;
        }
        hub_photochem(S.g, dq10 * 8760 / 1024);
        S.clock_q10 += dq10;
        S.updates++;
    }

    /* what goes back: the global column to every island (once there is one), migrants around the ring and to one more island */
    char outd[1200];
    int64_t gsum = 0;
    for (int g = 0; g < BIO_ATM_N; ++g) gsum += S.g[g];
    for (int i = 0; i < n_islands; ++i) {
        snprintf(outd, sizeof outd, "%s/out/%d", hub_dir, islands[i]);
        if (bio_mkdir_p(outd)) { bio_set_error("cannot create %s", outd); return -1; }
        if (gsum > 0) {
            snprintf(path, sizeof path, "%s/atm_global_%08lld.txt", outd, (long long)S.updates);
            FILE* f = fopen(path, "wb");
            if (f) {
                fprintf(f, "GMEATMG1\nplanet %016llx\nupdate %lld\nclock_years %lld\nscale_ppm %lld\numol_m2", (unsigned long long)S.planet,
                        (long long)S.updates, (long long)(S.clock_q10 / 1024), (long long)S.scale_ppm);
                for (int g = 0; g < BIO_ATM_N; ++g) fprintf(f, " %lld", (long long)S.g[g]);
                fprintf(f, "\n");
                fclose(f);
            }
        }
    }
    int routed = 0;
    for (int i = 0; i < n_islands && n_islands > 1; ++i) {
        HubIsland* h = hub_island(&S, islands[i]);
        if (!h) continue;
        char pre[32];
        snprintf(pre, sizeof pre, "mig_%d_", islands[i]);
        NameList L;
        names_list(in, pre, ".bin", &L);
        for (int q = 0; q < L.n; ++q) {
            const long long e0 = strtoll(L.name[q] + strlen(pre), NULL, 10);
            if (e0 <= h->mig_epoch) continue;
            snprintf(path, sizeof path, "%s/%s", in, L.name[q]);
            const int next = islands[(i + 1) % n_islands];
            int other = islands[(i + 1 + (int)(bio_hash(islands[i], (int32_t)e0, BH_IMPORT + 9, 0x4B10u) % (uint32_t)(n_islands - 1))) % n_islands];
            snprintf(to, sizeof to, "%s/out/%d/%s", hub_dir, next, L.name[q]);
            if (!file_copy(path, to)) ++routed;
            if (other != next && other != islands[i]) {
                snprintf(to, sizeof to, "%s/out/%d/%s", hub_dir, other, L.name[q]);
                if (!file_copy(path, to)) ++routed;
            }
            h->mig_epoch = e0;
        }
        names_free(&L);
    }
    /* in/ is a mailbox too: the 4 newest files of each island and kind are enough */
    for (int i = 0; i < n_islands; ++i)
        for (int q = 0; q < 2; ++q) {
            char pre[32];
            snprintf(pre, sizeof pre, q ? "mig_%d_" : "atm_%d_", islands[i]);
            NameList L;
            names_list(in, pre, q ? ".bin" : ".txt", &L);
            for (int k = 0; k + 4 < L.n; ++k) { snprintf(path, sizeof path, "%s/%s", in, L.name[k]); remove(path); }
            names_free(&L);
        }
    if (hub_save(statep, &S)) { bio_set_error("cannot write %s", statep); return -1; }

    /* the planet's atmosphere, to read; years until the reductants are used up at the present O2 supply */
    int64_t tot = 0;
    for (int g = 0; g < BIO_ATM_N; ++g) tot += S.g[g];
    snprintf(path, sizeof path, "%s/atm_global.txt", hub_dir);
    FILE* f = fopen(path, "wb");
    static const char* const GN[BIO_ATM_N] = { "H2S", "CH4", "NH3", "CO2", "O2", "N2", "H2" };
    rep_add(report, report_n, &ru, "hub: update %lld, clock %.1f sim-years, %d island(s) reported, %d migrant file(s) routed\n",
            (long long)S.updates, S.clock_q10 / 1024.0, nup, routed);
    rep_add(report, report_n, &ru, "hub: planet atmosphere:");
    if (f) fprintf(f, "GME planet atmosphere (hub update %lld, clock %.1f sim-years, box scale %lld ppm of the real column)\n", (long long)S.updates, S.clock_q10 / 1024.0, (long long)S.scale_ppm);
    for (int g = 0; g < BIO_ATM_N; ++g) {
        const double pc = tot > 0 ? 100.0 * (double)S.g[g] / (double)tot : 0.0;
        rep_add(report, report_n, &ru, " %s %.4f%%", GN[g], pc);
        if (f) fprintf(f, "%-4s %12.6f %%  %14.6g mol/m^2 (scaled)  net source %+.6g mol/m^2/yr\n", GN[g], pc, (double)S.g[g] / 1e6, (double)rate[g] / 1e6);
    }
    rep_add(report, report_n, &ru, "\n");
    {
        /* O2 needed to oxidise what the air holds: CH4 2, H2S 2, NH3 3/4, H2 1/2 */
        const double need = 2.0 * S.g[BIO_ATM_CH4] + 2.0 * S.g[BIO_ATM_H2S] + 0.75 * S.g[BIO_ATM_NH3] + 0.5 * S.g[BIO_ATM_H2] - (double)S.g[BIO_ATM_O2];
        /* net oxidant supply per year: O2 released minus the reductants released with it */
        const double sup = (double)rate[BIO_ATM_O2] - 2.0 * rate[BIO_ATM_CH4] - 2.0 * rate[BIO_ATM_H2S] - 0.75 * rate[BIO_ATM_NH3] - 0.5 * rate[BIO_ATM_H2];
        char eta[160];
        if (need <= 0) snprintf(eta, sizeof eta, "the planet's air is oxidised: O2 is in excess of the reductants");
        else if (!nup) snprintf(eta, sizeof eta, "no new island figures in this pass");
        else if (sup <= 0) snprintf(eta, sizeof eta, "no O2 rise at the present fluxes (net oxidant supply %.3g mol/m^2/yr)", sup / 1e6);
        else snprintf(eta, sizeof eta, "%.0f sim-years to the O2 rise at the present net oxidant supply (%.3g mol/m^2/yr)", need / sup, sup / 1e6);
        rep_add(report, report_n, &ru, "hub: titration: %s\n", eta);
        if (f) fprintf(f, "titration: %s\n", eta);
    }
    if (f) fclose(f);
    return nup;
}

/* ---- folder helpers for the hub's transport (local islands, tests) -------------------- */
/* copies the plain files of from_dir whose names start with prefix into to_dir
 * (created); returns the number copied, -1 on error */
int bio_dir_copy(const char* from_dir, const char* to_dir, const char* prefix) {
    NameList L;
    char a[1400], b[1400];
    if (names_list(from_dir, prefix ? prefix : "", "", &L) <= 0) { names_free(&L); return 0; }
    if (bio_mkdir_p(to_dir)) { names_free(&L); return -1; }
    int n = 0;
    for (int i = 0; i < L.n; ++i) {
        const size_t l = strlen(L.name[i]);
        if (l > 4 && !strcmp(L.name[i] + l - 4, ".tmp")) continue;
        snprintf(a, sizeof a, "%s/%s", from_dir, L.name[i]);
        snprintf(b, sizeof b, "%s/%s", to_dir, L.name[i]);
        if (!file_copy(a, b)) ++n;
    }
    names_free(&L);
    return n;
}
/* Makes the files of to_dir that start with prefix the same set as those of
 * from_dir: copies the ones missing or changed in size, removes the ones that
 * are gone. The hub's in/ then holds, per island, what that island's outbox
 * holds now, so an island founded anew does not hide behind its old files.
 * Does nothing when from_dir cannot be listed or holds none (an island that is
 * away keeps its last files). Returns the number copied, -1 on error. */
int bio_dir_mirror(const char* from_dir, const char* to_dir, const char* prefix) {
    NameList A, B;
    char a[1400], b[1400];
    if (names_list(from_dir, prefix, "", &A) <= 0) { names_free(&A); return 0; }
    if (bio_mkdir_p(to_dir)) { names_free(&A); return -1; }
    names_list(to_dir, prefix, "", &B);
    for (int i = 0; i < B.n; ++i) {
        const size_t l = strlen(B.name[i]);
        if (l > 4 && !strcmp(B.name[i] + l - 4, ".tmp")) continue;
        if (!bsearch(B.name[i], A.name, (size_t)A.n, 160, name_cmp)) { snprintf(b, sizeof b, "%s/%s", to_dir, B.name[i]); remove(b); }
    }
    int n = 0;
    for (int i = 0; i < A.n; ++i) {
        const size_t l = strlen(A.name[i]);
        if (l > 4 && !strcmp(A.name[i] + l - 4, ".tmp")) continue;
        snprintf(a, sizeof a, "%s/%s", from_dir, A.name[i]);
        snprintf(b, sizeof b, "%s/%s", to_dir, A.name[i]);
        if (bsearch(A.name[i], B.name, (size_t)B.n, 160, name_cmp)) {      /* there already: the same size means the same file */
            FILE* fa = fopen(a, "rb"); FILE* fb = fopen(b, "rb");
            long sa = -1, sb = -2;
            if (fa) { fseek(fa, 0, SEEK_END); sa = ftell(fa); fclose(fa); }
            if (fb) { fseek(fb, 0, SEEK_END); sb = ftell(fb); fclose(fb); }
            if (sa == sb) continue;
        }
        if (!file_copy(a, b)) ++n;
    }
    names_free(&A); names_free(&B);
    return n;
}
/* removes the plain files of dir, then dir itself when it is empty */
void bio_dir_clear(const char* dir) {
    NameList L;
    char a[1400];
    names_list(dir, "", "", &L);
    for (int i = 0; i < L.n; ++i) { snprintf(a, sizeof a, "%s/%s", dir, L.name[i]); remove(a); }
    names_free(&L);
#ifdef _WIN32
    _rmdir(dir);
#else
    rmdir(dir);
#endif
}
