// cosmos_kernels.cu - compiled at run time by NVRTC (see rtcuda.c).
// Self-contained: no includes. Units: AU, yr, solar masses.
//
// pos[i] = (x, y, z, mass)   vel[i] = (vx, vy, vz, collision radius)

#define TILE 256

extern "C" __global__ void k_kick_drift(int n, float4* pos, float4* vel, const float4* acc, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = pos[i], v = vel[i];
    const float4 a = acc[i];
    v.x += a.x * 0.5f * dt; v.y += a.y * 0.5f * dt; v.z += a.z * 0.5f * dt;
    p.x += v.x * dt;        p.y += v.y * dt;        p.z += v.z * dt;
    pos[i] = p;
    vel[i] = v;
}

extern "C" __global__ void k_kick(int n, float4* vel, const float4* acc, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = vel[i];
    const float4 a = acc[i];
    v.x += a.x * 0.5f * dt; v.y += a.y * 0.5f * dt; v.z += a.z * 0.5f * dt;
    vel[i] = v;
}

// Direct-sum gravity (tiled through shared memory), star, gas drag, and
// collision detection (pairs i < j whose collision spheres overlap).
extern "C" __global__ void __launch_bounds__(TILE)
k_accel(int n, const float4* __restrict__ pos, const float4* __restrict__ vel, float4* __restrict__ acc,
        float G, float star_gm, float eps2, float gas_on, float tau0, float m0,
        int2* pairs, int* pair_count, int pair_cap) {
    __shared__ float4 sp[TILE];
    __shared__ float  sr[TILE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float4 pi = i < n ? pos[i] : make_float4(1e9f, 0.f, 0.f, 0.f);
    const float  ri = i < n ? vel[i].w : 0.f;
    float ax = 0.f, ay = 0.f, az = 0.f;
    int hit = 0x7FFFFFFF;

    for (int base = 0; base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) { sp[threadIdx.x] = pos[j]; sr[threadIdx.x] = vel[j].w; }
        else       { sp[threadIdx.x] = make_float4(2e9f, 0.f, 0.f, 0.f); sr[threadIdx.x] = 0.f; }
        __syncthreads();
        const int lim = min(TILE, n - base);
        #pragma unroll 8
        for (int k = 0; k < lim; ++k) {
            const float4 q = sp[k];
            const float dx = q.x - pi.x, dy = q.y - pi.y, dz = q.z - pi.z;
            const float d2 = dx * dx + dy * dy + dz * dz;
            const float inv = rsqrtf(d2 + eps2);
            const float f = q.w * inv * inv * inv;
            ax += dx * f; ay += dy * f; az += dz * f;
            const float rr = ri + sr[k];
            if (d2 < rr * rr && base + k != i && q.w > 0.f && base + k < hit) hit = base + k;
        }
        __syncthreads();
    }
    if (i >= n) return;
    ax *= G; ay *= G; az *= G;

    const float r2 = pi.x * pi.x + pi.y * pi.y + pi.z * pi.z;
    const float ir = rsqrtf(r2);
    const float s = -star_gm * ir * ir * ir;
    ax += pi.x * s; ay += pi.y * s; az += pi.z * s;

    if (gas_on > 0.f && pi.w > 0.f) {
        // gas orbits on circular Keplerian paths in the midplane; drag pulls
        // bodies toward it (damps eccentricity and inclination). Smaller bodies
        // couple more strongly.
        const float vk = sqrtf(star_gm * ir);
        const float4 v = vel[i];
        const float gx = -pi.y * ir * vk, gy = pi.x * ir * vk;
        const float tau = tau0 * cbrtf(pi.w / m0);
        const float k = gas_on / tau;
        ax -= (v.x - gx) * k; ay -= (v.y - gy) * k; az -= v.z * k;
    }
    acc[i] = make_float4(ax, ay, az, 0.f);

    if (hit != 0x7FFFFFFF && i < hit && pi.w > 0.f) {
        const int slot = atomicAdd(pair_count, 1);
        if (slot < pair_cap) pairs[slot] = make_int2(i, hit);
    }
}
