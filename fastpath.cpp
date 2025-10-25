// fastpath.cpp (v0.5b) — HALO: SAXPY/Reduce + 2D-Image-Kern (LUT + AXPBY)
#ifndef HALO_FASTPATH_INCLUDED
#define HALO_FASTPATH_INCLUDED
//  - Persistenter Thread-Pool (kein Thread-Spawn pro Call)
//  - Auto-Scheduler: ST/MT & NT-Stores basierend auf Bytes/Breite/Alignment
//  - Affine-LUT Fast-Path (kein Gather bei LUT[i]≈a*i+b)
//  - AVX2-Pfade mit Alignment-Prolog, sfence genau 1x pro Thread/Call
//  - NEU v0.5b: halo_shutdown_pool() für sauberes Beenden des Pools
//
// Build (MinGW): g++ -O3 -march=native -pthread -shared -o halo_fastpath.dll fastpath.cpp
// Build (MSVC) : cl /O2 /arch:AVX2 /EHsc /LD fastpath.cpp /Fe:halo_fastpath.dll

#include <cstdint>
#include <cstring>
#include <chrono>
#include <limits>
#include <new>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  #define HALO_X86 1
  #include <immintrin.h>
#else
  #define HALO_X86 0
#endif

#if HALO_X86
#ifndef _mm256_set_m128i
  #define _mm256_set_m128i(hi, lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256((lo)), (hi), 1)
#endif
#ifndef _mm256_set_m128
  #define _mm256_set_m128(hi, lo) \
    _mm256_insertf128_ps(_mm256_castps128_ps256((lo)), (hi), 1)
#endif
#endif

#if !defined(_MSC_VER) && HALO_X86
  #include <cpuid.h>
#endif

#if defined(_WIN32)
  #define HALO_API extern "C" __declspec(dllexport)
#else
  #define HALO_API extern "C"
#endif

// ------------------ Globaler Zustand ------------------
static bool g_has_sse2   = false;
static bool g_has_avx2   = false;
static bool g_has_avx512 = false;

static bool    g_streaming_enabled = true;
static int64_t g_stream_threshold  = 1'000'000;  // SAXPY-NT-Schwelle (Elemente)
static int     g_threads           = 1;          // gewünschte Worker-Anzahl

static constexpr int      kNtMinWidthDefault   = 1024;
static constexpr int64_t  kNtMinBytesDefault   = 8ll << 20;   // 8 MiB
static constexpr int      kNtHighWidth         = 4096;
static constexpr int64_t  kNtHighBytes         = 32ll << 20;  // 32 MiB
static constexpr int64_t  kTargetTaskBytes     = 6ll << 20;   // ~6 MiB pro Task (LLC-optimiert)
static constexpr int64_t  kSmallTaskBytes      = 768ll << 10; // ~0.75 MiB, für feine Granularität

static inline double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
static inline bool is_aligned_32(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 31ull) == 0ull;
}

// ------------------ CPU-Feature-Init ------------------
#if HALO_X86
static void halo_cpuid_init() {
  int ecx=0, edx=0;
#if defined(_MSC_VER)
  int regs[4] = {0,0,0,0};
  __cpuid(regs, 1);
  ecx = regs[2]; edx = regs[3];
#else
  unsigned int a,b,c,d;
  __get_cpuid(1, &a, &b, &c, &d);
  ecx = (int)c; edx = (int)d;
#endif
  g_has_sse2 = (edx & (1<<26)) != 0;

  int ebx7=0, ecx7=0, edx7=0;
#if defined(_MSC_VER)
  int regs7[4] = {0,0,0,0};
  __cpuidex(regs7, 7, 0);
  ebx7 = regs7[1]; ecx7 = regs7[2]; edx7 = regs7[3];
#else
  unsigned int a7,b7,c7,d7;
  __get_cpuid_count(7, 0, &a7, &b7, &c7, &d7);
  ebx7 = (int)b7; ecx7 = (int)c7; edx7 = (int)d7;
#endif
  const bool avx2_bit = (ebx7 & (1<<5)) != 0;

  bool os_avx_supported = false;
  if (ecx & (1<<27)) {
    unsigned long long xcr0 = 0;
#if defined(_MSC_VER)
    xcr0 = _xgetbv(0);
#else
    unsigned int eax_x, edx_x;
    __asm__ volatile (".byte 0x0f, 0x01, 0xd0" : "=a"(eax_x), "=d"(edx_x) : "c"(0));
    xcr0 = ((unsigned long long)edx_x << 32) | eax_x;
#endif
    const bool xmm = (xcr0 & 0x2) != 0;
    const bool ymm = (xcr0 & 0x4) != 0;
    os_avx_supported = (xmm && ymm);
  }
  g_has_avx2   = (avx2_bit && os_avx_supported);
  g_has_avx512 = false;
}
#else
static void halo_cpuid_init() {
  g_has_sse2 = g_has_avx2 = g_has_avx512 = false;
}
#endif

// =====================================================
//  Persistenter Thread-Pool
// =====================================================
class ThreadPool {
public:
  ThreadPool(): desired_(1), stop_(false) {}
  ~ThreadPool() { shutdown(); }

  void set_threads(int t) {
    if (t < 1) t = 1;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw > 0 && (unsigned)t > hw) t = (int)hw;
    std::unique_lock<std::mutex> lk(mx_);
    desired_ = t;
    resize_unlocked();
  }

  int threads() const {
    std::lock_guard<std::mutex> lk(mx_);
    return (int)workers_.size();
  }

  // NEU v0.5b: explizit von außen beendbar (C-API)
  void stop_all() { shutdown(); }

  void parallel_for_rows(
    int rows,
    const std::function<void(int,int)>& fn,
    int min_rows_per_task=1,
    int chunk_rows_hint=0
  ) {
    if (rows <= 0) return;

    // Falls nur 1 Thread oder wenig Arbeit: direkt ausführen
    int t;
    {
      std::lock_guard<std::mutex> lk(mx_);
      t = std::max(1, (int)workers_.size());
    }
    int max_tasks = t;
    if (rows < max_tasks * min_rows_per_task) {
      fn(0, rows);
      return;
    }

    if (chunk_rows_hint > 0) {
      if (chunk_rows_hint < min_rows_per_task) chunk_rows_hint = min_rows_per_task;
    }

    int chunk = chunk_rows_hint > 0
      ? chunk_rows_hint
      : std::max(min_rows_per_task, (rows + max_tasks - 1) / max_tasks);
    if (chunk <= 0) chunk = min_rows_per_task;

    int tasks = (rows + chunk - 1) / chunk;
    if (tasks <= 1) {
      fn(0, rows);
      return;
    }

    // Latch für Fertigstellung
    struct Latch {
      std::mutex m;
      std::condition_variable cv;
      int count;
    };
    auto latch = std::make_shared<Latch>();
    latch->count = tasks;

    int y0 = 0;
    for (int i=0; i<tasks; ++i) {
      int y1 = y0 + chunk;
      if (y1 > rows) y1 = rows;
      int job_y0 = y0;
      int job_y1 = y1;
      y0 = y1;
      enqueue([=]() {
        fn(job_y0, job_y1);
        std::unique_lock<std::mutex> lk2(latch->m);
        if (--latch->count == 0) latch->cv.notify_one();
      });
    }

    // Warten
    std::unique_lock<std::mutex> lk(latch->m);
    latch->cv.wait(lk, [&]{ return latch->count == 0; });
  }

  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }

private:
  void enqueue(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> lk(mx_);
      resize_unlocked();
      jobs_.emplace_back(std::move(job));
    }
    cv_.notify_one();
  }

  void worker_loop() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(mx_);
        cv_.wait(lk, [&]{ return stop_ || !jobs_.empty() || (int)workers_.size() > desired_; });

        if (stop_) return;

        // Shrink?
        if ((int)workers_.size() > desired_) {
          return; // Thread beendet sich
        }

        if (!jobs_.empty()) {
          job = std::move(jobs_.front());
          jobs_.pop_front();
        } else {
          continue;
        }
      }
      job();
    }
  }

  void resize_unlocked() {
    // grow
    while ((int)workers_.size() < desired_) {
      workers_.emplace_back([this]{ worker_loop(); });
    }
    // shrink: passiv — Threads verlassen loop sobald desired_ unterschritten wurde, Join in shutdown()
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mx_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& th : workers_) { if (th.joinable()) th.join(); }
    workers_.clear();
  }

private:
  mutable std::mutex mx_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> jobs_;
  std::vector<std::thread> workers_;
  int desired_;
  bool stop_;
};

// Helper: Pool-Größe anpassen, idempotent
static void ensure_pool(int t) {
  ThreadPool::instance().set_threads(t);
}

// =====================================================
//  SAXPY / SUM
// =====================================================
static void saxpy_scalar(float a, const float* x, float* y, int64_t n) {
  for (int64_t i=0;i<n;++i) y[i] = a * x[i] + y[i];
}

#if HALO_X86
static void saxpy_sse2(float a, const float* x, float* y, int64_t n) {
  const __m128 va = _mm_set1_ps(a);
  int64_t i=0;
  for (; i+4<=n; i+=4) {
    __m128 vx = _mm_loadu_ps(x+i);
    __m128 vy = _mm_loadu_ps(y+i);
    __m128 r  = _mm_add_ps(_mm_mul_ps(va, vx), vy);
    _mm_storeu_ps(y+i, r);
  }
  for (; i<n; ++i) y[i] = a * x[i] + y[i];
}

static void saxpy_avx2(float a, const float* x, float* y, int64_t n) {
  const __m256 va = _mm256_set1_ps(a);
  int64_t i=0;
  for (; i+8<=n; i+=8) {
    __m256 vx = _mm256_loadu_ps(x+i);
    __m256 vy = _mm256_loadu_ps(y+i);
    __m256 r  = _mm256_fmadd_ps(va, vx, vy);
    _mm256_storeu_ps(y+i, r);
  }
  for (; i<n; ++i) y[i] = a * x[i] + y[i];
}

static void saxpy_avx2_stream(float a, const float* x, float* y, int64_t n) {
  if (n < g_stream_threshold || !is_aligned_32(y)) {
    saxpy_avx2(a, x, y, n);
    return;
  }
  const __m256 va = _mm256_set1_ps(a);
  int64_t i=0;
  for (; i<n && ((reinterpret_cast<uintptr_t>(y+i) & 31ull) != 0ull); ++i) {
    y[i] = a * x[i] + y[i];
  }
  for (; i+8<=n; i+=8) {
    __m256 vx = _mm256_loadu_ps(x+i);
    __m256 vy = _mm256_loadu_ps(y+i);
    __m256 r  = _mm256_fmadd_ps(va, vx, vy);
    _mm256_stream_ps(y+i, r);
  }
  for (; i<n; ++i) y[i] = a * x[i] + y[i];
  _mm_sfence();
}
#endif // HALO_X86

static float sum_scalar(const float* x, int64_t n) {
  float acc=0.f;
  for (int64_t i=0;i<n;++i) acc += x[i];
  return acc;
}

#if HALO_X86
static float sum_avx2(const float* x, int64_t n) {
  __m256 vacc = _mm256_setzero_ps();
  int64_t i=0;
  for (; i+8<=n; i+=8) {
    __m256 vx = _mm256_loadu_ps(x+i);
    vacc = _mm256_add_ps(vacc, vx);
  }
  __m128 lo = _mm256_castps256_ps128(vacc);
  __m128 hi = _mm256_extractf128_ps(vacc, 1);
  __m128 s  = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(s);
  __m128 ss = _mm_add_ps(s, sh);
  sh = _mm_movehl_ps(sh, ss);
  ss = _mm_add_ss(ss, sh);
  float acc = _mm_cvtss_f32(ss);
  for (; i<n; ++i) acc += x[i];
  return acc;
}
#endif

// ------------------ Dispatch/Autotune ------------------
enum class Impl : int32_t { Scalar=0, SSE2=1, AVX2=2, AVX2_STREAM=3 };
static Impl g_best_saxpy = Impl::Scalar;
static Impl g_best_sum   = Impl::Scalar;

static void saxpy_dispatch(float a, const float* x, float* y, int64_t n) {
  switch (g_best_saxpy) {
    case Impl::Scalar: saxpy_scalar(a,x,y,n); break;
#if HALO_X86
    case Impl::SSE2:   saxpy_sse2(a,x,y,n);   break;
    case Impl::AVX2:   saxpy_avx2(a,x,y,n);   break;
    case Impl::AVX2_STREAM:
      if (g_streaming_enabled) saxpy_avx2_stream(a,x,y,n);
      else                     saxpy_avx2(a,x,y,n);
      break;
#endif
    default:           saxpy_scalar(a,x,y,n); break;
  }
}

static double bench_once_saxpy(Impl impl, float a, const float* x, float* y, int64_t n, int iters) {
  double t0 = now_sec();
  for (int i=0;i<iters;++i) {
    switch (impl) {
      case Impl::Scalar: saxpy_scalar(a,x,y,n); break;
#if HALO_X86
      case Impl::SSE2:   saxpy_sse2(a,x,y,n);   break;
      case Impl::AVX2:   saxpy_avx2(a,x,y,n);   break;
      case Impl::AVX2_STREAM: saxpy_avx2_stream(a,x,y,n); break;
#endif
      default: saxpy_scalar(a,x,y,n); break;
    }
  }
  return now_sec() - t0;
}

static double bench_once_sum(Impl impl, const float* x, int64_t n, int iters, volatile float* sink) {
  double t0 = now_sec();
  float acc=0.f;
  for (int i=0;i<iters;++i) {
    switch (impl) {
      case Impl::Scalar: acc = sum_scalar(x,n); break;
#if HALO_X86
      case Impl::SSE2:   acc = sum_scalar(x,n); break;
      case Impl::AVX2:   acc = sum_avx2(x,n);   break;
      case Impl::AVX2_STREAM: acc = sum_avx2(x,n); break;
#endif
      default: acc = sum_scalar(x,n); break;
    }
  }
  *sink = acc;
  return now_sec() - t0;
}

// ------------------ MT-Wrapper per Pool für SAXPY ------------------
static void saxpy_slice(float a, const float* x, float* y, int64_t begin, int64_t end) {
  saxpy_dispatch(a, x + begin, y + begin, end - begin);
}

static void saxpy_mt(float a, const float* x, float* y, int64_t n, int threads) {
  if (threads <= 1 || n < 100000) {
    saxpy_dispatch(a, x, y, n);
    return;
  }
  ensure_pool(threads);
  ThreadPool::instance().parallel_for_rows((int)threads, [&](int t0, int t1){
    int totalT = threads;
    int64_t chunk = n / totalT;
    int idx = t0; // 0..threads-1
    int64_t begin = idx * chunk;
    int64_t end   = (idx == totalT - 1) ? n : (begin + chunk);
    saxpy_slice(a, x, y, begin, end);
  }, /*min_rows_per_task=*/1, /*chunk_rows_hint=*/1);
}

// ------------------ LUT-Analyse ------------------
static inline bool lut_is_affine(const float* lut256, float& a, float& b) {
  b = lut256[0];
  a = lut256[1] - lut256[0];
  const float eps = 1e-6f;
  for (int i=2; i<256; ++i) {
    float want = a * i + b;
    if (std::fabs(lut256[i] - want) > eps) return false;
  }
  return true;
}

static inline int compute_min_rows_per_task(int width, int height, int64_t row_bytes) {
  if (height <= 1) return 1;
  int min_rows = 1;

  if (width < 512) min_rows = std::max(min_rows, 2);
  if (width < 256) min_rows = std::max(min_rows, 3);
  if (width < 192) min_rows = std::max(min_rows, 4);
  if (width < 128) min_rows = std::max(min_rows, 6);
  if (width < 64)  min_rows = std::max(min_rows, 8);

  if (row_bytes <= 512 * 1024) min_rows = std::max(min_rows, 2);
  if (row_bytes <= 256 * 1024) min_rows = std::max(min_rows, 3);
  if (row_bytes <= 128 * 1024) min_rows = std::max(min_rows, 4);
  if (row_bytes <= 64  * 1024) min_rows = std::max(min_rows, 6);
  if (row_bytes <= 32  * 1024) min_rows = std::max(min_rows, 8);
  if (row_bytes <= 16  * 1024) min_rows = std::max(min_rows, 12);
  if (row_bytes <= 8   * 1024) min_rows = std::max(min_rows, 16);

  if (min_rows > height) min_rows = height;
  if (min_rows < 1) min_rows = 1;
  return min_rows;
}

static inline int compute_chunk_rows_hint(int width, int height, int min_rows, int threads, int64_t row_bytes, int64_t total_bytes) {
  if (height <= min_rows) return height;
  int64_t target = (threads > 1) ? kTargetTaskBytes : kSmallTaskBytes;
  if (total_bytes < (int64_t)threads * kSmallTaskBytes) target = std::max<int64_t>(target / 2, kSmallTaskBytes);
  if (width >= 2048) target += 1ll << 20;
  if (width <= 256) target = std::max<int64_t>(target / 2, 384ll << 10);
  int64_t denom = std::max<int64_t>(row_bytes, 1);
  int chunk = (int)((target + denom - 1) / denom);
  if (chunk < min_rows) chunk = min_rows;
  if (chunk > height) chunk = height;
  return chunk;
}

static inline bool should_use_streaming(int width, int64_t total_bytes, bool dst_aligned, bool stride_aligned) {
  if (!g_streaming_enabled || !dst_aligned || !stride_aligned) return false;
  if (width >= kNtHighWidth && total_bytes >= kNtHighBytes) return true;
  return (width >= kNtMinWidthDefault) && (total_bytes >= kNtMinBytesDefault);
}

struct ScheduleDecision {
  int  min_rows;
  int  chunk_hint;
  bool do_mt;
};

static inline ScheduleDecision compute_schedule(int width, int height, int threads, int64_t row_work, int64_t total_bytes, bool use_mt) {
  ScheduleDecision d;
  d.min_rows = compute_min_rows_per_task(width, height, row_work);
  d.chunk_hint = compute_chunk_rows_hint(width, height, d.min_rows, threads, row_work, total_bytes);
  const bool enough_rows = height > d.min_rows;
  const bool enough_bytes = total_bytes >= (int64_t)threads * kSmallTaskBytes;
  d.do_mt = use_mt && threads > 1 && enough_rows && enough_bytes;
  return d;
}

// =====================================================
//  2D-Image-Kern (u8 → f32 via LUT + scale/offset, dann AXPBY)
// =====================================================

// Scalar-Referenz
static void img_u8_to_f32_lut_axpby_scalar(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  const float* lut256,
  float scale, float offset,
  float alpha, float beta
) {
  for (int y=0; y<height; ++y) {
    const uint8_t* srow = src + y * src_stride;
    float*         drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      float tmp = lut256[(int)srow[x]] * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }
  }
}

static void img_u16_to_f32_axpby_scalar(
  const uint16_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  uint16_t mask,
  float inv_max,
  float scale, float offset,
  float alpha, float beta
) {
  for (int y=0; y<height; ++y) {
    const uint16_t* srow = (const uint16_t*)((const uint8_t*)src + y * src_stride);
    float*          drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      float sample = float(srow[x] & mask) * inv_max;
      float tmp = sample * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }
  }
}

static void img_rgb_u8_to_f32_interleaved_scalar(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  float scale, float offset,
  float alpha, float beta
) {
  for (int y=0; y<height; ++y) {
    const uint8_t* srow = src + y * src_stride;
    float*         drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      const int si = x * 3;
      const int di = x * 3;
      for (int c=0; c<3; ++c) {
        float tmp = float(srow[si + c]) * scale + offset;
        drow[di + c] = alpha * drow[di + c] + beta * tmp;
      }
    }
  }
}

static void img_rgb_u8_to_f32_planar_scalar(
  const uint8_t* src_r, int64_t src_r_stride,
  const uint8_t* src_g, int64_t src_g_stride,
  const uint8_t* src_b, int64_t src_b_stride,
  float* dst_r, int64_t dst_r_stride,
  float* dst_g, int64_t dst_g_stride,
  float* dst_b, int64_t dst_b_stride,
  int width, int height,
  float scale, float offset,
  float alpha, float beta
) {
  for (int y=0; y<height; ++y) {
    const uint8_t* srow_r = src_r + y * src_r_stride;
    const uint8_t* srow_g = src_g + y * src_g_stride;
    const uint8_t* srow_b = src_b + y * src_b_stride;
    float* drow_r = (float*)((uint8_t*)dst_r + y * dst_r_stride);
    float* drow_g = (float*)((uint8_t*)dst_g + y * dst_g_stride);
    float* drow_b = (float*)((uint8_t*)dst_b + y * dst_b_stride);
    for (int x=0; x<width; ++x) {
      float tmp_r = float(srow_r[x]) * scale + offset;
      float tmp_g = float(srow_g[x]) * scale + offset;
      float tmp_b = float(srow_b[x]) * scale + offset;
      drow_r[x] = alpha * drow_r[x] + beta * tmp_r;
      drow_g[x] = alpha * drow_g[x] + beta * tmp_g;
      drow_b[x] = alpha * drow_b[x] + beta * tmp_b;
    }
  }
}

#if HALO_X86
// Gather-Pfad (beliebige LUT)
static bool img_u8_to_f32_lut_axpby_avx2_gather(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  const float* lut256,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream,
  bool allow_aligned
) {
  const __m256 vscale = _mm256_set1_ps(scale);
  const __m256 voffs  = _mm256_set1_ps(offset);
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta  = _mm256_set1_ps(beta);

  bool any_stream = false;

  for (int y=0; y<height; ++y) {
    const uint8_t* srow = src + y * src_stride;
    float*         drow = (float*)((uint8_t*)dst + y * dst_stride);

    int x = 0;
    for (; x < width && ((reinterpret_cast<uintptr_t>(drow + x) & 31ull) != 0ull); ++x) {
      float tmp = lut256[(int)srow[x]] * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }

    for (; x + 8 <= width; x += 8) {
      _mm_prefetch((const char*)(srow + x + 256), _MM_HINT_T0);

      __m128i bytes64 = _mm_loadl_epi64((const __m128i*)(srow + x));
      __m128i u16x8   = _mm_cvtepu8_epi16(bytes64);
      __m128i idx_lo  = _mm_cvtepu16_epi32(u16x8);
      __m128i idx_hi  = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(u16x8, _mm_setzero_si128()));
      __m256i idx8    = _mm256_set_m128i(idx_hi, idx_lo);

      __m256 vL   = _mm256_i32gather_ps(lut256, idx8, 4);
      __m256 vtmp = _mm256_fmadd_ps(vL, vscale, voffs);
      __m256 vdst = allow_aligned ? _mm256_load_ps(drow + x)
                                  : _mm256_loadu_ps(drow + x);
      __m256 vout = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));

      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
      } else if (allow_aligned) {
        _mm256_store_ps(drow + x, vout);
      } else {
        _mm256_storeu_ps(drow + x, vout);
      }
    }

    for (; x < width; ++x) {
      float tmp = lut256[(int)srow[x]] * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }
  }

  return any_stream;
}

// Affiner LUT-Fast-Path (kein Gather)
static bool img_u8_to_f32_lut_axpby_avx2_affine(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  float slope, float intercept,
  float alpha, float beta,
  bool allow_stream,
  bool allow_aligned
) {
  const __m256 vslope = _mm256_set1_ps(slope);
  const __m256 vinter = _mm256_set1_ps(intercept);
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta  = _mm256_set1_ps(beta);

  bool any_stream = false;

  for (int y=0; y<height; ++y) {
    const uint8_t* srow = src + y * src_stride;
    float*         drow = (float*)((uint8_t*)dst + y * dst_stride);

    int x = 0;
    for (; x < width && ((reinterpret_cast<uintptr_t>(drow + x) & 31ull) != 0ull); ++x) {
      float tmp = slope * (float)srow[x] + intercept;
      drow[x] = alpha * drow[x] + beta * tmp;
    }

    for (; x + 8 <= width; x += 8) {
      _mm_prefetch((const char*)(srow + x + 256), _MM_HINT_T0);

      __m128i bytes64 = _mm_loadl_epi64((const __m128i*)(srow + x));
      __m128i u16x8   = _mm_cvtepu8_epi16(bytes64);
      __m128i i32_lo  = _mm_cvtepu16_epi32(u16x8);
      __m128i i32_hi  = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(u16x8, _mm_setzero_si128()));
      __m256i i32x8   = _mm256_set_m128i(i32_hi, i32_lo);
      __m256  vf8     = _mm256_cvtepi32_ps(i32x8);

      __m256 vtmp = _mm256_fmadd_ps(vslope, vf8, vinter);
      __m256 vdst = allow_aligned ? _mm256_load_ps(drow + x)
                                  : _mm256_loadu_ps(drow + x);
      __m256 vout = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));

      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
      } else if (allow_aligned) {
        _mm256_store_ps(drow + x, vout);
      } else {
        _mm256_storeu_ps(drow + x, vout);
      }
    }

    for (; x < width; ++x) {
      float tmp = slope * (float)srow[x] + intercept;
      drow[x] = alpha * drow[x] + beta * tmp;
    }
  }
  return any_stream;
}

static bool img_u16_to_f32_axpby_avx2(
  const uint16_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  uint16_t mask,
  float inv_max,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream,
  bool allow_aligned
) {
  const __m256 vinv   = _mm256_set1_ps(inv_max);
  const __m256 vscale = _mm256_set1_ps(scale);
  const __m256 voffs  = _mm256_set1_ps(offset);
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta  = _mm256_set1_ps(beta);
  const __m256i vmask = _mm256_set1_epi32((int)mask);

  bool any_stream = false;

  for (int y=0; y<height; ++y) {
    const uint16_t* srow = (const uint16_t*)((const uint8_t*)src + y * src_stride);
    float*          drow = (float*)((uint8_t*)dst + y * dst_stride);

    int x = 0;
    for (; x < width && ((reinterpret_cast<uintptr_t>(drow + x) & 31ull) != 0ull); ++x) {
      float sample = float(srow[x] & mask) * inv_max;
      float tmp = sample * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }

    for (; x + 8 <= width; x += 8) {
      __m128i vsrc16 = _mm_loadu_si128((const __m128i*)(srow + x));
      __m256i vi32   = _mm256_cvtepu16_epi32(vsrc16);
      vi32 = _mm256_and_si256(vi32, vmask);
      __m256 vfloat = _mm256_mul_ps(_mm256_cvtepi32_ps(vi32), vinv);
      __m256 vtmp   = _mm256_fmadd_ps(vfloat, vscale, voffs);
      __m256 vdst   = allow_aligned ? _mm256_load_ps(drow + x)
                                    : _mm256_loadu_ps(drow + x);
      __m256 vout   = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));
      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
      } else if (allow_aligned) {
        _mm256_store_ps(drow + x, vout);
      } else {
        _mm256_storeu_ps(drow + x, vout);
      }
    }

    for (; x < width; ++x) {
      float sample = float(srow[x] & mask) * inv_max;
      float tmp = sample * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }
  }

  return any_stream;
}
#endif // HALO_X86

// Worker: verarbeitet [y0,y1)
static void img_kernel_rows(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int y0, int y1,
  const float* lut256,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream,
  bool allow_aligned
) {
#if HALO_X86
  if (g_has_avx2) {
    float a=0.f, b=0.f;
    bool is_aff = lut_is_affine(lut256, a, b);
    bool used = false;
    if (is_aff) {
      float slope = a*scale;
      float inter = b*scale + offset;
      used = img_u8_to_f32_lut_axpby_avx2_affine(
        src + (int64_t)y0 * src_stride, src_stride,
        (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride), dst_stride,
        width, (y1 - y0),
        slope, inter, alpha, beta,
        allow_stream,
        allow_aligned
      );
    } else {
      used = img_u8_to_f32_lut_axpby_avx2_gather(
        src + (int64_t)y0 * src_stride, src_stride,
        (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride), dst_stride,
        width, (y1 - y0),
        lut256, scale, offset, alpha, beta,
        allow_stream,
        allow_aligned
      );
    }
    if (used) _mm_sfence();
    return;
  }
#endif
  img_u8_to_f32_lut_axpby_scalar(
    src + (int64_t)y0 * src_stride, src_stride,
    (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride), dst_stride,
    width, (y1 - y0),
    lut256, scale, offset, alpha, beta
  );
}

static void img_u16_kernel_rows(
  const uint16_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int y0, int y1,
  uint16_t mask, float inv_max,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream,
  bool allow_aligned
) {
#if HALO_X86
  if (g_has_avx2) {
    const uint16_t* src_base = (const uint16_t*)((const uint8_t*)src + (int64_t)y0 * src_stride);
    float* dst_base = (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride);
    bool used = img_u16_to_f32_axpby_avx2(
      src_base, src_stride,
      dst_base, dst_stride,
      width, (y1 - y0),
      mask, inv_max, scale, offset, alpha, beta,
      allow_stream, allow_aligned
    );
    if (used) _mm_sfence();
    return;
  }
#endif
  const uint16_t* src_base = (const uint16_t*)((const uint8_t*)src + (int64_t)y0 * src_stride);
  float* dst_base = (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride);
  img_u16_to_f32_axpby_scalar(
    src_base, src_stride,
    dst_base, dst_stride,
    width, (y1 - y0),
    mask, inv_max, scale, offset, alpha, beta
  );
}

static void img_rgb_interleaved_kernel_rows(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int y0, int y1,
  float scale, float offset,
  float alpha, float beta
) {
#if HALO_X86
  if (g_has_avx2 && width >= 4) {
    const __m128 vscale = _mm_set1_ps(scale);
    const __m128 voffset = _mm_set1_ps(offset);
    const __m128 valpha = _mm_set1_ps(alpha);
    const __m128 vbeta  = _mm_set1_ps(beta);
    const __m128i idx_r = _mm_setr_epi32(0, 3, 6, 9);
    const __m128i idx_g = _mm_setr_epi32(1, 4, 7, 10);
    const __m128i idx_b = _mm_setr_epi32(2, 5, 8, 11);
    const __m128i mask_r = _mm_setr_epi8(
      0, 3, 6, 9,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80
    );
    const __m128i mask_g = _mm_setr_epi8(
      1, 4, 7, 10,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80
    );
    const __m128i mask_b = _mm_setr_epi8(
      2, 5, 8, 11,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80,
      (char)0x80, (char)0x80, (char)0x80, (char)0x80
    );

    alignas(16) float out_r[4];
    alignas(16) float out_g[4];
    alignas(16) float out_b[4];

    for (int y = y0; y < y1; ++y) {
      const uint8_t* srow = src + (int64_t)y * src_stride;
      float* drow = (float*)(((uint8_t*)dst) + (int64_t)y * dst_stride);
      int x = 0;
      for (; x + 8 <= width; x += 4) {
        const uint8_t* sp = srow + (int64_t)x * 3;
        __m128i block = _mm_loadu_si128((const __m128i*)sp);
        __m128i rbytes = _mm_shuffle_epi8(block, mask_r);
        __m128i gbytes = _mm_shuffle_epi8(block, mask_g);
        __m128i bbytes = _mm_shuffle_epi8(block, mask_b);

        __m128 vr = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(rbytes));
        __m128 vg = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(gbytes));
        __m128 vb = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(bbytes));

        __m128 tmp_r = _mm_fmadd_ps(vscale, vr, voffset);
        __m128 tmp_g = _mm_fmadd_ps(vscale, vg, voffset);
        __m128 tmp_b = _mm_fmadd_ps(vscale, vb, voffset);

        float* dptr = drow + (int64_t)x * 3;
        __m128 dst_r = _mm_i32gather_ps(dptr, idx_r, 4);
        __m128 dst_g = _mm_i32gather_ps(dptr, idx_g, 4);
        __m128 dst_b = _mm_i32gather_ps(dptr, idx_b, 4);

        __m128 outR = _mm_fmadd_ps(valpha, dst_r, _mm_mul_ps(vbeta, tmp_r));
        __m128 outG = _mm_fmadd_ps(valpha, dst_g, _mm_mul_ps(vbeta, tmp_g));
        __m128 outB = _mm_fmadd_ps(valpha, dst_b, _mm_mul_ps(vbeta, tmp_b));

        _mm_store_ps(out_r, outR);
        _mm_store_ps(out_g, outG);
        _mm_store_ps(out_b, outB);
        for (int lane = 0; lane < 4; ++lane) {
          float* pixel = dptr + lane * 3;
          pixel[0] = out_r[lane];
          pixel[1] = out_g[lane];
          pixel[2] = out_b[lane];
        }
      }
      for (; x < width; ++x) {
        const int si = x * 3;
        const int di = x * 3;
        for (int c = 0; c < 3; ++c) {
          float tmp = float(srow[si + c]) * scale + offset;
          drow[di + c] = alpha * drow[di + c] + beta * tmp;
        }
      }
    }
    return;
  }
#endif
  img_rgb_u8_to_f32_interleaved_scalar(
    src + (int64_t)y0 * src_stride, src_stride,
    (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride), dst_stride,
    width, (y1 - y0),
    scale, offset, alpha, beta
  );
}

static void img_rgb_planar_kernel_rows(
  const uint8_t* src_r, int64_t src_r_stride,
  const uint8_t* src_g, int64_t src_g_stride,
  const uint8_t* src_b, int64_t src_b_stride,
  float* dst_r, int64_t dst_r_stride,
  float* dst_g, int64_t dst_g_stride,
  float* dst_b, int64_t dst_b_stride,
  int width, int y0, int y1,
  float scale, float offset,
  float alpha, float beta
) {
  img_rgb_u8_to_f32_planar_scalar(
    src_r + (int64_t)y0 * src_r_stride, src_r_stride,
    src_g + (int64_t)y0 * src_g_stride, src_g_stride,
    src_b + (int64_t)y0 * src_b_stride, src_b_stride,
    (float*)(((uint8_t*)dst_r) + (int64_t)y0 * dst_r_stride), dst_r_stride,
    (float*)(((uint8_t*)dst_g) + (int64_t)y0 * dst_g_stride), dst_g_stride,
    (float*)(((uint8_t*)dst_b) + (int64_t)y0 * dst_b_stride), dst_b_stride,
    width, (y1 - y0),
    scale, offset, alpha, beta
  );
}

// =====================================================
//  C-API
// =====================================================
HALO_API int halo_init_features() { halo_cpuid_init(); return 0; }
HALO_API const char* halo_version() { return "HALO v0.5b"; }

HALO_API int halo_query_features(int* out_sse2, int* out_avx2, int* out_avx512) {
  if (!out_sse2 || !out_avx2 || !out_avx512) return -1;
  *out_sse2   = g_has_sse2   ? 1 : 0;
  *out_avx2   = g_has_avx2   ? 1 : 0;
  *out_avx512 = g_has_avx512 ? 1 : 0;
  return 0;
}

// Laufzeit-Konfiguration (Streaming/Threshold/Threads)
HALO_API int halo_configure(int enable_streaming, long long stream_threshold, int threads) {
  g_streaming_enabled = (enable_streaming != 0);
  if (stream_threshold > 0) g_stream_threshold = (int64_t)stream_threshold;
  if (threads > 0)          g_threads          = threads;
  ensure_pool(std::max(1, g_threads));
  return 0;
}

HALO_API int halo_autotune(long long n, int iters, int* out_saxpy_impl, int* out_sum_impl) {
  if (!out_saxpy_impl || !out_sum_impl) return -1;
  if (n <= 0 || iters <= 0) return -2;

  float* x = new(std::nothrow) float[n];
  float* y = new(std::nothrow) float[n];
  if (!x || !y) { delete[] x; delete[] y; return -3; }
  for (int64_t i=0;i<n;++i) { x[i] = float(i%100)*0.5f; y[i] = 1.0f; }
  const float a = 0.25f;
  volatile float sink=0.f;

  Impl cands[4] = { Impl::Scalar, Impl::SSE2, Impl::AVX2, Impl::AVX2_STREAM };
  int ccount = 1;
#if HALO_X86
  if (g_has_sse2) ccount = 2;
  if (g_has_avx2) ccount = 3;
  if (g_has_avx2) ccount = 4; // STREAM
#endif

  double best_t = std::numeric_limits<double>::infinity();
  Impl best     = Impl::Scalar;
  // SAXPY
  for (int i=0;i<ccount;++i) {
    std::memset(y, 0, sizeof(float)*n);
    for (int64_t j=0;j<n;++j) y[j] = 1.0f;
    double t = bench_once_saxpy(cands[i], a, x, y, n, iters);
    if (t < best_t) { best_t = t; best = cands[i]; }
  }
  g_best_saxpy = best;

  // SUM
  best_t = std::numeric_limits<double>::infinity();
  best   = Impl::Scalar;
  for (int i=0;i<ccount;++i) {
    double t = bench_once_sum(cands[i], x, n, iters, &sink);
    if (t < best_t) { best_t = t; best = cands[i]; }
  }
  g_best_sum = best;

  delete[] x; delete[] y;
  *out_saxpy_impl = (int)g_best_saxpy;
  *out_sum_impl   = (int)g_best_sum;
  return 0;
}

HALO_API int halo_set_impls(int saxpy_impl, int sum_impl) {
  if (saxpy_impl < 0 || saxpy_impl > 3) return -1;
  if (sum_impl   < 0 || sum_impl   > 3) return -1;
  g_best_saxpy = (Impl)saxpy_impl;
  g_best_sum   = (Impl)sum_impl;
  return 0;
}

HALO_API int halo_saxpy_f32(float a, const float* x, float* y, long long n) {
  if (!x || !y || n < 0) return -1;
  saxpy_dispatch(a, x, y, (int64_t)n);
  return 0;
}

HALO_API int halo_saxpy_f32_mt(float a, const float* x, float* y, long long n) {
  if (!x || !y || n < 0) return -1;
  ensure_pool(std::max(1, g_threads));
  saxpy_mt(a, x, y, (int64_t)n, std::max(1, g_threads));
  return 0;
}

// SUM (ST)
HALO_API int halo_sum_f32(const float* x, long long n, float* out_sum) {
  if (!x || !out_sum || n < 0) return -1;
  float acc=0.f;
#if HALO_X86
  if (g_has_avx2) acc = sum_avx2(x, (int64_t)n);
  else            acc = sum_scalar(x, (int64_t)n);
#else
  acc = sum_scalar(x, (int64_t)n);
#endif
  *out_sum = acc;
  return 0;
}

// -----------------------------------------------------
//  2D-Image-Kern — Export + Auto-Scheduler
// -----------------------------------------------------
HALO_API int halo_img_u8_to_f32_lut_axpby(
  const unsigned char* src, long long src_stride,
  float* dst,           long long dst_stride,
  int width, int height,
  const float* lut256,
  float scale, float offset,
  float alpha, float beta,
  int use_mt /* caller hint */
) {
  if (!src || !dst || !lut256) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < width || dst_stride < (long long)width * 4) return -3;

  const int64_t pixels        = (int64_t)width * height;
  const int64_t bytes         = pixels * 5; // ~1B read + 4B write
  const int64_t row_work      = (int64_t)width * 5;
  const bool dst_al32         = is_aligned_32(dst);
  const bool dst_stride_al32  = ((dst_stride & 31ll) == 0ll);
  const bool allow_aligned    = dst_al32 && dst_stride_al32;
  bool allow_stream = should_use_streaming(width, bytes, dst_al32, dst_stride_al32);

  int t = std::max(1, g_threads);
  ensure_pool(t);
  ScheduleDecision sched = compute_schedule(width, height, t, row_work, bytes, use_mt != 0);
  const bool do_mt = sched.do_mt;

  if (!do_mt) {
#if HALO_X86
    if (g_has_avx2) {
      float a=0.f, b=0.f;
      bool is_aff = lut_is_affine(lut256, a, b);
      bool used=false;
      if (is_aff) {
        float slope = a*scale;
        float inter = b*scale + offset;
        used = img_u8_to_f32_lut_axpby_avx2_affine(
          (const uint8_t*)src, (int64_t)src_stride, dst, (int64_t)dst_stride,
          width, height, slope, inter, alpha, beta, allow_stream, allow_aligned
        );
      } else {
        used = img_u8_to_f32_lut_axpby_avx2_gather(
          (const uint8_t*)src, (int64_t)src_stride, dst, (int64_t)dst_stride,
          width, height, lut256, scale, offset, alpha, beta, allow_stream, allow_aligned
        );
      }
      if (used) _mm_sfence();
      return 0;
    }
#endif
    img_u8_to_f32_lut_axpby_scalar(
      (const uint8_t*)src, (int64_t)src_stride, dst, (int64_t)dst_stride,
      width, height, lut256, scale, offset, alpha, beta
    );
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    [&](int y0, int y1){
      img_kernel_rows(
        (const uint8_t*)src, (int64_t)src_stride,
        dst, (int64_t)dst_stride,
        width, y0, y1,
        lut256, scale, offset, alpha, beta,
        allow_stream,
        allow_aligned
      );
    },
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

static inline int clamp_int(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline float clamp_float(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

enum class MorphOp { Min, Max };

static float morph_combine(MorphOp op, float a, float b) {
  return (op == MorphOp::Min) ? std::min(a, b) : std::max(a, b);
}

static int morph3x3_generic(
  const float* src, int64_t src_stride,
  float* dst,       int64_t dst_stride,
  int width, int height,
  MorphOp op,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (int64_t)width * 4 || dst_stride < (int64_t)width * 4) return -3;

  const int64_t row_work = (int64_t)width * 12;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  std::vector<float> tmp((size_t)width * height, 0.0f);

  auto horizontal = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* trow = tmp.data() + (size_t)y * width;
      for (int x = 0; x < width; ++x) {
        int xm1 = clamp_int(x - 1, 0, width - 1);
        int xp1 = clamp_int(x + 1, 0, width - 1);
        float v0 = srow[xm1];
        float v1 = srow[x];
        float v2 = srow[xp1];
        float best = morph_combine(op, v0, v1);
        best = morph_combine(op, best, v2);
        trow[x] = best;
      }
    }
  };

  auto vertical = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x = 0; x < width; ++x) {
        int ym1 = clamp_int(y - 1, 0, height - 1);
        int yp1 = clamp_int(y + 1, 0, height - 1);
        float v0 = tmp[(size_t)ym1 * width + x];
        float v1 = tmp[(size_t)y * width + x];
        float v2 = tmp[(size_t)yp1 * width + x];
        float best = morph_combine(op, v0, v1);
        best = morph_combine(op, best, v2);
        drow[x] = best;
      }
    }
  };

  if (!sched.do_mt) {
    horizontal(0, height);
    vertical(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    horizontal,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  ThreadPool::instance().parallel_for_rows(
    height,
    vertical,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

static void box_blur_f32_scalar(
  const float* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  int radius
) {
  if (radius <= 0) {
    for (int y=0; y<height; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + y * src_stride);
      float* drow = (float*)((uint8_t*)dst + y * dst_stride);
      std::memcpy(drow, srow, sizeof(float)*width);
    }
    return;
  }
  const int kernel = radius * 2 + 1;
  std::vector<float> tmp((size_t)width * height, 0.0f);
  for (int y=0; y<height; ++y) {
    const float* srow = (const float*)((const uint8_t*)src + y * src_stride);
    float* trow = tmp.data() + (size_t)y * width;
    for (int x=0; x<width; ++x) {
      float acc = 0.0f;
      for (int k=-radius; k<=radius; ++k) {
        int ix = clamp_int(x + k, 0, width-1);
        acc += srow[ix];
      }
      trow[x] = acc / kernel;
    }
  }
  for (int y=0; y<height; ++y) {
    float* drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      float acc = 0.0f;
      for (int k=-radius; k<=radius; ++k) {
        int iy = clamp_int(y + k, 0, height-1);
        acc += tmp[(size_t)iy * width + x];
      }
      drow[x] = acc / kernel;
    }
  }
}

static std::vector<float> make_gaussian_kernel(float sigma) {
  int radius = std::max(1, (int)std::ceil(3.0f * sigma));
  std::vector<float> w((size_t)radius * 2 + 1);
  const float inv_sigma2 = 1.0f / (2.0f * sigma * sigma);
  float sum = 0.0f;
  for (int i=-radius; i<=radius; ++i) {
    float val = std::exp(-float(i*i) * inv_sigma2);
    w[(size_t)(i + radius)] = val;
    sum += val;
  }
  for (float& v : w) v /= sum;
  return w;
}

static void gaussian_blur_f32_scalar(
  const float* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  float sigma
) {
  if (sigma <= 0.0f) {
    for (int y=0; y<height; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + y * src_stride);
      float* drow = (float*)((uint8_t*)dst + y * dst_stride);
      std::memcpy(drow, srow, sizeof(float)*width);
    }
    return;
  }
  std::vector<float> kernel = make_gaussian_kernel(sigma);
  int radius = (int)kernel.size() / 2;
  std::vector<float> tmp((size_t)width * height, 0.0f);
  for (int y=0; y<height; ++y) {
    const float* srow = (const float*)((const uint8_t*)src + y * src_stride);
    float* trow = tmp.data() + (size_t)y * width;
    for (int x=0; x<width; ++x) {
      float acc = 0.0f;
      for (int k=-radius; k<=radius; ++k) {
        int ix = clamp_int(x + k, 0, width-1);
        acc += srow[ix] * kernel[(size_t)(k + radius)];
      }
      trow[x] = acc;
    }
  }
  for (int y=0; y<height; ++y) {
    float* drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      float acc = 0.0f;
      for (int k=-radius; k<=radius; ++k) {
        int iy = clamp_int(y + k, 0, height-1);
        acc += tmp[(size_t)iy * width + x] * kernel[(size_t)(k + radius)];
      }
      drow[x] = acc;
    }
  }
}

static void sobel_f32_scalar(
  const float* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height
) {
  for (int y=0; y<height; ++y) {
    float* drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      float gx = 0.0f;
      float gy = 0.0f;
      for (int ky=-1; ky<=1; ++ky) {
        int sy = clamp_int(y + ky, 0, height-1);
        const float* srow = (const float*)((const uint8_t*)src + sy * src_stride);
        for (int kx=-1; kx<=1; ++kx) {
          int sx = clamp_int(x + kx, 0, width-1);
          float val = srow[sx];
          int ix = kx + 1;
          int iy = ky + 1;
          static const int sobel_x[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
          };
          static const int sobel_y[3][3] = {
            { 1,  2,  1},
            { 0,  0,  0},
            {-1, -2, -1}
          };
          gx += val * sobel_x[iy][ix];
          gy += val * sobel_y[iy][ix];
        }
      }
      drow[x] = std::sqrt(gx*gx + gy*gy);
    }
  }
}

static void resize_bilinear_f32_scalar(
  const float* src, int64_t src_stride,
  int src_w, int src_h,
  float* dst, int64_t dst_stride,
  int dst_w, int dst_h
) {
  const float scale_x = (src_w > 1) ? float(src_w - 1) / float(dst_w - 1) : 0.0f;
  const float scale_y = (src_h > 1) ? float(src_h - 1) / float(dst_h - 1) : 0.0f;
  for (int y=0; y<dst_h; ++y) {
    float fy = scale_y * y;
    int y0 = clamp_int((int)std::floor(fy), 0, src_h - 1);
    int y1 = clamp_int(y0 + 1, 0, src_h - 1);
    float wy = fy - y0;
    const float* row0 = (const float*)((const uint8_t*)src + y0 * src_stride);
    const float* row1 = (const float*)((const uint8_t*)src + y1 * src_stride);
    float* drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<dst_w; ++x) {
      float fx = scale_x * x;
      int x0 = clamp_int((int)std::floor(fx), 0, src_w - 1);
      int x1 = clamp_int(x0 + 1, 0, src_w - 1);
      float wx = fx - x0;
      float v0 = row0[x0] * (1.0f - wx) + row0[x1] * wx;
      float v1 = row1[x0] * (1.0f - wx) + row1[x1] * wx;
      drow[x] = v0 * (1.0f - wy) + v1 * wy;
    }
  }
}

static float cubic_interp(float a, float b, float c, float d, float t) {
  float A = -0.5f*a + 1.5f*b - 1.5f*c + 0.5f*d;
  float B = a - 2.5f*b + 2.0f*c - 0.5f*d;
  float C = -0.5f*a + 0.5f*c;
  float D = b;
  return ((A*t + B)*t + C)*t + D;
}

static inline void catmull_rom_weights(float t, float out[4]) {
  float t2 = t * t;
  float t3 = t2 * t;
  out[0] = -0.5f * t3 + t2 - 0.5f * t;
  out[1] =  1.5f * t3 - 2.5f * t2 + 1.0f;
  out[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
  out[3] =  0.5f * t3 - 0.5f * t2;
}

static void resize_bicubic_f32_scalar(
  const float* src, int64_t src_stride,
  int src_w, int src_h,
  float* dst, int64_t dst_stride,
  int dst_w, int dst_h
) {
  const float scale_x = (float)src_w / (float)dst_w;
  const float scale_y = (float)src_h / (float)dst_h;
  for (int y=0; y<dst_h; ++y) {
    float fy = (y + 0.5f) * scale_y - 0.5f;
    int y1 = clamp_int((int)std::floor(fy), 0, src_h - 1);
    float wy = fy - y1;
    float* drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<dst_w; ++x) {
      float fx = (x + 0.5f) * scale_x - 0.5f;
      int x1 = clamp_int((int)std::floor(fx), 0, src_w - 1);
      float wx = fx - x1;
      float samples[4];
      for (int k=0; k<4; ++k) {
        int sy = clamp_int(y1 + k - 1, 0, src_h - 1);
        const float* row = (const float*)((const uint8_t*)src + sy * src_stride);
        float col[4];
        for (int j=0; j<4; ++j) {
          int sx = clamp_int(x1 + j - 1, 0, src_w - 1);
          col[j] = row[sx];
        }
        samples[k] = cubic_interp(col[0], col[1], col[2], col[3], wx);
      }
      drow[x] = cubic_interp(samples[0], samples[1], samples[2], samples[3], wy);
    }
  }
}

static void relu_clamp_axpby_scalar(
  const float* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  float alpha, float beta,
  float clamp_min, float clamp_max,
  bool apply_relu
) {
  for (int y=0; y<height; ++y) {
    const float* srow = (const float*)((const uint8_t*)src + y * src_stride);
    float* drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      float val = alpha * drow[x] + beta * srow[x];
      if (apply_relu && val < 0.0f) val = 0.0f;
      if (val < clamp_min) val = clamp_min;
      if (val > clamp_max) val = clamp_max;
      drow[x] = val;
    }
  }
}

HALO_API int halo_img_u16_to_f32_axpby(
  const unsigned short* src, long long src_stride,
  float* dst,               long long dst_stride,
  int width, int height,
  int bit_depth,
  float scale, float offset,
  float alpha, float beta,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 2 || dst_stride < (long long)width * 4) return -3;
  if (bit_depth <= 0 || bit_depth > 16) return -4;

  const uint16_t mask = (bit_depth == 16) ? 0xFFFFu : (uint16_t)((1u << bit_depth) - 1u);
  const float inv_max = 1.0f / float((bit_depth == 16) ? 65535.0f : float((1u << bit_depth) - 1u));

  const int64_t row_work = (int64_t)width * 6;
  const int64_t total_bytes = row_work * height;

  const bool dst_al32        = is_aligned_32(dst);
  const bool dst_stride_al32 = ((dst_stride & 31ll) == 0ll);
  const bool allow_aligned   = dst_al32 && dst_stride_al32;
  bool allow_stream = should_use_streaming(width, total_bytes, dst_al32, dst_stride_al32);

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  if (!sched.do_mt) {
#if HALO_X86
    if (g_has_avx2) {
      bool used = img_u16_to_f32_axpby_avx2(
        (const uint16_t*)src, (int64_t)src_stride,
        dst, (int64_t)dst_stride,
        width, height,
        mask, inv_max, scale, offset, alpha, beta,
        allow_stream, allow_aligned
      );
      if (used) _mm_sfence();
      return 0;
    }
#endif
    img_u16_to_f32_axpby_scalar(
      (const uint16_t*)src, (int64_t)src_stride,
      dst, (int64_t)dst_stride,
      width, height,
      mask, inv_max, scale, offset, alpha, beta
    );
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    [&](int y0, int y1){
      img_u16_kernel_rows(
        (const uint16_t*)src, (int64_t)src_stride,
        dst, (int64_t)dst_stride,
        width, y0, y1,
        mask, inv_max,
        scale, offset, alpha, beta,
        allow_stream,
        allow_aligned
      );
    },
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_img_rgb_u8_to_f32_interleaved(
  const unsigned char* src, long long src_stride,
  float* dst,               long long dst_stride,
  int width, int height,
  float scale, float offset,
  float alpha, float beta,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 3 || dst_stride < (long long)width * 3 * 4) return -3;

  const int64_t row_work = (int64_t)width * 15;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  if (!sched.do_mt) {
    img_rgb_u8_to_f32_interleaved_scalar(
      src, (int64_t)src_stride,
      dst, (int64_t)dst_stride,
      width, height,
      scale, offset, alpha, beta
    );
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    [&](int y0, int y1){
      img_rgb_interleaved_kernel_rows(
        src, (int64_t)src_stride,
        dst, (int64_t)dst_stride,
        width, y0, y1,
        scale, offset, alpha, beta
      );
    },
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_img_rgb_u8_to_f32_planar(
  const unsigned char* src_r, long long src_r_stride,
  const unsigned char* src_g, long long src_g_stride,
  const unsigned char* src_b, long long src_b_stride,
  float* dst_r, long long dst_r_stride,
  float* dst_g, long long dst_g_stride,
  float* dst_b, long long dst_b_stride,
  int width, int height,
  float scale, float offset,
  float alpha, float beta,
  int use_mt
) {
  if (!src_r || !src_g || !src_b || !dst_r || !dst_g || !dst_b) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_r_stride < width || src_g_stride < width || src_b_stride < width) return -3;
  if (dst_r_stride < (long long)width * 4 || dst_g_stride < (long long)width * 4 || dst_b_stride < (long long)width * 4) return -4;

  const int64_t row_work = (int64_t)width * 15;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  if (!sched.do_mt) {
    img_rgb_u8_to_f32_planar_scalar(
      src_r, (int64_t)src_r_stride,
      src_g, (int64_t)src_g_stride,
      src_b, (int64_t)src_b_stride,
      dst_r, (int64_t)dst_r_stride,
      dst_g, (int64_t)dst_g_stride,
      dst_b, (int64_t)dst_b_stride,
      width, height,
      scale, offset, alpha, beta
    );
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    [&](int y0, int y1){
      img_rgb_planar_kernel_rows(
        src_r, (int64_t)src_r_stride,
        src_g, (int64_t)src_g_stride,
        src_b, (int64_t)src_b_stride,
        dst_r, (int64_t)dst_r_stride,
        dst_g, (int64_t)dst_g_stride,
        dst_b, (int64_t)dst_b_stride,
        width, y0, y1,
        scale, offset, alpha, beta
      );
    },
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_box_blur_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int radius,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
  if (radius < 0) return -4;

  const int64_t row_work = (int64_t)width * (8 + std::max(1, radius) * 4);
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  std::vector<float> tmp((size_t)width * height, 0.0f);
  const int kernel = radius * 2 + 1;
  const float inv_kernel = 1.0f / (float)kernel;

  auto horizontal = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* trow = tmp.data() + (size_t)y * width;
#if HALO_X86
      if (g_has_avx2 && width >= 8) {
        const __m256 vnorm = _mm256_set1_ps(inv_kernel);
        int x = 0;
        int edge = std::min(width, radius);
        for (; x < edge; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int ix = clamp_int(x + k, 0, width-1);
            acc += srow[ix];
          }
          trow[x] = acc * inv_kernel;
        }
        int interior_end = width - radius;
        if (interior_end < edge) interior_end = edge;
        for (; x + 8 <= interior_end; x += 8) {
          __m256 acc = _mm256_setzero_ps();
          for (int k=-radius; k<=radius; ++k) {
            __m256 vals = _mm256_loadu_ps(srow + x + k);
            acc = _mm256_add_ps(acc, vals);
          }
          _mm256_storeu_ps(trow + x, _mm256_mul_ps(acc, vnorm));
        }
        for (; x < width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int ix = clamp_int(x + k, 0, width-1);
            acc += srow[ix];
          }
          trow[x] = acc * inv_kernel;
        }
      } else
#endif
      {
        for (int x=0; x<width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int ix = clamp_int(x + k, 0, width-1);
            acc += srow[ix];
          }
          trow[x] = acc * inv_kernel;
        }
      }
    }
  };

  auto vertical = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
#if HALO_X86
      if (g_has_avx2 && width >= 8) {
        const __m256 vnorm = _mm256_set1_ps(inv_kernel);
        int x = 0;
        for (; x + 8 <= width; x += 8) {
          __m256 acc = _mm256_setzero_ps();
          for (int k=-radius; k<=radius; ++k) {
            int iy = clamp_int(y + k, 0, height-1);
            const float* srow = tmp.data() + (size_t)iy * width;
            __m256 vals = _mm256_loadu_ps(srow + x);
            acc = _mm256_add_ps(acc, vals);
          }
          _mm256_storeu_ps(drow + x, _mm256_mul_ps(acc, vnorm));
        }
        for (; x < width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int iy = clamp_int(y + k, 0, height-1);
            acc += tmp[(size_t)iy * width + x];
          }
          drow[x] = acc * inv_kernel;
        }
      } else
#endif
      {
        for (int x=0; x<width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int iy = clamp_int(y + k, 0, height-1);
            acc += tmp[(size_t)iy * width + x];
          }
          drow[x] = acc * inv_kernel;
        }
      }
    }
  };

  if (!sched.do_mt) {
    horizontal(0, height);
    vertical(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    horizontal,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  ThreadPool::instance().parallel_for_rows(
    height,
    vertical,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_gaussian_blur_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  float sigma,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
  if (sigma < 0.0f) return -4;

  if (sigma == 0.0f) {
    for (int y=0; y<height; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      std::memcpy(drow, srow, sizeof(float)*width);
    }
    return 0;
  }

  std::vector<float> kernel = make_gaussian_kernel(sigma);
  int radius = (int)kernel.size() / 2;
  const float* kernel_data = kernel.data();

  const int64_t row_work = (int64_t)width * (8 + (radius * 4));
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  std::vector<float> tmp((size_t)width * height, 0.0f);

  auto horizontal = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* trow = tmp.data() + (size_t)y * width;
#if HALO_X86
      if (g_has_avx2 && width >= 8) {
        int x = 0;
        int edge = std::min(width, radius);
        for (; x < edge; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int ix = clamp_int(x + k, 0, width-1);
            acc += srow[ix] * kernel_data[k + radius];
          }
          trow[x] = acc;
        }
        int interior_end = width - radius;
        if (interior_end < edge) interior_end = edge;
        for (; x + 8 <= interior_end; x += 8) {
          __m256 vacc = _mm256_setzero_ps();
          for (int k=-radius; k<=radius; ++k) {
            __m256 vals = _mm256_loadu_ps(srow + x + k);
            __m256 wk = _mm256_set1_ps(kernel_data[k + radius]);
            vacc = _mm256_fmadd_ps(vals, wk, vacc);
          }
          _mm256_storeu_ps(trow + x, vacc);
        }
        for (; x < width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int ix = clamp_int(x + k, 0, width-1);
            acc += srow[ix] * kernel_data[k + radius];
          }
          trow[x] = acc;
        }
      } else
#endif
      {
        for (int x=0; x<width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int ix = clamp_int(x + k, 0, width-1);
            acc += srow[ix] * kernel_data[k + radius];
          }
          trow[x] = acc;
        }
      }
    }
  };

  auto vertical = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
#if HALO_X86
      if (g_has_avx2 && width >= 8) {
        int x = 0;
        for (; x + 8 <= width; x += 8) {
          __m256 vacc = _mm256_setzero_ps();
          for (int k=-radius; k<=radius; ++k) {
            int iy = clamp_int(y + k, 0, height-1);
            const float* srow = tmp.data() + (size_t)iy * width;
            __m256 vals = _mm256_loadu_ps(srow + x);
            __m256 wk = _mm256_set1_ps(kernel_data[k + radius]);
            vacc = _mm256_fmadd_ps(vals, wk, vacc);
          }
          _mm256_storeu_ps(drow + x, vacc);
        }
        for (; x < width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int iy = clamp_int(y + k, 0, height-1);
            acc += tmp[(size_t)iy * width + x] * kernel_data[k + radius];
          }
          drow[x] = acc;
        }
      } else
#endif
      {
        for (int x=0; x<width; ++x) {
          float acc = 0.0f;
          for (int k=-radius; k<=radius; ++k) {
            int iy = clamp_int(y + k, 0, height-1);
            acc += tmp[(size_t)iy * width + x] * kernel_data[k + radius];
          }
          drow[x] = acc;
        }
      }
    }
  };

  if (!sched.do_mt) {
    horizontal(0, height);
    vertical(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    horizontal,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  ThreadPool::instance().parallel_for_rows(
    height,
    vertical,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_sobel_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;

  const int64_t row_work = (int64_t)width * 16;
  const int64_t total_bytes = row_work * height;
  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  static const int sobel_x_kernel[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
  };
  static const int sobel_y_kernel[3][3] = {
    { 1,  2,  1},
    { 0,  0,  0},
    {-1, -2, -1}
  };

  auto worker = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
#if HALO_X86
      if (g_has_avx2 && width >= 8 && y > 0 && y < height-1) {
        const float* rowm1 = (const float*)((const uint8_t*)src + (int64_t)(y-1) * src_stride);
        const float* row0  = (const float*)((const uint8_t*)src + (int64_t)y     * src_stride);
        const float* rowp1 = (const float*)((const uint8_t*)src + (int64_t)(y+1) * src_stride);
        int x = 0;
        for (; x < 1; ++x) {
          float gx = 0.0f;
          float gy = 0.0f;
          for (int ky=-1; ky<=1; ++ky) {
            int sy = clamp_int(y + ky, 0, height-1);
            const float* srow = (const float*)((const uint8_t*)src + (int64_t)sy * src_stride);
            for (int kx=-1; kx<=1; ++kx) {
              int sx = clamp_int(x + kx, 0, width-1);
              float val = srow[sx];
              gx += val * sobel_x_kernel[ky+1][kx+1];
              gy += val * sobel_y_kernel[ky+1][kx+1];
            }
          }
          drow[x] = std::sqrt(gx*gx + gy*gy);
        }
        int limit = width - 1;
        const __m256 c1 = _mm256_set1_ps(1.0f);
        const __m256 c2 = _mm256_set1_ps(2.0f);
        for (; x + 8 <= limit; x += 8) {
          __m256 top_l = _mm256_loadu_ps(rowm1 + x - 1);
          __m256 top_c = _mm256_loadu_ps(rowm1 + x);
          __m256 top_r = _mm256_loadu_ps(rowm1 + x + 1);
          __m256 mid_l = _mm256_loadu_ps(row0  + x - 1);
          __m256 mid_r = _mm256_loadu_ps(row0  + x + 1);
          __m256 bot_l = _mm256_loadu_ps(rowp1 + x - 1);
          __m256 bot_c = _mm256_loadu_ps(rowp1 + x);
          __m256 bot_r = _mm256_loadu_ps(rowp1 + x + 1);

          __m256 gx = _mm256_setzero_ps();
          gx = _mm256_fnmadd_ps(top_l, c1, gx);
          gx = _mm256_fnmadd_ps(mid_l, c2, gx);
          gx = _mm256_fnmadd_ps(bot_l, c1, gx);
          gx = _mm256_fmadd_ps(top_r, c1, gx);
          gx = _mm256_fmadd_ps(mid_r, c2, gx);
          gx = _mm256_fmadd_ps(bot_r, c1, gx);

          __m256 gy = _mm256_setzero_ps();
          gy = _mm256_fmadd_ps(top_l, c1, gy);
          gy = _mm256_fmadd_ps(top_c, c2, gy);
          gy = _mm256_fmadd_ps(top_r, c1, gy);
          gy = _mm256_fnmadd_ps(bot_l, c1, gy);
          gy = _mm256_fnmadd_ps(bot_c, c2, gy);
          gy = _mm256_fnmadd_ps(bot_r, c1, gy);

          __m256 mag = _mm256_add_ps(_mm256_mul_ps(gx, gx), _mm256_mul_ps(gy, gy));
          mag = _mm256_sqrt_ps(mag);
          _mm256_storeu_ps(drow + x, mag);
        }
        for (; x < width; ++x) {
          float gx = 0.0f;
          float gy = 0.0f;
          for (int ky=-1; ky<=1; ++ky) {
            int sy = clamp_int(y + ky, 0, height-1);
            const float* srow = (const float*)((const uint8_t*)src + (int64_t)sy * src_stride);
            for (int kx=-1; kx<=1; ++kx) {
              int sx = clamp_int(x + kx, 0, width-1);
              float val = srow[sx];
              gx += val * sobel_x_kernel[ky+1][kx+1];
              gy += val * sobel_y_kernel[ky+1][kx+1];
            }
          }
          drow[x] = std::sqrt(gx*gx + gy*gy);
        }
        continue;
      }
#endif
      for (int x=0; x<width; ++x) {
        float gx = 0.0f;
        float gy = 0.0f;
        for (int ky=-1; ky<=1; ++ky) {
          int sy = clamp_int(y + ky, 0, height-1);
          const float* srow = (const float*)((const uint8_t*)src + (int64_t)sy * src_stride);
          for (int kx=-1; kx<=1; ++kx) {
            int sx = clamp_int(x + kx, 0, width-1);
            float val = srow[sx];
            gx += val * sobel_x_kernel[ky+1][kx+1];
            gy += val * sobel_y_kernel[ky+1][kx+1];
          }
        }
        drow[x] = std::sqrt(gx*gx + gy*gy);
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    worker,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_resize_bilinear_f32(
  const float* src, long long src_stride,
  int src_width, int src_height,
  float* dst, long long dst_stride,
  int dst_width, int dst_height,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0) return -2;
  if (src_stride < (long long)src_width * 4 || dst_stride < (long long)dst_width * 4) return -3;

  const int64_t row_work = (int64_t)dst_width * 16;
  const int64_t total_bytes = row_work * dst_height;
  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(dst_width, dst_height, threads, row_work, total_bytes, use_mt != 0);

  std::vector<int>  x0_idx(dst_width);
  std::vector<int>  x1_idx(dst_width);
  std::vector<float> wx_factor(dst_width);
  const float scale_x = (src_width > 1 && dst_width > 1) ? float(src_width - 1) / float(dst_width - 1) : 0.0f;
  for (int x=0; x<dst_width; ++x) {
    float fx = scale_x * x;
    int x0s = clamp_int((int)std::floor(fx), 0, src_width - 1);
    int x1s = clamp_int(x0s + 1, 0, src_width - 1);
    x0_idx[x] = x0s;
    x1_idx[x] = x1s;
    wx_factor[x] = fx - (float)x0s;
  }

  const float scale_y = (src_height > 1 && dst_height > 1) ? float(src_height - 1) / float(dst_height - 1) : 0.0f;

  auto worker = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      float fy = scale_y * y;
      int y0s = clamp_int((int)std::floor(fy), 0, src_height - 1);
      int y1s = clamp_int(y0s + 1, 0, src_height - 1);
      float wy = fy - (float)y0s;
      const float* row0 = (const float*)((const uint8_t*)src + (int64_t)y0s * src_stride);
      const float* row1 = (const float*)((const uint8_t*)src + (int64_t)y1s * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
#if HALO_X86
      if (g_has_avx2 && dst_width >= 8) {
        const __m256 wy_v   = _mm256_set1_ps(wy);
        const __m256 one    = _mm256_set1_ps(1.0f);
        const __m256 w1_y   = _mm256_sub_ps(one, wy_v);
        int x = 0;
        for (; x + 8 <= dst_width; x += 8) {
          __m256 wx = _mm256_loadu_ps(wx_factor.data() + x);
          __m256 w0 = _mm256_sub_ps(one, wx);
          __m256 w1 = wx;
          __m256i ix0 = _mm256_loadu_si256((const __m256i*)(x0_idx.data() + x));
          __m256i ix1 = _mm256_loadu_si256((const __m256i*)(x1_idx.data() + x));
          __m256 top0 = _mm256_i32gather_ps(row0, ix0, 4);
          __m256 top1 = _mm256_i32gather_ps(row0, ix1, 4);
          __m256 bot0 = _mm256_i32gather_ps(row1, ix0, 4);
          __m256 bot1 = _mm256_i32gather_ps(row1, ix1, 4);
          __m256 top = _mm256_fmadd_ps(top1, w1, _mm256_mul_ps(top0, w0));
          __m256 bot = _mm256_fmadd_ps(bot1, w1, _mm256_mul_ps(bot0, w0));
          __m256 out = _mm256_fmadd_ps(bot, wy_v, _mm256_mul_ps(top, w1_y));
          _mm256_storeu_ps(drow + x, out);
        }
        for (; x < dst_width; ++x) {
          int x0s = x0_idx[x];
          int x1s = x1_idx[x];
          float wx = wx_factor[x];
          float w0 = 1.0f - wx;
          float top = row0[x0s] * w0 + row0[x1s] * wx;
          float bot = row1[x0s] * w0 + row1[x1s] * wx;
          drow[x] = top * (1.0f - wy) + bot * wy;
        }
      } else
#endif
      {
        for (int x=0; x<dst_width; ++x) {
          int x0s = x0_idx[x];
          int x1s = x1_idx[x];
          float wx = wx_factor[x];
          float w0 = 1.0f - wx;
          float top = row0[x0s] * w0 + row0[x1s] * wx;
          float bot = row1[x0s] * w0 + row1[x1s] * wx;
          drow[x] = top * (1.0f - wy) + bot * wy;
        }
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, dst_height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    dst_height,
    worker,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_resize_bicubic_f32(
  const float* src, long long src_stride,
  int src_width, int src_height,
  float* dst, long long dst_stride,
  int dst_width, int dst_height,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0) return -2;
  if (src_stride < (long long)src_width * 4 || dst_stride < (long long)dst_width * 4) return -3;

  const int64_t row_work = (int64_t)dst_width * 32;
  const int64_t total_bytes = row_work * dst_height;
  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(dst_width, dst_height, threads, row_work, total_bytes, use_mt != 0);

  std::vector<int>   x_idx0(dst_width), x_idx1(dst_width), x_idx2(dst_width), x_idx3(dst_width);
  std::vector<float> x_w0(dst_width),   x_w1(dst_width),   x_w2(dst_width),   x_w3(dst_width);
  const float inv_dst_w = (dst_width > 0) ? 1.0f / (float)dst_width : 0.0f;
  for (int x=0; x<dst_width; ++x) {
    float fx = (float)src_width * (float(x) + 0.5f) * inv_dst_w - 0.5f;
    int x1 = clamp_int((int)std::floor(fx), 0, src_width - 1);
    float tx = fx - (float)x1;
    float w[4];
    catmull_rom_weights(tx, w);
    x_idx0[x] = clamp_int(x1 - 1, 0, src_width - 1);
    x_idx1[x] = clamp_int(x1 + 0, 0, src_width - 1);
    x_idx2[x] = clamp_int(x1 + 1, 0, src_width - 1);
    x_idx3[x] = clamp_int(x1 + 2, 0, src_width - 1);
    x_w0[x] = w[0];
    x_w1[x] = w[1];
    x_w2[x] = w[2];
    x_w3[x] = w[3];
  }

  auto worker = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      float fy = (float)src_height * (float(y) + 0.5f) / (float)dst_height - 0.5f;
      int y1s = clamp_int((int)std::floor(fy), 0, src_height - 1);
      float wy = fy - y1s;
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
#if HALO_X86
      if (g_has_avx2 && dst_width >= 8) {
        int sy_idx[4];
        float wy_weights[4];
        catmull_rom_weights(wy, wy_weights);
        for (int k=0; k<4; ++k) {
          sy_idx[k] = clamp_int(y1s + k - 1, 0, src_height - 1);
        }
        int x = 0;
        for (; x + 8 <= dst_width; x += 8) {
          __m256 accum = _mm256_setzero_ps();
          __m256 w0 = _mm256_loadu_ps(x_w0.data() + x);
          __m256 w1 = _mm256_loadu_ps(x_w1.data() + x);
          __m256 w2 = _mm256_loadu_ps(x_w2.data() + x);
          __m256 w3 = _mm256_loadu_ps(x_w3.data() + x);
          __m256i ix0 = _mm256_loadu_si256((const __m256i*)(x_idx0.data() + x));
          __m256i ix1 = _mm256_loadu_si256((const __m256i*)(x_idx1.data() + x));
          __m256i ix2 = _mm256_loadu_si256((const __m256i*)(x_idx2.data() + x));
          __m256i ix3 = _mm256_loadu_si256((const __m256i*)(x_idx3.data() + x));
          for (int ky=0; ky<4; ++ky) {
            const float* row = (const float*)((const uint8_t*)src + (int64_t)sy_idx[ky] * src_stride);
            __m256 s0 = _mm256_i32gather_ps(row, ix0, 4);
            __m256 s1 = _mm256_i32gather_ps(row, ix1, 4);
            __m256 s2 = _mm256_i32gather_ps(row, ix2, 4);
            __m256 s3 = _mm256_i32gather_ps(row, ix3, 4);
            __m256 lane = _mm256_mul_ps(s0, w0);
            lane = _mm256_fmadd_ps(s1, w1, lane);
            lane = _mm256_fmadd_ps(s2, w2, lane);
            lane = _mm256_fmadd_ps(s3, w3, lane);
            __m256 wyv = _mm256_set1_ps(wy_weights[ky]);
            accum = _mm256_fmadd_ps(lane, wyv, accum);
          }
          _mm256_storeu_ps(drow + x, accum);
        }
        for (; x < dst_width; ++x) {
          float col_vals[4];
          for (int ky=0; ky<4; ++ky) {
            const float* row = (const float*)((const uint8_t*)src + (int64_t)sy_idx[ky] * src_stride);
            float val = row[x_idx0[x]] * x_w0[x]
                      + row[x_idx1[x]] * x_w1[x]
                      + row[x_idx2[x]] * x_w2[x]
                      + row[x_idx3[x]] * x_w3[x];
            col_vals[ky] = val;
          }
          float result = 0.0f;
          for (int ky=0; ky<4; ++ky) {
            result += col_vals[ky] * wy_weights[ky];
          }
          drow[x] = result;
        }
        continue;
      }
#endif
      for (int x=0; x<dst_width; ++x) {
        float fx = (float)src_width * (float(x) + 0.5f) / (float)dst_width - 0.5f;
        int x1s = clamp_int((int)std::floor(fx), 0, src_width - 1);
        float wx = fx - x1s;
        float samples[4];
        for (int k=0; k<4; ++k) {
          int sy = clamp_int(y1s + k - 1, 0, src_height - 1);
          const float* row = (const float*)((const uint8_t*)src + (int64_t)sy * src_stride);
          float cols[4];
          for (int j=0; j<4; ++j) {
            int sx = clamp_int(x1s + j - 1, 0, src_width - 1);
            cols[j] = row[sx];
          }
          samples[k] = cubic_interp(cols[0], cols[1], cols[2], cols[3], wx);
        }
        drow[x] = cubic_interp(samples[0], samples[1], samples[2], samples[3], wy);
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, dst_height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    dst_height,
    worker,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_flip_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int flip_horizontal, int flip_vertical,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;

  const bool do_h = (flip_horizontal != 0);
  const bool do_v = (flip_vertical   != 0);

  if (!do_h && !do_v) {
    if (src == dst && src_stride == dst_stride) return 0;
    for (int y = 0; y < height; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      std::memcpy(drow, srow, sizeof(float) * width);
    }
    return 0;
  }

  if (src == dst && src_stride == dst_stride) {
    uint8_t* base = (uint8_t*)dst;
    if (do_v) {
      std::vector<float> swap_buf((size_t)width);
      for (int y = 0; y < height / 2; ++y) {
        float* top = (float*)(base + (int64_t)y * dst_stride);
        float* bottom = (float*)(base + (int64_t)(height - 1 - y) * dst_stride);
        std::memcpy(swap_buf.data(), top, sizeof(float) * width);
        std::memcpy(top, bottom, sizeof(float) * width);
        std::memcpy(bottom, swap_buf.data(), sizeof(float) * width);
      }
    }
    if (do_h) {
      for (int y = 0; y < height; ++y) {
        float* row = (float*)(base + (int64_t)y * dst_stride);
        int l = 0;
        int r = width - 1;
        while (l < r) {
          std::swap(row[l], row[r]);
          ++l;
          --r;
        }
      }
    }
    return 0;
  }

  const int64_t row_work = (int64_t)width * 12;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

#if HALO_X86
  const __m256i perm_rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
#endif

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      int dy = do_v ? (height - 1 - y) : y;
      float* drow = (float*)((uint8_t*)dst + (int64_t)dy * dst_stride);
      if (!do_h) {
        std::memcpy(drow, srow, sizeof(float) * width);
        continue;
      }
#if HALO_X86
      if (g_has_avx2 && width >= 8) {
        int x = 0;
        for (; x + 8 <= width; x += 8) {
          __m256 v = _mm256_loadu_ps(srow + x);
          v = _mm256_permutevar8x32_ps(v, perm_rev);
          _mm256_storeu_ps(drow + (width - x - 8), v);
        }
        for (int xi = x; xi < width; ++xi) {
          drow[width - 1 - xi] = srow[xi];
        }
      } else
#endif
      {
        for (int xi = 0; xi < width; ++xi) {
          drow[width - 1 - xi] = srow[xi];
        }
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    worker,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_rotate90_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int quarter_turns,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4) return -3;

  int rot = quarter_turns % 4;
  if (rot < 0) rot += 4;

  if (rot == 0) {
    if (src == dst && src_stride == dst_stride) return 0;
    if (dst_stride < (long long)width * 4) return -4;
    for (int y = 0; y < height; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      std::memcpy(drow, srow, sizeof(float) * width);
    }
    return 0;
  }

  if (rot == 2) {
    if (dst_stride < (long long)width * 4) return -4;
    return halo_flip_f32(src, src_stride, dst, dst_stride, width, height, 1, 1, use_mt);
  }

  const int dst_width  = (rot % 2 == 0) ? width  : height;
  const int dst_height = (rot % 2 == 0) ? height : width;
  if (dst_stride < (long long)dst_width * 4) return -4;

  if (src == dst && src_stride == dst_stride) {
    return -5;
  }

  std::vector<const float*> src_rows((size_t)height);
  for (int y = 0; y < height; ++y) {
    src_rows[(size_t)y] = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
  }

  const int64_t row_work = (int64_t)dst_width * 12;
  const int64_t total_bytes = row_work * dst_height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(dst_width, dst_height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      if (rot == 1) {
        int sx = y;
        for (int x = 0; x < dst_width; ++x) {
          int sy = height - 1 - x;
          drow[x] = src_rows[(size_t)sy][sx];
        }
      } else { // rot == 3
        int sx = width - 1 - y;
        for (int x = 0; x < dst_width; ++x) {
          int sy = x;
          drow[x] = src_rows[(size_t)sy][sx];
        }
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, dst_height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    dst_height,
    worker,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

HALO_API int halo_invert_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  float min_val, float max_val,
  int use_range,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;

  float lo = use_range ? min_val : 0.0f;
  float hi = use_range ? max_val : 1.0f;
  if (!(hi > lo)) return -4;

  const int64_t row_work = (int64_t)width * 8;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x = 0; x < width; ++x) {
        float inv = lo + hi - srow[x];
        drow[x] = clamp_float(inv, lo, hi);
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(height, worker, sched.min_rows, sched.chunk_hint);
  return 0;
}

HALO_API int halo_gamma_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  float gamma,
  float gain,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
  if (!(gamma > 0.0f)) return -4;

  const int64_t row_work = (int64_t)width * 8;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x = 0; x < width; ++x) {
        float normalized = srow[x];
        if (normalized < 0.0f) normalized = 0.0f;
        float mapped = std::pow(normalized, gamma);
        drow[x] = gain * mapped;
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(height, worker, sched.min_rows, sched.chunk_hint);
  return 0;
}

HALO_API int halo_levels_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  float in_low, float in_high,
  float out_low, float out_high,
  float gamma,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
  if (!(in_high > in_low)) return -4;
  if (!(gamma > 0.0f)) gamma = 1.0f;

  const float inv_in_range = 1.0f / (in_high - in_low);
  const float out_range = out_high - out_low;

  const int64_t row_work = (int64_t)width * 12;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x = 0; x < width; ++x) {
        float val = (srow[x] - in_low) * inv_in_range;
        val = clamp_float(val, 0.0f, 1.0f);
        if (gamma != 1.0f) {
          val = std::pow(val, gamma);
        }
        drow[x] = out_low + val * out_range;
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(height, worker, sched.min_rows, sched.chunk_hint);
  return 0;
}

HALO_API int halo_threshold_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  float low, float high,
  float low_value, float high_value,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;

  const bool has_range = (high > low);

  const int64_t row_work = (int64_t)width * 8;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x = 0; x < width; ++x) {
        float val = srow[x];
        if (!has_range) {
          drow[x] = (val >= low) ? high_value : low_value;
        } else {
          if (val <= low) {
            drow[x] = low_value;
          } else if (val >= high) {
            drow[x] = high_value;
          } else {
            float t = (val - low) / (high - low);
            drow[x] = low_value + t * (high_value - low_value);
          }
        }
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(height, worker, sched.min_rows, sched.chunk_hint);
  return 0;
}

HALO_API int halo_median3x3_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;

  const int64_t row_work = (int64_t)width * 24;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x = 0; x < width; ++x) {
        float vals[9];
        int idx = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          int sy = clamp_int(y + ky, 0, height - 1);
          const float* srow = (const float*)((const uint8_t*)src + (int64_t)sy * src_stride);
          for (int kx = -1; kx <= 1; ++kx) {
            int sx = clamp_int(x + kx, 0, width - 1);
            vals[idx++] = srow[sx];
          }
        }
        std::sort(vals, vals + 9);
        drow[x] = vals[4];
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(height, worker, sched.min_rows, sched.chunk_hint);
  return 0;
}

HALO_API int halo_erode3x3_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int use_mt
) {
  return morph3x3_generic(src, src_stride, dst, dst_stride, width, height, MorphOp::Min, use_mt);
}

HALO_API int halo_dilate3x3_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int use_mt
) {
  return morph3x3_generic(src, src_stride, dst, dst_stride, width, height, MorphOp::Max, use_mt);
}

HALO_API int halo_open3x3_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;

  std::vector<float> tmp((size_t)width * height, 0.0f);
  int rc = morph3x3_generic(src, src_stride, tmp.data(), (long long)width * 4, width, height, MorphOp::Min, use_mt);
  if (rc != 0) return rc;
  rc = morph3x3_generic(tmp.data(), (long long)width * 4, dst, dst_stride, width, height, MorphOp::Max, use_mt);
  return rc;
}

HALO_API int halo_close3x3_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;

  std::vector<float> tmp((size_t)width * height, 0.0f);
  int rc = morph3x3_generic(src, src_stride, tmp.data(), (long long)width * 4, width, height, MorphOp::Max, use_mt);
  if (rc != 0) return rc;
  rc = morph3x3_generic(tmp.data(), (long long)width * 4, dst, dst_stride, width, height, MorphOp::Min, use_mt);
  return rc;
}

HALO_API int halo_unsharp_mask_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  float sigma,
  float amount,
  float threshold,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
  if (sigma < 0.0f) return -4;
  if (threshold < 0.0f) threshold = 0.0f;

  std::vector<float> blurred((size_t)width * height, 0.0f);

  if (sigma == 0.0f) {
    for (int y = 0; y < height; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* brow = blurred.data() + (size_t)y * width;
      std::memcpy(brow, srow, sizeof(float) * width);
    }
  } else {
    int rc = halo_gaussian_blur_f32(
      src, src_stride,
      blurred.data(), (long long)width * 4,
      width, height,
      sigma,
      use_mt
    );
    if (rc != 0) return rc;
  }

  const int64_t row_work = (int64_t)width * 16;
  const int64_t total_bytes = row_work * height;

  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y = y0; y < y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      const float* brow = blurred.data() + (size_t)y * width;
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x = 0; x < width; ++x) {
        float detail = srow[x] - brow[x];
        if (std::fabs(detail) < threshold) detail = 0.0f;
        drow[x] = srow[x] + amount * detail;
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(height, worker, sched.min_rows, sched.chunk_hint);
  return 0;
}

HALO_API int halo_relu_clamp_axpby_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  float alpha, float beta,
  float clamp_min, float clamp_max,
  int apply_relu,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
  if (clamp_min > clamp_max) return -4;

  const int64_t row_work = (int64_t)width * 16;
  const int64_t total_bytes = row_work * height;
  int threads = std::max(1, g_threads);
  ensure_pool(threads);
  ScheduleDecision sched = compute_schedule(width, height, threads, row_work, total_bytes, use_mt != 0);

  auto worker = [&](int y0, int y1){
    for (int y=y0; y<y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x=0; x<width; ++x) {
        float val = alpha * drow[x] + beta * srow[x];
        if (apply_relu && val < 0.0f) val = 0.0f;
        if (val < clamp_min) val = clamp_min;
        if (val > clamp_max) val = clamp_max;
        drow[x] = val;
      }
    }
  };

  if (!sched.do_mt) {
    worker(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    worker,
    /*min_rows_per_task=*/sched.min_rows,
    /*chunk_rows_hint=*/sched.chunk_hint
  );

  return 0;
}

// -----------------------------------------------------
//  NEU: expliziter Pool-Shutdown (für Python atexit)
// -----------------------------------------------------
HALO_API void halo_shutdown_pool() {
  ThreadPool::instance().stop_all();
}

#endif // HALO_FASTPATH_INCLUDED
