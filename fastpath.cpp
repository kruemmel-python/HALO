// fastpath.cpp (v0.5b) — HALO: SAXPY/Reduce + 2D-Image-Kern (LUT + AXPBY)
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

  void parallel_for_rows(int rows, const std::function<void(int,int)>& fn, int min_rows_per_task=1) {
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

    // Latch für Fertigstellung
    struct Latch {
      std::mutex m;
      std::condition_variable cv;
      int count;
    };
    auto latch = std::make_shared<Latch>();
    latch->count = max_tasks;

    int chunk = rows / max_tasks;
    int y = 0;
    for (int i=0; i<max_tasks-1; ++i) {
      int y0 = y;
      int y1 = y0 + chunk;
      y = y1;
      enqueue([=]() {
        fn(y0, y1);
        std::unique_lock<std::mutex> lk2(latch->m);
        if (--latch->count == 0) latch->cv.notify_one();
      });
    }
    int y0 = y, y1 = rows;
    enqueue([=]() {
      fn(y0, y1);
      std::unique_lock<std::mutex> lk2(latch->m);
      if (--latch->count == 0) latch->cv.notify_one();
    });

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
  }, /*min_rows_per_task=*/1);
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

#if HALO_X86
// Gather-Pfad (beliebige LUT)
static bool img_u8_to_f32_lut_axpby_avx2_gather(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  const float* lut256,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream
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
      __m256 vdst = _mm256_loadu_ps(drow + x);
      __m256 vout = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));

      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
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
  bool allow_stream
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
      __m256 vdst = _mm256_loadu_ps(drow + x);
      __m256 vout = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));

      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
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
#endif // HALO_X86

// Worker: verarbeitet [y0,y1)
static void img_kernel_rows(
  const uint8_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int y0, int y1,
  const float* lut256,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream
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
        allow_stream
      );
    } else {
      used = img_u8_to_f32_lut_axpby_avx2_gather(
        src + (int64_t)y0 * src_stride, src_stride,
        (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride), dst_stride,
        width, (y1 - y0),
        lut256, scale, offset, alpha, beta,
        allow_stream
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

// =====================================================
//  C-API
// =====================================================
HALO_API int halo_init_features() { halo_cpuid_init(); return 0; }

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

  const int64_t pixels  = (int64_t)width * height;
  const int64_t bytes   = pixels * 5; // ~1B read + 4B write
  const bool dst_al32   = is_aligned_32(dst);
  bool allow_stream = g_streaming_enabled && dst_al32 && width >= 2048 && bytes >= (16ll<<20);

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool big_enough = (bytes >= (16ll<<20)) && (height >= 2*t);
  const bool do_mt = (use_mt != 0) && (t > 1) && big_enough;

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
          width, height, slope, inter, alpha, beta, allow_stream
        );
      } else {
        used = img_u8_to_f32_lut_axpby_avx2_gather(
          (const uint8_t*)src, (int64_t)src_stride, dst, (int64_t)dst_stride,
          width, height, lut256, scale, offset, alpha, beta, allow_stream
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
        allow_stream
      );
    },
    /*min_rows_per_task=*/2
  );

  return 0;
}

// -----------------------------------------------------
//  NEU: expliziter Pool-Shutdown (für Python atexit)
// -----------------------------------------------------
HALO_API void halo_shutdown_pool() {
  ThreadPool::instance().stop_all();
}
