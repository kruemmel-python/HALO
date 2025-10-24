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

  void parallel_for_rows(int rows,
                         const std::function<void(int,int)>& fn,
                         int min_rows_per_task=1,
                         int64_t work_per_row=0) {
    if (rows <= 0) return;

    // Falls nur 1 Thread oder wenig Arbeit: direkt ausführen
    int t;
    {
      std::lock_guard<std::mutex> lk(mx_);
      t = std::max(1, (int)workers_.size());
    }
    int max_tasks = t;
    int64_t eff_min_rows = std::max(1, min_rows_per_task);
    if (work_per_row > 0) {
      const int64_t row_bytes = std::max<int64_t>(1, work_per_row);
      const int64_t target_bytes = 256 * 1024; // ~256 KiB je Task
      int64_t min_rows_from_bytes = (target_bytes + row_bytes - 1) / row_bytes;
      eff_min_rows = std::max<int64_t>(eff_min_rows, min_rows_from_bytes);
    }

    if (rows < max_tasks * eff_min_rows) {
      fn(0, rows);
      return;
    }

    // Latch für Fertigstellung
    struct Latch {
      std::mutex m;
      std::condition_variable cv;
      int count;
    };
    int64_t chunk_rows = rows / (max_tasks * 2);
    if (chunk_rows <= 0) chunk_rows = rows / max_tasks;
    if (chunk_rows <= 0) chunk_rows = rows;
    chunk_rows = std::max<int64_t>(chunk_rows, eff_min_rows);

    if (work_per_row > 0) {
      const int64_t row_bytes = std::max<int64_t>(1, work_per_row);
      const int64_t target_bytes = 384 * 1024; // etwas größer bei hohem Durchsatz
      int64_t rows_from_bytes = (target_bytes + row_bytes - 1) / row_bytes;
      chunk_rows = std::max<int64_t>(chunk_rows, rows_from_bytes);
    }

    int tasks = (int)((rows + chunk_rows - 1) / chunk_rows);
    tasks = std::min(tasks, max_tasks * 4);
    if (tasks <= 1) {
      fn(0, rows);
      return;
    }

    auto latch = std::make_shared<Latch>();
    latch->count = tasks;

    int y = 0;
    for (int i=0; i<tasks; ++i) {
      int y0 = y;
      int y1 = std::min(rows, y0 + (int)chunk_rows);
      if (y0 >= y1) {
        std::unique_lock<std::mutex> lk2(latch->m);
        if (--latch->count == 0) latch->cv.notify_one();
        continue;
      }
      y = y1;
      enqueue([=]() {
        fn(y0, y1);
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

static void img_u16_to_f32_axpby_scalar(
  const uint16_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int height,
  int bits,
  float scale, float offset,
  float alpha, float beta
) {
  const int shift = std::max(0, 16 - bits);
  for (int y=0; y<height; ++y) {
    const uint16_t* srow = (const uint16_t*)((const uint8_t*)src + y * src_stride);
    float*          drow = (float*)((uint8_t*)dst + y * dst_stride);
    for (int x=0; x<width; ++x) {
      float tmp = float(srow[x] >> shift) * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }
  }
}

static void img_rgb_interleaved_to_f32_scalar(
  const uint8_t* src, int64_t src_stride,
  float* dst_r, int64_t dst_stride_r,
  float* dst_g, int64_t dst_stride_g,
  float* dst_b, int64_t dst_stride_b,
  int width, int height,
  float scale, float offset,
  float alpha, float beta
) {
  for (int y=0; y<height; ++y) {
    const uint8_t* srow = src + y * src_stride;
    float* drow_r = (float*)((uint8_t*)dst_r + y * dst_stride_r);
    float* drow_g = (float*)((uint8_t*)dst_g + y * dst_stride_g);
    float* drow_b = (float*)((uint8_t*)dst_b + y * dst_stride_b);
    for (int x=0; x<width; ++x) {
      const int idx = x * 3;
      float rv = float(srow[idx]);
      float gv = float(srow[idx + 1]);
      float bv = float(srow[idx + 2]);
      rv = rv * scale + offset;
      gv = gv * scale + offset;
      bv = bv * scale + offset;
      drow_r[x] = alpha * drow_r[x] + beta * rv;
      drow_g[x] = alpha * drow_g[x] + beta * gv;
      drow_b[x] = alpha * drow_b[x] + beta * bv;
    }
  }
}

static void img_rgb_planar_to_f32_scalar(
  const uint8_t* src_r, int64_t src_stride_r,
  const uint8_t* src_g, int64_t src_stride_g,
  const uint8_t* src_b, int64_t src_stride_b,
  float* dst_r, int64_t dst_stride_r,
  float* dst_g, int64_t dst_stride_g,
  float* dst_b, int64_t dst_stride_b,
  int width, int height,
  float scale, float offset,
  float alpha, float beta
) {
  for (int y=0; y<height; ++y) {
    const uint8_t* srow_r = src_r + y * src_stride_r;
    const uint8_t* srow_g = src_g + y * src_stride_g;
    const uint8_t* srow_b = src_b + y * src_stride_b;
    float* drow_r = (float*)((uint8_t*)dst_r + y * dst_stride_r);
    float* drow_g = (float*)((uint8_t*)dst_g + y * dst_stride_g);
    float* drow_b = (float*)((uint8_t*)dst_b + y * dst_stride_b);
    for (int x=0; x<width; ++x) {
      float rv = float(srow_r[x]) * scale + offset;
      float gv = float(srow_g[x]) * scale + offset;
      float bv = float(srow_b[x]) * scale + offset;
      drow_r[x] = alpha * drow_r[x] + beta * rv;
      drow_g[x] = alpha * drow_g[x] + beta * gv;
      drow_b[x] = alpha * drow_b[x] + beta * bv;
    }
  }
}

static void convolve_horizontal_rows(
  const float* src, int64_t src_stride,
  float* tmp, int width,
  int y0, int y1,
  const std::vector<float>& kernel,
  int radius
) {
  for (int y=y0; y<y1; ++y) {
    const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
    float* trow = tmp + (int64_t)y * width;
    for (int x=0; x<width; ++x) {
      float acc = 0.f;
      for (int k=-radius; k<=radius; ++k) {
        int xx = x + k;
        if (xx < 0) xx = 0;
        if (xx >= width) xx = width - 1;
        acc += kernel[k + radius] * srow[xx];
      }
      trow[x] = acc;
    }
  }
}

static void convolve_vertical_rows(
  const float* tmp, float* dst, int64_t dst_stride,
  int width, int height,
  int y0, int y1,
  const std::vector<float>& kernel,
  int radius
) {
  for (int y=y0; y<y1; ++y) {
    float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
    for (int x=0; x<width; ++x) {
      float acc = 0.f;
      for (int k=-radius; k<=radius; ++k) {
        int yy = y + k;
        if (yy < 0) yy = 0;
        if (yy >= height) yy = height - 1;
        acc += kernel[k + radius] * tmp[(int64_t)yy * width + x];
      }
      drow[x] = acc;
    }
  }
}

static std::vector<float> make_box_kernel(int radius) {
  std::vector<float> kernel((size_t)radius * 2 + 1, 1.0f);
  float norm = 1.0f / (float)kernel.size();
  for (float& v : kernel) v *= norm;
  return kernel;
}

static std::vector<float> make_gauss_kernel(int radius, float sigma) {
  std::vector<float> kernel((size_t)radius * 2 + 1, 0.f);
  const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma + 1e-12f);
  float sum = 0.f;
  for (int i=-radius; i<=radius; ++i) {
    float w = std::exp(-(float)(i*i) * inv_two_sigma2);
    kernel[i + radius] = w;
    sum += w;
  }
  if (sum <= 0.f) sum = 1.f;
  float inv = 1.0f / sum;
  for (float& v : kernel) v *= inv;
  return kernel;
}

static inline float cubic_weight(float x) {
  x = std::fabs(x);
  const float a = -0.5f;
  if (x <= 1.0f) {
    return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
  } else if (x < 2.0f) {
    return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
  } else {
    return 0.0f;
  }
}

static float sample_bilinear(
  const float* src, int64_t src_stride,
  int width, int height,
  float fx, float fy
) {
  float x = std::floor(fx);
  float y = std::floor(fy);
  int x0 = (int)x;
  int y0 = (int)y;
  float dx = fx - x;
  float dy = fy - y;
  int x1 = std::min(width - 1, x0 + 1);
  int y1 = std::min(height - 1, y0 + 1);
  x0 = std::max(0, x0);
  y0 = std::max(0, y0);

  const float* row0 = (const float*)((const uint8_t*)src + (int64_t)y0 * src_stride);
  const float* row1 = (const float*)((const uint8_t*)src + (int64_t)y1 * src_stride);

  float p00 = row0[x0];
  float p10 = row0[x1];
  float p01 = row1[x0];
  float p11 = row1[x1];

  float a = p00 * (1.f - dx) + p10 * dx;
  float b = p01 * (1.f - dx) + p11 * dx;
  return a * (1.f - dy) + b * dy;
}

static float sample_bicubic(
  const float* src, int64_t src_stride,
  int width, int height,
  float fx, float fy
) {
  int ix = (int)std::floor(fx);
  int iy = (int)std::floor(fy);
  float dx = fx - (float)ix;
  float dy = fy - (float)iy;
  float sum = 0.f;
  float norm = 0.f;
  for (int m=-1; m<=2; ++m) {
    int yy = std::clamp(iy + m, 0, height - 1);
    float wy = cubic_weight((float)m - dy);
    const float* row = (const float*)((const uint8_t*)src + (int64_t)yy * src_stride);
    for (int n=-1; n<=2; ++n) {
      int xx = std::clamp(ix + n, 0, width - 1);
      float wx = cubic_weight((float)n - dx);
      float w = wy * wx;
      norm += w;
      sum  += w * row[xx];
    }
  }
  if (norm == 0.f) return sum;
  return sum / norm;
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
      __m256 vdst = allow_aligned ? _mm256_load_ps(drow + x) : _mm256_loadu_ps(drow + x);
      __m256 vout = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));

      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
      } else {
        if (allow_aligned) _mm256_store_ps(drow + x, vout);
        else                _mm256_storeu_ps(drow + x, vout);
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
      __m256 vdst = allow_aligned ? _mm256_load_ps(drow + x) : _mm256_loadu_ps(drow + x);
      __m256 vout = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));

      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
      } else {
        if (allow_aligned) _mm256_store_ps(drow + x, vout);
        else                _mm256_storeu_ps(drow + x, vout);
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
  int bits,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream,
  bool allow_aligned
) {
  const __m256 vscale = _mm256_set1_ps(scale);
  const __m256 voffs  = _mm256_set1_ps(offset);
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta  = _mm256_set1_ps(beta);
  const int shift = std::max(0, 16 - bits);
  const __m256i vshift = _mm256_set1_epi32(shift);

  bool any_stream = false;

  for (int y=0; y<height; ++y) {
    const uint16_t* srow = (const uint16_t*)((const uint8_t*)src + y * src_stride);
    float*          drow = (float*)((uint8_t*)dst + y * dst_stride);

    int x = 0;
    for (; x < width && ((reinterpret_cast<uintptr_t>(drow + x) & 31ull) != 0ull); ++x) {
      float tmp = float(srow[x] >> shift) * scale + offset;
      drow[x] = alpha * drow[x] + beta * tmp;
    }

    for (; x + 8 <= width; x += 8) {
      __m128i pix0 = _mm_loadu_si128((const __m128i*)(srow + x));
      __m256i pix = _mm256_cvtepu16_epi32(pix0);
      if (shift > 0) {
        pix = _mm256_srl_epi32(pix, vshift);
      }
      __m256 vf   = _mm256_cvtepi32_ps(pix);
      __m256 vtmp = _mm256_fmadd_ps(vf, vscale, voffs);
      __m256 vdst = allow_aligned ? _mm256_load_ps(drow + x) : _mm256_loadu_ps(drow + x);
      __m256 vout = _mm256_fmadd_ps(valpha, vdst, _mm256_mul_ps(vbeta, vtmp));
      if (allow_stream) {
        _mm256_stream_ps(drow + x, vout);
        any_stream = true;
      } else {
        if (allow_aligned) _mm256_store_ps(drow + x, vout);
        else                _mm256_storeu_ps(drow + x, vout);
      }
    }

    for (; x < width; ++x) {
      float tmp = float(srow[x] >> shift) * scale + offset;
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

static void img_kernel_rows_u16(
  const uint16_t* src, int64_t src_stride,
  float* dst, int64_t dst_stride,
  int width, int y0, int y1,
  int bits,
  float scale, float offset,
  float alpha, float beta,
  bool allow_stream,
  bool allow_aligned
) {
#if HALO_X86
  if (g_has_avx2) {
    bool used = img_u16_to_f32_axpby_avx2(
      (const uint16_t*)((const uint8_t*)src + (int64_t)y0 * src_stride), src_stride,
      (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride), dst_stride,
      width, (y1 - y0), bits, scale, offset, alpha, beta, allow_stream, allow_aligned
    );
    if (used) _mm_sfence();
    return;
  }
#endif
  img_u16_to_f32_axpby_scalar(
    (const uint16_t*)((const uint8_t*)src + (int64_t)y0 * src_stride), src_stride,
    (float*)(((uint8_t*)dst) + (int64_t)y0 * dst_stride), dst_stride,
    width, (y1 - y0), bits, scale, offset, alpha, beta
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

  const int64_t row_bytes_f32 = (int64_t)width * 4;
  const int64_t pixels  = (int64_t)width * height;
  const int64_t bytes   = pixels * 5; // ~1B read + 4B write
  const bool dst_al32   = is_aligned_32(dst);
  const bool stride_al32 = ((dst_stride & 31ll) == 0);
  const bool allow_aligned = dst_al32 && stride_al32;

  int64_t nt_byte_threshold = (dst_stride == row_bytes_f32) ? (8ll<<20) : (12ll<<20);
  if (height > g_threads * 4) nt_byte_threshold /= 2;
  if (nt_byte_threshold < (4ll<<20)) nt_byte_threshold = (4ll<<20);

  const bool width_ok = width >= (allow_aligned ? 1024 : 1536);
  bool allow_stream = g_streaming_enabled && allow_aligned && width_ok && bytes >= nt_byte_threshold;

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool big_enough = (bytes >= (8ll<<20)) && (height >= 2*t);
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
    /*min_rows_per_task=*/2,
    row_bytes_f32 + width
  );

  return 0;
}

HALO_API int halo_img_u16_to_f32_axpby(
  const unsigned short* src, long long src_stride,
  float* dst,             long long dst_stride,
  int width, int height,
  int bits_per_sample,
  float scale, float offset,
  float alpha, float beta,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (bits_per_sample <= 0 || bits_per_sample > 16) return -3;
  const int64_t row_bytes_src = (int64_t)width * 2;
  const int64_t row_bytes_dst = (int64_t)width * 4;
  if (src_stride < row_bytes_src || dst_stride < row_bytes_dst) return -4;

  const bool dst_al32 = is_aligned_32(dst);
  const bool stride_al32 = ((dst_stride & 31ll) == 0);
  const bool allow_aligned = dst_al32 && stride_al32;
  const int64_t pixels = (int64_t)width * height;
  const int64_t bytes = pixels * 6; // grobe Abschätzung
  int64_t nt_byte_threshold = (dst_stride == row_bytes_dst) ? (8ll<<20) : (12ll<<20);
  if (height > g_threads * 4) nt_byte_threshold /= 2;
  if (nt_byte_threshold < (4ll<<20)) nt_byte_threshold = (4ll<<20);
  bool allow_stream = g_streaming_enabled && allow_aligned && width >= 1024 && bytes >= nt_byte_threshold;

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool big_enough = (bytes >= (8ll<<20)) && (height >= 2*t);
  const bool do_mt = (use_mt != 0) && (t > 1) && big_enough;

  if (!do_mt) {
#if HALO_X86
    if (g_has_avx2) {
      bool used = img_u16_to_f32_axpby_avx2(
        (const uint16_t*)src, (int64_t)src_stride, dst, (int64_t)dst_stride,
        width, height, bits_per_sample, scale, offset, alpha, beta,
        allow_stream, allow_aligned
      );
      if (used) _mm_sfence();
      return 0;
    }
#endif
    img_u16_to_f32_axpby_scalar(
      (const uint16_t*)src, (int64_t)src_stride, dst, (int64_t)dst_stride,
      width, height, bits_per_sample, scale, offset, alpha, beta
    );
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    [&](int y0, int y1){
      img_kernel_rows_u16(
        (const uint16_t*)src, (int64_t)src_stride,
        dst, (int64_t)dst_stride,
        width, y0, y1,
        bits_per_sample, scale, offset, alpha, beta,
        allow_stream,
        allow_aligned
      );
    },
    /*min_rows_per_task=*/2,
    row_bytes_dst + row_bytes_src
  );

  return 0;
}

HALO_API int halo_img_rgb_u8_to_f32_interleaved(
  const unsigned char* src, long long src_stride,
  float* dst_r, long long dst_stride_r,
  float* dst_g, long long dst_stride_g,
  float* dst_b, long long dst_stride_b,
  int width, int height,
  float scale, float offset,
  float alpha, float beta,
  int use_mt
) {
  if (!src || !dst_r || !dst_g || !dst_b) return -1;
  if (width <= 0 || height <= 0) return -2;
  const int64_t src_row = (int64_t)width * 3;
  const int64_t dst_row = (int64_t)width * 4;
  if (src_stride < src_row || dst_stride_r < dst_row || dst_stride_g < dst_row || dst_stride_b < dst_row) return -3;

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const int64_t bytes = (int64_t)height * (src_row + 3 * dst_row);
  const bool do_mt = (use_mt != 0) && (t > 1) && bytes >= (6ll<<20) && height >= 2 * t;

  auto work = [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const uint8_t* srow = src + (int64_t)y * src_stride;
      float* drow_r = (float*)((uint8_t*)dst_r + (int64_t)y * dst_stride_r);
      float* drow_g = (float*)((uint8_t*)dst_g + (int64_t)y * dst_stride_g);
      float* drow_b = (float*)((uint8_t*)dst_b + (int64_t)y * dst_stride_b);
      for (int x=0; x<width; ++x) {
        int idx = x * 3;
        float rv = float(srow[idx]) * scale + offset;
        float gv = float(srow[idx+1]) * scale + offset;
        float bv = float(srow[idx+2]) * scale + offset;
        drow_r[x] = alpha * drow_r[x] + beta * rv;
        drow_g[x] = alpha * drow_g[x] + beta * gv;
        drow_b[x] = alpha * drow_b[x] + beta * bv;
      }
    }
  };

  if (!do_mt) {
    work(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    [&](int y0, int y1){ work(y0, y1); },
    /*min_rows_per_task=*/1,
    src_row + 3 * dst_row
  );
  return 0;
}

HALO_API int halo_img_rgb_u8_to_f32_planar(
  const unsigned char* src_r, long long src_stride_r,
  const unsigned char* src_g, long long src_stride_g,
  const unsigned char* src_b, long long src_stride_b,
  float* dst_r, long long dst_stride_r,
  float* dst_g, long long dst_stride_g,
  float* dst_b, long long dst_stride_b,
  int width, int height,
  float scale, float offset,
  float alpha, float beta,
  int use_mt
) {
  if (!src_r || !src_g || !src_b || !dst_r || !dst_g || !dst_b) return -1;
  if (width <= 0 || height <= 0) return -2;
  const int64_t src_row = (int64_t)width;
  const int64_t dst_row = (int64_t)width * 4;
  if (src_stride_r < src_row || src_stride_g < src_row || src_stride_b < src_row) return -3;
  if (dst_stride_r < dst_row || dst_stride_g < dst_row || dst_stride_b < dst_row) return -4;

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const int64_t bytes = (int64_t)height * (3 * src_row + 3 * dst_row);
  const bool do_mt = (use_mt != 0) && (t > 1) && bytes >= (6ll<<20) && height >= 2 * t;

  auto work = [&](int y0, int y1) {
    for (int y=y0; y<y1; ++y) {
      const uint8_t* srow_r = src_r + (int64_t)y * src_stride_r;
      const uint8_t* srow_g = src_g + (int64_t)y * src_stride_g;
      const uint8_t* srow_b = src_b + (int64_t)y * src_stride_b;
      float* drow_r = (float*)((uint8_t*)dst_r + (int64_t)y * dst_stride_r);
      float* drow_g = (float*)((uint8_t*)dst_g + (int64_t)y * dst_stride_g);
      float* drow_b = (float*)((uint8_t*)dst_b + (int64_t)y * dst_stride_b);
      for (int x=0; x<width; ++x) {
        float rv = float(srow_r[x]) * scale + offset;
        float gv = float(srow_g[x]) * scale + offset;
        float bv = float(srow_b[x]) * scale + offset;
        drow_r[x] = alpha * drow_r[x] + beta * rv;
        drow_g[x] = alpha * drow_g[x] + beta * gv;
        drow_b[x] = alpha * drow_b[x] + beta * bv;
      }
    }
  };

  if (!do_mt) {
    work(0, height);
    return 0;
  }

  ThreadPool::instance().parallel_for_rows(
    height,
    [&](int y0, int y1){ work(y0, y1); },
    /*min_rows_per_task=*/1,
    3 * (src_row + dst_row)
  );
  return 0;
}

HALO_API int halo_img_box_blur_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int radius,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (radius < 0) return -3;
  const int64_t row_bytes = (int64_t)width * 4;
  if (src_stride < row_bytes || dst_stride < row_bytes) return -4;
  if (radius == 0) {
    for (int y=0; y<height; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      std::memcpy(drow, srow, sizeof(float) * width);
    }
    return 0;
  }

  std::vector<float> kernel = make_box_kernel(radius);
  std::vector<float> tmp((size_t)width * height, 0.f);

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool do_mt = (use_mt != 0) && (t > 1) && ((int64_t)width * height >= 65536);

  auto horiz = [&](int y0, int y1){
    convolve_horizontal_rows(src, src_stride, tmp.data(), width, y0, y1, kernel, radius);
  };
  auto vert = [&](int y0, int y1){
    convolve_vertical_rows(tmp.data(), dst, dst_stride, width, height, y0, y1, kernel, radius);
  };

  if (do_mt) {
    ThreadPool::instance().parallel_for_rows(height, horiz, /*min_rows_per_task=*/1, row_bytes);
    ThreadPool::instance().parallel_for_rows(height, vert, /*min_rows_per_task=*/1, row_bytes);
  } else {
    horiz(0, height);
    vert(0, height);
  }

  return 0;
}

HALO_API int halo_img_gauss_blur_f32(
  const float* src, long long src_stride,
  float* dst,       long long dst_stride,
  int width, int height,
  int radius,
  float sigma,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  if (radius < 0 || sigma <= 0.f) return -3;
  const int64_t row_bytes = (int64_t)width * 4;
  if (src_stride < row_bytes || dst_stride < row_bytes) return -4;

  std::vector<float> kernel = make_gauss_kernel(radius, sigma);
  std::vector<float> tmp((size_t)width * height, 0.f);

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool do_mt = (use_mt != 0) && (t > 1) && ((int64_t)width * height >= 65536);

  auto horiz = [&](int y0, int y1){
    convolve_horizontal_rows(src, src_stride, tmp.data(), width, y0, y1, kernel, radius);
  };
  auto vert = [&](int y0, int y1){
    convolve_vertical_rows(tmp.data(), dst, dst_stride, width, height, y0, y1, kernel, radius);
  };

  if (do_mt) {
    ThreadPool::instance().parallel_for_rows(height, horiz, /*min_rows_per_task=*/1, row_bytes);
    ThreadPool::instance().parallel_for_rows(height, vert, /*min_rows_per_task=*/1, row_bytes);
  } else {
    horiz(0, height);
    vert(0, height);
  }

  return 0;
}

HALO_API int halo_img_sobel_f32(
  const float* src, long long src_stride,
  float* dst_gx, long long dst_stride_gx,
  float* dst_gy, long long dst_stride_gy,
  int width, int height,
  int use_mt
) {
  if (!src || !dst_gx || !dst_gy) return -1;
  if (width <= 1 || height <= 1) return -2;
  const int64_t row_bytes = (int64_t)width * 4;
  if (src_stride < row_bytes || dst_stride_gx < row_bytes || dst_stride_gy < row_bytes) return -3;

  auto worker = [&](int y0, int y1) {
    for (int y=y0; y<y1; ++y) {
      int ym1 = std::max(0, y-1);
      int yp1 = std::min(height-1, y+1);
      const float* row_m1 = (const float*)((const uint8_t*)src + (int64_t)ym1 * src_stride);
      const float* row_0  = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      const float* row_p1 = (const float*)((const uint8_t*)src + (int64_t)yp1 * src_stride);
      float* gx_row = (float*)((uint8_t*)dst_gx + (int64_t)y * dst_stride_gx);
      float* gy_row = (float*)((uint8_t*)dst_gy + (int64_t)y * dst_stride_gy);
      for (int x=0; x<width; ++x) {
        int xm1 = std::max(0, x-1);
        int xp1 = std::min(width-1, x+1);
        float tl = row_m1[xm1];
        float tc = row_m1[x];
        float tr = row_m1[xp1];
        float cl = row_0[xm1];
        float cr = row_0[xp1];
        float bl = row_p1[xm1];
        float bc = row_p1[x];
        float br = row_p1[xp1];
        float gx = -tl - 2.f * cl - bl + tr + 2.f * cr + br;
        float gy = -tl - 2.f * tc - tr + bl + 2.f * bc + br;
        gx_row[x] = gx;
        gy_row[x] = gy;
      }
    }
  };

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool do_mt = (use_mt != 0) && (t > 1) && ((int64_t)width * height >= 32768);
  if (do_mt) {
    ThreadPool::instance().parallel_for_rows(height, worker, /*min_rows_per_task=*/1, row_bytes);
  } else {
    worker(0, height);
  }
  return 0;
}

HALO_API int halo_img_resize_f32(
  const float* src, long long src_stride,
  int src_width, int src_height,
  float* dst, long long dst_stride,
  int dst_width, int dst_height,
  int filter,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0) return -2;
  const int64_t src_row_bytes = (int64_t)src_width * 4;
  const int64_t dst_row_bytes = (int64_t)dst_width * 4;
  if (src_stride < src_row_bytes || dst_stride < dst_row_bytes) return -3;

  double scale_x = (double)src_width / (double)dst_width;
  double scale_y = (double)src_height / (double)dst_height;

  auto worker = [&](int y0, int y1) {
    for (int dy=y0; dy<y1; ++dy) {
      float* drow = (float*)((uint8_t*)dst + (int64_t)dy * dst_stride);
      double sy = (dy + 0.5) * scale_y - 0.5;
      for (int dx=0; dx<dst_width; ++dx) {
        double sx = (dx + 0.5) * scale_x - 0.5;
        float val = 0.f;
        if (filter == 1) {
          val = sample_bicubic(src, src_stride, src_width, src_height, (float)sx, (float)sy);
        } else {
          val = sample_bilinear(src, src_stride, src_width, src_height, (float)sx, (float)sy);
        }
        drow[dx] = val;
      }
    }
  };

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool do_mt = (use_mt != 0) && (t > 1) && ((int64_t)dst_width * dst_height >= 32768);
  if (do_mt) {
    ThreadPool::instance().parallel_for_rows(dst_height, worker, /*min_rows_per_task=*/1, dst_row_bytes);
  } else {
    worker(0, dst_height);
  }
  return 0;
}

HALO_API int halo_img_relu_clamp_axpby_f32(
  const float* src, long long src_stride,
  float* dst, long long dst_stride,
  int width, int height,
  float alpha, float beta,
  float clamp_min, float clamp_max,
  int apply_relu,
  int use_mt
) {
  if (!src || !dst) return -1;
  if (width <= 0 || height <= 0) return -2;
  const int64_t row_bytes = (int64_t)width * 4;
  if (src_stride < row_bytes || dst_stride < row_bytes) return -3;
  if (clamp_min > clamp_max) std::swap(clamp_min, clamp_max);

  auto worker = [&](int y0, int y1) {
    for (int y=y0; y<y1; ++y) {
      const float* srow = (const float*)((const uint8_t*)src + (int64_t)y * src_stride);
      float* drow = (float*)((uint8_t*)dst + (int64_t)y * dst_stride);
      for (int x=0; x<width; ++x) {
        float v = srow[x];
        if (apply_relu) v = std::max(v, 0.0f);
        if (v < clamp_min) v = clamp_min;
        if (v > clamp_max) v = clamp_max;
        drow[x] = alpha * drow[x] + beta * v;
      }
    }
  };

  int t = std::max(1, g_threads);
  ensure_pool(t);
  const bool do_mt = (use_mt != 0) && (t > 1) && ((int64_t)width * height >= 32768);
  if (do_mt) {
    ThreadPool::instance().parallel_for_rows(height, worker, /*min_rows_per_task=*/1, row_bytes);
  } else {
    worker(0, height);
  }
  return 0;
}

// -----------------------------------------------------
//  NEU: expliziter Pool-Shutdown (für Python atexit)
// -----------------------------------------------------
HALO_API void halo_shutdown_pool() {
  ThreadPool::instance().stop_all();
}
