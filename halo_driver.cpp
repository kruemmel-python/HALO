#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

// -----------------------------------------------------------------------------
//  CPU fast-path implementation
// -----------------------------------------------------------------------------
#include "fastpath.cpp"

// -----------------------------------------------------------------------------
//  GPU OpenCL driver implementation
// -----------------------------------------------------------------------------
#include "CipherCore_OpenCl.c"

namespace {

std::mutex g_gpu_mutex;
cl_program g_image_program = NULL;
cl_kernel g_box_blur_kernel = NULL;
cl_kernel g_gaussian_kernel = NULL;
cl_kernel g_sobel_kernel = NULL;
cl_kernel g_median_kernel = NULL;
cl_kernel g_invert_kernel = NULL;
cl_kernel g_gamma_kernel = NULL;
cl_kernel g_levels_kernel = NULL;
cl_kernel g_threshold_kernel = NULL;
cl_kernel g_unsharp_kernel = NULL;
int g_prepared_device = -1;

const char kImageKernelSource[] = R"CLC(
inline int clamp_int(int v, int lo, int hi) {
    return clamp(v, lo, hi);
}

inline float read_pixel(__global const float* src,
                        int stride,
                        int width,
                        int height,
                        int x,
                        int y) {
    int xx = clamp_int(x, 0, width - 1);
    int yy = clamp_int(y, 0, height - 1);
    return src[yy * stride + xx];
}

__kernel void box_blur_f32(__global const float* src,
                           __global float* dst,
                           int width,
                           int height,
                           int stride_in,
                           int stride_out,
                           int radius) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    int kernel_width = radius * 2 + 1;
    if (kernel_width <= 0) {
        dst[y * stride_out + x] = read_pixel(src, stride_in, width, height, x, y);
        return;
    }
    float sum = 0.0f;
    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            sum += read_pixel(src, stride_in, width, height, x + dx, y + dy);
            ++count;
        }
    }
    float value = count > 0 ? sum / (float)count : read_pixel(src, stride_in, width, height, x, y);
    dst[y * stride_out + x] = value;
}

__kernel void gaussian_blur_f32(__global const float* src,
                                __global float* dst,
                                int width,
                                int height,
                                int stride_in,
                                int stride_out,
                                float sigma,
                                int radius) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    if (sigma <= 0.0f || radius <= 0) {
        dst[y * stride_out + x] = read_pixel(src, stride_in, width, height, x, y);
        return;
    }
    float sum = 0.0f;
    float norm = 0.0f;
    float two_sigma_sq = 2.0f * sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float weight = exp(-((float)(dx * dx + dy * dy)) / two_sigma_sq);
            sum += weight * read_pixel(src, stride_in, width, height, x + dx, y + dy);
            norm += weight;
        }
    }
    dst[y * stride_out + x] = sum / fmax(norm, 1e-6f);
}

__kernel void sobel_f32(__global const float* src,
                        __global float* dst,
                        int width,
                        int height,
                        int stride_in,
                        int stride_out) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    float tl = read_pixel(src, stride_in, width, height, x - 1, y - 1);
    float tc = read_pixel(src, stride_in, width, height, x,     y - 1);
    float tr = read_pixel(src, stride_in, width, height, x + 1, y - 1);
    float ml = read_pixel(src, stride_in, width, height, x - 1, y);
    float mr = read_pixel(src, stride_in, width, height, x + 1, y);
    float bl = read_pixel(src, stride_in, width, height, x - 1, y + 1);
    float bc = read_pixel(src, stride_in, width, height, x,     y + 1);
    float br = read_pixel(src, stride_in, width, height, x + 1, y + 1);

    float gx = -tl - 2.0f * ml - bl + tr + 2.0f * mr + br;
    float gy =  tl + 2.0f * tc + tr - bl - 2.0f * bc - br;
    dst[y * stride_out + x] = sqrt(gx * gx + gy * gy);
}

__kernel void median3x3_f32(__global const float* src,
                            __global float* dst,
                            int width,
                            int height,
                            int stride_in,
                            int stride_out) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    float window[9];
    int idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            window[idx++] = read_pixel(src, stride_in, width, height, x + dx, y + dy);
        }
    }
    for (int i = 1; i < 9; ++i) {
        float key = window[i];
        int j = i - 1;
        while (j >= 0 && window[j] > key) {
            window[j + 1] = window[j];
            --j;
        }
        window[j + 1] = key;
    }
    dst[y * stride_out + x] = window[4];
}

__kernel void invert_f32_gpu(__global const float* src,
                             __global float* dst,
                             int width,
                             int height,
                             int stride_in,
                             int stride_out,
                             float min_val,
                             float max_val,
                             int use_range) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    float lo = use_range ? min_val : 0.0f;
    float hi = use_range ? max_val : 1.0f;
    float value = read_pixel(src, stride_in, width, height, x, y);
    float inv = lo + hi - value;
    dst[y * stride_out + x] = clamp(inv, lo, hi);
}

__kernel void gamma_f32_gpu(__global const float* src,
                            __global float* dst,
                            int width,
                            int height,
                            int stride_in,
                            int stride_out,
                            float gamma,
                            float gain) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    float value = read_pixel(src, stride_in, width, height, x, y);
    value = fmax(value, 0.0f);
    float mapped = (gamma == 1.0f) ? value : pow(value, gamma);
    dst[y * stride_out + x] = gain * mapped;
}

__kernel void levels_f32_gpu(__global const float* src,
                             __global float* dst,
                             int width,
                             int height,
                             int stride_in,
                             int stride_out,
                             float in_low,
                             float in_high,
                             float out_low,
                             float out_high,
                             float gamma) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    float inv_in_range = 1.0f / fmax(in_high - in_low, 1e-6f);
    float out_range = out_high - out_low;
    float value = (read_pixel(src, stride_in, width, height, x, y) - in_low) * inv_in_range;
    value = clamp(value, 0.0f, 1.0f);
    if (gamma != 1.0f) {
        value = pow(value, gamma);
    }
    dst[y * stride_out + x] = out_low + value * out_range;
}

__kernel void threshold_f32_gpu(__global const float* src,
                                __global float* dst,
                                int width,
                                int height,
                                int stride_in,
                                int stride_out,
                                float low,
                                float high,
                                float low_value,
                                float high_value) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    float value = read_pixel(src, stride_in, width, height, x, y);
    if (!(high > low)) {
        dst[y * stride_out + x] = (value >= low) ? high_value : low_value;
        return;
    }
    if (value <= low) {
        dst[y * stride_out + x] = low_value;
    } else if (value >= high) {
        dst[y * stride_out + x] = high_value;
    } else {
        float t = (value - low) / (high - low);
        dst[y * stride_out + x] = low_value + t * (high_value - low_value);
    }
}

__kernel void unsharp_mask_f32_gpu(__global const float* src,
                                   __global float* dst,
                                   int width,
                                   int height,
                                   int stride_in,
                                   int stride_out,
                                   float sigma,
                                   float amount,
                                   float threshold,
                                   int radius) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    float original = read_pixel(src, stride_in, width, height, x, y);
    float blurred = original;
    if (sigma > 0.0f && radius > 0) {
        float sum = 0.0f;
        float norm = 0.0f;
        float two_sigma_sq = 2.0f * sigma * sigma;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                float weight = exp(-((float)(dx * dx + dy * dy)) / two_sigma_sq);
                sum += weight * read_pixel(src, stride_in, width, height, x + dx, y + dy);
                norm += weight;
            }
        }
        blurred = sum / fmax(norm, 1e-6f);
    }
    float detail = original - blurred;
    if (fabs(detail) < threshold) {
        detail = 0.0f;
    }
    dst[y * stride_out + x] = original + amount * detail;
}
)CLC";

void release_image_program_locked() {
    if (g_box_blur_kernel) { clReleaseKernel(g_box_blur_kernel); g_box_blur_kernel = NULL; }
    if (g_gaussian_kernel) { clReleaseKernel(g_gaussian_kernel); g_gaussian_kernel = NULL; }
    if (g_sobel_kernel) { clReleaseKernel(g_sobel_kernel); g_sobel_kernel = NULL; }
    if (g_median_kernel) { clReleaseKernel(g_median_kernel); g_median_kernel = NULL; }
    if (g_invert_kernel) { clReleaseKernel(g_invert_kernel); g_invert_kernel = NULL; }
    if (g_gamma_kernel) { clReleaseKernel(g_gamma_kernel); g_gamma_kernel = NULL; }
    if (g_levels_kernel) { clReleaseKernel(g_levels_kernel); g_levels_kernel = NULL; }
    if (g_threshold_kernel) { clReleaseKernel(g_threshold_kernel); g_threshold_kernel = NULL; }
    if (g_unsharp_kernel) { clReleaseKernel(g_unsharp_kernel); g_unsharp_kernel = NULL; }
    if (g_image_program) { clReleaseProgram(g_image_program); g_image_program = NULL; }
}

cl_int build_image_program_locked() {
    if (!context || !device_id) {
        return CL_INVALID_CONTEXT;
    }
    if (g_image_program) {
        return CL_SUCCESS;
    }
    size_t source_length = std::strlen(kImageKernelSource);
    const char* source_ptr = kImageKernelSource;
    cl_int err = CL_SUCCESS;
    g_image_program = clCreateProgramWithSource(context, 1, &source_ptr, &source_length, &err);
    if (err != CL_SUCCESS) {
        return err;
    }
    err = clBuildProgram(g_image_program, 1, &device_id, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(g_image_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 1) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(g_image_program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
            std::fprintf(stderr, "[HALO][GPU] Kernel build failed: %s\n", log.data());
        }
        release_image_program_locked();
        return err;
    }
    g_box_blur_kernel = clCreateKernel(g_image_program, "box_blur_f32", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_gaussian_kernel = clCreateKernel(g_image_program, "gaussian_blur_f32", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_sobel_kernel = clCreateKernel(g_image_program, "sobel_f32", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_median_kernel = clCreateKernel(g_image_program, "median3x3_f32", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_invert_kernel = clCreateKernel(g_image_program, "invert_f32_gpu", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_gamma_kernel = clCreateKernel(g_image_program, "gamma_f32_gpu", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_levels_kernel = clCreateKernel(g_image_program, "levels_f32_gpu", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_threshold_kernel = clCreateKernel(g_image_program, "threshold_f32_gpu", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    g_unsharp_kernel = clCreateKernel(g_image_program, "unsharp_mask_f32_gpu", &err);
    if (err != CL_SUCCESS) { release_image_program_locked(); return err; }
    return CL_SUCCESS;
}

int ensure_image_program_locked() {
    if (!context || !queue) {
        return -100;
    }
    cl_int err = build_image_program_locked();
    return (err == CL_SUCCESS) ? 0 : err;
}

template <typename ExtraSetter>
cl_int dispatch_image_kernel_locked(cl_kernel kernel,
                                    const float* src,
                                    long long src_stride_bytes,
                                    float* dst,
                                    long long dst_stride_bytes,
                                    int width,
                                    int height,
                                    ExtraSetter&& set_extra_args) {
    if (!kernel || !context || !queue) {
        return CL_INVALID_OPERATION;
    }
    size_t src_pitch = static_cast<size_t>(src_stride_bytes);
    size_t dst_pitch = static_cast<size_t>(dst_stride_bytes);
    size_t src_bytes = src_pitch * static_cast<size_t>(height);
    size_t dst_bytes = dst_pitch * static_cast<size_t>(height);

    cl_int err = CL_SUCCESS;
    cl_mem src_mem = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    src_bytes,
                                    const_cast<float*>(src),
                                    &err);
    if (err != CL_SUCCESS) {
        if (src_mem) { clReleaseMemObject(src_mem); }
        return err;
    }
    cl_mem dst_mem = clCreateBuffer(context,
                                    CL_MEM_WRITE_ONLY,
                                    dst_bytes,
                                    NULL,
                                    &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(src_mem);
        return err;
    }

    int width_i = width;
    int height_i = height;
    int src_stride_elems = static_cast<int>(src_stride_bytes / sizeof(float));
    int dst_stride_elems = static_cast<int>(dst_stride_bytes / sizeof(float));
    int arg_index = 0;
    err  = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src_mem);
    err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_mem);
    err |= clSetKernelArg(kernel, arg_index++, sizeof(int), &width_i);
    err |= clSetKernelArg(kernel, arg_index++, sizeof(int), &height_i);
    err |= clSetKernelArg(kernel, arg_index++, sizeof(int), &src_stride_elems);
    err |= clSetKernelArg(kernel, arg_index++, sizeof(int), &dst_stride_elems);
    if (err == CL_SUCCESS) {
        err = set_extra_args(kernel, arg_index);
    }
    if (err == CL_SUCCESS) {
        const size_t global[2] = {
            static_cast<size_t>(width),
            static_cast<size_t>(height)
        };
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    }
    if (err == CL_SUCCESS) {
        err = clFinish(queue);
    }
    if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, dst_mem, CL_TRUE, 0, dst_bytes, dst, 0, NULL, NULL);
    }

    clReleaseMemObject(src_mem);
    clReleaseMemObject(dst_mem);
    return err;
}

}  // namespace

#if defined(_WIN32)
#define HALO_COMBINED_EXPORT extern "C" __declspec(dllexport)
#else
#define HALO_COMBINED_EXPORT extern "C" __attribute__((visibility("default")))
#endif

HALO_COMBINED_EXPORT int halo_gpu_prepare(int device_index) {
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    if (g_prepared_device != device_index || !context || !queue) {
        release_image_program_locked();
        int rc = initialize_gpu(device_index);
        if (rc != 0) {
            g_prepared_device = -1;
            return rc;
        }
        g_prepared_device = device_index;
    }
    return ensure_image_program_locked();
}

HALO_COMBINED_EXPORT void halo_gpu_release(int device_index) {
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    release_image_program_locked();
    shutdown_gpu(device_index);
    if (g_prepared_device == device_index) {
        g_prepared_device = -1;
    }
}

HALO_COMBINED_EXPORT int halo_gpu_box_blur_f32(const float* src,
                                               long long src_stride,
                                               float* dst,
                                               long long dst_stride,
                                               int width,
                                               int height,
                                               int radius,
                                               int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    if (radius < 0) return -4;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    auto setter = [radius](cl_kernel kernel, int arg_index) -> cl_int {
        int r = radius;
        return clSetKernelArg(kernel, arg_index, sizeof(int), &r);
    };
    cl_int err = dispatch_image_kernel_locked(g_box_blur_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_gaussian_blur_f32(const float* src,
                                                    long long src_stride,
                                                    float* dst,
                                                    long long dst_stride,
                                                    int width,
                                                    int height,
                                                    float sigma,
                                                    int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    if (sigma < 0.0f) return -4;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    int radius = (sigma <= 0.0f) ? 0 : std::max(1, (int)std::ceil(sigma * 3.0f));
    auto setter = [sigma, radius](cl_kernel kernel, int arg_index) -> cl_int {
        float s = sigma;
        int r = radius;
        cl_int err = clSetKernelArg(kernel, arg_index, sizeof(float), &s);
        if (err != CL_SUCCESS) return err;
        return clSetKernelArg(kernel, arg_index + 1, sizeof(int), &r);
    };
    cl_int err = dispatch_image_kernel_locked(g_gaussian_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_sobel_f32(const float* src,
                                            long long src_stride,
                                            float* dst,
                                            long long dst_stride,
                                            int width,
                                            int height,
                                            int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    auto setter = [](cl_kernel, int) -> cl_int { return CL_SUCCESS; };
    cl_int err = dispatch_image_kernel_locked(g_sobel_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_median3x3_f32(const float* src,
                                                long long src_stride,
                                                float* dst,
                                                long long dst_stride,
                                                int width,
                                                int height,
                                                int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    auto setter = [](cl_kernel, int) -> cl_int { return CL_SUCCESS; };
    cl_int err = dispatch_image_kernel_locked(g_median_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_invert_f32(const float* src,
                                             long long src_stride,
                                             float* dst,
                                             long long dst_stride,
                                             int width,
                                             int height,
                                             float min_val,
                                             float max_val,
                                             int use_range,
                                             int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    float lo = use_range ? min_val : 0.0f;
    float hi = use_range ? max_val : 1.0f;
    if (!(hi > lo)) return -4;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    auto setter = [min_val, max_val, use_range](cl_kernel kernel, int arg_index) -> cl_int {
        float lo = min_val;
        float hi = max_val;
        int range = use_range;
        cl_int err = clSetKernelArg(kernel, arg_index, sizeof(float), &lo);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 1, sizeof(float), &hi);
        if (err != CL_SUCCESS) return err;
        return clSetKernelArg(kernel, arg_index + 2, sizeof(int), &range);
    };
    cl_int err = dispatch_image_kernel_locked(g_invert_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_gamma_f32(const float* src,
                                            long long src_stride,
                                            float* dst,
                                            long long dst_stride,
                                            int width,
                                            int height,
                                            float gamma,
                                            float gain,
                                            int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    if (!(gamma > 0.0f)) return -4;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    auto setter = [gamma, gain](cl_kernel kernel, int arg_index) -> cl_int {
        float g = gamma;
        float ga = gain;
        cl_int err = clSetKernelArg(kernel, arg_index, sizeof(float), &g);
        if (err != CL_SUCCESS) return err;
        return clSetKernelArg(kernel, arg_index + 1, sizeof(float), &ga);
    };
    cl_int err = dispatch_image_kernel_locked(g_gamma_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_levels_f32(const float* src,
                                             long long src_stride,
                                             float* dst,
                                             long long dst_stride,
                                             int width,
                                             int height,
                                             float in_low,
                                             float in_high,
                                             float out_low,
                                             float out_high,
                                             float gamma,
                                             int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    if (!(in_high > in_low)) return -4;
    if (!(gamma > 0.0f)) gamma = 1.0f;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    auto setter = [in_low, in_high, out_low, out_high, gamma](cl_kernel kernel, int arg_index) -> cl_int {
        float il = in_low;
        float ih = in_high;
        float ol = out_low;
        float oh = out_high;
        float g = gamma;
        cl_int err = clSetKernelArg(kernel, arg_index, sizeof(float), &il);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 1, sizeof(float), &ih);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 2, sizeof(float), &ol);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 3, sizeof(float), &oh);
        if (err != CL_SUCCESS) return err;
        return clSetKernelArg(kernel, arg_index + 4, sizeof(float), &g);
    };
    cl_int err = dispatch_image_kernel_locked(g_levels_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_threshold_f32(const float* src,
                                                long long src_stride,
                                                float* dst,
                                                long long dst_stride,
                                                int width,
                                                int height,
                                                float low,
                                                float high,
                                                float low_value,
                                                float high_value,
                                                int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    auto setter = [low, high, low_value, high_value](cl_kernel kernel, int arg_index) -> cl_int {
        float lo = low;
        float hi = high;
        float lv = low_value;
        float hv = high_value;
        cl_int err = clSetKernelArg(kernel, arg_index, sizeof(float), &lo);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 1, sizeof(float), &hi);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 2, sizeof(float), &lv);
        if (err != CL_SUCCESS) return err;
        return clSetKernelArg(kernel, arg_index + 3, sizeof(float), &hv);
    };
    cl_int err = dispatch_image_kernel_locked(g_threshold_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

HALO_COMBINED_EXPORT int halo_gpu_unsharp_mask_f32(const float* src,
                                                   long long src_stride,
                                                   float* dst,
                                                   long long dst_stride,
                                                   int width,
                                                   int height,
                                                   float sigma,
                                                   float amount,
                                                   float threshold,
                                                   int /*use_mt*/) {
    if (!src || !dst) return -1;
    if (width <= 0 || height <= 0) return -2;
    if (src_stride < (long long)width * 4 || dst_stride < (long long)width * 4) return -3;
    if (sigma < 0.0f) return -4;
    if (threshold < 0.0f) threshold = 0.0f;
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    int rc = ensure_image_program_locked();
    if (rc != 0) return rc;
    int radius = (sigma <= 0.0f) ? 0 : std::max(1, (int)std::ceil(sigma * 3.0f));
    auto setter = [sigma, amount, threshold, radius](cl_kernel kernel, int arg_index) -> cl_int {
        float s = sigma;
        float a = amount;
        float t = threshold;
        int r = radius;
        cl_int err = clSetKernelArg(kernel, arg_index, sizeof(float), &s);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 1, sizeof(float), &a);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, arg_index + 2, sizeof(float), &t);
        if (err != CL_SUCCESS) return err;
        return clSetKernelArg(kernel, arg_index + 3, sizeof(int), &r);
    };
    cl_int err = dispatch_image_kernel_locked(g_unsharp_kernel, src, src_stride, dst, dst_stride, width, height, setter);
    return (err == CL_SUCCESS) ? 0 : err;
}

