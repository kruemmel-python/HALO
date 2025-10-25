/**
 * @file opencl_driver.c
 * @brief C implementation of an OpenCL driver providing GPU compute capabilities.
 *
 * This file contains the C interface for interacting with an OpenCL-capable GPU.
 * It includes functions for initialization, memory management, data transfer,
 * kernel compilation, and execution of various computational kernels commonly
 * used in deep learning (matrix multiplication, activations, normalization, etc.),
 * including specialized kernels for prototype-based models and spiking elements.
 *
 * The driver is designed to be compiled into a shared library (DLL/SO)
 * and called from a higher-level language (like Python).
 *
 * This version is adapted to be compatible with OpenCL 1.2, 2.x, and 3.x runtimes,
 * preferring modern API calls where available but maintaining compatibility.
 * It specifically handles the conditional compilation of kernels requiring atomics.
 * Includes loss shaping functionality based on a list of critical pairs.
 */

#define _CRT_SECURE_NO_WARNINGS /* For Visual Studio (if sprintf/sprintf_s is used) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h> /* For FLT_MAX, HUGE_VALF */
#include <stdint.h> // For uintptr_t

// Include OpenCL headers based on the operating system
#if defined(__APPLE__)
    #include <OpenCL/cl.h>
#elif defined(_WIN32) || defined(_WIN64)
    // Prüfe, ob die Header direkt erreichbar sind (wie bei deiner Struktur)
    #if __has_include("cl.h")
        #include "cl.h"
    #elif __has_include(<CL/cl.h>)
        #include <CL/cl.h>
    #else
        #error "OpenCL Header (cl.h) nicht gefunden! Bitte Pfad oder -I Option prüfen."
    #endif
#else
    #include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// --- Platform Specific Defines ---
#ifndef M_PI
/** @brief Definition of PI if not already defined. */
#define M_PI 3.14159265358979323846
#endif
/** @brief Constant 1/sqrt(2*pi), used in GELU backward calculation. */
#define M_1_SQRT2PI 0.39894228040143267794f

/**
 * @brief Defines the floating-point type used within the OpenCL kernels (e.g., float, half).
 * Affects kernel compilation options and buffer sizes.
 */
#define KERNEL_FP_TYPE float
/** @brief String representation of KERNEL_FP_TYPE, used in kernel build options. */
#define KERNEL_FP_TYPE_STR "float"

// --- Platform Specific Abstractions and Placeholders ---
#ifndef __linux__
// Windows specific definitions/placeholders
#define PROT_READ 1       /**< Placeholder memory protection flag (read). */
#define PROT_WRITE 2      /**< Placeholder memory protection flag (write). */
#define MAP_SHARED 1      /**< Placeholder memory mapping flag (shared). */
#define MAP_FAILED ((void *) -1) /**< Placeholder for failed memory map. */
/** @brief Placeholder mmap function for non-Linux systems. Returns MAP_FAILED. */
void* mmap(void* addr, size_t length, int prot, int flags, int fd, long offset) { return MAP_FAILED; }
/** @brief Placeholder munmap function for non-Linux systems. Returns -1. */
int munmap(void* addr, size_t length) { return -1; }
/** @brief Placeholder function to read PCI config space (returns 0). */
unsigned int read_pci_config(int gpu_index, int offset) { return 0; }
/** @brief Macro for exporting functions from a DLL on Windows. */
#ifndef DLLEXPORT
#define DLLEXPORT __declspec(dllexport)
#endif
#else
// Linux specific includes and definitions
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
/** @brief Placeholder function to read PCI config space (returns 0). */
unsigned int read_pci_config(int gpu_index, int offset) { return 0; }
/** @brief Macro for exporting functions with default visibility on Linux/GCC. */
#ifndef DLLEXPORT
#define DLLEXPORT __attribute__((visibility("default")))
#endif
#endif

// --- Global Data Type ---
/** @brief Defines the primary floating-point type used on the host side. */
#define FP_TYPE KERNEL_FP_TYPE

// --- OpenCL Globals ---
/** @brief Handle to the OpenCL context. */
cl_context context = NULL;
/** @brief Handle to the OpenCL command queue. */
cl_command_queue queue = NULL;
/** @brief Handle to the selected OpenCL device ID. */
cl_device_id device_id = NULL;
/** @brief Handle to the selected OpenCL platform ID. */
cl_platform_id platform_id = NULL;
/** @brief Flag indicating if the selected device supports double-precision floating-point (FP64). */
int has_fp64_support = 0;
/**
 * @brief Flag indicating if the device supports necessary atomic operations.
 * Specifically checks for `cl_khr_global_int32_base_atomics`, required by
 * kernels like `proto_segmented_sum_atomic`. Set during initialization.
 */
int has_atomics_support = 0;

// --- Kernel/Program Variables (Global Handles) ---
cl_program matmul_program = NULL;                 cl_kernel matmul_kernel = NULL;
cl_program softmax_program = NULL;                cl_kernel softmax_kernel = NULL;
cl_program gelu_program = NULL;                   cl_kernel gelu_kernel = NULL;
cl_program add_program = NULL;                    cl_kernel add_kernel = NULL;
cl_program mul_program = NULL;                    cl_kernel mul_kernel = NULL;
cl_program layernorm_program = NULL;              cl_kernel layernorm_kernel = NULL;
cl_program transpose_program = NULL;              cl_kernel transpose_kernel = NULL;
cl_program gelu_backward_program = NULL;          cl_kernel gelu_backward_kernel = NULL;
cl_program matmul_backward_da_program = NULL;     cl_kernel matmul_backward_da_kernel = NULL;
cl_program matmul_backward_db_program = NULL;     cl_kernel matmul_backward_db_kernel = NULL;
cl_program layernorm_backward_program = NULL;     cl_kernel layernorm_backward_kernel = NULL;
cl_program adam_program = NULL;                   cl_kernel adam_kernel = NULL;
cl_program softmax_backward_program = NULL;       cl_kernel softmax_backward_kernel = NULL;
cl_program mul_backward_program = NULL;           cl_kernel mul_backward_kernel = NULL;
cl_program transpose_backward_program = NULL;     cl_kernel transpose_backward_kernel = NULL;
cl_program embedding_lookup_program = NULL;       cl_kernel embedding_lookup_kernel = NULL;
cl_program reduce_sum_program = NULL;             cl_kernel reduce_sum_kernel = NULL;
cl_program broadcast_add_program = NULL;          cl_kernel broadcast_add_kernel = NULL;
cl_program transpose_batched_program = NULL;      cl_kernel transpose_batched_kernel = NULL;
cl_program transpose_12_batched_program = NULL;   cl_kernel transpose_12_batched_kernel = NULL;
cl_program matmul_batched_program = NULL;         cl_kernel matmul_batched_kernel = NULL;
cl_program matmul_batched_backward_da_program = NULL; cl_kernel matmul_batched_backward_da_kernel = NULL;
cl_program matmul_batched_backward_db_program = NULL; cl_kernel matmul_batched_backward_db_kernel = NULL;
cl_program log_softmax_program = NULL;            cl_kernel log_softmax_kernel = NULL;
cl_program cross_entropy_program = NULL;          cl_kernel cross_entropy_kernel = NULL;
cl_program add_broadcast_pe_program = NULL;       cl_kernel add_broadcast_pe_kernel = NULL;
cl_program threshold_spike_program = NULL;        cl_kernel threshold_spike_kernel = NULL;
cl_program add_bias_mn_program = NULL;            cl_kernel add_bias_mn_kernel = NULL;
cl_program dynamic_token_assign_program = NULL;   cl_kernel dynamic_token_assign_kernel = NULL;
cl_program pairwise_similarity_program = NULL;    cl_kernel pairwise_similarity_kernel = NULL;
cl_program hebbian_update_local_reduce_program = NULL; cl_kernel hebbian_update_local_reduce_kernel = NULL;
cl_program embedding_backward_calc_delta_local_program = NULL; cl_kernel embedding_backward_calc_delta_local_kernel = NULL;
// Prototype Update Kernels
cl_program proto_segmented_sum_program = NULL;   cl_kernel proto_segmented_sum_kernel = NULL;
cl_program proto_update_step_program = NULL;     cl_kernel proto_update_step_kernel = NULL;
// Loss Shaping Kernels (Keep both for potential compatibility)
cl_program shape_loss_reward_penalty_program = NULL; cl_kernel shape_loss_reward_penalty_kernel = NULL;
cl_program shape_loss_reward_penalty_list_program = NULL; cl_kernel shape_loss_reward_penalty_list_kernel = NULL; // NEU


// --- Driver level bookkeeping for extended API ---
static int g_active_gpu_index = -1;
static char g_device_info_buffer[512] = "";
static double g_last_kernel_time_ms = -1.0;
static int g_last_kernel_time_valid = 0;


/**
 * @brief Enumeration of available GPU commands that can be submitted via the driver.
 * Each enum value corresponds to a specific OpenCL kernel or operation.
 */
typedef enum {
    COMMAND_MATRIX_MULTIPLY = 1,                /**< Standard matrix multiply (C = A @ B). */
    COMMAND_SOFTMAX_ROWWISE = 2,                /**< Row-wise numerically stable softmax. */
    COMMAND_GELU_ELEMENTWISE = 3,               /**< Element-wise GELU activation. */
    COMMAND_ADD_ELEMENTWISE = 4,                /**< Element-wise addition (C = A + B). Also used for Embedding Bwd Pass 2. */
    COMMAND_MUL_ELEMENTWISE = 5,                /**< Element-wise multiplication (C = A * B). */
    COMMAND_LAYER_NORM = 6,                     /**< Layer normalization (row-wise, no affine params). */
    COMMAND_CLONE = 7,                          /**< Simple buffer copy (clEnqueueCopyBuffer). */
    COMMAND_TRANSPOSE = 8,                      /**< Basic 2D matrix transpose. */
    COMMAND_GELU_BACKWARD_ELEMENTWISE = 9,      /**< Element-wise backward pass for GELU. */
    COMMAND_MATMUL_BACKWARD_DA = 10,            /**< Backward pass for matmul, calculating gradient dA. */
    COMMAND_MATMUL_BACKWARD_DB = 11,            /**< Backward pass for matmul, calculating gradient dB. */
    COMMAND_LAYER_NORM_BACKWARD = 12,           /**< Backward pass for layer normalization. */
    COMMAND_ADAM_UPDATE = 13,                   /**< Adam optimizer parameter update step. */
    COMMAND_SOFTMAX_BACKWARD = 14,              /**< Backward pass for softmax. */
    COMMAND_MUL_BACKWARD = 15,                  /**< Backward pass for element-wise multiplication. */
    COMMAND_TRANSPOSE_BACKWARD = 16,            /**< Backward pass for basic 2D transpose (which is another transpose). */
    COMMAND_EMBEDDING_LOOKUP = 17,              /**< Embedding table lookup using indices. */
    COMMAND_EMBEDDING_BACKWARD_PASS1 = 18,      /**< Embedding backward: Calculate delta gradients (uses local reduction). */
    COMMAND_REDUCE_SUM_AXIS01 = 19,             /**< Reduce sum over first two axes (B, M) of a (B, M, N) tensor, output (N). Used for bias gradient. */
    COMMAND_BROADCAST_ADD_BIAS = 20,            /**< Broadcast add bias vector (N) to tensor (B, M, N). */
    COMMAND_TRANSPOSE_BATCHED = 21,             /**< Transpose the last two dimensions of a batched tensor (..., D1, D2) -> (..., D2, D1). */
    COMMAND_MATRIX_MULTIPLY_BATCHED = 22,       /**< Batched matrix multiply (C[b] = A[b] @ B[b]). */
    COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DA = 23, /**< Backward pass for batched matmul, calculating gradient dA. */
    COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DB = 24, /**< Backward pass for batched matmul, calculating gradient dB. */
    COMMAND_TRANSPOSE_12_BATCHED = 25,          /**< Transpose dimensions 1 and 2 of a 4D tensor (B, D1, D2, D3) -> (B, D2, D1, D3). */
    COMMAND_LOG_SOFTMAX_STABLE = 26,            /**< Row-wise numerically stable log-softmax. */
    COMMAND_CROSS_ENTROPY_LOSS_GRAD = 27,       /**< Calculate cross-entropy loss and gradient w.r.t. logits (input expected to be log-probabilities). */
    COMMAND_ADD_BROADCAST_PE = 28,              /**< Broadcast add positional encoding (S, E) to input (B, S, E). */
    COMMAND_HEBBIAN_OUTER_PRODUCT_UPDATE = 29,  /**< Hebbian weight update using outer product (uses local reduction). */
    COMMAND_THRESHOLD_SPIKE = 30,               /**< Generate binary spikes (0 or 1) based on thresholding activations. */
    COMMAND_ADD_BIAS_MN = 31,                   /**< Add Bias Vector (N) to Matrix (M, N). */
    COMMAND_DYNAMIC_TOKEN_ASSIGNMENT = 32,      /**< Assign activation vector to the closest prototype based on dot product similarity. */
    COMMAND_PAIRWISE_SIMILARITY = 33,           /**< Compute pairwise similarity matrix (dot product) between state vectors. */
    COMMAND_PROTO_SEGMENTED_SUM = 34,           /**< Atomically sum activations per prototype based on indices (Requires Atomics). */
    COMMAND_PROTO_UPDATE_STEP = 35,             /**< Update prototypes using accumulated sums and counts from segmented sum. */
    COMMAND_SHAPE_LOSS_REWARD_PENALTY = 36,     /**< Adjust loss based on reward/penalty rules (single pair). */
    COMMAND_SHAPE_LOSS_REWARD_PENALTY_LIST = 37 /**< Adjust loss based on reward/penalty rules (list of pairs). */ // NEU
} GPUCommand;

// --- Forward Declarations for Exported Functions ---
DLLEXPORT int initialize_gpu(int gpu_index);
DLLEXPORT void *allocate_gpu_memory(int gpu_index, size_t size);
DLLEXPORT void free_gpu_memory(int gpu_index, void* buffer_handle);
DLLEXPORT int write_host_to_gpu_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, const void* host_source_ptr);
DLLEXPORT int read_gpu_to_host_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, void* host_destination_ptr);
DLLEXPORT unsigned int simulated_get_compute_unit_count(int gpu_index); // Kept for dummy mode
DLLEXPORT void shutdown_gpu(int gpu_index);
DLLEXPORT int get_num_platforms(void);
DLLEXPORT int get_num_devices_on_platform(int platform_index);
DLLEXPORT int get_device_name(int platform_index, int device_index, char *out, int out_len);

static void update_device_info_buffer(void);
static void reset_last_kernel_time(void);

// Kernel Execution Function Exports
DLLEXPORT int execute_matmul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K);
DLLEXPORT int execute_softmax_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size);
DLLEXPORT int execute_gelu_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_elements);
DLLEXPORT int execute_add_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements);
DLLEXPORT int execute_add_bias_on_gpu(int gpu_index, void* buffer_a_or_c, void* buffer_b_bias, int M, int N);
DLLEXPORT int execute_mul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements);

// Extended API for handwriting workloads
DLLEXPORT int init_driver(int device_index);
DLLEXPORT const char* get_device_info(void);
DLLEXPORT int compile_kernels(const char* kernel_source, char** build_log_out);
DLLEXPORT void* allocate_buffer(size_t bytes);
DLLEXPORT int free_buffer(void* handle);
DLLEXPORT int upload_buffer(void* handle, const void* host_ptr, size_t bytes);
DLLEXPORT int download_buffer(void* handle, void* host_ptr, size_t bytes);
DLLEXPORT int run_kernel(const char* kernel_name,
                        const void** args,
                        const size_t* arg_sizes,
                        const size_t* global_work_size,
                        const size_t* local_work_size);
DLLEXPORT int enqueue_async(const char* kernel_name,
                            const void** args,
                            const size_t* arg_sizes,
                            const size_t* global_work_size,
                            const size_t* local_work_size,
                            void** event_handle_out);
DLLEXPORT int get_last_kernel_time_ms(double* milliseconds_out);
DLLEXPORT int conv2d_forward(int device_index,
                             const float* input,
                             const float* weights,
                             const float* bias,
                             float* output,
                             int batch,
                             int in_channels,
                             int in_height,
                             int in_width,
                             int out_channels,
                             int kernel_h,
                             int kernel_w,
                             int stride_h,
                             int stride_w,
                             int pad_h,
                             int pad_w,
                             int dilation_h,
                             int dilation_w,
                             int groups);
DLLEXPORT int maxpool_forward(int device_index,
                              const float* input,
                              float* output,
                              int batch,
                              int channels,
                              int height,
                              int width,
                              int kernel_size,
                              int stride);
DLLEXPORT int activation_forward(int device_index,
                                 const float* input,
                                 float* output,
                                 int num_elements,
                                 int activation_type,
                                 int apply_spike);
DLLEXPORT int stdp_update_kernel(int device_index,
                                 const float* pre_activations,
                                 const float* post_activations,
                                 float* weights,
                                 int pre_neurons,
                                 int post_neurons,
                                 float learning_rate,
                                 float decay);
DLLEXPORT int execute_layernorm_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size, float eps);
DLLEXPORT int execute_clone_on_gpu(int gpu_index, void* src_buffer, void* dst_buffer, size_t size);
DLLEXPORT int execute_transpose_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int rows, int cols);
DLLEXPORT int execute_gelu_backward_on_gpu(int gpu_index, void* buffer_input, void* buffer_grad_output, void* buffer_grad_input, int num_elements);
DLLEXPORT int execute_matmul_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K);
DLLEXPORT int execute_layernorm_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_x, void* buffer_dx, int num_rows, int row_size, float eps);
DLLEXPORT int execute_adam_update_on_gpu(int gpu_index, void* param_buffer, void* grad_buffer, void* m_buffer, void* v_buffer, int num_elements, int t, float lr, float beta1, float beta2, float eps, float weight_decay);
DLLEXPORT int execute_softmax_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_y, void* buffer_dx, int num_rows, int row_size);
DLLEXPORT int execute_mul_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_A, void* buffer_B, void* buffer_dA, void* buffer_dB, int num_elements);
DLLEXPORT int execute_transpose_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_dA, int rows_A, int cols_A);
DLLEXPORT int execute_embedding_lookup_gpu(int gpu_index, void* idx, void* w, void* o, int b, int s, int d, int v);
DLLEXPORT int execute_embedding_backward_gpu(int gpu_index, void* d_o, void* idx, void* d_w, int b, int s, int d, int v);
DLLEXPORT int execute_reduce_sum_gpu(int gpu_index, void* in, void* out, int B, int M, int N);
DLLEXPORT int execute_broadcast_add_gpu(int gpu_index, void* a, void* b, void* c, int B, int M, int N);
DLLEXPORT int execute_transpose_batched_gpu(int gpu_index, void* in, void* out, int B_flat, int d1, int d2);
DLLEXPORT int execute_transpose_12_batched_gpu(int gpu_index, void* buffer_in, void* buffer_out, int B, int D1, int D2, int D3);
DLLEXPORT int execute_matmul_batched_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K);
DLLEXPORT int execute_matmul_batched_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K);
DLLEXPORT int execute_log_softmax_stable_gpu(int gpu_index, void* input_logits, void* output_log_probs, int B_S_rows, int V_cols);
DLLEXPORT int execute_cross_entropy_loss_grad_gpu(int gpu_index, void* log_probs, void* target_indices, void* grad_input, void* loss_per_sample, int num_rows, int V);
DLLEXPORT int execute_add_broadcast_pe_gpu(int gpu_index, void* input, void* pe_slice, void* output, int B, int S, int E);
DLLEXPORT int execute_hebbian_update_on_gpu(int gpu_index, void* buffer_a, void* buffer_c, void* buffer_w, float learning_rate, int B, int M, int N, int K);
DLLEXPORT int execute_threshold_spike_on_gpu(int gpu_index, void* buffer_activations, void* buffer_spikes, float threshold, int num_elements);
DLLEXPORT int execute_dynamic_token_assignment_gpu(int gpu_index, void* activations_bse, void* prototypes_te, void* output_indices_bs, int B, int S, int E, int T);
DLLEXPORT int execute_pairwise_similarity_gpu(int gpu_index, void* states_nd, void* output_similarity_nn, int N, int D);
DLLEXPORT int execute_proto_segmented_sum_gpu(int gpu_index, void* activations_flat, void* indices_flat, void* proto_sums, void* proto_counts, int num_elements_flat, int E, int T);
DLLEXPORT int execute_proto_update_step_gpu(int gpu_index, void* prototypes, void* proto_sums, void* proto_counts, float learning_rate, int E, int T);
// Loss Shaping Exports
DLLEXPORT int execute_shape_loss_with_reward_penalty_gpu(int gpu_index, void* loss_per_sample_in, void* predictions, void* targets, void* loss_per_sample_out, int num_samples, int num_classes, float penalty_weight, float reward_weight, float high_confidence_threshold, int critical_target_class, int critical_predicted_class);
DLLEXPORT int execute_shape_loss_with_reward_penalty_list_gpu(int gpu_index, void* loss_per_sample_in, void* predictions, void* targets, void* loss_per_sample_out, void* critical_pairs, int num_samples, int num_classes, int num_critical_pairs, float penalty_weight, float reward_weight, float high_confidence_threshold); // NEU

// --- Internal Helper Function Declarations ---
cl_int compile_opencl_kernel(const char* kernel_source, const char* kernel_name, cl_program* program_out, cl_kernel* kernel_out);
const char* clGetErrorString(cl_int error);
int submit_kernel_command(int gpu_index, GPUCommand command, void *data);
int finish_queue_and_check(int gpu_index, const char* func_name);
void shutdown_driver();
unsigned int get_compute_unit_count(int gpu_index);
int zero_gpu_buffer(int gpu_index, void* gpu_buffer_handle, size_t size_bytes);
static cl_int get_reduction_params_helper(size_t* lws_out, size_t* local_mem_bytes_out);


// --- Kernel Source Code Strings ---
// (Alle bisherigen Kernel-Strings bleiben hier unverändert eingefügt)
// Matmul (Standard, Handles 3D @ 2D)
const char *matmul_kernel_src =
"#ifndef M_PI\n"
"#define M_PI 3.14159265358979323846f\n"
"#endif\n"
"__kernel void matrix_multiply(__global const FP_TYPE *a,       /* Input A (B, M, K) or (M, K) */\n"
"                            __global const FP_TYPE *b,       /* Input B (K, N) */\n"
"                            __global FP_TYPE *c,       /* Output C (B, M, N) or (M, N) */\n"
"                            const int B, const int M, const int N, const int K) {\n"
"    int col = get_global_id(0); /* N dimension */\n"
"    int row = get_global_id(1); /* M dimension */\n"
"    int batch_idx = get_global_id(2); /* B dimension */\n"
"\n"
"    /* Check bounds for the output element C[batch_idx, row, col] */\n"
"    if (batch_idx < B && row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        /* Calculate offset for A based on batch index. If B=1, this offset is 0. */\n"
"        size_t a_batch_offset = (size_t)batch_idx * M * K;\n"
"        /* Calculate offset for C based on batch index. */\n"
"        size_t c_batch_offset = (size_t)batch_idx * M * N;\n"
"\n"
"        /* Perform dot product: sum over k (A[batch, row, k] * B[k, col]) */\n"
"        for (int k = 0; k < K; ++k) {\n"
"             /* Access A using batch offset + row/k indices */\n"
"             /* Access B using standard k/col indices (implicitly broadcasted over B) */\n"
"             sum += (float)a[a_batch_offset + row * K + k] * (float)b[(size_t)k * N + col];\n"
"        }\n"
"        /* Write result to output C */\n"
"        c[c_batch_offset + row * N + col] = (FP_TYPE)sum;\n"
"    }\n"
"}";
// Matmul Backward dA (Standard)
const char *matmul_backward_dA_kernel_src =
"/* dA[b,m,k] = sum_n dC[b,m,n] * B[k,n] (equivalent to dC @ B^T) */\n"
"__kernel void matmul_backward_da(__global const FP_TYPE *dC, /* Gradient dC (B, M, N) */\n"
"                               __global const FP_TYPE *B,  /* Original Input B (K, N) */\n"
"                               __global FP_TYPE *dA, /* Output Gradient dA (B, M, K) */\n"
"                               const int B_dim, const int M_dim, const int N_dim, const int K_dim) {\n"
"    int k = get_global_id(0); /* K dimension */\n"
"    int m = get_global_id(1); /* M dimension */\n"
"    int b = get_global_id(2); /* B dimension */\n"
"\n"
"    /* Bounds check for dA element dA[b, m, k] */\n"
"    if (b < B_dim && m < M_dim && k < K_dim) {\n"
"        float gradient_sum = 0.0f;\n"
"        size_t dc_batch_offset = (size_t)b * M_dim * N_dim;\n"
"        size_t da_batch_offset = (size_t)b * M_dim * K_dim;\n"
"\n"
"        /* Sum over N dimension */\n"
"        for (int n = 0; n < N_dim; ++n) {\n"
"            /* dC[b, m, n] * B[k, n] */\n"
"            gradient_sum += (float)dC[dc_batch_offset + m * N_dim + n] * (float)B[(size_t)k * N_dim + n];\n"
"        }\n"
"        dA[da_batch_offset + m * K_dim + k] = (FP_TYPE)gradient_sum;\n"
"    }\n"
"}";
// Matmul Backward dB (Standard)
const char *matmul_backward_dB_kernel_src =
"/* dB[k,n] = sum_b sum_m A[b,m,k] * dC[b,m,n] (equivalent to A^T @ dC, summed over B) */\n"
"__kernel void matmul_backward_db(__global const FP_TYPE *A,  /* Original Input A (B, M, K) */\n"
"                               __global const FP_TYPE *dC, /* Gradient dC (B, M, N) */\n"
"                               __global FP_TYPE *dB, /* Output Gradient dB (K, N) */\n"
"                               const int B_dim, const int M_dim, const int N_dim, const int K_dim) {\n"
"    int n = get_global_id(0); /* N dimension */\n"
"    int k = get_global_id(1); /* K dimension */\n"
"\n"
"    /* Bounds check for dB element dB[k, n] */\n"
"    if (k < K_dim && n < N_dim) {\n"
"        float gradient_sum = 0.0f;\n"
"        /* Sum over Batch dimension B */\n"
"        for (int b = 0; b < B_dim; ++b) {\n"
"            size_t a_batch_offset = (size_t)b * M_dim * K_dim;\n"
"            size_t dc_batch_offset = (size_t)b * M_dim * N_dim;\n"
"            /* Sum over M dimension */\n"
"            for (int m = 0; m < M_dim; ++m) {\n"
"                /* A[b, m, k] * dC[b, m, n] */\n"
"                gradient_sum += (float)A[a_batch_offset + m * K_dim + k] * (float)dC[dc_batch_offset + m * N_dim + n];\n"
"            }\n"
"        }\n"
"        /* Write the final summed gradient to dB */\n"
"        dB[(size_t)k * N_dim + n] = (FP_TYPE)gradient_sum;\n"
"    }\n"
"}";
// Softmax (Row-wise, Numerically Stable)
const char *softmax_kernel_src =
"#ifndef HUGE_VALF /* Standard C float constant for infinity */\n"
"#define HUGE_VALF (__builtin_huge_valf()) /* Use compiler built-in if available */\n"
"#endif\n"
"#ifndef native_exp /* Use standard exp if native_exp is not defined/available */\n"
"#define native_exp exp\n"
"#endif\n"
"\n"
"__kernel void softmax_rowwise(__global const FP_TYPE *input, /* Input tensor (num_rows, row_size) flattened */\n"
"                            __global FP_TYPE *output,      /* Output tensor (num_rows, row_size) flattened */\n"
"                            const int num_rows, const int row_size) {\n"
"    int row = get_global_id(0); /* Index for the row (0 to num_rows-1) */\n"
"\n"
"    if (row < num_rows) {\n"
"        size_t offset = (size_t)row * row_size; /* Base offset for this row */\n"
"        __global const FP_TYPE* in_row = input + offset;\n"
"        __global FP_TYPE* out_row = output + offset;\n"
"\n"
"        /* 1. Find max value in the row for numerical stability */\n"
"        float max_val = -HUGE_VALF;\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            if ((float)in_row[i] > max_val) {\n"
"                max_val = (float)in_row[i];\n"
"            }\n"
"        }\n"
"\n"
"        /* 2. Calculate sum of exponentials (shifted by max_val) */\n"
"        float sum_exp = 0.0f;\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            sum_exp += native_exp((float)in_row[i] - max_val);\n"
"        }\n"
"\n"
"        /* 3. Calculate inverse sum (with epsilon for stability if sum_exp is close to zero) */\n"
"        float inv_sum = 1.0f / (sum_exp + 1e-9f);\n"
"\n"
"        /* 4. Calculate softmax probabilities: exp(x_i - max_val) / sum(exp(x_j - max_val)) */\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            out_row[i] = (FP_TYPE)(native_exp((float)in_row[i] - max_val) * inv_sum);\n"
"        }\n"
"    }\n"
"}";
// LogSoftmax (Row-wise, Numerically Stable)
const char *log_softmax_stable_kernel_src =
"#define native_exp exp\n"
"#define native_log log\n"
"\n"
"#ifndef HUGE_VALF\n"
"#define HUGE_VALF (__builtin_huge_valf())\n"
"#endif\n"
"\n"
"__kernel void log_softmax_stable_rowwise(\n"
"                    __global const FP_TYPE *input_logits, /* Input (B * S, V) flattened */\n"
"                    __global FP_TYPE *output_log_probs,   /* Output (B * S, V) flattened */\n"
"                    const int num_rows,  /* B * S */\n"
"                    const int row_size   /* V (Vocabulary size) */\n"
"                    ) {\n"
"    int row = get_global_id(0); /* Index from 0 to B*S - 1 */\n"
"\n"
"    if (row < num_rows) {\n"
"        size_t offset = (size_t)row * row_size;\n"
"        __global const FP_TYPE* in_row = input_logits + offset;\n"
"        __global FP_TYPE* out_row = output_log_probs + offset;\n"
"\n"
"        /* 1. Find max value in the row for numerical stability */\n"
"        float max_val = -HUGE_VALF;\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            if ((float)in_row[i] > max_val) {\n"
"                max_val = (float)in_row[i];\n"
"            }\n"
"        }\n"
"\n"
"        /* 2. Calculate sum of exponentials (shifted by max_val) */\n"
"        float sum_exp = 0.0f;\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            sum_exp += native_exp((float)in_row[i] - max_val);\n"
"        }\n"
"\n"
"        /* 3. Calculate log of the sum of exponentials (LogSumExp trick part 2) */\n"
"        float log_sum_exp = native_log(sum_exp + 1e-9f);\n"
"\n"
"        /* 4. Calculate log probabilities: log_prob = x - max - log(sum(exp(x - max))) */\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            out_row[i] = (FP_TYPE)(((float)in_row[i] - max_val) - log_sum_exp);\n"
"        }\n"
"    }\n"
"}";
// Cross Entropy Loss + Gradient w.r.t Logits
const char *cross_entropy_loss_grad_kernel_src =
"#ifndef native_exp\n"
"#define native_exp exp\n"
"#endif\n"
"\n"
"/* Calculates loss and gradient for cross-entropy. */\n"
"/* Assumes log_probs input is from a log_softmax operation. */\n"
"/* Target indices are integer class IDs. */\n"
"__kernel void cross_entropy_loss_grad(\n"
"                __global const FP_TYPE* log_probs,      /* Input: Log probabilities (B, S, V) flattened (B*S, V) */\n"
"                __global const int* target_indices,   /* Input: Target class indices (B, S) flattened (B*S,) */\n"
"                __global FP_TYPE* grad_input,         /* Output: Gradient w.r.t logits (B, S, V) flattened (B*S, V) */\n"
"                __global FP_TYPE* loss_per_sample,    /* Output: Loss per sample/token (B, S) flattened (B*S,) */\n"
"                const int num_rows, /* B * S */\n"
"                const int V /* Vocabulary size (row_size) */\n"
"                ) {\n"
"\n"
"     /* Global ID maps to the row (token/sample) index */\n"
"    int row = get_global_id(0); /* Index from 0 to num_rows-1 */\n"
"\n"
"    if (row < num_rows) {\n"
"        size_t base_offset = (size_t)row * V; /* Offset for log_probs and grad_input row */\n"
"        __global const FP_TYPE* log_probs_row = log_probs + base_offset;\n"
"        __global FP_TYPE* grad_input_row = grad_input + base_offset;\n"
"\n"
"        /* Get the target index for this row (sample/token) */\n"
"        int target_idx = target_indices[row];\n"
"\n"
"        /* --- Calculate Gradient: grad = probs - one_hot --- */\n"
"        /* This requires calculating probs = exp(log_probs) */\n"
"        for (int v = 0; v < V; ++v) {\n"
"            float current_log_prob = (float)log_probs_row[v];\n"
"            float current_prob = native_exp(current_log_prob);\n"
"            float grad_val = current_prob; /* Initialize gradient with probability */\n"
"\n"
"            /* Subtract 1.0f if this is the target class index */\n"
"            if (v == target_idx) {\n"
"                grad_val -= 1.0f;\n"
"            }\n"
"            grad_input_row[v] = (FP_TYPE)grad_val;\n"
"        }\n"
"\n"
"        /* --- Calculate Loss: loss = -log_prob[target_idx] --- */\n"
"        /* Ensure target_idx is valid before accessing log_probs */\n"
"        if (target_idx >= 0 && target_idx < V) {\n"
"            float target_log_prob = (float)log_probs_row[target_idx];\n"
"            /* Ensure loss is non-negative using fmax (built-in OpenCL function) */\n"
"            loss_per_sample[row] = (FP_TYPE)(fmax(0.0f, -target_log_prob));\n"
"        } else {\n"
"            /* Handle invalid target index (e.g., padding index like -1 or specific id) */\n"
"            /* Assign 0 loss for invalid/padding targets. */\n"
"            loss_per_sample[row] = (FP_TYPE)(0.0f);\n"
"        }\n"
"    }\n"
"}";
// Softmax Backward
const char *softmax_backward_kernel_src =
"#ifdef CL_HAS_FP64 /* Use double for accumulation if supported */\n"
"    typedef double ACCUM_TYPE;\n"
"    #define ACCUM_CONST(x) (double)(x)\n"
"#else /* Fallback to float accumulation */\n"
"    typedef float ACCUM_TYPE;\n"
"    #define ACCUM_CONST(x) (float)(x)\n"
"#endif\n"
"\n"
"/* Computes dL/dx = (dL/dy - sum(dL/dy * y)) * y */\n"
"__kernel void softmax_backward(__global const FP_TYPE *dy_in, /* Gradient dL/dy (num_rows, row_size) */\n"
"                               __global const FP_TYPE *y,    /* Output of forward softmax y (num_rows, row_size) */\n"
"                               __global FP_TYPE *dx,   /* Output Gradient dL/dx (num_rows, row_size) */\n"
"                               const int num_rows, const int row_size) {\n"
"    int row = get_global_id(0); /* Row index */\n"
"\n"
"    if (row < num_rows) {\n"
"        size_t offset = (size_t)row * row_size;\n"
"        __global const FP_TYPE* dy_row = dy_in + offset;\n"
"        __global const FP_TYPE* y_row = y + offset;\n"
"        __global FP_TYPE* dx_row = dx + offset;\n"
"\n"
"        /* 1. Calculate dot product: sum(dL/dy * y) for this row */\n"
"        ACCUM_TYPE dot_product = ACCUM_CONST(0.0);\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            dot_product += (ACCUM_TYPE)dy_row[i] * (ACCUM_TYPE)y_row[i];\n"
"        }\n"
"\n"
"        /* 2. Calculate gradient dL/dx for each element in the row */\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            ACCUM_TYPE dy_val = (ACCUM_TYPE)dy_row[i];\n"
"            ACCUM_TYPE y_val = (ACCUM_TYPE)y_row[i];\n"
"            /* dx_i = (dy_i - dot_product) * y_i */\n"
"            ACCUM_TYPE dx_val = (dy_val - dot_product) * y_val;\n"
"            dx_row[i] = (FP_TYPE)dx_val; /* Cast back to original FP_TYPE */\n"
"        }\n"
"    }\n"
"}";
// GELU Activation (Elementwise)
const char *gelu_kernel_src =
"/* Define constants used by GELU */\n"
"#ifndef M_PI\n"
"#define M_PI 3.14159265358979323846f\n"
"#endif\n"
"#ifndef M_SQRT1_2 /* 1/sqrt(2) */\n"
"#define M_SQRT1_2 0.70710678118654752440f\n"
"#endif\n"
"\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable /* Enable FP64 if available for erf calculation */\n"
"#ifndef native_erf /* Use standard erf if native version is not available/defined */\n"
"#define native_erf erf\n"
"#endif\n"
"\n"
"/* GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) */\n"
"__kernel void gelu_elementwise(__global const FP_TYPE *input, /* Input tensor */\n"
"                               __global FP_TYPE *output,      /* Output tensor */\n"
"                               const int num_elements) {\n"
"    /* Total number of elements */\n"
"    int idx = get_global_id(0); /* Global element index */\n"
"\n"
"    if (idx < num_elements) {\n"
"        float x = (float)input[idx]; /* Read input as float */\n"
"        /* Calculate GELU using native erf if possible */\n"
"        float gelu_val = 0.5f * x * (1.0f + native_erf(x * M_SQRT1_2));\n"
"        output[idx] = (FP_TYPE)gelu_val; /* Write result, cast back to FP_TYPE */\n"
"    }\n"
"}";
// GELU Backward (Elementwise)
const char *gelu_backward_kernel_src =
"/* Define constants used by GELU backward */\n"
"#ifndef M_PI\n"
"#define M_PI 3.14159265358979323846f\n"
"#endif\n"
"#ifndef M_SQRT1_2 /* 1/sqrt(2) */\n"
"#define M_SQRT1_2 0.70710678118654752440f\n"
"#endif\n"
"#ifndef M_1_SQRT2PI /* 1/sqrt(2*pi) - Used in PDF */\n"
"#define M_1_SQRT2PI 0.39894228040143267794f\n"
"#endif\n"
"\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable /* Enable FP64 for erf/exp if available */\n"
"#ifndef native_erf /* Use standard erf if native is not defined */\n"
"#define native_erf erf\n"
"#endif\n"
"#ifndef native_exp /* Use standard exp if native is not defined */\n"
"#define native_exp exp\n"
"#endif\n"
"\n"
"/* dGELU/dx = 0.5 * (1 + erf(x / sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-0.5 * x^2) */\n"
"/*           = CDF(x) + x * PDF(x) */\n"
"/* dL/dx = dL/dy * dGELU/dx */\n"
"__kernel void gelu_backward_elementwise(__global const FP_TYPE *input,       /* Original input x to GELU forward */\n"
"                                       __global const FP_TYPE *grad_output, /* Gradient dL/dy from subsequent layer */\n"
"                                       __global FP_TYPE *grad_input,  /* Output gradient dL/dx */\n"
"                                       const int num_elements) {\n"
"\n"
"     /* Total number of elements */\n"
"    int idx = get_global_id(0); /* Global element index */\n"
"\n"
"    if (idx < num_elements) {\n"
"        float x = (float)input[idx];       /* Original input value */\n"
"        float dy = (float)grad_output[idx]; /* Incoming gradient */\n"
"\n"
"        /* Calculate CDF term: 0.5 * (1 + erf(x / sqrt(2))) */\n"
"        float cdf_term = 0.5f * (1.0f + native_erf(x * M_SQRT1_2));\n"
"        /* Calculate PDF term: (1/sqrt(2*pi)) * exp(-0.5 * x^2) */\n"
"        float pdf_term = M_1_SQRT2PI * native_exp(-0.5f * x * x);\n"
"        /* Calculate dGELU/dx = CDF(x) + x * PDF(x) */\n"
"        float dgelu_dx = cdf_term + x * pdf_term;\n"
"\n"
"        /* Calculate final gradient: dL/dx = dL/dy * dGELU/dx */\n"
"        grad_input[idx] = (FP_TYPE)(dy * dgelu_dx); /* Write result, cast back to FP_TYPE */\n"
"    }\n"
"}";
// Add (Elementwise) - Used for general add and Embedding Bwd Pass 2
const char *add_kernel_src =
"/* c[i] = a[i] + b[i] */\n"
"__kernel void add_elementwise(__global const FP_TYPE *a, /* Input tensor A */\n"
"                             __global const FP_TYPE *b, /* Input tensor B */\n"
"                             __global FP_TYPE *c, /* Output tensor C */\n"
"                             const int num_elements) { /* Total number of elements */\n"
"    int idx = get_global_id(0); /* Global element index */\n"
"\n"
"    if (idx < num_elements) {\n"
"        c[idx] = (FP_TYPE)((float)a[idx] + (float)b[idx]); /* Perform addition and cast back */\n"
"    }\n"
"}";
// Multiply (Elementwise)
const char *mul_kernel_src =
"/* c[i] = a[i] * b[i] */\n"
"__kernel void mul_elementwise(__global const FP_TYPE *a, /* Input tensor A */\n"
"                             __global const FP_TYPE *b, /* Input tensor B */\n"
"                             __global FP_TYPE *c, /* Output tensor C */\n"
"                             const int num_elements) { /* Total number of elements */\n"
"    int idx = get_global_id(0); /* Global element index */\n"
"\n"
"    if (idx < num_elements) {\n"
"        c[idx] = (FP_TYPE)((float)a[idx] * (float)b[idx]); /* Perform multiplication and cast back */\n"
"    }\n"
"}";
// Multiply Backward (Elementwise)
const char *mul_backward_kernel_src =
"/* Computes gradients for elementwise multiplication C = A * B */\n"
"/* dA = dC * B */\n"
"/* dB = dC * A */\n"
"__kernel void mul_backward(__global const FP_TYPE *dC, /* Gradient dL/dC from subsequent layer */\n"
"                         __global const FP_TYPE *A,  /* Original Input A from forward pass */\n"
"                         __global const FP_TYPE *B,  /* Original Input B from forward pass */\n"
"                         __global FP_TYPE *dA, /* Output gradient dL/dA (can be NULL conceptually, but kernel expects a buffer if arg is set) */\n"
"                         __global FP_TYPE *dB, /* Output gradient dL/dB (can be NULL conceptually, but kernel expects a buffer if arg is set) */\n"
"                         const int num_elements) { /* Total number of elements */\n"
"    int idx = get_global_id(0); /* Global element index */\n"
"\n"
"    if (idx < num_elements) {\n"
"        float dC_val = (float)dC[idx]; /* Incoming gradient */\n"
"        float A_val = (float)A[idx];   /* Original input A */\n"
"        float B_val = (float)B[idx];   /* Original input B */\n"
"\n"
"        /* Calculate gradient w.r.t. A: dA = dC * B */\n"
"        /* Host code MUST ensure only valid buffers are passed if grads are needed. */\n"
"        dA[idx] = (FP_TYPE)(dC_val * B_val);\n"
"\n"
"        /* Calculate gradient w.r.t. B: dB = dC * A */\n"
"        /* Host code MUST ensure only valid buffers are passed if grads are needed. */\n"
"        dB[idx] = (FP_TYPE)(dC_val * A_val);\n"
"    }\n"
"}";
// Layer Normalization (Row-wise)
const char *layernorm_kernel_src =
"/* Define accumulation type based on FP64 support */\n"
"#ifdef CL_HAS_FP64\n"
"    typedef double ACCUM_TYPE;\n"
"    #define ACCUM_CONST(x) (double)(x)\n"
"#else\n"
"    typedef float ACCUM_TYPE;\n"
"    #define ACCUM_CONST(x) (float)(x)\n"
"#endif\n"
"\n"
"#ifndef native_rsqrt /* Use standard rsqrt if native version is not available */\n"
"#define native_rsqrt rsqrt\n"
"#endif\n"
"\n"
"/* Performs Layer Normalization along the last dimension. */\n"
"__kernel void layer_norm(__global const FP_TYPE *input, /* Input tensor (num_rows, row_size) flattened */\n"
"                         __global FP_TYPE *output,      /* Output tensor (num_rows, row_size) flattened */\n"
"                         const int num_rows, const int row_size, const float cl_eps) { /* Epsilon added in C host code */\n"
"    int row = get_global_id(0); /* Row index */\n"
"\n"
"    if (row < num_rows) {\n"
"        size_t offset = (size_t)row * row_size; /* Base offset for this row */\n"
"        __global const FP_TYPE* in_row = input + offset;\n"
"        __global FP_TYPE* out_row = output + offset;\n"
"\n"
"        /* 1. Calculate mean of the row */\n"
"        ACCUM_TYPE mean = ACCUM_CONST(0.0);\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            mean += (ACCUM_TYPE)in_row[i];\n"
"        }\n"
"        mean /= ACCUM_CONST(row_size);\n"
"\n"
"        /* 2. Calculate variance of the row */\n"
"        ACCUM_TYPE variance = ACCUM_CONST(0.0);\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            ACCUM_TYPE diff = (ACCUM_TYPE)in_row[i] - mean;\n"
"            variance += diff * diff;\n"
"        }\n"
"        variance /= ACCUM_CONST(row_size);\n"
"\n"
"        /* 3. Calculate inverse standard deviation (with epsilon) */\n"
"        /* Use native_rsqrt for potential performance improvement */\n"
"        ACCUM_TYPE eps_accum = (ACCUM_TYPE)cl_eps;\n"
"        ACCUM_TYPE inv_stddev = native_rsqrt(variance + eps_accum);\n"
"\n"
"        /* 4. Normalize the row: output = (input - mean) * inv_stddev */\n"
"        for (int i = 0; i < row_size; ++i) {\n"
"            out_row[i] = (FP_TYPE)(((ACCUM_TYPE)in_row[i] - mean) * inv_stddev);\n"
"        }\n"
"    }\n"
"}";
// Layer Normalization Backward
const char *layernorm_backward_kernel_src =
"#ifdef CL_HAS_FP64\n"
"    typedef double ACCUM_TYPE;\n"
"    #define ACCUM_CONST(x) (double)(x)\n"
"#else\n"
"    typedef float ACCUM_TYPE;\n"
"    #define ACCUM_CONST(x) (float)(x)\n"
"#endif\n"
"\n"
"#ifndef native_rsqrt\n"
"#define native_rsqrt rsqrt\n"
"#endif\n"
"\n"
"/* Calculates gradient for Layer Normalization (without affine parameters gamma/beta). */\n"
"__kernel void layer_norm_backward(__global const FP_TYPE *dy, /* Gradient dL/dy from subsequent layer */\n"
"                                __global const FP_TYPE *x,  /* Original input x to forward LayerNorm */\n"
"                                __global FP_TYPE *dx, /* Output gradient dL/dx */\n"
"                                const int num_rows, const int row_size, const float cl_eps) {\n"
"    int row = get_global_id(0); /* Row index */\n"
"\n"
"    if (row < num_rows) {\n"
"        size_t offset = (size_t)row * row_size;\n"
"        __global const FP_TYPE* dy_row = dy + offset;\n"
"        __global const FP_TYPE* x_row = x + offset;\n"
"        __global FP_TYPE* dx_row = dx + offset;\n"
"\n"
"        /* --- Recompute mean and variance (needed for backward) --- */\n"
"        ACCUM_TYPE mean = ACCUM_CONST(0.0);\n"
"        for (int i = 0; i < row_size; ++i) { mean += (ACCUM_TYPE)x_row[i]; }\n"
"        mean /= ACCUM_CONST(row_size);\n"
"\n"
"        ACCUM_TYPE variance = ACCUM_CONST(0.0);\n"
"        for (int i = 0; i < row_size; ++i) { ACCUM_TYPE diff = (ACCUM_TYPE)x_row[i] - mean; variance += diff * diff; }\n"
"        variance /= ACCUM_CONST(row_size);\n"
"\n"
"        ACCUM_TYPE eps_accum = (ACCUM_TYPE)cl_eps;\n"
"        ACCUM_TYPE inv_stddev = native_rsqrt(variance + eps_accum); /* 1 / sqrt(var + eps) */\n"
"        ACCUM_TYPE N_accum = ACCUM_CONST(row_size);\n"
"\n"
"        /* --- Calculate intermediate sums needed for the gradient --- */\n"
"        ACCUM_TYPE sum_dy = ACCUM_CONST(0.0);           /* sum(dy) */\n"
"        ACCUM_TYPE sum_dy_xhat = ACCUM_CONST(0.0);    /* sum(dy * x_hat) */\n"
"        /* Calculate x_hat = (x - mean) * inv_stddev on the fly */\n"
"        for (int i = 0; i < row_size; i++) {\n"
"            ACCUM_TYPE x_hat = ((ACCUM_TYPE)x_row[i] - mean) * inv_stddev;\n"
"            ACCUM_TYPE dy_accum = (ACCUM_TYPE)dy_row[i];\n"
"            sum_dy += dy_accum;\n"
"            sum_dy_xhat += dy_accum * x_hat;\n"
"        }\n"
"\n"
"        /* --- Calculate gradient dL/dx for each element --- */\n"
"        /* Formula (simplified, without affine params): */\n"
"        /* dx = (1/N) * inv_stddev * [ N*dy - sum(dy) - x_hat * sum(dy * x_hat) ] */\n"
"        for (int i = 0; i < row_size; i++) {\n"
"            ACCUM_TYPE x_hat = ((ACCUM_TYPE)x_row[i] - mean) * inv_stddev; /* Recompute x_hat */\n"
"            ACCUM_TYPE dy_accum = (ACCUM_TYPE)dy_row[i];\n"
"\n"
"            ACCUM_TYPE term1 = N_accum * dy_accum; /* N * dy_i */\n"
"            ACCUM_TYPE term2 = sum_dy;             /* sum(dy) */\n"
"            ACCUM_TYPE term3 = x_hat * sum_dy_xhat; /* x_hat_i * sum(dy * x_hat) */\n"
"\n"
"            /* Combine terms and scale */\n"
"            ACCUM_TYPE dx_accum = (ACCUM_CONST(1.0) / N_accum) * inv_stddev * (term1 - term2 - term3);\n"
"\n"
"            dx_row[i] = (FP_TYPE)dx_accum; /* Write final gradient */\n"
"        }\n"
"    }\n"
"}";
// Transpose (Basic 2D)
const char *transpose_kernel_src =
"/* Transposes a 2D matrix. Output[col, row] = Input[row, col] */\n"
"__kernel void transpose(__global const FP_TYPE *input, /* Input matrix (rows, cols) */\n"
"                        __global FP_TYPE *output,      /* Output matrix (cols, rows) */\n"
"                        const int rows, const int cols) {\n"
"    /* Use 2D global IDs corresponding to the OUTPUT matrix dimensions */\n"
"    int out_row = get_global_id(0); /* Corresponds to input cols (0 to cols-1) */\n"
"    int out_col = get_global_id(1); /* Corresponds to input rows (0 to rows-1) */\n"
"\n"
"    /* Bounds check for output indices */\n"
"    if (out_row < cols && out_col < rows) {\n"
"        /* Calculate linear index for output[out_row, out_col] (stride is rows) */\n"
"        size_t output_idx = (size_t)out_row * rows + out_col;\n"
"        /* Calculate linear index for input[out_col, out_row] (stride is cols) */\n"
"        size_t input_idx = (size_t)out_col * cols + out_row;\n"
"\n"
"        output[output_idx] = input[input_idx];\n"
"    }\n"
"}";
// Transpose Backward (Basic 2D)
const char *transpose_backward_kernel_src =
"/* Backward of transpose Y=X^T is dX = (dY)^T */\n"
"/* This kernel effectively performs another transpose on the incoming gradient dY. */\n"
"__kernel void transpose_backward(__global const FP_TYPE *dC, /* Gradient dL/dC (dims: cols_A x rows_A) */\n"
"                               __global FP_TYPE *dA,       /* Output gradient dL/dA (dims: rows_A x cols_A) */\n"
"                               const int rows_A, const int cols_A) {\n"
"    /* Use 2D global IDs corresponding to the OUTPUT gradient dA dimensions */\n"
"    int dA_row = get_global_id(0); /* 0 to rows_A-1 */\n"
"    int dA_col = get_global_id(1); /* 0 to cols_A-1 */\n"
"\n"
"    /* Bounds check for dA indices */\n"
"    if (dA_row < rows_A && dA_col < cols_A) {\n"
"        /* Calculate linear index for dA[dA_row, dA_col] (stride is cols_A) */\n"
"        size_t dA_idx = (size_t)dA_row * cols_A + dA_col;\n"
"        /* Calculate corresponding linear index in dC (transposed access) */\n"
"        /* dC has dimensions (cols_A, rows_A), so dC[dA_col, dA_row] */\n"
"        size_t dC_idx = (size_t)dA_col * rows_A + dA_row; /* Stride is rows_A */\n"
"\n"
"        dA[dA_idx] = dC[dC_idx]; /* Perform the transpose copy */\n"
"    }\n"
"}";
// Adam Optimizer Update
const char *adam_kernel_src =
"/* Use standard sqrt if native version is not available */\n"
"#ifndef native_sqrt\n"
"#define native_sqrt sqrt\n"
"#endif\n"
"\n"
"/* Performs Adam weight update step. */\n"
"/* Note: m and v states are expected to be float, regardless of KERNEL_FP_TYPE. */\n"
"__kernel void adam_update(__global FP_TYPE *param,       /* Parameter tensor (to be updated) */\n"
"                         __global const FP_TYPE *grad,       /* Gradient tensor dL/dparam */\n"
"                         __global float *m,           /* Adam state m (1st moment, float) */\n"
"                         __global float *v,           /* Adam state v (2nd moment, float) */\n"
"                         const int num_elements,   /* Total number of elements */\n"
"                         const float lr,             /* Learning rate */\n"
"                         const float beta1,          /* Adam beta1 hyperparameter */\n"
"                         const float beta2,          /* Adam beta2 hyperparameter */\n"
"                         const float epsilon,        /* Adam epsilon hyperparameter */\n"
"                         const float weight_decay,   /* Weight decay factor (L2 regularization) */\n"
"                         const float beta1_t,        /* Precomputed beta1^t */\n"
"                         const float beta2_t) {      /* Precomputed beta2^t */\n"
"    int idx = get_global_id(0); /* Global element index */\n"
"\n"
"    if (idx < num_elements) {\n"
"        /* Read values, using float for internal Adam calculations for stability/consistency */\n"
"        float p = (float)param[idx];\n"
"        float g = (float)grad[idx];\n"
"        float m_curr = m[idx]; /* Read current m state */\n"
"        float v_curr = v[idx]; /* Read current v state */\n"
"\n"
"        /* Apply weight decay (L2 regularization) if enabled */\n"
"        if (weight_decay > 0.0f) {\n"
"            g += weight_decay * p; /* Add weight decay term to the gradient */\n"
"        }\n"
"\n"
"        /* Update biased first moment estimate (m) */\n"
"        float m_new = beta1 * m_curr + (1.0f - beta1) * g;\n"
"        /* Update biased second raw moment estimate (v) */\n"
"        float v_new = beta2 * v_curr + (1.0f - beta2) * (g * g);\n"
"\n"
"        /* Compute bias-corrected first moment estimate (m_hat) */\n"
"        /* Add small epsilon to denominator for numerical stability, although 1-beta1_t is usually far from 0 early on. */\n"
"        float m_hat = m_new / (1.0f - beta1_t + 1e-9f);\n"
"        /* Compute bias-corrected second raw moment estimate (v_hat) */\n"
"        float v_hat = v_new / (1.0f - beta2_t + 1e-9f);\n"
"\n"
"        /* Compute the parameter update step */\n"
"        /* update = lr * m_hat / (sqrt(v_hat) + epsilon) */\n"
"        float update = lr * m_hat / (native_sqrt(v_hat) + epsilon);\n"
"\n"
"        /* Apply the update to the parameter */\n"
"        float p_new = p - update;\n"
"\n"
"        /* Write back updated parameter and Adam states */\n"
"        param[idx] = (FP_TYPE)p_new; /* Cast back to original parameter type */\n"
"        m[idx] = m_new;             /* Write updated m state (float) */\n"
"        v[idx] = v_new;             /* Write updated v state (float) */\n"
"    }\n"
"}";
// Embedding Lookup (GPU Version)
const char *embedding_lookup_kernel_src =
"/* Performs embedding lookup: output[b, s, :] = weights[indices[b, s], :] */\n"
"__kernel void embedding_lookup(\n"
"             __global const int* indices,     /* Input: Indices tensor (B, S) flattened (B*S,) */\n"
"             __global const FP_TYPE* weights, /* Input: Weight matrix (V, D) */\n"
"             __global FP_TYPE* output,        /* Output: Output tensor (B, S, D) flattened (B*S, D) */\n"
"             const int seq_len,     /* S */\n"
"             const int embed_dim,   /* D */\n"
"             const int vocab_size   /* V */\n"
"             /* B is implicit via global size dim 1 */\n"
"             ) {\n"
"    /* Use 2D global IDs mapping to (s, b) */\n"
"    int s = get_global_id(0); /* Sequence dimension index (0 to S-1) */\n"
"    int b = get_global_id(1); /* Batch dimension index (0 to B-1) */\n"
"\n"
"    /* Calculate linear index for the input indices array (B*S,) */\n"
"    size_t indices_idx = (size_t)b * seq_len + s;\n"
"\n"
"    /* Read the vocabulary index for this (b, s) position */\n"
"    int vocab_idx = indices[indices_idx];\n"
"\n"
"    /* Calculate the base offset for the output tensor row (B*S, D) */\n"
"    size_t output_offset = ((size_t)b * seq_len + s) * embed_dim;\n"
"\n"
"    /* --- Bounds Check for Vocabulary Index --- */\n"
"    /* Check if the vocabulary index is valid (within [0, vocab_size-1]) */\n"
"    if (vocab_idx < 0 || vocab_idx >= vocab_size) {\n"
"        /* Handle out-of-bounds index (e.g., padding or error) -> Output zeros */\n"
"        for(int d = 0; d < embed_dim; ++d) {\n"
"            output[output_offset + d] = (FP_TYPE)0.0;\n"
"        }\n"
"        return; /* Exit early for this work-item */\n"
"    }\n"
"    /* ----------------------------------------- */\n"
"\n"
"    /* Calculate the base offset for the corresponding row in the weight matrix (V, D) */\n"
"    size_t weight_offset = (size_t)vocab_idx * embed_dim;\n"
"\n"
"    /* Copy the embedding vector from weights to output for the full embedding dimension D */\n"
"    for (int d = 0; d < embed_dim; ++d) {\n"
"        output[output_offset + d] = weights[weight_offset + d];\n"
"    }\n"
"}";
// Embedding Backward Pass 1 Kernel (Local Reduction, No Atomics)
const char *embedding_backward_calc_delta_local_kernel_src =
"/* Define work-group size for reduction (can be tuned) */\n"
"#ifndef REDUCE_WG_SIZE\n"
"#define REDUCE_WG_SIZE 256\n"
"#endif\n"
"\n"
"/* Define accumulation type based on FP64 support */\n"
"#ifdef CL_HAS_FP64\n"
"    typedef double REDUCE_ACCUM_TYPE;\n"
"    #define REDUCE_ACCUM_CONST(x) (double)(x)\n"
"#else\n"
"    typedef float REDUCE_ACCUM_TYPE;\n"
"    #define REDUCE_ACCUM_CONST(x) (float)(x)\n"
"#endif\n"
"\n"
"__kernel void embedding_backward_calc_delta_local(\n"
"                 __global const FP_TYPE* grad_output, /* Input: Gradient dL/dOutput (B, S, D) flattened (B*S, D) */\n"
"                 __global const int* indices,         /* Input: Indices used in forward (B, S) flattened (B*S,) */\n"
"                 __global FP_TYPE* delta_dw,          /* Output: Temporary Delta Gradient (V, D), zero-initialized */\n"
"                 const int B_dim,      /* Batch size B */\n"
"                 const int S_dim,      /* Sequence length S */\n"
"                 const int D_dim,      /* Embedding dimension D */\n"
"                 const int V_dim,      /* Vocabulary size V */\n"
"                 __local REDUCE_ACCUM_TYPE* local_sums /* Local memory buffer, size = REDUCE_WG_SIZE */\n"
"                 ) {\n"
"\n"
"    /* --- Work-item / Work-group IDs --- */\n"
"    /* Each work-group computes one element delta_dw[v_out, d_out] */\n"
"    /* Kernel is launched with 1D GWS = V * D (number of groups) * LWS */\n"
"    size_t group_id = get_group_id(0); /* Group ID maps conceptually to output element index (0 to V*D - 1) */\n"
"    int tid = get_local_id(0);       /* Local thread ID within the work-group (0 to WGS-1) */\n"
"    int wg_size = get_local_size(0); /* Work-group size (REDUCE_WG_SIZE) */\n"
"\n"
"    /* Decompose linear group ID into the target vocabulary (v) and dimension (d) indices */\n"
"    int v_out = group_id / D_dim;\n"
"    int d_out = group_id % D_dim;\n"
"\n"
"    /* --- Bounds Check for the Group --- */\n"
"    if (v_out >= V_dim || d_out >= D_dim) {\n"
"        local_sums[tid] = REDUCE_ACCUM_CONST(0.0);\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        return;\n"
"    }\n"
"\n"
"    /* Total number of (b, s) pairs to potentially check */\n"
"    size_t items_to_reduce = (size_t)B_dim * S_dim;\n"
"    /* Accumulator for this thread's partial sum */\n"
"    REDUCE_ACCUM_TYPE thread_sum = REDUCE_ACCUM_CONST(0.0);\n"
"\n"
"    /* --- Grid-Stride Loop for Initial Summation --- */\n"
"    for (size_t i = tid; i < items_to_reduce; i += wg_size) {\n"
"        int b = i / S_dim;\n"
"        int s = i % S_dim;\n"
"        size_t indices_idx = (size_t)b * S_dim + s;\n"
"        int current_vocab_idx = indices[indices_idx];\n"
"\n"
"        /* --- Check if this (b, s) contributes to the target v_out --- */\n"
"        if (current_vocab_idx == v_out) {\n"
"            size_t grad_output_idx = ((size_t)b * S_dim + s) * D_dim + d_out;\n"
"            thread_sum += (REDUCE_ACCUM_TYPE)grad_output[grad_output_idx];\n"
"        }\n"
"    } /* End of loop over items_to_reduce */\n"
"\n"
"    local_sums[tid] = thread_sum;\n"
"\n"
"    /* --- Work-Group Reduction using Local Memory --- */\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int offset = wg_size / 2; offset > 0; offset /= 2) {\n"
"        if (tid < offset) {\n"
"            local_sums[tid] += local_sums[tid + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    /* --- Write Final Result --- */\n"
"    if (tid == 0) {\n"
"        size_t delta_dw_idx = (size_t)v_out * D_dim + d_out;\n"
"        delta_dw[delta_dw_idx] = (FP_TYPE)local_sums[0];\n"
"    }\n"
"}";
// Reduce Sum (Axis 0 and 1 for Bias Gradient)
const char *reduce_sum_kernel_src =
"/* Enable extensions if needed for local memory atomics (though not used here) */\n"
"#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n"
"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable /* For ACCUM_TYPE if needed */\n"
"\n"
"/* Define work-group size for reduction (can be tuned) */\n"
"#ifndef WORK_GROUP_SIZE_REDUCE\n"
"#define WORK_GROUP_SIZE_REDUCE 256\n"
"#endif\n"
"\n"
"#ifdef CL_HAS_FP64\n"
"    typedef double REDUCE_ACCUM_TYPE;\n"
"    #define REDUCE_ACCUM_CONST(x) (double)(x)\n"
"#else\n"
"    typedef float REDUCE_ACCUM_TYPE;\n"
"    #define REDUCE_ACCUM_CONST(x) (float)(x)\n"
"#endif\n"
"\n"
"/* Performs reduction sum over axes 0 (B) and 1 (M) of a 3D tensor (B, M, N). */\n"
"/* Output is a 1D tensor of size N. */\n"
"/* Uses local memory for efficient work-group reduction. */\n"
"__kernel void reduce_sum_axis01(\n"
"                __global const FP_TYPE* input, /* Input tensor (B, M, N) */\n"
"                __global FP_TYPE* output,      /* Output tensor (N) */\n"
"                const int B, const int M, const int N,\n"
"                __local REDUCE_ACCUM_TYPE* local_sums    /* Local memory buffer, size = WORK_GROUP_SIZE_REDUCE */\n"
"                ) {\n"
"    /* --- Work-item / Work-group IDs --- */\n"
"    int n_out_idx = get_group_id(0); /* Index for the output element N this group calculates (0 to N-1) */\n"
"    int tid = get_local_id(0);       /* Local thread ID within the work-group (0 to WGS-1) */\n"
"    int wg_size = get_local_size(0); /* Work-group size (WORK_GROUP_SIZE_REDUCE) */\n"
"\n"
"    /* Total number of elements to sum over per output element n_out_idx (B * M) */\n"
"    size_t items_to_reduce = (size_t)B * M;\n"
"    /* Accumulator for this thread's partial sum */\n"
"    REDUCE_ACCUM_TYPE thread_sum = REDUCE_ACCUM_CONST(0.0);\n"
"\n"
"    /* --- Grid-Stride Loop for Initial Summation --- */\n"
"    if (n_out_idx < N) { /* Ensure the group works on a valid output index */\n"
"        for (size_t i = tid; i < items_to_reduce; i += wg_size) {\n"
"            int b = i / M;\n"
"            int m = i % M;\n"
"            size_t input_idx = (size_t)b * M * N + (size_t)m * N + n_out_idx;\n"
"            thread_sum += (REDUCE_ACCUM_TYPE)input[input_idx];\n"
"        }\n"
"    }\n"
"    local_sums[tid] = thread_sum;\n"
"\n"
"    /* --- Work-Group Reduction using Local Memory --- */\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int offset = wg_size / 2; offset > 0; offset /= 2) {\n"
"        if (tid < offset) { /* Only threads in the first half of the current range add */\n"
"            local_sums[tid] += local_sums[tid + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    /* --- Write Final Result --- */\n"
"    if (tid == 0 && n_out_idx < N) { /* Check group index validity again before writing */\n"
"        output[n_out_idx] = (FP_TYPE)local_sums[0]; /* Cast back to output type */\n"
"    }\n"
"}";
// Broadcast Add (General Bias Vector - 3D + 1D)
const char *broadcast_add_kernel_src =
"/* Performs broadcast addition: C[b, m, n] = A[b, m, n] + B_bias[n] */\n"
"__kernel void broadcast_add_bias(\n"
"                __global const FP_TYPE* a,     /* Input tensor A (B, M, N) */\n"
"                __global const FP_TYPE* b_bias,/* Input bias vector B (N) */\n"
"                __global FP_TYPE* c,           /* Output tensor C (B, M, N) */\n"
"                const int M, const int N      /* Dimensions M and N (B is implicit from GWS dim 2) */\n"
"                ) {\n"
"    int n = get_global_id(0); /* Index along dimension N (0 to N-1) */\n"
"    int m = get_global_id(1); /* Index along dimension M (0 to M-1) */\n"
"    int b = get_global_id(2); /* Index along dimension B (0 to B-1) */\n"
"\n"
"    if (n < N && m < M) {\n"
"       size_t idx_a_c = (size_t)b * M * N + (size_t)m * N + n;\n"
"       int idx_b = n;\n"
"       c[idx_a_c] = a[idx_a_c] + b_bias[idx_b];\n"
"    }\n"
"}";
// Bias Addition Kernel (Matrix[M, N] + Vector[N])
const char *add_bias_mn_kernel_src =
"/* Performs broadcast addition: C[m, n] = A[m, n] + B_bias[n] */\n"
"/* Assumes A and C have shape (M, N), B_bias has shape (N) */\n"
"__kernel void add_bias_mn(\n"
"                __global const FP_TYPE* a,     /* Input tensor A (M, N) */\n"
"                __global const FP_TYPE* b_bias,/* Input bias vector B (N) */\n"
"                __global FP_TYPE* c,           /* Output tensor C (M, N) */\n"
"                const int M, const int N      /* Dimensions M and N */\n"
"                ) {\n"
"    int n = get_global_id(0); /* Index along dimension N (0 to N-1) */\n"
"    int m = get_global_id(1); /* Index along dimension M (0 to M-1) */\n"
"\n"
"    if (n < N && m < M) {\n"
"       size_t idx_ac = (size_t)m * N + n;\n"
"       int idx_b = n;\n"
"       c[idx_ac] = a[idx_ac] + b_bias[idx_b];\n"
"    }\n"
"}";
// Transpose Last Two Dimensions (Batched)
const char *transpose_batched_kernel_src =
"/* Transposes the last two dimensions of a tensor: (..., D1, D2) -> (..., D2, D1) */\n"
"__kernel void transpose_batched_last_two(\n"
"                __global const FP_TYPE* input, /* Input tensor (..., D1, D2) */\n"
"                __global FP_TYPE* output,      /* Output tensor (..., D2, D1) */\n"
"                const int Dim1,           /* Size of the dimension originally at -2 */\n"
"                const int Dim2            /* Size of the dimension originally at -1 */\n"
"                /* Leading dimensions (...) are flattened into GWS dim 2 (b_linear) */\n"
"                ) {\n"
"    int d1_out = get_global_id(0); /* Index along the new Dim1 (output dim -2, size Dim2) */\n"
"    int d2_out = get_global_id(1); /* Index along the new Dim2 (output dim -1, size Dim1) */\n"
"    int b_linear = get_global_id(2); /* Linearized index for the leading batch dimensions */\n"
"\n"
"    int d1_in = d2_out; /* Input dim1 index maps from output dim2 index */\n"
"    int d2_in = d1_out; /* Input dim2 index maps from output dim1 index */\n"
"\n"
"    if (d1_out < Dim2 && d2_out < Dim1) {\n"
"        size_t slice_stride = (size_t)Dim1 * Dim2;\n"
"        size_t batch_offset = (size_t)b_linear * slice_stride;\n"
"        size_t input_idx  = batch_offset + (size_t)d1_in * Dim2 + d2_in;\n"
"        size_t output_idx = batch_offset + (size_t)d1_out * Dim1 + d2_out; /* Stride is Dim1 now */\n"
"        output[output_idx] = input[input_idx];\n"
"    }\n"
"}";
// Transpose Dimensions 1 and 2 (Batched, 4D)
const char *transpose_12_batched_kernel_src =
"/* Transposes dimensions 1 and 2 of a 4D tensor: (B, D1, D2, D3) -> (B, D2, D1, D3) */\n"
"__kernel void transpose_12_batched(\n"
"                __global const FP_TYPE* input,  /* Input tensor (B, D1, D2, D3) */\n"
"                __global FP_TYPE* output, /* Output tensor (B, D2, D1, D3) */\n"
"                const int B, const int D1, const int D2, const int D3\n"
"                ) {\n"
"    int d3_idx = get_global_id(0);\n"
"    int d1_out_idx = get_global_id(1);\n"
"    int d2_b_linear = get_global_id(2);\n"
"\n"
"    int d2_out_idx = d2_b_linear % D2;\n"
"    int b_idx      = d2_b_linear / D2;\n"
"\n"
"    if (b_idx < B && d1_out_idx < D1 && d2_out_idx < D2 && d3_idx < D3) {\n"
"         int d1_in_idx = d1_out_idx;\n"
"         int d2_in_idx = d2_out_idx;\n"
"         size_t input_idx = (size_t)b_idx * D1 * D2 * D3 + \n"
"                           (size_t)d1_in_idx * D2 * D3 +  \n"
"                           (size_t)d2_in_idx * D3 +       \n"
"                           d3_idx;\n"
"         size_t output_idx = (size_t)b_idx * D2 * D1 * D3 + \n"
"                            (size_t)d2_out_idx * D1 * D3 + \n"
"                            (size_t)d1_out_idx * D3 +    \n"
"                            d3_idx;\n"
"         output[output_idx] = input[input_idx];\n"
"    }\n"
"}";
// Matmul (Batched, 3D @ 3D)
const char *matmul_batched_kernel_src =
"/* Performs batched matrix multiplication: C[b,:,:] = A[b,:,:] @ B[b,:,:] */\n"
"__kernel void matmul_batched(__global const FP_TYPE *a, /* Input A (B, M, K) */\n"
"                           __global const FP_TYPE *b, /* Input B (B, K, N) */\n"
"                           __global FP_TYPE *c, /* Output C (B, M, N) */\n"
"                           const int B, const int M, const int N, const int K) {\n"
"    int col = get_global_id(0); /* Index along N dimension (0 to N-1) */\n"
"    int row = get_global_id(1); /* Index along M dimension (0 to M-1) */\n"
"    int batch_idx = get_global_id(2); /* Index along B dimension (0 to B-1) */\n"
"\n"
"    if (batch_idx < B && row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        size_t a_batch_offset = (size_t)batch_idx * M * K;\n"
"        size_t b_batch_offset = (size_t)batch_idx * K * N;\n"
"        size_t c_batch_offset = (size_t)batch_idx * M * N;\n"
"\n"
"        for (int k = 0; k < K; ++k) {\n"
"             sum += (float)a[a_batch_offset + row * K + k] * (float)b[b_batch_offset + k * N + col];\n"
"        }\n"
"        c[c_batch_offset + row * N + col] = (FP_TYPE)sum;\n"
"    }\n"
"}";
// Matmul Backward dA (Batched)
const char *matmul_batched_backward_dA_kernel_src =
"/* dA[b,m,k] = sum_n dC[b,m,n] * B[b,k,n] (equivalent to dC @ B^T, batched) */\n"
"__kernel void matmul_batched_backward_da(__global const FP_TYPE *dC, /* Gradient dC (B, M, N) */\n"
"                                       __global const FP_TYPE *B,  /* Original Input B (B, K, N) */\n"
"                                       __global FP_TYPE *dA, /* Output Gradient dA (B, M, K) */\n"
"                                       const int B_dim, const int M_dim, const int N_dim, const int K_dim) {\n"
"    int k = get_global_id(0); /* Index along K dimension (0 to K_dim-1) */\n"
"    int m = get_global_id(1); /* Index along M dimension (0 to M_dim-1) */\n"
"    int b = get_global_id(2); /* Index along B dimension (0 to B_dim-1) */\n"
"\n"
"    if (b < B_dim && m < M_dim && k < K_dim) {\n"
"        float gradient_sum = 0.0f;\n"
"        size_t dc_batch_offset = (size_t)b * M_dim * N_dim;\n"
"        size_t b_batch_offset  = (size_t)b * K_dim * N_dim;\n"
"        size_t da_batch_offset = (size_t)b * M_dim * K_dim;\n"
"\n"
"        for (int n = 0; n < N_dim; ++n) {\n"
"            gradient_sum += (float)dC[dc_batch_offset + m * N_dim + n] * (float)B[b_batch_offset + k * N_dim + n];\n"
"        }\n"
"        dA[da_batch_offset + m * K_dim + k] = (FP_TYPE)gradient_sum;\n"
"    }\n"
"}";
// Matmul Backward dB (Batched)
const char *matmul_batched_backward_dB_kernel_src =
"/* dB[b,k,n] = sum_m A[b,m,k] * dC[b,m,n] (equivalent to A^T @ dC, batched) */\n"
"__kernel void matmul_batched_backward_db(__global const FP_TYPE *A,  /* Original Input A (B, M, K) */\n"
"                                       __global const FP_TYPE *dC, /* Gradient dC (B, M, N) */\n"
"                                       __global FP_TYPE *dB, /* Output Gradient dB (B, K, N) */\n"
"                                       const int B_dim, const int M_dim, const int N_dim, const int K_dim) {\n"
"    int n = get_global_id(0); /* Index along N dimension (0 to N_dim-1) */\n"
"    int k = get_global_id(1); /* Index along K dimension (0 to K_dim-1) */\n"
"    int b = get_global_id(2); /* Index along B dimension (0 to B_dim-1) */\n"
"\n"
"    if (b < B_dim && k < K_dim && n < N_dim) {\n"
"        float gradient_sum = 0.0f;\n"
"        size_t a_batch_offset  = (size_t)b * M_dim * K_dim;\n"
"        size_t dc_batch_offset = (size_t)b * M_dim * N_dim;\n"
"        size_t db_batch_offset = (size_t)b * K_dim * N_dim;\n"
"\n"
"        for (int m = 0; m < M_dim; ++m) {\n"
"            gradient_sum += (float)A[a_batch_offset + m * K_dim + k] * (float)dC[dc_batch_offset + m * N_dim + n];\n"
"        }\n"
"        dB[db_batch_offset + k * N_dim + n] = (FP_TYPE)gradient_sum;\n"
"    }\n"
"}";
// Broadcast Add for Positional Encoding
const char *add_broadcast_pe_kernel_src =
"/* Performs broadcast addition: Output[b, s, e] = Input[b, s, e] + PE[s, e] */\n"
"__kernel void add_broadcast_pe(\n"
"                __global const FP_TYPE* input,  /* Input tensor (B, S, E) */\n"
"                __global const FP_TYPE* pe,     /* Positional Encoding tensor (S, E) - Slice matching S */\n"
"                __global FP_TYPE* output, /* Output tensor (B, S, E) */\n"
"                const int S, const int E        /* Dimensions S and E (B is implicit from GWS dim 2) */\n"
"                ) {\n"
"    int e = get_global_id(0); /* Index along dimension E (0 to E-1) */\n"
"    int s = get_global_id(1); /* Index along dimension S (0 to S-1) */\n"
"    int b = get_global_id(2); /* Index along dimension B (0 to B-1) */\n"
"\n"
"    if (s < S && e < E) {\n"
"       size_t idx_bse = (size_t)b * S * E + (size_t)s * E + e;\n"
"       size_t idx_pe = (size_t)s * E + e;\n"
"       output[idx_bse] = input[idx_bse] + pe[idx_pe];\n"
"    }\n"
"}";
// Hebbian Update (Local Reduction, No Atomics)
const char *hebbian_update_local_reduce_kernel_src =
"/* Define work-group size for reduction (can be tuned) */\n"
"#ifndef REDUCE_WG_SIZE\n"
"#define REDUCE_WG_SIZE 256\n"
"#endif\n"
"\n"
"/* Define accumulation type based on FP64 support */\n"
"#ifdef CL_HAS_FP64\n"
"    typedef double REDUCE_ACCUM_TYPE;\n"
"    #define REDUCE_ACCUM_CONST(x) (double)(x)\n"
"#else\n"
"    typedef float REDUCE_ACCUM_TYPE;\n"
"    #define REDUCE_ACCUM_CONST(x) (float)(x)\n"
"#endif\n"
"\n"
"__kernel void hebbian_update_local_reduce(\n"
"                                __global const FP_TYPE *A,  /* Pre-synaptic activations (B, M, K) */\n"
"                                __global const FP_TYPE *C,  /* Post-synaptic activations (B, M, N) */\n"
"                                __global FP_TYPE *W,        /* Weights to be updated (K, N) */\n"
"                                const float learning_rate,\n"
"                                const int B_dim, const int M_dim, const int N_dim, const int K_dim,\n"
"                                __local REDUCE_ACCUM_TYPE* local_sums /* Local memory buffer, size = REDUCE_WG_SIZE */\n"
"                                ) {\n"
"    size_t group_id = get_group_id(0);\n"
"    int tid = get_local_id(0);\n"
"    int wg_size = get_local_size(0);\n"
"\n"
"    int k_out = group_id / N_dim;\n"
"    int n_out = group_id % N_dim;\n"
"\n"
"    if (k_out >= K_dim || n_out >= N_dim) {\n"
"        local_sums[tid] = REDUCE_ACCUM_CONST(0.0);\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        return;\n"
"    }\n"
"\n"
"    size_t items_to_reduce = (size_t)B_dim * M_dim;\n"
"    REDUCE_ACCUM_TYPE thread_sum = REDUCE_ACCUM_CONST(0.0);\n"
"\n"
"    for (size_t i = tid; i < items_to_reduce; i += wg_size) {\n"
"        int b = i / M_dim;\n"
"        int m = i % M_dim;\n"
"        size_t a_idx = (size_t)b * M_dim * K_dim + (size_t)m * K_dim + k_out;\n"
"        size_t c_idx = (size_t)b * M_dim * N_dim + (size_t)m * N_dim + n_out;\n"
"        thread_sum += (REDUCE_ACCUM_TYPE)A[a_idx] * (REDUCE_ACCUM_TYPE)C[c_idx];\n"
"    }\n"
"\n"
"    local_sums[tid] = thread_sum;\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int offset = wg_size / 2; offset > 0; offset /= 2) {\n"
"        if (tid < offset) {\n"
"            local_sums[tid] += local_sums[tid + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    if (tid == 0) {\n"
"        size_t w_idx = (size_t)k_out * N_dim + n_out;\n"
"        W[w_idx] += (FP_TYPE)(learning_rate * local_sums[0]);\n"
"    }\n"
"}";
// Threshold Spike Generation
const char *threshold_spike_kernel_src =
"__kernel void threshold_spike( __global const FP_TYPE *activations,\n"
"                               __global FP_TYPE *spikes, /* Output: 0.0f or 1.0f */\n"
"                               const float threshold,\n"
"                               const int num_elements) {\n"
"    int idx = get_global_id(0); /* Global element index */\n"
"\n"
"    if (idx < num_elements) {\n"
"        spikes[idx] = (activations[idx] > threshold) ? (FP_TYPE)1.0f : (FP_TYPE)0.0f;\n"
"    }\n"
"}";
// Dynamic Token Assignment: Find closest prototype (max dot product)
const char *dynamic_token_assign_kernel_src =
"#ifndef HUGE_VALF\n"
"#define HUGE_VALF (__builtin_huge_valf())\n"
"#endif\n"
"\n"
"/* Assigns each input activation vector to the index of the prototype with the highest dot product similarity. */\n"
"__kernel void dynamic_token_assignment(\n"
"                            __global const FP_TYPE *activations, /* Input activations (B, S, E) flattened */\n"
"                            __global const FP_TYPE *prototypes,  /* Token prototypes (T, E) flattened */\n"
"                            __global int *output_indices,      /* Output token indices (B, S) flattened */\n"
"                            const int S, /* Sequence length */\n"
"                            const int E, /* Embedding dimension */\n"
"                            const int T  /* Number of token prototypes */\n"
"                            /* B is implicit via GWS dim 1 */\n"
"                            ) {\n"
"    int s = get_global_id(0); /* Sequence dimension index (0 to S-1) */\n"
"    int b = get_global_id(1); /* Batch dimension index (0 to B-1) */\n"
"\n"
"    size_t activation_offset = ((size_t)b * S + s) * E; /* Offset for activations[b, s, :] */\n"
"    size_t output_idx = (size_t)b * S + s;              /* Offset for output_indices[b, s] */\n"
"\n"
"    float max_similarity = -HUGE_VALF;\n"
"    int best_token_idx = -1; /* Initialize with invalid index */\n"
"\n"
"    /* Iterate through all token prototypes */\n"
"    for (int t = 0; t < T; ++t) {\n"
"        size_t prototype_offset = (size_t)t * E; /* Offset for prototypes[t, :] */\n"
"        float current_similarity = 0.0f;\n"
"\n"
"        /* Compute dot product between activation and prototype */\n"
"        for (int e = 0; e < E; ++e) {\n"
"            current_similarity += activations[activation_offset + e] * prototypes[prototype_offset + e];\n"
"        }\n"
"\n"
"        /* Update best match if current similarity is higher */\n"
"        if (current_similarity > max_similarity) {\n"
"            max_similarity = current_similarity;\n"
"            best_token_idx = t;\n"
"        }\n"
"    }\n"
"\n"
"    /* Write the index of the best matching prototype */\n"
"    output_indices[output_idx] = best_token_idx;\n"
"}";
// Pairwise Similarity (Dot Product)
const char *pairwise_similarity_kernel_src =
"/* Computes the pairwise dot product similarity matrix for a set of state vectors. */\n"
"__kernel void pairwise_similarity_dot(\n"
"                            __global const FP_TYPE *states, /* Input state vectors (N, D) flattened */\n"
"                            __global FP_TYPE *similarity,   /* Output similarity matrix (N, N) flattened */\n"
"                            const int N, /* Number of state vectors */\n"
"                            const int D  /* Dimension of state vectors */\n"
"                            ) {\n"
"    int i = get_global_id(0); /* Row index for the similarity matrix (0 to N-1) */\n"
"    int j = get_global_id(1); /* Column index for the similarity matrix (0 to N-1) */\n"
"\n"
"    if (i < N && j < N) {\n"
"        size_t state_i_offset = (size_t)i * D;\n"
"        size_t state_j_offset = (size_t)j * D;\n"
"        size_t output_idx = (size_t)i * N + j;\n"
"\n"
"        float dot_product = 0.0f;\n"
"        for (int d = 0; d < D; ++d) {\n"
"            dot_product += states[state_i_offset + d] * states[state_j_offset + d];\n"
"        }\n"
"        similarity[output_idx] = (FP_TYPE)dot_product;\n"
"    }\n"
"}";
// GPU Prototype Update Kernel Sources
const char *proto_segmented_sum_atomic_kernel_src =
"/* This kernel requires the cl_khr_global_int32_base_atomics extension */\n"
"/* The CL_HAS_ATOMICS define MUST be passed by the host if the extension is supported */\n"
"#ifdef CL_HAS_ATOMICS\n"
"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
"/* Required for atomic_add_float */\n"
"#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable /* Often needed for cmpxchg on 64-bit */\n"
"#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n"
"\n"
"/* Performs atomic add for floats using compare-and-swap (cmpxchg). */\n"
"inline void atomic_add_float(__global float *addr, float val) {\n"
"    union {\n"
"        unsigned int u32;\n"
"        float f32;\n"
"    } next, expected, current;\n"
"    /* Cast the float pointer to a global pointer to unsigned int for atom_cmpxchg */\n"
"    __global unsigned int *u_addr = (__global unsigned int *)addr;\n"
"    current.f32 = *addr; // Read current value non-atomically (initial guess)\n"
"    do {\n"
"        expected.f32 = current.f32; // Expected value for cmpxchg\n"
"        next.f32 = expected.f32 + val; // Calculate the desired new value\n"
"        // Atomically compare the value at u_addr with expected.u32.\n"
"        // If they match, replace the value at u_addr with next.u32.\n"
"        // Return the *original* value that was at u_addr before the attempt.\n"
"        current.u32 = atom_cmpxchg(u_addr, expected.u32, next.u32);\n"
"    } while (current.u32 != expected.u32); // Loop if the value was changed by another thread between read and write\n"
"}\n"
"\n"
"/* Sums activations belonging to the same prototype index atomically. */\n"
"__kernel void proto_segmented_sum_atomic(\n"
"        __global const FP_TYPE* activations_flat, /* Flattened activations (M_flat * E) */\n"
"        __global const int* indices_flat,       /* Flattened indices (M_flat) mapping activations to prototypes */\n"
"        __global FP_TYPE* proto_sums,           /* Output sums per prototype (T * E), MUST be zero-initialized by host */\n"
"        __global int* proto_counts,             /* Output counts per prototype (T), MUST be zero-initialized by host */\n"
"        const int M_flat, /* Total number of activation vectors (e.g., B * S) */\n"
"        const int E,      /* Embedding dimension */\n"
"        const int T       /* Number of prototypes */\n"
"        ) {\n"
"    int idx = get_global_id(0); // Global index iterates through all activation vectors\n"
"\n"
"    // Check if this work-item is within the bounds of the activation vectors\n"
"    if (idx < M_flat) {\n"
"        // Get the prototype index assigned to this activation vector\n"
"        int proto_idx = indices_flat[idx];\n"
"\n"
"        // Ensure the prototype index is valid\n"
"        if (proto_idx >= 0 && proto_idx < T) {\n"
"            // Atomically increment the counter for this prototype\n"
"            atom_inc(&proto_counts[proto_idx]);\n"
"\n"
"            // Calculate the offset for the current activation vector's data\n"
"            size_t activation_offset = (size_t)idx * E;\n"
"            // Calculate the base offset for the target prototype's sum data\n"
"            size_t sum_offset = (size_t)proto_idx * E;\n"
"\n"
"            // Iterate through the embedding dimension\n"
"            for (int e = 0; e < E; ++e) {\n"
"                // Atomically add the activation component to the corresponding prototype sum\n"
"                atomic_add_float(&proto_sums[sum_offset + e], activations_flat[activation_offset + e]);\n"
"            }\n"
"        }\n"
"        // Ignore activations assigned to invalid prototype indices (e.g., -1 for padding)\n"
"    }\n"
"}\n"
"#else\n"
"/* If atomics are NOT supported, provide a dummy kernel to avoid compile errors, */\n"
"/* but this kernel will do nothing. The host should prevent its execution. */\n"
"__kernel void proto_segmented_sum_atomic(\n"
"        __global const FP_TYPE* activations_flat,\n"
"        __global const int* indices_flat,\n"
"        __global FP_TYPE* proto_sums,\n"
"        __global int* proto_counts,\n"
"        const int M_flat, const int E, const int T) {\n"
"        /* Atomic operations not supported or enabled. This kernel does nothing. */\n"
"        /* Host code should have checked has_atomics_support before enqueuing. */\n"
"}\n"
"#endif\n";
const char *proto_update_step_kernel_src =
"/* Updates prototypes using the accumulated sums and counts */\n"
"__kernel void proto_update_step(\n"
"        __global FP_TYPE* prototypes,     /* Prototypes to be updated (T * E) */\n"
"        __global const FP_TYPE* proto_sums, /* Input sums per prototype (T * E) from segmented_sum */\n"
"        __global const int* proto_counts,   /* Input counts per prototype (T) from segmented_sum */\n"
"        const float learning_rate,        /* Learning rate (alpha) for the update */\n"
"        const int E,                      /* Embedding dimension */\n"
"        const int T                       /* Number of prototypes */\n"
"        ) {\n"
"    // Global ID corresponds to the prototype index\n"
"    int t = get_global_id(0);\n"
"\n"
"    // Check if this work-item is within the bounds of the prototypes\n"
"    if (t < T) {\n"
"        // Get the number of activations assigned to this prototype\n"
"        int count = proto_counts[t];\n"
"\n"
"        // Only update prototypes that received at least one activation vector\n"
"        if (count > 0) {\n"
"            // Calculate base offset for this prototype's data\n"
"            size_t base_offset = (size_t)t * E;\n"
"            // Calculate inverse count for averaging\n"
"            float inv_count = 1.0f / (float)count;\n"
"            // Precompute learning rate factors\n"
"            float lr = learning_rate;\n"
"            float one_minus_lr = 1.0f - lr;\n"
"\n"
"            // Iterate through the embedding dimension\n"
"            for (int e = 0; e < E; ++e) {\n"
"                // Calculate the index for the current dimension\n"
"                size_t current_idx = base_offset + e;\n"
"                // Read the current prototype value\n"
"                float old_proto = prototypes[current_idx];\n"
"                // Calculate the mean activation value for this dimension\n"
"                float mean_activation = proto_sums[current_idx] * inv_count;\n"
"                // Apply the exponential moving average update rule:\n"
"                // new_proto = (1 - lr) * old_proto + lr * mean_activation\n"
"                prototypes[current_idx] = one_minus_lr * old_proto + lr * mean_activation;\n"
"            }\n"
"        }\n"
"        // Prototypes with count == 0 remain unchanged.\n"
"    }\n"
"}";
// Loss Shaping Kernel (Single Pair - Original)
const char *shape_loss_reward_penalty_kernel_src =
"/* Applies reward/penalty adjustments to pre-calculated loss values. */\n"
"/* Assumes 'predictions' buffer contains probabilities (output of softmax). */\n"
"__kernel void shape_loss_reward_penalty(\n"
"        __global const FP_TYPE* loss_in,           /* Input: Original loss per sample (num_samples) */\n"
"        __global const FP_TYPE* predictions,       /* Input: Model prediction probabilities (num_samples, num_classes) */\n"
"        __global const int* targets,             /* Input: Target class indices (num_samples) */\n"
"        __global FP_TYPE* loss_out,          /* Output: Shaped loss per sample (num_samples) */\n"
"        const int num_samples,             /* Total number of samples/tokens */\n"
"        const int num_classes,             /* Number of output classes (V) */\n"
"        const float penalty_weight,        /* Amount to ADD to loss for critical error */\n"
"        const float reward_weight,         /* Amount to SUBTRACT from loss for high-confidence correct prediction */\n"
"        const float high_confidence_threshold, /* Probability threshold for reward */\n"
"        const int critical_target_class,   /* Target class index for penalty check */\n"
"        const int critical_predicted_class /* Predicted class index for penalty check */\n"
"        )\n"
"{\n"
"    int idx = get_global_id(0); /* Global index for the sample/token */\n"
"\n"
"    if (idx < num_samples)\n"
"    {\n"
"        FP_TYPE current_loss = loss_in[idx];\n"
"        int target_label = targets[idx];\n"
"\n"
"        /* Handle padding/invalid target labels: Do not apply reward/penalty */\n"
"        if (target_label < 0 || target_label >= num_classes) {\n"
"            loss_out[idx] = current_loss;\n"
"            return;\n"
"        }\n"
"\n"
"        /* Find predicted class and its probability, and probability of correct class */\n"
"        size_t pred_offset = (size_t)idx * num_classes;\n"
"        int predicted_label = 0;\n"
"        FP_TYPE max_prob = -1.0f;\n"
"        for (int v = 0; v < num_classes; ++v) {\n"
"            FP_TYPE prob = predictions[pred_offset + v];\n"
"            if (prob > max_prob) {\n"
"                max_prob = prob;\n"
"                predicted_label = v;\n"
"            }\n"
"        }\n"
"        FP_TYPE correct_class_prob = predictions[pred_offset + target_label];\n"
"\n"
"        /* Calculate adjustment */\n"
"        float adjustment = 0.0f;\n"
"\n"
"        /* Penalty Logic */\n"
"        bool is_critical_error = (target_label == critical_target_class) && (predicted_label == critical_predicted_class);\n"
"        if (is_critical_error) {\n"
"            adjustment += penalty_weight;\n"
"        }\n"
"\n"
"        /* Reward Logic */\n"
"        bool is_correct = (predicted_label == target_label);\n"
"        bool is_high_confidence = (correct_class_prob >= high_confidence_threshold);\n"
"        if (is_correct && is_high_confidence) {\n"
"            adjustment -= reward_weight;\n"
"        }\n"
"\n"
"        /* Apply adjustment to the original loss */\n"
"        loss_out[idx] = current_loss + (FP_TYPE)adjustment;\n"
"    }\n"
"}";

// --- NEU: Loss Shaping Kernel (mit Liste) ---
const char *shape_loss_reward_penalty_list_kernel_src =
"/* Applies reward/penalty adjustments based on a list of critical pairs. */\n"
"/* Assumes 'predictions' buffer contains probabilities (output of softmax). */\n"
"__kernel void shape_loss_reward_penalty_list(\n"
"        __global const FP_TYPE* loss_in,           /* Input: Original loss per sample (num_samples) */\n"
"        __global const FP_TYPE* predictions,       /* Input: Model prediction probabilities (num_samples, num_classes) */\n"
"        __global const int* targets,             /* Input: Target class indices (num_samples) */\n"
"        __global FP_TYPE* loss_out,          /* Output: Shaped loss per sample (num_samples) */\n"
"        __global const int* critical_pairs,      /* Input: List of [target_id, predicted_id] pairs flattened (num_critical_pairs * 2) */\n"
"        const int num_samples,             /* Total number of samples/tokens */\n"
"        const int num_classes,             /* Number of output classes (V) */\n"
"        const int num_critical_pairs,      /* Number of critical pairs in the list */\n"
"        const float penalty_weight,        /* Amount to ADD to loss for critical error */\n"
"        const float reward_weight,         /* Amount to SUBTRACT from loss for high-confidence correct prediction */\n"
"        const float high_confidence_threshold /* Probability threshold for reward */\n"
"        )\n"
"{\n"
"    int idx = get_global_id(0); /* Global index for the sample/token */\n"
"\n"
"    if (idx < num_samples)\n"
"    {\n"
"        FP_TYPE current_loss = loss_in[idx];\n"
"        int target_label = targets[idx];\n"
"\n"
"        /* Handle padding/invalid target labels: Do not apply reward/penalty */\n"
"        if (target_label < 0 || target_label >= num_classes) {\n"
"            loss_out[idx] = current_loss;\n"
"            return;\n"
"        }\n"
"\n"
"        /* Find predicted class and its probability, and probability of correct class */\n"
"        size_t pred_offset = (size_t)idx * num_classes;\n"
"        int predicted_label = 0;\n"
"        FP_TYPE max_prob = -1.0f;\n"
"        for (int v = 0; v < num_classes; ++v) {\n"
"            FP_TYPE prob = predictions[pred_offset + v];\n"
"            if (prob > max_prob) {\n"
"                max_prob = prob;\n"
"                predicted_label = v;\n"
"            }\n"
"        }\n"
"        FP_TYPE correct_class_prob = predictions[pred_offset + target_label];\n"
"\n"
"        /* Calculate adjustment */\n"
"        float adjustment = 0.0f;\n"
"\n"
"        /* --- NEU: Penalty Logic mit Liste --- */\n"
"        bool is_critical_error = false;\n"
"        // Durchlaufe die Liste der kritischen Paare\n"
"        if (num_critical_pairs > 0 && critical_pairs != 0) { // Check for non-empty list and valid pointer\n"
"            for (int i = 0; i < num_critical_pairs; ++i) {\n"
"                int crit_target = critical_pairs[i * 2 + 0]; // Target ist an geraden Indizes\n"
"                int crit_pred   = critical_pairs[i * 2 + 1]; // Predicted ist an ungeraden Indizes\n"
"                if ((target_label == crit_target) && (predicted_label == crit_pred)) {\n"
"                    is_critical_error = true;\n"
"                    break; // Ein Treffer reicht\n"
"                }\n"
"            }\n"
"        }\n"
"        if (is_critical_error) {\n"
"            adjustment += penalty_weight;\n"
"        }\n"
"        /* --- ENDE NEU --- */\n"
"\n"
"        /* Reward Logic (unverändert) */\n"
"        bool is_correct = (predicted_label == target_label);\n"
"        bool is_high_confidence = (correct_class_prob >= high_confidence_threshold);\n"
"        if (is_correct && is_high_confidence) {\n"
"            adjustment -= reward_weight;\n"
"        }\n"
"\n"
"        /* Apply adjustment to the original loss */\n"
"        loss_out[idx] = current_loss + (FP_TYPE)adjustment;\n"
"    }\n"
"}";

// ----------------------------------------------------------------------------------

// --- Helper Function Implementations ---

/**
 * @brief Returns a human-readable string for an OpenCL error code.
 * Maps standard OpenCL error codes (negative integers) to descriptive strings.
 * @param error The cl_int error code.
 * @return A constant C string describing the error. Returns "Unknown OpenCL error %d" if the code is not recognized.
 */
const char* clGetErrorString(cl_int error) {
     // Static map of error codes to strings (standard OpenCL errors)
    static const char *errStr[] = {
        "CL_SUCCESS", "CL_DEVICE_NOT_FOUND", "CL_DEVICE_NOT_AVAILABLE", "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE", "CL_MEM_COPY_OVERLAP", "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED", "CL_BUILD_PROGRAM_FAILURE", "CL_MAP_FAILURE",
        /* Placeholder for codes -13 to -29 */
        "CL_MISALIGNED_SUB_BUFFER_OFFSET", "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", "CL_COMPILE_PROGRAM_FAILURE",
        "CL_LINKER_NOT_AVAILABLE", "CL_LINK_PROGRAM_FAILURE", "CL_DEVICE_PARTITION_FAILED", "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
        "", "", "", "", "", "", "", "", "", "",
        "CL_INVALID_VALUE", "CL_INVALID_DEVICE_TYPE", "CL_INVALID_PLATFORM", "CL_INVALID_DEVICE", "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES", "CL_INVALID_COMMAND_QUEUE", "CL_INVALID_HOST_PTR", "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", "CL_INVALID_IMAGE_SIZE", "CL_INVALID_SAMPLER", "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS", "CL_INVALID_PROGRAM", "CL_INVALID_PROGRAM_EXECUTABLE", "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION", "CL_INVALID_KERNEL", "CL_INVALID_ARG_INDEX", "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE", "CL_INVALID_KERNEL_ARGS", "CL_INVALID_WORK_DIMENSION", "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE", "CL_INVALID_GLOBAL_OFFSET", "CL_INVALID_EVENT_WAIT_LIST", "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION", "CL_INVALID_GL_OBJECT", "CL_INVALID_BUFFER_SIZE", "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE", "CL_INVALID_PROPERTY", "CL_INVALID_IMAGE_DESCRIPTOR", "CL_INVALID_COMPILER_OPTIONS",
        "CL_INVALID_LINKER_OPTIONS", "CL_INVALID_DEVICE_PARTITION_COUNT",
        /* Add more specific error codes for newer OpenCL versions if needed */
        "CL_INVALID_PIPE_SIZE", "CL_INVALID_DEVICE_QUEUE" /* Examples for 2.0+ */
    };
    const int errCount = sizeof(errStr) / sizeof(errStr[0]);
    const int index = -error; /* Error codes are negative integers */

    /* Check if the index is within the bounds of our static map */
    if (index >= 0 && index < errCount) {
        const char* err = errStr[index];
        /* Return the string if it's valid and not empty */
        if (err && err[0] != '\0') {
             return err;
        }
    }
    /* If the error code is unknown or the string is empty, return a generic message */
    static char unknown_error[64];
    /* Use snprintf (C99) for better portability and safety */
    snprintf(unknown_error, sizeof(unknown_error), "Unknown OpenCL error %d", error);
    unknown_error[sizeof(unknown_error) - 1] = '\0'; /* Ensure null termination */
    return unknown_error;
}

/**
 * @brief Compiles an OpenCL kernel from source code.
 */
cl_int compile_opencl_kernel(const char* kernel_source, const char* kernel_name,
                             cl_program* program_out, cl_kernel* kernel_out) {
    cl_int err;
    size_t source_size;

    // Initialize output pointers
    *program_out = NULL;
    *kernel_out = NULL;

    if (!kernel_source) {
         fprintf(stderr, "[C] compile_opencl_kernel: Error - kernel_source is NULL for '%s'.\n", kernel_name ? kernel_name : "UNKNOWN");
         return CL_INVALID_VALUE;
    }
    source_size = strlen(kernel_source);

    if (!context || !device_id) {
        fprintf(stderr, "[C] compile_opencl_kernel: Error - No context or device available for compiling '%s'.\n", kernel_name ? kernel_name : "UNKNOWN");
        return CL_INVALID_CONTEXT; // Or CL_INVALID_DEVICE if more appropriate
    }

    // Create program object
    *program_out = clCreateProgramWithSource(context, 1, &kernel_source, &source_size, &err);
    if (!*program_out || err != CL_SUCCESS) {
        fprintf(stderr, "[C] compile_opencl_kernel: clCreateProgramWithSource failed for '%s': %s (%d)\n",
                kernel_name ? kernel_name : "UNKNOWN", clGetErrorString(err), err);
        return err;
    }

    // --- Construct Build Options ---
    char build_options[512];
    snprintf(build_options, sizeof(build_options),
             "-cl-std=CL1.2 -Werror -D FP_TYPE=%s %s %s -DFP_TYPE_SIZE=%zu", // Base options, use CL1.2
             KERNEL_FP_TYPE_STR,
             has_fp64_support ? "-D CL_HAS_FP64" : "",          // Define if FP64 is supported
             has_atomics_support ? "-D CL_HAS_ATOMICS" : "",    // Define if required KHR atomics are supported
             sizeof(FP_TYPE)                                    // Define FP_TYPE_SIZE
             );
    build_options[sizeof(build_options) - 1] = '\0'; // Ensure null termination

    // Build the program
    err = clBuildProgram(*program_out, 1, &device_id, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] compile_opencl_kernel: clBuildProgram failed for '%s' with options '%s': %s (%d)\n",
                kernel_name ? kernel_name : "UNKNOWN", build_options, clGetErrorString(err), err);

        // Get and print the build log
        size_t log_size = 0;
        clGetProgramBuildInfo(*program_out, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            if (log) {
                clGetProgramBuildInfo(*program_out, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                fprintf(stderr, "--- OpenCL Build Log (%s) ---\n%s\n-----------------------------\n", kernel_name ? kernel_name : "UNKNOWN", log);
                free(log);
            } else {
                fprintf(stderr, "[C] compile_opencl_kernel: Failed to allocate memory (%zu bytes) for build log.\n", log_size);
            }
        } else {
             fprintf(stderr, "[C] compile_opencl_kernel: Build log is empty or unavailable.\n");
        }

        // Cleanup partially created resources
        clReleaseProgram(*program_out);
        *program_out = NULL;
        return err;
    }

    // Create the kernel object
    *kernel_out = clCreateKernel(*program_out, kernel_name, &err);
    if (!*kernel_out || err != CL_SUCCESS) {
        fprintf(stderr, "[C] compile_opencl_kernel: clCreateKernel failed for '%s': %s (%d)\n",
                kernel_name ? kernel_name : "UNKNOWN", clGetErrorString(err), err);
        // Cleanup program if kernel creation fails
        clReleaseProgram(*program_out);
        *program_out = NULL;
        return err;
    }

    // Success
    return CL_SUCCESS;
}

/**
 * @brief Releases all allocated OpenCL resources.
 */
void shutdown_driver() {
    printf("[C] shutdown_driver: Starting OpenCL resource cleanup...\n");

    // Release all kernel objects
    #define RELEASE_KERNEL(k) if (k) { clReleaseKernel(k); k = NULL; }
    RELEASE_KERNEL(matmul_kernel);
    RELEASE_KERNEL(softmax_kernel);
    RELEASE_KERNEL(gelu_kernel);
    RELEASE_KERNEL(add_kernel);
    RELEASE_KERNEL(mul_kernel);
    RELEASE_KERNEL(layernorm_kernel);
    RELEASE_KERNEL(transpose_kernel);
    RELEASE_KERNEL(gelu_backward_kernel);
    RELEASE_KERNEL(matmul_backward_da_kernel);
    RELEASE_KERNEL(matmul_backward_db_kernel);
    RELEASE_KERNEL(layernorm_backward_kernel);
    RELEASE_KERNEL(adam_kernel);
    RELEASE_KERNEL(softmax_backward_kernel);
    RELEASE_KERNEL(mul_backward_kernel);
    RELEASE_KERNEL(transpose_backward_kernel);
    RELEASE_KERNEL(embedding_lookup_kernel);
    RELEASE_KERNEL(reduce_sum_kernel);
    RELEASE_KERNEL(broadcast_add_kernel);
    RELEASE_KERNEL(transpose_batched_kernel);
    RELEASE_KERNEL(transpose_12_batched_kernel);
    RELEASE_KERNEL(matmul_batched_kernel);
    RELEASE_KERNEL(matmul_batched_backward_da_kernel);
    RELEASE_KERNEL(matmul_batched_backward_db_kernel);
    RELEASE_KERNEL(log_softmax_kernel);
    RELEASE_KERNEL(cross_entropy_kernel);
    RELEASE_KERNEL(add_broadcast_pe_kernel);
    RELEASE_KERNEL(threshold_spike_kernel);
    RELEASE_KERNEL(add_bias_mn_kernel);
    RELEASE_KERNEL(dynamic_token_assign_kernel);
    RELEASE_KERNEL(pairwise_similarity_kernel);
    RELEASE_KERNEL(hebbian_update_local_reduce_kernel);
    RELEASE_KERNEL(embedding_backward_calc_delta_local_kernel);
    RELEASE_KERNEL(proto_segmented_sum_kernel);
    RELEASE_KERNEL(proto_update_step_kernel);
    RELEASE_KERNEL(shape_loss_reward_penalty_kernel);
    RELEASE_KERNEL(shape_loss_reward_penalty_list_kernel); // NEU
    #undef RELEASE_KERNEL
    printf("[C] shutdown_driver: Kernels released.\n");

    // Release all program objects
    #define RELEASE_PROGRAM(p) if (p) { clReleaseProgram(p); p = NULL; }
    RELEASE_PROGRAM(matmul_program);
    RELEASE_PROGRAM(softmax_program);
    RELEASE_PROGRAM(gelu_program);
    RELEASE_PROGRAM(add_program);
    RELEASE_PROGRAM(mul_program);
    RELEASE_PROGRAM(layernorm_program);
    RELEASE_PROGRAM(transpose_program);
    RELEASE_PROGRAM(gelu_backward_program);
    RELEASE_PROGRAM(matmul_backward_da_program);
    RELEASE_PROGRAM(matmul_backward_db_program);
    RELEASE_PROGRAM(layernorm_backward_program);
    RELEASE_PROGRAM(adam_program);
    RELEASE_PROGRAM(softmax_backward_program);
    RELEASE_PROGRAM(mul_backward_program);
    RELEASE_PROGRAM(transpose_backward_program);
    RELEASE_PROGRAM(embedding_lookup_program);
    RELEASE_PROGRAM(reduce_sum_program);
    RELEASE_PROGRAM(broadcast_add_program);
    RELEASE_PROGRAM(transpose_batched_program);
    RELEASE_PROGRAM(transpose_12_batched_program);
    RELEASE_PROGRAM(matmul_batched_program);
    RELEASE_PROGRAM(matmul_batched_backward_da_program);
    RELEASE_PROGRAM(matmul_batched_backward_db_program);
    RELEASE_PROGRAM(log_softmax_program);
    RELEASE_PROGRAM(cross_entropy_program);
    RELEASE_PROGRAM(add_broadcast_pe_program);
    RELEASE_PROGRAM(threshold_spike_program);
    RELEASE_PROGRAM(add_bias_mn_program);
    RELEASE_PROGRAM(dynamic_token_assign_program);
    RELEASE_PROGRAM(pairwise_similarity_program);
    RELEASE_PROGRAM(hebbian_update_local_reduce_program);
    RELEASE_PROGRAM(embedding_backward_calc_delta_local_program);
    RELEASE_PROGRAM(proto_segmented_sum_program);
    RELEASE_PROGRAM(proto_update_step_program);
    RELEASE_PROGRAM(shape_loss_reward_penalty_program);
    RELEASE_PROGRAM(shape_loss_reward_penalty_list_program); // NEU
    #undef RELEASE_PROGRAM
    printf("[C] shutdown_driver: Programs released.\n");

    // Finish pending commands and release queue
    if (queue) {
        cl_int finish_err = clFinish(queue);
        if(finish_err != CL_SUCCESS) {
            fprintf(stderr, "[C] shutdown_driver: Warning - clFinish failed before releasing queue: %s (%d)\n", clGetErrorString(finish_err), finish_err);
        }
        clReleaseCommandQueue(queue);
        queue = NULL;
        printf("[C] shutdown_driver: Command queue released.\n");
    }

    // Release context
    if (context) {
        clReleaseContext(context);
        context = NULL;
        printf("[C] shutdown_driver: Context released.\n");
    }

    // Reset device/platform handles and flags
    device_id = NULL;
    platform_id = NULL;
    has_fp64_support = 0;
    has_atomics_support = 0;

    printf("[C] shutdown_driver: Cleanup finished.\n");
}

/**
 * @brief Queries and returns the number of compute units (CUs) on the selected device.
 */
unsigned int get_compute_unit_count(int gpu_index) {
    if (!device_id) { return 0; }
    cl_uint cu_count = 0;
    cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu_count, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] get_compute_unit_count: clGetDeviceInfo failed for CL_DEVICE_MAX_COMPUTE_UNITS: %s (%d)\n", clGetErrorString(err), err);
        return 0;
    }
    return (unsigned int)cu_count;
}

// --- Exported Functions ---

/**
 * @brief Initializes the OpenCL environment for a specific GPU.
 */
static int pick_device_global_index(int wanted, cl_platform_id *out_plat, cl_device_id *out_dev) {
    if (out_plat) {
        *out_plat = NULL;
    }
    if (out_dev) {
        *out_dev = NULL;
    }

    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplat);
    if (err != CL_SUCCESS || nplat == 0) {
        return -1;
    }

    cl_platform_id *plats = (cl_platform_id*)malloc(nplat * sizeof(*plats));
    if (!plats) {
        return -2;
    }

    err = clGetPlatformIDs(nplat, plats, NULL);
    if (err != CL_SUCCESS) {
        free(plats);
        return -3;
    }

    int running = 0;
    for (cl_uint p = 0; p < nplat; ++p) {
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &ndev);
        if (err != CL_SUCCESS || ndev == 0) {
            running += (int)ndev;
            continue;
        }
        if (wanted >= running && wanted < running + (int)ndev) {
            cl_device_id *devs = (cl_device_id*)malloc(ndev * sizeof(*devs));
            if (!devs) {
                free(plats);
                return -4;
            }
            err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, ndev, devs, NULL);
            if (err != CL_SUCCESS) {
                free(devs);
                free(plats);
                return -5;
            }
            if (out_plat) {
                *out_plat = plats[p];
            }
            if (out_dev) {
                *out_dev = devs[wanted - running];
            }
            free(devs);
            free(plats);
            return 0;
        }
        running += (int)ndev;
    }

    free(plats);
    return -6;
}

DLLEXPORT int get_num_platforms(void) {
    cl_uint n = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &n);
    if (err != CL_SUCCESS) {
        return -1;
    }
    return (int)n;
}

DLLEXPORT int get_num_devices_on_platform(int platform_index) {
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplat);
    if (err != CL_SUCCESS) {
        return -1;
    }
    if (platform_index < 0 || (cl_uint)platform_index >= nplat) {
        return -2;
    }

    cl_platform_id *plats = (cl_platform_id*)malloc(nplat * sizeof(*plats));
    if (!plats) {
        return -3;
    }

    err = clGetPlatformIDs(nplat, plats, NULL);
    if (err != CL_SUCCESS) {
        free(plats);
        return -4;
    }

    cl_uint ndev = 0;
    err = clGetDeviceIDs(plats[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &ndev);
    free(plats);
    if (err != CL_SUCCESS) {
        return -5;
    }

    return (int)ndev;
}

DLLEXPORT int get_device_name(int platform_index, int device_index, char *out, int out_len) {
    if (!out || out_len <= 0) {
        return -1;
    }
    out[0] = '\0';

    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplat);
    if (err != CL_SUCCESS) {
        return -2;
    }
    if (platform_index < 0 || (cl_uint)platform_index >= nplat) {
        return -3;
    }

    cl_platform_id *plats = (cl_platform_id*)malloc(nplat * sizeof(*plats));
    if (!plats) {
        return -4;
    }

    err = clGetPlatformIDs(nplat, plats, NULL);
    if (err != CL_SUCCESS) {
        free(plats);
        return -5;
    }

    cl_uint ndev = 0;
    err = clGetDeviceIDs(plats[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &ndev);
    if (err != CL_SUCCESS) {
        free(plats);
        return -6;
    }
    if (device_index < 0 || (cl_uint)device_index >= ndev) {
        free(plats);
        return -7;
    }

    cl_device_id *devs = (cl_device_id*)malloc(ndev * sizeof(*devs));
    if (!devs) {
        free(plats);
        return -8;
    }

    err = clGetDeviceIDs(plats[platform_index], CL_DEVICE_TYPE_ALL, ndev, devs, NULL);
    if (err != CL_SUCCESS) {
        free(devs);
        free(plats);
        return -9;
    }

    char name[256] = {0};
    char vendor[256] = {0};
    if (clGetDeviceInfo(devs[device_index], CL_DEVICE_NAME, sizeof(name), name, NULL) != CL_SUCCESS) {
        strncpy(name, "Unknown", sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
    }
    if (clGetDeviceInfo(devs[device_index], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL) != CL_SUCCESS) {
        strncpy(vendor, "Unknown", sizeof(vendor) - 1);
        vendor[sizeof(vendor) - 1] = '\0';
    }

    int wrote = 0;
#ifdef CL_DEVICE_PCI_BUS_ID_NV
    cl_uint nv_bus = 0;
    cl_uint nv_slot = 0;
    if (clGetDeviceInfo(devs[device_index], CL_DEVICE_PCI_BUS_ID_NV, sizeof(nv_bus), &nv_bus, NULL) == CL_SUCCESS &&
        clGetDeviceInfo(devs[device_index], CL_DEVICE_PCI_SLOT_ID_NV, sizeof(nv_slot), &nv_slot, NULL) == CL_SUCCESS) {
        snprintf(out, out_len, "%s (%s) [PCI %u:%u]", name, vendor, nv_bus, nv_slot);
        wrote = 1;
    }
#endif
#ifdef CL_DEVICE_PCI_BUS_ID_AMD
    if (!wrote) {
        cl_uint amd_bus = 0;
        cl_uint amd_dev = 0;
        if (clGetDeviceInfo(devs[device_index], CL_DEVICE_PCI_BUS_ID_AMD, sizeof(amd_bus), &amd_bus, NULL) == CL_SUCCESS &&
            clGetDeviceInfo(devs[device_index], CL_DEVICE_PCI_DEVICE_ID_AMD, sizeof(amd_dev), &amd_dev, NULL) == CL_SUCCESS) {
            snprintf(out, out_len, "%s (%s) [PCI %u:%u]", name, vendor, amd_bus, amd_dev);
            wrote = 1;
        }
    }
#endif
    if (!wrote) {
        snprintf(out, out_len, "%s (%s)", name, vendor);
    }

    free(devs);
    free(plats);
    return 0;
}

DLLEXPORT int initialize_gpu(int global_device_index) {
    cl_int err = CL_SUCCESS;

    if (context || queue || device_id) {
         fprintf(stderr, "[C] initialize_gpu: Warning - Already initialized. Re-initialization attempt for index %d ignored.\n", global_device_index);
         return 1;
    }

    printf("[C] initialize_gpu: Initializing OpenCL for GPU index %d...\n", global_device_index);

    cl_platform_id chosen_platform = NULL;
    cl_device_id chosen_device = NULL;
    int pick_status = pick_device_global_index(global_device_index, &chosen_platform, &chosen_device);
    if (pick_status != 0 || !chosen_platform || !chosen_device) {
        fprintf(stderr, "[C] initialize_gpu: invalid device index %d (status=%d).\n", global_device_index, pick_status);
        return 0;
    }

    platform_id = chosen_platform;
    device_id = chosen_device;
    g_active_gpu_index = global_device_index;

    char platformName[256] = {0};
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(platformName) - 1, platformName, NULL);
    if (err != CL_SUCCESS) {
        strncpy(platformName, "Unknown Platform", sizeof(platformName) - 1);
        platformName[sizeof(platformName) - 1] = '\0';
    }
    printf("[C] initialize_gpu: Using platform: %s\n", platformName);

    char deviceName[256] = {0};
    if (clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName) - 1, deviceName, NULL) != CL_SUCCESS) {
        strncpy(deviceName, "Unknown Device", sizeof(deviceName) - 1);
        deviceName[sizeof(deviceName) - 1] = '\0';
    }

    char vendorName[256] = {0};
    if (clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendorName) - 1, vendorName, NULL) != CL_SUCCESS) {
        strncpy(vendorName, "Unknown Vendor", sizeof(vendorName) - 1);
        vendorName[sizeof(vendorName) - 1] = '\0';
    }

    int local_index = 0;
    cl_uint local_device_count = 0;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &local_device_count);
    if (err == CL_SUCCESS && local_device_count > 0) {
        cl_device_id *local_devices = (cl_device_id*)malloc(local_device_count * sizeof(*local_devices));
        if (local_devices) {
            cl_int local_err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, local_device_count, local_devices, NULL);
            if (local_err == CL_SUCCESS) {
                for (cl_uint i = 0; i < local_device_count; ++i) {
                    if (local_devices[i] == device_id) {
                        local_index = (int)i;
                        break;
                    }
                }
            }
            free(local_devices);
        }
    }

    printf("[C] initialize_gpu: Using device %d (global index %d): %s (%s)\n", local_index, global_device_index, deviceName, vendorName);

    // --- Check Device Capabilities ---
    cl_device_fp_config fp_config;
    err = clGetDeviceInfo(device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
    has_fp64_support = (err == CL_SUCCESS && (fp_config & CL_FP_FMA));
    printf("[C] initialize_gpu: FP64 Support (CL_FP_FMA flag): %s\n", has_fp64_support ? "Yes" : "No");

    has_atomics_support = 0;
    char* extensions_str = NULL;
    size_t extensions_size = 0;
    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, NULL, &extensions_size);
    if (err == CL_SUCCESS && extensions_size > 1) {
        extensions_str = (char*)malloc(extensions_size);
        if (extensions_str) {
            err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, extensions_size, extensions_str, NULL);
            if (err == CL_SUCCESS) {
                if (strstr(extensions_str, "cl_khr_global_int32_base_atomics") != NULL) {
                    printf("[C] initialize_gpu: Found 'cl_khr_global_int32_base_atomics'. Basic 32-bit global atomics SUPPORTED.\n");
                    if (strstr(extensions_str, "cl_khr_int64_base_atomics") != NULL) {
                         printf("[C] initialize_gpu: Found 'cl_khr_int64_base_atomics'. 64-bit atomics SUPPORTED (may be needed by atomic_add_float).\n");
                         has_atomics_support = 1;
                    } else {
                         printf("[C] initialize_gpu: WARNING - 'cl_khr_global_int32_base_atomics' found, but 'cl_khr_int64_base_atomics' NOT found. The custom atomic_add_float might fail.\n");
                         has_atomics_support = 0;
                    }
                } else {
                    printf("[C] initialize_gpu: Extension 'cl_khr_global_int32_base_atomics' NOT FOUND. GPU Proto Update (segmented sum) will FAIL if attempted.\n");
                }
            } else {
                fprintf(stderr, "[C] initialize_gpu: Warning - Failed to query CL_DEVICE_EXTENSIONS content: %s (%d)\n", clGetErrorString(err), err);
            }
            free(extensions_str); extensions_str = NULL;
        } else {
            fprintf(stderr, "[C] initialize_gpu: Warning - Failed to allocate memory (%zu bytes) for extensions string.\n", extensions_size);
        }
    } else {
        fprintf(stderr, "[C] initialize_gpu: Warning - Failed to query CL_DEVICE_EXTENSIONS size or size is trivial: %s (%d), size=%zu\n", clGetErrorString(err), err, extensions_size);
    }
    printf("[C] initialize_gpu: Atomics Support Flag (has_atomics_support): %d\n", has_atomics_support);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        fprintf(stderr, "[C] initialize_gpu: clCreateContext failed: %s (%d)\n", clGetErrorString(err), err);
        shutdown_driver();
        return 0;
    }
    printf("[C] initialize_gpu: Context created.\n");

    // --- Create Command Queue ---
    cl_command_queue_properties queue_props = 0;
    #if CL_TARGET_OPENCL_VERSION >= 200
        queue = clCreateCommandQueueWithProperties(context, device_id, &queue_props, &err);
        if (!queue || err != CL_SUCCESS) {
            fprintf(stderr, "[C] initialize_gpu: clCreateCommandQueueWithProperties failed: %s (%d). Trying deprecated clCreateCommandQueue...\n", clGetErrorString(err), err);
            #if defined(__GNUC__) || defined(__clang__)
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            #endif
            #ifdef _MSC_VER
            #pragma warning(push)
            #pragma warning(disable: 4996)
            #endif
            queue = clCreateCommandQueue(context, device_id, 0, &err);
            #ifdef _MSC_VER
            #pragma warning(pop)
            #endif
            #if defined(__GNUC__) || defined(__clang__)
            #pragma GCC diagnostic pop
            #endif
        }
    #else
        #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        #endif
        #ifdef _MSC_VER
        #pragma warning(push)
        #pragma warning(disable: 4996)
        #endif
        queue = clCreateCommandQueue(context, device_id, 0, &err);
        #ifdef _MSC_VER
        #pragma warning(pop)
        #endif
        #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC diagnostic pop
        #endif
    #endif

    if (!queue || err != CL_SUCCESS) {
        fprintf(stderr, "[C] initialize_gpu: Failed to create command queue: %s (%d)\n", clGetErrorString(err), err);
        shutdown_driver();
        return 0;
    }
    printf("[C] initialize_gpu: Command queue created.\n");

    // --- Compile All Kernels ---
    printf("[C] initialize_gpu: Compiling ALL OpenCL kernels...\n");
    cl_int compile_err;
    #define COMPILE_KERNEL(src, name, prog_var, kern_var) \
        printf("[C] initialize_gpu: Compiling kernel '%s'...\n", name); \
        compile_err = compile_opencl_kernel(src, name, prog_var, kern_var); \
        if (compile_err != CL_SUCCESS) { \
            fprintf(stderr, "[C] initialize_gpu: FATAL ERROR - Failed to compile kernel '%s'. Shutting down.\n", name); \
            shutdown_driver(); \
            return 0; \
        }

    // Compile each kernel
    COMPILE_KERNEL(matmul_kernel_src, "matrix_multiply", &matmul_program, &matmul_kernel);
    COMPILE_KERNEL(softmax_kernel_src, "softmax_rowwise", &softmax_program, &softmax_kernel);
    COMPILE_KERNEL(gelu_kernel_src, "gelu_elementwise", &gelu_program, &gelu_kernel);
    COMPILE_KERNEL(add_kernel_src, "add_elementwise", &add_program, &add_kernel);
    COMPILE_KERNEL(mul_kernel_src, "mul_elementwise", &mul_program, &mul_kernel);
    COMPILE_KERNEL(layernorm_kernel_src, "layer_norm", &layernorm_program, &layernorm_kernel);
    COMPILE_KERNEL(transpose_kernel_src, "transpose", &transpose_program, &transpose_kernel);
    COMPILE_KERNEL(gelu_backward_kernel_src, "gelu_backward_elementwise", &gelu_backward_program, &gelu_backward_kernel);
    COMPILE_KERNEL(matmul_backward_dA_kernel_src, "matmul_backward_da", &matmul_backward_da_program, &matmul_backward_da_kernel);
    COMPILE_KERNEL(matmul_backward_dB_kernel_src, "matmul_backward_db", &matmul_backward_db_program, &matmul_backward_db_kernel);
    COMPILE_KERNEL(layernorm_backward_kernel_src, "layer_norm_backward", &layernorm_backward_program, &layernorm_backward_kernel);
    COMPILE_KERNEL(adam_kernel_src, "adam_update", &adam_program, &adam_kernel);
    COMPILE_KERNEL(softmax_backward_kernel_src, "softmax_backward", &softmax_backward_program, &softmax_backward_kernel);
    // Note: Mul backward uses same program/kernel as forward Mul
    // COMPILE_KERNEL(mul_backward_kernel_src, "mul_backward", &mul_backward_program, &mul_backward_kernel); // Uses mul_kernel
    COMPILE_KERNEL(transpose_backward_kernel_src, "transpose_backward", &transpose_backward_program, &transpose_backward_kernel);
    COMPILE_KERNEL(embedding_lookup_kernel_src, "embedding_lookup", &embedding_lookup_program, &embedding_lookup_kernel);
    COMPILE_KERNEL(reduce_sum_kernel_src, "reduce_sum_axis01", &reduce_sum_program, &reduce_sum_kernel);
    COMPILE_KERNEL(broadcast_add_kernel_src, "broadcast_add_bias", &broadcast_add_program, &broadcast_add_kernel);
    COMPILE_KERNEL(transpose_batched_kernel_src, "transpose_batched_last_two", &transpose_batched_program, &transpose_batched_kernel);
    COMPILE_KERNEL(transpose_12_batched_kernel_src, "transpose_12_batched", &transpose_12_batched_program, &transpose_12_batched_kernel);
    COMPILE_KERNEL(matmul_batched_kernel_src, "matmul_batched", &matmul_batched_program, &matmul_batched_kernel);
    COMPILE_KERNEL(matmul_batched_backward_dA_kernel_src, "matmul_batched_backward_da", &matmul_batched_backward_da_program, &matmul_batched_backward_da_kernel);
    COMPILE_KERNEL(matmul_batched_backward_dB_kernel_src, "matmul_batched_backward_db", &matmul_batched_backward_db_program, &matmul_batched_backward_db_kernel);
    COMPILE_KERNEL(log_softmax_stable_kernel_src, "log_softmax_stable_rowwise", &log_softmax_program, &log_softmax_kernel);
    COMPILE_KERNEL(cross_entropy_loss_grad_kernel_src, "cross_entropy_loss_grad", &cross_entropy_program, &cross_entropy_kernel);
    COMPILE_KERNEL(add_broadcast_pe_kernel_src, "add_broadcast_pe", &add_broadcast_pe_program, &add_broadcast_pe_kernel);
    COMPILE_KERNEL(threshold_spike_kernel_src, "threshold_spike", &threshold_spike_program, &threshold_spike_kernel);
    COMPILE_KERNEL(add_bias_mn_kernel_src, "add_bias_mn", &add_bias_mn_program, &add_bias_mn_kernel);
    COMPILE_KERNEL(dynamic_token_assign_kernel_src, "dynamic_token_assignment", &dynamic_token_assign_program, &dynamic_token_assign_kernel);
    COMPILE_KERNEL(pairwise_similarity_kernel_src, "pairwise_similarity_dot", &pairwise_similarity_program, &pairwise_similarity_kernel);
    COMPILE_KERNEL(hebbian_update_local_reduce_kernel_src, "hebbian_update_local_reduce", &hebbian_update_local_reduce_program, &hebbian_update_local_reduce_kernel);
    COMPILE_KERNEL(embedding_backward_calc_delta_local_kernel_src, "embedding_backward_calc_delta_local", &embedding_backward_calc_delta_local_program, &embedding_backward_calc_delta_local_kernel);
    COMPILE_KERNEL(proto_segmented_sum_atomic_kernel_src, "proto_segmented_sum_atomic", &proto_segmented_sum_program, &proto_segmented_sum_kernel);
    COMPILE_KERNEL(proto_update_step_kernel_src, "proto_update_step", &proto_update_step_program, &proto_update_step_kernel);
    COMPILE_KERNEL(shape_loss_reward_penalty_kernel_src, "shape_loss_reward_penalty", &shape_loss_reward_penalty_program, &shape_loss_reward_penalty_kernel);
    // NEU: Compile Loss Shaping Kernel (List)
    COMPILE_KERNEL(shape_loss_reward_penalty_list_kernel_src, "shape_loss_reward_penalty_list", &shape_loss_reward_penalty_list_program, &shape_loss_reward_penalty_list_kernel);

    #undef COMPILE_KERNEL

    printf("[C] initialize_gpu: All kernels compiled successfully.\n");
    printf("[C] initialize_gpu: Initialization OK for GPU %d (%s).\n", global_device_index, deviceName);

    update_device_info_buffer();
    reset_last_kernel_time();
    return 1;
}

DLLEXPORT void *allocate_gpu_memory(int gpu_index, size_t size) {
    cl_int err;
    if (!context) { fprintf(stderr, "[C] allocate_gpu_memory: Error - No OpenCL context available.\n"); return NULL; }
    if (size == 0) { fprintf(stderr, "[C] allocate_gpu_memory: Warning - Attempted to allocate 0 bytes. Returning NULL.\n"); return NULL; }
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (!buffer || err != CL_SUCCESS) {
        fprintf(stderr, "[C] allocate_gpu_memory: Error - clCreateBuffer failed: %s (%d) for size %zu bytes.\n", clGetErrorString(err), err, size);
        return NULL;
    }
    return (void*)buffer;
}

/**
 * @brief Frees memory previously allocated on the GPU device.
 */
DLLEXPORT void free_gpu_memory(int gpu_index, void* buffer_handle) {
     if (!buffer_handle) { return; }
    cl_mem buffer = (cl_mem)buffer_handle;
     if (!context) { return; }
    cl_int err = clReleaseMemObject(buffer);
    if (err != CL_SUCCESS && err != CL_INVALID_MEM_OBJECT) { // Ignore errors if already freed
         fprintf(stderr, "[C] free_gpu_memory: Error - clReleaseMemObject failed for buffer %p: %s (%d)\n", buffer_handle, clGetErrorString(err), err);
    }
}

/**
 * @brief Writes data from host memory to a GPU buffer (blocking).
 */
DLLEXPORT int write_host_to_gpu_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, const void* host_source_ptr) {
     if (!gpu_buffer_handle) { fprintf(stderr, "[C] write_host_to_gpu_blocking: Error - Invalid GPU buffer handle (NULL).\n"); return 0; }
    if (size > 0 && !host_source_ptr) { fprintf(stderr, "[C] write_host_to_gpu_blocking: Error - Host source pointer is NULL but size > 0 (%zu).\n", size); return 0; }
    if (!queue) { fprintf(stderr, "[C] write_host_to_gpu_blocking: Error - Command queue is NULL.\n"); return 0; }
    if (size == 0) { return 1; }
    cl_mem gpu_buffer = (cl_mem)gpu_buffer_handle;
    cl_int err = clEnqueueWriteBuffer(queue, gpu_buffer, CL_TRUE, offset, size, host_source_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "[C] write_host_to_gpu_blocking: Error - clEnqueueWriteBuffer failed: %s (%d) [offset=%zu, size=%zu]\n", clGetErrorString(err), err, offset, size); return 0; }
    return 1;
}

/**
 * @brief Reads data from a GPU buffer to host memory (blocking).
 */
DLLEXPORT int read_gpu_to_host_blocking(int gpu_index, void* gpu_buffer_handle, size_t offset, size_t size, void* host_destination_ptr) {
     if (!gpu_buffer_handle) { fprintf(stderr, "[C] read_gpu_to_host_blocking: Error - Invalid GPU buffer handle (NULL).\n"); return 0; }
     if (size > 0 && !host_destination_ptr) { fprintf(stderr, "[C] read_gpu_to_host_blocking: Error - Host destination pointer is NULL but size > 0 (%zu).\n", size); return 0; }
     if (!queue) { fprintf(stderr, "[C] read_gpu_to_host_blocking: Error - Command queue is NULL.\n"); return 0; }
     if (size == 0) { return 1; }
    cl_mem gpu_buffer = (cl_mem)gpu_buffer_handle;
    cl_int err = clEnqueueReadBuffer(queue, gpu_buffer, CL_TRUE, offset, size, host_destination_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "[C] read_gpu_to_host_blocking: Error - clEnqueueReadBuffer failed: %s (%d) [offset=%zu, size=%zu]\n", clGetErrorString(err), err, offset, size); return 0; }
    return 1;
}

/**
 * @brief Shuts down the OpenCL driver and releases all resources.
 */
DLLEXPORT void shutdown_gpu(int gpu_index) {
    printf("[C] shutdown_gpu: Received shutdown request for GPU index %d. Shutting down global OpenCL resources.\n", gpu_index);
    shutdown_driver();
}


// --- Command Data Structures (Used by submit_kernel_command) ---
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int B; int M; int N; int K; } BMMCommandData;
typedef struct { void* buffer_input; void* buffer_output; int num_rows; int row_size; } SoftmaxCommandData;
typedef struct { void* buffer_input; void* buffer_output; int num_elements; } GeluCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int num_elements; } AddCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int num_elements; } MulCommandData;
typedef struct { void* buffer_input; void* buffer_output; int num_rows; int row_size; float eps; } LayerNormCommandData;
typedef struct { void* src_buffer; void* dst_buffer; size_t size; } CloneCommandData;
typedef struct { void* buffer_input; void* buffer_output; int rows; int cols; } TransposeCommandData;
typedef struct { void* buffer_input; void* buffer_grad_output; void* buffer_grad_input; int num_elements; } GeluBackwardCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_dc; void* buffer_da; void* buffer_db; int B, M, N, K; } MatMulBackwardData;
typedef struct { void* buffer_dy; void* buffer_x; void* buffer_dx; int num_rows; int row_size; float eps; } LayerNormBackwardCommandData;
typedef struct { void* param_buffer; void* grad_buffer; void* m_buffer; void* v_buffer; int num_elements; int t_step; float lr,beta1,beta2,eps,weight_decay,beta1_t,beta2_t; } AdamCommandData;
typedef struct { void* buffer_dy; void* buffer_y; void* buffer_dx; int num_rows; int row_size; } SoftmaxBackwardCommandData;
typedef struct { void* buffer_dC; void* buffer_A; void* buffer_B; void* buffer_dA; void* buffer_dB; int num_elements; } MulBackwardCommandData;
typedef struct { void* buffer_dC; void* buffer_dA; int rows_A; int cols_A; } TransposeBackwardCommandData;
typedef struct { void* idx; void* w; void* o; int b, s, d, v; } EmbeddingLookupCommandData;
typedef struct { void* in; void* out; int B, M, N; } ReduceSumCommandData;
typedef struct { void* a; void* b; void* c; int B, M, N; } BroadcastAddCommandData;
typedef struct { void* in; void* out; int B_flat, d1, d2; } TransposeBatchedCommandData;
typedef struct { void* in; void* out; int B, D1, D2, D3; } Transpose12BatchedCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_c; int B; int M; int N; int K; } BMMBatchedCommandData;
typedef struct { void* buffer_a; void* buffer_b; void* buffer_dc; void* buffer_da; void* buffer_db; int B, M, N, K; } BMMBatchedBackwardData;
typedef struct { void* input_logits; void* output_log_probs; int B_S_rows; int V_cols; } LogSoftmaxStableCommandData;
typedef struct { void* log_probs; void* target_indices; void* grad_input; void* loss_per_sample; int B_S_rows; int V_cols; } CrossEntropyLossGradCommandData;typedef struct { void* input; void* pe_slice; void* output; int B; int S; int E; } AddBroadcastPECommandData;
typedef struct { void* buffer_a; void* buffer_c; void* buffer_w; float learning_rate; int B; int M; int N; int K; } HebbianUpdateLocalReduceCommandData;
typedef struct { void* buffer_activations; void* buffer_spikes; float threshold; int num_elements; } ThresholdSpikeCommandData;
typedef struct { void* a_or_c; void* b_bias; int M; int N; } AddBiasMNCommandData;
typedef struct { void* d_o; void* idx; void* delta_dw; int b; int s; int d; int v; } EmbeddingBackwardPass1CommandData;
typedef struct { void* activations_bse; void* prototypes_te; void* output_indices_bs; int B; int S; int E; int T; } DynamicTokenAssignmentCommandData;
typedef struct { void* states_nd; void* output_similarity_nn; int N; int D; } PairwiseSimilarityCommandData;
typedef struct {
    void* activations_flat; void* indices_flat; void* proto_sums; void* proto_counts;
    int M_flat; int E; int T;
} ProtoSegmentedSumCommandData;
typedef struct {
    void* prototypes; void* proto_sums; void* proto_counts;
    float learning_rate; int E; int T;
} ProtoUpdateStepCommandData;
// Struct for Loss Shaping Kernel (Single Pair)
typedef struct {
    void* loss_per_sample_in;
    void* predictions;
    void* targets;
    void* loss_per_sample_out;
    int num_samples;
    int num_classes;
    float penalty_weight;
    float reward_weight;
    float high_confidence_threshold;
    int critical_target_class;
    int critical_predicted_class;
} ShapeLossRewardPenaltyCommandData;
// NEU: Struct for Loss Shaping Kernel (List of Pairs)
typedef struct {
    void* loss_per_sample_in;
    void* predictions;
    void* targets;
    void* loss_per_sample_out;
    void* critical_pairs; // Handle zum Buffer der ID-Paare
    int num_samples;
    int num_classes;
    int num_critical_pairs; // Anzahl der Paare
    float penalty_weight;
    float reward_weight;
    float high_confidence_threshold;
} ShapeLossRewardPenaltyListCommandData;
// ------------------------------------

/**
 * @brief Zeros out a specified number of bytes in a GPU buffer.
 */
int zero_gpu_buffer(int gpu_index, void* gpu_buffer_handle, size_t size_bytes) {
    FP_TYPE* zeros_host = NULL;
    size_t num_elements;
    int success = 1;

    if (!gpu_buffer_handle) { fprintf(stderr, "[C] zero_gpu_buffer: Error - GPU buffer handle is NULL.\n"); return 0; }
    if (size_bytes == 0) { return 1; }
    if (size_bytes % sizeof(FP_TYPE) != 0) { fprintf(stderr, "[C] zero_gpu_buffer: Error - size_bytes %zu is not a multiple of FP_TYPE size %zu.\n", size_bytes, sizeof(FP_TYPE)); return 0; }
    num_elements = size_bytes / sizeof(FP_TYPE);

    zeros_host = (FP_TYPE*)malloc(size_bytes);
    if (!zeros_host) { fprintf(stderr, "[C] zero_gpu_buffer: Error - Failed to malloc %zu bytes for host zero buffer.\n", size_bytes); return 0; }

    for (size_t i = 0; i < num_elements; ++i) { zeros_host[i] = (FP_TYPE)0.0; }

    if (!write_host_to_gpu_blocking(gpu_index, gpu_buffer_handle, 0, size_bytes, zeros_host)) {
        fprintf(stderr, "[C] zero_gpu_buffer: Error - Failed to write zeros to GPU buffer.\n");
        success = 0;
    }

    free(zeros_host);
    return success;
}

/** @brief Default work-group size for reduction kernels. Can be tuned. */
#ifndef REDUCE_WG_SIZE
#define REDUCE_WG_SIZE 256
#endif

/**
 * @brief Helper function to determine parameters for reduction kernels.
 */
static cl_int get_reduction_params_helper(size_t* lws_out, size_t* local_mem_bytes_out) {
    *lws_out = REDUCE_WG_SIZE;
    *local_mem_bytes_out = 0;
    if (!device_id) { fprintf(stderr, "[C] ERROR (Reduction Setup): No device ID available.\n"); return CL_INVALID_DEVICE; }

    size_t max_wg_size = 0;
    cl_int lws_err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
    if (lws_err == CL_SUCCESS) {
        if (*lws_out > max_wg_size) {
            fprintf(stderr, "[C] WARN (Reduction Setup): Requested LWS %zu exceeds device max %zu, clamping LWS to %zu.\n", *lws_out, max_wg_size, max_wg_size);
            *lws_out = max_wg_size;
        }
    } else {
         fprintf(stderr, "[C] WARN (Reduction Setup): Failed to query max WGS (%s), using default LWS %zu without clamping check.\n", clGetErrorString(lws_err), *lws_out);
    }
    if (*lws_out == 0) { fprintf(stderr, "[C] ERROR (Reduction Setup): Calculated Local Work Size (LWS) is zero.\n"); return CL_INVALID_WORK_GROUP_SIZE; }

    #ifdef CL_HAS_FP64
        typedef double REDUCE_ACCUM_TYPE_HOST;
    #else
        typedef float REDUCE_ACCUM_TYPE_HOST;
    #endif
    *local_mem_bytes_out = (*lws_out) * sizeof(REDUCE_ACCUM_TYPE_HOST);

    cl_ulong max_lmem_size_ulong = 0;
    cl_int lmem_err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &max_lmem_size_ulong, NULL);
    if (lmem_err == CL_SUCCESS) {
         if (*local_mem_bytes_out > (size_t)max_lmem_size_ulong) {
             fprintf(stderr, "[C] ERROR (Reduction Setup): Calculated local memory size %zu bytes exceeds device max %llu bytes for LWS %zu.\n",
                     *local_mem_bytes_out, (unsigned long long)max_lmem_size_ulong, *lws_out);
             return CL_INVALID_WORK_GROUP_SIZE;
         }
     } else {
         fprintf(stderr, "[C] WARN (Reduction Setup): Failed to query CL_DEVICE_LOCAL_MEM_SIZE (%s), cannot verify limit for %zu bytes needed.\n", clGetErrorString(lmem_err), *local_mem_bytes_out);
     }
    return CL_SUCCESS;
}

/**
 * @brief Submits a command to the OpenCL command queue for execution.
 */
int submit_kernel_command(int gpu_index, GPUCommand command, void *data) {
    cl_int err = CL_SUCCESS;
    if (!queue) { fprintf(stderr, "[C] submit_kernel_command: Error - Invalid command queue (NULL).\n"); return 0; }

    #define CHECK_CL_ERR(call, kernel_name_str) \
        err = (call); \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "[C] OpenCL Error (%s): %s (%d) during '%s' in %s line %d\n", \
                    kernel_name_str, clGetErrorString(err), err, #call, __FILE__, __LINE__); \
            return 0; \
        }

    size_t lws_reduce; size_t local_mem_bytes;

    switch(command) {
        // --- Standard Kernels ---
        case COMMAND_MATRIX_MULTIPLY: {
            BMMCommandData* cmd = (BMMCommandData*)data;
            if (!matmul_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit MatMul: Invalid args or kernel.\n"); return 0;}
            if (cmd->B <= 0 || cmd->M <= 0 || cmd->N <= 0) { if ((size_t)cmd->B * cmd->M * cmd->N == 0) return 1; fprintf(stderr, "[C] Submit MatMul: Invalid dimensions B/M/N.\n"); return 0; }
            if (cmd->K <= 0) { fprintf(stderr, "[C] Submit MatMul: Invalid dimension K.\n"); return 0;}
            cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 0, sizeof(cl_mem), &a), "BMM Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 1, sizeof(cl_mem), &b), "BMM Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 2, sizeof(cl_mem), &c), "BMM Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 3, sizeof(cl_int), &cmd->B), "BMM Fwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 4, sizeof(cl_int), &cmd->M), "BMM Fwd Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 5, sizeof(cl_int), &cmd->N), "BMM Fwd Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_kernel, 6, sizeof(cl_int), &cmd->K), "BMM Fwd Arg 6");
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->M, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "BMM Fwd Enqueue");
            return 1;
        }
        case COMMAND_SOFTMAX_ROWWISE: {
            SoftmaxCommandData* cmd = (SoftmaxCommandData*)data;
            if (!softmax_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit Softmax: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_rows <= 0 || cmd->row_size <= 0) { if (cmd->num_rows == 0) return 1; fprintf(stderr, "[C] Submit Softmax: Invalid dimensions.\n"); return 0; }
            cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &in), "Softmax Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &out), "Softmax Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 2, sizeof(cl_int), &cmd->num_rows), "Softmax Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(softmax_kernel, 3, sizeof(cl_int), &cmd->row_size), "Softmax Fwd Arg 3");
            size_t gws[1] = { (size_t)cmd->num_rows };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Softmax Fwd Enqueue");
            return 1;
        }
        case COMMAND_GELU_ELEMENTWISE: {
            GeluCommandData* cmd = (GeluCommandData*)data;
            if (!gelu_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit GELU: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_elements <= 0) { if (cmd->num_elements == 0) return 1; fprintf(stderr, "[C] Submit GELU: Invalid dimensions.\n"); return 0; }
            cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
            CHECK_CL_ERR(clSetKernelArg(gelu_kernel, 0, sizeof(cl_mem), &in), "GELU Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(gelu_kernel, 1, sizeof(cl_mem), &out), "GELU Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(gelu_kernel, 2, sizeof(cl_int), &cmd->num_elements), "GELU Fwd Arg 2");
            size_t gws[1] = { (size_t)cmd->num_elements };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, gelu_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "GELU Fwd Enqueue");
            return 1;
        }
        case COMMAND_ADD_ELEMENTWISE: {
             AddCommandData* cmd = (AddCommandData*)data;
             if (!add_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit Add: Invalid args or kernel.\n"); return 0; }
             if (cmd->num_elements <= 0) { if (cmd->num_elements == 0) return 1; fprintf(stderr, "[C] Submit Add: Invalid dimensions.\n"); return 0; }
             cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 0, sizeof(cl_mem), &a), "Add Fwd Arg 0");
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 1, sizeof(cl_mem), &b), "Add Fwd Arg 1");
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 2, sizeof(cl_mem), &c), "Add Fwd Arg 2");
             CHECK_CL_ERR(clSetKernelArg(add_kernel, 3, sizeof(cl_int), &cmd->num_elements), "Add Fwd Arg 3");
             size_t gws[1] = { (size_t)cmd->num_elements };
             CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, add_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Add Fwd Enqueue");
             return 1;
        }
        case COMMAND_MUL_ELEMENTWISE: {
            MulCommandData* cmd = (MulCommandData*)data;
            if (!mul_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit Mul: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_elements <= 0) { if (cmd->num_elements == 0) return 1; fprintf(stderr, "[C] Submit Mul: Invalid dimensions.\n"); return 0; }
            cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
            CHECK_CL_ERR(clSetKernelArg(mul_kernel, 0, sizeof(cl_mem), &a), "Mul Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(mul_kernel, 1, sizeof(cl_mem), &b), "Mul Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(mul_kernel, 2, sizeof(cl_mem), &c), "Mul Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(mul_kernel, 3, sizeof(cl_int), &cmd->num_elements), "Mul Fwd Arg 3");
            size_t gws[1] = { (size_t)cmd->num_elements };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, mul_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Mul Fwd Enqueue");
            return 1;
        }
        case COMMAND_LAYER_NORM: {
            LayerNormCommandData* cmd = (LayerNormCommandData*)data;
            if (!layernorm_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit LayerNorm: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_rows <= 0 || cmd->row_size <= 0) { if (cmd->num_rows == 0) return 1; fprintf(stderr, "[C] Submit LayerNorm: Invalid dimensions.\n"); return 0; }
            cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
            float effective_eps = (cmd->eps > 0) ? cmd->eps : 1e-5f;
            CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 0, sizeof(cl_mem), &in), "LayerNorm Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 1, sizeof(cl_mem), &out), "LayerNorm Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 2, sizeof(cl_int), &cmd->num_rows), "LayerNorm Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 3, sizeof(cl_int), &cmd->row_size), "LayerNorm Fwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(layernorm_kernel, 4, sizeof(cl_float), &effective_eps), "LayerNorm Fwd Arg 4");
            size_t gws[1] = { (size_t)cmd->num_rows };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, layernorm_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "LayerNorm Fwd Enqueue");
            return 1;
        }
        case COMMAND_CLONE: {
            CloneCommandData* cmd = (CloneCommandData*)data;
            if (!cmd || !cmd->src_buffer || !cmd->dst_buffer) { fprintf(stderr, "[C] Submit Clone: Invalid args.\n"); return 0; }
            if (cmd->size == 0) return 1;
            cl_mem src = (cl_mem)cmd->src_buffer;
            cl_mem dst = (cl_mem)cmd->dst_buffer;
            CHECK_CL_ERR(clEnqueueCopyBuffer(queue, src, dst, 0, 0, cmd->size, 0, NULL, NULL), "Clone Enqueue (CopyBuffer)");
            return 1;
        }
        case COMMAND_TRANSPOSE: {
            TransposeCommandData* cmd = (TransposeCommandData*)data;
            if (!transpose_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_output) { fprintf(stderr, "[C] Submit Transpose2D: Invalid args or kernel.\n"); return 0; }
            if (cmd->rows <= 0 || cmd->cols <= 0) { if ((size_t)cmd->rows * cmd->cols == 0) return 1; fprintf(stderr, "[C] Submit Transpose2D: Invalid dimensions.\n"); return 0; }
            cl_mem in = (cl_mem)cmd->buffer_input, out = (cl_mem)cmd->buffer_output;
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), &in), "Transpose Fwd (2D) Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 1, sizeof(cl_mem), &out), "Transpose Fwd (2D) Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 2, sizeof(cl_int), &cmd->rows), "Transpose Fwd (2D) Arg 2");
            CHECK_CL_ERR(clSetKernelArg(transpose_kernel, 3, sizeof(cl_int), &cmd->cols), "Transpose Fwd (2D) Arg 3");
            size_t gws[2] = { (size_t)cmd->cols, (size_t)cmd->rows };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "Transpose Fwd (2D) Enqueue");
            return 1;
        }
        case COMMAND_GELU_BACKWARD_ELEMENTWISE: {
            GeluBackwardCommandData* cmd = (GeluBackwardCommandData*)data;
            if (!gelu_backward_kernel || !cmd || !cmd->buffer_input || !cmd->buffer_grad_output || !cmd->buffer_grad_input) { fprintf(stderr, "[C] Submit GELU Bwd: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_elements <= 0) { if (cmd->num_elements == 0) return 1; fprintf(stderr, "[C] Submit GELU Bwd: Invalid dimensions.\n"); return 0; }
            cl_mem input_mem = (cl_mem)cmd->buffer_input; cl_mem grad_output_mem = (cl_mem)cmd->buffer_grad_output; cl_mem grad_input_mem = (cl_mem)cmd->buffer_grad_input;
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 0, sizeof(cl_mem), &input_mem), "GELU Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 1, sizeof(cl_mem), &grad_output_mem), "GELU Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 2, sizeof(cl_mem), &grad_input_mem), "GELU Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(gelu_backward_kernel, 3, sizeof(cl_int), &cmd->num_elements), "GELU Bwd Arg 3");
            size_t gws[1] = { (size_t)cmd->num_elements };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, gelu_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "GELU Bwd Enqueue");
            return 1;
        }
        case COMMAND_MATMUL_BACKWARD_DA: {
            MatMulBackwardData* cmd = (MatMulBackwardData*)data;
            if (!matmul_backward_da_kernel || !cmd || !cmd->buffer_dc || !cmd->buffer_b || !cmd->buffer_da) { fprintf(stderr, "[C] Submit MatMul dA: Invalid args or kernel.\n"); return 0; }
             if (cmd->B <= 0 || cmd->M <= 0 || cmd->K <= 0) { if ((size_t)cmd->B * cmd->M * cmd->K == 0) return 1; fprintf(stderr, "[C] Submit MatMul dA: Invalid dimensions B/M/K.\n"); return 0;}
             if (cmd->N <= 0) { fprintf(stderr, "[C] Submit MatMul dA: Invalid dimension N.\n"); return 0;}
            cl_mem dc = (cl_mem)cmd->buffer_dc, b_mem = (cl_mem)cmd->buffer_b, da = (cl_mem)cmd->buffer_da;
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 0, sizeof(cl_mem), &dc), "MatMul dA Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 1, sizeof(cl_mem), &b_mem), "MatMul dA Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 2, sizeof(cl_mem), &da), "MatMul dA Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul dA Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul dA Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul dA Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_da_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul dA Arg 6");
            size_t gws[3] = { (size_t)cmd->K, (size_t)cmd->M, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_backward_da_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "MatMul dA Enqueue");
            return 1;
        }
        case COMMAND_MATMUL_BACKWARD_DB: {
            MatMulBackwardData* cmd = (MatMulBackwardData*)data;
            if (!matmul_backward_db_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_dc || !cmd->buffer_db) { fprintf(stderr, "[C] Submit MatMul dB: Invalid args or kernel.\n"); return 0; }
            if (cmd->K <= 0 || cmd->N <= 0) { if ((size_t)cmd->K * cmd->N == 0) return 1; fprintf(stderr, "[C] Submit MatMul dB: Invalid dimensions K/N.\n"); return 0;}
            if (cmd->B <= 0 || cmd->M <= 0) { fprintf(stderr, "[C] Submit MatMul dB: Invalid dimensions B/M.\n"); return 0;}
            cl_mem a_mem = (cl_mem)cmd->buffer_a, dc = (cl_mem)cmd->buffer_dc, db = (cl_mem)cmd->buffer_db;
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 0, sizeof(cl_mem), &a_mem), "MatMul dB Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 1, sizeof(cl_mem), &dc), "MatMul dB Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 2, sizeof(cl_mem), &db), "MatMul dB Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul dB Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul dB Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul dB Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_backward_db_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul dB Arg 6");
            size_t gws[2] = { (size_t)cmd->N, (size_t)cmd->K };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_backward_db_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "MatMul dB Enqueue");
            return 1;
        }
        case COMMAND_LAYER_NORM_BACKWARD: {
            LayerNormBackwardCommandData* cmd = (LayerNormBackwardCommandData*)data;
            if (!layernorm_backward_kernel || !cmd || !cmd->buffer_dy || !cmd->buffer_x || !cmd->buffer_dx) { fprintf(stderr, "[C] Submit LayerNorm Bwd: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_rows <= 0 || cmd->row_size <= 0) { if (cmd->num_rows == 0) return 1; fprintf(stderr, "[C] Submit LayerNorm Bwd: Invalid dimensions.\n"); return 0; }
            cl_mem dy_mem = (cl_mem)cmd->buffer_dy; cl_mem x_mem = (cl_mem)cmd->buffer_x; cl_mem dx_mem = (cl_mem)cmd->buffer_dx;
            float effective_eps = (cmd->eps > 0) ? cmd->eps : 1e-5f;
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 0, sizeof(cl_mem), &dy_mem), "LayerNorm Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 1, sizeof(cl_mem), &x_mem), "LayerNorm Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 2, sizeof(cl_mem), &dx_mem), "LayerNorm Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 3, sizeof(cl_int), &cmd->num_rows), "LayerNorm Bwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 4, sizeof(cl_int), &cmd->row_size), "LayerNorm Bwd Arg 4");
            CHECK_CL_ERR(clSetKernelArg(layernorm_backward_kernel, 5, sizeof(cl_float), &effective_eps), "LayerNorm Bwd Arg 5");
            size_t gws[1] = { (size_t)cmd->num_rows };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, layernorm_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "LayerNorm Bwd Enqueue");
            return 1;
        }
        case COMMAND_ADAM_UPDATE: {
            AdamCommandData* cmd = (AdamCommandData*)data;
            if (!adam_kernel || !cmd || !cmd->param_buffer || !cmd->grad_buffer || !cmd->m_buffer || !cmd->v_buffer) { fprintf(stderr, "[C] Submit Adam: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_elements <= 0) { if (cmd->num_elements == 0) return 1; fprintf(stderr, "[C] Submit Adam: Invalid dimensions.\n"); return 0; }
             if (cmd->t_step <= 0 || cmd->lr < 0.0f || cmd->beta1 < 0.0f || cmd->beta1 >= 1.0f || cmd->beta2 < 0.0f || cmd->beta2 >= 1.0f || cmd->eps < 0.0f || cmd->weight_decay < 0.0f) {
                 fprintf(stderr, "[C] Submit Adam: Invalid hyperparameters (t=%d, lr=%f, b1=%f, b2=%f, eps=%f, wd=%f).\n", cmd->t_step, cmd->lr, cmd->beta1, cmd->beta2, cmd->eps, cmd->weight_decay);
                 return 0;
             }
            cl_mem p = (cl_mem)cmd->param_buffer; cl_mem g = (cl_mem)cmd->grad_buffer; cl_mem m = (cl_mem)cmd->m_buffer; cl_mem v = (cl_mem)cmd->v_buffer;
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 0, sizeof(cl_mem), &p), "Adam Arg 0");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 1, sizeof(cl_mem), &g), "Adam Arg 1");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 2, sizeof(cl_mem), &m), "Adam Arg 2");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 3, sizeof(cl_mem), &v), "Adam Arg 3");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 4, sizeof(cl_int), &cmd->num_elements), "Adam Arg 4");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 5, sizeof(cl_float), &cmd->lr), "Adam Arg 5");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 6, sizeof(cl_float), &cmd->beta1), "Adam Arg 6");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 7, sizeof(cl_float), &cmd->beta2), "Adam Arg 7");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 8, sizeof(cl_float), &cmd->eps), "Adam Arg 8");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 9, sizeof(cl_float), &cmd->weight_decay), "Adam Arg 9");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 10, sizeof(cl_float), &cmd->beta1_t), "Adam Arg 10");
            CHECK_CL_ERR(clSetKernelArg(adam_kernel, 11, sizeof(cl_float), &cmd->beta2_t), "Adam Arg 11");
            size_t gws[1] = { (size_t)cmd->num_elements };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, adam_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Adam Update Enqueue");
            return 1;
        }
        case COMMAND_SOFTMAX_BACKWARD: {
            SoftmaxBackwardCommandData* cmd = (SoftmaxBackwardCommandData*)data;
            if (!softmax_backward_kernel || !cmd || !cmd->buffer_dy || !cmd->buffer_y || !cmd->buffer_dx) { fprintf(stderr, "[C] Submit Softmax Bwd: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_rows <= 0 || cmd->row_size <= 0) { if (cmd->num_rows == 0) return 1; fprintf(stderr, "[C] Submit Softmax Bwd: Invalid dimensions.\n"); return 0; }
            cl_mem dy = (cl_mem)cmd->buffer_dy; cl_mem y = (cl_mem)cmd->buffer_y; cl_mem dx = (cl_mem)cmd->buffer_dx;
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 0, sizeof(cl_mem), &dy), "Softmax Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 1, sizeof(cl_mem), &y), "Softmax Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 2, sizeof(cl_mem), &dx), "Softmax Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 3, sizeof(cl_int), &cmd->num_rows), "Softmax Bwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(softmax_backward_kernel, 4, sizeof(cl_int), &cmd->row_size), "Softmax Bwd Arg 4");
            size_t gws[1] = { (size_t)cmd->num_rows };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, softmax_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Softmax Bwd Enqueue");
            return 1;
        }
         case COMMAND_MUL_BACKWARD: {
            MulBackwardCommandData* cmd = (MulBackwardCommandData*)data;
            if (!mul_backward_kernel || !cmd || !cmd->buffer_dC || !cmd->buffer_A || !cmd->buffer_B || (!cmd->buffer_dA && !cmd->buffer_dB)) {
                if (cmd && !cmd->buffer_dA && !cmd->buffer_dB) return 1;
                fprintf(stderr, "[C] Submit Mul Bwd: Invalid args or kernel.\n"); return 0;
            }
            if (cmd->num_elements <= 0) { if (cmd->num_elements == 0) return 1; fprintf(stderr, "[C] Submit Mul Bwd: Invalid dimensions.\n"); return 0; }
            cl_mem dC = (cl_mem)cmd->buffer_dC; cl_mem A_mem = (cl_mem)cmd->buffer_A; cl_mem B_mem = (cl_mem)cmd->buffer_B;
            cl_mem dA_mem = (cl_mem)cmd->buffer_dA; cl_mem dB_mem = (cl_mem)cmd->buffer_dB;
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 0, sizeof(cl_mem), &dC), "Mul Bwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 1, sizeof(cl_mem), &A_mem), "Mul Bwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 2, sizeof(cl_mem), &B_mem), "Mul Bwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 3, sizeof(cl_mem), &dA_mem), "Mul Bwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 4, sizeof(cl_mem), &dB_mem), "Mul Bwd Arg 4");
            CHECK_CL_ERR(clSetKernelArg(mul_backward_kernel, 5, sizeof(cl_int), &cmd->num_elements), "Mul Bwd Arg 5");
            size_t gws[1] = { (size_t)cmd->num_elements };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, mul_backward_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Mul Bwd Enqueue");
            return 1;
        }
        case COMMAND_TRANSPOSE_BACKWARD: {
            TransposeBackwardCommandData* cmd = (TransposeBackwardCommandData*)data;
            if (!transpose_backward_kernel || !cmd || !cmd->buffer_dC || !cmd->buffer_dA ) { fprintf(stderr, "[C] Submit Transpose2D Bwd: Invalid args or kernel.\n"); return 0; }
            if (cmd->rows_A <= 0 || cmd->cols_A <= 0) { if ((size_t)cmd->rows_A * cmd->cols_A == 0) return 1; fprintf(stderr, "[C] Submit Transpose2D Bwd: Invalid dimensions.\n"); return 0; }
            cl_mem dC = (cl_mem)cmd->buffer_dC; cl_mem dA = (cl_mem)cmd->buffer_dA;
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 0, sizeof(cl_mem), &dC), "Transpose Bwd (2D) Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 1, sizeof(cl_mem), &dA), "Transpose Bwd (2D) Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 2, sizeof(cl_int), &cmd->rows_A), "Transpose Bwd (2D) Arg 2");
            CHECK_CL_ERR(clSetKernelArg(transpose_backward_kernel, 3, sizeof(cl_int), &cmd->cols_A), "Transpose Bwd (2D) Arg 3");
            size_t gws[2] = { (size_t)cmd->rows_A, (size_t)cmd->cols_A };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_backward_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "Transpose Bwd (2D) Enqueue");
            return 1;
        }
        case COMMAND_EMBEDDING_LOOKUP: {
            EmbeddingLookupCommandData* cmd = (EmbeddingLookupCommandData*)data;
            if (!embedding_lookup_kernel || !cmd || !cmd->idx || !cmd->w || !cmd->o) { fprintf(stderr, "[C] Submit Embed Lookup: Invalid args or kernel.\n"); return 0; }
            if (cmd->b <= 0 || cmd->s <= 0) { if ((size_t)cmd->b * cmd->s == 0) return 1; fprintf(stderr, "[C] Submit Embed Lookup: Invalid dimensions B/S.\n"); return 0; }
            if (cmd->d <= 0 || cmd->v <= 0) { fprintf(stderr, "[C] Submit Embed Lookup: Invalid dimensions D/V.\n"); return 0; }
            cl_mem idx_mem = (cl_mem)cmd->idx, w_mem = (cl_mem)cmd->w, o_mem = (cl_mem)cmd->o;
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 0, sizeof(cl_mem), &idx_mem), "Embedding Lookup Arg 0");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 1, sizeof(cl_mem), &w_mem), "Embedding Lookup Arg 1");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 2, sizeof(cl_mem), &o_mem), "Embedding Lookup Arg 2");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 3, sizeof(cl_int), &cmd->s), "Embedding Lookup Arg 3");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 4, sizeof(cl_int), &cmd->d), "Embedding Lookup Arg 4");
            CHECK_CL_ERR(clSetKernelArg(embedding_lookup_kernel, 5, sizeof(cl_int), &cmd->v), "Embedding Lookup Arg 5");
            size_t gws[2] = { (size_t)cmd->s, (size_t)cmd->b };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, embedding_lookup_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "Embedding Lookup Enqueue");
            return 1;
        }
        case COMMAND_EMBEDDING_BACKWARD_PASS1: {
            EmbeddingBackwardPass1CommandData* cmd = (EmbeddingBackwardPass1CommandData*)data;
            if (!embedding_backward_calc_delta_local_kernel || !cmd || !cmd->d_o || !cmd->idx || !cmd->delta_dw) { fprintf(stderr, "[C] Submit Embed Bwd P1: Invalid args or kernel.\n"); return 0; }
             if (cmd->b <= 0 || cmd->s <= 0) { if ((size_t)cmd->b * cmd->s == 0) return 1; fprintf(stderr, "[C] Submit Embed Bwd P1: Invalid dimensions B/S.\n"); return 0; }
             if (cmd->d <= 0 || cmd->v <= 0) { if ((size_t)cmd->v * cmd->d == 0) return 1; fprintf(stderr, "[C] Submit Embed Bwd P1: Invalid dimensions D/V.\n"); return 0; }
            cl_mem d_o_mem = (cl_mem)cmd->d_o; cl_mem idx_mem = (cl_mem)cmd->idx; cl_mem delta_dw_mem = (cl_mem)cmd->delta_dw;
            if (get_reduction_params_helper(&lws_reduce, &local_mem_bytes) != CL_SUCCESS) { fprintf(stderr, "[C] Submit Embed Bwd P1: Failed to get reduction parameters.\n"); return 0; }
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 0, sizeof(cl_mem), &d_o_mem), "Embed Bwd P1 Arg 0");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 1, sizeof(cl_mem), &idx_mem), "Embed Bwd P1 Arg 1");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 2, sizeof(cl_mem), &delta_dw_mem), "Embed Bwd P1 Arg 2");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 3, sizeof(cl_int), &cmd->b), "Embed Bwd P1 Arg 3 (B)");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 4, sizeof(cl_int), &cmd->s), "Embed Bwd P1 Arg 4 (S)");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 5, sizeof(cl_int), &cmd->d), "Embed Bwd P1 Arg 5 (D)");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 6, sizeof(cl_int), &cmd->v), "Embed Bwd P1 Arg 6 (V)");
            CHECK_CL_ERR(clSetKernelArg(embedding_backward_calc_delta_local_kernel, 7, local_mem_bytes, NULL), "Embed Bwd P1 Arg 7 (Local Mem)");
            size_t num_groups = (size_t)cmd->v * cmd->d;
            if (num_groups == 0) return 1;
            size_t gws_aligned[1] = { num_groups * lws_reduce };
            size_t lws[1] = { lws_reduce };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, embedding_backward_calc_delta_local_kernel, 1, NULL, gws_aligned, lws, 0, NULL, NULL), "Embed Bwd P1 Enqueue");
            return 1;
        }
        case COMMAND_REDUCE_SUM_AXIS01: {
            ReduceSumCommandData* cmd = (ReduceSumCommandData*)data;
            if (!reduce_sum_kernel || !cmd || !cmd->in || !cmd->out) { fprintf(stderr, "[C] Submit ReduceSum01: Invalid args or kernel.\n"); return 0; }
            if (cmd->B <= 0 || cmd->M <= 0 || cmd->N <= 0) { if ((size_t)cmd->B * cmd->M == 0 || cmd->N == 0) return 1; fprintf(stderr, "[C] Submit ReduceSum01: Invalid dimensions.\n"); return 0; }
            cl_mem in_mem = (cl_mem)cmd->in, out_mem = (cl_mem)cmd->out;
            if (get_reduction_params_helper(&lws_reduce, &local_mem_bytes) != CL_SUCCESS) { fprintf(stderr, "[C] Submit ReduceSum01: Failed to get reduction parameters.\n"); return 0; }
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 0, sizeof(cl_mem), &in_mem), "ReduceSum Arg 0");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 1, sizeof(cl_mem), &out_mem), "ReduceSum Arg 1");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 2, sizeof(cl_int), &cmd->B), "ReduceSum Arg 2");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 3, sizeof(cl_int), &cmd->M), "ReduceSum Arg 3");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 4, sizeof(cl_int), &cmd->N), "ReduceSum Arg 4");
            CHECK_CL_ERR(clSetKernelArg(reduce_sum_kernel, 5, local_mem_bytes, NULL), "ReduceSum Arg 5 (Local Mem)");
            size_t num_groups = (size_t)cmd->N;
            size_t gws[1] = { num_groups * lws_reduce };
            size_t lws[1] = { lws_reduce };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, reduce_sum_kernel, 1, NULL, gws, lws, 0, NULL, NULL), "ReduceSum Axis01 Enqueue");
            return 1;
        }
        case COMMAND_BROADCAST_ADD_BIAS: {
            BroadcastAddCommandData* cmd = (BroadcastAddCommandData*)data;
            if (!broadcast_add_kernel || !cmd || !cmd->a || !cmd->b || !cmd->c) { fprintf(stderr, "[C] Submit BroadcastAdd: Invalid args or kernel.\n"); return 0; }
            if (cmd->B <= 0 || cmd->M <= 0 || cmd->N <= 0) { if ((size_t)cmd->B * cmd->M * cmd->N == 0) return 1; fprintf(stderr, "[C] Submit BroadcastAdd: Invalid dimensions.\n"); return 0; }
            cl_mem a = (cl_mem)cmd->a, b_bias = (cl_mem)cmd->b, c = (cl_mem)cmd->c;
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 0, sizeof(cl_mem), &a), "BroadcastAdd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 1, sizeof(cl_mem), &b_bias), "BroadcastAdd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 2, sizeof(cl_mem), &c), "BroadcastAdd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 3, sizeof(cl_int), &cmd->M), "BroadcastAdd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(broadcast_add_kernel, 4, sizeof(cl_int), &cmd->N), "BroadcastAdd Arg 4");
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->M, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, broadcast_add_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "BroadcastAdd Enqueue");
            return 1;
        }
        case COMMAND_TRANSPOSE_BATCHED: {
            TransposeBatchedCommandData* cmd = (TransposeBatchedCommandData*)data;
            if (!transpose_batched_kernel || !cmd || !cmd->in || !cmd->out) { fprintf(stderr, "[C] Submit TransposeBatched: Invalid args or kernel.\n"); return 0; }
            if (cmd->B_flat <= 0 || cmd->d1 <= 0 || cmd->d2 <= 0) { if ((size_t)cmd->B_flat * cmd->d1 * cmd->d2 == 0) return 1; fprintf(stderr, "[C] Submit TransposeBatched: Invalid dimensions.\n"); return 0; }
            cl_mem in_mem = (cl_mem)cmd->in, out_mem = (cl_mem)cmd->out;
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 0, sizeof(cl_mem), &in_mem), "TransposeBatched Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 1, sizeof(cl_mem), &out_mem), "TransposeBatched Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 2, sizeof(cl_int), &cmd->d1), "TransposeBatched Arg 2");
            CHECK_CL_ERR(clSetKernelArg(transpose_batched_kernel, 3, sizeof(cl_int), &cmd->d2), "TransposeBatched Arg 3");
            size_t gws[3] = { (size_t)cmd->d2, (size_t)cmd->d1, (size_t)cmd->B_flat };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_batched_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "TransposeBatched (LastTwo) Enqueue");
            return 1;
        }
        case COMMAND_MATRIX_MULTIPLY_BATCHED: {
            BMMBatchedCommandData* cmd = (BMMBatchedCommandData*)data;
             if (!matmul_batched_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_b || !cmd->buffer_c) { fprintf(stderr, "[C] Submit BMM Batched: Invalid args or kernel.\n"); return 0;}
             if (cmd->B <= 0 || cmd->M <= 0 || cmd->N <= 0) { if ((size_t)cmd->B * cmd->M * cmd->N == 0) return 1; fprintf(stderr, "[C] Submit BMM Batched: Invalid dimensions B/M/N.\n"); return 0;}
             if (cmd->K <= 0) { fprintf(stderr, "[C] Submit BMM Batched: Invalid dimension K.\n"); return 0;}
            cl_mem a = (cl_mem)cmd->buffer_a, b = (cl_mem)cmd->buffer_b, c = (cl_mem)cmd->buffer_c;
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 0, sizeof(cl_mem), &a), "BMM Batched Fwd Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 1, sizeof(cl_mem), &b), "BMM Batched Fwd Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 2, sizeof(cl_mem), &c), "BMM Batched Fwd Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 3, sizeof(cl_int), &cmd->B), "BMM Batched Fwd Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 4, sizeof(cl_int), &cmd->M), "BMM Batched Fwd Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 5, sizeof(cl_int), &cmd->N), "BMM Batched Fwd Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_kernel, 6, sizeof(cl_int), &cmd->K), "BMM Batched Fwd Arg 6");
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->M, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_batched_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "BMM Batched Fwd Enqueue");
            return 1;
        }
        case COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DA: {
            BMMBatchedBackwardData* cmd = (BMMBatchedBackwardData*)data;
            if (!matmul_batched_backward_da_kernel || !cmd || !cmd->buffer_dc || !cmd->buffer_b || !cmd->buffer_da) { fprintf(stderr, "[C] Submit BMM Batched dA: Invalid args or kernel.\n"); return 0; }
            if (cmd->B <= 0 || cmd->M <= 0 || cmd->K <= 0) { if ((size_t)cmd->B * cmd->M * cmd->K == 0) return 1; fprintf(stderr, "[C] Submit BMM Batched dA: Invalid dimensions B/M/K.\n"); return 0; }
            if (cmd->N <= 0) { fprintf(stderr, "[C] Submit BMM Batched dA: Invalid dimension N.\n"); return 0; }
            cl_mem dc = (cl_mem)cmd->buffer_dc, b_in = (cl_mem)cmd->buffer_b, da = (cl_mem)cmd->buffer_da;
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 0, sizeof(cl_mem), &dc), "MatMul Batched dA Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 1, sizeof(cl_mem), &b_in), "MatMul Batched dA Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 2, sizeof(cl_mem), &da), "MatMul Batched dA Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul Batched dA Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul Batched dA Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul Batched dA Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_da_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul Batched dA Arg 6");
            size_t gws[3] = { (size_t)cmd->K, (size_t)cmd->M, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_batched_backward_da_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "MatMul Batched dA Enqueue");
            return 1;
        }
        case COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DB: {
            BMMBatchedBackwardData* cmd = (BMMBatchedBackwardData*)data;
            if (!matmul_batched_backward_db_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_dc || !cmd->buffer_db) { fprintf(stderr, "[C] Submit BMM Batched dB: Invalid args or kernel.\n"); return 0; }
             if (cmd->B <= 0 || cmd->K <= 0 || cmd->N <= 0) { if ((size_t)cmd->B * cmd->K * cmd->N == 0) return 1; fprintf(stderr, "[C] Submit BMM Batched dB: Invalid dimensions B/K/N.\n"); return 0; }
             if (cmd->M <= 0) { fprintf(stderr, "[C] Submit BMM Batched dB: Invalid dimension M.\n"); return 0; }
            cl_mem a_in = (cl_mem)cmd->buffer_a, dc = (cl_mem)cmd->buffer_dc, db = (cl_mem)cmd->buffer_db;
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 0, sizeof(cl_mem), &a_in), "MatMul Batched dB Arg 0");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 1, sizeof(cl_mem), &dc), "MatMul Batched dB Arg 1");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 2, sizeof(cl_mem), &db), "MatMul Batched dB Arg 2");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 3, sizeof(cl_int), &cmd->B), "MatMul Batched dB Arg 3");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 4, sizeof(cl_int), &cmd->M), "MatMul Batched dB Arg 4");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 5, sizeof(cl_int), &cmd->N), "MatMul Batched dB Arg 5");
            CHECK_CL_ERR(clSetKernelArg(matmul_batched_backward_db_kernel, 6, sizeof(cl_int), &cmd->K), "MatMul Batched dB Arg 6");
            size_t gws[3] = { (size_t)cmd->N, (size_t)cmd->K, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, matmul_batched_backward_db_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "MatMul Batched dB Enqueue");
            return 1;
        }
        case COMMAND_TRANSPOSE_12_BATCHED: {
            Transpose12BatchedCommandData* cmd = (Transpose12BatchedCommandData*)data;
            if (!transpose_12_batched_kernel || !cmd || !cmd->in || !cmd->out) { fprintf(stderr, "[C] Submit Transpose12B: Invalid args or kernel.\n"); return 0; }
            if (cmd->B <= 0 || cmd->D1 <= 0 || cmd->D2 <= 0 || cmd->D3 <= 0) { if ((size_t)cmd->B * cmd->D1 * cmd->D2 * cmd->D3 == 0) return 1; fprintf(stderr, "[C] Submit Transpose12B: Invalid dimensions.\n"); return 0; }
            cl_mem in_mem = (cl_mem)cmd->in; cl_mem out_mem = (cl_mem)cmd->out;
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 0, sizeof(cl_mem), &in_mem), "Transpose12 Arg 0");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 1, sizeof(cl_mem), &out_mem), "Transpose12 Arg 1");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 2, sizeof(cl_int), &cmd->B), "Transpose12 Arg 2");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 3, sizeof(cl_int), &cmd->D1), "Transpose12 Arg 3");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 4, sizeof(cl_int), &cmd->D2), "Transpose12 Arg 4");
            CHECK_CL_ERR(clSetKernelArg(transpose_12_batched_kernel, 5, sizeof(cl_int), &cmd->D3), "Transpose12 Arg 5");
            size_t gws[3] = { (size_t)cmd->D3, (size_t)cmd->D1, (size_t)cmd->D2 * cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, transpose_12_batched_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "Transpose12Batched Enqueue");
            return 1;
        }
        case COMMAND_LOG_SOFTMAX_STABLE: {
            LogSoftmaxStableCommandData* cmd = (LogSoftmaxStableCommandData*)data;
            if (!log_softmax_kernel || !cmd || !cmd->input_logits || !cmd->output_log_probs) { fprintf(stderr, "[C] Submit LogSoftmax: Invalid args or kernel.\n"); return 0; }
            if (cmd->B_S_rows <= 0 || cmd->V_cols <= 0) { if (cmd->B_S_rows == 0) return 1; fprintf(stderr, "[C] Submit LogSoftmax: Invalid dimensions.\n"); return 0; }
            cl_mem in_logits = (cl_mem)cmd->input_logits; cl_mem out_log_probs = (cl_mem)cmd->output_log_probs;
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 0, sizeof(cl_mem), &in_logits), "LogSoftmaxStable Arg 0");
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 1, sizeof(cl_mem), &out_log_probs), "LogSoftmaxStable Arg 1");
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 2, sizeof(cl_int), &cmd->B_S_rows), "LogSoftmaxStable Arg 2");
            CHECK_CL_ERR(clSetKernelArg(log_softmax_kernel, 3, sizeof(cl_int), &cmd->V_cols), "LogSoftmaxStable Arg 3");
            size_t gws[1] = { (size_t)cmd->B_S_rows };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, log_softmax_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "LogSoftmaxStable Enqueue");
            return 1;
        }
		case COMMAND_CROSS_ENTROPY_LOSS_GRAD: {
            CrossEntropyLossGradCommandData* cmd = (CrossEntropyLossGradCommandData*)data;
            if (!cross_entropy_kernel || !cmd || !cmd->log_probs || !cmd->target_indices || !cmd->grad_input || !cmd->loss_per_sample) { fprintf(stderr, "[C] Submit CrossEntropy: Invalid args or kernel.\n"); return 0; }
            if (cmd->B_S_rows <= 0 || cmd->V_cols <= 0) { if (cmd->B_S_rows == 0) return 1; fprintf(stderr, "[C] Submit CrossEntropy: Invalid dimensions.\n"); return 0; }
            cl_mem log_probs_mem = (cl_mem)cmd->log_probs; cl_mem target_indices_mem = (cl_mem)cmd->target_indices; cl_mem grad_input_mem = (cl_mem)cmd->grad_input; cl_mem loss_per_sample_mem = (cl_mem)cmd->loss_per_sample;
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 0, sizeof(cl_mem), &log_probs_mem), "CrossEntropyLossGrad Arg 0");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 1, sizeof(cl_mem), &target_indices_mem), "CrossEntropyLossGrad Arg 1");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 2, sizeof(cl_mem), &grad_input_mem), "CrossEntropyLossGrad Arg 2");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 3, sizeof(cl_mem), &loss_per_sample_mem), "CrossEntropyLossGrad Arg 3");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 4, sizeof(cl_int), &cmd->B_S_rows), "CrossEntropyLossGrad Arg 4 (num_rows)");
            CHECK_CL_ERR(clSetKernelArg(cross_entropy_kernel, 5, sizeof(cl_int), &cmd->V_cols), "CrossEntropyLossGrad Arg 5 (V)");
            size_t gws[1] = { (size_t)cmd->B_S_rows };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, cross_entropy_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "CrossEntropyLossGrad Enqueue");
            return 1;
        }
        case COMMAND_ADD_BROADCAST_PE: {
            AddBroadcastPECommandData* cmd = (AddBroadcastPECommandData*)data;
            if (!add_broadcast_pe_kernel || !cmd || !cmd->input || !cmd->pe_slice || !cmd->output) { fprintf(stderr, "[C] Submit AddBroadcastPE: Invalid args or kernel.\n"); return 0; }
            if (cmd->B <= 0 || cmd->S <= 0 || cmd->E <= 0) { if ((size_t)cmd->B * cmd->S * cmd->E == 0) return 1; fprintf(stderr, "[C] Submit AddBroadcastPE: Invalid dimensions.\n"); return 0; }
            cl_mem input_mem = (cl_mem)cmd->input; cl_mem pe_slice_mem = (cl_mem)cmd->pe_slice; cl_mem output_mem = (cl_mem)cmd->output;
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 0, sizeof(cl_mem), &input_mem), "AddBroadcastPE Arg 0");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 1, sizeof(cl_mem), &pe_slice_mem), "AddBroadcastPE Arg 1");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 2, sizeof(cl_mem), &output_mem), "AddBroadcastPE Arg 2");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 3, sizeof(cl_int), &cmd->S), "AddBroadcastPE Arg 3");
            CHECK_CL_ERR(clSetKernelArg(add_broadcast_pe_kernel, 4, sizeof(cl_int), &cmd->E), "AddBroadcastPE Arg 4");
            size_t gws[3] = { (size_t)cmd->E, (size_t)cmd->S, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, add_broadcast_pe_kernel, 3, NULL, gws, NULL, 0, NULL, NULL), "AddBroadcastPE Enqueue");
            return 1;
        }
        case COMMAND_HEBBIAN_OUTER_PRODUCT_UPDATE: {
            HebbianUpdateLocalReduceCommandData* cmd = (HebbianUpdateLocalReduceCommandData*)data;
            if (!hebbian_update_local_reduce_kernel || !cmd || !cmd->buffer_a || !cmd->buffer_c || !cmd->buffer_w) { fprintf(stderr, "[C] Submit HebbianLR: Invalid args or kernel.\n"); return 0; }
            if (cmd->K <= 0 || cmd->N <= 0) { if ((size_t)cmd->K * cmd->N == 0) return 1; fprintf(stderr, "[C] Submit HebbianLR: Invalid dimensions K/N.\n"); return 0; }
            if (cmd->B <= 0 || cmd->M <= 0) { fprintf(stderr, "[C] Submit HebbianLR: Invalid dimensions B/M.\n"); return 0; }
            cl_mem a_mem = (cl_mem)cmd->buffer_a; cl_mem c_mem = (cl_mem)cmd->buffer_c; cl_mem w_mem = (cl_mem)cmd->buffer_w;
            if (get_reduction_params_helper(&lws_reduce, &local_mem_bytes) != CL_SUCCESS) { fprintf(stderr, "[C] Submit HebbianLR: Failed to get reduction parameters.\n"); return 0; }
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 0, sizeof(cl_mem), &a_mem), "HebbianLR Arg 0 (A)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 1, sizeof(cl_mem), &c_mem), "HebbianLR Arg 1 (C)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 2, sizeof(cl_mem), &w_mem), "HebbianLR Arg 2 (W)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 3, sizeof(cl_float), &cmd->learning_rate), "HebbianLR Arg 3 (LR)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 4, sizeof(cl_int), &cmd->B), "HebbianLR Arg 4 (B)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 5, sizeof(cl_int), &cmd->M), "HebbianLR Arg 5 (M)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 6, sizeof(cl_int), &cmd->N), "HebbianLR Arg 6 (N)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 7, sizeof(cl_int), &cmd->K), "HebbianLR Arg 7 (K)");
            CHECK_CL_ERR(clSetKernelArg(hebbian_update_local_reduce_kernel, 8, local_mem_bytes, NULL), "HebbianLR Arg 8 (Local Mem)");
            size_t num_groups = (size_t)cmd->K * cmd->N;
            if (num_groups == 0) return 1;
            size_t gws_aligned[1] = { num_groups * lws_reduce };
            size_t lws[1] = { lws_reduce };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, hebbian_update_local_reduce_kernel, 1, NULL, gws_aligned, lws, 0, NULL, NULL), "Hebbian Update Local Reduce Enqueue");
            return 1;
        }
        case COMMAND_THRESHOLD_SPIKE: {
            ThresholdSpikeCommandData* cmd = (ThresholdSpikeCommandData*)data;
            if (!threshold_spike_kernel || !cmd || !cmd->buffer_activations || !cmd->buffer_spikes) { fprintf(stderr, "[C] Submit ThresholdSpike: Invalid args or kernel.\n"); return 0; }
            if (cmd->num_elements <= 0) { if (cmd->num_elements == 0) return 1; fprintf(stderr, "[C] Submit ThresholdSpike: Invalid dimensions.\n"); return 0; }
            cl_mem act_mem = (cl_mem)cmd->buffer_activations; cl_mem spk_mem = (cl_mem)cmd->buffer_spikes;
            CHECK_CL_ERR(clSetKernelArg(threshold_spike_kernel, 0, sizeof(cl_mem), &act_mem), "Threshold Spike Arg 0");
            CHECK_CL_ERR(clSetKernelArg(threshold_spike_kernel, 1, sizeof(cl_mem), &spk_mem), "Threshold Spike Arg 1");
            CHECK_CL_ERR(clSetKernelArg(threshold_spike_kernel, 2, sizeof(cl_float), &cmd->threshold), "Threshold Spike Arg 2");
            CHECK_CL_ERR(clSetKernelArg(threshold_spike_kernel, 3, sizeof(cl_int), &cmd->num_elements), "Threshold Spike Arg 3");
            size_t gws[1] = { (size_t)cmd->num_elements };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, threshold_spike_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Threshold Spike Enqueue");
            return 1;
        }
        case COMMAND_ADD_BIAS_MN: {
             AddBiasMNCommandData* cmd = (AddBiasMNCommandData*)data;
             if (!add_bias_mn_kernel || !cmd || !cmd->a_or_c || !cmd->b_bias) { fprintf(stderr, "[C] Submit AddBiasMN: Invalid args or kernel.\n"); return 0; }
             if (cmd->M <= 0 || cmd->N <= 0) { if ((size_t)cmd->M * cmd->N == 0) return 1; fprintf(stderr, "[C] Submit AddBiasMN: Invalid dimensions.\n"); return 0; }
             cl_mem a_or_c_mem = (cl_mem)cmd->a_or_c; cl_mem b_bias_mem = (cl_mem)cmd->b_bias;
             CHECK_CL_ERR(clSetKernelArg(add_bias_mn_kernel, 0, sizeof(cl_mem), &a_or_c_mem), "AddBiasMN Arg 0 (A)");
             CHECK_CL_ERR(clSetKernelArg(add_bias_mn_kernel, 1, sizeof(cl_mem), &b_bias_mem), "AddBiasMN Arg 1 (B)");
             CHECK_CL_ERR(clSetKernelArg(add_bias_mn_kernel, 2, sizeof(cl_mem), &a_or_c_mem), "AddBiasMN Arg 2 (C)");
             CHECK_CL_ERR(clSetKernelArg(add_bias_mn_kernel, 3, sizeof(cl_int), &cmd->M), "AddBiasMN Arg 3 (M)");
             CHECK_CL_ERR(clSetKernelArg(add_bias_mn_kernel, 4, sizeof(cl_int), &cmd->N), "AddBiasMN Arg 4 (N)");
             size_t gws[2] = { (size_t)cmd->N, (size_t)cmd->M };
             CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, add_bias_mn_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "AddBiasMN Enqueue");
             return 1;
        }
        case COMMAND_DYNAMIC_TOKEN_ASSIGNMENT: {
            DynamicTokenAssignmentCommandData* cmd = (DynamicTokenAssignmentCommandData*)data;
            if (!dynamic_token_assign_kernel || !cmd || !cmd->activations_bse || !cmd->prototypes_te || !cmd->output_indices_bs) { fprintf(stderr, "[C] Submit DynTokenAssign: Invalid args or kernel.\n"); return 0; }
            if (cmd->B <= 0 || cmd->S <= 0) { if ((size_t)cmd->B * cmd->S == 0) return 1; fprintf(stderr, "[C] Submit DynTokenAssign: Invalid dimensions B/S.\n"); return 0; }
            if (cmd->E <= 0 || cmd->T <= 0) { fprintf(stderr, "[C] Submit DynTokenAssign: Invalid dimensions E/T.\n"); return 0; }
            cl_mem act_mem = (cl_mem)cmd->activations_bse; cl_mem proto_mem = (cl_mem)cmd->prototypes_te; cl_mem idx_mem = (cl_mem)cmd->output_indices_bs;
            CHECK_CL_ERR(clSetKernelArg(dynamic_token_assign_kernel, 0, sizeof(cl_mem), &act_mem), "DynToken Assign Arg 0");
            CHECK_CL_ERR(clSetKernelArg(dynamic_token_assign_kernel, 1, sizeof(cl_mem), &proto_mem), "DynToken Assign Arg 1");
            CHECK_CL_ERR(clSetKernelArg(dynamic_token_assign_kernel, 2, sizeof(cl_mem), &idx_mem), "DynToken Assign Arg 2");
            CHECK_CL_ERR(clSetKernelArg(dynamic_token_assign_kernel, 3, sizeof(cl_int), &cmd->S), "DynToken Assign Arg 3");
            CHECK_CL_ERR(clSetKernelArg(dynamic_token_assign_kernel, 4, sizeof(cl_int), &cmd->E), "DynToken Assign Arg 4");
            CHECK_CL_ERR(clSetKernelArg(dynamic_token_assign_kernel, 5, sizeof(cl_int), &cmd->T), "DynToken Assign Arg 5");
            size_t gws[2] = { (size_t)cmd->S, (size_t)cmd->B };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, dynamic_token_assign_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "DynToken Assign Enqueue");
            return 1;
        }
        case COMMAND_PAIRWISE_SIMILARITY: {
            PairwiseSimilarityCommandData* cmd = (PairwiseSimilarityCommandData*)data;
            if (!pairwise_similarity_kernel || !cmd || !cmd->states_nd || !cmd->output_similarity_nn) { fprintf(stderr, "[C] Submit PairwiseSim: Invalid args or kernel.\n"); return 0; }
            if (cmd->N <= 0) { if (cmd->N == 0) return 1; fprintf(stderr, "[C] Submit PairwiseSim: Invalid dimension N.\n"); return 0; }
            if (cmd->D <= 0) { fprintf(stderr, "[C] Submit PairwiseSim: Invalid dimension D.\n"); return 0; }
            cl_mem states_mem = (cl_mem)cmd->states_nd; cl_mem sim_mem = (cl_mem)cmd->output_similarity_nn;
            CHECK_CL_ERR(clSetKernelArg(pairwise_similarity_kernel, 0, sizeof(cl_mem), &states_mem), "PairwiseSim Arg 0");
            CHECK_CL_ERR(clSetKernelArg(pairwise_similarity_kernel, 1, sizeof(cl_mem), &sim_mem), "PairwiseSim Arg 1");
            CHECK_CL_ERR(clSetKernelArg(pairwise_similarity_kernel, 2, sizeof(cl_int), &cmd->N), "PairwiseSim Arg 2");
            CHECK_CL_ERR(clSetKernelArg(pairwise_similarity_kernel, 3, sizeof(cl_int), &cmd->D), "PairwiseSim Arg 3");
            size_t gws[2] = { (size_t)cmd->N, (size_t)cmd->N };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, pairwise_similarity_kernel, 2, NULL, gws, NULL, 0, NULL, NULL), "PairwiseSim Enqueue");
            return 1;
        }
        case COMMAND_PROTO_SEGMENTED_SUM: {
            ProtoSegmentedSumCommandData* cmd = (ProtoSegmentedSumCommandData*)data;
            if (!proto_segmented_sum_kernel || !cmd || !cmd->activations_flat || !cmd->indices_flat || !cmd->proto_sums || !cmd->proto_counts) { fprintf(stderr, "[C] Submit Proto Segmented Sum: Error - Invalid arguments or kernel handle missing.\n"); return 0; }
            if (!has_atomics_support) { fprintf(stderr, "[C] Submit Proto Segmented Sum: Error - Required atomic operations not supported by the device/driver! Cannot execute.\n"); return 0; }
            if (cmd->M_flat <= 0) { if (cmd->M_flat == 0) return 1; fprintf(stderr, "[C] Submit Proto Segmented Sum: Invalid dimension M_flat.\n"); return 0;}
            if (cmd->E <= 0 || cmd->T <= 0) { fprintf(stderr, "[C] Submit Proto Segmented Sum: Invalid dimensions E/T.\n"); return 0;}
            cl_mem act_mem = (cl_mem)cmd->activations_flat; cl_mem idx_mem = (cl_mem)cmd->indices_flat; cl_mem sums_mem = (cl_mem)cmd->proto_sums; cl_mem counts_mem = (cl_mem)cmd->proto_counts;
            CHECK_CL_ERR(clSetKernelArg(proto_segmented_sum_kernel, 0, sizeof(cl_mem), &act_mem), "ProtoSum Arg 0");
            CHECK_CL_ERR(clSetKernelArg(proto_segmented_sum_kernel, 1, sizeof(cl_mem), &idx_mem), "ProtoSum Arg 1");
            CHECK_CL_ERR(clSetKernelArg(proto_segmented_sum_kernel, 2, sizeof(cl_mem), &sums_mem), "ProtoSum Arg 2");
            CHECK_CL_ERR(clSetKernelArg(proto_segmented_sum_kernel, 3, sizeof(cl_mem), &counts_mem), "ProtoSum Arg 3");
            CHECK_CL_ERR(clSetKernelArg(proto_segmented_sum_kernel, 4, sizeof(cl_int), &cmd->M_flat), "ProtoSum Arg 4");
            CHECK_CL_ERR(clSetKernelArg(proto_segmented_sum_kernel, 5, sizeof(cl_int), &cmd->E), "ProtoSum Arg 5");
            CHECK_CL_ERR(clSetKernelArg(proto_segmented_sum_kernel, 6, sizeof(cl_int), &cmd->T), "ProtoSum Arg 6");
            size_t gws[1] = { (size_t)cmd->M_flat };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, proto_segmented_sum_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Proto Segmented Sum Enqueue");
            return 1;
        }
        case COMMAND_PROTO_UPDATE_STEP: {
            ProtoUpdateStepCommandData* cmd = (ProtoUpdateStepCommandData*)data;
            if (!proto_update_step_kernel || !cmd || !cmd->prototypes || !cmd->proto_sums || !cmd->proto_counts) { fprintf(stderr, "[C] Submit Proto Update Step: Error - Invalid arguments or kernel handle missing.\n"); return 0; }
            if (cmd->T <= 0) { if (cmd->T == 0) return 1; fprintf(stderr, "[C] Submit Proto Update Step: Invalid dimension T.\n"); return 0;}
            if (cmd->E <= 0) { fprintf(stderr, "[C] Submit Proto Update Step: Invalid dimension E.\n"); return 0;}
            if (cmd->learning_rate < 0.0f || cmd->learning_rate > 1.0f) { fprintf(stderr, "[C] Submit Proto Update Step: Warning - Invalid learning_rate (%f). Should be in [0, 1].\n", cmd->learning_rate); }
            cl_mem proto_mem = (cl_mem)cmd->prototypes; cl_mem sums_mem = (cl_mem)cmd->proto_sums; cl_mem counts_mem = (cl_mem)cmd->proto_counts;
            CHECK_CL_ERR(clSetKernelArg(proto_update_step_kernel, 0, sizeof(cl_mem), &proto_mem), "ProtoUpdate Arg 0");
            CHECK_CL_ERR(clSetKernelArg(proto_update_step_kernel, 1, sizeof(cl_mem), &sums_mem), "ProtoUpdate Arg 1");
            CHECK_CL_ERR(clSetKernelArg(proto_update_step_kernel, 2, sizeof(cl_mem), &counts_mem), "ProtoUpdate Arg 2");
            CHECK_CL_ERR(clSetKernelArg(proto_update_step_kernel, 3, sizeof(cl_float), &cmd->learning_rate), "ProtoUpdate Arg 3");
            CHECK_CL_ERR(clSetKernelArg(proto_update_step_kernel, 4, sizeof(cl_int), &cmd->E), "ProtoUpdate Arg 4");
            CHECK_CL_ERR(clSetKernelArg(proto_update_step_kernel, 5, sizeof(cl_int), &cmd->T), "ProtoUpdate Arg 5");
            size_t gws[1] = { (size_t)cmd->T };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, proto_update_step_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Proto Update Step Enqueue");
            return 1;
        }
        case COMMAND_SHAPE_LOSS_REWARD_PENALTY: {
            ShapeLossRewardPenaltyCommandData* cmd = (ShapeLossRewardPenaltyCommandData*)data;
            if (!shape_loss_reward_penalty_kernel || !cmd || !cmd->loss_per_sample_in || !cmd->predictions || !cmd->targets || !cmd->loss_per_sample_out) {
                fprintf(stderr, "[C] Submit ShapeLoss: Invalid args or kernel.\n"); return 0;
            }
            if (cmd->num_samples <= 0 || cmd->num_classes <= 0) {
                if (cmd->num_samples == 0) return 1;
                fprintf(stderr, "[C] Submit ShapeLoss: Invalid dimensions (samples=%d, classes=%d).\n", cmd->num_samples, cmd->num_classes); return 0;
            }
             if (cmd->penalty_weight < 0.0f || cmd->reward_weight < 0.0f || cmd->high_confidence_threshold < 0.0f || cmd->high_confidence_threshold > 1.0f || cmd->critical_target_class < 0 || cmd->critical_target_class >= cmd->num_classes || cmd->critical_predicted_class < 0 || cmd->critical_predicted_class >= cmd->num_classes) {
                 fprintf(stderr, "[C] Submit ShapeLoss: Warning - Potentially invalid shaping parameters provided (penalty=%.2f, reward=%.2f, thresh=%.2f, crit_target=%d, crit_pred=%d).\n",
                         cmd->penalty_weight, cmd->reward_weight, cmd->high_confidence_threshold, cmd->critical_target_class, cmd->critical_predicted_class);
             }
            cl_mem loss_in_mem = (cl_mem)cmd->loss_per_sample_in; cl_mem pred_mem = (cl_mem)cmd->predictions; cl_mem targets_mem = (cl_mem)cmd->targets; cl_mem loss_out_mem = (cl_mem)cmd->loss_per_sample_out;
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 0, sizeof(cl_mem), &loss_in_mem), "ShapeLoss Arg 0 (loss_in)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 1, sizeof(cl_mem), &pred_mem), "ShapeLoss Arg 1 (predictions)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 2, sizeof(cl_mem), &targets_mem), "ShapeLoss Arg 2 (targets)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 3, sizeof(cl_mem), &loss_out_mem), "ShapeLoss Arg 3 (loss_out)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 4, sizeof(cl_int), &cmd->num_samples), "ShapeLoss Arg 4 (num_samples)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 5, sizeof(cl_int), &cmd->num_classes), "ShapeLoss Arg 5 (num_classes)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 6, sizeof(cl_float), &cmd->penalty_weight), "ShapeLoss Arg 6 (penalty_weight)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 7, sizeof(cl_float), &cmd->reward_weight), "ShapeLoss Arg 7 (reward_weight)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 8, sizeof(cl_float), &cmd->high_confidence_threshold), "ShapeLoss Arg 8 (high_confidence_threshold)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 9, sizeof(cl_int), &cmd->critical_target_class), "ShapeLoss Arg 9 (critical_target_class)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_kernel, 10, sizeof(cl_int), &cmd->critical_predicted_class), "ShapeLoss Arg 10 (critical_predicted_class)");
            size_t gws[1] = { (size_t)cmd->num_samples };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, shape_loss_reward_penalty_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Shape Loss Reward/Penalty Enqueue");
            return 1;
        }

        // --- NEU: Loss Shaping (List) ---
        case COMMAND_SHAPE_LOSS_REWARD_PENALTY_LIST: {
            ShapeLossRewardPenaltyListCommandData* cmd = (ShapeLossRewardPenaltyListCommandData*)data;
            if (!shape_loss_reward_penalty_list_kernel || !cmd || !cmd->loss_per_sample_in || !cmd->predictions || !cmd->targets || !cmd->loss_per_sample_out) {
                fprintf(stderr, "[C] Submit ShapeLossList: Invalid args or kernel.\n"); return 0;
            }
            // Prüfe kritischen Paar-Buffer nur, wenn Paare > 0 sind
            if (cmd->num_critical_pairs > 0 && !cmd->critical_pairs) {
                 fprintf(stderr, "[C] Submit ShapeLossList: Critical pairs buffer is NULL but count > 0.\n"); return 0;
            }
            if (cmd->num_samples <= 0 || cmd->num_classes <= 0) {
                if (cmd->num_samples == 0) return 1; // Trivial case
                fprintf(stderr, "[C] Submit ShapeLossList: Invalid dimensions (samples=%d, classes=%d).\n", cmd->num_samples, cmd->num_classes); return 0;
            }
             // Basic validation of parameters
             if (cmd->penalty_weight < 0.0f || cmd->reward_weight < 0.0f || cmd->high_confidence_threshold < 0.0f || cmd->high_confidence_threshold > 1.0f || cmd->num_critical_pairs < 0) {
                 fprintf(stderr, "[C] Submit ShapeLossList: Warning - Potentially invalid shaping parameters provided (penalty=%.2f, reward=%.2f, thresh=%.2f, num_pairs=%d).\n",
                         cmd->penalty_weight, cmd->reward_weight, cmd->high_confidence_threshold, cmd->num_critical_pairs);
             }

            cl_mem loss_in_mem = (cl_mem)cmd->loss_per_sample_in;
            cl_mem pred_mem = (cl_mem)cmd->predictions;
            cl_mem targets_mem = (cl_mem)cmd->targets;
            cl_mem loss_out_mem = (cl_mem)cmd->loss_per_sample_out;
            cl_mem crit_pairs_mem = (cl_mem)cmd->critical_pairs; // Handle zum Paar-Buffer

            // Argument Indices anpassen!
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 0, sizeof(cl_mem), &loss_in_mem), "ShapeLossList Arg 0 (loss_in)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 1, sizeof(cl_mem), &pred_mem), "ShapeLossList Arg 1 (predictions)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 2, sizeof(cl_mem), &targets_mem), "ShapeLossList Arg 2 (targets)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 3, sizeof(cl_mem), &loss_out_mem), "ShapeLossList Arg 3 (loss_out)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 4, sizeof(cl_mem), &crit_pairs_mem), "ShapeLossList Arg 4 (critical_pairs)"); // NEU
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 5, sizeof(cl_int), &cmd->num_samples), "ShapeLossList Arg 5 (num_samples)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 6, sizeof(cl_int), &cmd->num_classes), "ShapeLossList Arg 6 (num_classes)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 7, sizeof(cl_int), &cmd->num_critical_pairs), "ShapeLossList Arg 7 (num_critical_pairs)"); // NEU
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 8, sizeof(cl_float), &cmd->penalty_weight), "ShapeLossList Arg 8 (penalty_weight)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 9, sizeof(cl_float), &cmd->reward_weight), "ShapeLossList Arg 9 (reward_weight)");
            CHECK_CL_ERR(clSetKernelArg(shape_loss_reward_penalty_list_kernel, 10, sizeof(cl_float), &cmd->high_confidence_threshold), "ShapeLossList Arg 10 (high_confidence_threshold)");

            size_t gws[1] = { (size_t)cmd->num_samples };
            CHECK_CL_ERR(clEnqueueNDRangeKernel(queue, shape_loss_reward_penalty_list_kernel, 1, NULL, gws, NULL, 0, NULL, NULL), "Shape Loss Reward/Penalty List Enqueue");
            return 1;
        }
        // --- Ende NEU: Loss Shaping (List) ---

        default:
            fprintf(stderr, "[C] submit_kernel_command: Error - Unknown or unhandled command code: %d\n", command);
            return 0;
    } // end switch

    #undef CHECK_CL_ERR
    fprintf(stderr, "[C] submit_kernel_command: Error - Reached end of switch without successful command submission (Command code: %d).\n", command);
    return 0;
}

/**
 * @brief Blocks until all previously enqueued commands in the OpenCL queue have finished execution.
 */
int finish_queue_and_check(int gpu_index, const char* func_name) {
     if (!queue) { fprintf(stderr, "[C] %s: Error - Command queue is NULL. Cannot finish.\n", func_name ? func_name : "finish_queue_and_check"); return 0; }
    cl_int err = clFinish(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[C] %s: Error during clFinish after submitting commands: %s (%d)\n", func_name ? func_name : "finish_queue_and_check", clGetErrorString(err), err);
        return 0;
    }
    return 1;
}

// --- Exported Function Definitions (Wrappers for Kernel Execution) ---

DLLEXPORT int execute_matmul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_matmul_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B <= 0 || M <= 0 || N <= 0) { if ((size_t)B * M * N == 0) return 1; fprintf(stderr, "[C] execute_matmul_on_gpu: Error - Invalid non-positive dimensions (B=%d, M=%d, N=%d).\n", B, M, N); return 0; }
    if (K <= 0) { fprintf(stderr, "[C] execute_matmul_on_gpu: Error - Invalid non-positive dimension K=%d.\n", K); return 0; }
    BMMCommandData cmd_data = { buffer_a, buffer_b, buffer_c, B, M, N, K };
    if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_softmax_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_softmax_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) { if (num_rows == 0) return 1; fprintf(stderr, "[C] execute_softmax_on_gpu: Error - Invalid non-positive dimensions (rows=%d, size=%d).\n", num_rows, row_size); return 0; }
    SoftmaxCommandData cmd_data = { buffer_input, buffer_output, num_rows, row_size };
    if (!submit_kernel_command(gpu_index, COMMAND_SOFTMAX_ROWWISE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_gelu_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_elements) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_gelu_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_elements <= 0) { if (num_elements == 0) return 1; fprintf(stderr, "[C] execute_gelu_on_gpu: Error - Invalid non-positive number of elements (%d).\n", num_elements); return 0; }
    GeluCommandData cmd_data = { buffer_input, buffer_output, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_GELU_ELEMENTWISE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_add_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements) {
    if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_add_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_elements <= 0) { if (num_elements == 0) return 1; fprintf(stderr, "[C] execute_add_on_gpu: Error - Invalid non-positive number of elements (%d).\n", num_elements); return 0; }
    AddCommandData cmd_data = { buffer_a, buffer_b, buffer_c, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_ADD_ELEMENTWISE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_add_bias_on_gpu(int gpu_index, void* buffer_a_or_c, void* buffer_b_bias, int M, int N) {
    if (!buffer_a_or_c || !buffer_b_bias) { fprintf(stderr, "[C] execute_add_bias_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (M <= 0 || N <= 0) { if ((size_t)M * N == 0) return 1; fprintf(stderr, "[C] execute_add_bias_on_gpu: Error - Invalid non-positive dimensions (M=%d, N=%d).\n", M, N); return 0; }
    AddBiasMNCommandData cmd_data = { buffer_a_or_c, buffer_b_bias, M, N };
    if (!submit_kernel_command(gpu_index, COMMAND_ADD_BIAS_MN, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_mul_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int num_elements) {
    if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_mul_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_elements <= 0) { if (num_elements == 0) return 1; fprintf(stderr, "[C] execute_mul_on_gpu: Error - Invalid non-positive number of elements (%d).\n", num_elements); return 0; }
    MulCommandData cmd_data = { buffer_a, buffer_b, buffer_c, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_MUL_ELEMENTWISE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_layernorm_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int num_rows, int row_size, float eps) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_layernorm_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) { if (num_rows == 0) return 1; fprintf(stderr, "[C] execute_layernorm_on_gpu: Error - Invalid non-positive dimensions (rows=%d, size=%d).\n", num_rows, row_size); return 0; }
    float effective_eps = (eps > 0) ? eps : 1e-5f;
    LayerNormCommandData cmd_data = { buffer_input, buffer_output, num_rows, row_size, effective_eps };
    if (!submit_kernel_command(gpu_index, COMMAND_LAYER_NORM, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_clone_on_gpu(int gpu_index, void* src_buffer, void* dst_buffer, size_t size) {
    if (!src_buffer || !dst_buffer) { fprintf(stderr, "[C] execute_clone_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (size == 0) return 1;
    CloneCommandData cmd_data = { src_buffer, dst_buffer, size };
    if (!submit_kernel_command(gpu_index, COMMAND_CLONE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_transpose_on_gpu(int gpu_index, void* buffer_input, void* buffer_output, int rows, int cols) {
    if (!buffer_input || !buffer_output) { fprintf(stderr, "[C] execute_transpose_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (rows <= 0 || cols <= 0) { if ((size_t)rows * cols == 0) return 1; fprintf(stderr, "[C] execute_transpose_on_gpu: Error - Invalid non-positive dimensions (rows=%d, cols=%d).\n", rows, cols); return 0; }
    TransposeCommandData cmd_data = { buffer_input, buffer_output, rows, cols };
    if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_gelu_backward_on_gpu(int gpu_index, void* buffer_input, void* buffer_grad_output, void* buffer_grad_input, int num_elements) {
    if (!buffer_input || !buffer_grad_output || !buffer_grad_input) { fprintf(stderr, "[C] execute_gelu_backward_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_elements <= 0) { if (num_elements == 0) return 1; fprintf(stderr, "[C] execute_gelu_backward_on_gpu: Error - Invalid non-positive number of elements (%d).\n", num_elements); return 0; }
    GeluBackwardCommandData cmd_data = { buffer_input, buffer_grad_output, buffer_grad_input, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_GELU_BACKWARD_ELEMENTWISE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_matmul_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_dc) { fprintf(stderr, "[C] execute_matmul_backward_on_gpu: Error - NULL required input buffer handle provided (A, B, or dC).\n"); return 0; }
    if (!buffer_da && !buffer_db) { return 1; }
    int need_da = (buffer_da != NULL);
    int need_db = (buffer_db != NULL);
    if (B <= 0 || M <= 0 || N <= 0 || K <= 0) {
        int da_zero = need_da && ((size_t)B*M*K == 0);
        int db_zero = need_db && ((size_t)K*N == 0);
        if(need_da && need_db && (da_zero || db_zero)) { }
        else if (need_da && da_zero && !need_db) { }
        else if (need_db && db_zero && !need_da) { }
        else if (!need_da && !need_db) { return 1; }
        else {
            fprintf(stderr, "[C] execute_matmul_backward_on_gpu: Error - Invalid non-positive dimensions (B=%d, M=%d, N=%d, K=%d) for requested gradient.\n", B, M, N, K);
            return 0;
        }
    }
    MatMulBackwardData cmd_data = { buffer_a, buffer_b, buffer_dc, buffer_da, buffer_db, B, M, N, K };
    int success = 1;
    if (need_da && (size_t)B * M * K > 0) {
        if (!submit_kernel_command(gpu_index, COMMAND_MATMUL_BACKWARD_DA, &cmd_data)) { success = 0; }
    }
    if (need_db && (size_t)K * N > 0) {
        if (!submit_kernel_command(gpu_index, COMMAND_MATMUL_BACKWARD_DB, &cmd_data)) { success = 0; }
    }
    return success;
}
DLLEXPORT int execute_layernorm_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_x, void* buffer_dx, int num_rows, int row_size, float eps) {
    if (!buffer_dy || !buffer_x || !buffer_dx) { fprintf(stderr, "[C] execute_layernorm_backward_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) { if (num_rows == 0) return 1; fprintf(stderr, "[C] execute_layernorm_backward_on_gpu: Error - Invalid non-positive dimensions (rows=%d, size=%d).\n", num_rows, row_size); return 0; }
    float effective_eps = (eps > 0) ? eps : 1e-5f;
    LayerNormBackwardCommandData cmd_data = { buffer_dy, buffer_x, buffer_dx, num_rows, row_size, effective_eps };
    if (!submit_kernel_command(gpu_index, COMMAND_LAYER_NORM_BACKWARD, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_adam_update_on_gpu(int gpu_index, void* param_buffer, void* grad_buffer, void* m_buffer, void* v_buffer, int num_elements, int t, float lr, float beta1, float beta2, float eps, float weight_decay) {
    float beta1_t, beta2_t;
    if (!param_buffer || !grad_buffer || !m_buffer || !v_buffer) { fprintf(stderr, "[C] execute_adam_update_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_elements <= 0) { if (num_elements == 0) return 1; fprintf(stderr, "[C] execute_adam_update_on_gpu: Error - Invalid non-positive number of elements (%d).\n", num_elements); return 0; }
    if (t <= 0 || lr < 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || eps < 0.0f || weight_decay < 0.0f) {
         fprintf(stderr, "[C] execute_adam_update_on_gpu: Error - Invalid hyperparameters (t=%d, lr=%f, b1=%f, b2=%f, eps=%f, wd=%f).\n", t, lr, beta1, beta2, eps, weight_decay);
         return 0;
    }
    beta1_t = (float)pow((double)beta1, (double)t);
    beta2_t = (float)pow((double)beta2, (double)t);
    AdamCommandData cmd_data = { param_buffer, grad_buffer, m_buffer, v_buffer, num_elements, t, lr, beta1, beta2, eps, weight_decay, beta1_t, beta2_t };
    if (!submit_kernel_command(gpu_index, COMMAND_ADAM_UPDATE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_softmax_backward_on_gpu(int gpu_index, void* buffer_dy, void* buffer_y, void* buffer_dx, int num_rows, int row_size) {
    if (!buffer_dy || !buffer_y || !buffer_dx) { fprintf(stderr, "[C] execute_softmax_backward_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_rows <= 0 || row_size <= 0) { if (num_rows == 0) return 1; fprintf(stderr, "[C] execute_softmax_backward_on_gpu: Error - Invalid non-positive dimensions (rows=%d, size=%d).\n", num_rows, row_size); return 0; }
    SoftmaxBackwardCommandData cmd_data = { buffer_dy, buffer_y, buffer_dx, num_rows, row_size };
    if (!submit_kernel_command(gpu_index, COMMAND_SOFTMAX_BACKWARD, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_mul_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_A, void* buffer_B, void* buffer_dA, void* buffer_dB, int num_elements) {
    if (!buffer_dC || !buffer_A || !buffer_B) { fprintf(stderr, "[C] execute_mul_backward_on_gpu: Error - NULL required input buffer handle provided (dC, A, or B).\n"); return 0; }
    if (!buffer_dA && !buffer_dB) { return 1; }
    if (num_elements <= 0) { if (num_elements == 0) return 1; fprintf(stderr, "[C] execute_mul_backward_on_gpu: Error - Invalid non-positive number of elements (%d).\n", num_elements); return 0; }
    MulBackwardCommandData cmd_data = { buffer_dC, buffer_A, buffer_B, buffer_dA, buffer_dB, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_MUL_BACKWARD, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_transpose_backward_on_gpu(int gpu_index, void* buffer_dC, void* buffer_dA, int rows_A, int cols_A) {
    if (!buffer_dC || !buffer_dA) { fprintf(stderr, "[C] execute_transpose_backward_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (rows_A <= 0 || cols_A <= 0) { if ((size_t)rows_A * cols_A == 0) return 1; fprintf(stderr, "[C] execute_transpose_backward_on_gpu: Error - Invalid non-positive dimensions (rows_A=%d, cols_A=%d).\n", rows_A, cols_A); return 0; }
    TransposeBackwardCommandData cmd_data = { buffer_dC, buffer_dA, rows_A, cols_A };
    if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE_BACKWARD, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_embedding_lookup_gpu(int gpu_index, void* idx, void* w, void* o, int b, int s, int d, int v) {
    if (!idx || !w || !o) { fprintf(stderr, "[C] execute_embedding_lookup_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (b <= 0 || s <= 0) { if ((size_t)b * s == 0) return 1; fprintf(stderr, "[C] execute_embedding_lookup_gpu: Error - Invalid non-positive dimensions (b=%d, s=%d).\n", b, s); return 0; }
    if (d <= 0 || v <= 0) { fprintf(stderr, "[C] execute_embedding_lookup_gpu: Error - Invalid non-positive dimensions (d=%d, v=%d).\n", d, v); return 0; }
    EmbeddingLookupCommandData cd = { idx, w, o, b, s, d, v };
    if (!submit_kernel_command(gpu_index, COMMAND_EMBEDDING_LOOKUP, &cd)) { return 0; }
    return 1;
}
DLLEXPORT int execute_embedding_backward_gpu(int gpu_index, void* d_o, void* idx, void* d_w, int b, int s, int d, int v) {
    size_t num_grad_elements;
    void* delta_dw_buffer = NULL;
    size_t delta_dw_size_bytes;
    int success = 1;

    if (!d_o || !idx || !d_w) { fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (b <= 0 || s <= 0) { if ((size_t)b * s == 0) return 1; fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - Invalid non-positive dimensions (b=%d, s=%d).\n", b, s); return 0; }
    if (d <= 0 || v <= 0) { if ((size_t)v * d == 0) return 1; fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - Invalid non-positive dimensions (d=%d, v=%d).\n", d, v); return 0; }
    if (!embedding_backward_calc_delta_local_kernel || !add_kernel) { fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - Required kernels not compiled/available.\n"); return 0; }

    num_grad_elements = (size_t)v * d;
    delta_dw_size_bytes = num_grad_elements * sizeof(FP_TYPE);

    delta_dw_buffer = allocate_gpu_memory(gpu_index, delta_dw_size_bytes);
    if (!delta_dw_buffer) { fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - Failed to allocate temporary delta_dw buffer.\n"); return 0; }

    if (!zero_gpu_buffer(gpu_index, delta_dw_buffer, delta_dw_size_bytes)) {
        fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - Failed to zero temporary delta_dw buffer.\n");
        free_gpu_memory(gpu_index, delta_dw_buffer);
        return 0;
    }

    EmbeddingBackwardPass1CommandData pass1_cd = { d_o, idx, delta_dw_buffer, b, s, d, v };
    if (!submit_kernel_command(gpu_index, COMMAND_EMBEDDING_BACKWARD_PASS1, &pass1_cd)) {
        fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - Failed submitting Pass 1 (delta calculation).\n");
        free_gpu_memory(gpu_index, delta_dw_buffer);
        return 0;
    }

    AddCommandData pass2_cd = { d_w, delta_dw_buffer, d_w, (int)num_grad_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_ADD_ELEMENTWISE, &pass2_cd)) {
        fprintf(stderr, "[C] execute_embedding_backward_gpu: Error - Failed submitting Pass 2 (gradient accumulation).\n");
        success = 0;
    }

    free_gpu_memory(gpu_index, delta_dw_buffer);
    return success;
}
DLLEXPORT int execute_reduce_sum_gpu(int gpu_index, void* in, void* out, int B, int M, int N) {
    if (!in || !out) { fprintf(stderr, "[C] execute_reduce_sum_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B <= 0 || M <= 0 || N <= 0) { if ((size_t)B * M == 0 || N == 0) return 1; fprintf(stderr, "[C] execute_reduce_sum_gpu: Error - Invalid non-positive dimensions (B=%d, M=%d, N=%d).\n", B, M, N); return 0; }
    ReduceSumCommandData cd = { in, out, B, M, N };
    if (!submit_kernel_command(gpu_index, COMMAND_REDUCE_SUM_AXIS01, &cd)) { return 0; }
    return 1;
}
DLLEXPORT int execute_broadcast_add_gpu(int gpu_index, void* a, void* b, void* c, int B, int M, int N) {
    if (!a || !b || !c) { fprintf(stderr, "[C] execute_broadcast_add_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B <= 0 || M <= 0 || N <= 0) { if ((size_t)B * M * N == 0) return 1; fprintf(stderr, "[C] execute_broadcast_add_gpu: Error - Invalid non-positive dimensions (B=%d, M=%d, N=%d).\n", B, M, N); return 0; }
    BroadcastAddCommandData cd = { a, b, c, B, M, N };
    if (!submit_kernel_command(gpu_index, COMMAND_BROADCAST_ADD_BIAS, &cd)) { return 0; }
    return 1;
}
DLLEXPORT int execute_transpose_batched_gpu(int gpu_index, void* in, void* out, int B_flat, int d1, int d2) {
    if (!in || !out) { fprintf(stderr, "[C] execute_transpose_batched_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B_flat <= 0 || d1 <= 0 || d2 <= 0) { if ((size_t)B_flat * d1 * d2 == 0) return 1; fprintf(stderr, "[C] execute_transpose_batched_gpu: Error - Invalid non-positive dimensions (B_flat=%d, d1=%d, d2=%d).\n", B_flat, d1, d2); return 0; }
    TransposeBatchedCommandData cd = { in, out, B_flat, d1, d2 };
    if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE_BATCHED, &cd)) { return 0; }
    return 1;
}
DLLEXPORT int execute_transpose_12_batched_gpu(int gpu_index, void* buffer_in, void* buffer_out, int B, int D1, int D2, int D3) {
    if (!buffer_in || !buffer_out) { fprintf(stderr, "[C] execute_transpose_12_batched_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B <= 0 || D1 <= 0 || D2 <= 0 || D3 <= 0) { if ((size_t)B * D1 * D2 * D3 == 0) return 1; fprintf(stderr, "[C] execute_transpose_12_batched_gpu: Error - Invalid non-positive dimensions (B=%d, D1=%d, D2=%d, D3=%d).\n", B, D1, D2, D3); return 0; }
    Transpose12BatchedCommandData cmd_data = { buffer_in, buffer_out, B, D1, D2, D3 };
    if (!submit_kernel_command(gpu_index, COMMAND_TRANSPOSE_12_BATCHED, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_matmul_batched_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_c, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_c) { fprintf(stderr, "[C] execute_matmul_batched_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B <= 0 || M <= 0 || N <= 0) { if ((size_t)B * M * N == 0) return 1; fprintf(stderr, "[C] execute_matmul_batched_on_gpu: Error - Invalid non-positive dimensions (B=%d, M=%d, N=%d).\n", B, M, N); return 0; }
    if (K <= 0) { fprintf(stderr, "[C] execute_matmul_batched_on_gpu: Error - Invalid non-positive dimension K=%d.\n", K); return 0; }
    BMMBatchedCommandData cmd_data = { buffer_a, buffer_b, buffer_c, B, M, N, K };
    if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY_BATCHED, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_matmul_batched_backward_on_gpu(int gpu_index, void* buffer_a, void* buffer_b, void* buffer_dc, void* buffer_da, void* buffer_db, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_b || !buffer_dc ) { fprintf(stderr, "[C] execute_matmul_batched_backward_on_gpu: Error - NULL required input buffer handle provided (A, B, or dC).\n"); return 0; }
    if (!buffer_da && !buffer_db) { return 1; }
    int need_da = (buffer_da != NULL);
    int need_db = (buffer_db != NULL);
     if (B <= 0 || M <= 0 || N <= 0 || K <= 0) {
        int da_zero = need_da && ((size_t)B*M*K == 0);
        int db_zero = need_db && ((size_t)B*K*N == 0);
        if(need_da && need_db && (da_zero || db_zero)) {}
        else if (need_da && da_zero && !need_db) {}
        else if (need_db && db_zero && !need_da) {}
        else if (!need_da && !need_db) { return 1; }
        else {
            fprintf(stderr, "[C] execute_matmul_batched_backward_on_gpu: Error - Invalid non-positive dimensions (B=%d, M=%d, N=%d, K=%d) for requested gradient.\n", B, M, N, K);
            return 0;
        }
    }
    BMMBatchedBackwardData cmd_data = { buffer_a, buffer_b, buffer_dc, buffer_da, buffer_db, B, M, N, K };
    int success = 1;
    if (need_da && (size_t)B * M * K > 0) {
        if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DA, &cmd_data)) { success = 0; }
    }
    if (need_db && (size_t)B * K * N > 0) {
        if (!submit_kernel_command(gpu_index, COMMAND_MATRIX_MULTIPLY_BATCHED_BACKWARD_DB, &cmd_data)) { success = 0; }
    }
    return success;
}
DLLEXPORT int execute_log_softmax_stable_gpu(int gpu_index, void* input_logits, void* output_log_probs, int B_S_rows, int V_cols) {
    if (!input_logits || !output_log_probs) { fprintf(stderr, "[C] execute_log_softmax_stable_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B_S_rows <= 0 || V_cols <= 0) {
        if (B_S_rows == 0) return 1;
        fprintf(stderr, "[C] execute_log_softmax_stable_gpu: Error - Invalid non-positive dimensions (B_S_rows=%d, V_cols=%d).\n", B_S_rows, V_cols); return 0;
    }
    LogSoftmaxStableCommandData cmd_data = { input_logits, output_log_probs, B_S_rows, V_cols };
    if (!submit_kernel_command(gpu_index, COMMAND_LOG_SOFTMAX_STABLE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_cross_entropy_loss_grad_gpu(int gpu_index, void* log_probs, void* target_indices, void* grad_input, void* loss_per_sample, int num_rows, int V) {
    if (!log_probs || !target_indices || !grad_input || !loss_per_sample) { fprintf(stderr, "[C] execute_cross_entropy_loss_grad_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_rows <= 0 || V <= 0) { if (num_rows == 0) return 1; fprintf(stderr, "[C] execute_cross_entropy_loss_grad_gpu: Error - Invalid non-positive dimensions (num_rows=%d, V=%d).\n", num_rows, V); return 0; }
    CrossEntropyLossGradCommandData cmd_data = { log_probs, target_indices, grad_input, loss_per_sample, num_rows, V };
    if (!submit_kernel_command(gpu_index, COMMAND_CROSS_ENTROPY_LOSS_GRAD, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_add_broadcast_pe_gpu(int gpu_index, void* input, void* pe_slice, void* output, int B, int S, int E) {
    if (!input || !pe_slice || !output) { fprintf(stderr, "[C] execute_add_broadcast_pe_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B <= 0 || S <= 0 || E <= 0) { if ((size_t)B * S * E == 0) return 1; fprintf(stderr, "[C] execute_add_broadcast_pe_gpu: Error - Invalid non-positive dimensions (B=%d, S=%d, E=%d).\n", B, S, E); return 0; }
    AddBroadcastPECommandData cmd_data = { input, pe_slice, output, B, S, E };
    if (!submit_kernel_command(gpu_index, COMMAND_ADD_BROADCAST_PE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_hebbian_update_on_gpu(int gpu_index, void* buffer_a, void* buffer_c, void* buffer_w, float learning_rate, int B, int M, int N, int K) {
    if (!buffer_a || !buffer_c || !buffer_w) { fprintf(stderr, "[C] execute_hebbian_update_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (K <= 0 || N <= 0) { if ((size_t)K*N == 0) return 1; fprintf(stderr, "[C] execute_hebbian_update_on_gpu: Error - Invalid non-positive output dimensions (K=%d, N=%d).\n", K, N); return 0; }
    if (B <= 0 || M <= 0) { fprintf(stderr, "[C] execute_hebbian_update_on_gpu: Error - Invalid non-positive reduction dimensions (B=%d, M=%d).\n", B, M); return 0; }
    if (!hebbian_update_local_reduce_kernel) { fprintf(stderr, "[C] execute_hebbian_update_on_gpu: Error - Hebbian kernel not compiled/available.\n"); return 0; }
    HebbianUpdateLocalReduceCommandData cmd_data = { buffer_a, buffer_c, buffer_w, learning_rate, B, M, N, K };
    if (!submit_kernel_command(gpu_index, COMMAND_HEBBIAN_OUTER_PRODUCT_UPDATE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_threshold_spike_on_gpu(int gpu_index, void* buffer_activations, void* buffer_spikes, float threshold, int num_elements) {
    if (!buffer_activations || !buffer_spikes) { fprintf(stderr, "[C] execute_threshold_spike_on_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (num_elements <= 0) { if (num_elements == 0) return 1; fprintf(stderr, "[C] execute_threshold_spike_on_gpu: Error - Invalid non-positive number of elements (%d).\n", num_elements); return 0; }
    ThresholdSpikeCommandData cmd_data = { buffer_activations, buffer_spikes, threshold, num_elements };
    if (!submit_kernel_command(gpu_index, COMMAND_THRESHOLD_SPIKE, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_dynamic_token_assignment_gpu(int gpu_index, void* activations_bse, void* prototypes_te, void* output_indices_bs, int B, int S, int E, int T) {
    if (!activations_bse || !prototypes_te || !output_indices_bs) { fprintf(stderr, "[C] execute_dynamic_token_assignment_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (B <= 0 || S <= 0) { if ((size_t)B * S == 0) return 1; fprintf(stderr, "[C] execute_dynamic_token_assignment_gpu: Error - Invalid non-positive dimensions (B=%d, S=%d).\n", B, S); return 0; }
    if (E <= 0 || T <= 0) { fprintf(stderr, "[C] execute_dynamic_token_assignment_gpu: Error - Invalid non-positive dimensions (E=%d, T=%d).\n", E, T); return 0; }
    DynamicTokenAssignmentCommandData cmd_data = { activations_bse, prototypes_te, output_indices_bs, B, S, E, T };
    if (!submit_kernel_command(gpu_index, COMMAND_DYNAMIC_TOKEN_ASSIGNMENT, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_pairwise_similarity_gpu(int gpu_index, void* states_nd, void* output_similarity_nn, int N, int D) {
    if (!states_nd || !output_similarity_nn) { fprintf(stderr, "[C] execute_pairwise_similarity_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (N <= 0) { if (N == 0) return 1; fprintf(stderr, "[C] execute_pairwise_similarity_gpu: Error - Invalid non-positive dimension N=%d.\n", N); return 0; }
    if (D <= 0) { fprintf(stderr, "[C] execute_pairwise_similarity_gpu: Error - Invalid non-positive dimension D=%d.\n", D); return 0; }
    PairwiseSimilarityCommandData cmd_data = { states_nd, output_similarity_nn, N, D };
    if (!submit_kernel_command(gpu_index, COMMAND_PAIRWISE_SIMILARITY, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_proto_segmented_sum_gpu(int gpu_index, void* activations_flat, void* indices_flat, void* proto_sums, void* proto_counts, int num_elements_flat, int E, int T) {
    if (!activations_flat || !indices_flat || !proto_sums || !proto_counts) { fprintf(stderr, "[C] execute_proto_segmented_sum_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (!has_atomics_support) { fprintf(stderr, "[C] execute_proto_segmented_sum_gpu: Error - Required atomics support is NOT available on this device. Cannot execute.\n"); return 0; }
    if (num_elements_flat <= 0) { if (num_elements_flat == 0) return 1; fprintf(stderr, "[C] execute_proto_segmented_sum_gpu: Error - Invalid non-positive num_elements_flat (%d).\n", num_elements_flat); return 0;}
    if (E <= 0 || T <= 0) { fprintf(stderr, "[C] execute_proto_segmented_sum_gpu: Error - Invalid non-positive dimensions (E=%d, T=%d).\n", E, T); return 0;}
    ProtoSegmentedSumCommandData cmd_data = { activations_flat, indices_flat, proto_sums, proto_counts, num_elements_flat, E, T };
    if (!submit_kernel_command(gpu_index, COMMAND_PROTO_SEGMENTED_SUM, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_proto_update_step_gpu(int gpu_index, void* prototypes, void* proto_sums, void* proto_counts, float learning_rate, int E, int T) {
    if (!prototypes || !proto_sums || !proto_counts) { fprintf(stderr, "[C] execute_proto_update_step_gpu: Error - NULL buffer handle provided.\n"); return 0; }
    if (T <= 0) { if (T == 0) return 1; fprintf(stderr, "[C] execute_proto_update_step_gpu: Error - Invalid non-positive dimension T (%d).\n", T); return 0;}
    if (E <= 0) { fprintf(stderr, "[C] execute_proto_update_step_gpu: Error - Invalid non-positive dimension E (%d).\n", E); return 0;}
    if (learning_rate < 0.0f || learning_rate > 1.0f) { fprintf(stderr, "[C] execute_proto_update_step_gpu: Warning - Invalid learning_rate (%f). Should be in [0, 1].\n", learning_rate); }
    ProtoUpdateStepCommandData cmd_data = { prototypes, proto_sums, proto_counts, learning_rate, E, T };
    if (!submit_kernel_command(gpu_index, COMMAND_PROTO_UPDATE_STEP, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_shape_loss_with_reward_penalty_gpu(
    int gpu_index,
    void* loss_per_sample_in,
    void* predictions,
    void* targets,
    void* loss_per_sample_out,
    int num_samples,
    int num_classes,
    float penalty_weight,
    float reward_weight,
    float high_confidence_threshold,
    int critical_target_class,
    int critical_predicted_class
) {
    if (!loss_per_sample_in || !predictions || !targets || !loss_per_sample_out) {
        fprintf(stderr, "[C] execute_shape_loss_gpu: Error - NULL buffer handle provided.\n"); return 0;
    }
    if (num_samples <= 0 || num_classes <= 0) {
        if (num_samples == 0) return 1;
        fprintf(stderr, "[C] execute_shape_loss_gpu: Error - Invalid non-positive dimensions (samples=%d, classes=%d).\n", num_samples, num_classes); return 0;
    }
    if (!shape_loss_reward_penalty_kernel) {
         fprintf(stderr, "[C] execute_shape_loss_gpu: Error - Loss shaping kernel not available/compiled.\n"); return 0;
    }
    ShapeLossRewardPenaltyCommandData cmd_data = {
        loss_per_sample_in, predictions, targets, loss_per_sample_out,
        num_samples, num_classes, penalty_weight, reward_weight,
        high_confidence_threshold, critical_target_class, critical_predicted_class
    };
    if (!submit_kernel_command(gpu_index, COMMAND_SHAPE_LOSS_REWARD_PENALTY, &cmd_data)) { return 0; }
    return 1;
}
DLLEXPORT int execute_shape_loss_with_reward_penalty_list_gpu(
    int gpu_index,
    void* loss_per_sample_in,
    void* predictions,
    void* targets,
    void* loss_per_sample_out,
    void* critical_pairs,
    int num_samples,
    int num_classes,
    int num_critical_pairs,
    float penalty_weight,
    float reward_weight,
    float high_confidence_threshold
) {
    if (!loss_per_sample_in || !predictions || !targets || !loss_per_sample_out) {
        fprintf(stderr, "[C] execute_shape_loss_list_gpu: Error - NULL required buffer handle provided.\n"); return 0;
    }
    if (num_critical_pairs > 0 && !critical_pairs) {
         fprintf(stderr, "[C] execute_shape_loss_list_gpu: Error - Critical pairs buffer is NULL but count is %d.\n", num_critical_pairs); return 0;
    }
    if (num_samples <= 0 || num_classes <= 0) {
        if (num_samples == 0) return 1;
        fprintf(stderr, "[C] execute_shape_loss_list_gpu: Error - Invalid non-positive dimensions (samples=%d, classes=%d).\n", num_samples, num_classes); return 0;
    }
    if (!shape_loss_reward_penalty_list_kernel) {
         fprintf(stderr, "[C] execute_shape_loss_list_gpu: Error - Loss shaping list kernel not available/compiled.\n"); return 0;
    }
    ShapeLossRewardPenaltyListCommandData cmd_data = {
        loss_per_sample_in, predictions, targets, loss_per_sample_out,
        critical_pairs,
        num_samples, num_classes, num_critical_pairs,
        penalty_weight, reward_weight, high_confidence_threshold
    };
    if (!submit_kernel_command(gpu_index, COMMAND_SHAPE_LOSS_REWARD_PENALTY_LIST, &cmd_data)) {
        return 0;
    }
    return 1;
}


// --- Simulation Layer (Dummy implementations) ---
static void update_device_info_buffer(void) {
    if (device_id == NULL) {
        snprintf(g_device_info_buffer, sizeof(g_device_info_buffer), "(device not initialized)");
        return;
    }

    char device_name[256] = {0};
    char vendor_name[256] = {0};
    cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (err != CL_SUCCESS) {
        strncpy(device_name, "Unknown", sizeof(device_name) - 1);
        device_name[sizeof(device_name) - 1] = '\0';
    }

    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
    if (err != CL_SUCCESS) {
        strncpy(vendor_name, "Unknown", sizeof(vendor_name) - 1);
        vendor_name[sizeof(vendor_name) - 1] = '\0';
    }

    int wrote = 0;
#ifdef CL_DEVICE_PCI_BUS_ID_NV
    cl_uint nv_bus = 0;
    cl_uint nv_slot = 0;
    if (clGetDeviceInfo(device_id, CL_DEVICE_PCI_BUS_ID_NV, sizeof(nv_bus), &nv_bus, NULL) == CL_SUCCESS &&
        clGetDeviceInfo(device_id, CL_DEVICE_PCI_SLOT_ID_NV, sizeof(nv_slot), &nv_slot, NULL) == CL_SUCCESS) {
        snprintf(g_device_info_buffer, sizeof(g_device_info_buffer), "%s (%s) [PCI %u:%u]", device_name, vendor_name, nv_bus, nv_slot);
        wrote = 1;
    }
#endif
#ifdef CL_DEVICE_PCI_BUS_ID_AMD
    if (!wrote) {
        cl_uint amd_bus = 0;
        cl_uint amd_dev = 0;
        if (clGetDeviceInfo(device_id, CL_DEVICE_PCI_BUS_ID_AMD, sizeof(amd_bus), &amd_bus, NULL) == CL_SUCCESS &&
            clGetDeviceInfo(device_id, CL_DEVICE_PCI_DEVICE_ID_AMD, sizeof(amd_dev), &amd_dev, NULL) == CL_SUCCESS) {
            snprintf(g_device_info_buffer, sizeof(g_device_info_buffer), "%s (%s) [PCI %u:%u]", device_name, vendor_name, amd_bus, amd_dev);
            wrote = 1;
        }
    }
#endif
    if (!wrote) {
        snprintf(g_device_info_buffer, sizeof(g_device_info_buffer), "%s (%s)", device_name, vendor_name);
    }
    if (g_active_gpu_index >= 0) {
        char indexed_buffer[sizeof(g_device_info_buffer)];
        snprintf(indexed_buffer, sizeof(indexed_buffer), "#%d %s", g_active_gpu_index, g_device_info_buffer);
        strncpy(g_device_info_buffer, indexed_buffer, sizeof(g_device_info_buffer) - 1);
        g_device_info_buffer[sizeof(g_device_info_buffer) - 1] = '\0';
    }
}

static void reset_last_kernel_time(void) {
    g_last_kernel_time_ms = -1.0;
    g_last_kernel_time_valid = 0;
}

DLLEXPORT int init_driver(int device_index) {
    int status = initialize_gpu(device_index);
    if (status) {
        g_active_gpu_index = device_index;
        update_device_info_buffer();
        reset_last_kernel_time();
    }
    return status;
}

DLLEXPORT const char* get_device_info(void) {
    if (g_device_info_buffer[0] == '\0') {
        update_device_info_buffer();
    }
    return g_device_info_buffer;
}

DLLEXPORT int compile_kernels(const char* kernel_source, char** build_log_out) {
    if (!kernel_source) {
        if (build_log_out) {
            *build_log_out = NULL;
        }
        return 0;
    }
    if (!context || !device_id) {
        if (build_log_out) {
            *build_log_out = NULL;
        }
        return 0;
    }

    cl_int err = CL_SUCCESS;
    const char* sources[1] = { kernel_source };
    cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
    if (!program || err != CL_SUCCESS) {
        if (build_log_out) {
            *build_log_out = NULL;
        }
        return 0;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (build_log_out) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 0) {
            char* log_buffer = (char*)malloc(log_size + 1);
            if (log_buffer) {
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log_buffer, NULL);
                log_buffer[log_size] = '\0';
                *build_log_out = log_buffer;
            } else {
                *build_log_out = NULL;
            }
        } else {
            *build_log_out = NULL;
        }
    }

    clReleaseProgram(program);
    return (err == CL_SUCCESS);
}

DLLEXPORT void* allocate_buffer(size_t bytes) {
    if (g_active_gpu_index < 0) {
        return NULL;
    }
    return allocate_gpu_memory(g_active_gpu_index, bytes);
}

DLLEXPORT int free_buffer(void* handle) {
    if (g_active_gpu_index < 0 || !handle) {
        return 0;
    }
    free_gpu_memory(g_active_gpu_index, handle);
    return 1;
}

DLLEXPORT int upload_buffer(void* handle, const void* host_ptr, size_t bytes) {
    if (g_active_gpu_index < 0 || !handle || !host_ptr || bytes == 0) {
        return 0;
    }
    return write_host_to_gpu_blocking(g_active_gpu_index, handle, 0, bytes, host_ptr);
}

DLLEXPORT int download_buffer(void* handle, void* host_ptr, size_t bytes) {
    if (g_active_gpu_index < 0 || !handle || !host_ptr || bytes == 0) {
        return 0;
    }
    return read_gpu_to_host_blocking(g_active_gpu_index, handle, 0, bytes, host_ptr);
}

DLLEXPORT int conv2d_forward(int device_index,
                             const float* input,
                             const float* weights,
                             const float* bias,
                             float* output,
                             int batch,
                             int in_channels,
                             int in_height,
                             int in_width,
                             int out_channels,
                             int kernel_h,
                             int kernel_w,
                             int stride_h,
                             int stride_w,
                             int pad_h,
                             int pad_w,
                             int dilation_h,
                             int dilation_w,
                             int groups) {
    (void)device_index;
    if (!input || !weights || !output || batch <= 0 || in_channels <= 0 || out_channels <= 0) {
        return 0;
    }
    if (groups <= 0) {
        groups = 1;
    }

    if (stride_h <= 0 || stride_w <= 0 || kernel_h <= 0 || kernel_w <= 0 || dilation_h <= 0 || dilation_w <= 0) {
        return 0;
    }

    if ((in_channels % groups) != 0 || (out_channels % groups) != 0) {
        return 0;
    }

    int effective_in_channels = in_channels / groups;
    int effective_out_channels = out_channels / groups;

    int out_height = ((in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    int out_width = ((in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;
    if (out_height <= 0 || out_width <= 0) {
        return 0;
    }

    for (int b = 0; b < batch; ++b) {
        for (int g = 0; g < groups; ++g) {
            int in_channel_offset = g * effective_in_channels;
            int out_channel_offset = g * effective_out_channels;
            for (int oc = 0; oc < effective_out_channels; ++oc) {
                int global_oc = out_channel_offset + oc;
                const float bias_value = (bias != NULL) ? bias[global_oc] : 0.0f;
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        float acc = bias_value;
                        for (int ic = 0; ic < effective_in_channels; ++ic) {
                            int global_ic = in_channel_offset + ic;
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                int in_y = oh * stride_h - pad_h + kh * dilation_h;
                                if (in_y < 0 || in_y >= in_height) {
                                    continue;
                                }
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int in_x = ow * stride_w - pad_w + kw * dilation_w;
                                    if (in_x < 0 || in_x >= in_width) {
                                        continue;
                                    }
                                    size_t in_index = (((size_t)b * in_channels + (size_t)global_ic) * in_height + (size_t)in_y) * in_width + (size_t)in_x;
                                    size_t weight_index = ((((size_t)global_oc) * effective_in_channels + (size_t)ic) * kernel_h + (size_t)kh) * kernel_w + (size_t)kw;
                                    acc += input[in_index] * weights[weight_index];
                                }
                            }
                        }
                        size_t out_index = (((size_t)b * out_channels + (size_t)global_oc) * out_height + (size_t)oh) * out_width + (size_t)ow;
                        output[out_index] = acc;
                    }
                }
            }
        }
    }
    return 1;
}

DLLEXPORT int maxpool_forward(int device_index,
                              const float* input,
                              float* output,
                              int batch,
                              int channels,
                              int height,
                              int width,
                              int kernel_size,
                              int stride) {
    (void)device_index;
    if (!input || !output || batch <= 0 || channels <= 0 || kernel_size <= 0 || stride <= 0) {
        return 0;
    }
    if (height < kernel_size || width < kernel_size) {
        return 0;
    }
    int out_height = ((height - kernel_size) / stride) + 1;
    int out_width = ((width - kernel_size) / stride) + 1;
    if (out_height <= 0 || out_width <= 0) {
        return 0;
    }

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        int in_y = oh * stride + kh;
                        if (in_y >= height) {
                            continue;
                        }
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int in_x = ow * stride + kw;
                            if (in_x >= width) {
                                continue;
                            }
                            size_t in_index = (((size_t)b * channels + (size_t)c) * height + (size_t)in_y) * width + (size_t)in_x;
                            float value = input[in_index];
                            if (value > max_val) {
                                max_val = value;
                            }
                        }
                    }
                    size_t out_index = (((size_t)b * channels + (size_t)c) * out_height + (size_t)oh) * out_width + (size_t)ow;
                    output[out_index] = max_val;
                }
            }
        }
    }
    return 1;
}

DLLEXPORT int activation_forward(int device_index,
                                 const float* input,
                                 float* output,
                                 int num_elements,
                                 int activation_type,
                                 int apply_spike) {
    (void)device_index;
    if (!input || !output || num_elements <= 0) {
        return 0;
    }
    for (int i = 0; i < num_elements; ++i) {
        float value = input[i];
        switch (activation_type) {
            case 0: // ReLU
                value = (value > 0.0f) ? value : 0.0f;
                break;
            case 1: // Sigmoid
                value = 1.0f / (1.0f + expf(-value));
                break;
            case 2: // Tanh
                value = tanhf(value);
                break;
            default:
                break;
        }
        if (apply_spike) {
            value = (value > 0.5f) ? 1.0f : 0.0f;
        }
        output[i] = value;
    }
    return 1;
}

DLLEXPORT int stdp_update_kernel(int device_index,
                                 const float* pre_activations,
                                 const float* post_activations,
                                 float* weights,
                                 int pre_neurons,
                                 int post_neurons,
                                 float learning_rate,
                                 float decay) {
    (void)device_index;
    if (!pre_activations || !post_activations || !weights || pre_neurons <= 0 || post_neurons <= 0) {
        return 0;
    }

    for (int post_idx = 0; post_idx < post_neurons; ++post_idx) {
        float post_val = post_activations[post_idx];
        for (int pre_idx = 0; pre_idx < pre_neurons; ++pre_idx) {
            float pre_val = pre_activations[pre_idx];
            size_t w_index = (size_t)post_idx * (size_t)pre_neurons + (size_t)pre_idx;
            float delta = learning_rate * (post_val * pre_val);
            weights[w_index] += delta;
            weights[w_index] -= decay * weights[w_index];
        }
    }
    return 1;
}

DLLEXPORT int run_kernel(const char* kernel_name,
                        const void** args,
                        const size_t* arg_sizes,
                        const size_t* global_work_size,
                        const size_t* local_work_size) {
    (void)arg_sizes;
    (void)global_work_size;
    (void)local_work_size;
    reset_last_kernel_time();
    if (!kernel_name) {
        return 0;
    }

    if (strcmp(kernel_name, "matmul") == 0 && args) {
        const void* buffer_a = args[0];
        const void* buffer_b = args[1];
        void* buffer_c = (void*)args[2];
        const int* dims = (const int*)args[3];
        if (!buffer_a || !buffer_b || !buffer_c || !dims) {
            return 0;
        }
        int B = dims[0];
        int M = dims[1];
        int N = dims[2];
        int K = dims[3];
        return execute_matmul_on_gpu(g_active_gpu_index >= 0 ? g_active_gpu_index : 0,
                                     (void*)buffer_a,
                                     (void*)buffer_b,
                                     buffer_c,
                                     B,
                                     M,
                                     N,
                                     K);
    }

    if (strcmp(kernel_name, "conv2d") == 0 && args) {
        const float* input = (const float*)args[0];
        const float* weights = (const float*)args[1];
        const float* bias = (const float*)args[2];
        float* output = (float*)args[3];
        const int* params = (const int*)args[4];
        if (!input || !weights || !output || !params) {
            return 0;
        }
        /* legacy generic dispatch does not supply group count – default to 1 */
        return conv2d_forward(g_active_gpu_index >= 0 ? g_active_gpu_index : 0,
                              input,
                              weights,
                              bias,
                              output,
                              params[0],
                              params[1],
                              params[2],
                              params[3],
                              params[4],
                              params[5],
                              params[6],
                              params[7],
                              params[8],
                              params[9],
                              params[10],
                              params[11],
                              params[12],
                              1);
    }

    return 0;
}

DLLEXPORT int enqueue_async(const char* kernel_name,
                            const void** args,
                            const size_t* arg_sizes,
                            const size_t* global_work_size,
                            const size_t* local_work_size,
                            void** event_handle_out) {
    int result = run_kernel(kernel_name, args, arg_sizes, global_work_size, local_work_size);
    if (event_handle_out) {
        *event_handle_out = NULL;
    }
    return result;
}

DLLEXPORT int get_last_kernel_time_ms(double* milliseconds_out) {
    if (!milliseconds_out || !g_last_kernel_time_valid) {
        return 0;
    }
    *milliseconds_out = g_last_kernel_time_ms;
    return 1;
}

DLLEXPORT unsigned long long simulated_kernel_allocate(int gpu_index, size_t size) {
    if (size == 0) return 0;
    void* ptr = malloc(size);
    if (!ptr) { fprintf(stderr, "[C SIM] simulated_kernel_allocate: malloc failed for size %zu.\n", size); return 0; }
    return (unsigned long long)(uintptr_t)ptr;
}
DLLEXPORT void simulated_kernel_free(int gpu_index, unsigned long long address, size_t size) {
    if (address == 0) return;
    free((void*)(uintptr_t)address);
}
DLLEXPORT void simulated_kernel_write(int gpu_index, unsigned long long address, size_t size, const void *source) {
    if (address == 0 || size == 0 || source == NULL) return;
    memcpy((void*)(uintptr_t)address, source, size);
}
DLLEXPORT unsigned int simulated_get_compute_unit_count(int gpu_index) {
    return 4;
}

// --- End of File ---
#ifdef __cplusplus
}
#endif
