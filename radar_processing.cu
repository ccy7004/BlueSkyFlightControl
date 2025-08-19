#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Radar parameters definition (corresponding to MATLAB code)
#define C 3.0e8f                    // Speed of light (m/s)
#define RF 3.14e10f                // Radar frequency 31.4GHz  雷达射频
#define LAMBDA (C/RF)               // Radar wavelength
#define PULSE_NUMBER 256          // Number of pulses   仿真脉冲数
#define BANDWIDTH 20e6f             // Transmit signal bandwidth   调制信号带宽
#define TIME_WIDTH 4e-6f            // Transmit signal time width  脉冲宽度
#define PRT 100e-6f                 // Pulse repetition time (s)  脉冲重复周期
#define PRF (1.0f/PRT)              // Pulse repetition frequency
#define FS 12.8e6f                  // Sampling frequency  采样频率
#define NOISE_POWER 0.0f            // Noise power (dB)

// Calculation parameters
#define SAMPLE_NUMBER ((int)(FS*PRT))           // Samples per pulse repetition period
#define TOTAL_NUMBER (SAMPLE_NUMBER*PULSE_NUMBER) // Total number of samples
#define BLIND_NUMBER ((int)(FS*TIME_WIDTH))     // Blind zone samples
#define CHIRP_LENGTH ((int)(FS*TIME_WIDTH))     // Linear chirp pulse length

// Target parameters
#define TARGET_NUMBER 4
#define DECEPTION_JAMMER_NUMBER 2

// Jamming parameters
#define BARRAGE_JAMMER_POWER 10.0f  // Barrage jamming power (dB)
#define BARRAGE_JAMMER_BW 50e6f     // Barrage jamming bandwidth
#define BARRAGE_START_TIME 20e-6f   // Barrage jamming start time
#define BARRAGE_END_TIME 80e-6f     // Barrage jamming end time

// Wiener filter parameters
#define WIENER_ORDER 20

// CFAR parameters
#define CFAR_NR 40
#define CFAR_NP 20
#define CFAR_PFA 0.0001f

// Optimized thread block sizes for better GPU utilization
#define BLOCK_SIZE 512
#define BLOCK_SIZE_2D 32
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Time measurement function
long long getCurrentTimeMicros() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// GPU warm-up
__global__ void warmUpGPU() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f, b = 0.0f;
    b += a + tid;
}

// Generate linear chirp pulse (corresponding to MATLAB Chirp generation)
__global__ void generateChirpKernel(cufftComplex* chirp, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Pre-compute constants in shared memory
    __shared__ float freq_rate_shared;
    __shared__ float inv_fs;
    __shared__ int half_length;
    
    if (threadIdx.x == 0) {
        freq_rate_shared = M_PI * (BANDWIDTH/TIME_WIDTH);
        inv_fs = 1.0f / FS;
        half_length = length / 2;
    }
    __syncthreads();
    
    // Grid-stride loop for better memory throughput
    for (int i = idx; i < length; i += stride) {
        int offset = i - half_length;
        float t = (float)offset * inv_fs;
        float phase = freq_rate_shared * t * t;
        
        // Use sincos for better performance
        float sin_val, cos_val;
        sincosf(phase, &sin_val, &cos_val);
        
        chirp[i].x = cos_val;
        chirp[i].y = sin_val;
    }
}

// Generate Hamming window function
__global__ void generateHammingWindowKernel(float* window, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Pre-compute constants in shared memory
    __shared__ float two_pi_over_n;
    if (threadIdx.x == 0) {
        two_pi_over_n = 2.0f * M_PI / (float)(length - 1);
    }
    __syncthreads();
    
    // Grid-stride loop for vectorized operations
    for (int i = idx; i < length; i += stride) {
        window[i] = 0.54f - 0.46f * cosf(two_pi_over_n * (float)i);
    }
}

// Apply window function to pulse compression coefficients
__global__ void applyWindowKernel(cufftComplex* coeff, float* window, cufftComplex* coeff_windowed, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better memory coalescing
    for (int i = idx; i < length; i += stride) {
        float window_val = window[i];
        coeff_windowed[i].x = coeff[i].x * window_val;
        coeff_windowed[i].y = coeff[i].y * window_val;
    }
}

// Generate pulse compression coefficients (corresponding to MATLAB coeff = conj(fliplr(Chirp)))
__global__ void generateMatchedFilterKernel(cufftComplex* chirp, cufftComplex* coeff, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        int flip_idx = length - 1 - idx;
        // Complex conjugate
        coeff[idx].x = chirp[flip_idx].x;
        coeff[idx].y = -chirp[flip_idx].y;
    }
}

// Optimized target signal generation with coalesced memory access
__global__ void generateTargetSignalKernel(cufftComplex* signal, cufftComplex* chirp, 
                                          float* target_distances, float* target_velocities,
                                          float* target_powers, int target_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Cache target parameters in shared memory
    __shared__ float s_distance, s_velocity, s_power, s_phase_shift;
    if (threadIdx.x == 0) {
        s_distance = target_distances[target_idx];
        s_velocity = target_velocities[target_idx];
        s_power = sqrtf(target_powers[target_idx]);
        s_phase_shift = (target_idx == 3) ? M_PI/3.0f : 0.0f;
    }
    __syncthreads();
    
    int delay_samples = (int)(FS * 2 * s_distance / C);
    float doppler_freq = 2 * s_velocity / LAMBDA;
    float cos_shift = cosf(s_phase_shift);
    float sin_shift = sinf(s_phase_shift);
    
    for (int i = idx; i < TOTAL_NUMBER; i += stride) {
        int sample_idx = i % SAMPLE_NUMBER;
        
        if (sample_idx >= delay_samples && sample_idx < delay_samples + CHIRP_LENGTH) {
            int chirp_idx = sample_idx - delay_samples;
            float time = (float)i / FS;
            float doppler_phase = 2 * M_PI * doppler_freq * time;
            float cos_doppler = cosf(doppler_phase);
            float sin_doppler = sinf(doppler_phase);
            
            float chirp_real = chirp[chirp_idx].x * cos_shift - chirp[chirp_idx].y * sin_shift;
            float chirp_imag = chirp[chirp_idx].x * sin_shift + chirp[chirp_idx].y * cos_shift;
            
            float result_real = s_power * (chirp_real * cos_doppler - chirp_imag * sin_doppler);
            float result_imag = s_power * (chirp_real * sin_doppler + chirp_imag * cos_doppler);
            
            atomicAdd(&signal[i].x, result_real);
            atomicAdd(&signal[i].y, result_imag);
        }
    }
}

// Optimized barrage jamming generation with reduced divergence
__global__ void generateBarrageJammingKernel(cufftComplex* jamming, curandState* states, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Pre-compute constants
    const int barrage_start_sample = (int)(BARRAGE_START_TIME * FS);
    const int barrage_end_sample = (int)(BARRAGE_END_TIME * FS);
    const float amplitude = sqrtf(powf(10.0f, BARRAGE_JAMMER_POWER/10.0f));
    
    for (int i = idx; i < length; i += stride) {
        int sample_idx = i % SAMPLE_NUMBER;
        
        if (sample_idx >= barrage_start_sample && sample_idx <= barrage_end_sample) {
            curandState localState = states[i];
            
            jamming[i].x = amplitude * curand_normal(&localState);
            jamming[i].y = amplitude * curand_normal(&localState);
            
            states[i] = localState;
        } else {
            jamming[i].x = 0.0f;
            jamming[i].y = 0.0f;
        }
    }
}

// Generate deception jamming signal (corresponding to MATLAB DeceptionJammer)
__global__ void generateDeceptionJammingKernel(cufftComplex* jamming, cufftComplex* chirp,
                                              float* deception_distances, float* deception_velocities,
                                              float* deception_powers, curandState* states,
                                              int deception_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TOTAL_NUMBER) {
        int sample_idx = idx % SAMPLE_NUMBER;
        
        // Calculate deception jamming delay
        int delay_samples = (int)(FS * 2 * deception_distances[deception_idx] / C);
        
        // Calculate Doppler frequency shift
        float doppler_freq = 2 * deception_velocities[deception_idx] / LAMBDA;
        float time = (float)idx / FS;
        float doppler_phase = 2 * M_PI * doppler_freq * time;
        
        if (sample_idx >= delay_samples && sample_idx < delay_samples + CHIRP_LENGTH) {
            int chirp_idx = sample_idx - delay_samples;
            float power = sqrtf(powf(10.0f, deception_powers[deception_idx]/10.0f));
            
            // Random phase shift
            curandState localState = states[idx];
            float phase_shift = curand_uniform(&localState) * 2.0f * M_PI;
            states[idx] = localState;
            
            float cos_phase = cosf(phase_shift);
            float sin_phase = sinf(phase_shift);
            float cos_doppler = cosf(doppler_phase);
            float sin_doppler = sinf(doppler_phase);
            
            float chirp_real = chirp[chirp_idx].x * cos_phase - chirp[chirp_idx].y * sin_phase;
            float chirp_imag = chirp[chirp_idx].x * sin_phase + chirp[chirp_idx].y * cos_phase;
            
            atomicAdd(&jamming[idx].x, power * (chirp_real * cos_doppler - chirp_imag * sin_doppler));
            atomicAdd(&jamming[idx].y, power * (chirp_real * sin_doppler + chirp_imag * cos_doppler));
        }
    }
}

// Generate system noise
__global__ void generateSystemNoiseKernel(cufftComplex* noise, curandState* states, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Cache noise amplitude in shared memory
    __shared__ float amplitude_shared;
    if (threadIdx.x == 0) {
        amplitude_shared = sqrtf(powf(10.0f, NOISE_POWER/10.0f));
    }
    __syncthreads();
    
    // Grid-stride loop for vectorized noise generation
    for (int i = idx; i < length; i += stride) {
        curandState localState = states[i];
        
        // Generate two normal random numbers at once
        float2 randn = curand_normal2(&localState);
        
        noise[i].x = amplitude_shared * randn.x;
        noise[i].y = amplitude_shared * randn.y;
        
        states[i] = localState;
    }
}

// Generate auxiliary antenna signal (corresponding to MATLAB AuxiNoise)
__global__ void generateAuxiliarySignalKernel(cufftComplex* aux_signal, cufftComplex* barrage_jammer,
                                              cufftComplex* deception_jammer, curandState* states, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Cache auxiliary noise amplitude in shared memory
    __shared__ float amplitude_shared;
    if (threadIdx.x == 0) {
        amplitude_shared = 1.5f * sqrtf(powf(10.0f, NOISE_POWER/10.0f));
    }
    __syncthreads();
    
    // Grid-stride loop for vectorized auxiliary signal generation
    for (int i = idx; i < length; i += stride) {
        curandState localState = states[i];
        
        // Generate two normal random numbers at once
        float2 randn = curand_normal2(&localState);
        
        // Base noise
        aux_signal[i].x = amplitude_shared * randn.x;
        aux_signal[i].y = amplitude_shared * randn.y;
        
        // Add barrage jamming (attenuated by 0.2x)
        aux_signal[i].x += 0.2f * barrage_jammer[i].x;
        aux_signal[i].y += 0.2f * barrage_jammer[i].y;
        
        // Add deception jamming (attenuated by 0.3x)
        aux_signal[i].x += 0.3f * deception_jammer[i].x;
        aux_signal[i].y += 0.3f * deception_jammer[i].y;
        
        states[i] = localState;
    }
}

// Initialize random number generator
__global__ void initCurandKernel(curandState* states, unsigned long seed, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better parallelization
    for (int i = idx; i < length; i += stride) {
        curand_init(seed, i, 0, &states[i]);
    }
}

// Optimized complex addition with vectorized operations
__global__ void complexAddKernel(cufftComplex* a, cufftComplex* b, cufftComplex* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread for better memory throughput
    for (int i = idx; i < length; i += stride) {
        result[i].x = a[i].x + b[i].x;
        result[i].y = a[i].y + b[i].y;
    }
}

// Optimized complex subtraction with vectorized operations
__global__ void complexSubKernel(cufftComplex* a, cufftComplex* b, cufftComplex* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < length; i += stride) {
        result[i].x = a[i].x - b[i].x;
        result[i].y = a[i].y - b[i].y;
    }
}

// Optimized complex multiplication with vectorized operations
__global__ void complexMultiplyKernel(cufftComplex* a, cufftComplex* b, cufftComplex* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < length; i += stride) {
        result[i] = cuCmulf(a[i], b[i]);
    }
}

// Set blind zone to zero
__global__ void setBlindZoneKernel(cufftComplex* signal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for efficient memory access
    for (int pulse_idx = idx; pulse_idx < PULSE_NUMBER; pulse_idx += stride) {
        for (int i = 0; i < CHIRP_LENGTH; i++) {
            int sample_idx = pulse_idx * SAMPLE_NUMBER + i;
            signal[sample_idx].x = 0.0f;
            signal[sample_idx].y = 0.0f;
        }
    }
}

// Simplified Wiener filtering (corresponding to MATLAB Wiener filtering section)
__global__ void wienerFilterKernel(cufftComplex* input, cufftComplex* auxiliary, 
                                  cufftComplex* output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Cache adaptation coefficient in shared memory
    __shared__ float alpha_shared;
    if (threadIdx.x == 0) {
        alpha_shared = 0.1f; // Filter coefficient
    }
    __syncthreads();
    
    // Grid-stride loop for better memory coalescing
    for (int i = idx; i < length; i += stride) {
        // Vectorized Wiener filtering: output = input - alpha * auxiliary
        output[i].x = input[i].x - alpha_shared * auxiliary[i].x;
        output[i].y = input[i].y - alpha_shared * auxiliary[i].y;
    }
}

// Pulse compression scaling
__global__ void scalePulseCompressionKernel(cufftComplex* input, cufftComplex* output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Pre-compute scale factor in shared memory
    __shared__ float scale_shared;
    if (threadIdx.x == 0) {
        scale_shared = 1.0f / (float)length;
    }
    __syncthreads();
    
    // Grid-stride loop for vectorized operations
    for (int i = idx; i < length; i += stride) {
        output[i].x = input[i].x * scale_shared;
        output[i].y = input[i].y * scale_shared;
    }
}

// Optimized MTI processing with improved memory coalescing
__global__ void mtiProcessingKernel(cufftComplex* input, cufftComplex* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Use shared memory for better cache utilization
    __shared__ cufftComplex s_current[BLOCK_SIZE_2D * BLOCK_SIZE_2D];
    __shared__ cufftComplex s_next[BLOCK_SIZE_2D * BLOCK_SIZE_2D];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    if (i < SAMPLE_NUMBER && j < PULSE_NUMBER - 1) {
        int current_idx = j * SAMPLE_NUMBER + i;
        int next_idx = (j + 1) * SAMPLE_NUMBER + i;
        
        // Load data into shared memory
        s_current[tid] = input[current_idx];
        s_next[tid] = input[next_idx];
        __syncthreads();
        
        // Compute MTI difference
        output[current_idx].x = s_next[tid].x - s_current[tid].x;
        output[current_idx].y = s_next[tid].y - s_current[tid].y;
    }
    
    // Keep the last pulse unchanged
    if (i < SAMPLE_NUMBER && j == PULSE_NUMBER - 1) {
        int idx = j * SAMPLE_NUMBER + i;
        output[idx] = input[idx];
    }
}

// Optimized complex magnitude calculation with vectorized operations
__global__ void complexAbsKernel(cufftComplex* input, float* output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < length; i += stride) {
        float real = input[i].x;
        float imag = input[i].y;
        output[i] = sqrtf(real * real + imag * imag);
    }
}

// Optimized CFAR detection with shared memory and reduced divergence
__global__ void cfarDetectionKernel(float* input, int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Pre-compute CFAR constants
    const float alpha = 2.0f * CFAR_NR * (powf(CFAR_PFA, -1.0f/(2.0f * CFAR_NR)) - 1.0f);
    const int half_nr = CFAR_NR / 2;
    const int half_np = CFAR_NP / 2;
    
    if (i < SAMPLE_NUMBER && j < PULSE_NUMBER) {
        int idx = j * SAMPLE_NUMBER + i;
        
        // Calculate reference window bounds more efficiently
        int left_start = max(0, j - half_nr - half_np);
        int left_end = max(0, j - half_np);
        int right_start = min(PULSE_NUMBER, j + half_np + 1);
        int right_end = min(PULSE_NUMBER, j + half_nr + half_np + 1);
        
        // Calculate sum using unrolled loops for better performance
        float sum_ref = 0.0f;
        int ref_count = 0;
        
        // Left reference window - unrolled
        for (int k = left_start; k < left_end; k++) {
            sum_ref += input[k * SAMPLE_NUMBER + i];
            ref_count++;
        }
        
        // Right reference window - unrolled
        for (int k = right_start; k < right_end; k++) {
            sum_ref += input[k * SAMPLE_NUMBER + i];
            ref_count++;
        }
        
        float threshold = (ref_count > 0) ? (sum_ref / ref_count) * alpha : 10000.0f;
        output[idx] = (input[idx] > threshold) ? 1 : 0;
    }
}

// ==================== CPU Versions of GPU Kernel Functions ====================

// CPU version of generateChirpKernel
void generateChirpCPU(cufftComplex* chirp, int length) {
    for (int i = 0; i < length; i++) {
        float t = (float)i / FS;
        float phase = 2 * M_PI * (RF * t + 0.5f * BANDWIDTH / TIME_WIDTH * t * t);
        chirp[i].x = cosf(phase);
        chirp[i].y = sinf(phase);
    }
}

// CPU version of generateMatchedFilterKernel
void generateMatchedFilterCPU(cufftComplex* chirp, cufftComplex* coeff, int length) {
    for (int i = 0; i < length; i++) {
        int flip_idx = length - 1 - i;
        coeff[i].x = chirp[flip_idx].x;
        coeff[i].y = -chirp[flip_idx].y;
    }
}

// CPU version of generateHammingWindowKernel
void generateHammingWindowCPU(float* window, int length) {
    float two_pi_over_n = 2.0f * M_PI / (float)(length - 1);
    for (int i = 0; i < length; i++) {
        window[i] = 0.54f - 0.46f * cosf(two_pi_over_n * (float)i);
    }
}

// CPU version of applyWindowKernel
void applyWindowCPU(cufftComplex* coeff, float* window, cufftComplex* coeff_windowed, int length) {
    for (int i = 0; i < length; i++) {
        float window_val = window[i];
        coeff_windowed[i].x = coeff[i].x * window_val;
        coeff_windowed[i].y = coeff[i].y * window_val;
    }
}

// CPU version of generateTargetSignalKernel
void generateTargetSignalCPU(cufftComplex* signal, cufftComplex* chirp, 
                            float* target_distances, float* target_velocities,
                            float* target_powers, int target_idx) {
    float distance = target_distances[target_idx];
    float velocity = target_velocities[target_idx];
    float power = sqrtf(target_powers[target_idx]);
    float phase_shift = (target_idx == 3) ? M_PI/3.0f : 0.0f;
    
    int delay_samples = (int)(FS * 2 * distance / C);
    float doppler_freq = 2 * velocity / LAMBDA;
    float cos_shift = cosf(phase_shift);
    float sin_shift = sinf(phase_shift);
    
    for (int i = 0; i < TOTAL_NUMBER; i++) {
        int sample_idx = i % SAMPLE_NUMBER;
        
        if (sample_idx >= delay_samples && sample_idx < delay_samples + CHIRP_LENGTH) {
            int chirp_idx = sample_idx - delay_samples;
            float time = (float)i / FS;
            float doppler_phase = 2 * M_PI * doppler_freq * time;
            float cos_doppler = cosf(doppler_phase);
            float sin_doppler = sinf(doppler_phase);
            
            float chirp_real = chirp[chirp_idx].x * cos_shift - chirp[chirp_idx].y * sin_shift;
            float chirp_imag = chirp[chirp_idx].x * sin_shift + chirp[chirp_idx].y * cos_shift;
            
            float result_real = power * (chirp_real * cos_doppler - chirp_imag * sin_doppler);
            float result_imag = power * (chirp_real * sin_doppler + chirp_imag * cos_doppler);
            
            signal[i].x += result_real;
            signal[i].y += result_imag;
        }
    }
}

// CPU version of generateBarrageJammingKernel
void generateBarrageJammingCPU(cufftComplex* jamming, int length) {
    const int barrage_start_sample = (int)(BARRAGE_START_TIME * FS);
    const int barrage_end_sample = (int)(BARRAGE_END_TIME * FS);
    const float amplitude = sqrtf(powf(10.0f, BARRAGE_JAMMER_POWER/10.0f));
    
    // Initialize random number generator
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < length; i++) {
        int sample_idx = i % SAMPLE_NUMBER;
        
        if (sample_idx >= barrage_start_sample && sample_idx <= barrage_end_sample) {
            // Generate Gaussian noise using Box-Muller transform
            static bool has_spare = false;
            static float spare;
            
            if (has_spare) {
                has_spare = false;
                jamming[i].x = amplitude * spare;
            } else {
                has_spare = true;
                float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                float s = u * u + v * v;
                while (s >= 1.0f || s == 0.0f) {
                    u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    s = u * u + v * v;
                }
                float factor = sqrtf(-2.0f * logf(s) / s);
                jamming[i].x = amplitude * u * factor;
                spare = v * factor;
            }
            
            // Generate imaginary part
            float u2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float v2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float s2 = u2 * u2 + v2 * v2;
            while (s2 >= 1.0f || s2 == 0.0f) {
                u2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                v2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                s2 = u2 * u2 + v2 * v2;
            }
            float factor2 = sqrtf(-2.0f * logf(s2) / s2);
            jamming[i].y = amplitude * u2 * factor2;
        } else {
            jamming[i].x = 0.0f;
            jamming[i].y = 0.0f;
        }
    }
}

// CPU version of generateDeceptionJammingKernel
void generateDeceptionJammingCPU(cufftComplex* jamming, cufftComplex* chirp,
                                float* deception_distances, float* deception_velocities,
                                float* deception_powers, int jammer_idx, int length) {
    float distance = deception_distances[jammer_idx];
    float velocity = deception_velocities[jammer_idx];
    float power = sqrtf(powf(10.0f, deception_powers[jammer_idx]/10.0f));
    
    int delay_samples = (int)(FS * 2 * distance / C);
    float doppler_freq = 2 * velocity / LAMBDA;
    
    for (int i = 0; i < length; i++) {
        int sample_idx = i % SAMPLE_NUMBER;
        
        if (sample_idx >= delay_samples && sample_idx < delay_samples + CHIRP_LENGTH) {
            int chirp_idx = sample_idx - delay_samples;
            float time = (float)i / FS;
            float doppler_phase = 2 * M_PI * doppler_freq * time;
            float cos_doppler = cosf(doppler_phase);
            float sin_doppler = sinf(doppler_phase);
            
            float result_real = power * (chirp[chirp_idx].x * cos_doppler - chirp[chirp_idx].y * sin_doppler);
            float result_imag = power * (chirp[chirp_idx].x * sin_doppler + chirp[chirp_idx].y * cos_doppler);
            
            jamming[i].x += result_real;
            jamming[i].y += result_imag;
        }
    }
}

// CPU version of generateSystemNoiseKernel
void generateSystemNoiseCPU(cufftComplex* noise, int length) {
    const float amplitude = sqrtf(NOISE_POWER);
    
    // Initialize random number generator
    srand((unsigned int)time(NULL) + 12345); // Different seed from barrage jamming
    
    for (int i = 0; i < length; i++) {
        // Generate Gaussian noise using Box-Muller transform
        static bool has_spare = false;
        static float spare;
        
        if (has_spare) {
            has_spare = false;
            noise[i].x = amplitude * spare;
        } else {
            has_spare = true;
            float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float s = u * u + v * v;
            while (s >= 1.0f || s == 0.0f) {
                u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                s = u * u + v * v;
            }
            float factor = sqrtf(-2.0f * logf(s) / s);
            noise[i].x = amplitude * u * factor;
            spare = v * factor;
        }
        
        // Generate imaginary part
        float u2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float v2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float s2 = u2 * u2 + v2 * v2;
        while (s2 >= 1.0f || s2 == 0.0f) {
            u2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            v2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            s2 = u2 * u2 + v2 * v2;
        }
        float factor2 = sqrtf(-2.0f * logf(s2) / s2);
        noise[i].y = amplitude * u2 * factor2;
    }
}

// CPU version of generateAuxiliarySignalKernel
void generateAuxiliarySignalCPU(cufftComplex* aux_signal, cufftComplex* barrage_jammer,
                               cufftComplex* deception_jammer, int length) {
    const float amplitude = sqrtf(NOISE_POWER);
    
    // Initialize random number generator
    srand((unsigned int)time(NULL) + 54321); // Different seed
    
    for (int i = 0; i < length; i++) {
        // Generate base noise
        static bool has_spare = false;
        static float spare;
        
        if (has_spare) {
            has_spare = false;
            aux_signal[i].x = amplitude * spare;
        } else {
            has_spare = true;
            float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float s = u * u + v * v;
            while (s >= 1.0f || s == 0.0f) {
                u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                s = u * u + v * v;
            }
            float factor = sqrtf(-2.0f * logf(s) / s);
            aux_signal[i].x = amplitude * u * factor;
            spare = v * factor;
        }
        
        // Generate imaginary part
        float u2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float v2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float s2 = u2 * u2 + v2 * v2;
        while (s2 >= 1.0f || s2 == 0.0f) {
            u2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            v2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            s2 = u2 * u2 + v2 * v2;
        }
        float factor2 = sqrtf(-2.0f * logf(s2) / s2);
        aux_signal[i].y = amplitude * u2 * factor2;
        
        // Add barrage jamming (attenuated by 0.2x)
        aux_signal[i].x += 0.2f * barrage_jammer[i].x;
        aux_signal[i].y += 0.2f * barrage_jammer[i].y;
        
        // Add deception jamming (attenuated by 0.3x)
        aux_signal[i].x += 0.3f * deception_jammer[i].x;
        aux_signal[i].y += 0.3f * deception_jammer[i].y;
    }
}

// CPU version of complexAddKernel
void complexAddCPU(cufftComplex* a, cufftComplex* b, cufftComplex* result, int length) {
    for (int i = 0; i < length; i++) {
        result[i].x = a[i].x + b[i].x;
        result[i].y = a[i].y + b[i].y;
    }
}

// CPU version of setBlindZoneKernel
void setBlindZoneCPU(cufftComplex* signal) {
    for (int pulse_idx = 0; pulse_idx < PULSE_NUMBER; pulse_idx++) {
        for (int i = 0; i < CHIRP_LENGTH; i++) {
            int sample_idx = pulse_idx * SAMPLE_NUMBER + i;
            signal[sample_idx].x = 0.0f;
            signal[sample_idx].y = 0.0f;
        }
    }
}

// Save results to file
void saveResultsToFile(const char* filename, float* data, int rows, int cols) {
    FILE* file = fopen(filename, "w");
    if (file) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(file, "%.6f ", data[i * cols + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
}

// ==================== Additional CPU processing helpers ====================

void wienerFilterCPU(const cufftComplex* input, const cufftComplex* auxiliary, cufftComplex* output, int length) {
    const float alpha = 0.1f;
    for (int i = 0; i < length; ++i) {
        output[i].x = input[i].x - alpha * auxiliary[i].x;
        output[i].y = input[i].y - alpha * auxiliary[i].y;
    }
}

// Circular convolution with short filter of length CHIRP_LENGTH
void pulseCompressionCPU(const cufftComplex* input, const cufftComplex* coeff_windowed, cufftComplex* output, int length) {
    for (int i = 0; i < length; ++i) {
        float acc_x = 0.0f;
        float acc_y = 0.0f;
        for (int k = 0; k < CHIRP_LENGTH; ++k) {
            int idx = i - k;
            if (idx < 0) idx += length;
            float a_x = input[idx].x;
            float a_y = input[idx].y;
            float b_x = coeff_windowed[k].x;
            float b_y = coeff_windowed[k].y;
            // acc += a * b
            acc_x += a_x * b_x - a_y * b_y;
            acc_y += a_x * b_y + a_y * b_x;
        }
        // Scale to match GPU scaling
        output[i].x = acc_x / (float)length;
        output[i].y = acc_y / (float)length;
    }
}

void mtiProcessingCPU(const cufftComplex* input, cufftComplex* output) {
    for (int j = 0; j < PULSE_NUMBER - 1; ++j) {
        const int base = j * SAMPLE_NUMBER;
        const int next = (j + 1) * SAMPLE_NUMBER;
        for (int i = 0; i < SAMPLE_NUMBER; ++i) {
            int idx = base + i;
            int idx_next = next + i;
            output[idx].x = input[idx_next].x - input[idx].x;
            output[idx].y = input[idx_next].y - input[idx].y;
        }
    }
    // last pulse unchanged
    int base_last = (PULSE_NUMBER - 1) * SAMPLE_NUMBER;
    for (int i = 0; i < SAMPLE_NUMBER; ++i) {
        int idx = base_last + i;
        output[idx] = input[idx];
    }
}

// Naive DFT across pulse dimension for each range cell
void mtdProcessingCPU(const cufftComplex* input, cufftComplex* output) {
    const float two_pi_over_N = 2.0f * (float)M_PI / (float)PULSE_NUMBER;
    for (int i = 0; i < SAMPLE_NUMBER; ++i) {
        for (int k = 0; k < PULSE_NUMBER; ++k) {
            float acc_x = 0.0f;
            float acc_y = 0.0f;
            for (int j = 0; j < PULSE_NUMBER; ++j) {
                int idx = j * SAMPLE_NUMBER + i;
                float angle = -two_pi_over_N * (float)k * (float)j;
                float c = cosf(angle);
                float s = sinf(angle);
                float a_x = input[idx].x;
                float a_y = input[idx].y;
                // a * (c + i s)
                acc_x += a_x * c - a_y * s;
                acc_y += a_x * s + a_y * c;
            }
            int out_idx = k * SAMPLE_NUMBER + i;
            output[out_idx].x = acc_x;
            output[out_idx].y = acc_y;
        }
    }
}

void complexAbsCPU(const cufftComplex* input, float* output, int length) {
    for (int i = 0; i < length; ++i) {
        float x = input[i].x;
        float y = input[i].y;
        output[i] = sqrtf(x * x + y * y);
    }
}

void cfarDetectionCPU(const float* input, int* output) {
    const float alpha = 2.0f * CFAR_NR * (powf(CFAR_PFA, -1.0f/(2.0f * CFAR_NR)) - 1.0f);
    const int half_nr = CFAR_NR / 2;
    const int half_np = CFAR_NP / 2;
    for (int j = 0; j < PULSE_NUMBER; ++j) {
        for (int i = 0; i < SAMPLE_NUMBER; ++i) {
            int idx = j * SAMPLE_NUMBER + i;
            int left_start = std::max(0, j - half_nr - half_np);
            int left_end = std::max(0, j - half_np);
            int right_start = std::min(PULSE_NUMBER, j + half_np + 1);
            int right_end = std::min(PULSE_NUMBER, j + half_nr + half_np + 1);
            float sum_ref = 0.0f;
            int ref_count = 0;
            for (int k = left_start; k < left_end; ++k) {
                sum_ref += input[k * SAMPLE_NUMBER + i];
                ref_count++;
            }
            for (int k = right_start; k < right_end; ++k) {
                sum_ref += input[k * SAMPLE_NUMBER + i];
                ref_count++;
            }
            float threshold = (ref_count > 0) ? (sum_ref / ref_count) * alpha : 10000.0f;
            output[idx] = (input[idx] > threshold) ? 1 : 0;
        }
    }
}

int main() {
    printf("Radar Sidelobe Suppression Processing System - CUDA Implementation\n");
    printf("==========================================\n");
    printf("Two-Phase Processing: Generate 100 signal groups, then process them\n");
    printf("==========================================\n");
    
    // Time measurement variables for 100 iterations
    const int NUM_ITERATIONS = 100;
    long long total_start_time = getCurrentTimeMicros();
    
    // Cumulative time variables for averaging
    long long total_signal_gen_time = 0, total_interference_gen_time = 0;
    long long total_sidelobe_cancellation_time = 0, total_sidelobe_blanking_time = 0;
    long long total_pulse_compression_time = 0;
    long long total_mti_time = 0, total_mtd_time = 0, total_cfar_time = 0;
    long long total_processing_time = 0;
    
    // Individual iteration time variables
    long long sidelobe_cancellation_time = 0, sidelobe_blanking_time = 0;
    long long pulse_compression_time = 0;
    long long mti_time = 0, mtd_time = 0, cfar_time = 0;
    
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    float elapsed_time;
    
    // Declare FFT variables at function level (single GPU path)
    cufftComplex *d_echo_fft = nullptr, *d_coeff_fft = nullptr, *d_pc_fft = nullptr;
    cufftHandle plan, plan_2d;
    bool fft_initialized = false, fft_2d_initialized = false;
    
    // GPU warm-up
    warmUpGPU<<<1, 256>>>();
    cudaDeviceSynchronize();
    
    printf("Radar Parameters:\n");
    printf("- Pulse Number: %d\n", PULSE_NUMBER);
    printf("- Sample Number per Pulse: %d\n", SAMPLE_NUMBER);
    printf("- Total Sample Number: %d\n", TOTAL_NUMBER);
    printf("- Chirp Length: %d\n", CHIRP_LENGTH);
    printf("- Number of Iterations: %d\n", NUM_ITERATIONS);
    printf("\n");
    
    // Allocate host memory
    float target_distances[TARGET_NUMBER] = {3000, 6000, 9000, 4000};
    float target_velocities[TARGET_NUMBER] = {20, 0, 10, 30};
    float target_powers[TARGET_NUMBER] = {1, 1, 1, 1};
    
    float deception_distances[DECEPTION_JAMMER_NUMBER] = {3500, 6500};
    float deception_velocities[DECEPTION_JAMMER_NUMBER] = {15, -10};
    float deception_powers[DECEPTION_JAMMER_NUMBER] = {5, 3};
    
    printf("Allocating optimized GPU memory...\n");
    
    // Create CUDA streams for overlapping computation and memory transfer (single GPU path)
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    
    // Allocate GPU memory for single iteration processing (single GPU path variables)
    cufftComplex *d_chirp, *d_coeff, *d_coeff_windowed;
    cufftComplex *d_signal_all, *d_barrage_jammer, *d_deception_jammer;
    cufftComplex *d_system_noise, *d_aux_signal, *d_echo, *d_true_signal;
    cufftComplex *d_wiener_output, *d_pc_result, *d_mti_result, *d_mtd_result;
    float *d_target_distances, *d_target_velocities, *d_target_powers;
    float *d_deception_distances, *d_deception_velocities, *d_deception_powers;
    float *d_window, *d_mtd_abs;
    int *d_cfar_result;
    curandState *d_rand_states;
    
    // Use pinned memory for faster host-device transfers
    cufftComplex *h_echo_pinned, *h_aux_pinned;
    cudaMallocHost(&h_echo_pinned, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMallocHost(&h_aux_pinned, TOTAL_NUMBER * sizeof(cufftComplex));
    
    // Allocate GPU memory for storing 100 groups of data (reserved, not used further)
    cufftComplex *d_echo_groups, *d_aux_groups;
    
    // Allocate memory for single iteration (single GPU path variables)
    cudaMalloc(&d_chirp, CHIRP_LENGTH * sizeof(cufftComplex));
    cudaMalloc(&d_coeff, CHIRP_LENGTH * sizeof(cufftComplex));
    cudaMalloc(&d_coeff_windowed, CHIRP_LENGTH * sizeof(cufftComplex));
    cudaMalloc(&d_signal_all, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_barrage_jammer, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_deception_jammer, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_system_noise, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_aux_signal, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_echo, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_true_signal, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_wiener_output, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_pc_result, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_mti_result, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_mtd_result, TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_target_distances, TARGET_NUMBER * sizeof(float));
    cudaMalloc(&d_target_velocities, TARGET_NUMBER * sizeof(float));
    cudaMalloc(&d_target_powers, TARGET_NUMBER * sizeof(float));
    cudaMalloc(&d_deception_distances, DECEPTION_JAMMER_NUMBER * sizeof(float));
    cudaMalloc(&d_deception_velocities, DECEPTION_JAMMER_NUMBER * sizeof(float));
    cudaMalloc(&d_deception_powers, DECEPTION_JAMMER_NUMBER * sizeof(float));
    cudaMalloc(&d_window, CHIRP_LENGTH * sizeof(float));
    cudaMalloc(&d_mtd_abs, TOTAL_NUMBER * sizeof(float));
    cudaMalloc(&d_cfar_result, TOTAL_NUMBER * sizeof(int));
    cudaMalloc(&d_rand_states, TOTAL_NUMBER * sizeof(curandState));
    
    // Allocate memory for storing 100 groups of data on device (reserved)
    cudaMalloc(&d_echo_groups, NUM_ITERATIONS * TOTAL_NUMBER * sizeof(cufftComplex));
    cudaMalloc(&d_aux_groups, NUM_ITERATIONS * TOTAL_NUMBER * sizeof(cufftComplex));
    
    // Copy parameters to GPU (single GPU path variables)
    cudaMemcpy(d_target_distances, target_distances, TARGET_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_velocities, target_velocities, TARGET_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_powers, target_powers, TARGET_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deception_distances, deception_distances, DECEPTION_JAMMER_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deception_velocities, deception_velocities, DECEPTION_JAMMER_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deception_powers, deception_powers, DECEPTION_JAMMER_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize random number generator with optimized grid size
    int grid_size = min((TOTAL_NUMBER + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    int chirp_grid_size = min((CHIRP_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    
    initCurandKernel<<<grid_size, BLOCK_SIZE, 0, stream1>>>(d_rand_states, time(NULL), TOTAL_NUMBER);
    cudaStreamSynchronize(stream1);
    
    // ==================== First Loop: Generate 100 Groups of Signal and Interference Data (CPU) ====================
    printf("Phase 1: Generating 100 groups of signal and interference data on CPU...\n");
    printf("Progress: ");
    
    // Allocate CPU memory for signal generation
    cufftComplex *h_chirp = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
    cufftComplex *h_coeff = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
    cufftComplex *h_coeff_windowed = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
    float *h_window = (float*)malloc(CHIRP_LENGTH * sizeof(float));
    cufftComplex *h_signal_all = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
    cufftComplex *h_barrage_jammer = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
    cufftComplex *h_deception_jammer = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
    cufftComplex *h_system_noise = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
    cufftComplex *h_aux_signal = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
    cufftComplex *h_echo = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
    
    // Allocate CPU memory for storing all iterations data
    cufftComplex *h_echo_all_iterations = (cufftComplex*)malloc(NUM_ITERATIONS * TOTAL_NUMBER * sizeof(cufftComplex));
    cufftComplex *h_aux_all_iterations = (cufftComplex*)malloc(NUM_ITERATIONS * TOTAL_NUMBER * sizeof(cufftComplex));
    
    // Generate linear chirp pulse (only once) on CPU
    generateChirpCPU(h_chirp, CHIRP_LENGTH);
    
    // Generate pulse compression coefficients (only once) on CPU
    generateMatchedFilterCPU(h_chirp, h_coeff, CHIRP_LENGTH);
    
    // Generate Hamming window (only once) on CPU
    generateHammingWindowCPU(h_window, CHIRP_LENGTH);
    
    // Apply window function (only once) on CPU
    applyWindowCPU(h_coeff, h_window, h_coeff_windowed, CHIRP_LENGTH);
    
    cudaEventRecord(cuda_start);
    long long data_gen_start_time = getCurrentTimeMicros();
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Progress indicator
        if ((iter + 1) % 10 == 0) {
            printf("%d ", iter + 1);
            fflush(stdout);
        }
        
        // ==================== Signal Generation Phase (CPU) ====================
        // Initialize signal array
        memset(h_signal_all, 0, TOTAL_NUMBER * sizeof(cufftComplex));
        
        // Generate target signals on CPU
        for (int i = 0; i < TARGET_NUMBER; i++) {
            generateTargetSignalCPU(h_signal_all, h_chirp, 
                                   target_distances, target_velocities,
                                   target_powers, i);
        }
        
        // ==================== Interference Generation Phase (CPU) ====================
        // Initialize interference arrays
        memset(h_barrage_jammer, 0, TOTAL_NUMBER * sizeof(cufftComplex));
        memset(h_deception_jammer, 0, TOTAL_NUMBER * sizeof(cufftComplex));
        
        // Generate barrage jamming on CPU
        generateBarrageJammingCPU(h_barrage_jammer, TOTAL_NUMBER);
        
        // Generate deception jamming on CPU
        for (int i = 0; i < DECEPTION_JAMMER_NUMBER; i++) {
            generateDeceptionJammingCPU(h_deception_jammer, h_chirp,
                                       deception_distances, deception_velocities,
                                       deception_powers, i, TOTAL_NUMBER);
        }
        
        // Generate system noise on CPU
        generateSystemNoiseCPU(h_system_noise, TOTAL_NUMBER);
        
        // Generate auxiliary antenna signal on CPU
        generateAuxiliarySignalCPU(h_aux_signal, h_barrage_jammer,
                                  h_deception_jammer, TOTAL_NUMBER);
        
        // Synthesize total echo signal on CPU
        complexAddCPU(h_signal_all, h_barrage_jammer, h_echo, TOTAL_NUMBER);
        complexAddCPU(h_echo, h_deception_jammer, h_echo, TOTAL_NUMBER);
        complexAddCPU(h_echo, h_system_noise, h_echo, TOTAL_NUMBER);
        
        // Set blind zone on CPU
        setBlindZoneCPU(h_echo);
        setBlindZoneCPU(h_aux_signal);
        
        // Store the generated data in CPU memory for later transfer
        memcpy(&h_echo_all_iterations[iter * TOTAL_NUMBER], h_echo, TOTAL_NUMBER * sizeof(cufftComplex));
        memcpy(&h_aux_all_iterations[iter * TOTAL_NUMBER], h_aux_signal, TOTAL_NUMBER * sizeof(cufftComplex));
    }
    
    // Free CPU memory used only for generation
    free(h_chirp);
    free(h_coeff);
    free(h_coeff_windowed);
    free(h_window);
    free(h_signal_all);
    free(h_barrage_jammer);
    free(h_deception_jammer);
    free(h_system_noise);
    free(h_aux_signal);
    free(h_echo);
    
    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);
    cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
    long long data_gen_end_time = getCurrentTimeMicros();
    long long total_data_gen_time = data_gen_end_time - data_gen_start_time;
    
    printf("\nCompleted! Data generation time: %lld microseconds\n", total_data_gen_time);
    
    // ==================== Second Loop: Process 100 Groups of Data ====================
    printf("\nPhase 2: Processing 100 groups of data...\n");
    printf("请选择数据处理模式: 1=CPU, 2=单GPU, 3=双GPU -> ");
    int mode = 2;
    if (scanf("%d", &mode) != 1) {
        mode = 2;
    }
    if (mode < 1 || mode > 3) mode = 2;
    // Early fallback if user selected dual-GPU but system has <2 GPUs
    if (mode == 3) {
        int device_count_check = 0;
        cudaGetDeviceCount(&device_count_check);
        if (device_count_check < 2) {
            printf("\n[Warning] 系统检测到GPU数量不足2块，自动回退到单GPU模式。\n");
            mode = 2;
        }
    }
    printf("Progress: ");
    
    long long processing_start_time = getCurrentTimeMicros();
    
    // ====== CPU Processing Mode ======
    if (mode == 1) {
        long long total_cpu2gpu_transfer_time = 0; // not used in CPU mode
        long long total_gpu2cpu_transfer_time = 0; // not used in CPU mode
        
        // Prepare matched filter coefficients for processing (re-generate)
        cufftComplex *h_chirp_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        cufftComplex *h_coeff_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        float *h_window_proc = (float*)malloc(CHIRP_LENGTH * sizeof(float));
        cufftComplex *h_coeff_windowed_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        generateChirpCPU(h_chirp_proc, CHIRP_LENGTH);
        generateMatchedFilterCPU(h_chirp_proc, h_coeff_proc, CHIRP_LENGTH);
        generateHammingWindowCPU(h_window_proc, CHIRP_LENGTH);
        applyWindowCPU(h_coeff_proc, h_window_proc, h_coeff_windowed_proc, CHIRP_LENGTH);
        
        // Allocate CPU processing buffers
        cufftComplex *cpu_wiener_output = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
        cufftComplex *cpu_pc_result = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
        cufftComplex *cpu_mti_result = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
        cufftComplex *cpu_mtd_result = (cufftComplex*)malloc(TOTAL_NUMBER * sizeof(cufftComplex));
        float *cpu_mtd_abs = (float*)malloc(TOTAL_NUMBER * sizeof(float));
        int *cpu_cfar_result = (int*)malloc(TOTAL_NUMBER * sizeof(int));
        
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            if ((iter + 1) % 10 == 0) {
                printf("%d ", iter + 1);
                fflush(stdout);
            }
            long long iter_start_time = getCurrentTimeMicros();
            
            const cufftComplex* echo = &h_echo_all_iterations[iter * TOTAL_NUMBER];
            const cufftComplex* aux = &h_aux_all_iterations[iter * TOTAL_NUMBER];
            
            // Sidelobe Cancellation (Wiener)
            long long t0 = getCurrentTimeMicros();
            wienerFilterCPU(echo, aux, cpu_wiener_output, TOTAL_NUMBER);
            long long t1 = getCurrentTimeMicros();
            sidelobe_cancellation_time = t1 - t0;
            
            // Sidelobe Blanking (placeholder)
            t0 = getCurrentTimeMicros();
            // No-op for CPU mode; keep as placeholder for timing consistency
            t1 = getCurrentTimeMicros();
            sidelobe_blanking_time = t1 - t0;
            
            // Pulse Compression (circular convolution with short filter)
            t0 = getCurrentTimeMicros();
            pulseCompressionCPU(cpu_wiener_output, h_coeff_windowed_proc, cpu_pc_result, TOTAL_NUMBER);
            t1 = getCurrentTimeMicros();
            pulse_compression_time = t1 - t0;
            
            // MTI
            t0 = getCurrentTimeMicros();
            mtiProcessingCPU(cpu_pc_result, cpu_mti_result);
            t1 = getCurrentTimeMicros();
            mti_time = t1 - t0;
            
            // MTD (DFT across pulses)
            t0 = getCurrentTimeMicros();
            mtdProcessingCPU(cpu_mti_result, cpu_mtd_result);
            t1 = getCurrentTimeMicros();
            mtd_time = t1 - t0;
            
            // Magnitude + CFAR
            t0 = getCurrentTimeMicros();
            complexAbsCPU(cpu_mtd_result, cpu_mtd_abs, TOTAL_NUMBER);
            
            // CFAR
            cfarDetectionCPU(cpu_mtd_abs, cpu_cfar_result);
            t1 = getCurrentTimeMicros();
            cfar_time = t1 - t0;
            
            long long iter_end_time = getCurrentTimeMicros();
            long long iter_processing_time = iter_end_time - iter_start_time;
            
            total_sidelobe_cancellation_time += sidelobe_cancellation_time;
            total_sidelobe_blanking_time += sidelobe_blanking_time;
            total_pulse_compression_time += pulse_compression_time;
            total_mti_time += mti_time;
            total_mtd_time += mtd_time;
            total_cfar_time += cfar_time;
            total_processing_time += iter_processing_time;
        }
        
        // Free CPU processing buffers
        free(cpu_wiener_output);
        free(cpu_pc_result);
        free(cpu_mti_result);
        free(cpu_mtd_result);
        free(cpu_mtd_abs);
        free(cpu_cfar_result);
        free(h_chirp_proc);
        free(h_coeff_proc);
        free(h_window_proc);
        free(h_coeff_windowed_proc);
        
        long long processing_end_time = getCurrentTimeMicros();
        long long total_processing_phase_time = processing_end_time - processing_start_time;
        
        long long avg_signal_gen_time = (total_data_gen_time / 2) / NUM_ITERATIONS;
        long long avg_interference_gen_time = (total_data_gen_time / 2) / NUM_ITERATIONS;
        long long avg_sidelobe_cancellation_time = total_sidelobe_cancellation_time / NUM_ITERATIONS;
        long long avg_sidelobe_blanking_time = total_sidelobe_blanking_time / NUM_ITERATIONS;
        long long avg_pulse_compression_time = total_pulse_compression_time / NUM_ITERATIONS;
        long long avg_mti_time = total_mti_time / NUM_ITERATIONS;
        long long avg_mtd_time = total_mtd_time / NUM_ITERATIONS;
        long long avg_cfar_time = total_cfar_time / NUM_ITERATIONS;
        long long avg_processing_time = total_processing_time / NUM_ITERATIONS;
        long long avg_gpu_time = avg_sidelobe_cancellation_time + avg_sidelobe_blanking_time + 
                                 avg_pulse_compression_time + avg_mti_time + avg_mtd_time + avg_cfar_time;
        
        printf("\nCompleted! Processing time: %lld microseconds\n", total_processing_phase_time);
        
        // ==================== Performance Statistics Output ====================
        printf("\n==========================================\n");
        printf("Two-Phase Processing Time Statistics (100 iterations):\n");
        printf("==========================================\n");
        printf("Phase 1 - Data Generation:\n");
        printf("  Total Data Generation Time: %8lld microseconds\n", total_data_gen_time);
        printf("  Avg Signal Generation:      %8lld microseconds\n", avg_signal_gen_time);
        printf("  Avg Interference Generation:%8lld microseconds\n", avg_interference_gen_time);
        printf("\nPhase 2 - Data Processing (CPU):\n");
        printf("  Avg Sidelobe Cancellation: %8lld microseconds\n", avg_sidelobe_cancellation_time);
        printf("  Avg Sidelobe Blanking:     %8lld microseconds\n", avg_sidelobe_blanking_time);
        printf("  Avg Pulse Compression:     %8lld microseconds\n", avg_pulse_compression_time);
        printf("  Avg MTI Processing:        %8lld microseconds\n", avg_mti_time);
        printf("  Avg MTD Processing:        %8lld microseconds\n", avg_mtd_time);
        printf("  Avg CFAR Detection:        %8lld microseconds\n", avg_cfar_time);
        printf("------------------------------------------\n");
        printf("Average CPU Processing Time: %8lld microseconds\n", avg_gpu_time);
        printf("Average Total Time per Iter: %8lld microseconds\n", avg_processing_time);
    }
    // ====== Single GPU Processing Mode (preserve current behavior) ======
    else if (mode == 2) {
        long long total_cpu2gpu_transfer_time = 0;  // 累计CPU到GPU传输时间
        
        // Prepare matched filter coefficients for processing (re-generate on CPU and copy once)
        cufftComplex *h_chirp_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        cufftComplex *h_coeff_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        float *h_window_proc = (float*)malloc(CHIRP_LENGTH * sizeof(float));
        cufftComplex *h_coeff_windowed_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        generateChirpCPU(h_chirp_proc, CHIRP_LENGTH);
        generateMatchedFilterCPU(h_chirp_proc, h_coeff_proc, CHIRP_LENGTH);
        generateHammingWindowCPU(h_window_proc, CHIRP_LENGTH);
        applyWindowCPU(h_coeff_proc, h_window_proc, h_coeff_windowed_proc, CHIRP_LENGTH);
        cudaMemcpy(d_coeff_windowed, h_coeff_windowed_proc, CHIRP_LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        free(h_chirp_proc);
        free(h_coeff_proc);
        free(h_window_proc);
        free(h_coeff_windowed_proc);
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Progress indicator
            if ((iter + 1) % 10 == 0) {
                printf("%d ", iter + 1);
                fflush(stdout);
            }
            
            long long iter_start_time = getCurrentTimeMicros();
            
            // ==================== CPU to GPU Data Transfer Phase ====================
            cudaEventRecord(cuda_start);
            long long transfer_start_time = getCurrentTimeMicros();
            
            // Transfer data from CPU to GPU using async copy
            cudaMemcpyAsync(d_echo, &h_echo_all_iterations[iter * TOTAL_NUMBER], 
                           TOTAL_NUMBER * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(d_aux_signal, &h_aux_all_iterations[iter * TOTAL_NUMBER], 
                           TOTAL_NUMBER * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream2);
            
            // Synchronize transfer completion
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
            
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            long long transfer_end_time = getCurrentTimeMicros();
            long long cpu2gpu_transfer_time = transfer_end_time - transfer_start_time;
            total_cpu2gpu_transfer_time += cpu2gpu_transfer_time;  // 累加传输时间
            
            if ((iter + 1) % 10 == 0) {
                printf("\nIteration %d - CPU to GPU transfer time: %lld microseconds\n", iter + 1, cpu2gpu_transfer_time);
            }
        
            // ==================== Sidelobe Cancellation Phase ====================
            cudaEventRecord(cuda_start);
            
            // Simplified Wiener filtering for sidelobe cancellation
            wienerFilterKernel<<<grid_size, BLOCK_SIZE, 0, stream1>>>(d_echo, d_aux_signal, d_wiener_output, TOTAL_NUMBER);
            
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            sidelobe_cancellation_time = (long long)(elapsed_time * 1000);
            
            // ==================== Sidelobe Blanking Phase ====================
            cudaEventRecord(cuda_start);
            
            // Placeholder to represent sidelobe blanking processing time
            cudaDeviceSynchronize();
            
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            sidelobe_blanking_time = (long long)(elapsed_time * 1000);
        
            // ==================== Pulse Compression Phase ====================
            cudaEventRecord(cuda_start);
            
            // Create optimized FFT plan with advanced settings (only for first iteration)
            if (!fft_initialized) {
                // Use advanced FFT plan creation for better performance
                size_t workSize;
                cufftCreate(&plan);
                cufftSetAutoAllocation(plan, 0); // Manual memory management
                cufftMakePlan1d(plan, TOTAL_NUMBER, CUFFT_C2C, 1, &workSize);
                
                // Allocate FFT work area
                cufftComplex* d_fft_work;
                cudaMalloc(&d_fft_work, workSize);
                cufftSetWorkArea(plan, d_fft_work);
                
                cufftSetStream(plan, stream1);
                
                // Allocate FFT buffers with optimal alignment
                cudaMalloc(&d_echo_fft, TOTAL_NUMBER * sizeof(cufftComplex));
                cudaMalloc(&d_coeff_fft, TOTAL_NUMBER * sizeof(cufftComplex));
                cudaMalloc(&d_pc_fft, TOTAL_NUMBER * sizeof(cufftComplex));
                fft_initialized = true;
            }
            
            // Zero-pad pulse compression coefficients asynchronously
            cudaMemsetAsync(d_coeff_fft, 0, TOTAL_NUMBER * sizeof(cufftComplex), stream2);
            cudaMemcpyAsync(d_coeff_fft, d_coeff_windowed, CHIRP_LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, stream2);
            
            // FFT with stream synchronization
            cufftExecC2C(plan, d_wiener_output, d_echo_fft, CUFFT_FORWARD);
            cudaStreamSynchronize(stream2);
            cufftExecC2C(plan, d_coeff_fft, d_coeff_fft, CUFFT_FORWARD);
            
            // Frequency domain multiplication
            complexMultiplyKernel<<<grid_size, BLOCK_SIZE, 0, stream1>>>(d_echo_fft, d_coeff_fft, d_pc_fft, TOTAL_NUMBER);
            
            // IFFT
            cufftExecC2C(plan, d_pc_fft, d_pc_result, CUFFT_INVERSE);
            
            // Scaling
            scalePulseCompressionKernel<<<grid_size, BLOCK_SIZE, 0, stream1>>>(d_pc_result, d_pc_result, TOTAL_NUMBER);
            
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            pulse_compression_time = (long long)(elapsed_time * 1000);
        
            // ==================== MTI Processing Phase ====================
            cudaEventRecord(cuda_start);
            
            // Optimized 2D grid configuration
            dim3 mti_grid(min((SAMPLE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535), 
                          min((PULSE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535));
            dim3 mti_block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            
            mtiProcessingKernel<<<mti_grid, mti_block, 0, stream1>>>(d_pc_result, d_mti_result);
            
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            mti_time = (long long)(elapsed_time * 1000);
        
            // ==================== MTD Processing Phase ====================
            cudaEventRecord(cuda_start);
            
            // Create optimized 2D FFT plan for MTD with advanced settings (only for first iteration)
            if (!fft_2d_initialized) {
                // Use advanced 2D FFT plan creation for better performance
                size_t workSize2D;
                cufftCreate(&plan_2d);
                cufftSetAutoAllocation(plan_2d, 0); // Manual memory management
                cufftMakePlan2d(plan_2d, PULSE_NUMBER, SAMPLE_NUMBER, CUFFT_C2C, &workSize2D);
                
                // Allocate 2D FFT work area
                cufftComplex* d_fft_work_2d;
                cudaMalloc(&d_fft_work_2d, workSize2D);
                cufftSetWorkArea(plan_2d, d_fft_work_2d);
                
                cufftSetStream(plan_2d, stream1);
                fft_2d_initialized = true;
            }
            
            // Rearrange data and perform FFT
            cufftExecC2C(plan_2d, d_mti_result, d_mtd_result, CUFFT_FORWARD);
            
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            mtd_time = (long long)(elapsed_time * 1000);
        
            // ==================== CFAR Detection Phase ====================
            cudaEventRecord(cuda_start);
            
            // Calculate MTD result magnitude
            complexAbsKernel<<<grid_size, BLOCK_SIZE, 0, stream1>>>(d_mtd_result, d_mtd_abs, TOTAL_NUMBER);
            
            // CFAR detection with optimized grid
            dim3 cfar_grid(min((SAMPLE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535), 
                           min((PULSE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535));
            dim3 cfar_block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            
            cfarDetectionKernel<<<cfar_grid, cfar_block, 0, stream1>>>(d_mtd_abs, d_cfar_result);
            
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            cfar_time = (long long)(elapsed_time * 1000);
            
            // Calculate iteration processing time
            long long iter_end_time = getCurrentTimeMicros();
            long long iter_processing_time = iter_end_time - iter_start_time;
            
            // Accumulate times for averaging
            total_sidelobe_cancellation_time += sidelobe_cancellation_time;
            total_sidelobe_blanking_time += sidelobe_blanking_time;
            total_pulse_compression_time += pulse_compression_time;
            total_mti_time += mti_time;
            total_mtd_time += mtd_time;
            total_cfar_time += cfar_time;
            total_processing_time += iter_processing_time;
        }
        
        long long processing_end_time = getCurrentTimeMicros();
        long long total_processing_phase_time = processing_end_time - processing_start_time;
        
        printf("\nCompleted! Processing time: %lld microseconds\n", total_processing_phase_time);
        
        long long total_end_time = getCurrentTimeMicros();
        long long total_time = total_end_time - total_start_time;
        
        // Calculate average data generation time per iteration
        total_signal_gen_time = total_data_gen_time / 2;  // Approximate split between signal and interference
        total_interference_gen_time = total_data_gen_time / 2;
        
        // ==================== Result Saving and Statistics ====================
        printf("\nSaving processing results...\n");
        
        // Initialize GPU to CPU transfer timing variables
        long long total_gpu2cpu_transfer_time = 0;
        
        // Time the GPU to CPU transfer for a single fetch (preserve current behavior)
        float *h_mtd_abs_single = (float*)malloc(TOTAL_NUMBER * sizeof(float));
        long long gpu2cpu_transfer_start = getCurrentTimeMicros();
        cudaMemcpy(h_mtd_abs_single, d_mtd_abs, TOTAL_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
        long long gpu2cpu_transfer_end = getCurrentTimeMicros();
        long long gpu2cpu_transfer_time = gpu2cpu_transfer_end - gpu2cpu_transfer_start;
        total_gpu2cpu_transfer_time = gpu2cpu_transfer_time;
        
        // Calculate average times
        long long avg_signal_gen_time = total_signal_gen_time / NUM_ITERATIONS;
        long long avg_interference_gen_time = total_interference_gen_time / NUM_ITERATIONS;
        long long avg_cpu2gpu_transfer_time = total_cpu2gpu_transfer_time / NUM_ITERATIONS;  // 平均CPU到GPU传输时间
        long long avg_gpu2cpu_transfer_time = total_gpu2cpu_transfer_time;  // GPU到CPU传输时间（单次传输）
        long long avg_sidelobe_cancellation_time = total_sidelobe_cancellation_time / NUM_ITERATIONS;
        long long avg_sidelobe_blanking_time = total_sidelobe_blanking_time / NUM_ITERATIONS;
        long long avg_pulse_compression_time = total_pulse_compression_time / NUM_ITERATIONS;
        long long avg_mti_time = total_mti_time / NUM_ITERATIONS;
        long long avg_mtd_time = total_mtd_time / NUM_ITERATIONS;
        long long avg_cfar_time = total_cfar_time / NUM_ITERATIONS;
        long long avg_processing_time = total_processing_time / NUM_ITERATIONS;
        long long avg_gpu_time = avg_sidelobe_cancellation_time + avg_sidelobe_blanking_time + 
                                avg_pulse_compression_time + avg_mti_time + avg_mtd_time + avg_cfar_time;
        
        // ==================== Performance Statistics Output ====================
        printf("\n==========================================\n");
        printf("Two-Phase Processing Time Statistics (100 iterations):\n");
        printf("==========================================\n");
        printf("Phase 1 - Data Generation:\n");
        printf("  Total Data Generation Time: %8lld microseconds\n", total_data_gen_time);
        printf("  Avg Signal Generation:      %8lld microseconds\n", avg_signal_gen_time);
        printf("  Avg Interference Generation:%8lld microseconds\n", avg_interference_gen_time);
        printf("\nPhase 2 - Data Processing (Single GPU):\n");
        printf("  Avg CPU to GPU Transfer:    %8lld microseconds\n", avg_cpu2gpu_transfer_time);
        printf("  Avg GPU to CPU Transfer:    %8lld microseconds\n", avg_gpu2cpu_transfer_time);
        printf("  Avg Sidelobe Cancellation: %8lld microseconds\n", avg_sidelobe_cancellation_time);
        printf("  Avg Sidelobe Blanking:     %8lld microseconds\n", avg_sidelobe_blanking_time);
        printf("  Avg Pulse Compression:     %8lld microseconds\n", avg_pulse_compression_time);
        printf("  Avg MTI Processing:        %8lld microseconds\n", avg_mti_time);
        printf("  Avg MTD Processing:        %8lld microseconds\n", avg_mtd_time);
        printf("  Avg CFAR Detection:        %8lld microseconds\n", avg_cfar_time);
        printf("------------------------------------------\n");
        printf("Average GPU Processing Time: %8lld microseconds\n", avg_gpu_time);
        printf("Average Total Time per Iter: %8lld microseconds\n", avg_processing_time);
        
        free(h_mtd_abs_single);
    }
    // ====== Dual GPU Processing Mode ======
    else {
        
        // Prepare matched filter coefficients for processing (re-generate on CPU)
        cufftComplex *h_chirp_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        cufftComplex *h_coeff_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        float *h_window_proc = (float*)malloc(CHIRP_LENGTH * sizeof(float));
        cufftComplex *h_coeff_windowed_proc = (cufftComplex*)malloc(CHIRP_LENGTH * sizeof(cufftComplex));
        generateChirpCPU(h_chirp_proc, CHIRP_LENGTH);
        generateMatchedFilterCPU(h_chirp_proc, h_coeff_proc, CHIRP_LENGTH);
        generateHammingWindowCPU(h_window_proc, CHIRP_LENGTH);
        applyWindowCPU(h_coeff_proc, h_window_proc, h_coeff_windowed_proc, CHIRP_LENGTH);
        
        // Per-device resources
        struct DeviceResources {
            int deviceId;
            cudaStream_t stream1, stream2, stream3;
            cufftComplex *d_echo, *d_aux_signal, *d_wiener_output, *d_coeff_windowed;
            cufftComplex *d_echo_fft, *d_coeff_fft, *d_pc_fft, *d_pc_result, *d_mti_result, *d_mtd_result;
            float *d_mtd_abs;
            int *d_cfar_result;
            cufftHandle plan1d, plan2d;
            bool plan1d_initialized, plan2d_initialized;
        };
        
        DeviceResources dev[2];
        for (int d = 0; d < 2; ++d) {
            dev[d].deviceId = d;
            cudaSetDevice(dev[d].deviceId);
            cudaStreamCreate(&dev[d].stream1);
            cudaStreamCreate(&dev[d].stream2);
            cudaStreamCreate(&dev[d].stream3);
            cudaMalloc(&dev[d].d_echo, TOTAL_NUMBER * sizeof(cufftComplex));
            cudaMalloc(&dev[d].d_aux_signal, TOTAL_NUMBER * sizeof(cufftComplex));
            cudaMalloc(&dev[d].d_wiener_output, TOTAL_NUMBER * sizeof(cufftComplex));
            cudaMalloc(&dev[d].d_coeff_windowed, CHIRP_LENGTH * sizeof(cufftComplex));
            cudaMemcpy(dev[d].d_coeff_windowed, h_coeff_windowed_proc, CHIRP_LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);
            cudaMalloc(&dev[d].d_pc_result, TOTAL_NUMBER * sizeof(cufftComplex));
            cudaMalloc(&dev[d].d_mti_result, TOTAL_NUMBER * sizeof(cufftComplex));
            cudaMalloc(&dev[d].d_mtd_result, TOTAL_NUMBER * sizeof(cufftComplex));
            cudaMalloc(&dev[d].d_mtd_abs, TOTAL_NUMBER * sizeof(float));
            cudaMalloc(&dev[d].d_cfar_result, TOTAL_NUMBER * sizeof(int));
            dev[d].d_echo_fft = nullptr;
            dev[d].d_coeff_fft = nullptr;
            dev[d].d_pc_fft = nullptr;
            dev[d].plan1d_initialized = false;
            dev[d].plan2d_initialized = false;
        }
        
        long long total_cpu2gpu_transfer_time = 0;
        long long total_gpu2cpu_transfer_time = 0;
        
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            if ((iter + 1) % 10 == 0) {
                printf("%d ", iter + 1);
                fflush(stdout);
            }
            int group_number = iter + 1; // 1-based
            int dsel = (group_number % 2 == 1) ? 0 : 1; // 奇数组->GPU0, 偶数组->GPU1
            DeviceResources &ctx = dev[dsel];
            cudaSetDevice(ctx.deviceId);
            
            long long iter_start_time = getCurrentTimeMicros();
            
            // CPU->GPU transfer (echo and aux)
            cudaEventRecord(cuda_start);
            long long t_transfer0 = getCurrentTimeMicros();
            cudaMemcpyAsync(ctx.d_echo, &h_echo_all_iterations[iter * TOTAL_NUMBER], TOTAL_NUMBER * sizeof(cufftComplex), cudaMemcpyHostToDevice, ctx.stream1);
            cudaMemcpyAsync(ctx.d_aux_signal, &h_aux_all_iterations[iter * TOTAL_NUMBER], TOTAL_NUMBER * sizeof(cufftComplex), cudaMemcpyHostToDevice, ctx.stream2);
            cudaStreamSynchronize(ctx.stream1);
            cudaStreamSynchronize(ctx.stream2);
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            long long t_transfer1 = getCurrentTimeMicros();
            total_cpu2gpu_transfer_time += (t_transfer1 - t_transfer0);
            
            // Sidelobe cancellation
            cudaEventRecord(cuda_start);
            wienerFilterKernel<<<grid_size, BLOCK_SIZE, 0, ctx.stream1>>>(ctx.d_echo, ctx.d_aux_signal, ctx.d_wiener_output, TOTAL_NUMBER);
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            sidelobe_cancellation_time += (long long)(elapsed_time * 1000);
            
            // Sidelobe blanking (placeholder)
            cudaEventRecord(cuda_start);
            cudaDeviceSynchronize();
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            sidelobe_blanking_time += (long long)(elapsed_time * 1000);
            
            // Pulse compression
            cudaEventRecord(cuda_start);
            if (!ctx.plan1d_initialized) {
                size_t workSize = 0;
                cufftCreate(&ctx.plan1d);
                cufftSetAutoAllocation(ctx.plan1d, 0);
                cufftMakePlan1d(ctx.plan1d, TOTAL_NUMBER, CUFFT_C2C, 1, &workSize);
                cufftComplex* work = nullptr;
                cudaMalloc(&work, workSize);
                cufftSetWorkArea(ctx.plan1d, work);
                cufftSetStream(ctx.plan1d, ctx.stream1);
                cudaMalloc(&ctx.d_echo_fft, TOTAL_NUMBER * sizeof(cufftComplex));
                cudaMalloc(&ctx.d_coeff_fft, TOTAL_NUMBER * sizeof(cufftComplex));
                cudaMalloc(&ctx.d_pc_fft, TOTAL_NUMBER * sizeof(cufftComplex));
                ctx.plan1d_initialized = true;
            }
            cudaMemsetAsync(ctx.d_coeff_fft, 0, TOTAL_NUMBER * sizeof(cufftComplex), ctx.stream2);
            cudaMemcpyAsync(ctx.d_coeff_fft, ctx.d_coeff_windowed, CHIRP_LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, ctx.stream2);
            cufftExecC2C(ctx.plan1d, ctx.d_wiener_output, ctx.d_echo_fft, CUFFT_FORWARD);
            cudaStreamSynchronize(ctx.stream2);
            cufftExecC2C(ctx.plan1d, ctx.d_coeff_fft, ctx.d_coeff_fft, CUFFT_FORWARD);
            complexMultiplyKernel<<<grid_size, BLOCK_SIZE, 0, ctx.stream1>>>(ctx.d_echo_fft, ctx.d_coeff_fft, ctx.d_pc_fft, TOTAL_NUMBER);
            cufftExecC2C(ctx.plan1d, ctx.d_pc_fft, ctx.d_pc_result, CUFFT_INVERSE);
            scalePulseCompressionKernel<<<grid_size, BLOCK_SIZE, 0, ctx.stream1>>>(ctx.d_pc_result, ctx.d_pc_result, TOTAL_NUMBER);
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            pulse_compression_time += (long long)(elapsed_time * 1000);
            
            // MTI
            cudaEventRecord(cuda_start);
            dim3 mti_grid(min((SAMPLE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535), 
                          min((PULSE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535));
            dim3 mti_block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            mtiProcessingKernel<<<mti_grid, mti_block, 0, ctx.stream1>>>(ctx.d_pc_result, ctx.d_mti_result);
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            mti_time += (long long)(elapsed_time * 1000);
            
            // MTD
            cudaEventRecord(cuda_start);
            if (!ctx.plan2d_initialized) {
                size_t workSize2D = 0;
                cufftCreate(&ctx.plan2d);
                cufftSetAutoAllocation(ctx.plan2d, 0);
                cufftMakePlan2d(ctx.plan2d, PULSE_NUMBER, SAMPLE_NUMBER, CUFFT_C2C, &workSize2D);
                cufftComplex* work2d = nullptr;
                cudaMalloc(&work2d, workSize2D);
                cufftSetWorkArea(ctx.plan2d, work2d);
                cufftSetStream(ctx.plan2d, ctx.stream1);
                ctx.plan2d_initialized = true;
            }
            cufftExecC2C(ctx.plan2d, ctx.d_mti_result, ctx.d_mtd_result, CUFFT_FORWARD);
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            mtd_time += (long long)(elapsed_time * 1000);
            
            // CFAR
            cudaEventRecord(cuda_start);
            complexAbsKernel<<<grid_size, BLOCK_SIZE, 0, ctx.stream1>>>(ctx.d_mtd_result, ctx.d_mtd_abs, TOTAL_NUMBER);
            dim3 cfar_grid(min((SAMPLE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535), 
                           min((PULSE_NUMBER + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 65535));
            dim3 cfar_block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            cfarDetectionKernel<<<cfar_grid, cfar_block, 0, ctx.stream1>>>(ctx.d_mtd_abs, ctx.d_cfar_result);
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            cudaEventElapsedTime(&elapsed_time, cuda_start, cuda_stop);
            cfar_time += (long long)(elapsed_time * 1000);
            
            // GPU->CPU transfer for results
            float *h_mtd_abs_iter = (float*)malloc(TOTAL_NUMBER * sizeof(float));
            long long t_back0 = getCurrentTimeMicros();
            cudaMemcpy(h_mtd_abs_iter, ctx.d_mtd_abs, TOTAL_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
            long long t_back1 = getCurrentTimeMicros();
            total_gpu2cpu_transfer_time += (t_back1 - t_back0);
            free(h_mtd_abs_iter);
            
            long long iter_end_time = getCurrentTimeMicros();
            total_processing_time += (iter_end_time - iter_start_time);
        }
        
        // Compute averages
        long long avg_signal_gen_time = (total_data_gen_time / 2) / NUM_ITERATIONS;
        long long avg_interference_gen_time = (total_data_gen_time / 2) / NUM_ITERATIONS;
        long long avg_cpu2gpu_transfer_time = total_cpu2gpu_transfer_time / NUM_ITERATIONS;
        long long avg_gpu2cpu_transfer_time = total_gpu2cpu_transfer_time / NUM_ITERATIONS;
        long long avg_sidelobe_cancellation_time = total_sidelobe_cancellation_time / NUM_ITERATIONS;
        long long avg_sidelobe_blanking_time = total_sidelobe_blanking_time / NUM_ITERATIONS;
        long long avg_pulse_compression_time = total_pulse_compression_time / NUM_ITERATIONS;
        long long avg_mti_time = total_mti_time / NUM_ITERATIONS;
        long long avg_mtd_time = total_mtd_time / NUM_ITERATIONS;
        long long avg_cfar_time = total_cfar_time / NUM_ITERATIONS;
        long long avg_processing_time = total_processing_time / NUM_ITERATIONS;
        long long avg_gpu_time = avg_sidelobe_cancellation_time + avg_sidelobe_blanking_time + 
                                 avg_pulse_compression_time + avg_mti_time + avg_mtd_time + avg_cfar_time;
        
        long long processing_end_time = getCurrentTimeMicros();
        long long total_processing_phase_time = processing_end_time - processing_start_time;
        printf("\nCompleted! Processing time: %lld microseconds\n", total_processing_phase_time);
        
        // Output stats
        printf("\n==========================================\n");
        printf("Two-Phase Processing Time Statistics (100 iterations):\n");
        printf("==========================================\n");
        printf("Phase 1 - Data Generation:\n");
        printf("  Total Data Generation Time: %8lld microseconds\n", total_data_gen_time);
        printf("  Avg Signal Generation:      %8lld microseconds\n", avg_signal_gen_time);
        printf("  Avg Interference Generation:%8lld microseconds\n", avg_interference_gen_time);
        printf("\nPhase 2 - Data Processing (Dual GPU):\n");
        printf("  Avg CPU to GPU Transfer:    %8lld microseconds\n", avg_cpu2gpu_transfer_time);
        printf("  Avg GPU to CPU Transfer:    %8lld microseconds\n", avg_gpu2cpu_transfer_time);
        printf("  Avg Sidelobe Cancellation: %8lld microseconds\n", avg_sidelobe_cancellation_time);
        printf("  Avg Sidelobe Blanking:     %8lld microseconds\n", avg_sidelobe_blanking_time);
        printf("  Avg Pulse Compression:     %8lld microseconds\n", avg_pulse_compression_time);
        printf("  Avg MTI Processing:        %8lld microseconds\n", avg_mti_time);
        printf("  Avg MTD Processing:        %8lld microseconds\n", avg_mtd_time);
        printf("  Avg CFAR Detection:        %8lld microseconds\n", avg_cfar_time);
        printf("------------------------------------------\n");
        printf("Average GPU Processing Time: %8lld microseconds\n", avg_gpu_time);
        printf("Average Total Time per Iter: %8lld microseconds\n", avg_processing_time);
        
        // Free device resources
        for (int d = 0; d < 2; ++d) {
            cudaSetDevice(dev[d].deviceId);
            if (dev[d].plan1d_initialized) cufftDestroy(dev[d].plan1d);
            if (dev[d].plan2d_initialized) cufftDestroy(dev[d].plan2d);
            cudaFree(dev[d].d_echo);
            cudaFree(dev[d].d_aux_signal);
            cudaFree(dev[d].d_wiener_output);
            cudaFree(dev[d].d_coeff_windowed);
            if (dev[d].d_echo_fft) cudaFree(dev[d].d_echo_fft);
            if (dev[d].d_coeff_fft) cudaFree(dev[d].d_coeff_fft);
            if (dev[d].d_pc_fft) cudaFree(dev[d].d_pc_fft);
            cudaFree(dev[d].d_pc_result);
            cudaFree(dev[d].d_mti_result);
            cudaFree(dev[d].d_mtd_result);
            cudaFree(dev[d].d_mtd_abs);
            cudaFree(dev[d].d_cfar_result);
            cudaStreamDestroy(dev[d].stream1);
            cudaStreamDestroy(dev[d].stream2);
            cudaStreamDestroy(dev[d].stream3);
        }
        
        free(h_chirp_proc);
        free(h_coeff_proc);
        free(h_window_proc);
        free(h_coeff_windowed_proc);
    }
    
    // Clean up arrays for all iterations
    free(h_echo_all_iterations);
    free(h_aux_all_iterations);
    
    // Clean up single-GPU allocated resources (if any allocated)
    cudaFree(d_chirp);
    cudaFree(d_coeff);
    cudaFree(d_coeff_windowed);
    cudaFree(d_signal_all);
    cudaFree(d_barrage_jammer);
    cudaFree(d_deception_jammer);
    cudaFree(d_system_noise);
    cudaFree(d_aux_signal);
    cudaFree(d_echo);
    cudaFree(d_true_signal);
    cudaFree(d_wiener_output);
    cudaFree(d_pc_result);
    cudaFree(d_mti_result);
    cudaFree(d_mtd_result);
    cudaFree(d_target_distances);
    cudaFree(d_target_velocities);
    cudaFree(d_target_powers);
    cudaFree(d_deception_distances);
    cudaFree(d_deception_velocities);
    cudaFree(d_deception_powers);
    cudaFree(d_window);
    cudaFree(d_mtd_abs);
    cudaFree(d_cfar_result);
    cudaFree(d_rand_states);
    cudaFree(d_echo_groups);
    cudaFree(d_aux_groups);
    cudaFreeHost(h_echo_pinned);
    cudaFreeHost(h_aux_pinned);
    
    if (fft_initialized) {
        cudaFree(d_echo_fft);
        cudaFree(d_coeff_fft);
        cudaFree(d_pc_fft);
        cufftDestroy(plan);
    }
    if (fft_2d_initialized) {
        cufftDestroy(plan_2d);
    }
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);
    
    return 0;
}

