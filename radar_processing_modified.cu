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

// Processing modes
enum ProcessingMode {
    CPU_PROCESSING = 1,
    SINGLE_GPU_PROCESSING = 2,
    DUAL_GPU_PROCESSING = 3
};

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