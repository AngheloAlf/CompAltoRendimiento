#include "launch_config.cuh"

__device__ void set_partials(float* mins, int* position, int tId){
    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
        if(tId < s){
            if(mins[tId] > mins[tId+s]){
                mins[tId] = mins[tId+s];
                position[tId] = position[tId+s];
            }
        }
        __syncthreads();
    }
}

__global__ void set_new_Ion(int new_ionIdx, float* dev_Q , float* new_ions_xs, float* new_ions_ys, float* partial_min, int* partial_min_pos){
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    float* mins = sdata;
    int* position = (int*)&sdata[blockDim.x];
    mins[tId] = partial_min[tId];
    position[tId] = partial_min_pos[tId];
    __syncthreads();
    set_partials(mins, position, tId);
    if(tId == 0){
        new_ions_xs[new_ionIdx] = (float)(position[0]%SIZE_MALLA);
        new_ions_ys[new_ionIdx] = (float)(position[0]/SIZE_MALLA);
        dev_Q[position[0]] = INFINITY;
    }
}

__global__ void partial_reduction(float* partial_min, int* partial_min_pos){
    int uThId = threadIdx.x + blockDim.x*blockIdx.x;
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    float* mins = sdata;
    int* position = (int*)&sdata[blockDim.x];
    mins[tId] = partial_min[uThId];
    position[tId] = partial_min_pos[uThId];
    __syncthreads();
    set_partials(mins, position, tId);
    if(tId == 0){
        partial_min[blockIdx.x] = mins[0];
        partial_min_pos[blockIdx.x] = position[0];
    }
}

__global__ void Q_reduction(float* dev_Q, float* partial_min, int* partial_min_pos){
    int uThId = threadIdx.x + blockDim.x*blockIdx.x;
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    float* mins = sdata;
    int* position = (int*)&sdata[blockDim.x];
    mins[tId] = dev_Q[uThId];
    position[tId] = uThId;
    __syncthreads();
    set_partials(mins, position, tId);
    if(tId == 0){
        partial_min[blockIdx.x] = mins[0];
        partial_min_pos[blockIdx.x] = position[0];
    }
}