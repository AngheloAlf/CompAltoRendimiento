#include "launch_config.cuh"

__device__ float distance(float p1x, float p1y, float p2x, float p2y){
    return sqrtf(powf(p1x-p2x, 2) + powf(p1y-p2y, 2));
}

__global__ void set_Qs(float* dev_Q){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    float q = 0;
    for(int i = 0; i<INITIAL_IONS; i++){
        float dist = distance((float)(uThId%SIZE_MALLA), (float)(uThId/SIZE_MALLA), dev_ini_ions_xs[i], dev_ini_ions_ys[i]);
        if(dist == 0){
            q=INFINITY;
            break;
        }
        else{
            q+= 1/dist;
        }
    }
    dev_Q[uThId] = q;
}

__global__ void update_Qs_by_chunk(float* dev_Q, int i, int chunk_size){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    float q = 0;
    for(int n = i*chunk_size; n<(i+1)*chunk_size; n++){
        float dist = distance((float)(uThId%SIZE_MALLA), (float)(uThId/SIZE_MALLA), dev_ini_ions_xs[n], dev_ini_ions_ys[n]);
        if(dist == 0){
            q=INFINITY;
            break;
        }
        else{
            q += 1/dist;
        }
    }
    dev_Q[uThId]+= q;
}

__global__ void update_Qs_by_chunk_r(float* dev_Q, int i, int chunk_size, float r){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    float q = 0;
    for(int n = i*chunk_size; n<(i+1)*chunk_size; n++){
        float dist = distance((float)(uThId%SIZE_MALLA), (float)(uThId/SIZE_MALLA), dev_ini_ions_xs[n], dev_ini_ions_ys[n]);
        if(dist < r){
            if(dist == 0){
                q=INFINITY;
                break;
            }
            else{
                q += 1/dist;
            }
        }
    }
    dev_Q[uThId]+= q;
}


__global__ void set_Qs_r(float* dev_Q, float r){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    float q = 0;
    for(int i = 0; i<INITIAL_IONS; i++){
        float dist = distance((float)(uThId%SIZE_MALLA), (float)(uThId/SIZE_MALLA), dev_ini_ions_xs[i], dev_ini_ions_ys[i]);
        if(dist<r){
            if(dist == 0){
                q=INFINITY;
                break;
            }
            else{
                q+= 1/dist;
            }
        }
    }
    dev_Q[uThId] = q;
}

__global__ void update_Qs(int new_ionIdx, float* dev_Q, float* new_ions_xs, float* new_ions_ys){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    float dist = distance((float)(uThId%SIZE_MALLA), (float)(uThId/SIZE_MALLA), new_ions_xs[new_ionIdx], new_ions_ys[new_ionIdx]);
    if(dist == 0){
        dev_Q[uThId]=INFINITY;
    }
    else{
        dev_Q[uThId] += 1/dist;
    }
}

__global__ void update_Qs_r(int new_ionIdx, float* dev_Q, float* new_ions_xs, float* new_ions_ys, float r){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    float dist = distance((float)(uThId%SIZE_MALLA), (float)(uThId/SIZE_MALLA), new_ions_xs[new_ionIdx], new_ions_ys[new_ionIdx]);
    if(dist<r){
        if(dist == 0){
            dev_Q[uThId]=INFINITY;
        }
        else{
            dev_Q[uThId] += 1/dist;
        }
    }
}