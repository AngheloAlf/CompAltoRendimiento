#include <stdio.h>
#include <stdlib.h>

#define SIZE_MALLA 8192//1024
#define BLOCK_SIZE 256//1024
#define INITIAL_IONS 100
#define MAX_IONS 1100

__constant__ float dev_ini_ions_xs[INITIAL_IONS];
__constant__ float dev_ini_ions_ys[INITIAL_IONS];

float uniform_rand(){
    return SIZE_MALLA*((float) rand() / (RAND_MAX));
}

void configSeed(unsigned seed){
    if(seed == 0){
        unsigned seed = (unsigned)time(NULL);
    }
    srand(seed);
    printf("results for seed %i\n", seed);  
}

void populate(float* ions_xs, float* ions_ys){
    for(int i = 0; i<INITIAL_IONS; i++){
        ions_xs[i]=uniform_rand();
        ions_ys[i]=uniform_rand();
    }
}

__device__ float distance(float p1x, float p1y, float p2x, float p2y){
    return sqrtf(powf(p1x-p2x, 2) + powf(p1y-p2y, 2));
}

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

__global__ void set_Qs(float* dev_Q, float* partial_min, int* partial_min_pos){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    float* mins = sdata;
    int* position = (int*)&sdata[BLOCK_SIZE];
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
    mins[tId] = q;
    position[tId] = uThId;
    __syncthreads();
    set_partials(mins, position, tId);
    if(tId == 0){
        partial_min[blockIdx.x] = mins[0];
        partial_min_pos[blockIdx.x] = position[0];
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

__global__ void reduction(float* partial_min, int* partial_min_pos){
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

__global__ void update_Qs(int new_ionIdx, float* dev_Q, float* new_ions_xs, float* new_ions_ys, float* partial_min, int* partial_min_pos){
    int uThId = threadIdx.x + blockDim.x * blockIdx.x;
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    float* mins = sdata;
    int* position = (int*)&sdata[BLOCK_SIZE];
    float dist = distance((float)(uThId%SIZE_MALLA), (float)(uThId/SIZE_MALLA), new_ions_xs[new_ionIdx], new_ions_ys[new_ionIdx]);
    if(dist == 0){
        dev_Q[uThId]=INFINITY;
    }
    else{
        dev_Q[uThId] += 1/dist;
    }
    mins[tId] = dev_Q[uThId];
    position[tId] = uThId;
    __syncthreads();
    set_partials(mins, position, tId);
    if(tId == 0){
        partial_min[blockIdx.x] = mins[0];
        partial_min_pos[blockIdx.x] = position[0];
    }
}

void setIon(int i, float* dev_Q, float* new_ions_xs, float* new_ions_ys, float* partial_min, int* partial_min_pos){
    int size;
    //printf("Ion %i\n", i);
    for(size = SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE; size>BLOCK_SIZE; size/=BLOCK_SIZE){
        //printf("size = %i\n", size);
        reduction<<<size/BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE*(sizeof(float) + sizeof(int))>>>(partial_min, partial_min_pos);
        int inf = cudaDeviceSynchronize();
        if(inf != 0){printf("fail in reduction, cuda code: %i\n", inf);}
    }
    //printf("size = %i\n", size);
    int sMemSize = size*(sizeof(float) + sizeof(int));
    set_new_Ion<<<1, size, sMemSize>>>(i, dev_Q, new_ions_xs, new_ions_ys, partial_min, partial_min_pos);
    int inf = cudaDeviceSynchronize();
    if(inf != 0){printf("fail in set_new_ion, cuda code: %i\n", inf);}
}

int ion_placement(){
    float* hst_ini_ions_xs = (float*)malloc(INITIAL_IONS*sizeof(float));
    float* hst_ini_ions_ys = (float*)malloc(INITIAL_IONS*sizeof(float));
    float* hst_ions_placed_xs = (float*)malloc((MAX_IONS - INITIAL_IONS)*sizeof(float));
    float* hst_ions_placed_ys = (float*)malloc((MAX_IONS - INITIAL_IONS)*sizeof(float));
    populate(hst_ini_ions_xs, hst_ini_ions_ys);
    cudaMemcpyToSymbol(dev_ini_ions_xs, hst_ini_ions_xs, INITIAL_IONS*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_ini_ions_ys, hst_ini_ions_ys, INITIAL_IONS*sizeof(float), 0, cudaMemcpyHostToDevice);
    float* dev_Q;
    float* partial_min;
    float* new_ions_xs;
    float* new_ions_ys;
    int* partial_min_pos;
    cudaMalloc(&dev_Q, SIZE_MALLA*SIZE_MALLA*sizeof(float));
    cudaMalloc(&partial_min, (SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE)*sizeof(float));
    cudaMalloc(&partial_min_pos, (SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE)*sizeof(int));
    cudaMalloc(&new_ions_xs,(MAX_IONS - INITIAL_IONS)*sizeof(float));
    cudaMalloc(&new_ions_ys,(MAX_IONS - INITIAL_IONS)*sizeof(float));
    int sMemSize = BLOCK_SIZE*(sizeof(float) + sizeof(int));
    set_Qs<<<SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE, BLOCK_SIZE, sMemSize>>>(dev_Q, partial_min, partial_min_pos);
    int inf = cudaDeviceSynchronize();
    if(inf != 0) {printf("fail0\n"); return inf;}
    setIon(0, dev_Q, new_ions_xs, new_ions_ys, partial_min, partial_min_pos);
    for(int i = 1; i< MAX_IONS-INITIAL_IONS; i++){
        update_Qs<<<SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE, BLOCK_SIZE, sMemSize>>>(i-1, dev_Q, new_ions_xs, new_ions_ys, partial_min, partial_min_pos);
        inf = cudaDeviceSynchronize();
        if(inf != 0) {printf("fail1\n"); return inf;}
        setIon(i, dev_Q, new_ions_xs, new_ions_ys, partial_min, partial_min_pos);
    }

    inf = cudaMemcpy(hst_ions_placed_xs, new_ions_xs, (MAX_IONS-INITIAL_IONS)*sizeof(float), cudaMemcpyDeviceToHost);
    if(inf != 0) return inf;
    inf = cudaMemcpy(hst_ions_placed_ys, new_ions_ys, (MAX_IONS-INITIAL_IONS)*sizeof(float), cudaMemcpyDeviceToHost);
    if(inf != 0) return inf;

    for(int i = 0; i<MAX_IONS-INITIAL_IONS; i++){
        printf("ION %i: (%f,%f)\n", i, hst_ions_placed_xs[i], hst_ions_placed_ys[i]);
    }

    cudaFree(dev_Q);
    cudaFree(partial_min);
    cudaFree(partial_min_pos);
    cudaFree(new_ions_xs);
    cudaFree(new_ions_ys);
    free(hst_ini_ions_xs);
    free(hst_ini_ions_ys);
    free(hst_ions_placed_xs);
    free(hst_ions_placed_ys);
    return 0;
}

int main(){
    configSeed(10);
    int error = ion_placement();
    printf("cuda code: %i\n", error);
    int dist = 2;
    printf("1/dist = %f\n", (dist==0)? INFINITY : (float)(2/dist));
    return 0;
}