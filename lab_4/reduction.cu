#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 16384//8192
#define BLOCK_SIZE 256//1024

float uniform_rand(){
    return SIZE*((float) rand() / (RAND_MAX));
}

void configSeed(unsigned seed){
    if(seed == 0){
        unsigned seed = (unsigned)time(NULL);
    }
    srand(seed);
    printf("results for seed %i\n", seed);  
}

void populate(float * Q){
    for(int i = 0; i<SIZE*SIZE; i++){
        Q[i] = uniform_rand();
    }
}

__global__ void cudaMin(float * dev_Q, float * ans){
    int uThId = threadIdx.x + blockIdx.x * blockDim.x;
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    sdata[tId] = dev_Q[uThId];
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tId < s){
            if(sdata[tId] > sdata[tId+s]){
                sdata[tId] = sdata[tId+s];
            }
        }
        __syncthreads();
    }
    if(tId == 0) ans[blockIdx.x] = sdata[0];
}

__global__ void cudaPartialsMinPos(float * dev_Q, float * ans_min, int * ans_pos){
    int uThId = threadIdx.x + blockIdx.x * blockDim.x;
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    float * numbers = sdata;
    int * position = (int*)&sdata[BLOCK_SIZE];
    numbers[tId] = dev_Q[uThId];
    position[tId] = uThId;
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tId < s){
            if(numbers[tId] > numbers[tId+s]){
                numbers[tId] = numbers[tId+s];
                position[tId] = position[tId+s];
            }
        }
        __syncthreads();
    }
    if(tId == 0){
        ans_min[blockIdx.x] = numbers[0];
        ans_pos[blockIdx.x] = position[0];
    }
}

__global__ void cudaMinPos(float * partials, int * partials_pos, float * min, int * min_pos){
    int uThId = threadIdx.x + blockIdx.x * blockDim.x;
    int tId = threadIdx.x;
    extern __shared__ float sdata[];
    float * numbers = sdata;
    int * position = (int*)&sdata[BLOCK_SIZE];
    numbers[tId] = partials[uThId];
    position[tId] = partials_pos[uThId];
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tId < s){
            if(numbers[tId] > numbers[tId+s]){
                numbers[tId] = numbers[tId+s];
                position[tId] = position[tId+s];
            }
        }
        __syncthreads();
    }
    if(tId == 0){
        min[0] = numbers[0];
        min_pos[0] = position[0];
    }
}

void print(float * ptr, int n){
    for(int i = 0; i<n; i++){
        printf("%f\n", ptr[i]);
    }
}

void searchMin(float * Q){
    float * min = (float*)malloc(sizeof(float));
    float * dev_Q;
    float * dev_partials;
    float * dev_min;
    float * partials = (float*)malloc((SIZE*SIZE/BLOCK_SIZE)*sizeof(float));
    cudaMalloc(&dev_Q, SIZE*SIZE*sizeof(float));
    cudaMalloc(&dev_partials, (SIZE*SIZE/BLOCK_SIZE)*sizeof(float));
    cudaMalloc(&dev_min, sizeof(float));
    cudaMemcpy(dev_Q, Q, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMin<<<(SIZE*SIZE)/BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(dev_Q, dev_partials);
    int inf = cudaMemcpy(partials, dev_partials, (SIZE*SIZE/BLOCK_SIZE)*sizeof(float), cudaMemcpyDeviceToHost);
    printf("cuda code %i\n", inf);
    print(partials,(SIZE*SIZE/BLOCK_SIZE));
    cudaMin<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(dev_partials, dev_min);
    cudaMemcpy(min, dev_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_partials);
    cudaFree(dev_min);
    cudaFree(dev_Q);
    free(partials);
    printf("The min is %f\n", *min);
    free(min);
}

int searchMinPos(float * dev_Q){
    float * dev_partials_ans;
    int * dev_partials_pos;
    float * dev_min;
    int * dev_min_pos;
    float * min = (float*)malloc(sizeof(float));
    int * pos = (int*)malloc(sizeof(int));
    int sMemSize = BLOCK_SIZE*(sizeof(float) + sizeof(int));
    cudaMalloc(&dev_partials_ans, (SIZE*SIZE/BLOCK_SIZE)*sizeof(float));
    cudaMalloc(&dev_partials_pos, (SIZE*SIZE/BLOCK_SIZE)*sizeof(int));
    cudaMalloc(&dev_min, sizeof(float));
    cudaMalloc(&dev_min_pos, sizeof(int));
    cudaPartialsMinPos<<<(SIZE*SIZE)/BLOCK_SIZE, BLOCK_SIZE, sMemSize>>>(dev_Q, dev_partials_ans, dev_partials_pos);
    cudaMinPos<<<1, BLOCK_SIZE, sMemSize>>>(dev_partials_ans, dev_partials_pos, dev_min, dev_min_pos);
    cudaMemcpy(min, dev_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos, dev_min_pos, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_Q);
    cudaFree(dev_partials_ans);
    cudaFree(dev_partials_pos);
    cudaFree(dev_min);
    cudaFree(dev_min_pos);
    int resp = *pos;
    free(min);
    free(pos);
    return resp;
}
int main(){
    float * Q = (float*)malloc(SIZE*SIZE*sizeof(float));
    float * dev_Q;
    cudaMalloc(&dev_Q, SIZE*SIZE*sizeof(float));
    configSeed(2134540);
    populate(Q);
    cudaMemcpy(dev_Q, Q, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
    print(Q, 7);
    //searchMin(Q);
    cudaEvent_t ct1, ct2;
    float dt;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    int pos = searchMinPos(dev_Q);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    printf("GPU min is %f and his pos is %i. Time: %f[ms]\n", Q[pos], pos, dt);
    clock_t t1, t2;
    t1 = clock();
    float minv = Q[0];
    int min_pos = 0;
    for(int i = 1; i<SIZE*SIZE; i++){
        if(minv>Q[i]){
            minv = Q[i];
            min_pos = i;
        }
    }
    t2 = clock();
    double dtime = 1000.0 * (double)(t2-t1) / CLOCKS_PER_SEC;
    printf("CPU min is %f and his pos is %i. Time %f[ms]\n", Q[min_pos], min_pos, dtime);
    cudaFree(dev_Q);
    free(Q);
    return 0;
}