#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID_SIZE 8192
//#define GRID_SIZE 1024
#define INITIAL_IONS 600
#define MAX_IONS 6000
#define THREADS_PER_BLOCK 256

float uniform_rand(){
    return GRID_SIZE*((float) rand() / (RAND_MAX));
}

void configSeed(unsigned seed){
    if (seed == 0){
        seed = (unsigned)time(NULL);
    }
    srand(seed);
    printf("results for seed %i\n", seed);
}

void populateIons(float *ions_xs, float *ions_ys){
    for(int i = 0; i<INITIAL_IONS; i++){
        ions_xs[i] = uniform_rand();
        ions_ys[i] = uniform_rand();
    }
}

void print_first_5(float *ions_xs, float *ions_ys){
    for(int i = 0; i<5; i++){
        printf("(%f,%f)\n", ions_xs[i], ions_ys[i]);
    }

}

__device__ float distance(float p_1x, float p_1y, float p_2x, float p_2y){
    return sqrtf(powf(p_1x - p_2x, 2) + powf(p_1y - p_2y, 2));
}

__global__ void update_Qs(float * dev_Q, float *dev_xs, float *dev_ys, int iter){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if(tId < GRID_SIZE*GRID_SIZE){
        int x = tId%GRID_SIZE;
        int y = tId/GRID_SIZE;
        if(dev_Q[tId] != INFINITY){
            dev_Q[tId] += 1/distance((float)x, (float)(y), dev_xs[INITIAL_IONS + iter - 1], dev_ys[INITIAL_IONS + iter - 1]);
        }
    }
}
 
__global__ void set_Qs(float *dev_Q, float *dev_xs, float *dev_ys){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if(tId < GRID_SIZE*GRID_SIZE){
        int x = tId%GRID_SIZE;
        int y = tId/GRID_SIZE;
        float q = 0;
        for(int i = 0; i<INITIAL_IONS; i++){
            q += 1/ distance((float)x, (float)y, dev_xs[i], dev_ys[i]);
        }
        dev_Q[tId] = q;      
        /*if(tId<10){
            printf("%f\n",dev_Q[tId]);
        }*/
    }
}

/*int is_in_Ions(int limit, int pos){
    for(int i = INITIAL_IONS; i<INITIAL_IONS + limit; i++){
        if((int)hst_Ions->xs[i] + (int)hst_Ions->ys[i] * GRID_SIZE == pos){
            return 1;
        }
    }
    return 0;
}*/
/*
void wea_q_no_funca(){
    float * Q;
    Ions * hst_Ions;
    float *dev_Q;

    float *hst_xs = (float*)malloc(sizeof(float)*MAX_IONS);
    float *hst_ys = (float*)malloc(sizeof(float)*MAX_IONS);

    Q = (float*)malloc(GRID_SIZE * GRID_SIZE * sizeof(float));
    configSeed(10);
    populateIons(hst_Ions);
    print_first_5(hst_Ions);

    float *dev_xs;
    float *dev_ys;
    cudaMalloc(&dev_xs, sizeof(float) * MAX_IONS);
    cudaMalloc(&dev_ys, sizeof(float) * MAX_IONS);
    
    for(int i=0; i<MAX_IONS-INITIAL_IONS; i++){
        cudaMemcpy(dev_Ions, hst_Ions, sizeof(Ions), cudaMemcpyHostToDevice);
        if(i==0){
            set_Qs<<<(GRID_SIZE*GRID_SIZE/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dev_Q, dev_Ions);
        }else{
            update_Qs<<<(GRID_SIZE*GRID_SIZE/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dev_Q, dev_Ions, i);
        }
        cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        float min = INFINITY;
        int n_min = -1;
        for(int n = 0; n<10; n++){
            printf("; %f",Q[n]);
        }
        for(int n = 0; n<GRID_SIZE*GRID_SIZE; n++){
            if(Q[n]<min){
                min = Q[n];
                n_min = n;
            }
        }
        hst_Ions->xs[INITIAL_IONS+i] = (float)(n_min%GRID_SIZE);
        hst_Ions->ys[INITIAL_IONS+i] = (float)(n_min/GRID_SIZE);
        Q[n_min] = INFINITY;
        cudaMemcpy(dev_Q, Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        printf("%i: q min= %f ; pos_min(%i,%i)\n", i, min, n_min%GRID_SIZE, n_min/GRID_SIZE);
    }
    for(int i = INITIAL_IONS; i<MAX_IONS; i++){
        printf("(%f,%f)\n", hst_Ions->xs[i], hst_Ions->ys[i]);
    }
    free(hst_Ions);
    free(Q);
    cudaFree(dev_Ions);
    cudaFree(dev_Q);
}*/

void wea_test(){
    long blocks = (long)ceil((float)GRID_SIZE*GRID_SIZE/THREADS_PER_BLOCK);
    float *Q = (float *)malloc(GRID_SIZE * GRID_SIZE * sizeof(float));

    float *hst_xs = (float*)malloc(sizeof(float)*MAX_IONS);
    float *hst_ys = (float*)malloc(sizeof(float)*MAX_IONS);

    float * dev_Q;

    configSeed(10);
    populateIons(hst_xs, hst_ys);
    print_first_5(hst_xs, hst_ys);
    
    float *dev_xs;
    float *dev_ys;
    cudaMalloc(&dev_xs, sizeof(float) * MAX_IONS);
    cudaMalloc(&dev_ys, sizeof(float) * MAX_IONS);

    int infcudamalloc = cudaMalloc(&dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float));
    printf("malloc code: %i\n", infcudamalloc);
    cudaMemset(dev_Q, 0, GRID_SIZE*GRID_SIZE * sizeof(float));
    cudaMemcpy(dev_xs, hst_xs, sizeof(float) * MAX_IONS, cudaMemcpyHostToDevice);
    int info = cudaMemcpy(dev_ys, hst_ys, sizeof(float) * MAX_IONS, cudaMemcpyHostToDevice);
    printf("cudaMemcpy code: %i\n", info);
    set_Qs<<<blocks, THREADS_PER_BLOCK>>>(dev_Q, dev_xs, dev_ys);
    ///info = cudaDeviceSynchronize();
    info = cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cuda code: %i\n", info);
    for(int n = 0; n<10; n++){
        printf("; %f",Q[n]);
    }
    cudaFree(dev_Q);
    //cudaFree(dev_Ions);
    free(Q);
    //free(hst_Ions);
}

int main(){
    //wea_q_no_funca();
    wea_test();
    return 0;
}