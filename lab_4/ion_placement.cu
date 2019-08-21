#include <stdio.h>
#include <stdlib.h>
//#include <math.h>

#include "reduction.cuh"

// #define GRID_SIZE 8192//2048
#define GRID_SIZE 1024//2048
#define INITIAL_IONS 5000//125   //5000
#define MAX_IONS 6000
#define BLOCK_SIZE 256//1024

__constant__ float dev_initial_ions[2*INITIAL_IONS];

void populateIons(float * ions){
    for(int i = 0; i<2*INITIAL_IONS; i++){
        ions[i] = uniform_rand();
    }
}

void print_first_5(float * ions){
    for(int i = 0; i<5; i++){
        printf("(%f,%f)\n", ions[i], ions[i+INITIAL_IONS]);
    }

}

void print_last(int n, float * ions){
    for(int i=INITIAL_IONS-n; i<INITIAL_IONS; i++){
        printf("(%f,%f)\n",ions[i],ions[i+INITIAL_IONS]);
    }
}

__device__ float distance(float p_1x, float p_1y, float p_2x, float p_2y){
    return sqrtf(powf(p_1x - p_2x, 2) + powf(p_1y - p_2y,2));
}


__global__ void update_Qs(float *dev_Q, int ion_min_pos){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tId%GRID_SIZE;
    int y = tId/GRID_SIZE;
    if(dev_Q[tId] != INFINITY){
        dev_Q[tId] += 1/distance((float)x, (float)(y), (float)(ion_min_pos%GRID_SIZE), (float)(ion_min_pos/GRID_SIZE));
    }
}

__global__ void set_Qs(float *dev_Q){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    float q = 0;
    for(int i = 0; i<INITIAL_IONS; i++){
        q += 1/distance((float)(tId%GRID_SIZE), (float)(tId/GRID_SIZE), dev_initial_ions[i], dev_initial_ions[i+INITIAL_IONS]);
    }
    dev_Q[tId] = q;
}

void wea_q_no_funca(){
    int *poscicion_iones = (int *)malloc(sizeof(int)*(MAX_IONS-INITIAL_IONS));

    float *Q = (float*)malloc(GRID_SIZE * GRID_SIZE * sizeof(float));
    float *dev_Q;
    
    float *hst_initial_ions = (float *)malloc(2*INITIAL_IONS*sizeof(float));
    populateIons(hst_initial_ions);

    cudaMemcpyToSymbol(dev_initial_ions, hst_initial_ions, 2*INITIAL_IONS*sizeof(float), 0, cudaMemcpyHostToDevice);

    cudaMalloc(&dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float));

    set_Qs<<<(GRID_SIZE*GRID_SIZE/BLOCK_SIZE), BLOCK_SIZE>>>(dev_Q);
    
    int inf = cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cuda code 0: %i\n", inf);
    poscicion_iones[0] = searchMinPos(dev_Q);
    printf("GPU min es %f en pos %i\n",Q[poscicion_iones[0]],poscicion_iones[0]);
    int min_pos = 0;
    float min = Q[0];
    for(int i = 1; i<GRID_SIZE*GRID_SIZE; i++){
        if(min>Q[i]){
            min_pos = i;
            min = Q[i];
        }
    }
    printf("CPU min es %f en pos %i\n", min, min_pos);
    Q[poscicion_iones[0]] = INFINITY;
    inf = cudaMemcpy(dev_Q, Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    printf("cuda code 1: %i\n", inf);

    for(int i=1; i<MAX_IONS-INITIAL_IONS; i++){
        
        update_Qs<<<(GRID_SIZE*GRID_SIZE/BLOCK_SIZE), BLOCK_SIZE>>>(dev_Q, poscicion_iones[i-1]);

        inf = cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        if(inf != 0) {printf("cuda code 2: %i\n", inf);break;}
        Q[poscicion_iones[i] = searchMinPos(dev_Q)] = INFINITY;
        inf = cudaMemcpy(dev_Q, Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        if(inf != 0) {printf("cuda code 3: %i\n", inf);break;}
    }

    //for(int i=0; i<MAX_IONS-INITIAL_IONS; i++) printf("%i\n", poscicion_iones[i]);

    free(Q);
    cudaFree(dev_Q);
    free(poscicion_iones);
}

void wea_test(){
    long blocks = (long)ceil((float)GRID_SIZE*GRID_SIZE/BLOCK_SIZE);
    float *Q = (float *)malloc(GRID_SIZE * GRID_SIZE * sizeof(float));
    float *hst_initial_ions = (float *)malloc(2*INITIAL_IONS*sizeof(float));
    float *dev_Q;
    //Ions * dev_Ions;
    populateIons(hst_initial_ions);
    print_first_5(hst_initial_ions);
    print_last(5, hst_initial_ions);
    cudaMemcpyToSymbol(dev_initial_ions, hst_initial_ions, 2*INITIAL_IONS*sizeof(float), 0, cudaMemcpyHostToDevice);
    int infcudamalloc = cudaMalloc(&dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float));
    printf("malloc code: %i\n", infcudamalloc);
    set_Qs<<<(GRID_SIZE*GRID_SIZE)/BLOCK_SIZE, BLOCK_SIZE>>>(dev_Q);
    //int info = cudaDeviceSynchronize();
    int info = cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cuda code: %i\n", info);
    for(int n = 0; n<10; n++){
        printf("; %f",Q[n]);
    }
    cudaFree(dev_Q);
    free(Q);
}

int main(){
    configSeed(10);//1566155583
    wea_q_no_funca();
    //wea_test();
    return 0;
}