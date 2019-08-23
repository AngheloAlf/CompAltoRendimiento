#include <stdio.h>
#include <stdlib.h>

#include "launch_config.cuh"
#include "reduction.cuh"
#include "kernels.cuh"

float uniform_rand(){
    return SIZE_MALLA*((float) rand() / (RAND_MAX));
}

void configSeed(unsigned seed){
    if(seed == 0){
        seed = (unsigned)time(NULL);
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

void setIon(int i, float* dev_Q, float* new_ions_xs, float* new_ions_ys, float* partial_min, int* partial_min_pos){
    int size;
    Q_reduction<<<SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE*(sizeof(float) + sizeof(int))>>>(dev_Q, partial_min, partial_min_pos);
    //printf("Ion %i\n", i);
    for(size = SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE; size>BLOCK_SIZE; size/=BLOCK_SIZE){
        //printf("size = %i\n", size);
        partial_reduction<<<size/BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE*(sizeof(float) + sizeof(int))>>>(partial_min, partial_min_pos);
        int inf = cudaDeviceSynchronize();
        if(inf != 0){printf("fail in partial_reduction, cuda code: %i\n", inf);}
    }
    //printf("size = %i\n", size);
    int sMemSize = size*(sizeof(float) + sizeof(int));
    set_new_Ion<<<1, size, sMemSize>>>(i, dev_Q, new_ions_xs, new_ions_ys, partial_min, partial_min_pos);
    int inf = cudaDeviceSynchronize();
    if(inf != 0){printf("fail in set_new_ion, cuda code: %i\n", inf);}
}

void printProgress (double percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}

void set_Qs_in_chunks(float* dev_Q, float r){
    int chunk_size = 50;
    for(int i = 0; i<INITIAL_IONS/chunk_size; i++){
        update_Qs_by_chunk_r<<<SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE, BLOCK_SIZE>>>(dev_Q, i, chunk_size, r);
        cudaDeviceSynchronize();
        printProgress((double)(i+1)/(INITIAL_IONS/chunk_size));
    }
    printf("\n");
}

int ion_placement(float r){
    float* hst_ini_ions_xs = (float*)malloc(INITIAL_IONS*sizeof(float));
    float* hst_ini_ions_ys = (float*)malloc(INITIAL_IONS*sizeof(float));
    float* hst_ions_placed_xs = (float*)malloc((MAX_IONS - INITIAL_IONS)*sizeof(float));
    float* hst_ions_placed_ys = (float*)malloc((MAX_IONS - INITIAL_IONS)*sizeof(float));
    populate(hst_ini_ions_xs, hst_ini_ions_ys);
    cudaMemcpyToSymbol(dev_ini_ions_xs, hst_ini_ions_xs, INITIAL_IONS*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_ini_ions_ys, hst_ini_ions_ys, INITIAL_IONS*sizeof(float), 0, cudaMemcpyHostToDevice);
    float* dev_Q;
    float* new_ions_xs;
    float* new_ions_ys;
    int* partial_min_pos;
    float* partial_min;
    cudaMalloc(&partial_min, (SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE)*sizeof(float));
    cudaMalloc(&partial_min_pos, (SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE)*sizeof(int));
    cudaMalloc(&dev_Q, SIZE_MALLA*SIZE_MALLA*sizeof(float));
    cudaMemset(dev_Q, 0, SIZE_MALLA*SIZE_MALLA*sizeof(float));
    cudaMalloc(&new_ions_xs,(MAX_IONS - INITIAL_IONS)*sizeof(float));
    cudaMalloc(&new_ions_ys,(MAX_IONS - INITIAL_IONS)*sizeof(float));
    printf("Working in set_Qs\n");
    set_Qs_in_chunks(dev_Q, r);
    setIon(0, dev_Q, new_ions_xs, new_ions_ys, partial_min, partial_min_pos);
    printf("Updating Qs\n");
    for(int i = 1; i< MAX_IONS-INITIAL_IONS; i++){
        update_Qs_r<<<SIZE_MALLA*SIZE_MALLA/BLOCK_SIZE, BLOCK_SIZE>>>(i-1, dev_Q, new_ions_xs, new_ions_ys, r);
        setIon(i, dev_Q, new_ions_xs, new_ions_ys, partial_min, partial_min_pos);
        cudaDeviceSynchronize();
        printProgress((double)(i+1)/(MAX_IONS-INITIAL_IONS));
    }
    int inf = cudaMemcpy(hst_ions_placed_xs, new_ions_xs, (MAX_IONS-INITIAL_IONS)*sizeof(float), cudaMemcpyDeviceToHost);
    if(inf != 0) return inf;
    inf = cudaMemcpy(hst_ions_placed_ys, new_ions_ys, (MAX_IONS-INITIAL_IONS)*sizeof(float), cudaMemcpyDeviceToHost);
    if(inf != 0) return inf;
    for(int i = 0; i<MAX_IONS-INITIAL_IONS; i++){
        printf("\nION %i: (%f,%f)", i, hst_ions_placed_xs[i], hst_ions_placed_ys[i]);
    }
    cudaFree(dev_Q);
    cudaFree(new_ions_xs);
    cudaFree(new_ions_ys);
    cudaFree(partial_min);
    cudaFree(partial_min_pos);
    free(hst_ini_ions_xs);
    free(hst_ini_ions_ys);
    free(hst_ions_placed_xs);
    free(hst_ions_placed_ys);
    return 0;
}

int main(){
    float radius = 100
    configSeed(1566440079);
    cudaEvent_t ct1, ct2;
    float dt1;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    printf("radio = %f\n", radius);
    cudaEventRecord(ct1);
    int error = ion_placement(radius);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt1, ct1, ct2);
    printf("\n\nCuda code: %i\n", error);
    printf("Total Time %f[ms]\n", dt1);
    return 0;
}