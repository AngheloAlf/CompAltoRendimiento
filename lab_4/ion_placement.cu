#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID_SIZE 1000
#define INITIAL_IONS 5000
#define MAX_IONS 6000
#define THREADS_PER_BLOCK 512

typedef struct point point;

point Ions[MAX_IONS];
float Q[GRID_SIZE*GRID_SIZE];


struct point{
    float x,y;
};

float uniform_rand(){
    return GRID_SIZE*((float) rand() / (RAND_MAX));
}

void configSeed(){
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    printf("results for seed %i\n", seed);
}

void populateIons(){
    for(int i = 0; i<INITIAL_IONS; i++){
        Ions[i] = *(point*)malloc(sizeof(point));
        Ions[i].x = uniform_rand();
        Ions[i].y = uniform_rand();
    }
}

void print_first_5(point Ions[]){
    for(int i = 0; i<5; i++){
        printf("(%f,%f)\n", Ions[i].x, Ions[i].y);
    }
}

__device__ float distance(float p_1x, float p_1y, float p_2x, float p_2y){
    return sqrtf(powf(p_1x - p_2x, 2) + powf(p_1y - p_2y,2));
}

__global__ void update_Qs(float * dev_Q, point * dev_Ions, int iter){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tId%GRID_SIZE;
    int y = tId/GRID_SIZE;
    dev_Q[tId] += 1 / distance((float)x, (float)(y), dev_Ions[INITIAL_IONS + iter].x, dev_Ions[INITIAL_IONS + iter].y);
}

__global__ void set_Qs(float * dev_Q, point * dev_Ions){
        int tId = threadIdx.x + blockIdx.x * blockDim.x;
        if(tId < GRID_SIZE*GRID_SIZE){
            int x = tId%GRID_SIZE;
            int y = tId/GRID_SIZE;
            float q = 0;
            for(int i = 0; i<INITIAL_IONS; i++){
                q += 1 / distance((float)x, (float)y, dev_Ions[i].x, dev_Ions[i].y);
            }
            dev_Q[tId] = q;      
        }
}

int is_in_Ions(int limit, int pos){
    for(int i = INITIAL_IONS; i<INITIAL_IONS + limit; i++){
        if((int)Ions[i].x + (int)Ions[i].y * GRID_SIZE == pos){
            return 1;
        }
    }
    return 0;
}

int main(){

    float *dev_Q;
    point *dev_Ions;
    configSeed();
    populateIons();
    cudaMalloc(&dev_Ions, MAX_IONS*sizeof(point));
    cudaMalloc(&dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float));
    //La chicha con el kernel
    for(int i=0; i<MAX_IONS-INITIAL_IONS; i++){
        cudaMemcpy(dev_Ions, Ions, MAX_IONS*sizeof(point), cudaMemcpyHostToDevice);
        if(i==0){
            set_Qs<<<GRID_SIZE*GRID_SIZE/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_Q, dev_Ions);
        }else{
            update_Qs<<<GRID_SIZE*GRID_SIZE/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_Q, dev_Ions, i);
        }
        cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        float min = INFINITY;
        for(int n = 0; n<GRID_SIZE*GRID_SIZE; n++){
            if(is_in_Ions(i,n) != 1){
                int x = n%GRID_SIZE;
                int y = n/GRID_SIZE;
                if(Q[n]<min){
                    min = Q[n];
                    Ions[INITIAL_IONS+i].x = (float)x;
                    Ions[INITIAL_IONS+i].y = (float)y;
                }
            }
        }
    }
    cudaFree(dev_Ions);
    cudaFree(dev_Q);
    return 0;
}