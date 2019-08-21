#include <stdio.h>
#include <stdlib.h>
//#include <math.h>

#define GRID_SIZE 8192//2048
#define INITIAL_IONS 5000//125   //5000
#define MAX_IONS 6000
#define BLOCK_SIZE 256//1024

typedef struct Ions Ions;

struct Ions{
    float xs[MAX_IONS-INITIAL_IONS];
    float ys[MAX_IONS-INITIAL_IONS];
};

__constant__ float dev_initial_ions[2*INITIAL_IONS];

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
    //if(!((p_2x<GRID_SIZE && p_2x>0) && (p_2y<GRID_SIZE && p_2y>0))) printf("miss!: p2=(%f,%f)\n",p_1x,p_1y,p_2x,p_2y);
    //return 1;
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

__global__ void update_Qs(float * dev_Q, Ions * dev_Ions, int iter){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if(tId < GRID_SIZE*GRID_SIZE){
        int x = tId%GRID_SIZE;
        int y = tId/GRID_SIZE;
        if(dev_Q[tId] != INFINITY){
            dev_Q[tId] += 1/distance((float)x, (float)(y), dev_Ions->xs[INITIAL_IONS + iter - 1], dev_Ions->ys[INITIAL_IONS + iter - 1]);
        }
    }
}   
 
__global__ void set_Qs(float * dev_Q){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    float q = 0;
    if(tId < GRID_SIZE*GRID_SIZE){
        for(int i = 0; i<INITIAL_IONS; i++){
            //float dist = distance((float)(tId%GRID_SIZE), (float)(tId/GRID_SIZE), dev_initial_ions[i], dev_initial_ions[i+INITIAL_IONS]);
            //if(dist == 0) printf("@#*\n");
            q += 1/distance((float)(tId%GRID_SIZE), (float)(tId/GRID_SIZE), dev_initial_ions[i], dev_initial_ions[i+INITIAL_IONS]);
        }
        dev_Q[tId] = q;      
        /*if(tId<10){
            printf("%f\n",dev_Q[tId]);
        }*/
    }
}

int searchMinPos(float * dev_Q){
    float * dev_partials_ans;
    int * dev_partials_pos;
    float * dev_min;
    int * dev_min_pos;
    float * min = (float*)malloc(sizeof(float));
    int * pos = (int*)malloc(sizeof(int));
    int sMemSize = BLOCK_SIZE*(sizeof(float) + sizeof(int));
    cudaMalloc(&dev_partials_ans, (GRID_SIZE*GRID_SIZE/BLOCK_SIZE)*sizeof(float));
    cudaMalloc(&dev_partials_pos, (GRID_SIZE*GRID_SIZE/BLOCK_SIZE)*sizeof(int));
    cudaMalloc(&dev_min, sizeof(float));
    cudaMalloc(&dev_min_pos, sizeof(int));
    cudaPartialsMinPos<<<(GRID_SIZE*GRID_SIZE)/BLOCK_SIZE, BLOCK_SIZE, sMemSize>>>(dev_Q, dev_partials_ans, dev_partials_pos);
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

/*void wea_q_no_funca(){
    float * Q;
    Ions * hst_Ions;
    float *dev_Q;
    Ions *dev_Ions;
    hst_Ions = (float*)malloc(sizeof(Ions));
    Q = (float*)malloc(GRID_SIZE * GRID_SIZE * sizeof(float));
    configSeed(10);
    populateIons(hst_Ions);
    print_first_5(hst_Ions);
    cudaMalloc(&dev_Ions, sizeof(Ions));
    cudaMalloc(&dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float));
    for(int i=0; i<MAX_IONS-INITIAL_IONS; i++){
        cudaMemcpy(dev_Ions, hst_Ions, sizeof(Ions), cudaMemcpyHostToDevice);
        if(i==0){
            set_Qs<<<(GRID_SIZE*GRID_SIZE/BLOCK_SIZE) + 1, BLOCK_SIZE>>>(dev_Q, dev_Ions);
        }else{
            update_Qs<<<(GRID_SIZE*GRID_SIZE/BLOCK_SIZE) + 1, BLOCK_SIZE>>>(dev_Q, dev_Ions, i);
        }
        cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        float min = INFINITY;
        int n_min = -1;
        for(int n = 0; n<10; n++){
            printf("; %f",Q[n]);
        }
        //for(int n =262*GRID_SIZE+288-2; n<262*GRID_SIZE+288+2; n++){
        //    printf("%f, pos(%i,%i)\n", Q[n], n%GRID_SIZE, n/GRID_SIZE);
        //}
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
    long blocks = (long)ceil((float)GRID_SIZE*GRID_SIZE/BLOCK_SIZE);
    float * Q = (float *)malloc(GRID_SIZE * GRID_SIZE * sizeof(float));
    float * hst_initial_ions = (float *)malloc(2*INITIAL_IONS*sizeof(float));
    float * dev_Q;
    //Ions * dev_Ions;
    configSeed(1566155583);//1566155583
    populateIons(hst_initial_ions);
    print_first_5(hst_initial_ions);
    print_last(5, hst_initial_ions);
    cudaMemcpyToSymbol(dev_initial_ions, hst_initial_ions, 2*INITIAL_IONS*sizeof(float), 0, cudaMemcpyHostToDevice);
    int infcudamalloc = cudaMalloc(&dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float));
    printf("malloc code: %i\n", infcudamalloc);
    set_Qs<<<(GRID_SIZE*GRID_SIZE)/BLOCK_SIZE, BLOCK_SIZE>>>(dev_Q);
    int info = cudaDeviceSynchronize();
    //int info = cudaMemcpy(Q, dev_Q, GRID_SIZE*GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cuda code: %i\n", info);
    /*for(int n = 0; n<10; n++){
        printf("; %f",Q[n]);
    }*/
    //cudaFree(dev_Q);
    free(Q);
}

int main(){
    //wea_q_no_funca();
    wea_test();
    return 0;
}