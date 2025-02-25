#include <stdlib.h>
#include <stdio.h>

void chk_args(int argc, char **argv){
    if(argc <= 2){
        printf("Usage: %s filename out_name\n", argv[0]);
        exit(-1);
    }
}

/** 
 * Loads the color channel from file into the array.
 * Inputs:
     - FILE *img: The already opened image file.
     - float *arr: The color channel.
     - long M: The size of the row of the image.
     - long N: The size of the column of the image.
 * Output:
     - void.
**/
void load_row(FILE *img, float *arr, long M, long N){
    for(long y = 0; y < M*N; ++y){
        if(fscanf(img, "%f", &arr[y]) != 1){
            fprintf(stderr, "Error while reading\n");
            exit(-2);
        }
    }
}

/** 
 * Reads the file and stores its RGB values in arrays
 * Inputs:
     - char *filename: The name of the file to read.
     - float **r_arr: Here will be stored the array of the red channel. The image is linearized. Each image is next to each other.
     - float **g_arr: Here will be stored the array of the green channel. The image is linearized. Each image is next to each other.
     - float **b_arr: Here will be stored the array of the blue channel. The image is linearized. Each image is next to each other.
     - long *M: Here will be stored the amounts of rows per image.
     - long *N: Here will be stored the amounts of columns per image.
 * Output:
     - void.
**/
void load_file(char *filename, float **r_arr, float **g_arr, float **b_arr, long *M, long *N){
    FILE *img = fopen(filename, "r");
    fscanf(img, "%li %li", M, N);
    *r_arr = (float *)malloc(sizeof(float) * (*M)*(*N));
    *g_arr = (float *)malloc(sizeof(float) * (*M)*(*N));
    *b_arr = (float *)malloc(sizeof(float) * (*M)*(*N));

    load_row(img, *r_arr, *M, *N);
    load_row(img, *g_arr, *M, *N);
    load_row(img, *b_arr, *M, *N);

    fclose(img);
}

void write_file(char *outname, long M, long N, float *r_arr, float *g_arr, float *b_arr){
    FILE *out_file = fopen(outname, "w");
    fprintf(out_file, "%li %li\n", M, N);

    for(long i = 0; i < M*N-1; ++i){
        fprintf(out_file, "%f ", r_arr[i]);
    }
    fprintf(out_file, "%f\n", r_arr[M*N-1]);
    for(long i = 0; i < M*N-1; ++i){
        fprintf(out_file, "%f ", g_arr[i]);
    }
    fprintf(out_file, "%f\n", g_arr[M*N-1]);
    for(long i = 0; i < M*N-1; ++i){
        fprintf(out_file, "%f ", b_arr[i]);
    }
    fprintf(out_file, "%f\n", b_arr[M*N-1]);

    fclose(out_file);
}

__global__ void intercalar(float *src_arr, long M, long N, long x){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if(tId < N/x/2*M){
        for(long i = 0; i < x; ++i){
            float temp = src_arr[tId*(2*x) + i];
            src_arr[tId*(2*x) + i] = src_arr[tId*(2*x) + i + x];
            src_arr[tId*(2*x) + i + x] = temp;
        }
    }
}

void generar_imagen(char *out_name, float *r_arr_gpu, float *g_arr_gpu, float *b_arr_gpu, long M, long N, long x){
    long block_size = 256;
    long grid_size = (long)ceil((float)N/x/2*M/block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    intercalar<<<grid_size, block_size>>>(r_arr_gpu, M, N, x);
    intercalar<<<grid_size, block_size>>>(g_arr_gpu, M, N, x);
    intercalar<<<grid_size, block_size>>>(b_arr_gpu, M, N, x);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf ("%f[ms]\n", milliseconds); 

    float *new_r_arr = (float *)malloc(M*N * sizeof(float));
    float *new_g_arr = (float *)malloc(M*N * sizeof(float));
    float *new_b_arr = (float *)malloc(M*N * sizeof(float));
    cudaMemcpy(new_r_arr, r_arr_gpu, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_g_arr, g_arr_gpu, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_b_arr, b_arr_gpu, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    write_file(out_name, M, N, new_r_arr, new_g_arr, new_b_arr);

    free(new_r_arr);
    free(new_g_arr);
    free(new_b_arr);
}


int main(int argc, char **argv){
    chk_args(argc, argv);
    float *r_arr, *g_arr, *b_arr;
    long M, N;
    load_file(argv[1], &r_arr, &g_arr, &b_arr, &M, &N);
    
    /* CUDA SETUP */
    float *r_arr_gpu;
    float *g_arr_gpu;
    float *b_arr_gpu;
    cudaMalloc(&r_arr_gpu, M*N * sizeof(float));
    cudaMalloc(&g_arr_gpu, M*N * sizeof(float));
    cudaMalloc(&b_arr_gpu, M*N * sizeof(float));
    
    cudaMemcpy(r_arr_gpu, r_arr, M*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_arr_gpu, g_arr, M*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_arr_gpu, b_arr, M*N * sizeof(float), cudaMemcpyHostToDevice);

    /* CUDA SETUP END */

    char *dst_name = (char *)malloc(strlen(argv[2]) + 3);
    strcpy(&dst_name[2], argv[2]);
    dst_name[1] = '_';

    for(long i = 0; i < 10; ++i){
        dst_name[0] = i + '0';
        long x = i == 0 ? 1 : 2 << (i-1);

        generar_imagen(dst_name, r_arr_gpu, g_arr_gpu, b_arr_gpu, M, N, x);
    }
    
    free(dst_name);

    cudaFree(r_arr_gpu);
    cudaFree(g_arr_gpu);
    cudaFree(b_arr_gpu);

    return 0;
}
