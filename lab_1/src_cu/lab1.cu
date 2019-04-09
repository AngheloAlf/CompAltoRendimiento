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
     - long img_amount: The image index.
     - long row: The size of the row of the image.
     - long column: The size of the column of the image.
 * Output:
     - void.
**/
void load_row(FILE *img, float *arr, long img_number, long row, long column){
    for(long y = 0; y < row*column; ++y){
        if(fscanf(img, "%f", &arr[y + img_number*row*column]) != 1){
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
     - long *img_amount: Here will be stored the amounts of images.
     - long *row: Here will be stored the amounts of rows per image.
     - long *column: Here will be stored the amounts of columns per image.
 * Output:
     - void.
**/
void load_file(char *filename, float **r_arr, float **g_arr, float **b_arr, long *img_amount, long *row, long *column){
    FILE *img = fopen(filename, "r");
    fscanf(img, "%li %li %li", img_amount, row, column);
    *r_arr = (float *)malloc(sizeof(float) * (*img_amount) * (*row)*(*column));
    *g_arr = (float *)malloc(sizeof(float) * (*img_amount) * (*row)*(*column));
    *b_arr = (float *)malloc(sizeof(float) * (*img_amount) * (*row)*(*column));

    for(long i = 0; i < (*img_amount); ++i){
        load_row(img, *r_arr, i, *row, *column);
        load_row(img, *g_arr, i, *row, *column);
        load_row(img, *b_arr, i, *row, *column);
    }

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

__global__ void mean(float *dst_arr, const float *src_arr, long img_amount, long row, long column){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if(tId < row*column){
        for(long j = 0; j < img_amount; ++j){
            dst_arr[tId] += src_arr[tId + j*row*column];
        }

        dst_arr[tId] /= img_amount;
    }
}


int main(int argc, char **argv){
    chk_args(argc, argv);
    float *r_arr, *g_arr, *b_arr;
    long img_amount, row, column;
    load_file(argv[1], &r_arr, &g_arr, &b_arr, &img_amount, &row, &column);

    /* CUDA */
    float *r_arr_gpu;
    float *g_arr_gpu;
    float *b_arr_gpu;
    cudaMalloc(&r_arr_gpu, img_amount*row*column * sizeof(float));
    cudaMalloc(&g_arr_gpu, img_amount*row*column * sizeof(float));
    cudaMalloc(&b_arr_gpu, img_amount*row*column * sizeof(float));
    printf("B\n");
    cudaMemcpy(r_arr_gpu, r_arr, img_amount*row*column * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_arr_gpu, g_arr, img_amount*row*column * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_arr_gpu, b_arr, img_amount*row*column * sizeof(float), cudaMemcpyHostToDevice);
    printf("C\n");

    float *dst_r_arr;
    float *dst_g_arr;
    float *dst_b_arr;
    cudaMalloc(&dst_r_arr, row*column * sizeof(float));
    cudaMemset(dst_r_arr, 0, row*column * sizeof(float));
    cudaMalloc(&dst_g_arr, row*column * sizeof(float));
    cudaMemset(dst_r_arr, 0, row*column * sizeof(float));
    cudaMalloc(&dst_b_arr, row*column * sizeof(float));
    cudaMemset(dst_r_arr, 0, row*column * sizeof(float));

    long block_size = 256;
    long grid_size = (long)ceil((float)row*column/block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mean<<<grid_size, block_size>>>(dst_r_arr, r_arr_gpu, img_amount, row, column);
    mean<<<grid_size, block_size>>>(dst_g_arr, g_arr_gpu, img_amount, row, column);
    mean<<<grid_size, block_size>>>(dst_b_arr, b_arr_gpu, img_amount, row, column);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf ("%f[ms].\n", milliseconds); 

    float *new_r_arr = (float *)malloc(row*column * sizeof(float));
    float *new_g_arr = (float *)malloc(row*column * sizeof(float));
    float *new_b_arr = (float *)malloc(row*column * sizeof(float));
    cudaMemcpy(new_r_arr, dst_r_arr, sizeof(float)*row*column, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_g_arr, dst_g_arr, sizeof(float)*row*column, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_b_arr, dst_b_arr, sizeof(float)*row*column, cudaMemcpyDeviceToHost);

    write_file(argv[2], row, column, new_r_arr, new_g_arr, new_b_arr);

    free(new_r_arr);
    free(new_g_arr);
    free(new_b_arr);
    return 0;
}
