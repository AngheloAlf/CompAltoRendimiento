#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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
     - long row: The size of the row of the image.
     - long column: The size of the column of the image.
 * Output:
     - void.
**/
void load_row(FILE *img, float *arr, long row, long column){
    for(long y = 0; y < row*column; ++y){
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
    *r_arr = malloc(sizeof(float) * (*M)*(*N));
    *g_arr = malloc(sizeof(float) * (*M)*(*N));
    *b_arr = malloc(sizeof(float) * (*M)*(*N));

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

float *intercalar(const float *arr, long M, long N, long x){
    float *final_img = calloc(M * N, sizeof(float));

    for(long amount = 0; amount < N/x; ++amount){
        for(long j = 0; j < M; ++j){
            for(long i = 0; i < x; ++i){
                final_img[amount*i + N*j] = arr[amount*i+x + N*j];
                final_img[amount*i+x + N*j] = arr[amount*i + N*j];
            }
        }
    }

    return final_img;
}

int main(int argc, char **argv){
    chk_args(argc, argv);
    float *r_arr, *g_arr, *b_arr;
    long M, N;
    load_file(argv[1], &r_arr, &g_arr, &b_arr, &M, &N);

    clock_t t = clock();
    long x = 64;
    float *new_r_arr = intercalar(r_arr, M, N, x);
    float *new_g_arr = intercalar(g_arr, M, N, x);
    float *new_b_arr = intercalar(b_arr, M, N, x);
    t = clock() - t;
    printf ("%f[ms].\n",((float)t)/CLOCKS_PER_SEC * 1000); /* http://www.cplusplus.com/reference/ctime/clock/ */

    write_file(argv[2], M, N, new_r_arr, new_g_arr, new_b_arr);

    free(new_r_arr);
    free(new_g_arr);
    free(new_b_arr);
    return 0;
}
