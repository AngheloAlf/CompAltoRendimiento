#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

void chk_args(int argc, char **argv){
    if(argc <= 2){
        printf("Usage: %s filename out_name\n", argv[0]);
        exit(-1);
    }
}

void load_row(FILE *img, float *arr, long M, long N){
    for(long i = 0; i < N; ++i){
        for(long j = 0; j < M; ++j){
            if(fscanf(img, "%f", &arr[j*N + i]) != 1){
                fprintf(stderr, "Error while reading\n");
                exit(-2);
            }
        }
    }
}

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



    for(long i = 0; i < N; ++i){
        for(long j = 0; j < M; ++j){
            if(M*N-1 == (i+1)*(j+1)-1){
                break;
            }
            fprintf(out_file, "%f ", r_arr[j*N + i]);
        }
    }
    fprintf(out_file, "%f\n", r_arr[M*N-1]);

    for(long i = 0; i < N; ++i){
        for(long j = 0; j < M; ++j){
            if(M*N-1 == (i+1)*(j+1)-1){
                break;
            }
            fprintf(out_file, "%f ", g_arr[j*N + i]);
        }
    }
    fprintf(out_file, "%f\n", g_arr[M*N-1]);

    for(long i = 0; i < N; ++i){
        for(long j = 0; j < M; ++j){
            if(M*N-1 == (i+1)*(j+1)-1){
                break;
            }
            fprintf(out_file, "%f ", b_arr[j*N + i]);
        }
    }
    fprintf(out_file, "%f\n", b_arr[M*N-1]);

    fclose(out_file);
}

float *intercalar(const float *arr, long M, long N, long x){
    float *final_img = calloc(M * N, sizeof(float));

    for(long i = 0; i < N/x; ++i){
        for(long tId = 0; tId < M*x; ++tId){
            final_img[i*x + tId] = arr[(i+1)*x + tId];
            final_img[(i+1)*x + tId] = arr[i*x + tId];
        }
    }



    return final_img;
}

int main(int argc, char **argv){
    chk_args(argc, argv);
    float *r_arr, *g_arr, *b_arr;
    long M, N;
    load_file(argv[1], &r_arr, &g_arr, &b_arr, &M, &N);

    char *dst_name = malloc(strlen(argv[2]) + 3);
    strcpy(&dst_name[2], argv[2]);
    dst_name[1] = '_';

    for(long x = 0; x < 10; ++x){
        dst_name[0] = x + '0';
        clock_t t = clock();
        float *new_r_arr = intercalar(r_arr, M, N, pow(2, x));
        float *new_g_arr = intercalar(g_arr, M, N, pow(2, x));
        float *new_b_arr = intercalar(b_arr, M, N, pow(2, x));
        t = clock() - t;
        printf ("%f[ms]\n",((float)t)/CLOCKS_PER_SEC * 1000); /* http://www.cplusplus.com/reference/ctime/clock/ */

        write_file(dst_name, M, N, new_r_arr, new_g_arr, new_b_arr);

        free(new_r_arr);
        free(new_g_arr);
        free(new_b_arr);
    }

    free(dst_name);

    return 0;
}
