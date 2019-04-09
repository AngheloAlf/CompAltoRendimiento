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
    *r_arr = malloc(sizeof(float) * (*img_amount) * (*row)*(*column));
    *g_arr = malloc(sizeof(float) * (*img_amount) * (*row)*(*column));
    *b_arr = malloc(sizeof(float) * (*img_amount) * (*row)*(*column));

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

float *mean(const float *arr, long img_amount, long row, long column){
    float *final_img = malloc(sizeof(float) * row * column);
    for(long i = 0; i < row*column; ++i){
        i[final_img] = 0;
    }

    for(long i = 0; i < row*column; ++i){
        for(long j = 0; j < img_amount; ++j){
            i[final_img] += arr[i + j*row*column];
        }
    }

    for(long i = 0; i < row*column; ++i){
        i[final_img] /= img_amount;
    }

    return final_img;
}

int main(int argc, char **argv){
    chk_args(argc, argv);
    float *r_arr, *g_arr, *b_arr;
    long img_amount, row, column;
    load_file(argv[1], &r_arr, &g_arr, &b_arr, &img_amount, &row, &column);

    clock_t t = clock();
    float *new_r_arr = mean(r_arr, img_amount, row, column);
    float *new_g_arr = mean(g_arr, img_amount, row, column);
    float *new_b_arr = mean(b_arr, img_amount, row, column);
    t = clock() - t;
    printf ("%f[ms].\n",((float)t)/CLOCKS_PER_SEC * 1000); /* http://www.cplusplus.com/reference/ctime/clock/ */

    write_file(argv[2], row, column, new_r_arr, new_g_arr, new_b_arr);

    free(new_r_arr);
    free(new_g_arr);
    free(new_b_arr);
    return 0;
}
