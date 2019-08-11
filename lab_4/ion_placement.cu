#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID_SIZE 8192
#define INITIAL_IONS 5000
#define MAX_IONS 7000

struct point{
    float x,y;
} typedef point;

float uniform_rand(){
    return GRID_SIZE*((float) rand() / (RAND_MAX));
}

void configSeed(){
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    printf("results for seed %i\n", seed);
}

void populate(point * Ions[]){
    for(int i = 0; i<INITIAL_IONS; i++){
        *Ions[i] = (*point)malloc(sizeof(point));
        *Ions[i].x = uniform_rand();
        *Ions[i].y = uniform_rand();
    }
}

void print_first_5(point * Ions[]){
    for(int i = 0; i<5; i++){
        printf("(%f,%f)\n", *Ions[i].x, *Ions[i].y);
    }
}

int main(){
    point Ions[MAX_IONS];
    float Q[GRID_SIZE*GRID_SIZE];
    configSeed();
    populate(&Ions);
    print_first_5(&Ions);
    return 0;
}