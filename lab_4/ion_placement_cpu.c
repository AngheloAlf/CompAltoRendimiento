#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define GRID_SIZE 2000

typedef struct Ion Ion;

struct Ion {
    float point[2];
    Ion * next;
};

Ion * ionList;

float Q[GRID_SIZE][GRID_SIZE];
int ion_population = 5000;

float uniform_rand(){
    return GRID_SIZE*((float) rand() / (RAND_MAX));
}

void configSeed(){
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    printf("results for seed %i\n", seed);
}

void pushIon(float x, float y){
    Ion * ion = (Ion *)malloc(sizeof(Ion));
    ion->point[0] = x;
    ion->point[1] = y;
    ion->next = ionList;
    ionList = ion;
}

Ion * getIon(int n){
    Ion * ion = ionList;
    for(int i=0; i<n; i++){
        ion = ion->next;
    }
    return ion;
}

float distance(float point_1[], float point_2[]){
    return sqrtf(powf(point_1[0] - point_2[0], 2) + powf(point_1[1] - point_2[1], 2));
}

void printIons(){
    Ion * ion = ionList;
    while(ion){
        printf("(%f,%f)\n", ion->point[0], ion->point[1]);
        ion = ion->next;
    }
}

int ion_in(int pos, int size){
    Ion * ion = ionList;
    for(int i=0; i<size; i++){
        if(ion->point[0] + ion->point[1] == pos){
            return 1;
        }
        ion = ion->next;
    }
    return 0;
}

void calculate_Qs(int iter){
    float vertex[2];
    int min[2];
    float q_min = INFINITY;
    if(iter = 0){
        for(int v = 0; v<GRID_SIZE*GRID_SIZE; v++){
            int x = v%GRID_SIZE;
            int y = v/GRID_SIZE;
            vertex[0] = x;
            vertex[1] = y;
            float q = 0;
            Ion * ion = ionList;
            while(ion){
                q += 1 / distance(ion->point, vertex);
                ion = ion->next;
            }
            Q[x][y] = q;
            if(q < q_min){
                q_min = q;
                min[0] = x;
                min[1] = y;
            }
        }
        printf("(%i,%i)-%f\n", min[0], min[1], q_min);
    }
    else{
        for(int v = 0; v<GRID_SIZE*GRID_SIZE; v++){
            if(ion_in(v, iter)==0){
                int x = v%GRID_SIZE;
                int y = v/GRID_SIZE;
                vertex[0] = x;
                vertex[1] = y;
                Q[x][y] += 1 / distance(ionList->point, vertex);
                if(Q[x][y] < q_min){
                    q_min = Q[x][y];
                    min[0] = x;
                    min[1] = y;
                }
            }
        }
    }
    if(q_min != INFINITY){
        pushIon(min[0],min[1]);
        ion_population++;
    }
}

void ion_populate(int size){
    for(int i=0; i<size; i++){
        pushIon(uniform_rand(),uniform_rand());
    }
}

int main(){
    configSeed();
    ion_populate(ion_population);
    for(int i = 0; i<1000; i++){
        calculate_Qs(i);
    }
    Ion * ion = ionList;
    for(int i = 0; i<1000; i++){
        printf("(%f,%f)\n", ion->point[0], ion->point[1]);
        ion = ion->next;
    }
    return 0;
}