#ifndef REDUCTION_H
#define REDUCTION_H

float uniform_rand();
void configSeed(unsigned seed);

void populate(float *Q);

__global__ void cudaMin(float *dev_Q, float *ans);
__global__ void cudaPartialsMinPos(float *dev_Q, float *ans_min, int *ans_pos);
__global__ void cudaMinPos(float *partials, int *partials_pos, float *min, int *min_pos);

void searchMin(float *Q);
int searchMinPos(float *dev_Q);

#endif /* REDUCTION_H */
