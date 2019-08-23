#ifndef KERNELS_H
#define KERNELS_H

__global__ void set_Qs(float* dev_Q);
__global__ void update_Qs_by_chunk(float* dev_Q, int i, int chunk_size);
__global__ void set_Qs_r(float* dev_Q, float r);
__global__ void update_Qs(int new_ionIdx, float* dev_Q, float* new_ions_xs, float* new_ions_ys);
__global__ void update_Qs_r(int new_ionIdx, float* dev_Q, float* new_ions_xs, float* new_ions_ys, float r);

#endif /* REDUCTION_H */