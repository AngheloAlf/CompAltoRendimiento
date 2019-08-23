#ifndef REDUCTION_H
#define REDUCTION_H

__global__ void Q_reduction(float* dev_Q, float* partial_min, int* partial_min_pos);
__global__ void partial_reduction(float* partial_min, int* partial_min_pos);
__global__ void set_new_Ion(int new_ionIdx, float* dev_Q , float* new_ions_xs, float* new_ions_ys, float* partial_min, int* partial_min_pos);

#endif /* REDUCTION_H */
