all: cpu gpu

cpu: src_c/lab1.c
	gcc src_c/lab1.c -o lab1_cpu.out -Wall -Wextra -Wpedantic -std=gnu11 -O2

cpu_debug: src_c/lab1.c
	gcc src_c/lab1.c -o lab1_cpu.out -Wall -Wextra -Wpedantic -std=gnu11 -g -Og

gpu: src_cu/lab1.cu
	nvcc src_cu/lab1.cu -o lab1_gpu
