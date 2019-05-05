all: cpu gpu_2 gpu_3 gpu_4

cpu:
	gcc src_c/main_cpu.c -o lab2_cpu -Wall -Wextra -Wpedantic -std=gnu11 -O2

cpu_debug:
	gcc src_c/main_cpu.c -o lab2_cpu -Wall -Wextra -Wpedantic -std=gnu11 -g -Og

gpu_2:
	nvcc src_cu_2/main.cu -o lab2_gpu_2

gpu_3:
	nvcc src_cu_3/main.cu -o lab2_gpu_3

gpu_4:
	# nvcc src_cu_3/main.cu -o lab2_gpu_3
