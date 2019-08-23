#define SIZE_MALLA 8192//1024
#define BLOCK_SIZE 256//1024
#define INITIAL_IONS 5000
#define MAX_IONS 6000
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

__constant__ float dev_ini_ions_xs[INITIAL_IONS];
__constant__ float dev_ini_ions_ys[INITIAL_IONS];