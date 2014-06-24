#include <stdio.h>
#include <stdlib.h>

void _check(cudaError_t cs, const char *file, long line)
{
  const char *errstr;

  if (cs != cudaSuccess) {
    errstr = cudaGetErrorString(cs);
    printf("CUDA error %s at %s:%ld.\n", errstr, file, line);
    exit(1);
  }
}
