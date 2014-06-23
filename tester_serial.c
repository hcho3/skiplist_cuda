#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "skip_serial.h"

int main(void)
{
  int N = 100;
  int i;
  Skiplist *sl;
  Node *cur;
  int *result;
  int result_dim;

  srand(time(NULL));

  sl = skiplist_create();
  for (i = 0; i < N; i++)
    skiplist_insert(sl, rand() % 100);

  printf("size = %d\n", skiplist_size(sl));
  result = skiplist_gather(sl, &result_dim);
  printf("result_dim = %d\n", result_dim);
  for (i = 0; i < N; i++)
    printf("%d ", result[i]);
  printf("\n");

  free(result);

  skiplist_remove(sl, 30); 
  printf("size = %d\n", skiplist_size(sl));
  result = skiplist_gather(sl, &result_dim);
  printf("result_dim = %d\n", result_dim);
  for (i = 0; i < result_dim; i++)
    printf("%d ", result[i]);
  printf("\n");

  skiplist_destroy(sl);
  free(result);

  return 0;
}
