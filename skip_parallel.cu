#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "skip_parallel.h"
#define MAX_LEVEL 32

#define MATCH_BARRIER 0x80000000

struct Node {
  Node **next;
  E val;
  int level;
};

struct Skiplist {
  Node *head;
};

/* utility functions: do not use outside of this source file */
__device__ static Node *node_search(Skiplist *sl, E elem, int desired_level);
__device__ static Node *node_create(E val, int level);
__device__ static void node_destroy(Node *node);
__global__ static void skiplist_size_internal(Skiplist *sl, int *size_out);
__global__ static void create_head(Skiplist *sl);
__global__ static void skiplist_destroy_traverse(Skiplist *sl, Node **to_free);
__global__ static void skiplist_destroy_free(Node **to_free, int size);
__device__ static int rand(unsigned int random);
__global__ static void skiplist_gather_internal(E *dest, Skiplist *sl);

/* Initializes the skiplist to contain nothing but an empty head node */
__global__ static void create_head(Skiplist *sl)
{
  sl->head = node_create(MIN_VAL, MAX_LEVEL);
  memset(sl->head->next, 0, MAX_LEVEL * sizeof(Node *));
}

/* Performs a traversal of the list and completes preliminary legwork so that
 * memory can be freed after it is used in the list.
 */
__global__ static void skiplist_destroy_traverse(Skiplist *sl, Node **to_free)
{
  Node *cur;

  int i = 0;

  cur = sl->head;

  // traverse through the nodes and delete each of them from memory
  while (cur != NULL) {
    to_free[i] = cur;

    cur = cur->next[0];
    i++;
  }
}

/* Uses the array of elements created above to actually free up Node memory. */
__global__ static void skiplist_destroy_free(Node **to_free, int size)
{
	// Parallelizing the process of freeing Nodes from memory.
  int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Frees Nodes like in skip_serial but uses threads to do this concurrently
  while (i < size) {
    free(to_free[i]->next);
    free(to_free[i]);

    i += blockDim.x * gridDim.x;
  }
}

/* Launches a kernel to create a header for the new skiplist. */
Skiplist *skiplist_create(void)
{
  Skiplist *sl;
  CHECK(cudaMalloc(&sl, sizeof(Skiplist)));

  create_head<<<1, 1>>>(sl);
  CHECK(cudaDeviceSynchronize());

  return sl;
}

/* Combines destory_traverse and destroy_free functions to actually free
 * memory once the skiplist need not be used again.
 */
void skiplist_destroy(Skiplist *sl)
{
  Node **to_free;
  int size = skiplist_size(sl);

  CHECK(cudaMalloc((void **)&to_free, size * sizeof(Node *)));

  skiplist_destroy_traverse<<<1, 1>>>(sl, to_free);
  skiplist_destroy_free<<<100, 512>>>(to_free, size);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaFree(sl));
  CHECK(cudaFree(to_free));
}

/* Inserts a new node into the list in parallel. */
__device__ void skiplist_insert(Skiplist *sl, E elem)
{
  Node *new_node;
  Node *dest;
  Node *link_first_read, *link_second_read;
  int i;
  int level;

  // Randomly generate i.
  // The level of the node is given by the number of successive 1-bits at
  // the tail of i, plus 1.
  level = 1;
  while (rand(clock()) == 1 && level < MAX_LEVEL)
    level++;

  new_node = node_create(elem, level);

  // insert the new node into the skiplist
  //int tries;
  for (i = 0; i < level; i++) {
    do {
			// We keep track of the original form of the list for checking purposes
      dest = node_search(sl, elem, i); // want to insert right after this node
      link_first_read = dest->next[i];

			// Checks for obvious errors before inserting
      if (link_first_read != NULL && link_first_read->val < elem) {
        continue;
      }

			// Allow each node/thread to set the forward facing pointer.
      new_node->next[i] = link_first_read;
      // check if dest->next[i] contains the same value as a while ago
      // if so, make it point to the new node. 
      // otherwise, declare failure and try again.
			// this would mean that someone else has already inserted.
      link_second_read
      = (Node *)atomicCAS((unsigned long long int *)&(dest->next[i]),
        *(unsigned long long int *)&link_first_read,
        *(unsigned long long int *)&new_node);
    } while (link_first_read != link_second_read);
  }
}
 
/* Simply traverses the bottom level of the skiplist and returns size */
int skiplist_size(Skiplist *sl)
{
  int *size, *size_dev;
  int size_result;
  CHECK(cudaHostAlloc(&size, sizeof(int), cudaHostAllocMapped));
  CHECK(cudaHostGetDevicePointer(&size_dev, size, 0));

  skiplist_size_internal<<<1, 1>>>(sl, size_dev);
  CHECK(cudaDeviceSynchronize());

  size_result = *size;
  CHECK(cudaFreeHost(size));

  return size_result;
}
/* Retrieves the size on the device and writes back to the kernel above. */
__global__ static void skiplist_size_internal(Skiplist *sl, int *size_out)
{
  Node *cur = skiplist_head(sl);
  int size = 0;

  if (cur->next[0] == NULL) {
    *size_out = 0;
    return;
  }

  cur = cur->next[0]; // skip the (empty) head node

  while (cur != NULL) {
    size++;
    cur = cur->next[0];
  }

  *size_out = size; // write across PCI channel
}

/* Returns an array of type E containing only sorted values from the list */
E *skiplist_gather(Skiplist *sl, int *dim)
{
  int size = skiplist_size(sl);
  E *dest_dev;
  E *dest = (E *)malloc(size * sizeof(E));

  CHECK(cudaMalloc(&dest_dev, size * sizeof(E)));

  skiplist_gather_internal<<<1, 1>>>(dest_dev, sl);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(dest, dest_dev, size * sizeof(E), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dest_dev));

  *dim = size;
  return dest;
}

/* Retrieves the elements of the list on the device much like size_internal. */
__global__ static void skiplist_gather_internal(E *dest, Skiplist *sl)
{
  Node *cur = skiplist_head(sl);
  int i;

  if (cur->next[0] == NULL)
    return;

  cur = cur->next[0];

  i = 0;
  while (cur != NULL) {
    dest[i] = cur->val;
    cur = cur->next[0];
    i++;
  }
}

/* Returns the pointer to the head of the skip list. */
__device__ Node *skiplist_head(Skiplist *sl)
{
  return sl->head;
}

/* Fetches the reference to the next Node in the skiplist. */
__device__ Node *node_next(Node *node)
{
  if (node == NULL)
    return NULL;
  else
    return node->next[0];
}

/* Constructor function for a single node belonging to the skiplist. */
__device__ static Node *node_create(E val, int level)
{
  Node *node = (Node *)malloc(sizeof(Node));

  node->val = val;
  node->level = level;
  node->next = (Node **)malloc(level * sizeof(Node *));

  return node;
}

/* Simplifies the process of freeing node data. */
__device__ static void node_destroy(Node *node)
{
  free(node->next);
  free(node);
}

/* Traverses the skiplist and searches for a particular element. */
__device__ static Node *node_search(Skiplist *sl, E elem, int desired_level)
{
  Node *cur = skiplist_head(sl);
  Node *next_node;
  int level;

  for (level = MAX_LEVEL - 1; level >= desired_level; level--) {
    next_node = cur->next[level];
    while (next_node != NULL && next_node->val < elem) {
      cur = next_node;
      next_node = cur->next[level];
    }
  }

  return cur;
}
/* Necessary function for generating a random number on the GPU. This is  
 * utilized when creating a new node and assigning an appropriate level value.
 */
__device__ static int rand(unsigned int random)
{
 	//See Figure 2 of 'GPU Random Numbers via the Tiny Encryption Algorithm', Zafar (2010).
	unsigned int sum, v0, v1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  //Start hashing.
  sum = 0;
  v0 = tid;
  v1 = random;
  
  sum += 0x9e3779b9;
  v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
  v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
  
  sum += 0x9e3779b9;
  v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
  v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
  
  sum += 0x9e3779b9;
  v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
  v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
  
  sum += 0x9e3779b9;
  v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
  v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
  
  return ((v0 + v1) < MATCH_BARRIER ? 0 : 1);
}
