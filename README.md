**`skiplist_cuda`: A parallel/CUDA implementation of skiplist**

[Skiplists](http://dl.acm.org/citation.cfm?id=78977) are a variant of linked
lists that allow insertions in O(log n) time. Inspired by a
[GTC 2013 talk]
(http://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf),
we build a parallel implementation of skiplist where multiple GPU threads can
insert simultaneously. We make heavy use of [atomic compare-and-swap]
(http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomiccas) operation.
By default, our skiplist contains integers; to use different types, modify the
type definition `E` in `skip_parallel.h`.

We assume that you are using a 64-bit machine.

Usage
----
**Create a skiplist on the host:**
```cuda
#include "spmat_parallel.h"
...
// set heap size of 128 MB
cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
Skiplist *sl = skiplist_create();
...
```
**Important**: It is crucial to set heap size large enough to contain all the
elements to be inserted. When `skiplist_insert()` crashes in the middle, try
increasing the heap size.

**Insert elements to the skiplist**
Simply call `skiplist_insert()`. The following code stub will insert all
elements in the array `a` into the skiplist `sl`:
```cuda
#include "spmat_parallel.h"
...
__global__ void add(Skiplist *sl, int *a, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    while (x < N) {
        skiplist_insert(sl, a[x]);
        x += blockDim.x * gridDim.x;
    }
}
...
add<<<100, 320>>>(sl, a_dev, N);
```
**Remove elements from the list:** Unfortunately, this feature is not completed
yet. We'll try to make some time in the future.

**Collect the content to the host:**
```cuda
E *result = skiplist_gather(sl, &result_dim);
// result points to a host buffer
// result_dim contains the number of elements
```
This function will allocate a buffer on the host and return its pointer. At
the same time, it will return the number of elements via the second argument.

**De-allocate the skiplist**
```cuda
skiplist_destroy(sl);
```

How to compile the library
----
```bash
make
```
This will compile the library, along with a sample program in
`tester_parallel.cu`.

How to link your program with the library
----
```bash
nvcc -o [your executable] [your object files] skip_parallel.o safety.o
```
`safety.o` contains a little macro that checks the return value of all
CUDA API functions.

Dependencies
----
  - [nVIDIA CUDA toolkit](http://docs.nvidia.com/cuda)

Credits
----
Hyunsu "Philip" Cho and Sam Johnson, Trinity College, Hartford, CT
