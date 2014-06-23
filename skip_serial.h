#include <limits.h>
typedef struct Node Node;
typedef struct Skiplist Skiplist;
typedef int E;
#define MIN_VAL INT_MIN

Skiplist *skiplist_create(void);
void skiplist_destroy(Skiplist *sl);
void skiplist_insert(Skiplist *sl, E elem);
void skiplist_remove(Skiplist *sl, E elem);
int skiplist_size(Skiplist *sl);
E *skiplist_gather(Skiplist *sl, int *dim);

/* for traversal */
Node *skiplist_head(Skiplist *sl);
E node_val(Node *node);
Node *node_next(Node *node);
