#include <stdlib.h>
#include <string.h>
#include "skip_serial.h"
#define MAX_LEVEL 33

struct Node {
  Node **next;
  E val;
  int level;
};

struct Skiplist {
  Node *head;
};

/* utility functions: do not use outside of this source file */
static Node *node_search(Skiplist *sl, E elem, int desired_level);
static Node *node_create(E val, int level);
static void node_destroy(Node *node);

/* Initializes the skiplist to contain nothing but an empty head node */
Skiplist *skiplist_create(void)
{
  int i;
  Skiplist *sl = malloc(sizeof(Skiplist));

  sl->head = node_create(0, MAX_LEVEL);
  // set all links null
  memset(sl->head->next, 0, MAX_LEVEL * sizeof(Node *));

  return sl;
}

/* Frees up the memory used for individual nodes in the skiplist */
void skiplist_destroy(Skiplist *sl)
{
  Node *cur;
  Node *next;

  cur = skiplist_head(sl);
  next = node_next(cur); // remember the node following cur

  // traverse through the nodes and delete each of them from memory
  while (cur != NULL) {
    free(cur->next);
    free(cur);

    cur = next;
    next = node_next(cur);
  }

  free(sl);
}

/* Creates a node which is inserted in order to the skiplist */
void skiplist_insert(Skiplist *sl, E elem)
{
  Node *new_node;
  Node *dest;
  int i;
  int level;

  // Randomly generate i.
  // The level of the node is given by the number of successive 1-bits at
  // the tail of i, plus 1.
  level = 1;
  for (i = rand(); (i & 1) == 1; i >>= 1)
    level++;

  new_node = node_create(elem, level);

  // insert the new node into the skiplist
  for (i = 0; i < level; i++) {
    dest = node_search(sl, elem, i);// want to insert right after this node
    new_node->next[i] = dest->next[i];
    dest->next[i] = new_node; // need atomics here when done on GPU
  }
}

/* Removes a certain element from the skiplist and appropriately reassigns
 * pointers once the element is taken out of the list.
 */
void skiplist_remove(Skiplist *sl, E elem)
{
  Node *prev_node = node_search(sl, elem, 0); // node previous to 'elem'
  Node *target_node = prev_node->next[0]; // node containing 'elem' if in list
  int i;

  if (target_node->val != elem)
    return; // elem not found

  // Bypass the removed node by adjusting pointers from the top down.
  // remove top level first
  for (i = target_node->level - 1; i >= 0; i--) {
    prev_node = node_search(sl, elem, i);
    prev_node->next[i] = target_node->next[i]; // need atomics here
  }

  node_destroy(target_node);
}

/* Simply traverses the bottom level of the skiplist and returns size */
int skiplist_size(Skiplist *sl)
{
  Node *cur = skiplist_head(sl);
  int size = 0;

  if (cur->next[0] == NULL)
    return 0;

  cur = cur->next[0]; // skip the (empty) head node

  while (cur != NULL) {
    size++;
    cur = cur->next[0];
  }

  return size;
}

/* Returns an array of type E containing only sorted values from the list */
E *skiplist_gather(Skiplist *sl, int *dim)
{
  int size = skiplist_size(sl);
  E *dest = (E *)malloc(size * sizeof(E));

  Node *cur = skiplist_head(sl);
  int i;

  if (cur->next[0] == NULL)
    return;

  i = 0;
  while (cur != NULL) {
    dest[i] = cur->val;
    cur = cur->next[0];
    i++;
  }

  *dim = size;
  return dest;
}

/* Returns the pointer to the head of the skip list. */
Node *skiplist_head(Skiplist *sl)
{
  return sl->head;
}

E node_val(Node *node)
{
  return node->val;
}

/* Fetches the reference to the next Node in the skiplist. */
Node *node_next(Node *node)
{
  if (node == NULL)
    return NULL;
  else
    return node->next[0];
}

/* Constructor function for a single node belonging to the skiplist. */
static Node *node_create(E val, int level)
{
  Node *node = malloc(sizeof(Node));

  node->val = val;
  node->level = level;
  node->next = malloc(level * sizeof(Node *));

  return node;
}

/* Simplifies the process of freeing node data. */
static void node_destroy(Node *node)
{
  free(node->next);
  free(node);
}

/* Traverses the skiplist and searches for a particular element. */
static Node *node_search(Skiplist *sl, E elem, int desired_level)
{
  Node *cur = skiplist_head(sl);
  int level;

  for (level = MAX_LEVEL - 1; level >= desired_level; level--) {
    while (cur->next[level] != NULL && cur->next[level]->val < elem)
      cur = cur->next[level];
  }

  return cur;
}
