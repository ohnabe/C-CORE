#ifndef THP_TYPES_INC
#define THP_TYPES_INC

#include <cstddef>

template <typename T> struct THPTypeInfo {};

namespace torch {

typedef struct THVoidStorage
{
  void *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  void *allocator;
  void *allocatorContext;
  THVoidStorage *view;
} THVoidStorage;

typedef struct THVoidTensor
{
   long *size;
   long *stride;
   int nDimension;
   THVoidStorage *storage;
   ptrdiff_t storageOffset;
   int refcount;
   char flag;
} THVoidTensor;

}  // namespace torch

#endif
