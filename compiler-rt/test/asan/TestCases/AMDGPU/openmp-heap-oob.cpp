// REQUIRES: asan-openmp-offload
// RUN: %clang_omp_offload_asan -O1 %s -o %t
// RUN: env HSA_XNACK=1 not %run %t 2>&1 | FileCheck %s

#include <omp.h>

int main() {
  int Dev = omp_get_default_device();
  int *A = static_cast<int *>(omp_target_alloc(100 * sizeof(int), Dev));
#pragma omp target is_device_ptr(A)
  for (int I = 0; I <= 100; ++I)
    A[I] = I;
  omp_target_free(A, Dev);
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device
// CHECK: WRITE of size 4
// CHECK: #0 {{.*}} in {{.*}}openmp-heap-oob.cpp:12
// CHECK: is located 0 bytes after 400-byte region
// CHECK: allocated by thread T0 here:
