// Recoverable reports are printed while the allocation is live, once per wave,
// and the kernel keeps running.

// REQUIRES: asan-openmp-offload
// RUN: %clang_omp_offload_asan -O1 -fsanitize-recover=address %s -o %t
// RUN: env HSA_XNACK=1 ASAN_OPTIONS=halt_on_error=0:detect_leaks=0 %run %t 2>&1 \
// RUN:   | FileCheck %s

#include <cstdio>
#include <omp.h>

int main() {
  const int N = 1000;
  int Dev = omp_get_default_device();
  int *A = static_cast<int *>(omp_target_alloc(N * sizeof(int), Dev));
#pragma omp target teams distribute parallel for is_device_ptr(A)
  for (int I = 0; I < N + 6; ++I)
    A[I] = I;
  omp_target_free(A, Dev);
  printf("after kernel\n");
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device
// CHECK: #0 {{.*}} in {{.*}}openmp-recover.cpp:18
// CHECK: is located 0 bytes after 4000-byte region
// CHECK-NOT: freed by thread
// CHECK: allocated by thread T0 here:
// CHECK-NOT: ERROR: AddressSanitizer
// CHECK: after kernel
