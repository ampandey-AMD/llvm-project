// REQUIRES: asan-openmp-offload
// RUN: %clang_omp_offload_asan -O1 %s -o %t
// RUN: env HSA_XNACK=1 not %run %t 2>&1 | FileCheck %s

#include <cstdlib>

#pragma omp declare target
int Global[16];
#pragma omp end declare target

int main(int argc, char **argv) {
  int Idx = 16 + argc - 1;
#pragma omp target firstprivate(Idx)
  Global[Idx] = 1;
  return 0;
}

// CHECK: ERROR: AddressSanitizer: global-buffer-overflow on amdgpu device
// CHECK: WRITE of size 4
// CHECK: #0 {{.*}} in {{.*}}openmp-global-oob.cpp:14
