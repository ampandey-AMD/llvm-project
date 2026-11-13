// Team memory is lowered to device memory allocated by the runtime.

// REQUIRES: asan-openmp-offload
// RUN: %clang_omp_offload_asan -O1 %s -o %t
// RUN: env HSA_XNACK=1 %run %t 0 2>&1 | FileCheck %s --check-prefix=OK
// RUN: env HSA_XNACK=1 not %run %t 16 2>&1 | FileCheck %s

#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(int argc, char **argv) {
  int Idx = atoi(argv[1]);
  int Out = 0;
#pragma omp target teams num_teams(1) map(tofrom : Out) firstprivate(Idx)
  {
    int S[16];
#pragma omp allocate(S) allocator(omp_pteam_mem_alloc)
    S[Idx % 32] = 1;
    Out = S[0];
  }
  printf("Out = %d\n", Out);
  return 0;
}

// OK-NOT: ERROR: AddressSanitizer
// OK: Out = 1

// CHECK: ERROR: AddressSanitizer: {{.*}} on amdgpu device
// CHECK: WRITE of size 4
// CHECK: #0 {{.*}} in {{.*}}openmp-lds-oob.cpp:19
