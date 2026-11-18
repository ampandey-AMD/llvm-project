// RUN: %clangxx_asan -O0 %s -o %t -Wl,-rpath,%compiler_rt_libdir
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not=AddressSanitizer
// RUN: env SANITIZER_HSA_RUNTIME_PATH=%rocm_lib/libhsa-runtime64.so.1 \
// RUN:   %run %t 2>&1 | FileCheck %s --implicit-check-not=AddressSanitizer
// RUN: %env_asan_opts=verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=VERBOSE
//
// The ASan runtime exports the HSA interceptors to every program. In a program
// that never loaded ROCr they must behave as if HSA is absent: return an error
// instead of calling a null REAL(), and never load a ROCr themselves.
//
// REQUIRES: linux, stable-runtime, rocm
// UNSUPPORTED: android

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int (*InitFn)();
typedef int (*FreeFn)(void *);
typedef int (*CopyFn)(void *, const void *, size_t);

static int RocrMapped() {
  FILE *F = fopen("/proc/self/maps", "r");
  char Line[4096];
  int Found = 0;
  while (fgets(Line, sizeof(Line), F))
    if (strstr(Line, "libhsa-runtime64"))
      Found = 1;
  fclose(F);
  return Found;
}

int main() {
  FreeFn PoolFree = (FreeFn)dlsym(RTLD_DEFAULT, "hsa_amd_memory_pool_free");
  CopyFn Copy = (CopyFn)dlsym(RTLD_DEFAULT, "hsa_memory_copy");
  InitFn Init = (InitFn)dlsym(RTLD_DEFAULT, "hsa_init");
  if (!PoolFree || !Copy || !Init) {
    fprintf(stderr, "missing interceptor\n");
    return 1;
  }

  char Buf[8];
  fprintf(stderr, "pool_free: 0x%x\n", PoolFree(nullptr));
  fprintf(stderr, "memory_copy: 0x%x\n", Copy(Buf, Buf + 4, 4));
  fprintf(stderr, "hsa_init: 0x%x\n", Init());
  fprintf(stderr, "rocr mapped: %d\n", RocrMapped());
  return 0;
}

// CHECK: pool_free: 0x100b
// CHECK: memory_copy: 0x100b
// CHECK: hsa_init: 0x100b
// CHECK: rocr mapped: 0

// VERBOSE: could not resolve hsa_init from a loaded ROCr
// VERBOSE: hsa_init: 0x100b
