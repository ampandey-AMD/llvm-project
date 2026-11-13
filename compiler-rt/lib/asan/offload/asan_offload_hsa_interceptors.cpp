//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// HSA interceptors for host-side AddressSanitizer offload support. The
/// 'hsa_init' interceptor lives in the main runtime and calls into this file
/// through '__asan_offload_hsa_init'.
///
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

#include "asan_offload.h"
#include "asan_report.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_platform.h"

#if !SANITIZER_LINUX
#  error "Offload ASan reporting is supported on Linux only"
#endif

using namespace __sanitizer;
using namespace __asan;

namespace {

static StaticSpinMutex InitMutex;
static StaticSpinMutex HsaMutex;
static atomic_uint8_t Initialized;
static void* HsaHandle;

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;
  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;
  Offload::Get().RegisterHandler(HandleOffloadReport);
  SetOffloadSymbolizer([](uptr PC) { return Offload::Get().Symbolize(PC); });
  Atexit([] { Offload::Get().UntrackImages(); });
  AddDieCallback([] { Offload::Get().UntrackImages(); });
  atomic_store(&Initialized, 1, memory_order_release);
}

}  // namespace

// DSOs with a static runtime all export these wrappers and interpose onto the
// first, so 'RTLD_NEXT' can loop back into it. Resolve from HSA directly
// without loading it.
static void* HsaSymbol(const char* Name) {
  SpinMutexLock L(&HsaMutex);
  constexpr const char* Libs[] = {SANITIZER_HSA_LIBRARY ".so.1",
                                  SANITIZER_HSA_LIBRARY ".so"};
  for (const char* Lib : Libs)
    if (!HsaHandle)
      HsaHandle = dlopen(Lib, RTLD_LAZY | RTLD_NOLOAD);
  return HsaHandle ? dlsym(HsaHandle, Name) : nullptr;
}

// The shared runtime exports these to every program, act as if HSA is absent
// when it is not loaded.
#define ASAN_HSA_ENTER(name)                                                \
  Initialize();                                                             \
  if (UNLIKELY(!REAL(name))) {                                              \
    REAL(name) = reinterpret_cast<decltype(REAL(name))>(HsaSymbol(#name));  \
    if (UNLIKELY(!REAL(name))) {                                            \
      VReport(1, "%s: cannot find %s in this process\n", SanitizerToolName, \
              #name);                                                       \
      return HSA_STATUS_ERROR;                                              \
    }                                                                       \
  }

#define ASAN_HSA_FORWARD(name, ...)      \
  ASAN_HSA_ENTER(name);                  \
  if (UNLIKELY(!Offload::Get().Ready())) \
    return REAL(name)(__VA_ARGS__);

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  ASAN_HSA_ENTER(hsa_shut_down);

  Offload::Get().Shutdown();
  return REAL(hsa_shut_down)();
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char* Options) {
  ASAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status == HSA_STATUS_SUCCESS)
    Offload::Get().TrackExecutable(Executable);
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  ASAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  Offload::Get().UntrackExecutable(Executable);
  return REAL(hsa_executable_destroy)(Executable);
}

extern "C" void __asan_offload_init() { Initialize(); }

// Every HSA client reaches this through the 'hsa_init' interceptor, including
// ones whose device code reports through other means, so failure only
// disables device reports.
extern "C" void __asan_offload_hsa_init() {
  Initialize();
  if (!Offload::Get().Init())
    Report(
        "WARNING: %s: cannot initialize HSA offload support, device "
        "errors will not be reported\n",
        SanitizerToolName);
}

__attribute__((constructor)) static void AsanOffloadDynInit() {
  WarnIfHsaNotInterposed();
}
