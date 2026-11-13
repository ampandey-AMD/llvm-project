//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Host-side AddressSanitizer support for device reports.
///
//===----------------------------------------------------------------------===//

#ifndef ASAN_OFFLOAD_H
#define ASAN_OFFLOAD_H

#include "asan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __asan {

__sanitizer::u32 HandleOffloadReport(void* Port, __sanitizer::u32 Lanes);

}  // namespace __asan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __asan_offload_init();
SANITIZER_INTERFACE_ATTRIBUTE void __asan_offload_hsa_init();
}

#endif  // ASAN_OFFLOAD_H
