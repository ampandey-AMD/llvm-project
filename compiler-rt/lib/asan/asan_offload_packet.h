//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Report packet sent from the GPU AddressSanitizer runtime to the host. Each
// active lane of the reporting wave fills its own packet.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_OFFLOAD_PACKET_H
#define ASAN_OFFLOAD_PACKET_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_offload_opcodes.h"

struct __asan_gpu_report {
  __sanitizer::u64 pc;
  __sanitizer::u64 addr;
  __sanitizer::u64 size;
  __sanitizer::u32 block[3];
  __sanitizer::u32 thread;
  __sanitizer::u8 is_write;
  __sanitizer::u8 fatal;
  __sanitizer::u8 reserved[22];
};

static_assert(sizeof(__asan_gpu_report) == 64,
              "Offload ASan report must fit one RPC packet");

#endif  // ASAN_OFFLOAD_PACKET_H
