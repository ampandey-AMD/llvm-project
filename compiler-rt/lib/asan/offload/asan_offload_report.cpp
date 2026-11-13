//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Converts device AddressSanitizer reports into host error reports.
///
//===----------------------------------------------------------------------===//

#include "asan_offload.h"
#include "asan_report.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "shared/rpc.h"

using namespace __sanitizer;

namespace __asan {

u32 HandleOffloadReport(void* PortPtr, u32) {
  auto& Port = *reinterpret_cast<rpc::Server::Port*>(PortPtr);
  if (Port.get_opcode() != SANITIZER_OFFLOAD_ASAN)
    return rpc::RPC_UNHANDLED_OPCODE;

  // Layout expected by ErrorNonSelfAMDGPU: the device ID, the block ID, then
  // one work-item ID per reporting lane.
  constexpr u32 MaxLanes = 64;
  uptr Addrs[MaxLanes];
  u64 Ids[4 + MaxLanes] = {};
  u32 Count = 0;
  __asan_gpu_report First = {};
  Port.recv([&](rpc::Buffer* Buffer, u32) {
    __asan_gpu_report R;
    internal_memcpy(&R, Buffer->data, sizeof(R));
    if (Count == 0) {
      First = R;
      Ids[1] = R.block[0];
      Ids[2] = R.block[1];
      Ids[3] = R.block[2];
    }
    if (Count < MaxLanes) {
      Addrs[Count] = static_cast<uptr>(R.addr);
      Ids[4 + Count] = R.thread;
      ++Count;
    }
  });
  if (!Count)
    return rpc::RPC_SUCCESS;

  uptr PC = static_cast<uptr>(First.pc);
  u32 Size = First.size > ~u32(0) ? ~u32(0) : static_cast<u32>(First.size);
  ReportNonselfError(&PC, 1, Addrs, Count, Ids, 4 + Count, First.is_write, Size,
                     First.fatal, "amdgpu", /*nonself_vma_adjust=*/0,
                     /*nonself_fd=*/-1, /*nonself_file_extent_size=*/0,
                     /*nonself_file_extent_start=*/0);

  // The device waits for this before continuing or stopping the kernel.
  Port.send([](rpc::Buffer*, u32) {});
  return rpc::RPC_SUCCESS;
}

}  // namespace __asan
