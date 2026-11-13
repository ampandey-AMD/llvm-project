//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AddressSanitizer runtime for GPU targets. Device code shares the host shadow
/// memory, so checks only read and write shadow bytes. Errors are forwarded to
/// the host runtime over RPC.
///
//===----------------------------------------------------------------------===//

#include <gpuintrin.h>

#include "asan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

using namespace __sanitizer;

#define INTERFACE extern "C" SANITIZER_INTERFACE_ATTRIBUTE
// The AMDGPU backend emits calls to these after LTO has internalized the
// runtime, e.g. when lowering LDS for sanitized kernels.
#define CODEGEN_INTERFACE INTERFACE __attribute__((used))

// The device link wraps malloc and free so that user allocations go through
// the instrumented allocator below.
extern "C" {
void* __real_malloc(uptr size);
void __real_free(void* ptr);
}

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client client asm("__llvm_rpc_client");

namespace {

// Must match the AMDGPU mapping in the AddressSanitizer pass.
constexpr uptr kShadowScale = 3;
constexpr uptr kShadowGranularity = 1ull << kShadowScale;
constexpr uptr kShadowOffset = 0x7fff8000;

constexpr u8 kHeapLeftRedzoneMagic = 0xfa;
constexpr u8 kHeapFreeMagic = 0xfd;
constexpr u8 kUserPoisonedMemoryMagic = 0xf7;
constexpr u8 kArrayCookieMagic = 0xac;
constexpr s8 kGlobalRedzoneMagic = static_cast<s8>(0xf9);

// Must match the device malloc header scanned by the host report.
constexpr u64 kAllocMagic = 0xfedcba1ee1abcdefull;
constexpr uptr kAllocHeaderBytes = 32;
constexpr uptr kAllocRedzoneBytes = 32;

struct AllocHeader {
  u64 magic;
  u64 base;
  u64 pc;
  u32 total;
  u32 size;
};
static_assert(sizeof(AllocHeader) == kAllocHeaderBytes,
              "header must fill the left redzone");

// Layout of the instrumented global descriptor emitted by the compiler.
struct GlobalSourceLocation {
  const char* filename;
  int line_no;
  int column_no;
};

struct Global {
  uptr beg;
  uptr size;
  uptr size_with_redzone;
  const char* name;
  const char* module_name;
  uptr has_dynamic_init;
  GlobalSourceLocation* location;
  uptr odr_indicator;
};

__gpu_global s8* mem_to_shadow(uptr addr) {
  return reinterpret_cast<__gpu_global s8*>((addr >> kShadowScale) +
                                            kShadowOffset);
}

uptr round_up(uptr size, uptr boundary) {
  return (size + boundary - 1) & ~(boundary - 1);
}

uptr round_down(uptr size, uptr boundary) { return size & ~(boundary - 1); }

bool is_aligned(uptr addr) { return (addr & (kShadowGranularity - 1)) == 0; }

void fill_shadow(uptr aligned_beg, uptr aligned_size, s8 value) {
  __gpu_global s8* shadow = mem_to_shadow(aligned_beg);
  for (uptr i = 0, e = aligned_size / kShadowGranularity; i < e; ++i)
    shadow[i] = value;
}

bool is_address_poisoned(uptr addr) {
  s8 shadow_value = *mem_to_shadow(addr);
  if (!shadow_value)
    return false;
  return static_cast<s8>(addr & (kShadowGranularity - 1)) >= shadow_value;
}

bool is_invalid_access(uptr addr, uptr size) {
  if (size <= kShadowGranularity) {
    s8 shadow_value = *mem_to_shadow(addr);
    return shadow_value && static_cast<s8>((addr & (kShadowGranularity - 1)) +
                                           size - 1) >= shadow_value;
  }
  return *reinterpret_cast<__gpu_global s16*>(mem_to_shadow(addr)) != 0;
}

uptr first_poisoned(uptr beg, uptr end) {
  for (uptr addr = beg; addr < end; ++addr)
    if (is_address_poisoned(addr))
      return addr;
  return 0;
}

// Returns the first poisoned byte in [beg, beg + size), or zero.
uptr region_is_poisoned(uptr beg, uptr size) {
  if (!size)
    return 0;
  uptr end = beg + size;
  uptr aligned_beg = round_up(beg, kShadowGranularity);
  uptr aligned_end = round_down(end, kShadowGranularity);
  if (aligned_beg >= aligned_end)
    return first_poisoned(beg, end);
  if (uptr addr = first_poisoned(beg, aligned_beg))
    return addr;
  __gpu_global s8* shadow = mem_to_shadow(aligned_beg);
  for (uptr i = 0, e = (aligned_end - aligned_beg) / kShadowGranularity; i < e;
       ++i) {
    // A positive shadow value is the number of addressable leading bytes.
    if (s8 value = shadow[i])
      return aligned_beg + i * kShadowGranularity + (value > 0 ? value : 0);
  }
  return first_poisoned(aligned_end, end);
}

// Linear work-item ID within the work-group.
u32 thread_id() {
#ifdef __AMDGPU__
  // Read the group sizes directly to avoid a reference to the OCLC ABI version.
  const u16 __gpu_constant* sizes = reinterpret_cast<const u16 __gpu_constant*>(
      reinterpret_cast<const u8 __gpu_constant*>(
          __builtin_amdgcn_implicitarg_ptr()) +
      12);
  const u32 nx = sizes[0], ny = sizes[1];
#else
  const u32 nx = __gpu_num_threads(__GPU_X_DIM);
  const u32 ny = __gpu_num_threads(__GPU_Y_DIM);
#endif
  return __gpu_thread_id(__GPU_X_DIM) +
         nx *
             (__gpu_thread_id(__GPU_Y_DIM) + ny * __gpu_thread_id(__GPU_Z_DIM));
}

// Shallow deduplication of recoverable reports, decided once per wave.
bool seen(u64 lane_mask, uptr pc) {
  static uptr table[64] = {};
  bool skip = false;
  if (__gpu_is_first_in_lane(lane_mask)) {
    u32 index = (pc * 0x9E3779B97F4A7C15ull) >> 58;
    skip = __scoped_atomic_exchange_n(&table[index], pc, __ATOMIC_RELAXED,
                                      __MEMORY_SCOPE_DEVICE) == pc;
  }
  return __gpu_read_first_lane_u32(lane_mask, skip);
}

[[gnu::cold, gnu::noinline]] void report(uptr pc, uptr addr, uptr size,
                                         bool is_write, bool fatal) {
  const u64 lane_mask = __gpu_lane_mask();
  if (!fatal && seen(lane_mask, pc))
    return;

  __asan_gpu_report rep = {};
  rep.pc = pc;
  rep.addr = addr;
  rep.size = size;
  rep.block[0] = __gpu_block_id(__GPU_X_DIM);
  rep.block[1] = __gpu_block_id(__GPU_Y_DIM);
  rep.block[2] = __gpu_block_id(__GPU_Z_DIM);
  rep.thread = thread_id();
  rep.is_write = is_write;
  rep.fatal = fatal;

  {
    rpc::Client::Port port = client.open<SANITIZER_OFFLOAD_ASAN>();
    port.send([&](rpc::Buffer* buf, u32) {
      __builtin_memcpy(buf->data, &rep, sizeof(rep));
    });
    // Wait until the host has printed the report, the faulting allocation must
    // still be live and a fatal report must be visible before the trap.
    port.recv([](rpc::Buffer*, u32) {});
  }

  if (fatal)
    __builtin_trap();
}

void check_range(const void* ptr, uptr size, bool is_write, uptr pc) {
  if (!size || __gpu_is_ptr_private(const_cast<void*>(ptr)) ||
      __gpu_is_ptr_local(const_cast<void*>(ptr)))
    return;
  if (uptr bad = region_is_poisoned(reinterpret_cast<uptr>(ptr), size))
    report(pc, bad, size, is_write, /*fatal=*/true);
}

void poison_redzones(const Global& g) {
  if (!is_aligned(g.beg) || !is_aligned(g.size_with_redzone))
    return;
  uptr aligned_size = round_up(g.size, kShadowGranularity);
  fill_shadow(g.beg + aligned_size, g.size_with_redzone - aligned_size,
              kGlobalRedzoneMagic);
  if (g.size != aligned_size)
    *mem_to_shadow(g.beg + round_down(g.size, kShadowGranularity)) =
        static_cast<s8>(g.size % kShadowGranularity);
}

void unpoison_redzones(const Global& g) {
  if (!is_aligned(g.beg) || !is_aligned(g.size_with_redzone))
    return;
  fill_shadow(g.beg, g.size_with_redzone, 0);
}

// Freed chunks stay poisoned here until evicted. Only then do they go back to
// the device allocator, which uninstrumented users share, so they must be
// addressable again by that point.
constexpr u64 kQuarantineSize = 1024;
u64 quarantine[kQuarantineSize] = {};
u64 quarantine_next = 0;

void release_chunk(uptr base) {
  AllocHeader* header = reinterpret_cast<AllocHeader*>(base);
  header->magic = 0;
  fill_shadow(base, header->total, 0);
  __real_free(reinterpret_cast<void*>(base));
}

}  // namespace

//===----------------------------------------------------------------------===//
// Error reporting (emitted by the AddressSanitizer pass)
//===----------------------------------------------------------------------===//

#define ASAN_REPORT_ERROR(type, size, is_write)                            \
  CODEGEN_INTERFACE void __asan_report_##type##size(uptr addr) {           \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/true);         \
  }                                                                        \
  CODEGEN_INTERFACE void __asan_report_##type##size##_noabort(uptr addr) { \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/false);        \
  }                                                                        \
  CODEGEN_INTERFACE void __asan_##type##size(uptr addr) {                  \
    if (is_invalid_access(addr, size))                                     \
      report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/true);       \
  }                                                                        \
  CODEGEN_INTERFACE void __asan_##type##size##_noabort(uptr addr) {        \
    if (is_invalid_access(addr, size))                                     \
      report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/false);      \
  }

ASAN_REPORT_ERROR(load, 1, false)
ASAN_REPORT_ERROR(load, 2, false)
ASAN_REPORT_ERROR(load, 4, false)
ASAN_REPORT_ERROR(load, 8, false)
ASAN_REPORT_ERROR(load, 16, false)
ASAN_REPORT_ERROR(store, 1, true)
ASAN_REPORT_ERROR(store, 2, true)
ASAN_REPORT_ERROR(store, 4, true)
ASAN_REPORT_ERROR(store, 8, true)
ASAN_REPORT_ERROR(store, 16, true)

#define ASAN_REPORT_ERROR_N(type, is_write)                               \
  CODEGEN_INTERFACE void __asan_report_##type##_n(uptr addr, uptr size) { \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/true);        \
  }                                                                       \
  CODEGEN_INTERFACE void __asan_report_##type##_n_noabort(uptr addr,      \
                                                          uptr size) {    \
    report(GET_CALLER_PC(), addr, size, is_write, /*fatal=*/false);       \
  }                                                                       \
  CODEGEN_INTERFACE void __asan_##type##N(uptr addr, uptr size) {         \
    if (uptr bad = region_is_poisoned(addr, size))                        \
      report(GET_CALLER_PC(), bad, size, is_write, /*fatal=*/true);       \
  }                                                                       \
  CODEGEN_INTERFACE void __asan_##type##N_noabort(uptr addr, uptr size) { \
    if (uptr bad = region_is_poisoned(addr, size))                        \
      report(GET_CALLER_PC(), bad, size, is_write, /*fatal=*/false);      \
  }

ASAN_REPORT_ERROR_N(load, false)
ASAN_REPORT_ERROR_N(store, true)

//===----------------------------------------------------------------------===//
// Memory intrinsics
//===----------------------------------------------------------------------===//

INTERFACE void* __asan_memcpy(void* to, const void* from, uptr size) {
  uptr pc = GET_CALLER_PC();
  check_range(from, size, /*is_write=*/false, pc);
  check_range(to, size, /*is_write=*/true, pc);
  return __builtin_memcpy(to, from, size);
}

INTERFACE void* __asan_memmove(void* to, const void* from, uptr size) {
  uptr pc = GET_CALLER_PC();
  check_range(from, size, /*is_write=*/false, pc);
  check_range(to, size, /*is_write=*/true, pc);
  return __builtin_memmove(to, from, size);
}

INTERFACE void* __asan_memset(void* s, int c, uptr n) {
  check_range(s, n, /*is_write=*/true, GET_CALLER_PC());
  return __builtin_memset(s, c, n);
}

//===----------------------------------------------------------------------===//
// Poisoning interface
//===----------------------------------------------------------------------===//

INTERFACE uptr __asan_region_is_poisoned(uptr beg, uptr size) {
  return region_is_poisoned(beg, size);
}

INTERFACE int __asan_address_is_poisoned(const void* addr) {
  return is_address_poisoned(reinterpret_cast<uptr>(addr));
}

INTERFACE void __asan_poison_memory_region(const void* addr, uptr size) {
  if (!size)
    return;
  uptr beg = reinterpret_cast<uptr>(addr);
  uptr end = beg + size;
  __gpu_global s8* beg_sp = mem_to_shadow(beg);
  __gpu_global s8* end_sp = mem_to_shadow(end);
  s8 beg_off = static_cast<s8>(beg & (kShadowGranularity - 1));
  s8 end_off = static_cast<s8>(end & (kShadowGranularity - 1));
  s8 beg_val = *beg_sp;
  s8 end_val = *end_sp;

  if (beg_sp == end_sp) {
    if (beg_val > 0 && beg_val <= end_off)
      *beg_sp = beg_off > 0 ? (beg_val < beg_off ? beg_val : beg_off)
                            : static_cast<s8>(kUserPoisonedMemoryMagic);
    return;
  }
  if (beg_off > 0) {
    *beg_sp = (beg_val == 0 || beg_val > beg_off) ? beg_off : beg_val;
    ++beg_sp;
  }
  for (; beg_sp < end_sp; ++beg_sp)
    *beg_sp = static_cast<s8>(kUserPoisonedMemoryMagic);
  if (end_val > 0 && end_val <= end_off)
    *end_sp = static_cast<s8>(kUserPoisonedMemoryMagic);
}

INTERFACE void __asan_unpoison_memory_region(const void* addr, uptr size) {
  if (!size)
    return;
  uptr beg = reinterpret_cast<uptr>(addr);
  uptr end = beg + size;
  __gpu_global s8* beg_sp = mem_to_shadow(beg);
  __gpu_global s8* end_sp = mem_to_shadow(end);
  s8 end_off = static_cast<s8>(end & (kShadowGranularity - 1));
  s8 beg_val = *beg_sp;
  s8 end_val = *end_sp;

  if (beg_sp == end_sp) {
    if (beg_val != 0)
      *beg_sp = beg_val > end_off ? beg_val : end_off;
    return;
  }
  for (; beg_sp < end_sp; ++beg_sp) *beg_sp = 0;
  if (end_off > 0 && end_val != 0)
    *end_sp = end_val > end_off ? end_val : end_off;
}

// Emitted by the AMDGPU LDS lowering to poison the redzones it inserts.
CODEGEN_INTERFACE void __asan_poison_region(u64 beg, u64 size) {
  if (!is_aligned(beg)) {
    uptr aligned = round_down(beg, kShadowGranularity);
    *mem_to_shadow(aligned) = static_cast<s8>(beg - aligned);
  }
  uptr aligned_beg = round_up(beg, kShadowGranularity);
  uptr aligned_end = round_down(beg + size, kShadowGranularity);
  if (aligned_end > aligned_beg)
    fill_shadow(aligned_beg, aligned_end - aligned_beg,
                static_cast<s8>(kHeapLeftRedzoneMagic));
}

//===----------------------------------------------------------------------===//
// Device heap allocation, used by the AMDGPU LDS lowering and the OpenMP
// device runtime, backed by the device malloc
//===----------------------------------------------------------------------===//

CODEGEN_INTERFACE u64 __asan_malloc_impl(u64 size, u64 pc) {
  uptr user_size = round_up(size ? size : 1, kShadowGranularity);
  uptr total = kAllocHeaderBytes + user_size + kAllocRedzoneBytes;
  uptr base = reinterpret_cast<uptr>(__real_malloc(total));
  if (!base)
    return 0;

  uptr user = base + kAllocHeaderBytes;
  AllocHeader* header = reinterpret_cast<AllocHeader*>(base);
  header->magic = kAllocMagic;
  header->base = base;
  header->pc = pc;
  header->total = static_cast<u32>(total);
  header->size = static_cast<u32>(size);

  fill_shadow(base, kAllocHeaderBytes, static_cast<s8>(kHeapLeftRedzoneMagic));
  fill_shadow(user, round_down(size, kShadowGranularity), 0);
  if (size % kShadowGranularity)
    *mem_to_shadow(user + round_down(size, kShadowGranularity)) =
        static_cast<s8>(size % kShadowGranularity);
  fill_shadow(user + user_size, kAllocRedzoneBytes,
              static_cast<s8>(kHeapLeftRedzoneMagic));
  return user;
}

CODEGEN_INTERFACE void __asan_free_impl(u64 ptr, u64 pc) {
  if (!ptr)
    return;
  AllocHeader* header = reinterpret_cast<AllocHeader*>(ptr - kAllocHeaderBytes);
  s8 shadow_value = *mem_to_shadow(ptr);
  if (header->magic != kAllocMagic ||
      shadow_value == static_cast<s8>(kHeapFreeMagic)) {
    report(pc, ptr, 1, /*is_write=*/true, /*fatal=*/true);
    return;
  }
  // The header is kept so the host report can still describe the allocation.
  fill_shadow(ptr, header->total - kAllocHeaderBytes,
              static_cast<s8>(kHeapFreeMagic));

  u64 slot = __scoped_atomic_fetch_add(&quarantine_next, 1, __ATOMIC_RELAXED,
                                       __MEMORY_SCOPE_DEVICE) %
             kQuarantineSize;
  if (u64 evicted =
          __scoped_atomic_exchange_n(&quarantine[slot], header->base,
                                     __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE))
    release_chunk(evicted);
}

INTERFACE [[gnu::noinline]] void* __wrap_malloc(uptr size) {
  return reinterpret_cast<void*>(__asan_malloc_impl(size, GET_CALLER_PC()));
}

INTERFACE [[gnu::noinline]] void __wrap_free(void* ptr) {
  __asan_free_impl(reinterpret_cast<u64>(ptr), GET_CALLER_PC());
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

INTERFACE void __asan_register_globals(uptr globals, uptr n) {
  const Global* g = reinterpret_cast<const Global*>(globals);
  for (uptr i = 0; i < n; ++i) poison_redzones(g[i]);
}

INTERFACE void __asan_unregister_globals(uptr globals, uptr n) {
  const Global* g = reinterpret_cast<const Global*>(globals);
  for (uptr i = 0; i < n; ++i) unpoison_redzones(g[i]);
}

INTERFACE void __asan_register_elf_globals(uptr flag, uptr start, uptr stop) {
  uptr* registered = reinterpret_cast<uptr*>(flag);
  if (!start || *registered)
    return;
  __asan_register_globals(start, (stop - start) / sizeof(Global));
  *registered = 1;
}

INTERFACE void __asan_unregister_elf_globals(uptr flag, uptr start, uptr stop) {
  uptr* registered = reinterpret_cast<uptr*>(flag);
  if (!start || !*registered)
    return;
  __asan_unregister_globals(start, (stop - start) / sizeof(Global));
  *registered = 0;
}

//===----------------------------------------------------------------------===//
// C++ array cookies
//===----------------------------------------------------------------------===//

INTERFACE void __asan_poison_cxx_array_cookie(uptr p) {
  *mem_to_shadow(p) = static_cast<s8>(kArrayCookieMagic);
}

INTERFACE uptr __asan_load_cxx_array_cookie(uptr* p) {
  uptr addr = reinterpret_cast<uptr>(p);
  s8 shadow_value = *mem_to_shadow(addr);
  if (shadow_value == static_cast<s8>(kHeapFreeMagic)) {
    report(GET_CALLER_PC(), addr, 1, /*is_write=*/false, /*fatal=*/true);
    return 0;
  }
  return *p;
}

//===----------------------------------------------------------------------===//
// Unsupported on the device
//===----------------------------------------------------------------------===//

INTERFACE void __asan_init() {}
INTERFACE void __asan_version_mismatch_check_v8() {}
INTERFACE void __asan_handle_no_return() {}
INTERFACE void __asan_before_dynamic_init(uptr) {}
INTERFACE void __asan_after_dynamic_init() {}
INTERFACE void __asan_register_image_globals(uptr) {}
INTERFACE void __asan_unregister_image_globals(uptr) {}
INTERFACE void __sanitizer_ptr_cmp(uptr, uptr) {}
INTERFACE void __sanitizer_ptr_sub(uptr, uptr) {}
