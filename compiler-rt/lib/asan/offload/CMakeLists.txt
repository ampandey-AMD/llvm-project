include(FindLibcCommonUtils)
if(NOT OS_NAME MATCHES "Linux" OR NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(ALL_ASAN_OFFLOAD_SUPPORTED_ARCH ${X86_64} ${ARM64} ${PPC64})
list_intersect(ASAN_OFFLOAD_SUPPORTED_ARCH
  ASAN_SUPPORTED_ARCH
  ALL_ASAN_OFFLOAD_SUPPORTED_ARCH)

set(ASAN_OFFLOAD_SOURCES
  asan_offload_hsa_interceptors.cpp
  asan_offload_report.cpp
  )

set(ASAN_OFFLOAD_HEADERS
  asan_offload.h
  ../asan_offload_packet.h
  )

include_directories(..)

add_compiler_rt_object_libraries(RTAsan_offload
  ARCHS ${ASAN_OFFLOAD_SUPPORTED_ARCH}
  SOURCES ${ASAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${ASAN_OFFLOAD_HEADERS}
  CFLAGS ${ASAN_CFLAGS}
  DEFS ${ASAN_COMMON_DEFINITIONS})
foreach(arch ${ASAN_OFFLOAD_SUPPORTED_ARCH})
  target_link_libraries(RTAsan_offload.${arch} PRIVATE
                        llvm-libc-common-utilities)
endforeach()

# The shared runtime embeds these objects directly, the static archive is only
# for static runtimes and is pulled in with -u.
add_compiler_rt_runtime(clang_rt.asan_offload
  STATIC
  ARCHS ${ASAN_OFFLOAD_SUPPORTED_ARCH}
  OBJECT_LIBS RTAsan_offload
              RTSanitizerOffload
  CFLAGS ${ASAN_CFLAGS}
  DEFS ${ASAN_COMMON_DEFINITIONS}
  PARENT_TARGET asan)
