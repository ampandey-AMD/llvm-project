; RUN: opt < %s -passes=amdgpu-sw-lower-lds -S -mtriple=amdgcn-amd-amdhsa | FileCheck %s

; The kernel reaches @leaf only through an indirect call to @wrapper. After
; lowering, @leaf reads the kernel id to find the LDS replacement, so
; amdgpu-no-lds-kernel-id must be removed from the callees of indirect call
; targets too, not just from the targets themselves.

@lds = internal addrspace(3) global i32 poison, align 4
@fn_table = addrspace(1) global ptr @wrapper, align 8

; CHECK-LABEL: define internal void @leaf(
; CHECK: call i32 @llvm.amdgcn.lds.kernel.id()
define internal void @leaf() #0 {
  store i32 1, ptr addrspace(3) @lds, align 4
  ret void
}

; CHECK-LABEL: define internal void @wrapper(
define internal void @wrapper() #0 {
  call void @leaf()
  ret void
}

define amdgpu_kernel void @k0(ptr %fp) sanitize_address {
  call void %fp()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"nosanitize_address", i32 1}

attributes #0 = { sanitize_address "amdgpu-no-lds-kernel-id" }

; CHECK-NOT: "amdgpu-no-lds-kernel-id"
