// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// End-to-end exercise of the RISC-V Vector Matrix Extension (M4 / VME) int8
// matrix multiply. The M4 instructions are not known to any assembler, so each
// is emitted as a raw 32-bit `.word`. The encodings here are the same ones
// validated by sim/test/coralnpu_m4_ops_test.cc (see the MakeVConfig /
// MakeVMatrix / MakeVtMem helpers there) and must match
// sim/coralnpu_m4.bin_fmt.
//
// Build (needs a riscv32 gcc/clang with the vector extension), then objdump to
// the .disassm the test harness consumes:
//   clang --target=riscv32 -march=rv32imv -nostdlib -O0 \
//       -Ttext=0x1000 coralnpu_m4_i8_matmul.c -o coralnpu_m4_i8_matmul.elf
//   llvm-objdump -d coralnpu_m4_i8_matmul.elf > coralnpu_m4_i8_matmul.disassm
//
// Computes C = A^T * B for 2x2 int8 matrices, K=2:
//   A = [[1,2],[3,4]] (row-major), B = [[5,6],[7,8]]
//   C = [[26,30],[38,44]] (int32), written to OUT_ADDR.

#include <stdint.h>

#define VME_WORD(w) __asm__ volatile(".word %0" ::"i"(w))

#define OUT_ADDR 0x10000u

// Inputs materialized in registers/memory by ordinary instructions (the
// header-load test path only writes .text, not .data).
static inline void store_i8x2(volatile int8_t* p, int8_t a, int8_t b) {
  p[0] = a;
  p[1] = b;
}

void _start(void) {
  // Scratch buffers for the A/B rows (on a fixed data address range).
  volatile int8_t* a0 = (volatile int8_t*)0x10100;  // A row 0.
  volatile int8_t* a1 = (volatile int8_t*)0x10110;  // A row 1.
  volatile int8_t* b0 = (volatile int8_t*)0x10120;  // B row 0.
  volatile int8_t* b1 = (volatile int8_t*)0x10130;  // B row 1.
  store_i8x2(a0, 1, 2);
  store_i8x2(a1, 3, 4);
  store_i8x2(b0, 5, 6);
  store_i8x2(b1, 7, 8);

  // 1) Configure: mtype = mtwiden=3 (TWIDEN=4), tk=2, tm=2; vtype = SEW=8.
  //    a3 = mtype (0x843), a4 = vtype (0). msetmtype a3, a4.
  __asm__ volatile("li a3, 0x843");
  __asm__ volatile("li a4, 0");
  VME_WORD(0x82E6F057);  // msetmtype rs1=a3(13), rs2=a4(14).
  //    tn = vl = 2 via msettn (a5 = 2).
  __asm__ volatile("li a5, 2");
  VME_WORD(0x8407F057);  // msettn rd=x0, rs1=a5(15).

  // 2) Load A rows -> v8, v10 and B rows -> v16, v18 (stride 8/KMAX = 2 regs).
  __asm__ volatile("vle8.v v8,  (%0)" ::"r"(a0));
  __asm__ volatile("vle8.v v10, (%0)" ::"r"(a1));
  __asm__ volatile("vle8.v v16, (%0)" ::"r"(b0));
  __asm__ volatile("vle8.v v18, (%0)" ::"r"(b1));

  // 3) C += A^T * B into tile mt0 (the tile starts zeroed).
  VME_WORD(0xF28800F7);  // vtmms.tvv mt0, v8, v16.

  // 4) Store the two result rows to OUT_ADDR via vtse32 (TSS row index in a6).
  __asm__ volatile("li a2, %0" ::"i"(OUT_ADDR));
  __asm__ volatile("li a6, 0");  // TSS: tile 0, pattern row, index 0.
  VME_WORD(0x53067027);          // vtse32 a6, (a2).
  __asm__ volatile("addi a2, a2, 8");
  __asm__ volatile("li a6, 1");  // TSS: row 1.
  VME_WORD(0x53067027);          // vtse32 a6, (a2).

  __asm__ volatile("ebreak");
  for (;;) {
  }
}
