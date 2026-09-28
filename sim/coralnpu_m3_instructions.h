// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Declarations of instruction semantic functions for the CoralNPU M3 ISA.
// Instruction semantic functions specific to the CoralNPU M3 architecture are
// declared here.

#ifndef SIM_CORALNPU_M3_INSTRUCTIONS_H_
#define SIM_CORALNPU_M3_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::Instruction;

// Semantic function for the scalar convert float to bfloat16 (fcvt.bf16.s)
// instruction. Converts single-precision float in fs1 to bfloat16 format
// in fd, taking rounding mode from rm source or fcsr.
// Source operand 0: floating point register containing single-precision input.
// Source operand 1: rounding mode (rm).
// Destination operand 0: floating point destination register.
// Destination operand 1: fflags/fcsr destination register.
void CoralNPUFcvtBf16S(Instruction* inst);

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_M3_INSTRUCTIONS_H_
