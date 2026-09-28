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

// This file declares the simulation semantic functions for the
// CoralNPU M4-specific custom instructions. These instructions interact
// with the Zvt matrix extension state, including memory-to-tile load
// and store operations with statically encoded element widths.

#ifndef SIM_CORALNPU_M4_INSTRUCTIONS_H_
#define SIM_CORALNPU_M4_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact::sim::riscv {

// Loads a tile row or column subset from memory.
// The effective element width (EEW) in bits is specified by `eew_bits`.
void CoralNPUM4Vtle(int eew_bits,
                    const ::mpact::sim::generic::Instruction* inst);

// Stores a tile row or column subset to memory.
// The effective element width (EEW) in bits is specified by `eew_bits`.
void CoralNPUM4Vtse(int eew_bits,
                    const ::mpact::sim::generic::Instruction* inst);

}  // namespace mpact::sim::riscv

#endif  // SIM_CORALNPU_M4_INSTRUCTIONS_H_
