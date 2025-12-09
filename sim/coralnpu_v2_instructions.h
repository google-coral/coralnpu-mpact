// Copyright 2025 Google LLC
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

#ifndef SIM_CORALNPU_V2_INSTRUCTIONS_H_
#define SIM_CORALNPU_V2_INSTRUCTIONS_H_

#include "absl/base/nullability.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::ExceptionCode;

// Semantic function for the mpause instruction. This instruction is used to
// halt execution.
// No source operands.
// No destination operands.
void CoralNPUV2Mpause(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the load word (lw) instruction. If the address access
// is valid, the data is loaded into the destination register. Otherwise, a
// trap is triggered.
// Source operand 1: scalar register containing the base address to load.
// Source operand 2: immediate operand containing the offset to load.
// Destination operand 1: scalar register to store the loaded data.
void CoralNPUV2Lw(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the load half word (lh) instruction. If the address
// access is valid, the data is loaded into the destination register. Otherwise,
// a trap is triggered.
// Source operand 1: scalar register containing the base address to load.
// Source operand 2: immediate operand containing the offset to load.
// Destination operand 1: scalar register to store the sign extended data.
void CoralNPUV2Lh(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the load unsigned half word (lhu) instruction. If the
// address access is valid, the data is loaded into the destination register.
// Otherwise, a trap is triggered.
// Source operand 1: scalar register containing the base address to load.
// Source operand 2: immediate operand containing the offset to load.
// Destination operand 1: scalar register to store the unsigned data.
void CoralNPUV2Lhu(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the load byte (lb) instruction. If the address access
// is valid, the data is loaded into the destination register. Otherwise, a
// trap is triggered.
// Source operand 1: scalar register containing the base address to load.
// Source operand 2: immediate operand containing the offset to load.
// Destination operand 1: scalar register to store the sign extended data.
void CoralNPUV2Lb(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the load unsigned byte (lbu) instruction. If the
// address access is valid, the data is loaded into the destination register.
// Otherwise, a trap is triggered.
// Source operand 1: scalar register containing the base address to load.
// Source operand 2: immediate operand containing the offset to load.
// Destination operand 1: scalar register to store the unsigned data.
void CoralNPUV2Lbu(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the store word (sw) instruction. If the address access
// is valid, the data is stored to the specified address. Otherwise, a trap is
// triggered.
// Source operand 1: scalar register containing the base address to store.
// Source operand 2: immediate operand containing the offset to store.
// Source operand 3: scalar register containing the data to store.
void CoralNPUV2Sw(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the store half word (sh) instruction. If the address
// access is valid, the data is stored to the specified address. Otherwise, a
// trap is triggered.
// Source operand 1: scalar register containing the base address to store.
// Source operand 2: immediate operand containing the offset to store.
// Source operand 3: scalar register containing the data to store. The lower
// 16 bits of this register are used to store.
void CoralNPUV2Sh(const Instruction* /*absl_nonnull*/ instruction);

// Semantic function for the store byte (sb) instruction. If the address access
// is valid, the data is stored to the specified address. Otherwise, a trap is
// triggered.
// Source operand 1: scalar register containing the base address to store.
// Source operand 2: immediate operand containing the offset to store.
// Source operand 3: scalar register containing the data to store. The lower 8
// bits of this register are used to store.
void CoralNPUV2Sb(const Instruction* /*absl_nonnull*/ instruction);

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_V2_INSTRUCTIONS_H_
