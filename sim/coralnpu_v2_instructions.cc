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

#include "sim/coralnpu_v2_instructions.h"

#include <cstdint>
#include <type_traits>

#include "sim/coralnpu_v2_state.h"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "riscv/riscv_f_instructions.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"

namespace {
using ::coralnpu::sim::CoralNPUV2State;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

template <typename Register, typename ValueType,
          ExceptionCode fault_exception_code>
bool AccessCheck(const Instruction* /*absl_nonnull*/ instruction) {
  using RegVal = typename Register::ValueType;
  using URegVal = typename std::make_unsigned<RegVal>::type;
  URegVal base = ::mpact::sim::generic::GetInstructionSource<URegVal>(
      instruction, /*index=*/0);
  RegVal offset = ::mpact::sim::generic::GetInstructionSource<RegVal>(
      instruction, /*index=*/1);
  URegVal address = base + offset;
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  if (!state) {
    LOG(ERROR) << "AccessCheck: state is nullptr.";
    return false;
  }
  if (!state->IsLsuAccessValid(address, sizeof(ValueType))) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0, *fault_exception_code,
                /*epc=*/instruction->address(), instruction);
    return false;
  }
  return true;
}
}  // namespace

namespace coralnpu::sim {

using ::coralnpu::sim::CoralNPUV2State;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVFpRegister;

void CoralNPUV2Mpause(const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  if (state) {
    state->MPause(instruction);
  }
}

void CoralNPUV2Lw(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed =
      AccessCheck<RV32Register, int32_t, ExceptionCode::kLoadAccessFault>(
          instruction);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, int32_t>(instruction);
  }
}

void CoralNPUV2Lh(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed =
      AccessCheck<RV32Register, int16_t, ExceptionCode::kLoadAccessFault>(
          instruction);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, int16_t>(instruction);
  }
}

void CoralNPUV2Lhu(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed =
      AccessCheck<RV32Register, uint16_t, ExceptionCode::kLoadAccessFault>(
          instruction);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, uint16_t>(instruction);
  }
}

void CoralNPUV2Lb(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed =
      AccessCheck<RV32Register, int8_t, ExceptionCode::kLoadAccessFault>(
          instruction);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, int8_t>(instruction);
  }
}

void CoralNPUV2Lbu(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed =
      AccessCheck<RV32Register, uint8_t, ExceptionCode::kLoadAccessFault>(
          instruction);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, uint8_t>(instruction);
  }
}

void CoralNPUV2Sw(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed =
      AccessCheck<RV32Register, uint32_t, ExceptionCode::kStoreAccessFault>(
          instruction);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RVStore<RV32Register, uint32_t>(instruction);
  }
}

void CoralNPUV2Sh(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed =
      AccessCheck<RV32Register, uint16_t, ExceptionCode::kStoreAccessFault>(
          instruction);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RVStore<RV32Register, uint16_t>(instruction);
  }
}

void CoralNPUV2Sb(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed =
      AccessCheck<RV32Register, uint8_t, ExceptionCode::kStoreAccessFault>(
          instruction);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RVStore<RV32Register, uint8_t>(instruction);
  }
}

void CoralNPUV2Fsw(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed =
      AccessCheck<RVFpRegister, uint64_t, ExceptionCode::kStoreAccessFault>(
          instruction);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RV32::RiscVFSw(instruction);
  }
}

}  // namespace coralnpu::sim
