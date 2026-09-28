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

#include "sim/coralnpu_m4_instructions.h"

#include <algorithm>
#include <cstdint>

#include "sim/coralnpu_m4_state.h"
#include "sim/coralnpu_v2_state.h"
#include "absl/log/check.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/riscv_zvt_instructions.h"
#include "riscv/riscv_zvt_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::operator*;  // NOLINT

namespace {

int TileSubsetCount(::mpact::sim::riscv::RiscVZvtMatrixState* matrix_state,
                    ::mpact::sim::riscv::RiscVVectorState* vector_state,
                    int tile_element_width) {
  const int kEffectiveTileElements = tile_element_width < 64
                                         ? matrix_state->tile_elements()
                                         : matrix_state->tile_elements() / 2;
  return std::min(vector_state->vector_length(), kEffectiveTileElements);
}

}  // namespace

void CoralNPUM4Vtle(int eew_bits,
                    const ::mpact::sim::generic::Instruction* inst) {
  auto m_result = ::mpact::sim::riscv::GetMatrixState(inst);
  CHECK(m_result.ok()) << m_result.status();
  auto* matrix_state = *m_result;
  auto* state = static_cast<::coralnpu::sim::CoralNPUM4State*>(
      matrix_state->riscv_state());
  auto* vector_state = state->rv_vector();
  const uint64_t kBase =
      ::mpact::sim::generic::GetInstructionSource<uint64_t>(inst, 0);
  const int kElementSize = eew_bits / 8;
  const int kCount = TileSubsetCount(matrix_state, vector_state, eew_bits);
  if (!state->HasPermission(kBase, kCount * kElementSize,
                            ::coralnpu::sim::MemoryPermission::kRead)) {
    state->Trap(/*is_interrupt=*/false, kBase,
                *::mpact::sim::riscv::ExceptionCode::kLoadAccessFault,
                inst->address(), inst);
    return;
  }

  ::mpact::sim::riscv::RiscVZvtVtle(eew_bits, inst);
}

void CoralNPUM4Vtse(int eew_bits,
                    const ::mpact::sim::generic::Instruction* inst) {
  auto m_result = ::mpact::sim::riscv::GetMatrixState(inst);
  CHECK(m_result.ok()) << m_result.status();
  auto* matrix_state = *m_result;
  auto* state = static_cast<::coralnpu::sim::CoralNPUM4State*>(
      matrix_state->riscv_state());
  auto* vector_state = state->rv_vector();
  const uint64_t kBase =
      ::mpact::sim::generic::GetInstructionSource<uint64_t>(inst, 0);
  const int kElementSize = eew_bits / 8;
  const int kCount = TileSubsetCount(matrix_state, vector_state, eew_bits);
  if (!state->HasPermission(kBase, kCount * kElementSize,
                            ::coralnpu::sim::MemoryPermission::kWrite)) {
    state->Trap(/*is_interrupt=*/false, kBase,
                *::mpact::sim::riscv::ExceptionCode::kStoreAccessFault,
                inst->address(), inst);
    return;
  }

  ::mpact::sim::riscv::RiscVZvtVtse(eew_bits, inst);
}

}  // namespace mpact::sim::riscv
