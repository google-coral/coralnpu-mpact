// Copyright 2025 Google LLC
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

#include "sim/coralnpu_v2_encoding.h"

#include <cstdint>

#include "sim/coralnpu_v2_bin_decoder.h"
#include "sim/coralnpu_v2_getters.h"
#include "sim/coralnpu_v2_state.h"
#include "absl/base/nullability.h"
#include "riscv/riscv_encoding_common.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"

namespace coralnpu::sim {

using ::coralnpu::sim::CoralNPUV2State;
using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::operator*;  // NOLINT
using ::mpact::sim::generic::SourceOperandInterface;
using Extractors = ::coralnpu::sim::encoding::Extractors;

CoralNPUV2Encoding::CoralNPUV2Encoding(CoralNPUV2State* /*absl_nonnull*/ state)
    : RiscVEncodingCommon(), state_(state) {
  source_op_getters_.emplace(
      *SourceOpEnum::kNone,
      []() -> SourceOperandInterface* { return nullptr; });
  dest_op_getters_.emplace(
      *DestOpEnum::kNone,
      [](int latency) -> DestinationOperandInterface* { return nullptr; });
  // Add CoralNPU V2 ISA source operand getters.
  AddCoralNPUV2SourceGetters<SourceOpEnum, Extractors>(source_op_getters_,
                                                       this);
  // Add CoralNPU V2 ISA destination operand getters.
  AddCoralNPUV2DestGetters<DestOpEnum, Extractors>(dest_op_getters_, this);
}

SourceOperandInterface* CoralNPUV2Encoding::GetSource(SlotEnum, int, OpcodeEnum,
                                                      SourceOpEnum op, int) {
  auto const& iter = source_op_getters_.find(*op);
  if (iter == source_op_getters_.end()) return nullptr;
  return iter->second();
}

DestinationOperandInterface* CoralNPUV2Encoding::GetDestination(
    SlotEnum, int, OpcodeEnum, DestOpEnum op, int, int latency) {
  auto const& iter = dest_op_getters_.find(*op);
  if (iter == dest_op_getters_.end()) return nullptr;
  return iter->second(latency);
}

void CoralNPUV2Encoding::ParseInstruction(uint32_t inst_word) {
  inst_word_ = inst_word;
  opcode_ = ::coralnpu::sim::encoding::DecodeCoralNPUV2Inst32(inst_word_);
}

}  // namespace coralnpu::sim
