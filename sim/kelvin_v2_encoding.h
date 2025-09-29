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

#ifndef SIM_KELVIN_V2_ENCODING_H_
#define SIM_KELVIN_V2_ENCODING_H_

#include <cstdint>

#include "sim/kelvin_v2_decoder.h"
#include "sim/kelvin_v2_enums.h"
#include "sim/kelvin_v2_state.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "riscv/riscv_encoding_common.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/simple_resource.h"

namespace kelvin::sim {

class KelvinV2Encoding : public ::kelvin::sim::isa32_v2::KelvinV2EncodingBase,
                         public ::mpact::sim::riscv::RiscVEncodingCommon {
 public:
  using DestinationOperandInterface =
      ::mpact::sim::generic::DestinationOperandInterface;
  using SimpleResourcePool = ::mpact::sim::generic::SimpleResourcePool;
  using SourceOperandInterface = ::mpact::sim::generic::SourceOperandInterface;
  using KelvinV2State = ::kelvin::sim::KelvinV2State;

  using SourceOpGetterMap =
      absl::flat_hash_map<int, absl::AnyInvocable<SourceOperandInterface*()>>;
  using DestOpGetterMap = absl::flat_hash_map<
      int, absl::AnyInvocable<DestinationOperandInterface*(int)>>;

  using OpcodeEnum = ::kelvin::sim::isa32_v2::OpcodeEnum;
  using SlotEnum = ::kelvin::sim::isa32_v2::SlotEnum;
  using SourceOpEnum = ::kelvin::sim::isa32_v2::SourceOpEnum;
  using DestOpEnum = ::kelvin::sim::isa32_v2::DestOpEnum;

  explicit KelvinV2Encoding(KelvinV2State* /*absl_nonnull*/ state);

  // Based on KelvinV2EncodingBase
  OpcodeEnum GetOpcode(SlotEnum, int) override { return opcode_; }

  // The following method returns a source operand that corresponds to the
  // particular operand field.
  SourceOperandInterface* GetSource(SlotEnum, int, OpcodeEnum, SourceOpEnum op,
                                    int source_no) override;

  // The following method returns a destination operand that corresponds to the
  // particular operand field.
  DestinationOperandInterface* GetDestination(SlotEnum, int, OpcodeEnum,
                                              DestOpEnum op, int dest_no,
                                              int latency) override;

  // This method returns latency for any destination operand for which the
  // latency specifier in the .isa file is '*'. Since there are none, just
  // return 0.
  int GetLatency(SlotEnum, int, OpcodeEnum, DestOpEnum, int) override {
    return 0;
  }

  // Based on RiscVEncodingCommon
  KelvinV2State* state() const override { return state_; }

  SimpleResourcePool* resource_pool() override { return nullptr; }

  uint32_t inst_word() const override { return inst_word_; }

  // Parses an instruction and determines the opcode.
  void ParseInstruction(uint32_t inst_word);

  const SourceOpGetterMap& source_op_getters() { return source_op_getters_; }
  const DestOpGetterMap& dest_op_getters() { return dest_op_getters_; }

 private:
  uint32_t inst_word_;
  OpcodeEnum opcode_;
  KelvinV2State* state_;
  SourceOpGetterMap source_op_getters_;
  DestOpGetterMap dest_op_getters_;
};

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_V2_ENCODING_H_
