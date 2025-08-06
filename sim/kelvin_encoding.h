/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SIM_KELVIN_ENCODING_H_
#define SIM_KELVIN_ENCODING_H_

#include <cstdint>
#include <string>

#include "sim/kelvin_decoder.h"
#include "sim/kelvin_enums.h"
#include "sim/kelvin_state.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "mpact/sim/generic/simple_resource.h"

namespace kelvin::sim::isa32 {

// This class provides the interface between the generated instruction decoder
// framework (which is agnostic of the actual bit representation of
// instructions) and the instruction representation. This class provides methods
// to return the opcode, source operands, and destination operands for
// instructions according to the operand fields in the encoding.
class KelvinEncoding : public KelvinEncodingBase {
 public:
  explicit KelvinEncoding(KelvinState* state);
  ~KelvinEncoding() override;

  // Parses an instruction and determines the opcode.
  void ParseInstruction(uint32_t inst_word);

  // Returns the opcode in the current instruction representation.
  OpcodeEnum GetOpcode(SlotEnum, int) override { return opcode_; }

  // There is no predicate, so return nullptr.
  PredicateOperandInterface* GetPredicate(SlotEnum, int, OpcodeEnum,
                                          PredOpEnum) override {
    return nullptr;
  }

  // Return the resource operand corresponding to the resource enum. If argument
  // is not kNone, it means that the resource enum is a pool of resources and
  // the resource element from the pool is specified by the
  // ResourceArgumentEnum. This is used for instance for register resources,
  // where the resource itself is a register bank, and the argument specifies
  // which register (or more precisely) which encoding "field" specifies the
  // register number.
  ResourceOperandInterface* GetSimpleResourceOperand(
      SlotEnum, int, OpcodeEnum, SimpleResourceVector& resource_vec,
      int end) override {
    return nullptr;
  }
  ResourceOperandInterface* GetComplexResourceOperand(
      SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
      int end) override {
    return nullptr;
  }

  // The following method returns a source operand that corresponds to the
  // particular operand field.
  SourceOperandInterface* GetSource(SlotEnum, int, OpcodeEnum, SourceOpEnum op,
                                    int source_no) override;

  // The following method returns a destination operand that corresponds to the
  // particular operand field.
  DestinationOperandInterface* GetDestination(SlotEnum, int, OpcodeEnum,
                                              DestOpEnum op, int,
                                              int latency) override;

  // This method returns latency for any destination operand for which the
  // latency specifier in the .isa file is '*'. Since there are none, just
  // return 0.
  int GetLatency(SlotEnum, int, OpcodeEnum, DestOpEnum, int) override {
    return 0;
  }

  // Getter.
  mpact::sim::generic::SimpleResourcePool* resource_pool() const {
    return resource_pool_;
  }

 protected:
  using SourceOpGetterMap =
      absl::flat_hash_map<int, absl::AnyInvocable<SourceOperandInterface*()>>;
  using DestOpGetterMap = absl::flat_hash_map<
      int, absl::AnyInvocable<DestinationOperandInterface*(int)>>;

  SourceOpGetterMap& source_op_getters() { return source_op_getters_; }
  DestOpGetterMap& dest_op_getters() { return dest_op_getters_; }

  KelvinState* state() const { return state_; }
  OpcodeEnum opcode() const { return opcode_; }
  uint32_t inst_word() const { return inst_word_; }

 private:
  std::string GetSimpleResourceName(SimpleResourceEnum resource_enum);
  // These methods initialize the source and destination operand getter
  // arrays, and the complex resource getter array.
  void InitializeSourceOperandGetters();
  void InitializeDestinationOperandGetters();
  bool IsWidenDestinationRegisterOp() const;
  int GetSrc1WidenFactor() const;

  SourceOpGetterMap source_op_getters_;
  DestOpGetterMap dest_op_getters_;
  KelvinState* state_;
  uint32_t inst_word_;
  OpcodeEnum opcode_;
  mpact::sim::generic::SimpleResourcePool* resource_pool_ = nullptr;
};

}  // namespace kelvin::sim::isa32

#endif  // SIM_KELVIN_ENCODING_H_
