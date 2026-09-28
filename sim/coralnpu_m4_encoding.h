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

// This file defines the CoralNPUM4Encoding class, which implements the
// instruction parser for the CoralNPU M4 architecture. It extends the
// V2 encoding with M4-specific operand getters.

#ifndef SIM_CORALNPU_M4_ENCODING_H_
#define SIM_CORALNPU_M4_ENCODING_H_

#include <cstdint>

#include "sim/coralnpu_m4_bin_decoder.h"
#include "sim/coralnpu_m4_decoder.h"
#include "sim/coralnpu_m4_enums.h"
#include "sim/coralnpu_m4_state.h"
#include "sim/coralnpu_v2_encoding_template.h"
#include "riscv/riscv_zvt_getters.h"

namespace coralnpu::sim {

// The encoding parser for the CoralNPU M4 architecture. This class parses
// 32-bit instruction words into their constituent fields and identifies the
// opcode, configuring the M4-specific source and destination operand getters
// (such as the matrix tile and vector group operands for the Zvt extension).
class CoralNPUM4Encoding
    : public CoralNPUV2EncodingTemplate<
          ::coralnpu::sim::isa32_m4::CoralNPUM4EncodingBase,
          ::coralnpu::sim::isa32_m4::OpcodeEnum,
          ::coralnpu::sim::isa32_m4::SlotEnum,
          ::coralnpu::sim::isa32_m4::SourceOpEnum,
          ::coralnpu::sim::isa32_m4::DestOpEnum,
          ::coralnpu::sim::encoding_m4::Extractors> {
 public:
  using Base = CoralNPUV2EncodingTemplate<
      ::coralnpu::sim::isa32_m4::CoralNPUM4EncodingBase,
      ::coralnpu::sim::isa32_m4::OpcodeEnum,
      ::coralnpu::sim::isa32_m4::SlotEnum,
      ::coralnpu::sim::isa32_m4::SourceOpEnum,
      ::coralnpu::sim::isa32_m4::DestOpEnum,
      ::coralnpu::sim::encoding_m4::Extractors>;

  explicit CoralNPUM4Encoding(CoralNPUM4State* state) : Base(state) {
    // Add the M4-specific operand getters on top of the inherited V2 getters.
    ::mpact::sim::riscv::AddRiscVZvtSourceGetters<
        ::coralnpu::sim::isa32_m4::SourceOpEnum,
        ::coralnpu::sim::encoding_m4::Extractors>(this->source_op_getters_,
                                                  this);
    ::mpact::sim::riscv::AddRiscVZvtDestGetters<
        ::coralnpu::sim::isa32_m4::DestOpEnum,
        ::coralnpu::sim::encoding_m4::Extractors>(this->dest_op_getters_, this);
  }

  // Parses an instruction and determines the opcode.
  void ParseInstruction(uint32_t inst_word);
};

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_M4_ENCODING_H_
