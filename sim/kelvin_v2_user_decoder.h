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

#ifndef SIM_KELVIN_V2_USER_DECODER_H_
#define SIM_KELVIN_V2_USER_DECODER_H_

#include <cstdint>
#include <memory>

#include "sim/kelvin_v2_decoder.h"
#include "sim/kelvin_v2_encoding.h"
#include "sim/kelvin_v2_state.h"
#include "absl/base/nullability.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace kelvin::sim {

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the riscv architecture
// only has a single slot, it's a pretty simple class.
class KelvinV2IsaFactory
    : public ::kelvin::sim::isa32_v2::KelvinV2InstructionSetFactory {
  using ArchState = ::mpact::sim::generic::ArchState;
  using KelvinV2Slot = ::kelvin::sim::isa32_v2::KelvinV2Slot;

 public:
  std::unique_ptr<KelvinV2Slot> CreateKelvinV2Slot(ArchState* state) override {
    return std::make_unique<KelvinV2Slot>(state);
  }
};

class KelvinV2UserDecoder : public ::mpact::sim::generic::DecoderInterface {
 public:
  using DataBuffer = ::mpact::sim::generic::DataBuffer;
  using Instruction = ::mpact::sim::generic::Instruction;
  using KelvinV2Encoding = ::kelvin::sim::KelvinV2Encoding;
  using KelvinV2InstructionSet =
      ::kelvin::sim::isa32_v2::KelvinV2InstructionSet;
  using MemoryInterface = ::mpact::sim::util::MemoryInterface;
  using ProgramError = ::mpact::sim::generic::ProgramError;

  KelvinV2UserDecoder(KelvinV2State* /*absl_nonnull*/ state,
                      MemoryInterface* /*absl_nonnull*/ memory);
  ~KelvinV2UserDecoder() override;

  // Decodes an instruction at the given address.
  Instruction* DecodeInstruction(uint64_t address) override;

  // Returns the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override;

  // Returns the name of the opcode at the given index.
  const char* GetOpcodeName(int index) const override;

 private:
  KelvinV2State* state_;
  MemoryInterface* memory_;
  DataBuffer* inst_db_;
  std::unique_ptr<ProgramError> decode_error_;
  std::unique_ptr<KelvinV2Encoding> kelvin_v2_encoding_;
  std::unique_ptr<KelvinV2IsaFactory> kelvin_v2_isa_factory_;
  std::unique_ptr<KelvinV2InstructionSet> kelvin_v2_isa_;
};

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_V2_USER_DECODER_H_
