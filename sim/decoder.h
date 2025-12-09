// Copyright 2023 Google LLC
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

#ifndef SIM_DECODER_H_
#define SIM_DECODER_H_

#include <cstdint>
#include <memory>

#include "sim/coralnpu_decoder.h"
#include "sim/coralnpu_encoding.h"
#include "sim/coralnpu_enums.h"
#include "sim/coralnpu_state.h"
#include "riscv/riscv_generic_decoder.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the RISC-V
// architecture only has a single slot, it's a pretty simple class.
class CoralNPUIsaFactory : public isa32::CoralNPUInstructionSetFactory {
 public:
  std::unique_ptr<isa32::CoralnpuSlot> CreateCoralnpuSlot(
      mpact::sim::generic::ArchState* state) override {
    return std::make_unique<isa32::CoralnpuSlot>(state);
  }
};

// This class implements the generic DecoderInterface and provides a bridge
// to the (isa specific) generated decoder classes.
class CoralNPUDecoder : public mpact::sim::generic::DecoderInterface {
 public:
  using SlotEnum = isa32::SlotEnum;
  using OpcodeEnum = isa32::OpcodeEnum;

  CoralNPUDecoder(CoralNPUState* state,
                  mpact::sim::util::MemoryInterface* memory);
  CoralNPUDecoder() = delete;
  ~CoralNPUDecoder() override;

  // This will always return a valid instruction that can be executed. In the
  // case of a decode error, the semantic function in the instruction object
  // instance will raise an internal simulator error when executed.
  mpact::sim::generic::Instruction* DecodeInstruction(
      uint64_t address) override;

  // Return the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override {
    return static_cast<int>(OpcodeEnum::kPastMaxValue);
  }
  // Return the name of the opcode at the given index.
  const char* GetOpcodeName(int index) const override {
    return isa32::kOpcodeNames[index];
  }

  // Getter.
  coralnpu::sim::isa32::CoralNPUEncoding* coralnpu_encoding() const {
    return coralnpu_encoding_;
  }

 private:
  CoralNPUState* state_;
  std::unique_ptr<mpact::sim::riscv::RiscVGenericDecoder<
      CoralNPUState, isa32::OpcodeEnum, isa32::CoralNPUEncoding,
      isa32::CoralNPUInstructionSet>>
      decoder_;
  std::unique_ptr<mpact::sim::generic::ProgramError> decode_error_;
  mpact::sim::generic::DataBuffer* inst_db_;
  isa32::CoralNPUEncoding* coralnpu_encoding_;
  CoralNPUIsaFactory* coralnpu_isa_factory_;
  isa32::CoralNPUInstructionSet* coralnpu_isa_;
};

}  // namespace coralnpu::sim

#endif  // SIM_DECODER_H_
