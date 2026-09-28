// Copyright 2026 Google LLC
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

// This file defines the user-level decoder for the CoralNPU M4 architecture.
// It integrates the M4 instruction set, encoding parser, and generic RISC-V
// decoder to translate binary instructions into executable semantic models.

#ifndef SIM_CORALNPU_M4_USER_DECODER_H_
#define SIM_CORALNPU_M4_USER_DECODER_H_

#include <cstdint>
#include <memory>

#include "sim/coralnpu_m4_decoder.h"
#include "sim/coralnpu_m4_encoding.h"
#include "sim/coralnpu_m4_enums.h"
#include "sim/coralnpu_m4_state.h"
#include "absl/base/nullability.h"
#include "riscv/riscv_generic_decoder.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

// Factory class needed by the generated decoder; creates the decoder for each
// slot instance (there is a single slot).
class CoralNPUM4IsaFactory
    : public ::coralnpu::sim::isa32_m4::CoralNPUM4InstructionSetFactory {
  using ArchState = ::mpact::sim::generic::ArchState;
  using CoralnpuM4Slot = ::coralnpu::sim::isa32_m4::CoralnpuM4Slot;

 public:
  std::unique_ptr<CoralnpuM4Slot> CreateCoralnpuM4Slot(
      ArchState* state) override {
    return std::make_unique<CoralnpuM4Slot>(state);
  }
};

// The main instruction decoder for the CoralNPU M4 architecture. It uses
// the generic RISC-V decoder infrastructure to map instruction addresses
// to their decoded mpact::sim::generic::Instruction representations.
class CoralNPUM4UserDecoder : public ::mpact::sim::generic::DecoderInterface {
 public:
  using CoralNPUM4Encoding = ::coralnpu::sim::CoralNPUM4Encoding;
  using CoralNPUM4InstructionSet =
      ::coralnpu::sim::isa32_m4::CoralNPUM4InstructionSet;
  using CoralNPUM4State = ::coralnpu::sim::CoralNPUM4State;
  using DataBuffer = ::mpact::sim::generic::DataBuffer;
  using Instruction = ::mpact::sim::generic::Instruction;
  using MemoryInterface = ::mpact::sim::util::MemoryInterface;
  using OpcodeEnum = ::coralnpu::sim::isa32_m4::OpcodeEnum;
  using ProgramError = ::mpact::sim::generic::ProgramError;
  using RiscVGenericDecoder =
      ::mpact::sim::riscv::RiscVGenericDecoder<CoralNPUM4State, OpcodeEnum,
                                               CoralNPUM4Encoding,
                                               CoralNPUM4InstructionSet>;

  explicit CoralNPUM4UserDecoder(CoralNPUM4State* /*absl_nonnull*/ state,
                                 MemoryInterface* /*absl_nonnull*/ memory);
  ~CoralNPUM4UserDecoder() override;

  // Decodes an instruction at the given address.
  Instruction* DecodeInstruction(uint64_t address) override;

  // Returns the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override;

  // Returns the name of the opcode at the given index.
  const char* GetOpcodeName(int index) const override;

 private:
  CoralNPUM4State* state_;
  std::unique_ptr<ProgramError> decode_error_;
  std::unique_ptr<CoralNPUM4Encoding> coralnpu_m4_encoding_;
  std::unique_ptr<CoralNPUM4IsaFactory> coralnpu_m4_isa_factory_;
  std::unique_ptr<CoralNPUM4InstructionSet> coralnpu_m4_isa_;
  std::unique_ptr<RiscVGenericDecoder> decoder_;
};

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_M4_USER_DECODER_H_
