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

#include "sim/coralnpu_v2_user_decoder.h"

#include <cstdint>
#include <memory>

#include "sim/coralnpu_v2_decoder.h"
#include "sim/coralnpu_v2_encoding.h"
#include "sim/coralnpu_v2_enums.h"
#include "sim/coralnpu_v2_state.h"
#include "absl/base/nullability.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {
using ::coralnpu::sim::CoralNPUV2Encoding;
using ::coralnpu::sim::CoralNPUV2State;
using ::coralnpu::sim::isa32_v2::CoralNPUV2InstructionSet;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ProgramErrorController;
using ::mpact::sim::generic::operator*;  // NOLINT
using ::coralnpu::sim::isa32_v2::kOpcodeNames;
using ::coralnpu::sim::isa32_v2::OpcodeEnum;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::util::MemoryInterface;

CoralNPUV2UserDecoder::CoralNPUV2UserDecoder(
    CoralNPUV2State* /*absl_nonnull*/ state, MemoryInterface* /*absl_nonnull*/ memory)
    : state_(state), memory_(memory) {
  // Need a data buffer to load instructions from memory. Allocate a single
  // buffer that can be reused for each instruction word.
  inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  coralnpu_v2_isa_factory_ = std::make_unique<CoralNPUV2IsaFactory>();
  coralnpu_v2_isa_ = std::make_unique<CoralNPUV2InstructionSet>(
      state, coralnpu_v2_isa_factory_.get());
  coralnpu_v2_encoding_ = std::make_unique<CoralNPUV2Encoding>(state);
  decode_error_ = state->program_error_controller()->GetProgramError(
      ProgramErrorController::kInternalErrorName);
  decoder_ = std::make_unique<RiscVGenericDecoder>(
      state, memory, coralnpu_v2_encoding_.get(), coralnpu_v2_isa_.get());
}

CoralNPUV2UserDecoder::~CoralNPUV2UserDecoder() { inst_db_->DecRef(); }

Instruction* CoralNPUV2UserDecoder::DecodeInstruction(uint64_t address) {
  if (!state_->IsAddressInItcmRange(address)) {
    Instruction* inst = new Instruction(0, state_);
    inst->set_size(1);
    inst->SetDisassemblyString("Invalid instruction address");
    inst->set_opcode(*::coralnpu::sim::isa32_v2::OpcodeEnum::kNone);
    inst->set_address(address);
    inst->set_semantic_function([this](Instruction* inst) {
      state_->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kInstructionAccessFault, inst->address(),
                   inst);
    });
    return inst;
  }

  return decoder_->DecodeInstruction(address);
}

int CoralNPUV2UserDecoder::GetNumOpcodes() const {
  return static_cast<int>(OpcodeEnum::kPastMaxValue);
}

const char* CoralNPUV2UserDecoder::GetOpcodeName(int index) const {
  return kOpcodeNames[index];
}

}  // namespace coralnpu::sim
