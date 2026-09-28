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

#include "sim/coralnpu_m4_user_decoder.h"

#include <cstdint>
#include <memory>

#include "sim/coralnpu_m4_decoder.h"
#include "sim/coralnpu_m4_encoding.h"
#include "sim/coralnpu_m4_enums.h"
#include "sim/coralnpu_m4_state.h"
#include "sim/coralnpu_v2_state.h"
#include "absl/base/nullability.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {
using ::coralnpu::sim::CoralNPUM4Encoding;
using ::coralnpu::sim::CoralNPUM4State;
using ::coralnpu::sim::isa32_m4::CoralNPUM4InstructionSet;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ProgramErrorController;
using ::mpact::sim::generic::operator*;  // NOLINT
using ::coralnpu::sim::isa32_m4::kOpcodeNames;
using ::coralnpu::sim::isa32_m4::OpcodeEnum;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::util::MemoryInterface;

CoralNPUM4UserDecoder::CoralNPUM4UserDecoder(
    CoralNPUM4State* /*absl_nonnull*/ state, MemoryInterface* /*absl_nonnull*/ memory)
    : state_(state) {
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  coralnpu_m4_isa_factory_ = std::make_unique<CoralNPUM4IsaFactory>();
  coralnpu_m4_isa_ = std::make_unique<CoralNPUM4InstructionSet>(
      state, coralnpu_m4_isa_factory_.get());
  coralnpu_m4_encoding_ = std::make_unique<CoralNPUM4Encoding>(state);
  decode_error_ = state->program_error_controller()->GetProgramError(
      ProgramErrorController::kInternalErrorName);
  decoder_ = std::make_unique<RiscVGenericDecoder>(
      state, memory, coralnpu_m4_encoding_.get(), coralnpu_m4_isa_.get());
}

CoralNPUM4UserDecoder::~CoralNPUM4UserDecoder() = default;

Instruction* CoralNPUM4UserDecoder::DecodeInstruction(uint64_t address) {
  if (!state_->HasPermission(static_cast<uint32_t>(address), 4,
                             MemoryPermission::kExecute)) {
    Instruction* inst = new Instruction(0, state_);
    inst->set_size(1);
    inst->SetDisassemblyString("Invalid instruction address");
    inst->set_opcode(*::coralnpu::sim::isa32_m4::OpcodeEnum::kNone);
    inst->set_address(address);
    inst->set_semantic_function([this](Instruction* inst) {
      state_->Trap(/*is_interrupt=*/false, /*trap_value=*/inst->address(),
                   *ExceptionCode::kInstructionAccessFault, inst->address(),
                   inst);
    });
    return inst;
  }

  return decoder_->DecodeInstruction(address);
}

int CoralNPUM4UserDecoder::GetNumOpcodes() const {
  return static_cast<int>(OpcodeEnum::kPastMaxValue);
}

const char* CoralNPUM4UserDecoder::GetOpcodeName(int index) const {
  return kOpcodeNames[index];
}

}  // namespace coralnpu::sim
