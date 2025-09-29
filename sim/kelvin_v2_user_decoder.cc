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

#include "sim/kelvin_v2_user_decoder.h"

#include <cstdint>
#include <memory>

#include "sim/kelvin_v2_decoder.h"
#include "sim/kelvin_v2_encoding.h"
#include "sim/kelvin_v2_enums.h"
#include "sim/kelvin_v2_state.h"
#include "absl/base/nullability.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace kelvin::sim {
using ::kelvin::sim::KelvinV2Encoding;
using ::kelvin::sim::KelvinV2State;
using ::kelvin::sim::isa32_v2::KelvinV2InstructionSet;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ProgramErrorController;
using ::mpact::sim::generic::operator*;  // NOLINT
using ::kelvin::sim::isa32_v2::kOpcodeNames;
using ::kelvin::sim::isa32_v2::OpcodeEnum;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::util::MemoryInterface;

KelvinV2UserDecoder::KelvinV2UserDecoder(KelvinV2State* /*absl_nonnull*/ state,
                                         MemoryInterface* /*absl_nonnull*/ memory)
    : state_(state), memory_(memory) {
  // Need a data buffer to load instructions from memory. Allocate a single
  // buffer that can be reused for each instruction word.
  inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  kelvin_v2_isa_factory_ = std::make_unique<KelvinV2IsaFactory>();
  kelvin_v2_isa_ = std::make_unique<KelvinV2InstructionSet>(
      state, kelvin_v2_isa_factory_.get());
  kelvin_v2_encoding_ = std::make_unique<KelvinV2Encoding>(state);
  decode_error_ = state->program_error_controller()->GetProgramError(
      ProgramErrorController::kInternalErrorName);
}

KelvinV2UserDecoder::~KelvinV2UserDecoder() { inst_db_->DecRef(); }

Instruction* KelvinV2UserDecoder::DecodeInstruction(uint64_t address) {
  // Address alignment check.
  if (address & 0x1) {
    Instruction* inst = new Instruction(0, state_);
    inst->set_size(1);
    inst->SetDisassemblyString("Misaligned instruction address");
    inst->set_opcode(*::kelvin::sim::isa32_v2::OpcodeEnum::kNone);
    inst->set_address(address);
    inst->set_semantic_function([this](Instruction* inst) {
      state_->Trap(/*is_interrupt*/ false, inst->address(),
                   *ExceptionCode::kInstructionAddressMisaligned,
                   inst->address() ^ 0x1, inst);
    });
    return inst;
  }

  // TODO - b/442008530: Trigger a decoder failure if address is outside the
  //                     ITCM range.

  // Read the instruction word from memory and parse it in the encoding parser.
  memory_->Load(address, inst_db_, nullptr, nullptr);
  uint32_t iword = inst_db_->Get<uint32_t>(0);
  kelvin_v2_encoding_->ParseInstruction(iword);

  // Call the isa decoder to obtain a new instruction object for the instruction
  // word that was parsed above.
  return kelvin_v2_isa_->Decode(address, kelvin_v2_encoding_.get());
}

int KelvinV2UserDecoder::GetNumOpcodes() const {
  return static_cast<int>(OpcodeEnum::kPastMaxValue);
}

const char* KelvinV2UserDecoder::GetOpcodeName(int index) const {
  return kOpcodeNames[index];
}

}  // namespace kelvin::sim
