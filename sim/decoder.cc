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

#include "sim/decoder.h"

#include <cstdint>
#include <memory>

#include "sim/coralnpu_decoder.h"
#include "sim/coralnpu_encoding.h"
#include "sim/coralnpu_enums.h"
#include "sim/coralnpu_state.h"
#include "riscv/riscv_generic_decoder.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

CoralNPUDecoder::CoralNPUDecoder(CoralNPUState* state,
                                 mpact::sim::util::MemoryInterface* memory)
    : state_(state) {
  // Get a handle to the internal error in the program error controller.
  decode_error_ = state->program_error_controller()->GetProgramError(
      mpact::sim::generic::ProgramErrorController::kInternalErrorName);

  // Need a data buffer to load instructions from memory. Allocate a single
  // buffer that can be reused for each instruction word.
  inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  // Allocate the isa factory class, the top level isa decoder
  // instance, and the encoding parser.
  coralnpu_isa_factory_ = new CoralNPUIsaFactory();
  coralnpu_isa_ =
      new isa32::CoralNPUInstructionSet(state, coralnpu_isa_factory_);
  coralnpu_encoding_ = new isa32::CoralNPUEncoding(state);
  decoder_ = std::make_unique<mpact::sim::riscv::RiscVGenericDecoder<
      CoralNPUState, isa32::OpcodeEnum, isa32::CoralNPUEncoding,
      isa32::CoralNPUInstructionSet>>(state, memory, coralnpu_encoding_,
                                      coralnpu_isa_);
  decode_error_ = state->program_error_controller()->GetProgramError(
      mpact::sim::generic::ProgramErrorController::kInternalErrorName);
}

CoralNPUDecoder::~CoralNPUDecoder() {
  inst_db_->DecRef();
  delete coralnpu_isa_;
  delete coralnpu_isa_factory_;
  delete coralnpu_encoding_;
}

mpact::sim::generic::Instruction* CoralNPUDecoder::DecodeInstruction(
    uint64_t address) {
  return decoder_->DecodeInstruction(address);
}

}  // namespace coralnpu::sim
