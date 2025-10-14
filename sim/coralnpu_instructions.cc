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

#include "sim/coralnpu_instructions.h"

#include <cstdint>
#include <string>

#include "sim/coralnpu_state.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

void CoralNPUIllegalInstruction(mpact::sim::generic::Instruction* inst) {
  auto* state = static_cast<CoralNPUState*>(inst->state());
  state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
              *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
              /*epc*/ inst->address(), inst);
}

void CoralNPUNopInstruction(mpact::sim::generic::Instruction* inst) {}

void CoralNPUIMpause(const mpact::sim::generic::Instruction* inst) {
  auto* state = static_cast<CoralNPUState*>(inst->state());
  state->MPause(inst);
}

// A helper function to determine if there is a \0 in a char[4] stored in
// uint32_t
bool WordHasZero(uint32_t data) {
  return (((data >> 24) & 0xff) == 0) || (((data >> 16) & 0xff) == 0) ||
         (((data >> 8) & 0xff) == 0) || ((data & 0xff) == 0);
}

// A helper function to load a string from the memory address by detecting the
// '\0' terminator
void CoralNPUStringLoadHelper(const mpact::sim::generic::Instruction* inst,
                              std::string* out_string) {
  auto* state = static_cast<CoralNPUState*>(inst->state());
  auto addr = mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0, 0);
  uint32_t data;
  auto* db = state->db_factory()->Allocate<uint32_t>(1);
  do {
    state->LoadMemory(inst, addr, db, nullptr, nullptr);
    data = db->Get<uint32_t>(0);
    *out_string +=
        std::string(reinterpret_cast<char*>(&data), sizeof(uint32_t));
    addr += 4;
  } while (!WordHasZero(data) && addr < state->max_physical_address());
  // Trim the string properly.
  out_string->resize(out_string->find('\0'));
  db->DecRef();
}

// Handle FLOG, SLOG, CLOG, and KLOG instructions
void CoralNPULogInstruction(int log_mode,
                            mpact::sim::generic::Instruction* inst) {
  auto* state = static_cast<CoralNPUState*>(inst->state());
  switch (log_mode) {
    case 0: {  // Format log op to set the format of the printout and print it.
      std::string format_string;
      CoralNPUStringLoadHelper(inst, &format_string);
      state->PrintLog(format_string);
      break;
    }
    case 1: {  // Scalar log op to load an integer argument.
      // The value is stored as an unsigned integer. The actual format will be
      // determined with the format specifier "d" or "u".
      auto data =
          mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0, 0);
      state->SetLogArgs(data);
      break;
    }
    case 2: {  // Character log op to load a group of char[4] as an argument.
      auto data =
          mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0, 0);
      auto* clog_string = state->clog_string();
      // CLOG can break a long character array as multiple CLOG calls, and they
      // need to be combined as a single string argument.
      *clog_string +=
          std::string(reinterpret_cast<char*>(&data), sizeof(uint32_t));
      if (WordHasZero(data)) {
        // Trim the string properly.
        clog_string->resize(clog_string->find('\0'));
        state->SetLogArgs(*clog_string);
        clog_string->clear();
      }
      break;
    }
    case 3: {  // String log to op load a string argument.
      std::string str_arg;
      CoralNPUStringLoadHelper(inst, &str_arg);
      state->SetLogArgs(str_arg);
      break;
    }
    default:
      break;
  }
}

// Handle Store instructions for mmap_uncached addresses
template <typename T>
void CoralNPUIStore(Instruction* inst) {
  uint32_t base = mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0);
  int32_t offset = mpact::sim::generic::GetInstructionSource<int32_t>(inst, 1);
  uint32_t address = base + offset;
  T value = mpact::sim::generic::GetInstructionSource<T>(inst, 2);
  auto* state = static_cast<CoralNPUState*>(inst->state());
  // Check and exclude the cache invalidation bit. However, the semihost tests
  // use the memory space greater than the coralnpu HW configuration and do not
  // comply to the magic bit setting. Exclude the check and mask for those
  // tests.
  if (state->max_physical_address() <=
      kCoralnpuMaxMemoryAddress) {  // exclude semihost tests
    address &= kMemMask;
  }
  auto* db = state->db_factory()->Allocate(sizeof(T));
  db->Set<T>(0, value);
  state->StoreMemory(inst, address, db);
  db->DecRef();
}

template void CoralNPUIStore<uint32_t>(mpact::sim::generic::Instruction* inst);
template void CoralNPUIStore<uint16_t>(mpact::sim::generic::Instruction* inst);
template void CoralNPUIStore<uint8_t>(mpact::sim::generic::Instruction* inst);

}  // namespace coralnpu::sim
