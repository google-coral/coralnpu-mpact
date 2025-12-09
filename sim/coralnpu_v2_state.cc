// Copyright 2025 Google LLC
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

#include "sim/coralnpu_v2_state.h"

#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::AtomicMemoryOpInterface;
using ::mpact::sim::util::MemoryInterface;

CoralNPUV2State::CoralNPUV2State(
    absl::string_view id, RiscVXlen xlen, MemoryInterface* /*absl_nonnull*/ memory,
    AtomicMemoryOpInterface* /*absl_nullable*/ atomic_memory)
    : RiscVState(id, xlen, memory, atomic_memory) {
  set_vector_register_width(kCoralnpuV2VectorByteLength);
}
CoralNPUV2State::~CoralNPUV2State() = default;

void CoralNPUV2State::MPause(const Instruction* instruction) {
  for (auto& handler : on_mpause_) {
    if (handler(instruction)) return;
  }
  // Set the return address to the current instruction.
  const uint64_t epc = (instruction != nullptr) ? instruction->address() : 0;
  Trap(/*is_interrupt=*/false, /*trap_value=*/0, /*exception_code=*/3, epc,
       instruction);
}

bool CoralNPUV2State::IsLsuAccessValid(uint32_t address, uint32_t size) {
  uint64_t request_start_address = address;
  uint64_t request_end_address = address + size;
  for (const auto& range : lsu_access_ranges_) {
    uint64_t access_start_address = range.start_address;
    uint64_t access_end_address = range.start_address + range.length;
    if (request_start_address >= access_start_address &&
        request_end_address <= access_end_address) {
      return true;
    }
  }
  return false;
}

}  // namespace coralnpu::sim
