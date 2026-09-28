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
#include <memory>
#include <string>

#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::MemoryInterface;

namespace {
// StretchMisa32 stretches the 32-bit value into a 64-bit value by moving the
// upper 2 bits of the 32-bit input to the upper 2 bits of the 64-bit output.
inline uint64_t StretchMisa32(uint32_t value) {
  uint64_t value64 = static_cast<uint64_t>(value);
  value64 = ((value64 & 0xc000'0000) << 32) | (value64 & 0x03ff'ffff);
  return value64;
}
}  // namespace

CoralNPUV2State::CoralNPUV2State(
    std::string id, RiscVXlen xlen, MemoryInterface* memory,
    ::mpact::sim::util::AtomicMemoryOpInterface* atomic_memory)
    : RiscVState(id, xlen, memory, atomic_memory) {
  // CoralNPU supports only IALIGN=32 (no C compressed extension).
  // Per RISC-V Privileged Architecture Spec 3.1.14, mepc[1:0] must be zero.
  auto mepc_result = csr_set()->GetCsr("mepc");
  if (mepc_result.ok()) {
    auto* mepc_csr =
        static_cast<::mpact::sim::riscv::RiscV32SimpleCsr*>(*mepc_result);
    mepc_csr->set_write_mask(mepc_csr->write_mask() &
                             ~static_cast<uint32_t>(2));
    mepc_csr->set_read_mask(mepc_csr->read_mask() & ~static_cast<uint32_t>(2));
  }
}

CoralNPUV2State::~CoralNPUV2State() = default;

void CoralNPUV2State::MPause(const Instruction* instruction) {
  for (auto& handler : on_mpause_) {
    if (handler(instruction)) return;
  }
  // Set the return address to the current instruction.
  const uint64_t epc = (instruction != nullptr) ? instruction->address() : 0;
  Trap(/*is_interrupt=*/false, /*trap_value=*/0, *ExceptionCode::kBreakpoint,
       epc, instruction);
}

// Returns true if the requested memory access has the required permissions.
// Note: This implementation currently requires the entire access to be
// contained within a single memory region. If an access spans across multiple
// regions, it will return false even if both regions have the required
// permissions.
bool CoralNPUV2State::HasPermission(uint32_t address, uint32_t size,
                                    MemoryPermission permissions) const {
  uint64_t request_start_address = address;
  uint64_t request_end_address = static_cast<uint64_t>(address) + size;
  for (const auto& region : memory_regions_) {
    uint64_t region_start_address = region.start_address;
    uint64_t region_end_address =
        static_cast<uint64_t>(region.start_address) + region.length;
    if (request_start_address >= region_start_address &&
        request_end_address <= region_end_address) {
      return (region.permissions & permissions) == permissions;
    }
  }
  return false;
}

std::unique_ptr<CoralNPUV2State> CreateCoralNPUV2State(
    std::string id, RiscVXlen xlen, MemoryInterface* memory,
    ::mpact::sim::util::AtomicMemoryOpInterface* atomic_memory,
    const CoralNPUV2StateConfig* config) {
  auto state =
      std::make_unique<CoralNPUV2State>(id, xlen, memory, atomic_memory);
  if (config != nullptr) {
    state->misa()->Set(StretchMisa32(config->initial_misa_value));
    for (const auto& region : config->memory_regions) {
      state->AddMemoryRegion(region.start_address, region.length,
                             region.permissions);
    }
  }
  return state;
}

}  // namespace coralnpu::sim
