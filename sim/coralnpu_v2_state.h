// Copyright 2024 Google LLC
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

#ifndef SIM_CORALNPU_V2_STATE_H_
#define SIM_CORALNPU_V2_STATE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

enum class MemoryPermission : uint8_t {
  kNone = 0,
  kRead = 1,
  kWrite = 2,
  kExecute = 4,
  kReadWrite = kRead | kWrite,
  kReadExecute = kRead | kExecute,
  kReadWriteExecute = kRead | kWrite | kExecute,
};

constexpr MemoryPermission operator|(MemoryPermission lhs,
                                     MemoryPermission rhs) {
  return static_cast<MemoryPermission>(static_cast<uint8_t>(lhs) |
                                       static_cast<uint8_t>(rhs));
}

constexpr MemoryPermission operator&(MemoryPermission lhs,
                                     MemoryPermission rhs) {
  return static_cast<MemoryPermission>(static_cast<uint8_t>(lhs) &
                                       static_cast<uint8_t>(rhs));
}

inline MemoryPermission& operator|=(MemoryPermission& lhs,
                                    MemoryPermission rhs) {
  lhs = lhs | rhs;
  return lhs;
}

struct CoralNPUV2MemoryRegion {
  uint32_t start_address;
  uint32_t length;
  MemoryPermission permissions;
};

constexpr int kCoralNPUV2VectorByteLength = 16;
constexpr uint32_t kCoralNPUV2DefaultItcmStartAddress = 0x0;
constexpr uint32_t kCoralNPUV2DefaultItcmLength = 0x2000;
constexpr uint32_t kCoralNPUV2DefaultDtcmStartAddress = 0x10000;
constexpr uint32_t kCoralNPUV2DefaultDtcmLength = 0x8000;
constexpr uint32_t kCoralNPUV2DefaultExtmemStartAddress = 0x20000000;
constexpr uint32_t kCoralNPUV2DefaultExtmemLength = 0x400000;
constexpr uint32_t kCoralNPUV2InstructionSize = 4;

constexpr uint32_t kCoralNPUV2DefaultInitialMisaValue =
    (static_cast<uint32_t>(::mpact::sim::riscv::RiscVXlen::RV32) << 30) |
    static_cast<uint32_t>(::mpact::sim::riscv::IsaExtension::kIntegerMulDiv) |
    static_cast<uint32_t>(::mpact::sim::riscv::IsaExtension::kRVIBaseIsa) |
    static_cast<uint32_t>(
        ::mpact::sim::riscv::IsaExtension::kSinglePrecisionFp) |
    static_cast<uint32_t>(::mpact::sim::riscv::IsaExtension::kVectorExtension);

class CoralNPUV2State : public ::mpact::sim::riscv::RiscVState {
 public:
  CoralNPUV2State(
      std::string id, ::mpact::sim::riscv::RiscVXlen xlen,
      ::mpact::sim::util::MemoryInterface* memory,
      ::mpact::sim::util::AtomicMemoryOpInterface* atomic_memory = nullptr);
  ~CoralNPUV2State() override;

  // Dispatches the mpause instruction to the registered handlers. If no
  // handlers return true, the instruction is trapped.
  void MPause(const ::mpact::sim::generic::Instruction* instruction);

  // Registers a handler for the `mpause` instruction.
  // When an `mpause` instruction is executed, handlers are invoked in
  // registration order. If a handler returns `true`, it is assumed to have
  // handled the instruction, and no further handlers are invoked.
  void AddMpauseHandler(
      ::absl::AnyInvocable<bool(const ::mpact::sim::generic::Instruction*)>
          handler) {
    on_mpause_.emplace_back(std::move(handler));
  }

  // Registers a memory region with the specified permissions.
  void AddMemoryRegion(uint32_t start_address, uint32_t length,
                       MemoryPermission permissions) {
    memory_regions_.push_back({.start_address = start_address,
                               .length = length,
                               .permissions = permissions});
  }

  void ClearMemoryRegions() { memory_regions_.clear(); }

  // Returns true if the requested memory access has the required permissions.
  // Note: This implementation currently requires the entire access to be
  // contained within a single memory region.
  bool HasPermission(uint32_t address, uint32_t size,
                     MemoryPermission permissions) const;

 private:
  std::vector<
      absl::AnyInvocable<bool(const ::mpact::sim::generic::Instruction*)>>
      on_mpause_;
  std::vector<CoralNPUV2MemoryRegion> memory_regions_;
};

struct CoralNPUV2StateConfig {
  // The initial value of the misa register.
  uint32_t initial_misa_value = kCoralNPUV2DefaultInitialMisaValue;
  // The memory regions with their permissions.
  std::vector<CoralNPUV2MemoryRegion> memory_regions = {
      {.start_address = kCoralNPUV2DefaultItcmStartAddress,
       .length = kCoralNPUV2DefaultItcmLength,
       .permissions = MemoryPermission::kReadExecute},
      {.start_address = kCoralNPUV2DefaultDtcmStartAddress,
       .length = kCoralNPUV2DefaultDtcmLength,
       .permissions = MemoryPermission::kReadWrite},
      {.start_address = kCoralNPUV2DefaultExtmemStartAddress,
       .length = kCoralNPUV2DefaultExtmemLength,
       .permissions = MemoryPermission::kReadWrite}};
};

std::unique_ptr<CoralNPUV2State> CreateCoralNPUV2State(
    std::string id, ::mpact::sim::riscv::RiscVXlen xlen,
    ::mpact::sim::util::MemoryInterface* memory,
    ::mpact::sim::util::AtomicMemoryOpInterface* atomic_memory = nullptr,
    const CoralNPUV2StateConfig* config = nullptr);

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_V2_STATE_H_
