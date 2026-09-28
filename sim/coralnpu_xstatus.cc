// Copyright 2026 Google LLC
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

#include "sim/coralnpu_xstatus.h"

#include <cstdint>

#include "riscv/riscv_misa.h"
#include "riscv/riscv_xstatus.h"
#include "mpact/sim/generic/arch_state.h"

namespace coralnpu {
namespace sim {

CoralNPUMStatus::CoralNPUMStatus(uint32_t initial_value,
                                 ::mpact::sim::generic::ArchState* state,
                                 ::mpact::sim::riscv::RiscVMIsa* misa)
    : ::mpact::sim::riscv::RiscVMStatus(initial_value, state, misa) {
  set_read_mask(kCoralNpuReadMask);
  set_write_mask(kCoralNpuWriteMask);
  Set(initial_value);
}

CoralNPUMStatus::CoralNPUMStatus(uint64_t initial_value,
                                 ::mpact::sim::generic::ArchState* state,
                                 ::mpact::sim::riscv::RiscVMIsa* misa)
    : ::mpact::sim::riscv::RiscVMStatus(initial_value, state, misa) {
  set_read_mask(kCoralNpuReadMask);
  set_write_mask(kCoralNpuWriteMask);
}

static inline uint64_t CoralNpuStretchMStatus32(uint32_t value) {
  uint64_t value64 = static_cast<uint64_t>(value);
  value64 = ((value64 & 0x80000000ULL) << 32) | (value64 & 0x7fffffffULL);
  return value64;
}

static inline uint32_t CoralNpuCompressMStatus64(uint64_t value) {
  uint32_t value32 = ((value >> 32) & 0x80000000ULL) | (value & 0x7fffffffULL);
  return value32;
}

uint32_t CoralNPUMStatus::GetUint32() {
  return CoralNpuCompressMStatus64(GetUint64());
}

uint32_t CoralNPUMStatus::AsUint32() {
  return GetUint32() & coralnpu_read_mask_32_;
}

void CoralNPUMStatus::Write(uint32_t value) {
  Set(value & coralnpu_write_mask_32_);
}

void CoralNPUMStatus::SetBits(uint32_t bits) {
  uint32_t new_value = GetUint32() | (bits & coralnpu_write_mask_32_);
  Set(new_value);
}

void CoralNPUMStatus::ClearBits(uint32_t bits) {
  uint32_t new_value = GetUint32() & ~(bits & coralnpu_write_mask_32_);
  Set(new_value);
}

void CoralNPUMStatus::Set(uint32_t value) {
  uint64_t new_value =
      (CoralNpuStretchMStatus32(value) & coralnpu_set_mask_from_32_) |
      (GetUint64() & ~coralnpu_set_mask_from_32_);
  ::mpact::sim::riscv::RiscVMStatus::Set(new_value);
}

}  // namespace sim
}  // namespace coralnpu
