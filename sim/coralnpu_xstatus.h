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

#ifndef SIM_CORALNPU_XSTATUS_H_
#define SIM_CORALNPU_XSTATUS_H_

#include <cstdint>

#include "riscv/riscv_misa.h"
#include "riscv/riscv_xstatus.h"
#include "mpact/sim/generic/arch_state.h"

namespace coralnpu {
namespace sim {

// The CoralNPUMStatus extends the RiscVMStatus to include matrix state dirty
// bits (MS bits, slice [30:29]) exclusive to the CoralNPU extensions.
class CoralNPUMStatus : public ::mpact::sim::riscv::RiscVMStatus {
 public:
  static constexpr uint64_t kCoralNpuReadMask = 0x8000'000f'607f'f9bbULL;
  static constexpr uint64_t kCoralNpuWriteMask = 0x0000'0000'607f'f9bbULL;

  CoralNPUMStatus() = delete;
  CoralNPUMStatus(uint32_t initial_value,
                  ::mpact::sim::generic::ArchState* state,
                  ::mpact::sim::riscv::RiscVMIsa* misa);
  CoralNPUMStatus(uint64_t initial_value,
                  ::mpact::sim::generic::ArchState* state,
                  ::mpact::sim::riscv::RiscVMIsa* misa);
  ~CoralNPUMStatus() override = default;

  bool sd() { return RiscVMStatus::sd() || ms() == 0b11; }

  // MS - matrix state dirty.
  int ms() { return (GetUint64() >> 29) & 0b11; }
  void set_ms(uint32_t value) {
    uint64_t mask = 0b11ULL << 29;
    uint64_t new_val =
        (GetUint64() & ~mask) | ((static_cast<uint64_t>(value) << 29) & mask);
    Set(new_val);
  }

  uint32_t GetUint32() override;
  uint32_t AsUint32() override;
  void Write(uint32_t value) override;
  void SetBits(uint32_t bits) override;
  void ClearBits(uint32_t bits) override;
  void Set(uint32_t value) override;

 protected:
  uint32_t coralnpu_write_mask_32_ = 0x607ff9bb;
  uint32_t coralnpu_read_mask_32_ = 0x607ff9bb | 0x80000000;
  uint64_t coralnpu_set_mask_from_32_ =
      0xffff'fff0'ffff'ffffULL | 0x60000000ULL;
};

}  // namespace sim
}  // namespace coralnpu

#endif  // SIM_CORALNPU_XSTATUS_H_
