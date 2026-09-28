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
  static constexpr uint64_t kVsMask = 0x0000'0600ULL;
  static constexpr uint64_t kMsMask = 0x6000'0000ULL;
  static constexpr uint64_t kCoralNpuReadMask =
      RiscVMStatus::kDefaultReadMask | kMsMask | kVsMask;
  static constexpr uint64_t kCoralNpuWriteMask =
      RiscVMStatus::kDefaultWriteMask | kMsMask | kVsMask;

  CoralNPUMStatus() = delete;
  CoralNPUMStatus(uint32_t initial_value,
                  ::mpact::sim::generic::ArchState* state,
                  ::mpact::sim::riscv::RiscVMIsa* misa);
  CoralNPUMStatus(uint64_t initial_value,
                  ::mpact::sim::generic::ArchState* state,
                  ::mpact::sim::riscv::RiscVMIsa* misa);
  ~CoralNPUMStatus() override = default;

  bool sd() {
    return RiscVMStatus::sd() || ms() == 0b11 || vs() == 0b11 || fs() == 0b11 ||
           xs() == 0b11;
  }

  // MS - matrix state dirty (bits [30:29]).
  int ms() { return GetterHelper<29, 0b11>(); }
  void set_ms(uint32_t value) { SetterHelper<29, 0b11>(value); }

  // VS - vector state dirty (bits [10:9]).
  int vs() { return GetterHelper<9, 0b11>(); }
  void set_vs(uint32_t value) { SetterHelper<9, 0b11>(value); }
};

}  // namespace sim
}  // namespace coralnpu

#endif  // SIM_CORALNPU_XSTATUS_H_
