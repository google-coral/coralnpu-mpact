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
    : ::mpact::sim::riscv::RiscVMStatus(
          initial_value, state, misa, kCoralNpuReadMask, kCoralNpuWriteMask) {}

CoralNPUMStatus::CoralNPUMStatus(uint64_t initial_value,
                                 ::mpact::sim::generic::ArchState* state,
                                 ::mpact::sim::riscv::RiscVMIsa* misa)
    : ::mpact::sim::riscv::RiscVMStatus(
          initial_value, state, misa, kCoralNpuReadMask, kCoralNpuWriteMask) {}

}  // namespace sim
}  // namespace coralnpu
