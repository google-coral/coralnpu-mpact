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

#ifndef SIM_CORALNPU_M4_STATE_H_
#define SIM_CORALNPU_M4_STATE_H_

#include <memory>
#include <string>

#include "sim/coralnpu_v2_state.h"  // IWYU pragma: export
#include "riscv/riscv_state.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

class CoralNPUM4State : public CoralNPUV2State {
 public:
  CoralNPUM4State(
      std::string id, ::mpact::sim::riscv::RiscVXlen xlen,
      ::mpact::sim::util::MemoryInterface* memory,
      ::mpact::sim::util::AtomicMemoryOpInterface* atomic_memory = nullptr);
  ~CoralNPUM4State() override;
};

std::unique_ptr<CoralNPUM4State> CreateCoralNPUM4State(
    std::string id, ::mpact::sim::riscv::RiscVXlen xlen,
    ::mpact::sim::util::MemoryInterface* memory,
    ::mpact::sim::util::AtomicMemoryOpInterface* atomic_memory = nullptr,
    const CoralNPUV2StateConfig* config = nullptr);

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_M4_STATE_H_
