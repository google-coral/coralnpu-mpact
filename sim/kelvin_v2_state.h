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

#ifndef SIM_KELVIN_V2_STATE_H_
#define SIM_KELVIN_V2_STATE_H_

#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace kelvin::sim {

constexpr uint32_t kKelvinV2MisaInitialValue = 0x40201120;
constexpr int kKelvinV2VectorByteLength = 16;

class KelvinV2State : public ::mpact::sim::riscv::RiscVState {
 public:
  using AtomicMemoryOpInterface = ::mpact::sim::util::AtomicMemoryOpInterface;
  using MemoryInterface = ::mpact::sim::util::MemoryInterface;
  using RiscVState = ::mpact::sim::riscv::RiscVState;
  using RiscVXlen = ::mpact::sim::riscv::RiscVXlen;

  KelvinV2State(absl::string_view id, RiscVXlen xlen,
                MemoryInterface* /*absl_nonnull*/ memory,
                AtomicMemoryOpInterface* /*absl_nullable*/ atomic_memory);
  KelvinV2State(absl::string_view id, RiscVXlen xlen,
                MemoryInterface* /*absl_nonnull*/ memory)
      : KelvinV2State(id, xlen, memory, nullptr) {}
  ~KelvinV2State() override;

  // Deleted Constructors and operators.
  KelvinV2State(const KelvinV2State&) = delete;
  KelvinV2State(KelvinV2State&&) = delete;
  KelvinV2State& operator=(const KelvinV2State&) = delete;
  KelvinV2State& operator=(KelvinV2State&&) = delete;
};
}  // namespace kelvin::sim

#endif  // SIM_KELVIN_V2_STATE_H_
