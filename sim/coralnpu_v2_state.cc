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

// StretchMisa32 stretches the 32-bit value into a 64-bit value by moving the
// upper 2 bits to the lower 32 bits.
static inline uint64_t StretchMisa32(uint32_t value) {
  uint64_t value64 = static_cast<uint64_t>(value);
  value64 = ((value64 & 0xc000'0000) << 32) | (value64 & 0x03ff'ffff);
  return value64;
}

CoralNPUV2State::CoralNPUV2State(
    absl::string_view id, RiscVXlen xlen, MemoryInterface* /*absl_nonnull*/ memory,
    AtomicMemoryOpInterface* /*absl_nullable*/ atomic_memory)
    : RiscVState(id, xlen, memory, atomic_memory) {
  // Set the initial value of the misa CSR to the CoralNPU V2 ISA value.
  misa()->Set(StretchMisa32(kCoralnpuV2MisaInitialValue));
  set_vector_register_width(kCoralnpuV2VectorByteLength);
}
CoralNPUV2State::~CoralNPUV2State() = default;

}  // namespace coralnpu::sim
