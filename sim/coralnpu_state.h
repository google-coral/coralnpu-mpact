/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SIM_CORALNPU_STATE_H_
#define SIM_CORALNPU_STATE_H_

#include <any>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

using Instruction = ::mpact::sim::generic::Instruction;

// CoralNPU HW reserves the 31st bit as the magic cache invalidation bit.
// SW can update the load/store address to include that bit to trigger immediate
// cache invalidation. The actual address should exclude that bit. In ISS the
// invalidation is no-op and the actual address should be in the lower bits.
//
// Note the core supports up to 2GB memory (4MB is actually integrated in RTL).
constexpr uint64_t kMemMask = 0x0000'0000'7fff'ffff;

// Default to 256 to match
// https://opensecura.googlesource.com/hw/kelvin/+/master/hdl/chisel/src/coralnpu/Parameters.scala.
inline constexpr uint32_t kVectorLengthInBits = 256;

inline constexpr int kNumVregs = 64;

constexpr uint64_t kCoralnpuMaxMemoryAddress = 0x3f'ffffULL;  // 4MB

template <typename T>
using AccArrayTemplate = std::array<T, kVectorLengthInBits / 32>;

using AccArrayType = AccArrayTemplate<uint32_t>;

using DwAccArray = std::array<uint32_t, 32>;

class CoralNPUState : public mpact::sim::riscv::RiscVState {
 public:
  CoralNPUState(absl::string_view id, mpact::sim::riscv::RiscVXlen xlen,
                mpact::sim::util::MemoryInterface* memory,
                mpact::sim::util::AtomicMemoryOpInterface* atomic_memory);
  CoralNPUState(absl::string_view id, mpact::sim::riscv::RiscVXlen xlen,
                mpact::sim::util::MemoryInterface* memory);
  ~CoralNPUState() override = default;

  // Deleted Constructors and operators.

  CoralNPUState(const CoralNPUState&) = delete;
  CoralNPUState(CoralNPUState&&) = delete;
  CoralNPUState& operator=(const CoralNPUState&) = delete;
  CoralNPUState& operator=(CoralNPUState&&) = delete;

  void set_vector_length(uint32_t length) { vector_length_ = length; }
  uint32_t vector_length() const { return vector_length_; }

  AccArrayType* acc_vec(int index) { return &(acc_register_[index]); }
  AccArrayTemplate<AccArrayType> acc_register() const { return acc_register_; }

  uint32_t* dw_acc_vec(int i) { return &depthwise_acc_register_[i]; }
  DwAccArray& dw_acc_register() { return depthwise_acc_register_; }
  const DwAccArray& dw_acc_register() const { return depthwise_acc_register_; }

  void SetLogArgs(std::any data) { log_args_.emplace_back(std::move(data)); }
  std::string* clog_string() { return &clog_string_; }
  void PrintLog(absl::string_view format_string);

  // Extra CoralNPU terminating state.
  void MPause(const Instruction* inst);

  // Add terminating state handler.
  void AddMpauseHandler(absl::AnyInvocable<bool(const Instruction*)> handler) {
    on_mpause_.emplace_back(std::move(handler));
  }

 private:
  uint32_t vector_length_{kVectorLengthInBits};

  // Variables to store the log arguments.
  std::vector<std::any> log_args_;
  std::string clog_string_;
  // Extra state handlers
  std::vector<absl::AnyInvocable<bool(const Instruction*)>> on_mpause_;

  // Convolution accumulation register, set to be uint32[VLENW][VLENW].
  AccArrayTemplate<AccArrayType> acc_register_;

  // Depthwise convolution accumulation register.
  DwAccArray depthwise_acc_register_;

  // CoralNPU-specific CSR, contains information about the CoralNPU ISA version.
  mpact::sim::riscv::RiscV32SimpleCsr kisa_;

  // minstret CSR.
  mpact::sim::riscv::RiscVCsrInterface* minstret_;
  mpact::sim::riscv::RiscVCsrInterface* minstreth_;
};

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_STATE_H_
