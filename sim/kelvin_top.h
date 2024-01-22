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

#ifndef SIM_KELVIN_TOP_H_
#define SIM_KELVIN_TOP_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "sim/kelvin_enums.h"
#include "sim/kelvin_state.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/declare.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "riscv/riscv_arm_semihost.h"
#include "riscv/riscv_breakpoint.h"
#include "riscv/riscv_fp_state.h"
#include "mpact/sim/generic/component.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/counters.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decode_cache.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/memory/memory_watcher.h"

ABSL_DECLARE_FLAG(bool, use_semihost);

namespace kelvin::sim {

using ::mpact::sim::generic::DataBuffer;
using HaltReason = mpact::sim::generic::CoreDebugInterface::HaltReason;
using HaltReasonValueType =
    mpact::sim::generic::CoreDebugInterface::HaltReasonValueType;

// Custom HaltReason for `ebreak`
const HaltReasonValueType kHaltAbort = *HaltReason::kUserSpecifiedMin + 1;

// Top level class for the Kelvin simulator. This is the main interface for
// interacting and controlling execution of programs running on the simulator.
// This class brings together the decoder, the architecture state, and control.
class KelvinTop : public mpact::sim::generic::Component,
                  public mpact::sim::generic::CoreDebugInterface {
 public:
  using RunStatus = mpact::sim::generic::CoreDebugInterface::RunStatus;

  explicit KelvinTop(std::string name);
  KelvinTop(std::string name, uint64_t memory_block_size_bytes,
            uint64_t memory_size_bytes, uint8_t **memory_block_ptr_list);

  ~KelvinTop() override;

  // Methods inherited from CoreDebugInterface.
  absl::Status Halt() override;
  absl::StatusOr<int> Step(int num) override;
  absl::Status Run() override;
  absl::Status Wait() override;

  absl::StatusOr<RunStatus> GetRunStatus() override;
  absl::StatusOr<HaltReasonValueType> GetLastHaltReason() override;

  // Register access by register name.
  absl::StatusOr<uint64_t> ReadRegister(const std::string &name) override;
  absl::Status WriteRegister(const std::string &name, uint64_t value) override;
  absl::StatusOr<DataBuffer *> GetRegisterDataBuffer(
      const std::string &name) override;

  // Read and Write memory methods bypass any semihosting.
  absl::StatusOr<size_t> ReadMemory(uint64_t address, void *buf,
                                    size_t length) override;
  absl::StatusOr<size_t> WriteMemory(uint64_t address, const void *buf,
                                     size_t length) override;

  bool HasBreakpoint(uint64_t address) override;
  absl::Status SetSwBreakpoint(uint64_t address) override;
  absl::Status ClearSwBreakpoint(uint64_t address) override;
  absl::Status ClearAllSwBreakpoints() override;

  // Return the instruction object for the instruction at the given address.
  absl::StatusOr<mpact::sim::generic::Instruction *> GetInstruction(
      uint64_t address) override;
  // Return the string representation for the instruction at the given address.
  absl::StatusOr<std::string> GetDisassembly(uint64_t address) override;

  // Called when a halt is requested.
  void RequestHalt(HaltReasonValueType halt_reason,
                   const mpact::sim::generic::Instruction *inst);
  void RequestHalt(HaltReason halt_reason,
                   const mpact::sim::generic::Instruction *inst);

  // Load a binary image of the SW program.
  absl::Status LoadImage(const std::string &image_path, uint64_t start_address);
  // Accessors.
  sim::KelvinState *state() const { return state_; }
  mpact::sim::util::MemoryInterface *memory() const { return memory_; }

  // Cycle helper function
  uint64_t GetCycleCount() const { return counter_num_cycles_.GetValue(); }

 private:
  // Initialize the top.
  void Initialize();
  // Helper method to step past a breakpoint.
  absl::Status StepPastBreakpoint();
  // Set the pc value.
  void SetPc(uint64_t value);
  // Increment the cycle count.
  void IncrementCycleCount(uint64_t value);
  void IncrementInstructionCount(uint64_t value);

  // The DB factory is used to manage data buffers for memory read/writes.
  mpact::sim::generic::DataBufferFactory db_factory_;
  // Current status and last halt reasons.
  RunStatus run_status_ = RunStatus::kHalted;
  HaltReasonValueType halt_reason_ =
      static_cast<HaltReasonValueType>(HaltReason::kNone);
  // Halting flag. This is set to true when execution must halt.
  bool halted_ = false;
  absl::Notification *run_halted_;
  // The local Kelvin state.
  sim::KelvinState *state_;
  mpact::sim::riscv::RiscVFPState *fp_state_;
  // Breakpoint manager.
  mpact::sim::riscv::RiscVBreakpointManager *rv_bp_manager_ = nullptr;
  // The pc register instance.
  mpact::sim::generic::RegisterBase *pc_;
  // Kelvin decoder decoder instance.
  mpact::sim::generic::DecoderInterface *kelvin_decoder_ = nullptr;
  // Decode cache, memory and memory watcher.
  mpact::sim::generic::DecodeCache *decode_cache_ = nullptr;
  mpact::sim::util::MemoryInterface *memory_ = nullptr;
  mpact::sim::util::MemoryWatcher *watcher_ = nullptr;
  // Counter for the number of instructions simulated.
  mpact::sim::generic::SimpleCounter<uint64_t>
      counter_opcode_[static_cast<int>(sim::isa32::OpcodeEnum::kPastMaxValue)];
  mpact::sim::generic::SimpleCounter<uint64_t> counter_num_instructions_;
  mpact::sim::generic::SimpleCounter<uint64_t> counter_num_cycles_;
  absl::flat_hash_map<uint32_t, std::string> register_id_map_;
  // Setup arm semihosting.
  mpact::sim::riscv::RiscVArmSemihost *semihost_ = nullptr;
};

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_TOP_H_
