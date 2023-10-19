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

#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_KELVIN_RENODE_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_KELVIN_RENODE_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "sim/kelvin_top.h"
#include "sim/renode/renode_debug_interface.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"

// This file defines a wrapper class for the KelvinTop. In addition, the .cc
// class defines a global namespace function that is used by the renode wrapper
// to create a top simulator instance.

extern kelvin::sim::renode::RenodeDebugInterface *CreateKelvinSim(
    std::string name);

extern kelvin::sim::renode::RenodeDebugInterface *CreateKelvinSim(
    std::string name, uint64_t memory_block_size_bytes,
    uint64_t memory_size_bytes, uint8_t **block_ptr_list);

namespace kelvin::sim {

class KelvinRenode : public renode::RenodeDebugInterface {
 public:
  using mpact::sim::generic::CoreDebugInterface::HaltReasonValueType;
  using mpact::sim::generic::CoreDebugInterface::RunStatus;
  using RenodeCpuRegister = kelvin::sim::renode::RenodeCpuRegister;

  explicit KelvinRenode(std::string name);
  explicit KelvinRenode(std::string name, uint64_t memory_block_size_bytes,
                        uint64_t memory_size_bytes, uint8_t **block_ptr_list);

  ~KelvinRenode() override;

  // Request that core stop running override;
  absl::Status Halt() override;
  // Step the core by num instructions.
  absl::StatusOr<int> Step(int num) override;
  // Allow the core to free-run. The loop to run the instructions should be
  // in a separate thread so that this method can return. This allows a user
  // interface built on top of this interface to handle multiple cores running
  // at the same time.
  absl::Status Run() override;
  // Wait until the current core halts execution.
  absl::Status Wait() override;
  // Returns the current run status.
  absl::StatusOr<RunStatus> GetRunStatus() override;
  // Returns the reason for the most recent halt.
  absl::StatusOr<HaltReasonValueType> GetLastHaltReason() override;
  // Read/write the named registers.
  absl::StatusOr<uint64_t> ReadRegister(const std::string &name) override;
  absl::Status WriteRegister(const std::string &name, uint64_t value) override;

  // Read/write the numeric id registers.
  absl::StatusOr<uint64_t> ReadRegister(uint32_t reg_id) override;
  absl::Status WriteRegister(uint32_t reg_id, uint64_t value) override;
  absl::StatusOr<mpact::sim::generic::DataBuffer *> GetRegisterDataBuffer(
      const std::string &name) override;
  // Read/write the buffers to memory.
  absl::StatusOr<size_t> ReadMemory(uint64_t address, void *buf,
                                    size_t length) override;
  absl::StatusOr<size_t> WriteMemory(uint64_t address, const void *buf,
                                     size_t length) override;
  bool HasBreakpoint(uint64_t address) override;
  // Set/Clear software breakpoints at the given addresses.
  absl::Status SetSwBreakpoint(uint64_t address) override;
  absl::Status ClearSwBreakpoint(uint64_t address) override;
  // Remove all software breakpoints.
  absl::Status ClearAllSwBreakpoints() override;
  absl::StatusOr<mpact::sim::generic::Instruction *> GetInstruction(
      uint64_t address) override;
  // Return the string representation for the instruction at the given address.
  absl::StatusOr<std::string> GetDisassembly(uint64_t address) override;
  // Return register information.
  int32_t GetRenodeRegisterInfoSize() const override;
  absl::Status GetRenodeRegisterInfo(int32_t index, int32_t max_len, char *name,
                                     RenodeCpuRegister &info) override;

  absl::Status LoadImage(const std::string &image_path,
                         uint64_t start_address) override;

 private:
  KelvinTop *kelvin_top_ = nullptr;
};

}  // namespace kelvin::sim

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_KELVIN_RENODE_H_
