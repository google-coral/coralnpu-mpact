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

#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_RENODE_MPACT_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_RENODE_MPACT_H_

#include <cstddef>
#include <cstdint>

#include "sim/renode/renode_debug_interface.h"
#include "absl/container/flat_hash_map.h"

// This file defines the interface that Renode uses to communicate with the
// simulator as well as support classes to implement the interface.
// C interface used by Renode.
extern "C" {
// Construct a debug instance connected to a simulator. Returns the non-zero
// id of the created instance. A return value of zero indicates an error.
int32_t construct(int32_t max_name_length);
// Construct a debug instance connected to a simulator with memory passed from
// renode. Returns the non-zero id of the created instance. A return value of
// zero indicates an error. Note the pointer array needs to be the last argument
// to comply with renode's import binding signature.
int32_t construct_with_memory(int32_t max_name_length,
                              uint64_t memory_block_size_bytes,
                              uint64_t memory_size_bytes,
                              uint8_t **mem_block_ptr_list);
// Destruct the given debug instance. A negative return value indicates an
// error.
int32_t destruct(int32_t id);
// Return the number of register entries.
int32_t get_reg_info_size(int32_t id);
// Return the register entry with the given index. The info pointer should
// store an object of type RenodeCpuRegister.
int32_t get_reg_info(int32_t id, int32_t index, char *name, void *info);
// Load the given executable into the instance with the given id. Return the
// entry point.
uint64_t load_executable(int32_t id, const char *elf_file_name,
                         int32_t *status);
// Load the content of the given file into memory, starting at the given
// address.
int32_t load_image(int32_t id, const char *file_name, uint64_t address);
// Read register reg_id in the instance id, store the value in the pointer
// given. A return value < 0 is an error.
int32_t read_register(int32_t id, uint32_t reg_id, uint64_t *value);
// Write register reg_id in the instance id. A return value < 0 is an error.
int32_t write_register(int32_t id, uint32_t reg_id, uint64_t value);

uint64_t read_memory(int32_t id, uint64_t address, char *buffer,
                     uint64_t length);
uint64_t write_memory(int32_t id, uint64_t address, const char *buffer,
                      uint64_t length);
// Reset the instance. A return value < 0 is an error.
int32_t reset(int32_t id);
// Step the instance id by num_to_step instructions. Return the number of
// instructions stepped. The status is written to the pointer *status.
uint64_t step(int32_t id, uint64_t num_to_step, int32_t *status);
// Halt a free running simulator.
int32_t halt(int32_t id, int32_t *status);
}

namespace kelvin::sim::renode {

// Execution results.
enum class ExecutionResult : int32_t {
  kOk = 0,
  kInterrupted = 1,
  kWaitingForInterrupt = 2,
  kStoppedAtBreakpoint = 3,
  kStoppedAtWatchpoint = 4,
  kExternalMmuFault = 5,
  kAborted = -1,
};
// Intermediary between the C interface above and the actual debug interface
// of the simulator.
class RenodeAgent {
 public:
  using RenodeCpuRegister = kelvin::sim::renode::RenodeCpuRegister;
  constexpr static size_t kBufferSize = 64 * 1024;
  // This is a singleton class, so need a static Instance method.
  static RenodeAgent *Instance() {
    if (instance_ != nullptr) return instance_;
    instance_ = new RenodeAgent();
    return instance_;
  }
  // These methods correspond to the C methods defined above.
  int32_t Construct(int32_t max_name_length);
  int32_t Construct(int32_t max_name_length, uint64_t memory_block_size_bytes,
                    uint64_t memory_size_bytes, uint8_t **mem_block_ptr_list);
  int32_t Destroy(int32_t id);
  int32_t Reset(int32_t id);
  int32_t GetRegisterInfoSize(int32_t id) const;
  int32_t GetRegisterInfo(int32_t id, int32_t index, char *name,
                          RenodeCpuRegister *info);
  int32_t ReadRegister(int32_t id, uint32_t reg_id, uint64_t *value);
  int32_t WriteRegister(int32_t id, uint32_t reg_id, uint64_t value);
  uint64_t ReadMemory(int32_t id, uint64_t address, char *buffer,
                      uint64_t length);
  uint64_t WriteMemory(int32_t id, uint64_t address, const char *buffer,
                       uint64_t length);
  uint64_t LoadExecutable(int32_t id, const char *elf_file_name,
                          int32_t *status);
  int32_t LoadImage(int32_t id, const char *file_name, uint64_t address);
  uint64_t Step(int32_t id, uint64_t num_to_step, int32_t *status);
  int32_t Halt(int32_t id, int32_t *status);
  // Accessor.
  kelvin::sim::renode::RenodeDebugInterface *core_dbg(int32_t id) const {
    auto ptr = core_dbg_instances_.find(id);
    if (ptr != core_dbg_instances_.end()) return ptr->second;
    return nullptr;
  }

 private:
  static RenodeAgent *instance_;
  static uint32_t count_;
  RenodeAgent() = default;
  absl::flat_hash_map<uint32_t, kelvin::sim::renode::RenodeDebugInterface *>
      core_dbg_instances_;
  absl::flat_hash_map<uint32_t, int32_t> name_length_map_;
};

}  // namespace kelvin::sim::renode

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_RENODE_MPACT_H_
