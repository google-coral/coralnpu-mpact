// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sim/renode/renode_mpact.h"

#include <cstdint>
#include <limits>
#include <string>

#include "sim/coralnpu_top.h"
#include "sim/renode/renode_debug_interface.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

// This function must be defined in the library.
extern coralnpu::sim::renode::RenodeDebugInterface* CreateCoralNPUSim(
    std::string);

extern coralnpu::sim::renode::RenodeDebugInterface* CreateCoralNPUSim(
    std::string, uint64_t, uint64_t, uint8_t**);

// External "C" functions visible to Renode.
using coralnpu::sim::renode::RenodeAgent;
using coralnpu::sim::renode::RenodeCpuRegister;

// Implementation of the C interface functions. They each forward the call to
// the corresponding method in RenodeAgent.
int32_t construct(int32_t max_name_length) {
  return RenodeAgent::Instance()->Construct(max_name_length);
}

int32_t construct_with_memory(int32_t max_name_length,
                              uint64_t memory_block_size_bytes,
                              uint64_t memory_size_bytes,
                              uint8_t** mem_block_ptr_list) {
  return RenodeAgent::Instance()->Construct(
      max_name_length, memory_block_size_bytes, memory_size_bytes,
      mem_block_ptr_list);
}

int32_t destruct(int32_t id) { return RenodeAgent::Instance()->Destroy(id); }
int32_t reset(int32_t id) { return RenodeAgent::Instance()->Reset(id); }
int32_t get_reg_info_size(int32_t id) {
  return RenodeAgent::Instance()->GetRegisterInfoSize(id);
}
int32_t get_reg_info(int32_t id, int32_t index, char* name, void* info) {
  if (info == nullptr) return -1;
  return RenodeAgent::Instance()->GetRegisterInfo(
      id, index, name, static_cast<RenodeCpuRegister*>(info));
}
uint64_t load_executable(int32_t id, const char* elf_file_name,
                         int32_t* status) {
  return RenodeAgent::Instance()->LoadExecutable(id, elf_file_name, status);
}
int32_t load_image(int32_t id, const char* file_name, uint64_t address) {
  return RenodeAgent::Instance()->LoadImage(id, file_name, address);
}
int32_t read_register(int32_t id, uint32_t reg_id, uint64_t* value) {
  return RenodeAgent::Instance()->ReadRegister(id, reg_id, value);
}
int32_t write_register(int32_t id, uint32_t reg_id, uint64_t value) {
  return RenodeAgent::Instance()->WriteRegister(id, reg_id, value);
}
uint64_t read_memory(int32_t id, uint64_t address, char* buffer,
                     uint64_t length) {
  return RenodeAgent::Instance()->ReadMemory(id, address, buffer, length);
}
uint64_t write_memory(int32_t id, uint64_t address, const char* buffer,
                      uint64_t length) {
  return RenodeAgent::Instance()->WriteMemory(id, address, buffer, length);
}
uint64_t step(int32_t id, uint64_t num_to_step, int32_t* status) {
  return RenodeAgent::Instance()->Step(id, num_to_step, status);
}
int32_t halt(int32_t id, int32_t* status) {
  return RenodeAgent::Instance()->Halt(id, status);
}

namespace coralnpu::sim::renode {

RenodeAgent* RenodeAgent::instance_ = nullptr;
uint32_t RenodeAgent::count_ = 0;

// Create the debug instance by calling the factory function.
int32_t RenodeAgent::Construct(int32_t max_name_length) {
  std::string name = absl::StrCat("renode", count_);
  auto* dbg = CreateCoralNPUSim(name);
  if (dbg == nullptr) {
    return -1;
  }
  core_dbg_instances_.emplace(RenodeAgent::count_, dbg);
  name_length_map_.emplace(RenodeAgent::count_, max_name_length);
  return RenodeAgent::count_++;
}

int32_t RenodeAgent::Construct(int32_t max_name_length,
                               uint64_t memory_block_size_bytes,
                               uint64_t memory_size_bytes,
                               uint8_t** mem_block_ptr_list) {
  std::string name = absl::StrCat("renode", count_);
  auto* dbg = CreateCoralNPUSim(name, memory_block_size_bytes,
                                memory_size_bytes, mem_block_ptr_list);
  if (dbg == nullptr) {
    return -1;
  }
  core_dbg_instances_.emplace(RenodeAgent::count_, dbg);
  name_length_map_.emplace(RenodeAgent::count_, max_name_length);
  return RenodeAgent::count_++;
}

// Destroy the debug instance.
int32_t RenodeAgent::Destroy(int32_t id) {
  // Check for valid instance.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) return -1;
  delete dbg_iter->second;
  core_dbg_instances_.erase(dbg_iter);
  return 0;
}

int32_t RenodeAgent::Reset(int32_t id) {
  // Check for valid instance.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) return -1;
  // For now, do nothing.
  return 0;
}

int32_t RenodeAgent::GetRegisterInfoSize(int32_t id) const {
  // Check for valid instance.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) return -1;
  auto* dbg = dbg_iter->second;
  return dbg->GetRenodeRegisterInfoSize();
}

int32_t RenodeAgent::GetRegisterInfo(int32_t id, int32_t index, char* name,
                                     RenodeCpuRegister* info) {
  // Check for valid instance.
  if (info == nullptr) return -1;
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) return -1;
  auto* dbg = dbg_iter->second;
  int32_t max_len = name_length_map_.at(id);
  auto result = dbg->GetRenodeRegisterInfo(index, max_len, name, *info);
  if (!result.ok()) return -1;
  return 0;
}

// Read the register given by the id.
int32_t RenodeAgent::ReadRegister(int32_t id, uint32_t reg_id,
                                  uint64_t* value) {
  // Check for valid instance.
  if (value == nullptr) return -1;
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) return -1;
  // Read register.
  auto* dbg = dbg_iter->second;
  auto result = dbg->ReadRegister(reg_id);
  if (!result.ok()) return -1;
  *value = result.value();
  return 0;
}

int32_t RenodeAgent::WriteRegister(int32_t id, uint32_t reg_id,
                                   uint64_t value) {
  // Check for valid instance.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) return -1;
  // Write register.
  auto* dbg = dbg_iter->second;
  auto result = dbg->WriteRegister(reg_id, value);
  if (!result.ok()) return -1;
  return 0;
}

uint64_t RenodeAgent::ReadMemory(int32_t id, uint64_t address, char* buffer,
                                 uint64_t length) {
  // Check for valid desktop.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) {
    LOG(ERROR) << "No such core dbg instance: " << id;
    return 0;
  }
  auto* dbg = dbg_iter->second;
  auto res = dbg->ReadMemory(address, buffer, length);
  if (!res.ok()) return 0;
  return res.value();
}

uint64_t RenodeAgent::WriteMemory(int32_t id, uint64_t address,
                                  const char* buffer, uint64_t length) {
  // Check for valid desktop.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) {
    LOG(ERROR) << "No such core dbg instance: " << id;
    return 0;
  }
  auto* dbg = dbg_iter->second;
  auto res = dbg->WriteMemory(address, buffer, length);
  if (!res.ok()) return 0;
  return res.value();
}

uint64_t RenodeAgent::LoadExecutable(int32_t id, const char* file_name,
                                     int32_t* status) {
  // Check for valid desktop.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) {
    LOG(ERROR) << "No such core dbg instance: " << id;
    *status = -1;
    return 0;
  }
  // Instantiate loader and try to load the file.
  auto* dbg = dbg_iter->second;
  mpact::sim::util::ElfProgramLoader loader(dbg);
  auto load_res = loader.LoadProgram(file_name);
  if (!load_res.ok()) {
    LOG(ERROR) << "Failed to load program: " << load_res.status().message();
    *status = -1;
    return 0;
  }
  uint64_t entry = load_res.value();
  if (!dbg->WriteRegister("pc", entry).ok()) {
    LOG(ERROR) << "Failed to write to the pc: " << load_res.status().message();
    *status = -1;
    return 0;
  }
  *status = 0;
  return entry;
}

int32_t RenodeAgent::LoadImage(int32_t id, const char* file_name,
                               uint64_t address) {
  // Get the debug interface.
  auto dbg_iter = core_dbg_instances_.find(id);
  if (dbg_iter == core_dbg_instances_.end()) {
    LOG(ERROR) << "No such core dbg instance: " << id;
    return -1;
  }
  auto* dbg = dbg_iter->second;
  auto res = dbg->LoadImage(file_name, address);
  if (!res.ok()) {
    LOG(ERROR) << "Failed to load image: " << res.message();
    return -1;
  }
  return 0;
}

uint64_t RenodeAgent::Step(int32_t id, uint64_t num_to_step, int32_t* status) {
  // Set the default execution status
  if (status != nullptr) {
    *status = static_cast<int32_t>(ExecutionResult::kAborted);
  }

  // Get the core debug if object.
  auto* dbg = RenodeAgent::Instance()->core_dbg(id);
  // Is the debug interface valid?
  if (dbg == nullptr) {
    return 0;
  }
  if (num_to_step == 0) {
    if (status != nullptr) {
      *status = static_cast<int32_t>(ExecutionResult::kOk);
    }
    return 0;
  }
  // Check the previous halt reason.
  auto halt_res = dbg->GetLastHaltReason();
  if (!halt_res.ok()) {
    return 0;
  }
  // If the previous halt reason was a semihost halt request, then we shouldn't
  // step any further. Just return with "waiting for interrupt" code.
  using mpact::sim::generic::operator*;  // NOLINT: used below.
  if (halt_res.value() == *HaltReason::kSemihostHaltRequest) {
    if (status != nullptr) {
      *status = static_cast<int32_t>(ExecutionResult::kAborted);
    }
    return 0;
  }
  // Perform the stepping.
  uint32_t total_executed = 0;
  while (num_to_step > 0) {
    // Check how far to step, and make multiple calls if the number
    // is greater than <int>::max();
    int step_count = (num_to_step > std::numeric_limits<int>::max())
                         ? std::numeric_limits<int>::max()
                         : static_cast<int>(num_to_step);
    auto res = dbg->Step(step_count);
    // An error occurred.
    if (!res.ok()) {
      return total_executed;
    }
    int num_executed = res.value();
    total_executed += num_executed;
    // Check if the execution was halted due to a semihosting halt request,
    // i.e., program exit.
    halt_res = dbg->GetLastHaltReason();
    if (!halt_res.ok()) {
      return total_executed;
    }
    switch (halt_res.value()) {
      case *HaltReason::kSemihostHaltRequest:
        return total_executed;
        break;
      case *HaltReason::kSoftwareBreakpoint:
      case *HaltReason::kHardwareBreakpoint:
        if (status != nullptr) {
          *status = static_cast<int32_t>(ExecutionResult::kStoppedAtBreakpoint);
        }
        return total_executed;
        break;
      case *HaltReason::kUserRequest:
        if (status != nullptr) {
          *status = static_cast<int32_t>(ExecutionResult::kOk);
        }
        return total_executed;
        break;
      case kHaltAbort:  // `ebreak` custom halt reason
        if (status != nullptr) {
          *status = static_cast<int32_t>(ExecutionResult::kAborted);
        }
        return total_executed;
        break;
      default:
        break;
    }

    // If we stepped fewer instructions than anticipated, stop stepping and
    // return with no error.
    if (num_executed < step_count) {
      if (status != nullptr) {
        *status = static_cast<int32_t>(ExecutionResult::kOk);
      }
      return total_executed;
    }
    num_to_step -= num_executed;
  }
  if (status != nullptr) {
    *status = static_cast<int32_t>(ExecutionResult::kOk);
  }
  return total_executed;
}

// Signal the simulator to halt.
int32_t RenodeAgent::Halt(int32_t id, int32_t* status) {
  // Get the core debug if object.
  auto* dbg = RenodeAgent::Instance()->core_dbg(id);
  // Is the debug interface valid?
  if (dbg == nullptr) {
    return -1;
  }
  // Request halt.
  auto halt_status = dbg->Halt();
  if (!halt_status.ok()) {
    if (status != nullptr) {
      *status = static_cast<int32_t>(ExecutionResult::kAborted);
    }
    return -1;
  }
  // Get the halt status.
  auto halt_res = dbg->GetLastHaltReason();
  if (!halt_res.ok()) {
    if (status != nullptr) {
      *status = static_cast<int32_t>(ExecutionResult::kAborted);
    }
    return -1;
  }
  // Map the halt status appropriately.
  using mpact::sim::generic::operator*;  // NOLINT: used below.
  if (status != nullptr) {
    switch (halt_res.value()) {
      case *HaltReason::kSemihostHaltRequest:
        *status = static_cast<int32_t>(ExecutionResult::kAborted);
        break;
      case *HaltReason::kSoftwareBreakpoint:
        *status = static_cast<int32_t>(ExecutionResult::kStoppedAtBreakpoint);
        break;
      case *HaltReason::kUserRequest:
        *status = static_cast<int32_t>(ExecutionResult::kInterrupted);
        break;
      case *HaltReason::kNone:
        *status = static_cast<int32_t>(ExecutionResult::kOk);
        break;
      case coralnpu::sim::kHaltAbort:
        *status = static_cast<int32_t>(ExecutionResult::kAborted);
        break;
      default:
        break;
    }
  }
  return 0;
}

}  // namespace coralnpu::sim::renode
