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

#include "sim/renode/coralnpu_renode.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "sim/coralnpu_top.h"
#include "sim/renode/coralnpu_renode_register_info.h"
#include "sim/renode/renode_debug_interface.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_debug_info.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"

coralnpu::sim::renode::RenodeDebugInterface* CreateCoralNPUSim(
    std::string name) {
  auto* top = new coralnpu::sim::CoralNPURenode(name);
  return top;
}

coralnpu::sim::renode::RenodeDebugInterface* CreateCoralNPUSim(
    std::string name, uint64_t memory_block_size_bytes,
    uint64_t memory_size_bytes, uint8_t** block_ptr_list) {
  auto* top = new coralnpu::sim::CoralNPURenode(
      name, memory_block_size_bytes, memory_size_bytes, block_ptr_list);
  return top;
}

namespace coralnpu::sim {

using HaltReasonValueType =
    mpact::sim::generic::CoreDebugInterface::HaltReasonValueType;
using RunStatus = mpact::sim::generic::CoreDebugInterface::RunStatus;
using Instruction = mpact::sim::generic::Instruction;
using RiscVDebugInfo = mpact::sim::riscv::RiscVDebugInfo;

CoralNPURenode::CoralNPURenode(std::string name) {
  coralnpu_top_ = new CoralNPUTop(name);
}

CoralNPURenode::CoralNPURenode(std::string name,
                               uint64_t memory_block_size_bytes,
                               uint64_t memory_size_bytes,
                               uint8_t** block_ptr_list) {
  coralnpu_top_ = new CoralNPUTop(name, memory_block_size_bytes,
                                  memory_size_bytes, block_ptr_list);
}

CoralNPURenode::~CoralNPURenode() { delete coralnpu_top_; }

absl::Status CoralNPURenode::Halt() { return coralnpu_top_->Halt(); }
absl::Status CoralNPURenode::Halt(HaltReason halt_reason) {
  return coralnpu_top_->Halt(halt_reason);
}
absl::Status CoralNPURenode::Halt(HaltReasonValueType halt_reason) {
  return coralnpu_top_->Halt(halt_reason);
}
absl::StatusOr<int> CoralNPURenode::Step(int num_steps) {
  return coralnpu_top_->Step(num_steps);
}
absl::Status CoralNPURenode::Run() { return coralnpu_top_->Run(); }
absl::Status CoralNPURenode::Wait() { return coralnpu_top_->Wait(); }
absl::StatusOr<RunStatus> CoralNPURenode::GetRunStatus() {
  return coralnpu_top_->GetRunStatus();
}
absl::StatusOr<HaltReasonValueType> CoralNPURenode::GetLastHaltReason() {
  return coralnpu_top_->GetLastHaltReason();
}

absl::StatusOr<uint64_t> CoralNPURenode::ReadRegister(const std::string& name) {
  return coralnpu_top_->ReadRegister(name);
}

absl::Status CoralNPURenode::WriteRegister(const std::string& name,
                                           uint64_t value) {
  return coralnpu_top_->WriteRegister(name, value);
}

absl::StatusOr<size_t> CoralNPURenode::ReadMemory(uint64_t address, void* buf,
                                                  size_t length) {
  return coralnpu_top_->ReadMemory(address, buf, length);
}

absl::StatusOr<size_t> CoralNPURenode::WriteMemory(uint64_t address,
                                                   const void* buf,
                                                   size_t length) {
  return coralnpu_top_->WriteMemory(address, buf, length);
}

absl::StatusOr<mpact::sim::generic::DataBuffer*>
CoralNPURenode::GetRegisterDataBuffer(const std::string& name) {
  return coralnpu_top_->GetRegisterDataBuffer(name);
}

bool CoralNPURenode::HasBreakpoint(uint64_t address) {
  return coralnpu_top_->HasBreakpoint(address);
}

absl::Status CoralNPURenode::SetSwBreakpoint(uint64_t address) {
  return coralnpu_top_->SetSwBreakpoint(address);
}

absl::Status CoralNPURenode::ClearSwBreakpoint(uint64_t address) {
  return coralnpu_top_->ClearSwBreakpoint(address);
}

absl::Status CoralNPURenode::ClearAllSwBreakpoints() {
  return coralnpu_top_->ClearAllSwBreakpoints();
}

absl::StatusOr<mpact::sim::generic::Instruction*>
CoralNPURenode::GetInstruction(uint64_t address) {
  return coralnpu_top_->GetInstruction(address);
}

absl::StatusOr<std::string> CoralNPURenode::GetDisassembly(uint64_t address) {
  return coralnpu_top_->GetDisassembly(address);
}

absl::Status CoralNPURenode::LoadImage(const std::string& image_path,
                                       uint64_t start_address) {
  return coralnpu_top_->LoadImage(image_path, start_address);
}

absl::StatusOr<uint64_t> CoralNPURenode::ReadRegister(uint32_t reg_id) {
  auto ptr = RiscVDebugInfo::Instance()->debug_register_map().find(reg_id);
  if (ptr == RiscVDebugInfo::Instance()->debug_register_map().end()) {
    return absl::NotFoundError(
        absl::StrCat("Not found reg id: ", absl::Hex(reg_id)));
  }

  return ReadRegister(ptr->second);
}

absl::Status CoralNPURenode::WriteRegister(uint32_t reg_id, uint64_t value) {
  auto ptr = RiscVDebugInfo::Instance()->debug_register_map().find(reg_id);
  if (ptr == RiscVDebugInfo::Instance()->debug_register_map().end()) {
    return absl::NotFoundError(
        absl::StrCat("Not found reg id: ", absl::Hex(reg_id)));
  }

  return WriteRegister(ptr->second, value);
}

int32_t CoralNPURenode::GetRenodeRegisterInfoSize() const {
  return CoralNPURenodeRegisterInfo::GetRenodeRegisterInfo().size();
}

absl::Status CoralNPURenode::GetRenodeRegisterInfo(int32_t index,
                                                   int32_t max_len, char* name,
                                                   RenodeCpuRegister& info) {
  auto const& register_info =
      CoralNPURenodeRegisterInfo::GetRenodeRegisterInfo();
  if ((index < 0 || index >= register_info.size())) {
    return absl::OutOfRangeError(
        absl::StrCat("Register info index (", index, ") out of range"));
  }
  info = register_info[index];
  auto const& reg_map = RiscVDebugInfo::Instance()->debug_register_map();
  auto ptr = reg_map.find(info.index);
  if (ptr == reg_map.end()) {
    name[0] = '\0';
  } else {
    strncpy(name, ptr->second.c_str(), max_len);
  }

  return absl::OkStatus();
}

}  // namespace coralnpu::sim
