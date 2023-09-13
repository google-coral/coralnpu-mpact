#include "sim/renode/kelvin_renode.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "sim/kelvin_top.h"
#include "sim/renode/kelvin_renode_register_info.h"
#include "sim/renode/renode_debug_interface.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_debug_info.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"

kelvin::sim::renode::RenodeDebugInterface *CreateKelvinSim(std::string name) {
  auto *top = new kelvin::sim::KelvinRenode(name);
  return top;
}

kelvin::sim::renode::RenodeDebugInterface *CreateKelvinSim(
    std::string name, uint64_t memory_block_size_bytes,
    uint64_t memory_size_bytes, uint8_t **block_ptr_list) {
  auto *top = new kelvin::sim::KelvinRenode(name, memory_block_size_bytes,
                                            memory_size_bytes, block_ptr_list);
  return top;
}

namespace kelvin::sim {

using HaltReasonValueType =
    mpact::sim::generic::CoreDebugInterface::HaltReasonValueType;
using RunStatus = mpact::sim::generic::CoreDebugInterface::RunStatus;
using Instruction = mpact::sim::generic::Instruction;
using RiscVDebugInfo = mpact::sim::riscv::RiscVDebugInfo;

KelvinRenode::KelvinRenode(std::string name) {
  kelvin_top_ = new KelvinTop(name);
}

KelvinRenode::KelvinRenode(std::string name, uint64_t memory_block_size_bytes,
                           uint64_t memory_size_bytes,
                           uint8_t **block_ptr_list) {
  kelvin_top_ = new KelvinTop(name, memory_block_size_bytes, memory_size_bytes,
                              block_ptr_list);
}

KelvinRenode::~KelvinRenode() { delete kelvin_top_; }

absl::Status KelvinRenode::Halt() { return kelvin_top_->Halt(); }
absl::StatusOr<int> KelvinRenode::Step(int num_steps) {
  return kelvin_top_->Step(num_steps);
}
absl::Status KelvinRenode::Run() { return kelvin_top_->Run(); }
absl::Status KelvinRenode::Wait() { return kelvin_top_->Wait(); }
absl::StatusOr<RunStatus> KelvinRenode::GetRunStatus() {
  return kelvin_top_->GetRunStatus();
}
absl::StatusOr<HaltReasonValueType> KelvinRenode::GetLastHaltReason() {
  return kelvin_top_->GetLastHaltReason();
}

absl::StatusOr<uint64_t> KelvinRenode::ReadRegister(const std::string &name) {
  return kelvin_top_->ReadRegister(name);
}

absl::Status KelvinRenode::WriteRegister(const std::string &name,
                                         uint64_t value) {
  return kelvin_top_->WriteRegister(name, value);
}

absl::StatusOr<size_t> KelvinRenode::ReadMemory(uint64_t address, void *buf,
                                                size_t length) {
  return kelvin_top_->ReadMemory(address, buf, length);
}

absl::StatusOr<size_t> KelvinRenode::WriteMemory(uint64_t address,
                                                 const void *buf,
                                                 size_t length) {
  return kelvin_top_->WriteMemory(address, buf, length);
}

absl::StatusOr<mpact::sim::generic::DataBuffer *>
KelvinRenode::GetRegisterDataBuffer(const std::string &name) {
  return kelvin_top_->GetRegisterDataBuffer(name);
}

bool KelvinRenode::HasBreakpoint(uint64_t address) {
  return kelvin_top_->HasBreakpoint(address);
}

absl::Status KelvinRenode::SetSwBreakpoint(uint64_t address) {
  return kelvin_top_->SetSwBreakpoint(address);
}

absl::Status KelvinRenode::ClearSwBreakpoint(uint64_t address) {
  return kelvin_top_->ClearSwBreakpoint(address);
}

absl::Status KelvinRenode::ClearAllSwBreakpoints() {
  return kelvin_top_->ClearAllSwBreakpoints();
}

absl::StatusOr<mpact::sim::generic::Instruction *> KelvinRenode::GetInstruction(
    uint64_t address) {
  return kelvin_top_->GetInstruction(address);
}

absl::StatusOr<std::string> KelvinRenode::GetDisassembly(uint64_t address) {
  return kelvin_top_->GetDisassembly(address);
}

absl::Status KelvinRenode::LoadImage(const std::string &image_path,
                                     uint64_t start_address) {
  return kelvin_top_->LoadImage(image_path, start_address);
}

absl::StatusOr<uint64_t> KelvinRenode::ReadRegister(uint32_t reg_id) {
  auto ptr = RiscVDebugInfo::Instance()->debug_register_map().find(reg_id);
  if (ptr == RiscVDebugInfo::Instance()->debug_register_map().end()) {
    return absl::NotFoundError(
        absl::StrCat("Not found reg id: ", absl::Hex(reg_id)));
  }

  return ReadRegister(ptr->second);
}

absl::Status KelvinRenode::WriteRegister(uint32_t reg_id, uint64_t value) {
  auto ptr = RiscVDebugInfo::Instance()->debug_register_map().find(reg_id);
  if (ptr == RiscVDebugInfo::Instance()->debug_register_map().end()) {
    return absl::NotFoundError(
        absl::StrCat("Not found reg id: ", absl::Hex(reg_id)));
  }

  return WriteRegister(ptr->second, value);
}

int32_t KelvinRenode::GetRenodeRegisterInfoSize() const {
  return KelvinRenodeRegisterInfo::GetRenodeRegisterInfo().size();
}

absl::Status KelvinRenode::GetRenodeRegisterInfo(int32_t index, int32_t max_len,
                                                 char *name,
                                                 RenodeCpuRegister &info) {
  auto const &register_info = KelvinRenodeRegisterInfo::GetRenodeRegisterInfo();
  if ((index < 0 || index >= register_info.size())) {
    return absl::OutOfRangeError(
        absl::StrCat("Register info index (", index, ") out of range"));
  }
  info = register_info[index];
  auto const &reg_map = RiscVDebugInfo::Instance()->debug_register_map();
  auto ptr = reg_map.find(info.index);
  if (ptr == reg_map.end()) {
    name[0] = '\0';
  } else {
    strncpy(name, ptr->second.c_str(), max_len);
  }

  return absl::OkStatus();
}

}  // namespace kelvin::sim
