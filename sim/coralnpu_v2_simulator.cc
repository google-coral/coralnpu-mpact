// Copyright 2025 Google LLC
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

#include "sim/coralnpu_v2_simulator.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "sim/coralnpu_v2_state.h"
#include "sim/coralnpu_v2_user_decoder.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::kFRegisterAliases;
using ::mpact::sim::riscv::kXRegisterAliases;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVTop;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::util::FlatDemandMemory;

CoralNPUV2Simulator::CoralNPUV2Simulator(
    const CoralNPUV2SimulatorOptions& options)
    : options_(options) {
  // Create the memory interface and the state.
  memory_ = std::make_unique<FlatDemandMemory>();
  state_ =
      std::make_unique<CoralNPUV2StateFactory>()
          ->SetItcmRange(options_.itcm_start_address, options_.itcm_length)
          ->SetInitialMisaValue(options_.initial_misa_value)
          ->AddLsuAccessRange(options_.dtcm_start_address, options_.dtcm_length)
          ->Create("CoralNPUV2", mpact::sim::riscv::RiscVXlen::RV32,
                   memory_.get(), /*atomic_memory=*/nullptr);

  // Add the scalar, floating point and vector registers to the state.
  std::string reg_name;
  for (int i = 0; i < 32; i++) {
    reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
    state_->AddRegister<RV32Register>(reg_name);
    CHECK_OK(
        state_->AddRegisterAlias<RV32Register>(reg_name, kXRegisterAliases[i]));

    reg_name = absl::StrCat(RiscVState::kFregPrefix, i);
    state_->AddRegister<RVFpRegister>(reg_name);
    CHECK_OK(
        state_->AddRegisterAlias<RVFpRegister>(reg_name, kFRegisterAliases[i]));

    reg_name = absl::StrCat(RiscVState::kVregPrefix, i);
    state_->AddRegister<RVVectorRegister>(reg_name,
                                          kCoralnpuV2VectorByteLength);
  }

  // Create the floating point and vector states.
  rv_fp_state_ =
      std::make_unique<RiscVFPState>(state_->csr_set(), state_.get());
  state_->set_rv_fp(rv_fp_state_.get());
  rvv_state_ = std::make_unique<RiscVVectorState>(state_.get(),
                                                  kCoralnpuV2VectorByteLength);

  decoder_ =
      std::make_unique<CoralNPUV2UserDecoder>(state_.get(), memory_.get());

  // Create the top level instance of the simulation engine.
  top_ = std::make_unique<RiscVTop>("CoralNPUV2", state_.get(), decoder_.get());

  // Add a handler to halt the simulation when an mpause instruction is
  // received.
  state_->AddMpauseHandler([this](const Instruction* inst) {
    std::cout << "mpause instruction received.\n";
    top_->RequestHalt(HaltReason::kUserRequest, inst);
    return true;
  });

  // Add a handler to halt the simulation when an ebreak instruction is
  // received.
  state_->AddEbreakHandler([this](const Instruction* inst) {
    std::cout << "ebreak instruction received. Instruction address: "
              << absl::StrFormat("0x%08x", inst->address()) << std::endl;
    if (options_.exit_on_ebreak) {
      top_->RequestHalt(HaltReason::kUserRequest, inst);
      return true;
    }
    return false;
  });

  elf_loader_ =
      std::make_unique<mpact::sim::util::ElfProgramLoader>(memory_.get());
}

absl::Status CoralNPUV2Simulator::LoadProgram(
    const std::string& file_name, std::optional<uint32_t> entry_point) {
  auto load_result = elf_loader_->LoadProgram(file_name);
  if (!load_result.ok()) {
    return load_result.status();
  }
  auto elf_entry_point = load_result.value();
  uint32_t final_entry_point =
      entry_point.has_value() ? entry_point.value() : elf_entry_point;

  if (elf_entry_point != final_entry_point) {
    LOG(WARNING) << absl::StrFormat(
        "ELF recorded entry point 0x%08x is different from the flag value "
        "0x%08x. The program may not start properly",
        elf_entry_point, final_entry_point);
  }

  // Initialize the PC to the entry point.
  return top_->WriteRegister("pc", final_entry_point);
}

absl::Status CoralNPUV2Simulator::Run() { return top_->Run(); }

absl::StatusOr<int> CoralNPUV2Simulator::Step(int num_steps) {
  return top_->Step(num_steps);
}

absl::StatusOr<uint64_t> CoralNPUV2Simulator::ReadRegister(
    const std::string& name) {
  return top_->ReadRegister(name);
}

absl::Status CoralNPUV2Simulator::WriteRegister(const std::string& name,
                                                uint64_t value) {
  return top_->WriteRegister(name, value);
}

absl::StatusOr<mpact::sim::generic::DataBuffer*>
CoralNPUV2Simulator::GetRegisterDataBuffer(const std::string& name) {
  return top_->GetRegisterDataBuffer(name);
}

absl::StatusOr<size_t> CoralNPUV2Simulator::ReadMemory(uint64_t address,
                                                       void* buf,
                                                       size_t length) {
  return top_->ReadMemory(address, buf, length);
}

absl::StatusOr<size_t> CoralNPUV2Simulator::WriteMemory(uint64_t address,
                                                        const void* buf,
                                                        size_t length) {
  return top_->WriteMemory(address, buf, length);
}

void CoralNPUV2Simulator::RunInteractive() {
  mpact::sim::riscv::DebugCommandShell cmd_shell;
  cmd_shell.AddCore({top_.get(), [this]() { return elf_loader_.get(); }});
  cmd_shell.Run(std::cin, std::cout);
}

absl::Status CoralNPUV2Simulator::Wait() { return top_->Wait(); }

absl::Status CoralNPUV2Simulator::Halt() { return top_->Halt(); }

uint64_t CoralNPUV2Simulator::GetCycleCount() const {
  return top_->counter_num_cycles()->GetValue();
}

absl::Status CoralNPUV2Simulator::SetSwBreakpoint(uint64_t address) {
  return top_->SetSwBreakpoint(address);
}

absl::Status CoralNPUV2Simulator::ClearSwBreakpoint(uint64_t address) {
  return top_->ClearSwBreakpoint(address);
}

}  // namespace coralnpu::sim
