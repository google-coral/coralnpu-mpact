// Copyright 2026 Google LLC
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

#include "sim/coralnpu_simulator.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "sim/coralnpu_architecture.h"
#include "sim/coralnpu_m3_user_decoder.h"
#include "sim/coralnpu_m4_state.h"
#include "sim/coralnpu_m4_user_decoder.h"
#include "sim/coralnpu_v2_state.h"
#include "sim/coralnpu_v2_user_decoder.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "elfio/elf_types.hpp"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv32_htif_semihost.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/riscv_zvt_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_watcher.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace coralnpu::sim {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::riscv::kFRegisterAliases;
using ::mpact::sim::riscv::kXRegisterAliases;
using ::mpact::sim::riscv::RiscV32HtifSemiHost;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVTop;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::util::ElfProgramLoader;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryWatcher;

std::optional<mpact::sim::riscv::RiscV32HtifSemiHost::SemiHostAddresses>
CoralNPUSimulator::GetHtifMagicAddresses(
    mpact::sim::util::ElfProgramLoader* loader) {
  RiscV32HtifSemiHost::SemiHostAddresses magic;
  auto result = loader->GetSymbol("tohost_ready");
  if (!result.ok()) return std::nullopt;
  magic.tohost_ready = result->first;

  result = loader->GetSymbol("tohost");
  if (!result.ok()) return std::nullopt;
  magic.tohost = result->first;

  result = loader->GetSymbol("fromhost_ready");
  if (!result.ok()) return std::nullopt;
  magic.fromhost_ready = result->first;

  result = loader->GetSymbol("fromhost");
  if (!result.ok()) return std::nullopt;
  magic.fromhost = result->first;

  LOG(INFO) << absl::StrFormat(
      "HTIF magic addresses: tohost=0x%08x, tohost_ready=0x%08x, "
      "fromhost=0x%08x, fromhost_ready=0x%08x",
      magic.tohost, magic.tohost_ready, magic.fromhost, magic.fromhost_ready);
  return magic;
}

CoralNPUSimulator::CoralNPUSimulator(const CoralNPUSimulatorOptions& options)
    : options_(options) {
  // Create the memory interface and the state.
  memory_ = std::make_unique<FlatDemandMemory>();

  CoralNPUV2StateConfig config = {
      .initial_misa_value = options_.initial_misa_value,
      .memory_regions = options_.memory_regions,
  };
  const std::string id = [&]() {
    switch (options_.architecture) {
      case Architecture::kM4:
        return "CoralNPUM4";
      case Architecture::kM3:
        return "CoralNPUM3";
      case Architecture::kV2:
        return "CoralNPUV2";
    }
    LOG(FATAL) << "Unknown architecture";
  }();
  if (options_.architecture == Architecture::kM4) {
    state_ = CreateCoralNPUM4State(id, mpact::sim::riscv::RiscVXlen::RV32,
                                   memory_.get(), /*atomic_memory=*/nullptr,
                                   &config);
  } else {
    state_ = CreateCoralNPUV2State(id, mpact::sim::riscv::RiscVXlen::RV32,
                                   memory_.get(), /*atomic_memory=*/nullptr,
                                   &config);
  }

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
                                          kCoralNPUV2VectorByteLength);
  }

  // Create the floating point and vector states.
  rv_fp_state_ =
      std::make_unique<RiscVFPState>(state_->csr_set(), state_.get());
  state_->set_rv_fp(rv_fp_state_.get());
  rvv_state_ = std::make_unique<RiscVVectorState>(state_.get(),
                                                  kCoralNPUV2VectorByteLength);

  if (options_.architecture == Architecture::kM4) {
    // The M4 generation adds the matrix tile state (and the mtype CSR, which
    // the matrix state registers into the CSR set).
    auto matrix_status =
        ::mpact::sim::riscv::RiscVZvtMatrixState::Create(state_.get());
    CHECK(matrix_status.ok())
        << "Failed to create matrix state: " << matrix_status.status();
    matrix_state_ = std::move(*matrix_status);
    decoder_ = std::make_unique<CoralNPUM4UserDecoder>(
        static_cast<CoralNPUM4State*>(state_.get()), memory_.get());
  } else if (options_.architecture == Architecture::kM3) {
    decoder_ =
        std::make_unique<CoralNPUM3UserDecoder>(state_.get(), memory_.get());
  } else {
    decoder_ =
        std::make_unique<CoralNPUV2UserDecoder>(state_.get(), memory_.get());
  }

  // Create the top level instance of the simulation engine.
  top_ = std::make_unique<RiscVTop>(id, state_.get(), decoder_.get());

  if (!options_.skip_default_handlers) {
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
                << absl::StrFormat("0x%08x", inst->address()) << "\n";
      if (options_.exit_on_ebreak) {
        top_->RequestHalt(HaltReason::kUserRequest, inst);
        return true;
      }
      return false;
    });
  }

  elf_loader_ = std::make_unique<ElfProgramLoader>(memory_.get());

  memory_watcher_ = std::make_unique<MemoryWatcher>(memory_.get());
}

CoralNPUSimulator::~CoralNPUSimulator() = default;

absl::Status CoralNPUSimulator::LoadProgram(
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

  if (options_.semihost_htif) {
    // When semihosting is enabled, we need to allow access to the program's
    // memory regions (data, bss, heap, stack, htif buffers). We use the
    // elf_loader to find the program's LOAD segments and add them as memory
    // regions.
    for (const auto& segment : elf_loader_->elf_reader()->segments) {
      if (segment->get_type() == ELFIO::PT_LOAD) {
        uint64_t addr = segment->get_physical_address();
        if (addr == 0) {
          addr = segment->get_virtual_address();
        }
        uint64_t size = segment->get_memory_size();
        if (size > 0) {
          MemoryPermission permissions = MemoryPermission::kNone;
          uint32_t flags = segment->get_flags();
          if (flags & ELFIO::PF_R) permissions |= MemoryPermission::kRead;
          if (flags & ELFIO::PF_W) permissions |= MemoryPermission::kWrite;
          if (flags & ELFIO::PF_X) permissions |= MemoryPermission::kExecute;

          LOG(INFO) << absl::StrFormat(
              "Adding memory region for segment: 0x%08x:0x%08x (0x%x)", addr,
              size, static_cast<uint8_t>(permissions));
          state_->AddMemoryRegion(static_cast<uint32_t>(addr),
                                  static_cast<uint32_t>(size), permissions);
        }
      }
    }

    // Add htif semihosting.
    if (auto magic_addresses = GetHtifMagicAddresses(elf_loader_.get())) {
      htif_semihost_ = std::make_unique<RiscV32HtifSemiHost>(
          memory_watcher_.get(), memory_.get(), *magic_addresses,
          [this]() {
            LOG(INFO) << "HTIF semihosting halt request received";
            top_->RequestHalt(RiscVTop::HaltReason::kSemihostHaltRequest,
                              nullptr);
          },
          [this](std::string error) {
            LOG(ERROR) << "HTIF semihosting error: " << error;
            top_->RequestHalt(RiscVTop::HaltReason::kSemihostHaltRequest,
                              nullptr);
          });
      state_->set_memory(memory_watcher_.get());
    } else {
      LOG(WARNING) << "HTIF semihosting enabled but magic addresses not found";
    }
  }

  // Initialize the PC to the entry point.
  return top_->WriteRegister("pc", final_entry_point);
}

absl::Status CoralNPUSimulator::Run() { return top_->Run(); }

absl::StatusOr<int> CoralNPUSimulator::Step(int num_steps) {
  return top_->Step(num_steps);
}

absl::StatusOr<uint64_t> CoralNPUSimulator::ReadRegister(
    const std::string& name) {
  return top_->ReadRegister(name);
}

absl::Status CoralNPUSimulator::WriteRegister(const std::string& name,
                                              uint64_t value) {
  return top_->WriteRegister(name, value);
}

absl::StatusOr<mpact::sim::generic::DataBuffer*>
CoralNPUSimulator::GetRegisterDataBuffer(const std::string& name) {
  return top_->GetRegisterDataBuffer(name);
}

absl::StatusOr<size_t> CoralNPUSimulator::ReadMemory(uint64_t address,
                                                     void* buf, size_t length) {
  return top_->ReadMemory(address, buf, length);
}

absl::StatusOr<size_t> CoralNPUSimulator::WriteMemory(uint64_t address,
                                                      const void* buf,
                                                      size_t length) {
  return top_->WriteMemory(address, buf, length);
}

void CoralNPUSimulator::RunInteractive() {
  mpact::sim::riscv::DebugCommandShell cmd_shell;
  cmd_shell.AddCore({top_.get(), [this]() { return elf_loader_.get(); }});
  cmd_shell.Run(std::cin, std::cout);
}

absl::Status CoralNPUSimulator::Wait() { return top_->Wait(); }

absl::Status CoralNPUSimulator::Halt() { return top_->Halt(); }

uint64_t CoralNPUSimulator::GetCycleCount() const {
  return top_->counter_num_cycles()->GetValue();
}

absl::Status CoralNPUSimulator::SetSwBreakpoint(uint64_t address) {
  return top_->SetSwBreakpoint(address);
}

absl::Status CoralNPUSimulator::ClearSwBreakpoint(uint64_t address) {
  return top_->ClearSwBreakpoint(address);
}

}  // namespace coralnpu::sim
