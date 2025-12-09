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

#include <signal.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sim/coralnpu_v2_state.h"
#include "sim/coralnpu_v2_user_decoder.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

using ::coralnpu::sim::CoralNPUV2StateFactory;
using ::coralnpu::sim::CoralNPUV2UserDecoder;
using ::coralnpu::sim::kCoralnpuV2VectorByteLength;
using ::mpact::sim::generic::DecoderInterface;
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
using ::mpact::sim::util::ElfProgramLoader;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryInterface;
using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;

// Flags for specifying interactive mode.
ABSL_FLAG(bool, i, false, "Interactive mode");
ABSL_FLAG(bool, interactive, false, "Interactive mode");

ABSL_FLAG(std::optional<uint32_t>, entry_point, std::nullopt,
          "Optionally set the entry point of the program.");

ABSL_FLAG(uint32_t, itcm_start_address, 0x0,
          "Set the start address of the ITCM range.");
ABSL_FLAG(uint32_t, itcm_length, 0x2000, "Set the length of the ITCM range.");
ABSL_FLAG(uint32_t, initial_misa_value, 0x40201120,
          "Set the initial value of the misa register.");
ABSL_FLAG(uint32_t, dtcm_start_address, 0x10000,
          "Set the start address of the DTCM range.");
ABSL_FLAG(uint32_t, dtcm_length, 0x8000, "Set the length of the DTCM range.");

ABSL_FLAG(bool, exit_on_ebreak, false, "Exit on ebreak instruction.");

// Static pointer to the top instance. Used by the control-C handler.
static mpact::sim::riscv::RiscVTop* g_top = nullptr;

// Control-c handler to interrupt any running simulation.
static void sim_sigint_handler(int arg) {
  if (g_top != nullptr) {
    absl::Status status = g_top->Halt();
    if (!status.ok()) {
      LOG(ERROR) << "Error halting simulation: " << status;
    }
    return;
  } else {
    exit(-1);
  }
}

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetProgramUsageMessage("CoralNPUV2 MPACT-Sim based CLI tool");
  auto out_args = absl::ParseCommandLine(argc, argv);
  argc = out_args.size();
  argv = &out_args[0];
  if (argc != 2) {
    LOG(ERROR) << "Only a single input file allowed";
    return -1;
  }
  std::string file_name = argv[1];

  // Create the memory interface and the state.
  std::unique_ptr<MemoryInterface> memory =
      std::make_unique<FlatDemandMemory>();
  auto state =
      std::make_unique<CoralNPUV2StateFactory>()
          ->SetItcmRange(absl::GetFlag(FLAGS_itcm_start_address),
                         absl::GetFlag(FLAGS_itcm_length))
          ->SetInitialMisaValue(absl::GetFlag(FLAGS_initial_misa_value))
          ->AddLsuAccessRange(absl::GetFlag(FLAGS_dtcm_start_address),
                              absl::GetFlag(FLAGS_dtcm_length))
          ->Create("CoralNPUV2", mpact::sim::riscv::RiscVXlen::RV32,
                   memory.get(), /*atomic_memory=*/nullptr);

  // Add the scalar, floating point and vector registers to the state.
  std::string reg_name;
  for (int i = 0; i < 32; i++) {
    reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
    state->AddRegister<RV32Register>(reg_name);
    CHECK_OK(
        state->AddRegisterAlias<RV32Register>(reg_name, kXRegisterAliases[i]));

    reg_name = absl::StrCat(RiscVState::kFregPrefix, i);
    state->AddRegister<RVFpRegister>(reg_name);
    CHECK_OK(
        state->AddRegisterAlias<RVFpRegister>(reg_name, kFRegisterAliases[i]));

    reg_name = absl::StrCat(RiscVState::kVregPrefix, i);
    state->AddRegister<RVVectorRegister>(reg_name, kCoralnpuV2VectorByteLength);
  }
  // Create the floating point and vector states.
  auto rv_fp_state =
      std::make_unique<RiscVFPState>(state->csr_set(), state.get());
  state->set_rv_fp(rv_fp_state.get());
  auto rvv_state = std::make_unique<RiscVVectorState>(
      state.get(), kCoralnpuV2VectorByteLength);
  std::unique_ptr<DecoderInterface> decoder =
      std::make_unique<CoralNPUV2UserDecoder>(state.get(), memory.get());
  // Create the top level instance of the simulation engine.
  auto top =
      std::make_unique<RiscVTop>("CoralNPUV2", state.get(), decoder.get());

  // Add a handler to halt the simulation when an mpause instruction is
  // received.
  state->AddMpauseHandler([&top](const Instruction* inst) {
    std::cout << "mpause instruction received.\n";
    top->RequestHalt(HaltReason::kUserRequest, inst);
    return true;
  });

  // Add a handler to halt the simulation when an ebreak instruction is
  // received. By default, this handler does nothing. If the `exit_on_ebreak`
  // flag is set, the handler will exit the simulation.
  state->AddEbreakHandler([&top](const Instruction* inst) {
    std::cout << "ebreak instruction received. Instruction address: "
              << absl::StrFormat("0x%08x", inst->address()) << std::endl;
    if (absl::GetFlag(FLAGS_exit_on_ebreak)) {
      top->RequestHalt(HaltReason::kUserRequest, inst);
      return true;
    }
    return false;
  });

  // Set up control-c handling.
  g_top = top.get();
  struct sigaction sa;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sigaddset(&sa.sa_mask, SIGINT);
  sa.sa_handler = &sim_sigint_handler;
  sigaction(SIGINT, &sa, nullptr);

  bool interactive = absl::GetFlag(FLAGS_i) || absl::GetFlag(FLAGS_interactive);

  uint32_t entry_point = 0;
  // Load the elf segments into memory.
  auto elf_loader = std::make_unique<ElfProgramLoader>(memory.get());
  auto load_result = elf_loader->LoadProgram(file_name);
  if (!load_result.ok()) {
    LOG(ERROR) << "Error while loading '" << file_name
               << "': " << load_result.status().message();
    return -1;
  }
  auto elf_entry_point = load_result.value();
  // Set the program entry point to based on the ELF info but can
  // be overridden by the `entry_point` flag.
  entry_point = (absl::GetFlag(FLAGS_entry_point).has_value())
                    ? absl::GetFlag(FLAGS_entry_point).value()
                    : elf_entry_point;
  if (elf_entry_point != entry_point) {
    LOG(WARNING) << absl::StrFormat(
        "ELF recorded entry point 0x%08x is different from the flag value "
        "0x%08x. The program may not start properly",
        elf_entry_point, entry_point);
  }

  // Initialize the PC to the entry point.
  auto pc_write = top->WriteRegister("pc", entry_point);
  if (!pc_write.ok()) {
    LOG(ERROR) << "Error writing to pc: " << pc_write.message();
    return -1;
  }

  // Determine if this is being run interactively or as a batch job.
  if (interactive) {
    mpact::sim::riscv::DebugCommandShell cmd_shell;
    cmd_shell.AddCore({top.get(), [&]() { return elf_loader.get(); }});
    cmd_shell.Run(std::cin, std::cout);
  } else {
    std::cout << "Starting simulation\n";
    auto t0 = absl::Now();

    auto run_status = top->Run();
    if (!run_status.ok()) {
      LOG(ERROR) << run_status.message();
    }

    auto wait_status = top->Wait();
    if (!wait_status.ok()) {
      LOG(ERROR) << wait_status.message();
    }
    auto sec = absl::ToDoubleSeconds(absl::Now() - t0);
    std::cout << "Total cycles: " << top->counter_num_cycles()->GetValue()
              << '\n';
    std::cout << absl::StrFormat("Simulation done: %0.3f sec\n", sec);
  }
}
