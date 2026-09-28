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

// This file contains the main entry point for the CoralNPU M4 simulator.
// It sets up the simulation environment, loads the target ELF program,
// configures memory regions and the initial architectural state, and
// runs the simulation (either interactively or to completion).

#include <signal.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "sim/coralnpu_architecture.h"
#include "sim/coralnpu_simulator.h"
#include "sim/coralnpu_v2_state.h"
#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mpact/sim/generic/type_helpers.h"

using ::coralnpu::sim::Architecture;
using ::coralnpu::sim::CoralNPUSimulator;
using ::coralnpu::sim::CoralNPUSimulatorOptions;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

// Flags for specifying interactive mode.
ABSL_FLAG(bool, i, false, "Interactive mode");
ABSL_FLAG(bool, interactive, false, "Interactive mode");

ABSL_FLAG(std::optional<uint32_t>, entry_point, std::nullopt,
          "Optionally set the entry point of the program.");

ABSL_FLAG(uint32_t, initial_misa_value, 0x40201120,
          "Set the initial value of the misa register.");
ABSL_FLAG(bool, exit_on_ebreak, false, "Exit on ebreak instruction.");

ABSL_FLAG(std::vector<std::string>, allow_memory_region,
          std::vector<std::string>({"0x0:0x2000:rx", "0x10000:0x8000:rw",
                                    "0x20000000:0x400000:rw"}),
          "Allowed memory region. Format is start_address:length:rwx. "
          "Repeat this option to specify multiple regions.");

ABSL_FLAG(bool, semihost_htif, false, "HTIF semihosting");

// Static pointer to the simulator instance. Used by the control-C handler.
static std::atomic<CoralNPUSimulator*> g_simulator = nullptr;

// Control-c handler to interrupt any running simulation.
static void sim_sigint_handler(int arg) {
  auto* sim = g_simulator.load();
  if (sim != nullptr) {
    absl::Status status = sim->Halt();
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
  absl::SetProgramUsageMessage(
      "CoralNPU M4 (Vector Matrix Extension) Instruction Set Simulator.\n"
      "Usage: coralnpu_m4_sim [options] <elf_file>");
  auto out_args = absl::ParseCommandLine(argc, argv);
  argc = out_args.size();
  argv = &out_args[0];
  if (argc != 2) {
    LOG(ERROR) << "Only a single input file allowed";
    return -1;
  }
  std::string file_name = argv[1];

  CoralNPUSimulatorOptions options;
  options.architecture = Architecture::kM4;
  options.initial_misa_value = absl::GetFlag(FLAGS_initial_misa_value);
  options.exit_on_ebreak = absl::GetFlag(FLAGS_exit_on_ebreak);
  options.memory_regions.clear();
  options.semihost_htif = absl::GetFlag(FLAGS_semihost_htif);

  for (const std::string& region_str :
       absl::GetFlag(FLAGS_allow_memory_region)) {
    std::vector<absl::string_view> parts = absl::StrSplit(region_str, ':');
    if (parts.size() != 3) {
      LOG(ERROR) << "Invalid memory region: " << region_str << ". The expected "
                 << "format is start_address:length:rwx.";
      return -1;
    }
    uint32_t start, length;
    std::string error;
    if (!absl::ParseFlag(parts[0], &start, &error)) {
      LOG(ERROR) << "Invalid memory region start: " << parts[0] << " " << error;
      return -1;
    }
    if (!absl::ParseFlag(parts[1], &length, &error)) {
      LOG(ERROR) << "Invalid memory region length: " << parts[1] << " "
                 << error;
      return -1;
    }
    ::coralnpu::sim::MemoryPermission permissions =
        ::coralnpu::sim::MemoryPermission::kNone;
    for (char c : parts[2]) {
      if (c == 'r') {
        permissions |= ::coralnpu::sim::MemoryPermission::kRead;
      } else if (c == 'w') {
        permissions |= ::coralnpu::sim::MemoryPermission::kWrite;
      } else if (c == 'x') {
        permissions |= ::coralnpu::sim::MemoryPermission::kExecute;
      } else {
        LOG(ERROR) << "Invalid permission char: " << c;
        return -1;
      }
    }
    options.memory_regions.push_back(
        {.start_address = start, .length = length, .permissions = permissions});
  }

  CoralNPUSimulator simulator(options);

  // Set up control-c handling.
  g_simulator = &simulator;
  struct sigaction sa;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sigaddset(&sa.sa_mask, SIGINT);
  sa.sa_handler = &sim_sigint_handler;
  sigaction(SIGINT, &sa, nullptr);

  bool interactive = absl::GetFlag(FLAGS_i) || absl::GetFlag(FLAGS_interactive);
  std::optional<uint32_t> entry_point = absl::GetFlag(FLAGS_entry_point);

  absl::Status load_status = simulator.LoadProgram(file_name, entry_point);
  if (!load_status.ok()) {
    LOG(ERROR) << "Error while loading '" << file_name
               << "': " << load_status.message();
    return -1;
  }

  if (interactive) {
    simulator.RunInteractive();
  } else {
    std::cout << "Starting simulation\n";
    auto t0 = absl::Now();
    absl::Status run_status = simulator.Run();
    if (!run_status.ok()) {
      LOG(ERROR) << "Error during simulation run: " << run_status.message();
      return -1;
    }
    absl::Status wait_status = simulator.Wait();
    if (!wait_status.ok()) {
      LOG(ERROR) << "Error while waiting for simulation: "
                 << wait_status.message();
      return -1;
    }
    auto sec = absl::ToDoubleSeconds(absl::Now() - t0);
    std::cout << "Total cycles: " << simulator.GetCycleCount() << '\n';
    std::cout << absl::StrFormat("Simulation done: %0.3f sec\n", sec);
  }

  g_simulator = nullptr;
  return 0;
}
