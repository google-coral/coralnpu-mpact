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
#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "sim/kelvin_state.h"
#include "sim/kelvin_top.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv_register_aliases.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"
#include "re2/re2.h"

// Flags for specifying interactive mode.
ABSL_FLAG(bool, i, false, "Interactive mode");
ABSL_FLAG(bool, interactive, false, "Interactive mode");

ABSL_FLAG(uint32_t, bin_memory_offset, 0,
          "Memory offset to load the binary file");
ABSL_FLAG(std::optional<uint32_t>, entry_point, std::nullopt,
          "Optionally set the entry point of the program.");

// Static pointer to the top instance. Used by the control-C handler.
static kelvin::sim::KelvinTop *top = nullptr;

// Control-c handler to interrupt any running simulation.
static void sim_sigint_handler(int arg) {
  if (top != nullptr) {
    (void)top->Halt();
    return;
  } else {
    exit(-1);
  }
}

// Custom debug command to print all the scalar register values.
static bool PrintRegisters(
    absl::string_view input,
    const mpact::sim::riscv::DebugCommandShell::CoreAccess &core_access,
    std::string &output) {
  LazyRE2 xreg_info_re{R"(\s*reg\s+info\s*)"};
  if (!RE2::FullMatch(input, *xreg_info_re)) {
    return false;
  }
  std::string output_str;
  for (int i = 0; i < 32; ++i) {
    std::string reg_name = absl::StrCat("x", i);
    auto result = core_access.debug_interface->ReadRegister(reg_name);
    if (!result.ok()) {
      // Skip the register if error occurs.
      continue;
    }
    output_str +=
        absl::StrCat(mpact::sim::riscv::kXRegisterAliases[i], "\t = [",
                     absl::Hex(result.value(), absl::kZeroPad8), "]\n");
  }
  output = output_str;
  return true;
}

// Custom debug command to print all the assigned vector register values.
static bool PrintVectorRegisters(
    absl::string_view input,
    const mpact::sim::riscv::DebugCommandShell::CoreAccess &core_access,
    std::string &output) {
  LazyRE2 vreg_info_re{R"(\s*vreg\s+info\s*)"};
  if (!RE2::FullMatch(input, *vreg_info_re)) {
    return false;
  }
  std::string output_str;
  for (int i = 0; i < kelvin::sim::kNumVregs; ++i) {
    std::string reg_name = absl::StrCat("v", i);
    auto result = core_access.debug_interface->GetRegisterDataBuffer(reg_name);
    if (!result.ok()) {
      // Skip the register if error occurs.
      continue;
    }
    auto *db = result.value();
    if (db == nullptr) {
      // Skip the register if the data buffer is not available.
      continue;
    }
    std::string data_str;
    std::string sep;
    for (int j = 0; j < kelvin::sim::kVectorLengthInBits / 32; ++j) {
      auto value = db->Get<uint32_t>(j);
      data_str += sep + absl::StrFormat("%08x", value);
      sep = ":";
    }
    output_str += absl::StrCat("v", i, "\t = [", data_str, "]\n");
  }

  output = output_str;
  return true;
}

// Use ELF file's magic word to determine if the input file is an ELF file.
static bool IsElfFile(std::string &file_name) {
  std::ifstream image_file;
  image_file.open(file_name, std::ios::in | std::ios::binary);
  if (image_file.good()) {
    uint32_t magic_word;
    image_file.read(reinterpret_cast<char *>(&magic_word), sizeof(magic_word));
    image_file.close();
    return magic_word == 0x464c457f;  // little endian ELF magic word.
  }
  return false;
}

int main(int argc, char **argv) {
  absl::InitializeLog();
  absl::SetProgramUsageMessage("Kelvin MPACT-Sim based CLI tool");
  auto out_args = absl::ParseCommandLine(argc, argv);
  argc = out_args.size();
  argv = &out_args[0];
  if (argc != 2) {
    LOG(ERROR) << "Only a single input file allowed";
    return -1;
  }
  std::string file_name = argv[1];

  kelvin::sim::KelvinTop kelvin_top("Kelvin");

  // Set up control-c handling.
  top = &kelvin_top;
  struct sigaction sa;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sigaddset(&sa.sa_mask, SIGINT);
  sa.sa_handler = &sim_sigint_handler;
  sigaction(SIGINT, &sa, nullptr);

  bool interactive = absl::GetFlag(FLAGS_i) || absl::GetFlag(FLAGS_interactive);
  auto is_elf_file = IsElfFile(file_name);

  uint32_t entry_point = 0;
  // Load the elf segments into memory.
  mpact::sim::util::ElfProgramLoader elf_loader(kelvin_top.memory());
  if (!is_elf_file && interactive) {
    LOG(ERROR) << "Interactive mode may misbehave without the ELF symbol";
    return -1;
  }
  if (is_elf_file) {
    auto load_result = elf_loader.LoadProgram(file_name);
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
      LOG(ERROR) << absl::StrFormat(
          "ELF recorded entry point 0x%08x is different from the flag value "
          "0x%08x. The program may not start properly",
          elf_entry_point, entry_point);
    }
  } else {  // Load binary file from the specified memory offset.
    // Required the flag `entry_point` to be specified for binary file.
    if (!absl::GetFlag(FLAGS_entry_point).has_value()) {
      LOG(ERROR) << "Need to specify the program entry point";
      return -1;
    }
    entry_point = absl::GetFlag(FLAGS_entry_point).value();
    auto res =
        kelvin_top.LoadImage(file_name, absl::GetFlag(FLAGS_bin_memory_offset));
    if (!res.ok()) {
      LOG(ERROR) << "Error while loading '" << file_name
                 << "': " << res.message();
      return -1;
    }
  }

  // Initialize the PC to the entry point.
  auto pc_write = kelvin_top.WriteRegister("pc", entry_point);
  if (!pc_write.ok()) {
    LOG(ERROR) << "Error writing to pc: " << pc_write.message();
    return -1;
  }

  // Determine if this is being run interactively or as a batch job.
  if (interactive) {
    mpact::sim::riscv::DebugCommandShell cmd_shell(
        {{&kelvin_top, &elf_loader}});
    // Add custom commands to interactive debug command shell.
    cmd_shell.AddCommand(
        "    reg info                       - print all scalar regs",
        PrintRegisters);
    cmd_shell.AddCommand(
        "    vreg info                      - print assigned vector regs",
        PrintVectorRegisters);
    cmd_shell.Run(std::cin, std::cout);
  } else {
    std::cout << "Starting simulation\n";
    auto t0 = absl::Now();

    auto run_status = kelvin_top.Run();
    if (!run_status.ok()) {
      LOG(ERROR) << run_status.message();
    }

    auto wait_status = kelvin_top.Wait();
    if (!wait_status.ok()) {
      LOG(ERROR) << wait_status.message();
    }
    auto sec = absl::ToDoubleSeconds(absl::Now() - t0);
    std::cout << "Total cycles: " << kelvin_top.GetCycleCount() << std::endl;
    std::cout << absl::StrFormat("Simulation done: %0.3f sec\n", sec);
  }
}
