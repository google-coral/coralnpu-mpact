#include <signal.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "sim/kelvin_state.h"
#include "sim/kelvin_top.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
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

int main(int argc, char **argv) {
  absl::SetProgramUsageMessage("Kelvin MPACT-Sim based CLI tool");
  auto out_args = absl::ParseCommandLine(argc, argv);
  argc = out_args.size();
  argv = &out_args[0];
  if (argc != 2) {
    std::cerr << "Only a single input file allowed" << std::endl;
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

  // Load the elf segments into memory.
  mpact::sim::util::ElfProgramLoader elf_loader(kelvin_top.memory());
  auto load_result = elf_loader.LoadProgram(file_name);
  if (!load_result.ok()) {
    std::cerr << "Error while loading '" << file_name
              << "': " << load_result.status().message();
  }

  // Initialize the PC to the entry point.
  uint32_t entry_point = load_result.value();
  auto pc_write = kelvin_top.WriteRegister("pc", entry_point);
  if (!pc_write.ok()) {
    std::cerr << "Error writing to pc: " << pc_write.message();
  }

  // Determine if this is being run interactively or as a batch job.
  bool interactive = absl::GetFlag(FLAGS_i) || absl::GetFlag(FLAGS_interactive);
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
      std::cerr << run_status.message() << std::endl;
    }

    auto wait_status = kelvin_top.Wait();
    if (!wait_status.ok()) {
      std::cerr << wait_status.message() << std::endl;
    }
    auto sec = absl::ToDoubleSeconds(absl::Now() - t0);
    std::cout << "Total cycles: " << kelvin_top.GetCycleCount() << std::endl;
    std::cout << absl::StrFormat("Simulation done: %0.3f sec\n", sec);
  }
}
