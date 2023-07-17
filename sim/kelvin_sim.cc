#include <signal.h>

#include <iostream>
#include <string>
#include <vector>

#include "sim/kelvin_top.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/log.h"
#include "riscv/debug_command_shell.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

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
    cmd_shell.Run(std::cin, std::cout);
    std::cout << "Total cycles: " << kelvin_top.GetCycleCount() << std::endl;
  } else {
    std::cerr << "Starting simulation\n";

    auto run_status = kelvin_top.Run();
    if (!run_status.ok()) {
      std::cerr << run_status.message() << std::endl;
    }

    auto wait_status = kelvin_top.Wait();
    if (!wait_status.ok()) {
      std::cerr << wait_status.message() << std::endl;
    }
    std::cout << "Total cycles: " << kelvin_top.GetCycleCount() << std::endl;
    std::cerr << "Simulation done\n";
  }
}
