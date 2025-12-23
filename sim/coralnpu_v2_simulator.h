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

#ifndef SIM_CORALNPU_V2_SIMULATOR_H_
#define SIM_CORALNPU_V2_SIMULATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "sim/coralnpu_v2_state.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace coralnpu::sim {

struct CoralNPUV2SimulatorOptions {
  uint32_t itcm_start_address = 0x0;
  uint32_t itcm_length = 0x2000;
  uint32_t initial_misa_value = 0x40201120;
  uint32_t dtcm_start_address = 0x10000;
  uint32_t dtcm_length = 0x8000;
  bool exit_on_ebreak = false;
};

class CoralNPUV2Simulator {
 public:
  using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;

  explicit CoralNPUV2Simulator(const CoralNPUV2SimulatorOptions& options);
  ~CoralNPUV2Simulator() = default;

  // Loads the program from the given ELF file.
  // If entry_point is provided, it overrides the ELF entry point.
  absl::Status LoadProgram(const std::string& file_name,
                           std::optional<uint32_t> entry_point = std::nullopt);

  // Runs the simulation in batch mode.
  // Returns when the simulation halts.
  absl::Status Run();

  // Runs the simulation for a specified number of steps.
  // Returns the number of steps actually executed.
  absl::StatusOr<int> Step(int num_steps);

  // Register access by register name.
  absl::StatusOr<uint64_t> ReadRegister(const std::string& name);
  absl::Status WriteRegister(const std::string& name, uint64_t value);
  absl::StatusOr<mpact::sim::generic::DataBuffer*> GetRegisterDataBuffer(
      const std::string& name);

  // Read and Write memory methods.
  absl::StatusOr<size_t> ReadMemory(uint64_t address, void* buf, size_t length);
  absl::StatusOr<size_t> WriteMemory(uint64_t address, const void* buf,
                                     size_t length);

  // Breakpoints.
  absl::Status SetSwBreakpoint(uint64_t address);
  absl::Status ClearSwBreakpoint(uint64_t address);

  // Runs the simulation in interactive mode using DebugCommandShell.
  void RunInteractive();

  // Waits for the simulation to complete.
  absl::Status Wait();

  // Requests the simulation to halt.
  absl::Status Halt();

  // Returns the number of cycles executed.
  uint64_t GetCycleCount() const;

  // Accessors
  mpact::sim::riscv::RiscVTop* top() const { return top_.get(); }
  mpact::sim::util::MemoryInterface* memory() const { return memory_.get(); }
  CoralNPUV2State* state() const { return state_.get(); }

 private:
  std::unique_ptr<mpact::sim::util::MemoryInterface> memory_;
  std::unique_ptr<CoralNPUV2State> state_;
  std::unique_ptr<mpact::sim::riscv::RiscVFPState> rv_fp_state_;
  std::unique_ptr<mpact::sim::riscv::RiscVVectorState> rvv_state_;
  std::unique_ptr<mpact::sim::generic::DecoderInterface> decoder_;
  std::unique_ptr<mpact::sim::riscv::RiscVTop> top_;
  std::unique_ptr<mpact::sim::util::ElfProgramLoader> elf_loader_;

  CoralNPUV2SimulatorOptions options_;
};

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_V2_SIMULATOR_H_
