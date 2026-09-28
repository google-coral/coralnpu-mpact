// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sim/coralnpu_simulator.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "sim/coralnpu_architecture.h"
#include "sim/coralnpu_m3_user_decoder.h"
#include "sim/coralnpu_m4_state.h"
#include "sim/coralnpu_m4_user_decoder.h"
#include "sim/coralnpu_v2_state.h"
#include "sim/coralnpu_v2_user_decoder.h"
#include "sim/coralnpu_xstatus.h"
#include "sim/test/coralnpu_v2_rvv_add_intrinsic_generated.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

/* start - new test macros */
#define ABSL_EXPECT_OK(expression) \
  EXPECT_THAT(expression, ::absl_testing::IsOk())
#define ABSL_ASSERT_OK(expression) \
  ASSERT_THAT(expression, ::absl_testing::IsOk())
/* end - new test macros */

/* start - helper test macro */
#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define ASSERT_OK_AND_ASSIGN(lhs, rexpr)                 \
  auto CONCAT(status_or_line_, __LINE__) = (rexpr);      \
  ABSL_ASSERT_OK(CONCAT(status_or_line_, __LINE__));          \
  lhs = std::move(CONCAT(status_or_line_, __LINE__)).value();
/* end - helper test macro */

constexpr uint32_t kOutputAddress = 0x10810;
constexpr int kNumOutputs = 1024;
constexpr uint32_t kMemSetBreakpointAddress = 0x174;
constexpr uint32_t kMemSetReturnInstructionAddress = 0x18c;
constexpr uint32_t kMpauseAddress = 0xf0;
// The original source code adds input_1 and input_2 vectors element by element
// and stores the result in the output array. input_1 is filled with 0x1 and
// input_2 is filled with 0x6. The final output should be 0x7.
constexpr uint16_t kExpectedOutput = 0x7;

// The depot path to the test directory.
constexpr std::string_view kDepotPath =
    "sim/test/";
constexpr std::string_view kHelloSemihostElf =
    "coralnpu_v2_rvv_add_intrinsic.elf";

namespace {

using ::coralnpu::sim::Architecture;
using ::coralnpu::sim::CoralNPUSimulator;
using ::coralnpu::sim::CoralNPUSimulatorOptions;
using ::coralnpu::sim::MemoryPermission;
using ::coralnpu::sim::test_data::coralnpu_v2_rvv_add_intrinsic::
    GetInstructions;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::operator*;  // NOLINT
using ::mpact::sim::util::ElfProgramLoader;
using ::mpact::sim::util::FlatDemandMemory;
using ::testing::Each;
using ::testing::HasSubstr;
using ::testing::Pointee;
using ::testing::Values;
using ::absl_testing::IsOkAndHolds;

std::string ArchitectureName(
    const ::testing::TestParamInfo<Architecture>& info) {
  switch (info.param) {
    case Architecture::kV2:
      return "V2";
    case Architecture::kM3:
      return "M3";
    case Architecture::kM4:
      return "M4";
    default:
      return "Unknown";
  }
}

std::string GetElfPath(std::string_view filename) {
  return absl::StrCat(kDepotPath, "testfiles/", filename);
}

bool HasTohostSymbol(const std::string& elf_path) {
  auto memory = std::make_unique<FlatDemandMemory>();
  auto loader = std::make_unique<ElfProgramLoader>(memory.get());
  auto status = loader->LoadProgram(elf_path);
  if (!status.ok()) return false;
  auto symbol = loader->GetSymbol("tohost");
  return symbol.ok();
}

class CoralNPUSimulatorTest : public ::testing::TestWithParam<Architecture> {
 public:
  void SetUp() override {
    simulator_options_.architecture = GetParam();
    simulator_options_.exit_on_ebreak = true;
    // Ensure we have a region that covers the instructions and data.
    simulator_options_.memory_regions.push_back(
        {.start_address = 0x0,
         .length = 0x1000000,
         .permissions = MemoryPermission::kReadWriteExecute});
    simulator_ = std::make_unique<CoralNPUSimulator>(simulator_options_);

    // Load the instructions into memory.
    for (const auto& inst : GetInstructions()) {
      ABSL_ASSERT_OK(simulator_->WriteMemory(inst.address, &inst.instruction,
                                        sizeof(inst.instruction)));
    }
  }

  absl::StatusOr<std::unique_ptr<std::vector<uint16_t>>> ReadOutputArray() {
    auto output_array = std::make_unique<std::vector<uint16_t>>(kNumOutputs);
    absl::StatusOr<size_t> status = simulator_->ReadMemory(
        kOutputAddress, output_array->data(), kNumOutputs * sizeof(uint16_t));
    if (!status.ok()) {
      return status.status();
    }
    return output_array;
  }

  // Runs the simulation while the `continue_condition` is met, the mpause
  // instruction is reached, or an error occurs.
  absl::Status KeepRunningUntilDone(
      std::function<bool(uint64_t)> continue_condition) {
    while (true) {
      absl::Status wait_status = simulator_->Wait();
      if (!wait_status.ok()) {
        return wait_status;
      }
      absl::StatusOr<uint64_t> pc = simulator_->ReadRegister("pc");
      if (!pc.ok()) {
        return pc.status();
      }
      if (*pc == kMpauseAddress + 4) {
        return absl::OkStatus();
      }
      if (continue_condition(*pc)) {
        absl::Status run_status = simulator_->Run();
        if (!run_status.ok()) {
          return run_status;
        }
      } else {
        return absl::OkStatus();
      }
    }
  }

 protected:
  CoralNPUSimulatorOptions simulator_options_;
  std::unique_ptr<CoralNPUSimulator> simulator_;
};

TEST_P(CoralNPUSimulatorTest, TestDecoderType) {
  if (GetParam() == Architecture::kM4) {
    EXPECT_NE(dynamic_cast<::coralnpu::sim::CoralNPUM4UserDecoder*>(
                  simulator_->decoder()),
              nullptr);
  } else if (GetParam() == Architecture::kM3) {
    EXPECT_NE(dynamic_cast<::coralnpu::sim::CoralNPUM3UserDecoder*>(
                  simulator_->decoder()),
              nullptr);
  } else {
    EXPECT_NE(dynamic_cast<::coralnpu::sim::CoralNPUV2UserDecoder*>(
                  simulator_->decoder()),
              nullptr);
  }
}

TEST_P(CoralNPUSimulatorTest, TestStateType) {
  if (GetParam() == Architecture::kM4) {
    EXPECT_NE(
        dynamic_cast<::coralnpu::sim::CoralNPUM4State*>(simulator_->state()),
        nullptr);
    EXPECT_NE(dynamic_cast<::coralnpu::sim::CoralNPUMStatus*>(
                  simulator_->state()->mstatus()),
              nullptr);
  } else {
    EXPECT_EQ(
        dynamic_cast<::coralnpu::sim::CoralNPUM4State*>(simulator_->state()),
        nullptr);
    EXPECT_EQ(dynamic_cast<::coralnpu::sim::CoralNPUMStatus*>(
                  simulator_->state()->mstatus()),
              nullptr);
  }
}

TEST_P(CoralNPUSimulatorTest, TestArchitectureId) {
  if (GetParam() == Architecture::kM4) {
    EXPECT_EQ(simulator_->state()->id(), "CoralNPUM4");
  } else if (GetParam() == Architecture::kM3) {
    EXPECT_EQ(simulator_->state()->id(), "CoralNPUM3");
  } else {
    EXPECT_EQ(simulator_->state()->id(), "CoralNPUV2");
  }
}

TEST_P(CoralNPUSimulatorTest, TestMatrixStateExistsForM4) {
  auto mtype_status =
      simulator_->state()->csr_set()->GetCsr(0xC23);  // mtype CSR index
  if (GetParam() == Architecture::kM4) {
    ABSL_EXPECT_OK(mtype_status) << "M4 should have matrix state and mtype CSR.";
    if (mtype_status.ok()) {
      // Read the CSR to ensure the matrix_state_ object is actually alive
      // (catches use-after-free).
      EXPECT_EQ((*mtype_status)->AsUint64(), 0);
    }
  } else {
    EXPECT_FALSE(mtype_status.ok())
        << "Non-M4 architectures should not have mtype CSR.";
  }
}

TEST_P(CoralNPUSimulatorTest, TestMisaStretching) {
  // The default options set initial_misa_value = 0x40201120.
  // bit 30 is set. StretchMisa32 should move it to bit 62.
  // For RV32, this results in the bit being preserved in the 32-bit register.
  auto misa_status = simulator_->ReadRegister("misa");
  ABSL_ASSERT_OK(misa_status);
  EXPECT_EQ(*misa_status, 0x40201120ULL);
}

TEST_P(CoralNPUSimulatorTest, TestIntegerRegisters) {
  for (int i = 0; i < 32; i++) {
    ABSL_ASSERT_OK(simulator_->WriteRegister(absl::StrCat("x", i), i));
    ASSERT_OK_AND_ASSIGN(uint32_t value,
                         simulator_->ReadRegister(absl::StrCat("x", i)));
    EXPECT_EQ(value, i);
  }
}

TEST_P(CoralNPUSimulatorTest, TestFloatingPointRegisters) {
  for (int i = 0; i < 32; i++) {
    ABSL_ASSERT_OK(simulator_->WriteRegister(absl::StrCat("f", i), i));
    ASSERT_OK_AND_ASSIGN(uint64_t value,
                         simulator_->ReadRegister(absl::StrCat("f", i)));
    EXPECT_EQ(value, i);
  }
}

TEST_P(CoralNPUSimulatorTest, TestVectorRegisters) {
  for (int i = 0; i < 32; i++) {
    ASSERT_OK_AND_ASSIGN(DataBuffer * reg_db, simulator_->GetRegisterDataBuffer(
                                                  absl::StrCat("v", i)));
    EXPECT_NE(reg_db, nullptr);
  }
}

TEST_P(CoralNPUSimulatorTest, TestRvvAddIntrinsicMpause) {
  // Run the simulation to the end.
  ABSL_ASSERT_OK(simulator_->Run());
  ABSL_ASSERT_OK(simulator_->Wait());

  // Verify that the simulation stopped after the mpause instruction.
  absl::StatusOr<uint64_t> pc = simulator_->ReadRegister("pc");
  ABSL_ASSERT_OK(pc);
  EXPECT_EQ(*pc, kMpauseAddress + 4);
}

TEST_P(CoralNPUSimulatorTest, TestRvvAddIntrinsicReadMemory) {
  // Run the simulation to the end.
  ABSL_ASSERT_OK(simulator_->Run());
  ABSL_ASSERT_OK(simulator_->Wait());

  // This elf is not a self checking test so we will manually verify the output
  // array.
  EXPECT_THAT(ReadOutputArray(), IsOkAndHolds(Pointee(Each(kExpectedOutput))));
}

TEST_P(CoralNPUSimulatorTest, TestRvvAddIntrinsicSetBreakpoint) {
  // Set the breakpoint and run the simulation.
  ABSL_ASSERT_OK(simulator_->SetSwBreakpoint(kMemSetBreakpointAddress));
  ABSL_ASSERT_OK(simulator_->Run());

  int breakpoint_hit_count = 0;
  bool unexpected_simulation_halt = false;
  ABSL_EXPECT_OK(KeepRunningUntilDone([&](uint64_t pc) {
    if (pc == kMemSetBreakpointAddress) {
      breakpoint_hit_count++;
      return true;  // Continue running after a breakpoint.
    } else {
      unexpected_simulation_halt = true;
      return false;  // Stop running.
    }
  }));

  EXPECT_GE(breakpoint_hit_count, 2);
  EXPECT_FALSE(unexpected_simulation_halt);
}

TEST_P(CoralNPUSimulatorTest, TestRvvAddIntrinsicWriteRegister) {
  // Set the breakpoint and run the simulation.
  ABSL_ASSERT_OK(simulator_->SetSwBreakpoint(kMemSetBreakpointAddress));
  ABSL_ASSERT_OK(simulator_->Run());

  bool unexpected_simulation_halt = false;
  std::vector<absl::Status> write_register_statuses;
  ABSL_EXPECT_OK(KeepRunningUntilDone([&](uint64_t pc) {
    if (pc == kMemSetBreakpointAddress) {
      write_register_statuses.push_back(
          simulator_->WriteRegister("pc", kMemSetReturnInstructionAddress));
      return true;  // Continue running after a breakpoint.
    } else {
      unexpected_simulation_halt = true;
      return false;  // Stop running.
    }
  }));
  EXPECT_FALSE(unexpected_simulation_halt);
  ASSERT_THAT(write_register_statuses, Each(absl_testing::IsOk()));

  // Since we skipped over the memset body, the output array should be
  // uninitialized (0x0).
  EXPECT_THAT(ReadOutputArray(), IsOkAndHolds(Pointee(Each(0x0))));
}

INSTANTIATE_TEST_SUITE_P(Architecture, CoralNPUSimulatorTest,
                         Values(Architecture::kV2, Architecture::kM3,
                                Architecture::kM4),
                         ArchitectureName);

using CoralNPUSimulatorParamTest = ::testing::TestWithParam<Architecture>;

TEST_P(CoralNPUSimulatorParamTest, TestLoadProgramPermissions) {
  CoralNPUSimulatorOptions options;
  options.architecture = GetParam();
  // Clear default memory regions.
  options.memory_regions.clear();
  options.semihost_htif = true;
  auto simulator = std::make_unique<CoralNPUSimulator>(options);

  std::string elf_path = GetElfPath(kHelloSemihostElf);
  ABSL_ASSERT_OK(simulator->LoadProgram(elf_path));

  // The ELF has an executable segment. Check if it's executable.
  // For coralnpu_v2_rvv_add_intrinsic.elf, the code is at 0x0.
  EXPECT_TRUE(
      simulator->state()->HasPermission(0x0, 4, MemoryPermission::kExecute));
}

TEST_P(CoralNPUSimulatorParamTest, TestLoadProgramElfSegments) {
  CoralNPUSimulatorOptions options;
  options.architecture = GetParam();
  // Clear default memory regions to ensure we only have what LoadProgram adds.
  options.memory_regions.clear();
  options.semihost_htif = true;
  auto simulator = std::make_unique<CoralNPUSimulator>(options);

  std::string elf_path = GetElfPath(kHelloSemihostElf);
  ABSL_ASSERT_OK(simulator->LoadProgram(elf_path));

  // For coralnpu_v2_rvv_add_intrinsic.elf:
  // .text is at 0x0, size > 0, flags R X
  // .data is at 0x10000, size > 0, flags R W
  EXPECT_TRUE(
      simulator->state()->HasPermission(0x0, 0x100, MemoryPermission::kRead));
  EXPECT_TRUE(simulator->state()->HasPermission(0x0, 0x100,
                                                MemoryPermission::kExecute));
  EXPECT_FALSE(
      simulator->state()->HasPermission(0x0, 0x100, MemoryPermission::kWrite));

  EXPECT_TRUE(simulator->state()->HasPermission(0x100000, 0x10,
                                                MemoryPermission::kRead));
  EXPECT_TRUE(simulator->state()->HasPermission(0x100000, 0x10,
                                                MemoryPermission::kWrite));
  EXPECT_FALSE(simulator->state()->HasPermission(0x100000, 0x10,
                                                 MemoryPermission::kExecute));
}

TEST_P(CoralNPUSimulatorParamTest, TestEbreakExitOnEbreak) {
  using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;
  using HaltReasonValueType = std::underlying_type_t<HaltReason>;

  CoralNPUSimulatorOptions options;
  options.architecture = GetParam();
  options.exit_on_ebreak = true;
  auto simulator = std::make_unique<CoralNPUSimulator>(options);

  // Add a memory region for code.
  simulator->state()->AddMemoryRegion(0x1000, 0x1000,
                                      MemoryPermission::kReadWriteExecute);

  // Write ebreak instruction (0x00100073) at 0x1000.
  uint32_t ebreak_inst = 0x00100073;
  ASSERT_OK_AND_ASSIGN(size_t written,
                       simulator->WriteMemory(0x1000, &ebreak_inst, 4));
  ASSERT_EQ(written, 4);

  // Set PC to 0x1000.
  ABSL_ASSERT_OK(simulator->WriteRegister("pc", 0x1000));

  // Run the simulation.
  ABSL_ASSERT_OK(simulator->Run());
  ABSL_ASSERT_OK(simulator->Wait());

  // Check if it halted due to ebreak (kUserRequest).
  ASSERT_OK_AND_ASSIGN(HaltReasonValueType halt_reason,
                       simulator->top()->GetLastHaltReason());
  EXPECT_EQ(halt_reason, *HaltReason::kUserRequest);
}

TEST_P(CoralNPUSimulatorParamTest, TestNoPermissionDecode) {
  using ::coralnpu::sim::CoralNPUM3UserDecoder;
  using ::coralnpu::sim::CoralNPUV2UserDecoder;
  using ::mpact::sim::generic::Instruction;
  using ::mpact::sim::riscv::ExceptionCode;

  CoralNPUSimulatorOptions options;
  options.architecture = GetParam();
  auto simulator = std::make_unique<CoralNPUSimulator>(options);

  // Address 0x10000000 has no permissions by default.
  uint32_t addr = 0x10000000;

  std::unique_ptr<::mpact::sim::generic::DecoderInterface> decoder;
  if (GetParam() == Architecture::kM3) {
    decoder = std::make_unique<CoralNPUM3UserDecoder>(simulator->state(),
                                                      simulator->memory());
  } else {
    decoder = std::make_unique<CoralNPUV2UserDecoder>(simulator->state(),
                                                      simulator->memory());
  }

  // Attempt to decode instruction at address 0x1000.
  auto* inst = decoder->DecodeInstruction(addr);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->size(), 1);
  EXPECT_EQ(inst->AsString(), "Invalid instruction address");

  // Execute the instruction, it should trap with InstructionAccessFault.
  bool trap_called = false;
  simulator->state()->set_on_trap(
      [&](bool, uint64_t, uint64_t code, uint64_t, const Instruction*) -> bool {
        trap_called = true;
        EXPECT_EQ(static_cast<ExceptionCode>(code),
                  ExceptionCode::kInstructionAccessFault);
        return true;
      });

  inst->Execute();
  EXPECT_TRUE(trap_called);
  inst->DecRef();
}

TEST_P(CoralNPUSimulatorParamTest, TestHtifAddresses) {
  std::string elf_path = GetElfPath(kHelloSemihostElf);
  auto memory = std::make_unique<FlatDemandMemory>();
  auto loader = std::make_unique<ElfProgramLoader>(memory.get());
  ABSL_ASSERT_OK(loader->LoadProgram(elf_path));

  auto magic = CoralNPUSimulator::GetHtifMagicAddresses(loader.get());
  ASSERT_TRUE(magic.has_value());
  EXPECT_EQ(magic->tohost, 0x100000);
  EXPECT_EQ(magic->tohost_ready, 0x100040);
  EXPECT_EQ(magic->fromhost, 0x100080);
  EXPECT_EQ(magic->fromhost_ready, 0x1000c0);
}

TEST_P(CoralNPUSimulatorParamTest, TestSemihostHtifDisabled) {
  std::string elf_path = GetElfPath(kHelloSemihostElf);
  if (!HasTohostSymbol(elf_path)) {
    GTEST_SKIP() << "ELF " << elf_path
                 << " does not have 'tohost' symbol, skipping HTIF test.";
    return;
  }

  CoralNPUSimulatorOptions options;
  options.architecture = GetParam();
  options.semihost_htif = false;

  auto simulator = std::make_unique<CoralNPUSimulator>(options);

  ABSL_ASSERT_OK(simulator->LoadProgram(elf_path));

  testing::internal::CaptureStdout();

  // Run the simulation for a short while.
  ABSL_ASSERT_OK(simulator->Step(1000));

  std::string output = testing::internal::GetCapturedStdout();
  // If semihost_htif is false, we should NOT see the semihosting output.
  EXPECT_THAT(output, Not(HasSubstr("this looks like atv's magic")));
}

TEST_P(CoralNPUSimulatorParamTest, TestInvalidCsrDecode) {
  using ::mpact::sim::generic::Instruction;

  CoralNPUSimulatorOptions options;
  options.architecture = GetParam();
  auto simulator = std::make_unique<CoralNPUSimulator>(options);

  // csrrs x1, 0xfff, x0 (invalid CSR 0xfff)
  // opcode = 1110011, funct3 = 010, rs1 = 0, rd = 1, imm = 0xfff
  // 0xfff 00000 010 00001 1110011 -> 0xfff020f3
  uint32_t inst_word = 0xfff020f3;
  uint32_t addr = 0x1000;
  simulator->state()->AddMemoryRegion(addr, 4,
                                      MemoryPermission::kReadWriteExecute);
  ABSL_ASSERT_OK(simulator->WriteMemory(addr, &inst_word, 4));

  auto* inst = simulator->decoder()->DecodeInstruction(addr);
  ASSERT_NE(inst, nullptr);
  // Just verify it decodes to something and doesn't crash.
  // The invalid CSR should still decode but might have an ImmediateOperand
  // for the CSR index instead of a register operand.
  EXPECT_GT(inst->SourcesSize(), 0);
  inst->DecRef();
}

INSTANTIATE_TEST_SUITE_P(Architecture, CoralNPUSimulatorParamTest,
                         Values(Architecture::kV2, Architecture::kM3,
                                Architecture::kM4),
                         ArchitectureName);

TEST(CoralNPUV2StateTest, TestMultipleMpauseHandlers) {
  using ::coralnpu::sim::CreateCoralNPUV2State;
  using ::mpact::sim::generic::Instruction;
  using ::mpact::sim::riscv::RiscVXlen;
  using ::mpact::sim::util::FlatDemandMemory;

  auto memory = std::make_unique<FlatDemandMemory>();
  auto state = CreateCoralNPUV2State("test", RiscVXlen::RV32, memory.get());

  int handler1_call_count = 0;
  int handler2_call_count = 0;
  int handler3_call_count = 0;

  state->AddMpauseHandler([&](const Instruction*) {
    handler1_call_count++;
    return false;  // Should continue to next handler.
  });
  state->AddMpauseHandler([&](const Instruction*) {
    handler2_call_count++;
    return true;  // Should stop here.
  });
  state->AddMpauseHandler([&](const Instruction*) {
    handler3_call_count++;
    return true;  // Should not be called.
  });

  state->MPause(nullptr);

  EXPECT_EQ(handler1_call_count, 1);
  EXPECT_EQ(handler2_call_count, 1);
  EXPECT_EQ(handler3_call_count, 0);
}

TEST(CoralNPUV2StateTest, TestMpauseDefaultTrap) {
  using ::coralnpu::sim::CreateCoralNPUV2State;
  using ::mpact::sim::generic::Instruction;
  using ::mpact::sim::riscv::ExceptionCode;
  using ::mpact::sim::riscv::RiscVXlen;
  using ::mpact::sim::util::FlatDemandMemory;

  auto memory = std::make_unique<FlatDemandMemory>();
  auto state = CreateCoralNPUV2State("test", RiscVXlen::RV32, memory.get());

  bool trap_called = false;
  ExceptionCode exception_code = ExceptionCode::kIllegalInstruction;

  state->set_on_trap(
      [&](bool, uint64_t, uint64_t code, uint64_t, const Instruction*) -> bool {
        trap_called = true;
        exception_code = static_cast<ExceptionCode>(code);
        return true;
      });

  state->MPause(nullptr);

  EXPECT_TRUE(trap_called);
  EXPECT_EQ(exception_code, ExceptionCode::kBreakpoint);
}

using CoralNPUSimulatorSemihostTest = ::testing::TestWithParam<Architecture>;

TEST_P(CoralNPUSimulatorSemihostTest, TestHelloWordSemihost) {
  using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;
  using HaltReasonValueType = std::underlying_type_t<HaltReason>;
  std::string elf_path = GetElfPath(kHelloSemihostElf);

  if (!HasTohostSymbol(elf_path)) {
    GTEST_SKIP() << "ELF " << elf_path
                 << " does not have 'tohost' symbol, skipping HTIF test.";
    return;
  }

  CoralNPUSimulatorOptions options;
  options.architecture = GetParam();
  options.semihost_htif = true;

  auto simulator = std::make_unique<CoralNPUSimulator>(options);

  ABSL_ASSERT_OK(simulator->LoadProgram(elf_path));

  testing::internal::CaptureStdout();

  // Run the simulation.
  ABSL_ASSERT_OK(simulator->Run());
  ABSL_ASSERT_OK(simulator->Wait());

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, HasSubstr("this looks like atv's magic"));

  // The simulator requests kSemihostHaltRequest when semihosting exits (e.g.
  // via exit syscall). However, this binary exits via mpause, which triggers
  // kUserRequest.
  ASSERT_OK_AND_ASSIGN(HaltReasonValueType halt_reason,
                       simulator->top()->GetLastHaltReason());
  EXPECT_EQ(halt_reason, *HaltReason::kUserRequest);
}

INSTANTIATE_TEST_SUITE_P(Architecture, CoralNPUSimulatorSemihostTest,
                         Values(Architecture::kV2, Architecture::kM3,
                                Architecture::kM4),
                         ArchitectureName);

}  // namespace
