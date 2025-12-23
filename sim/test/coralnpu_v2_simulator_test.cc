// Copyright 2025 Google LLC
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

#include "sim/coralnpu_v2_simulator.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "sim/test/coralnpu_v2_rvv_add_intrinsic_generated.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/data_buffer.h"

/* start - new test macros */
#define ABSL_EXPECT_OK(expression) \
  EXPECT_THAT(expression, ::absl_testing::IsOk())
#define ABSL_ASSERT_OK(expression) \
  ASSERT_THAT(expression, ::absl_testing::IsOk())
/* end - new test macros */

/* start - helper test macro */
#define ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  auto status_or = (rexpr); \
  ABSL_ASSERT_OK(status_or); \
  lhs = std::move(status_or).value();
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

namespace {

using ::coralnpu::sim::CoralNPUV2Simulator;
using ::coralnpu::sim::CoralNPUV2SimulatorOptions;
using ::coralnpu::sim::test_data::coralnpu_v2_rvv_add_intrinsic::
    GetInstructions;
using ::mpact::sim::generic::DataBuffer;
using ::testing::Each;
using ::testing::Pointee;
using ::absl_testing::IsOkAndHolds;

class CoralNPUV2SimulatorTest : public ::testing::Test {
 public:
  void SetUp() override {
    simulator_options_.exit_on_ebreak = true;
    simulator_ = std::make_unique<CoralNPUV2Simulator>(simulator_options_);

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
  CoralNPUV2SimulatorOptions simulator_options_;
  std::unique_ptr<CoralNPUV2Simulator> simulator_;
};

TEST_F(CoralNPUV2SimulatorTest, TestIntegerRegisters) {
  for (int i = 0; i < 32; ++i) {
    ABSL_ASSERT_OK(simulator_->WriteRegister(absl::StrCat("x", i), i));
    ASSERT_OK_AND_ASSIGN(uint32_t value,
                         simulator_->ReadRegister(absl::StrCat("x", i)));
    EXPECT_EQ(value, i);
  }
}

TEST_F(CoralNPUV2SimulatorTest, TestFloatingPointRegisters) {
  for (int i = 0; i < 32; ++i) {
    ABSL_ASSERT_OK(simulator_->WriteRegister(absl::StrCat("f", i), i));
    ASSERT_OK_AND_ASSIGN(uint64_t value,
                         simulator_->ReadRegister(absl::StrCat("f", i)));
    EXPECT_EQ(value, i);
  }
}

TEST_F(CoralNPUV2SimulatorTest, TestVectorRegisters) {
  for (int i = 0; i < 32; ++i) {
    ASSERT_OK_AND_ASSIGN(DataBuffer * reg_db, simulator_->GetRegisterDataBuffer(
                                                  absl::StrCat("v", i)));
    EXPECT_NE(reg_db, nullptr);
  }
}

TEST_F(CoralNPUV2SimulatorTest, TestRvvAddIntrinsicMpause) {
  // Run the simulation to the end.
  ABSL_ASSERT_OK(simulator_->Run());
  ABSL_ASSERT_OK(simulator_->Wait());

  // Verify that the simulation stopped after the mpause instruction.
  absl::StatusOr<uint64_t> pc = simulator_->ReadRegister("pc");
  ABSL_ASSERT_OK(pc);
  EXPECT_EQ(*pc, kMpauseAddress + 4);
}

TEST_F(CoralNPUV2SimulatorTest, TestRvvAddIntrinsicReadMemory) {
  // Run the simulation to the end.
  ABSL_ASSERT_OK(simulator_->Run());
  ABSL_ASSERT_OK(simulator_->Wait());

  // This elf is not a self checking test so we will manually verify the output
  // array.
  EXPECT_THAT(ReadOutputArray(), IsOkAndHolds(Pointee(Each(kExpectedOutput))));
}

TEST_F(CoralNPUV2SimulatorTest, TestRvvAddIntrinsicSetBreakpoint) {
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

TEST_F(CoralNPUV2SimulatorTest, TestRvvAddIntrinsicWriteRegister) {
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

}  // namespace
