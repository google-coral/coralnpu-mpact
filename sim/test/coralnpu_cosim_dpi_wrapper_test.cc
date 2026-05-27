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

#include <cstdint>
#include <string>
#include <vector>

#include "sim/coralnpu_architecture.h"
#include "sim/cosim/coralnpu_cosim_dpi.h"
#include "sim/test/align_test_generated.h"
#include "sim/test/frm_test_generated.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "external/svdpi_h_file/file/svdpi.h"

namespace {

constexpr uint32_t kLoadImmediateToX5 = 0b11011110101011011011'00101'0110111;
constexpr uint32_t kFmvX5ToF5 = 0b1111000'00000'00101'000'00101'1010011;
constexpr uint32_t kVsetivli_e32_m1 = 0b11'0011010000'01111'111'00110'1010111;
constexpr uint32_t kVmvX5ToV1 = 0b010111'1'00000'00101'100'00001'1010111;
constexpr uint32_t kAddImmediateToX5_2047 =
    0b011111111111'00101'000'00101'0010011;
constexpr uint32_t kAddImmediateToX5_1776 =
    0b011011110000'00101'000'00101'0010011;
constexpr uint32_t kExpectedX5Value = 0xdeadbeef;
constexpr uint32_t kNopInstruction = 0x00000013;  // x0 = x0 + 0 (nop)
constexpr uint32_t kMpause = 0b000'0100'00000'00000'000'00000'111'0011;
constexpr uint32_t kStoreX1ToX2 = 0b0000000'00001'00010'010'00000'0100011;

class CosimFixture : public ::testing::Test {
 public:
  CosimFixture() { mpact_init(); }
  ~CosimFixture() override { mpact_fini(); }

  int add_test_values_to_x5() {
    int status = 0;
    status = mpact_step_wrapper(kLoadImmediateToX5);
    if (status != 0) {
      return status;
    }
    status = mpact_step_wrapper(kAddImmediateToX5_2047);
    if (status != 0) {
      return status;
    }
    status = mpact_step_wrapper(kAddImmediateToX5_1776);
    return status;
  }

  int mpact_step_wrapper(uint32_t instruction) {
    int status = 0;
    svLogicVecVal instruction_struct;
    instruction_struct.aval = instruction;
    instruction_struct.bval = 0;
    status = mpact_step(&instruction_struct);
    return status;
  }
};

TEST_F(CosimFixture, GetPc) {
  uint32_t pc_value = 1;
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_EQ(pc_value, 0);
}

TEST_F(CosimFixture, GetPcAfterStep) {
  uint32_t pc_value = 1;
  EXPECT_EQ(mpact_step_wrapper(kNopInstruction), 0);
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_EQ(pc_value, 4);
}

TEST_F(CosimFixture, GetPcAfterReset) {
  uint32_t pc_value = 1;
  EXPECT_EQ(mpact_step_wrapper(kNopInstruction), 0);
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_NE(pc_value, 0);
  EXPECT_EQ(mpact_reset(), 0);
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_EQ(pc_value, 0);
}

TEST_F(CosimFixture, GetGpr) {
  uint32_t gpr_value = 1;
  EXPECT_EQ(mpact_get_register("x5", &gpr_value), 0);
  EXPECT_EQ(gpr_value, 0);
  EXPECT_EQ(add_test_values_to_x5(), 0);
  EXPECT_EQ(mpact_get_register("x5", &gpr_value), 0);
  EXPECT_EQ(gpr_value, kExpectedX5Value);
}

TEST_F(CosimFixture, GetMcycleCsr) {
  uint32_t mcycle_value = 12345;
  EXPECT_EQ(mpact_get_register("mcycle", &mcycle_value), 0);
  EXPECT_EQ(mcycle_value, 0);
  EXPECT_EQ(mpact_step_wrapper(kNopInstruction), 0);
  EXPECT_EQ(mpact_get_register("mcycle", &mcycle_value), 0);
  EXPECT_EQ(mcycle_value, 1);
}

TEST_F(CosimFixture, GetFpr) {
  uint32_t gpr_value = 1;
  uint32_t fpr_value = 1;
  EXPECT_EQ(mpact_get_register("x5", &gpr_value), 0);
  EXPECT_EQ(gpr_value, 0);
  EXPECT_EQ(mpact_get_register("f5", &fpr_value), 0);
  EXPECT_EQ(fpr_value, 0);
  EXPECT_EQ(add_test_values_to_x5(), 0);
  EXPECT_EQ(mpact_step_wrapper(kFmvX5ToF5), 0);
  EXPECT_EQ(mpact_get_register("f5", &fpr_value), 0);
  EXPECT_EQ(fpr_value, kExpectedX5Value);
}

TEST_F(CosimFixture, GetVectorRegister) {
  uint32_t x5_val = 0;
  EXPECT_EQ(add_test_values_to_x5(), 0);
  EXPECT_EQ(mpact_get_register("x5", &x5_val), 0);
  EXPECT_EQ(x5_val, kExpectedX5Value);
  // Configure the vector unit for 4 x 32 bit elements.
  EXPECT_EQ(mpact_step_wrapper(kVsetivli_e32_m1), 0);
  // Move the value in x5 into v1.
  EXPECT_EQ(mpact_step_wrapper(kVmvX5ToV1), 0);

  svLogicVecVal result[32];
  EXPECT_EQ(mpact_get_vector_register("v1", result), 0);

  std::vector<uint32_t> expected_avals(4, kExpectedX5Value);
  std::vector<uint32_t> result_avals;
  for (int i = 0; i < 4; ++i) {
    result_avals.push_back(result[i].aval);
    // Verilator doesn't support 4 state logic values. bval != 0 represent
    // states `x` and `z` we only expect `0` and `1` states.
    EXPECT_EQ(result[i].bval, 0);
  }
  EXPECT_EQ(result_avals, expected_avals);
}

TEST_F(CosimFixture, GetVectorRegisterErrors_NullValue) {
  EXPECT_NE(mpact_get_vector_register("v1", nullptr), 0);
}

TEST_F(CosimFixture, GetVectorRegisterErrors_NullRegName) {
  svLogicVecVal result[32];
  EXPECT_NE(mpact_get_vector_register(nullptr, result), 0);
}

TEST_F(CosimFixture, GetVectorRegisterErrors_HandleNotInitialized) {
  mpact_fini();  // Reset the handle to test the uninitialized case.
  svLogicVecVal result[32];
  EXPECT_NE(mpact_get_vector_register("v1", result), 0);
}

TEST_F(CosimFixture, GetPcWithCustomConfig) {
  uint32_t test_itcm_start_address = 0x4000;
  sim_config_t config_data = {
      .itcm_start_address = test_itcm_start_address,
      .itcm_length = 0x100000,
      .initial_misa_value = 0x0,
      .architecture = kCoralNPUV2,
  };
  EXPECT_EQ(mpact_config(&config_data), 0);
  uint32_t pc_value = 1;
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_EQ(pc_value, test_itcm_start_address);

  // This should work if the memory region was correctly updated.
  // If not, WriteMemory at test_itcm_start_address might work
  // (FlatDemandMemory), but Step(1) will trigger a fetch that fails permission
  // check and traps.
  EXPECT_EQ(mpact_step_wrapper(kNopInstruction), 0);
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_EQ(pc_value, test_itcm_start_address + 4);
}

TEST_F(CosimFixture, GetPcWithM3Config) {
  uint32_t test_itcm_start_address = 0x4000;
  sim_config_t config_data = {
      .itcm_start_address = test_itcm_start_address,
      .itcm_length = 0x100000,
      .initial_misa_value = 0x0,
      .architecture = kCoralNPUM3,
  };
  EXPECT_EQ(mpact_config(&config_data), 0);
  coralnpu_architecture_t arch = kCoralNPUV2;
  EXPECT_EQ(mpact_get_architecture(&arch), 0);
  EXPECT_EQ(arch, kCoralNPUM3);

  uint32_t pc_value = 1;
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_EQ(pc_value, test_itcm_start_address);

  EXPECT_EQ(mpact_step_wrapper(kNopInstruction), 0);
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  EXPECT_EQ(pc_value, test_itcm_start_address + 4);
}

// Verify that configuring ITCM does not affect DTCM.
TEST_F(CosimFixture, TestDtcmAccessAfterCustomItcmConfig) {
  uint32_t test_itcm_start_address = 0x4000;
  sim_config_t config_data = {
      .itcm_start_address = test_itcm_start_address,
      .itcm_length = 0x1000,
      .initial_misa_value = 0x40201120,
  };
  config_data.architecture = kCoralNPUV2;
  EXPECT_EQ(mpact_config(&config_data), 0);

  // Try to write to the default DTCM address (0x10000).
  // This should work if DTCM was not moved.
  uint32_t pc_value = 0;
  ASSERT_EQ(mpact_set_register("x1", 0x01234567), 0);
  ASSERT_EQ(mpact_set_register("x2", 0x10000), 0);
  ASSERT_EQ(mpact_step_wrapper(kStoreX1ToX2), 0);
  ASSERT_EQ(mpact_get_register("pc", &pc_value), 0);

  // The program counter will only go to the next instruction if the store
  // instruction didn't trigger a trap.
  EXPECT_EQ(pc_value, test_itcm_start_address + 4);
}

// Without calling mpact_config(), the simulator should use the default
// DTCM configuration.
TEST_F(CosimFixture, TestDtcmWriteWithLazyInitialization) {
  uint32_t pc_value = 0;
  ASSERT_EQ(mpact_set_register("x1", 0x01234567), 0);
  ASSERT_EQ(mpact_set_register("x2", 0x10000), 0);
  ASSERT_EQ(mpact_step_wrapper(kStoreX1ToX2), 0);
  ASSERT_EQ(mpact_get_register("pc", &pc_value), 0);

  // The program counter will only go to the next instruction if the store
  // instruction didn't trigger a trap.
  EXPECT_EQ(pc_value, 4);
}

// Verifies that the simulator architecture configuration (V2 vs M3) is
// correctly set and observable via the DPI. This test satisfies mutation
// testing requirements by ensuring that changes to the architecture assignment
// logic are detected.
TEST_F(CosimFixture, ArchitectureDifferentiation) {
  coralnpu_architecture_t arch = kCoralNPUM3;  // Initialize to wrong value.

  // Default architecture is V2.
  EXPECT_EQ(mpact_get_architecture(&arch), 0);
  EXPECT_EQ(arch, kCoralNPUV2);

  // Configure for M3.
  sim_config_t config_data = {
      .itcm_start_address = 0x0,
      .itcm_length = 0x10000,
      .architecture = kCoralNPUM3,
  };
  EXPECT_EQ(mpact_config(&config_data), 0);
  EXPECT_EQ(mpact_get_architecture(&arch), 0);
  EXPECT_EQ(arch, kCoralNPUM3);

  // Configure back to V2.
  mpact_fini();
  mpact_init();
  config_data.architecture = kCoralNPUV2;
  EXPECT_EQ(mpact_config(&config_data), 0);
  EXPECT_EQ(mpact_get_architecture(&arch), 0);
  EXPECT_EQ(arch, kCoralNPUV2);
}

TEST_F(CosimFixture, LoadProgramTest) {
  // Use the same ELF as the simulator tests.
  std::string elf_path =
      "sim/test/testfiles/"
      "coralnpu_v2_rvv_add_intrinsic.elf";

  // Initialize with architecture V2.
  sim_config_t config_data = {
      .itcm_start_address = 0x0,
      .itcm_length = 0x1000000,
      .initial_misa_value = 0x40201120,
      .architecture = kCoralNPUV2,
  };
  ASSERT_EQ(mpact_config(&config_data), 0);

  ASSERT_EQ(mpact_load_program(elf_path.c_str()), 0);

  uint32_t pc_value = 1;
  EXPECT_EQ(mpact_get_register("pc", &pc_value), 0);
  // The entry point for this ELF is 0x0.
  EXPECT_EQ(pc_value, 0);
}

// Test that the cosim wrapper can run the align_test elf program. This program
// makes use of external memory.
TEST_F(CosimFixture, AlignTest) {
  using InstructionData = coralnpu::sim::test_data::align_test::InstructionData;
  std::vector<InstructionData> instructions =
      coralnpu::sim::test_data::align_test::GetInstructions();
  absl::flat_hash_map<uint32_t, uint32_t> instruction_map;
  for (const InstructionData& instruction : instructions) {
    instruction_map[instruction.address] = instruction.instruction;
  }
  constexpr uint32_t kEntryPoint = 0x254;
  constexpr uint32_t kExitAddress = 0x1dc;
  constexpr uint32_t kTrapHandlerAddress = 0x350;
  constexpr uint32_t kExtMemStartAddress = 0x2000'0000;
  constexpr uint32_t kProgramExtMemPointerAddress = 0x0001'0000;
  constexpr int kMaxIterations = 10'000'000;

  sim_config_t config_data = {
      .itcm_start_address = 0x0,
      .itcm_length = 0x2000,
      .initial_misa_value = 0x40201120,
  };
  config_data.architecture = kCoralNPUV2;
  ASSERT_EQ(mpact_config(&config_data), 0);
  ASSERT_EQ(mpact_add_load_store_range(0x10000, 0x8000), 0);
  ASSERT_EQ(mpact_add_load_store_range(kExtMemStartAddress, 0x0040'0000), 0);

  // Set the variables in the elf .data section.
  ASSERT_EQ(mpact_set_register("pc", 0x800), 0);
  ASSERT_EQ(mpact_set_register("x1", kExtMemStartAddress), 0);
  ASSERT_EQ(mpact_set_register("x2", kProgramExtMemPointerAddress), 0);
  ASSERT_EQ(mpact_step_wrapper(kStoreX1ToX2), 0);
  ASSERT_EQ(mpact_set_register("x1", 0x0001'0020), 0);
  ASSERT_EQ(mpact_set_register("x2", 0x0001'0004), 0);
  ASSERT_EQ(mpact_step_wrapper(kStoreX1ToX2), 0);

  ASSERT_EQ(mpact_set_register("pc", kEntryPoint), 0);
  uint32_t instruction_data = 0;
  uint32_t pc_value = 0;
  uint32_t pc_after_step = 0;
  for (int i = 0; i < kMaxIterations; i++) {
    ASSERT_EQ(mpact_get_register("pc", &pc_value), 0);
    instruction_data = instruction_map[pc_value];
    ASSERT_EQ(mpact_step_wrapper(instruction_data), 0);
    ASSERT_EQ(mpact_get_register("pc", &pc_after_step), 0);
    ASSERT_NE(pc_after_step, kTrapHandlerAddress)
        << "an unexpected trap happened.";
    if (pc_value == kExitAddress || instruction_data == kMpause) {
      return;
    }
  }
  FAIL() << "Test did not exit. final pc: "
         << absl::StrFormat("0x%08x", pc_value);
}

// Test that the cosim wrapper can run the frm_test elf program. This program
// does a LOAD from the ITCM range.
TEST_F(CosimFixture, FrmTest) {
  using InstructionData = coralnpu::sim::test_data::frm_test::InstructionData;
  std::vector<InstructionData> instructions =
      coralnpu::sim::test_data::frm_test::GetInstructions();
  absl::flat_hash_map<uint32_t, uint32_t> instruction_map;
  for (const InstructionData& instruction : instructions) {
    instruction_map[instruction.address] = instruction.instruction;
  }
  constexpr uint32_t kEntryPoint = 0;
  constexpr uint32_t kLoopAddress = 0x000000f0;
  constexpr uint32_t kTrapHandlerAddress = 0x000000f4;
  constexpr int kMaxIterations = 10'000'000;

  sim_config_t config_data = {
      .itcm_start_address = 0x0,
      .itcm_length = 0x2000,
      .initial_misa_value = 0x40201120,
  };
  config_data.architecture = kCoralNPUV2;
  ASSERT_EQ(mpact_config(&config_data), 0);
  ASSERT_EQ(mpact_add_load_store_range(0x10000, 0x8000), 0);

  ASSERT_EQ(mpact_set_register("pc", kEntryPoint), 0);
  uint32_t instruction_data = 0;
  uint32_t pc_value = 0;
  uint32_t pc_after_step = 0;
  for (int i = 0; i < kMaxIterations; i++) {
    ASSERT_EQ(mpact_get_register("pc", &pc_value), 0);
    instruction_data = instruction_map[pc_value];
    ASSERT_EQ(mpact_step_wrapper(instruction_data), 0);
    ASSERT_EQ(mpact_get_register("pc", &pc_after_step), 0);
    ASSERT_NE(pc_after_step, kTrapHandlerAddress)
        << "an unexpected trap happened.";
    if (pc_value == kLoopAddress || instruction_data == kMpause) {
      return;
    }
  }
  FAIL() << "Test did not exit. final pc: "
         << absl::StrFormat("0x%08x", pc_value);
}
}  // namespace
