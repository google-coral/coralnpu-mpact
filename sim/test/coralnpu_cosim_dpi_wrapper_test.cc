#include <cstdint>
#include <vector>

#include "sim/cosim/coralnpu_cosim_dpi.h"
#include "googletest/include/gtest/gtest.h"
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
constexpr uint32_t kExpectedMisaValue = 0x40201120;

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

TEST_F(CosimFixture, GetMisaCsr) {
  uint32_t misa_value = 0;
  EXPECT_EQ(mpact_get_register("misa", &misa_value), 0);
  EXPECT_EQ(misa_value, kExpectedMisaValue);
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

  svLogicVecVal result[4];
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
  svLogicVecVal result[4];
  EXPECT_NE(mpact_get_vector_register(nullptr, result), 0);
}

TEST_F(CosimFixture, GetVectorRegisterErrors_HandleNotInitialized) {
  mpact_fini();  // Reset the handle to test the uninitialized case.
  svLogicVecVal result[4];
  EXPECT_NE(mpact_get_vector_register("v1", result), 0);
}

}  // namespace
