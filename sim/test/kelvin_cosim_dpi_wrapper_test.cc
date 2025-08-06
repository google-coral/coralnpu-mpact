#include <cstdint>

#include "sim/cosim/kelvin_cosim_dpi.h"
#include "googletest/include/gtest/gtest.h"
#include "external/svdpi_h_file/file/svdpi.h"

namespace {

const uint32_t kLoadImmediateToX5 = 0b11011110101011011011'00101'0110111;
const uint32_t kAddImmediateToX5_2047 = 0b011111111111'00101'000'00101'0010011;
const uint32_t kAddImmediateToX5_1776 = 0b011011110000'00101'000'00101'0010011;
const uint32_t kExpectedX5Value = 0xdeadbeef;
const uint32_t kNopInstruction = 0x00000013;  // x0 = x0 + 0 (nop)
const uint32_t kMcycleCsrAddress = 0xb00;

class CosimFixture : public ::testing::Test {
 public:
  CosimFixture() { mpact_init(); }
  ~CosimFixture() override { mpact_fini(); }
};

TEST_F(CosimFixture, Step) {
  svLogicVecVal instruction;
  instruction.aval = 0x00000000;
  EXPECT_EQ(mpact_step(&instruction), 0);
}

TEST_F(CosimFixture, GetPc) { EXPECT_EQ(mpact_get_pc(), 0); }

TEST_F(CosimFixture, GetPcAfterStep) {
  svLogicVecVal instruction;
  instruction.aval = kNopInstruction;
  EXPECT_EQ(mpact_step(&instruction), 0);
  EXPECT_EQ(mpact_get_pc(), 4);
}

TEST_F(CosimFixture, GetPcAfterReset) {
  svLogicVecVal instruction;
  instruction.aval = kNopInstruction;  // x0 = x0 + 0 (nop)
  EXPECT_EQ(mpact_step(&instruction), 0);
  EXPECT_NE(mpact_get_pc(), 0);
  EXPECT_EQ(mpact_reset(), 0);
  EXPECT_EQ(mpact_get_pc(), 0);
}

TEST_F(CosimFixture, CheckGpr) {
  EXPECT_EQ(mpact_get_gpr(5), 0);
  svLogicVecVal instruction;
  instruction.aval = kLoadImmediateToX5;
  EXPECT_EQ(mpact_step(&instruction), 0);
  instruction.aval = kAddImmediateToX5_2047;
  EXPECT_EQ(mpact_step(&instruction), 0);
  instruction.aval = kAddImmediateToX5_1776;
  EXPECT_EQ(mpact_step(&instruction), 0);
  EXPECT_EQ(mpact_get_gpr(5), kExpectedX5Value);
}

TEST_F(CosimFixture, GetMcycleCsr) {
  EXPECT_EQ(mpact_get_csr(kMcycleCsrAddress), 0);
  svLogicVecVal instruction;
  instruction.aval = kNopInstruction;  // x0 = x0 + 0 (nop)
  EXPECT_EQ(mpact_step(&instruction), 0);
  EXPECT_EQ(mpact_get_csr(kMcycleCsrAddress), 1);
}

}  // namespace
