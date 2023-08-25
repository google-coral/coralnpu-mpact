#include "sim/kelvin_vector_convolution_instructions.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "sim/test/kelvin_vector_instructions_test_base.h"
#include "sim/test/testfiles/kelvin_vector_convolution_testdata.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"

namespace {

using mpact::sim::generic::Instruction;

// Semantic functions.
using kelvin::sim::KelvinVConv;

class KelvinVectorConvolutionInstructionsTest
    : public kelvin::sim::test::KelvinVectorInstructionsTestBase {
 protected:
  void ConvolutionTestHelper(const kelvin::sim::vconv_cmd_t vconv_cmd,
                             bool expect_fail = false) {
    constexpr int kVs1 = 0;
    constexpr int kVs3 = 16;
    constexpr int kVd = 48;
    const uint32_t kVLenInByte = state_->vector_length() / 8;
    const uint32_t kVLenInWord = state_->vector_length() / 32;
    // Set vs1 and vs3
    std::vector<uint8_t> vs1_value(kVLenInWord * kVLenInByte);
    auto vs1_span = absl::Span<uint8_t>(vs1_value);
    memcpy(vs1_span.data(), kVConvIn1, sizeof(kVConvIn1));
    std::vector<uint8_t> vs3_value(kVLenInWord * kVLenInByte);
    auto vs3_span = absl::Span<uint8_t>(vs3_value);
    memcpy(vs3_span.data(), kVConvIn2, sizeof(kVConvIn2));
    for (int i = 0; i < kVLenInWord; ++i) {
      auto vs1_name = absl::StrCat("v", kVs1 + i);
      auto vs3_name = absl::StrCat("v", kVs3 + i);
      SetVectorRegisterValues<uint8_t>(
          {{vs1_name, vs1_span.subspan(i * kVLenInByte, kVLenInByte)},
           {vs3_name, vs3_span.subspan(i * kVLenInByte, kVLenInByte)}});
    }
    uint32_t vconv_cmd_value;
    memcpy(&vconv_cmd_value, &vconv_cmd, sizeof(vconv_cmd_value));
    SetRegisterValues<uint32_t>({{kelvin::sim::test::kRs2Name,
                                  static_cast<uint32_t>(vconv_cmd_value)}});

    // Reset accumulation register
    for (int i = 0; i < kVLenInWord; ++i) {
      auto acc_vec = state_->acc_vec(i);
      acc_vec->fill(0);
    }

    // Call VConv twice with the swapped vs1 and vs3
    std::array<InstructionPtr, 2> instructions = {CreateInstruction(),
                                                  CreateInstruction()};
    instructions[0]->set_semantic_function(KelvinVConv);
    AppendVectorRegisterOperands(instructions[0].get(), kVLenInWord,
                                 1 /* src1_widen_factor*/, kVs1, {},
                                 false /* widen_dst*/, {kVd});
    AppendRegisterOperands(instructions[0].get(), {kelvin::sim::test::kRs2Name},
                           {});
    AppendVectorRegisterOperands(instructions[0].get(), kVLenInWord,
                                 1 /* src3_widen_factor*/, kVs3, {},
                                 false /* widen_dst*/, {});

    instructions[1]->set_semantic_function(KelvinVConv);
    AppendVectorRegisterOperands(instructions[1].get(), 1,
                                 kVLenInWord /* src1_widen_factor*/, kVs3, {},
                                 false /* widen_dst*/, {kVd});
    AppendRegisterOperands(instructions[1].get(), {kelvin::sim::test::kRs2Name},
                           {});
    AppendVectorRegisterOperands(instructions[1].get(), 1,
                                 kVLenInWord /* src3_widen_factor*/, kVs1, {},
                                 false /* widen_dst*/, {});
    execution_fail_ = false;
    state_->set_on_trap(trap_call_back_);
    instructions[0]->Execute();
    if (expect_fail) {
      EXPECT_TRUE(execution_fail_);
      return;
    }
    instructions[1]->Execute();
    EXPECT_FALSE(execution_fail_);
    auto result_acc = state_->acc_register();
    for (int i = 0; i < result_acc.size(); ++i) {
      for (int j = 0; j < result_acc[i].size(); ++j) {
        EXPECT_EQ(result_acc[i][j], kVConvOutRef[i][j])
            << absl::StrCat("acc[", i, "][", j, "] != Ref[", i, "][", j, "]");
      }
    }
  }

 private:
  bool execution_fail_;
  std::function<bool(bool, uint64_t, uint64_t, uint64_t, const Instruction *)>
      trap_call_back_ = [this](bool is_interrupt, uint64_t trap_value,
                               uint64_t exception_code, uint64_t epc,
                               const Instruction *instruction) {
        auto code =
            static_cast<mpact::sim::riscv::ExceptionCode>(exception_code);
        if (code == mpact::sim::riscv::ExceptionCode::kIllegalInstruction) {
          this->execution_fail_ = true;
          return true;
        }
        return false;
      };
};

TEST_F(KelvinVectorConvolutionInstructionsTest, VConv) {
  // Set the convolution to have 8 filters (starting from index 0), with the
  // data bias of 86 (unsigned) and the filter bias of 188 (signed).
  kelvin::sim::vconv_cmd_t vconv_cmd{.mode = 0,
                                     .start = 0,
                                     .stop = 7,
                                     .sbias1 = 86,
                                     .sdata1 = false,
                                     .sbias2 = 188,
                                     .sdata2 = true};
  ConvolutionTestHelper(vconv_cmd);
}

TEST_F(KelvinVectorConvolutionInstructionsTest, VConvWrongMode) {
  // Set the convolution to work on 16-bit input/filter (illegal setting).
  kelvin::sim::vconv_cmd_t vconv_cmd{.mode = 1,
                                     .start = 0,
                                     .stop = 7,
                                     .sbias1 = 86,
                                     .sdata1 = false,
                                     .sbias2 = 188,
                                     .sdata2 = true};
  ConvolutionTestHelper(vconv_cmd, true);
}

TEST_F(KelvinVectorConvolutionInstructionsTest, VConvTooLargeStop) {
  // Set the convolution to work on 9 filters (too many filters).
  kelvin::sim::vconv_cmd_t vconv_cmd{.mode = 0,
                                     .start = 0,
                                     .stop = 8,
                                     .sbias1 = 86,
                                     .sdata1 = false,
                                     .sbias2 = 188,
                                     .sdata2 = true};
  ConvolutionTestHelper(vconv_cmd, true);
}

TEST_F(KelvinVectorConvolutionInstructionsTest, VConvWrongStop) {
  // Set the convolution to start from filter 7 and to stop at filter 5 (reverse
  // order).
  kelvin::sim::vconv_cmd_t vconv_cmd{.mode = 0,
                                     .start = 7,
                                     .stop = 5,
                                     .sbias1 = 86,
                                     .sdata1 = false,
                                     .sbias2 = 188,
                                     .sdata2 = true};
  ConvolutionTestHelper(vconv_cmd, true);
}
}  // namespace
