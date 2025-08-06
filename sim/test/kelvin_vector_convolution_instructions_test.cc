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

#include "sim/kelvin_vector_convolution_instructions.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

#include "sim/test/kelvin_vector_instructions_test_base.h"
#include "sim/test/testfiles/kelvin_vector_convolution_testdata.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"

namespace {

using mpact::sim::generic::Instruction;

// Semantic functions.
using kelvin::sim::KelvinVConv;
using kelvin::sim::KelvinVDwconv;

class KelvinVectorConvolutionInstructionsTest
    : public kelvin::sim::test::KelvinVectorInstructionsTestBase {
 protected:
  // Write [1-32] into register, accounting for internal dwconv swizzle
  template <typename T>
  void SetRegisterAscending(int reg, T offset) {
    std::vector<T> data(32);
    for (uint32_t i = 0; i < data.size(); i++) {
      uint32_t reg;
      switch ((i >> 3) & 0b11) {
        case 0:
          reg = 0;
          break;
        case 1:
          reg = 2;
          break;
        case 2:
          reg = 1;
          break;
        case 3:
          reg = 3;
          break;
      }
      uint32_t pos = i & 0b111;
      uint32_t target = (pos << 2) | reg;
      data[target] = i + offset;
    }

    auto reg_name = absl::StrCat("v", reg);
    SetVectorRegisterValues<T>({{reg_name, absl::Span<T>(data)}});
  }

  template <typename T>
  void SetRegisterConstant(int reg, T val) {
    std::vector<T> data(32, val);
    auto reg_name = absl::StrCat("v", reg);
    SetVectorRegisterValues<T>({{reg_name, absl::Span<T>(data)}});
  }

  void ResetDwAccumulator() { state_->dw_acc_register().fill(0); }

  template <bool kWriteAcc = true>
  void ExecuteDwconv(bool expect_fail = false) {
    constexpr int kVs1 = 0;
    constexpr int kVs3 = 16;
    constexpr int kVd = 48;
    InstructionPtr instruction = CreateInstruction();
    instruction->set_semantic_function(
        absl::bind_front(KelvinVDwconv, kWriteAcc));
    AppendVectorRegisterOperands(instruction.get(), 1, 9, kVs1, {},
                                 false /* widen_dst*/, {});
    AppendRegisterOperands(instruction.get(), {kelvin::sim::test::kRs2Name},
                           {});
    AppendVectorRegisterOperands(instruction.get(), 1, 3, kVs3, {},
                                 false /* widen_dst*/, {});
    if (kWriteAcc) {
      std::vector<kelvin::sim::test::RegisterBase*> reg_vec;
      for (int i = 0; i < 4; i++) {
        auto reg_name = absl::StrCat("v", kVd + i);
        reg_vec.push_back(
            state_->GetRegister<kelvin::sim::test::RVVectorRegister>(reg_name)
                .first);
      }
      auto* op = new kelvin::sim::test::RV32VectorDestinationOperand(
          absl::Span<kelvin::sim::test::RegisterBase*>(reg_vec), 0,
          absl::StrCat("v", kVd));
      instruction->AppendDestination(op);
    }

    execution_fail_ = false;
    state_->set_on_trap(trap_call_back_);
    instruction->Execute();
    EXPECT_EQ(expect_fail, execution_fail_);
  }

  template <bool kWriteAcc = true>
  void TestAccumulatorAndRegisters(
      std::function<void(int /*index*/, int32_t /*value*/)> f) {
    constexpr int kVd = 48;

    // Check internal accumulator.
    auto acc_vec = state_->dw_acc_register();
    for (int i = 0; i < 32; i++) {
      f(i, acc_vec[i]);
    }

    // Check Registers
    if (kWriteAcc) {
      for (int r = 0; r < 4; r++) {
        auto reg = state_
                       ->GetRegister<kelvin::sim::test::RVVectorRegister>(
                           absl::StrCat("v", kVd + r))
                       .first;
        auto reg_data = reg->data_buffer()->Get<int32_t>();
        for (int elem = 0; elem < 8; elem++) {
          int i = (r * 8) + elem;
          int32_t value = reg_data[elem];
          f(i, value);
        }
      }
    }
  }

  void DepthwiseConvolutionBiasTestHelper(uint32_t sbias1, uint32_t sbias2) {
    constexpr int kVs1 = 0;
    constexpr int kVs3 = 16;

    kelvin::sim::vdwconv_u8_t dwconv_cmd;
    memset(&dwconv_cmd, 0, sizeof(dwconv_cmd));
    dwconv_cmd.sdata1 = 1;
    dwconv_cmd.sdata2 = 1;
    dwconv_cmd.sbias1 = sbias1;
    dwconv_cmd.sbias2 = sbias2;
    uint32_t vdwconv_cmd_value;
    memcpy(&vdwconv_cmd_value, &dwconv_cmd, sizeof(vdwconv_cmd_value));
    SetRegisterValues<uint32_t>(
        {{kelvin::sim::test::kRs2Name, vdwconv_cmd_value}});

    ResetDwAccumulator();
    SetRegisterAscending<int8_t>(kVs1, 1 - sbias1);
    SetRegisterConstant<int8_t>(kVs1 + 1, -sbias1);
    SetRegisterConstant<int8_t>(kVs1 + 2, -sbias1);

    SetRegisterConstant<int8_t>(kVs3, 1 - sbias2);
    SetRegisterConstant<int8_t>(kVs3 + 1, -sbias2);
    SetRegisterConstant<int8_t>(kVs3 + 2, -sbias2);
    ExecuteDwconv();

    TestAccumulatorAndRegisters(
        [](int i, int32_t value) { EXPECT_EQ(i + 1, value); });
  }

  template <typename T, bool kWriteAcc = true>
  void DepthwiseConvolutionRegbaseTestHelper(int regbase, int prev, int curr,
                                             int next) {
    constexpr int kVs1 = 0;
    constexpr int kVs3 = 16;

    kelvin::sim::vdwconv_u8_t dwconv_cmd;
    memset(&dwconv_cmd, 0, sizeof(dwconv_cmd));
    dwconv_cmd.regbase = regbase;
    if (std::is_signed<T>::value) {
      dwconv_cmd.sdata1 = 1;
      dwconv_cmd.sdata2 = 1;
    }
    uint32_t vdwconv_cmd_value;
    memcpy(&vdwconv_cmd_value, &dwconv_cmd, sizeof(vdwconv_cmd_value));
    SetRegisterValues<uint32_t>(
        {{kelvin::sim::test::kRs2Name, vdwconv_cmd_value}});

    // Test prev reg
    {
      ResetDwAccumulator();

      SetRegisterAscending<T>(kVs1 + prev, 1);
      SetRegisterConstant<T>(kVs1 + curr, 0);
      SetRegisterConstant<T>(kVs1 + next, 0);

      SetRegisterConstant<T>(kVs3, 1);
      SetRegisterConstant<T>(kVs3 + 1, 0);
      SetRegisterConstant<T>(kVs3 + 2, 0);

      ExecuteDwconv<kWriteAcc>();
      TestAccumulatorAndRegisters<kWriteAcc>(
          [](int i, int32_t value) { EXPECT_EQ(i + 1, value); });
    }

    // Test curr reg
    {
      ResetDwAccumulator();

      SetRegisterConstant<T>(kVs1 + prev, 0);
      SetRegisterAscending<T>(kVs1 + curr, 1);
      SetRegisterConstant<T>(kVs1 + next, 0);

      SetRegisterConstant<T>(kVs3, 0);
      SetRegisterConstant<T>(kVs3 + 1, 2);
      SetRegisterConstant<T>(kVs3 + 2, 0);

      ExecuteDwconv<kWriteAcc>();
      TestAccumulatorAndRegisters<kWriteAcc>(
          [](int i, int32_t value) { EXPECT_EQ(2 * (i + 1), value); });
    }

    // Test next reg
    {
      ResetDwAccumulator();

      SetRegisterConstant<T>(kVs1 + prev, 0);
      SetRegisterConstant<T>(kVs1 + curr, 0);
      SetRegisterAscending<T>(kVs1 + next, 1);

      SetRegisterConstant<T>(kVs3, 0);
      SetRegisterConstant<T>(kVs3 + 1, 0);
      SetRegisterConstant<T>(kVs3 + 2, 3);

      ExecuteDwconv<kWriteAcc>();
      TestAccumulatorAndRegisters<kWriteAcc>(
          [](int i, int32_t value) { EXPECT_EQ(3 * (i + 1), value); });
    }
  }

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
  std::function<bool(bool, uint64_t, uint64_t, uint64_t, const Instruction*)>
      trap_call_back_ = [this](bool is_interrupt, uint64_t trap_value,
                               uint64_t exception_code, uint64_t epc,
                               const Instruction* instruction) {
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

TEST_F(KelvinVectorConvolutionInstructionsTest, VDwconvRegbase) {
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(0, 0, 1, 2);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(1, 1, 2, 3);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(2, 2, 3, 4);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(3, 3, 4, 5);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(4, 4, 5, 6);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(5, 5, 6, 7);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(6, 6, 7, 8);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(7, 1, 0, 2);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(8, 1, 2, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(9, 3, 4, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(10, 5, 6, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(11, 7, 8, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(12, 2, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(13, 4, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(14, 6, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, true>(15, 8, 0, 1);

  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(0, 0, 1, 2);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(1, 1, 2, 3);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(2, 2, 3, 4);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(3, 3, 4, 5);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(4, 4, 5, 6);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(5, 5, 6, 7);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(6, 6, 7, 8);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(7, 1, 0, 2);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(8, 1, 2, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(9, 3, 4, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(10, 5, 6, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(11, 7, 8, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(12, 2, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(13, 4, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(14, 6, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, true>(15, 8, 0, 1);

  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(0, 0, 1, 2);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(1, 1, 2, 3);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(2, 2, 3, 4);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(3, 3, 4, 5);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(4, 4, 5, 6);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(5, 5, 6, 7);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(6, 6, 7, 8);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(7, 1, 0, 2);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(8, 1, 2, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(9, 3, 4, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(10, 5, 6, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(11, 7, 8, 0);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(12, 2, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(13, 4, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(14, 6, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<uint8_t, false>(15, 8, 0, 1);

  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(0, 0, 1, 2);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(1, 1, 2, 3);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(2, 2, 3, 4);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(3, 3, 4, 5);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(4, 4, 5, 6);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(5, 5, 6, 7);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(6, 6, 7, 8);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(7, 1, 0, 2);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(8, 1, 2, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(9, 3, 4, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(10, 5, 6, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(11, 7, 8, 0);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(12, 2, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(13, 4, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(14, 6, 0, 1);
  DepthwiseConvolutionRegbaseTestHelper<int8_t, false>(15, 8, 0, 1);
}

TEST_F(KelvinVectorConvolutionInstructionsTest, VDwconvSignBiases) {
  DepthwiseConvolutionBiasTestHelper(2, 0);
  DepthwiseConvolutionBiasTestHelper(0, 3);
  DepthwiseConvolutionBiasTestHelper(5, 5);
}

TEST_F(KelvinVectorConvolutionInstructionsTest, VDwconvSparsity1) {
  constexpr int kVs1 = 0;
  constexpr int kVs3 = 16;

  kelvin::sim::vdwconv_u8_t dwconv_cmd;
  memset(&dwconv_cmd, 0, sizeof(dwconv_cmd));
  dwconv_cmd.regbase = 0;
  dwconv_cmd.sparsity = 1;
  uint32_t vdwconv_cmd_value;
  memcpy(&vdwconv_cmd_value, &dwconv_cmd, sizeof(vdwconv_cmd_value));
  SetRegisterValues<uint32_t>(
      {{kelvin::sim::test::kRs2Name, vdwconv_cmd_value}});

  {
    ResetDwAccumulator();

    SetRegisterConstant<uint8_t>(kVs1, 42);
    SetRegisterAscending<uint8_t>(kVs1 + 1, 1);
    SetRegisterConstant<uint8_t>(kVs1 + 2, 0);

    SetRegisterConstant<uint8_t>(kVs3, 1);
    SetRegisterConstant<uint8_t>(kVs3 + 1, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 2, 0);

    ExecuteDwconv();
    TestAccumulatorAndRegisters([](int i, int32_t value) {
      EXPECT_EQ((i % 8 == 0 ? 42 : i), value)
          << "Incorrect value at index " << i;
    });
  }

  {
    ResetDwAccumulator();

    SetRegisterConstant<uint8_t>(kVs1, 0);
    SetRegisterAscending<uint8_t>(kVs1 + 1, 1);
    SetRegisterConstant<uint8_t>(kVs1 + 2, 0);

    SetRegisterConstant<uint8_t>(kVs3, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 1, 1);
    SetRegisterConstant<uint8_t>(kVs3 + 2, 0);

    ExecuteDwconv();
    TestAccumulatorAndRegisters([](int i, int32_t value) {
      EXPECT_EQ(i + 1, value) << "Incorrect value at index " << i;
    });
  }

  {
    ResetDwAccumulator();

    SetRegisterConstant<uint8_t>(kVs1, 0);
    SetRegisterAscending<uint8_t>(kVs1 + 1, 0);
    SetRegisterConstant<uint8_t>(kVs1 + 2, 42);

    SetRegisterConstant<uint8_t>(kVs3, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 1, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 2, 1);

    ExecuteDwconv();
    TestAccumulatorAndRegisters([](int i, int32_t value) {
      EXPECT_EQ((i % 8 == 7 ? 42 : i + 1), value)
          << "Incorrect value at index " << i;
    });
  }
}

TEST_F(KelvinVectorConvolutionInstructionsTest, VDwconvSparsity2) {
  constexpr int kVs1 = 0;
  constexpr int kVs3 = 16;

  kelvin::sim::vdwconv_u8_t dwconv_cmd;
  memset(&dwconv_cmd, 0, sizeof(dwconv_cmd));
  dwconv_cmd.regbase = 0;
  dwconv_cmd.sparsity = 2;
  uint32_t vdwconv_cmd_value;
  memcpy(&vdwconv_cmd_value, &dwconv_cmd, sizeof(vdwconv_cmd_value));
  SetRegisterValues<uint32_t>(
      {{kelvin::sim::test::kRs2Name, vdwconv_cmd_value}});

  {
    ResetDwAccumulator();

    SetRegisterAscending<uint8_t>(kVs1, 1);
    SetRegisterConstant<uint8_t>(kVs1 + 1, 0);
    SetRegisterConstant<uint8_t>(kVs1 + 2, 0);

    SetRegisterConstant<uint8_t>(kVs3, 1);
    SetRegisterConstant<uint8_t>(kVs3 + 1, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 2, 0);

    ExecuteDwconv();
    TestAccumulatorAndRegisters([](int i, int32_t value) {
      EXPECT_EQ(i + 1, value) << "Incorrect value at index " << i;
    });
  }

  {
    ResetDwAccumulator();

    SetRegisterAscending<uint8_t>(kVs1, 0);
    SetRegisterConstant<uint8_t>(kVs1 + 1, 42);
    SetRegisterConstant<uint8_t>(kVs1 + 2, 0);

    SetRegisterConstant<uint8_t>(kVs3, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 1, 1);
    SetRegisterConstant<uint8_t>(kVs3 + 2, 0);

    ExecuteDwconv();
    TestAccumulatorAndRegisters([](int i, int32_t value) {
      EXPECT_EQ((i % 8 == 7 ? 42 : i + 1), value)
          << "Incorrect value at index " << i;
    });
  }

  {
    ResetDwAccumulator();

    SetRegisterAscending<uint8_t>(kVs1, 0);
    SetRegisterConstant<uint8_t>(kVs1 + 1, 42);
    SetRegisterConstant<uint8_t>(kVs1 + 2, 0);

    SetRegisterConstant<uint8_t>(kVs3, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 1, 0);
    SetRegisterConstant<uint8_t>(kVs3 + 2, 1);

    ExecuteDwconv();
    TestAccumulatorAndRegisters([](int i, int32_t value) {
      if (i % 8 >= 6) {
        EXPECT_EQ(42, value) << "Incorrect value at index " << i;
      } else {
        EXPECT_EQ(i + 2, value) << "Incorrect value at index " << i;
      }
    });
  }
}

TEST_F(KelvinVectorConvolutionInstructionsTest, VDwconvSparsity3) {
  // Sparsity value of 3 is invalid.
  kelvin::sim::vdwconv_u8_t dwconv_cmd;
  memset(&dwconv_cmd, 0, sizeof(dwconv_cmd));
  dwconv_cmd.regbase = 0;
  dwconv_cmd.sparsity = 3;
  uint32_t vdwconv_cmd_value;
  memcpy(&vdwconv_cmd_value, &dwconv_cmd, sizeof(vdwconv_cmd_value));
  SetRegisterValues<uint32_t>(
      {{kelvin::sim::test::kRs2Name, vdwconv_cmd_value}});
  ExecuteDwconv(/* expect_fail */ true);
}

}  // namespace
