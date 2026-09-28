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

#include "sim/coralnpu_m4_encoding.h"

#include <cstdint>
#include <memory>

#include "sim/coralnpu_m4_enums.h"
#include "sim/coralnpu_m4_state.h"
#include "googletest/include/gtest/gtest.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

namespace {

using ::coralnpu::sim::CoralNPUM4Encoding;
using ::coralnpu::sim::CreateCoralNPUM4State;
using ::mpact::sim::generic::operator*;  // NOLINT
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::FlatDemandMemory;
using OpcodeEnum = ::coralnpu::sim::isa32_m4::OpcodeEnum;
using SlotEnum = ::coralnpu::sim::isa32_m4::SlotEnum;

// Builds a VConfig (OP-V configuration) instruction word from its fields.
constexpr uint32_t kOpcodeOpV = 0b101'0111;
uint32_t MakeVConfig(uint32_t func7, uint32_t rs2, uint32_t rs1, uint32_t func3,
                     uint32_t rd) {
  return (func7 << 25) | (rs2 << 20) | (rs1 << 15) | (func3 << 12) | (rd << 7) |
         kOpcodeOpV;
}

class CoralNPUM4EncodingTest : public ::testing::Test {
 protected:
  CoralNPUM4EncodingTest()
      : memory_(std::make_unique<FlatDemandMemory>()),
        state_(CreateCoralNPUM4State("test", RiscVXlen::RV32, memory_.get())),
        encoding_(state_.get()) {}

  OpcodeEnum Decode(uint32_t inst_word) {
    encoding_.ParseInstruction(inst_word);
    return encoding_.GetOpcode(SlotEnum::kCoralnpuM4, 0);
  }

  std::unique_ptr<FlatDemandMemory> memory_;
  std::unique_ptr<::coralnpu::sim::CoralNPUM4State> state_;
  CoralNPUM4Encoding encoding_;
};

TEST_F(CoralNPUM4EncodingTest, DecodesMsetmtype) {
  // func7 == 0b100'0001, func3 == 0b111.
  EXPECT_EQ(Decode(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111,
                               /*rd=*/0)),
            OpcodeEnum::kMsetmtype);
}

TEST_F(CoralNPUM4EncodingTest, DecodesMsettn) {
  // func7 == 0b100'0010, rs2 == 0.
  EXPECT_EQ(Decode(MakeVConfig(0b100'0010, /*rs2=*/0, /*rs1=*/1, 0b111,
                               /*rd=*/2)),
            OpcodeEnum::kMsettn);
}

TEST_F(CoralNPUM4EncodingTest, DecodesMsettm) {
  // func7 == 0b100'0010, rs2 == 1.
  EXPECT_EQ(Decode(MakeVConfig(0b100'0010, /*rs2=*/1, /*rs1=*/1, 0b111,
                               /*rd=*/2)),
            OpcodeEnum::kMsettm);
}

TEST_F(CoralNPUM4EncodingTest, DecodesMsettk) {
  // func7 == 0b100'0010, rs2 == 2.
  EXPECT_EQ(Decode(MakeVConfig(0b100'0010, /*rs2=*/2, /*rs1=*/1, 0b111,
                               /*rd=*/2)),
            OpcodeEnum::kMsettk);
}

// A vsetvl (func7 == 0b100'0000) must NOT decode as a matrix config op.
TEST_F(CoralNPUM4EncodingTest, VsetvlIsNotMatrixConfig) {
  OpcodeEnum op = Decode(MakeVConfig(0b100'0000, /*rs2=*/0, /*rs1=*/1, 0b111,
                                     /*rd=*/2));
  EXPECT_NE(op, OpcodeEnum::kMsetmtype);
  EXPECT_NE(op, OpcodeEnum::kMsettn);
  EXPECT_NE(op, OpcodeEnum::kMsettm);
  EXPECT_NE(op, OpcodeEnum::kMsettk);
}

}  // namespace
