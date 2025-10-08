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

#include "sim/kelvin_v2_user_decoder.h"

#include <cstdint>
#include <memory>
#include <string>

#include "sim/kelvin_v2_encoding.h"
#include "sim/kelvin_v2_enums.h"
#include "sim/kelvin_v2_state.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace {

using ::kelvin::sim::KelvinV2Encoding;
using ::kelvin::sim::KelvinV2State;
using ::kelvin::sim::KelvinV2UserDecoder;
using ::kelvin::sim::isa32_v2::DestOpEnum;
using ::kelvin::sim::isa32_v2::kDestOpNames;
using ::kelvin::sim::isa32_v2::kSourceOpNames;
using ::kelvin::sim::isa32_v2::OpcodeEnum;
using ::kelvin::sim::isa32_v2::SourceOpEnum;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::riscv::Instruction;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryInterface;

// addi x1, x1, 0
constexpr uint32_t kNopAddiInstruction = 0b000000000000'00001'000'00001'0010011;
// vsetivli x0, 4, e32, m1, ta, ma
constexpr uint32_t kVsetivli_e32_m1 = 0b11'0000100000'00100'111'00000'1010111;

class KelvinV2UserDecoderFixture : public ::testing::Test {
 public:
  void SetUp() override {
    memory_ = std::make_unique<FlatDemandMemory>();
    state_ = std::make_unique<KelvinV2State>("KelvinV2", RiscVXlen::RV32,
                                             memory_.get());
    decoder_ =
        std::make_unique<KelvinV2UserDecoder>(state_.get(), memory_.get());
  }

  std::string DoVerifyShiftInstructionSourceOps(uint32_t base_instruction,
                                                OpcodeEnum expected_opcode) {
    std::string failure_string;
    for (int i = 0; i < 32; i++) {
      uint32_t shift_amount = i;
      uint64_t test_address = 4 * i;
      uint32_t instruction_word = base_instruction | (shift_amount << 20);
      DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
      inst_db->Set<uint32_t>(/*index=*/0, instruction_word);
      memory_->Store(test_address, inst_db);

      std::unique_ptr<Instruction> instruction =
          absl::WrapUnique(decoder_->DecodeInstruction(test_address));
      if (instruction->opcode() != *expected_opcode) {
        absl::StrAppend(&failure_string, "Opcode mismatch.",
                        " expected: ", *expected_opcode,
                        " observed: ", instruction->opcode(),
                        " instruction_word: ",
                        absl::StrFormat("%08x", instruction_word), "\n");
      }

      uint32_t observed_shift_amount = instruction->Source(1)->AsInt32(0);
      if (shift_amount != observed_shift_amount) {
        absl::StrAppend(
            &failure_string,
            "Shift amount mismatch for shift amount: ", shift_amount,
            " observed: ", observed_shift_amount,
            " instruction: ", instruction->AsString(),
            " instruction_word: ", absl::StrFormat("%08x", instruction_word),
            "\n");
      }
      inst_db->DecRef();
    }
    return failure_string;
  }

 protected:
  std::unique_ptr<KelvinV2State> state_;
  std::unique_ptr<MemoryInterface> memory_;
  std::unique_ptr<KelvinV2UserDecoder> decoder_;
};

TEST_F(KelvinV2UserDecoderFixture, TestGetNumOpcodes) {
  EXPECT_NE(decoder_->GetNumOpcodes(), 0);
}

TEST_F(KelvinV2UserDecoderFixture, DecodeInstruction) {
  uint64_t test_address = 0;
  DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
  inst_db->Set<uint32_t>(/*index=*/0, kNopAddiInstruction);
  memory_->Store(test_address, inst_db);
  std::unique_ptr<Instruction> instruction =
      absl::WrapUnique(decoder_->DecodeInstruction(test_address));
  EXPECT_NE(instruction.get(), nullptr);
  EXPECT_EQ(instruction->opcode(), *OpcodeEnum::kAddi);
  inst_db->DecRef();
}

TEST_F(KelvinV2UserDecoderFixture, VerifySRLIInstructionSourceOps) {
  // imm[11:5] - Hard coded to 0 for SRLI
  // uimim5 - shift amount
  // rs1 - source register
  // f3 - 101 for srli instruction
  // rd - destination register
  // opcode - 1110011 for srli instruction
  //                         imm[11:5] |uimm5| rs1 |f3 | rd  | opcode
  uint32_t srli_instruction = 0b0000000'00000'11111'101'11111'0010011;
  std::string shift_failures =
      DoVerifyShiftInstructionSourceOps(srli_instruction, OpcodeEnum::kSrli);
  EXPECT_TRUE(shift_failures.empty()) << shift_failures;
}

TEST_F(KelvinV2UserDecoderFixture, VerifySLLIInstructionSourceOps) {
  // imm[11:5] - Hard coded to 0 for SRLI
  // uimim5 - shift amount
  // rs1 - source register
  // f3 - 001 for slli instruction
  // rd - destination register
  // opcode - 1110011 for slli instruction
  //                         imm[11:5] |uimm5| rs1 |f3 | rd  | opcode
  uint32_t slli_instruction = 0b0000000'00000'11111'001'11111'0010011;
  std::string shift_failures =
      DoVerifyShiftInstructionSourceOps(slli_instruction, OpcodeEnum::kSlli);
  EXPECT_TRUE(shift_failures.empty()) << shift_failures;
}

TEST_F(KelvinV2UserDecoderFixture, VerifySRAIInstructionSourceOps) {
  // imm[11:5] - Hard coded to 0b0100000 for SRAI
  // uimim5 - shift amount
  // rs1 - source register
  // f3 - 101 for srai instruction
  // rd - destination register
  // opcode - 1110011 for srai instruction
  //                         imm[11:5] |uimm5| rs1 |f3 | rd  | opcode
  uint32_t srai_instruction = 0b0100000'00000'11111'101'11111'0010011;
  std::string shift_failures =
      DoVerifyShiftInstructionSourceOps(srai_instruction, OpcodeEnum::kSrai);
  EXPECT_TRUE(shift_failures.empty()) << shift_failures;
}

TEST_F(KelvinV2UserDecoderFixture, DecodeVectorInstruction) {
  uint64_t test_address = 0;
  DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
  // vsetivli x0, 4, e32, m1, ta, ma
  inst_db->Set<uint32_t>(/*index=*/0, kVsetivli_e32_m1);
  memory_->Store(test_address, inst_db);
  std::unique_ptr<Instruction> instruction =
      absl::WrapUnique(decoder_->DecodeInstruction(test_address));
  // Verify the vsetivli instruction was correctly decoded.
  EXPECT_NE(instruction.get(), nullptr);
  EXPECT_EQ(instruction->opcode(), *OpcodeEnum::kVsetivli);
  inst_db->DecRef();
}

class KelvinV2EncodingFixture : public ::testing::Test {
 public:
  void SetUp() override {
    memory_ = std::make_unique<FlatDemandMemory>();
    state_ = std::make_unique<KelvinV2State>("KelvinV2", RiscVXlen::RV32,
                                             memory_.get());
    encoding_ = std::make_unique<KelvinV2Encoding>(state_.get());
  }

 protected:
  std::unique_ptr<KelvinV2State> state_;
  std::unique_ptr<MemoryInterface> memory_;
  std::unique_ptr<KelvinV2Encoding> encoding_;
};

TEST_F(KelvinV2EncodingFixture, AllSourceOpsHaveGetters) {
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; i++) {
    EXPECT_TRUE(encoding_->source_op_getters().contains(i))
        << "No source operand for enum value " << i << " (" << kSourceOpNames[i]
        << ")";
  }
}

TEST_F(KelvinV2EncodingFixture, AllDestOpsHaveGetters) {
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; i++) {
    EXPECT_TRUE(encoding_->dest_op_getters().contains(i))
        << "No dest operand for enum value " << i << " (" << kDestOpNames[i]
        << ")";
  }
}

}  // namespace
