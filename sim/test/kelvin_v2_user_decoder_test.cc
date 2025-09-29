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

#include "sim/kelvin_v2_encoding.h"
#include "sim/kelvin_v2_enums.h"
#include "sim/kelvin_v2_state.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/memory/memory.h"
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

class KelvinV2UserDecoderFixture : public ::testing::Test {
 public:
  void SetUp() override {
    memory_ = std::make_unique<FlatDemandMemory>();
    state_ = std::make_unique<KelvinV2State>("KelvinV2", RiscVXlen::RV32,
                                             memory_.get());
    decoder_ =
        std::make_unique<KelvinV2UserDecoder>(state_.get(), memory_.get());
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
