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

#include "sim/coralnpu_v2_instructions.h"

#include <cstdint>
#include <functional>
#include <memory>

#include "sim/coralnpu_v2_state.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/any_invocable.h"
#include "riscv/riscv_i_instructions.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace {

using ::coralnpu::sim::CoralNPUV2Lb;
using ::coralnpu::sim::CoralNPUV2Lbu;
using ::coralnpu::sim::CoralNPUV2Lh;
using ::coralnpu::sim::CoralNPUV2Lhu;
using ::coralnpu::sim::CoralNPUV2Lw;
using ::coralnpu::sim::CoralNPUV2Mpause;
using ::coralnpu::sim::CoralNPUV2Sb;
using ::coralnpu::sim::CoralNPUV2Sh;
using ::coralnpu::sim::CoralNPUV2State;
using ::coralnpu::sim::CoralNPUV2Sw;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV32::RiscVILbChild;
using ::mpact::sim::riscv::RV32::RiscVILbuChild;
using ::mpact::sim::riscv::RV32::RiscVILhChild;
using ::mpact::sim::riscv::RV32::RiscVILhuChild;
using ::mpact::sim::riscv::RV32::RiscVILwChild;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryInterface;

constexpr uint32_t kGoodLsuAddress = 0x00014000;
constexpr uint32_t kBadLsuAddress = 0x100;
constexpr uint32_t kTestWord = 0xf0f0a5a5;
constexpr uint16_t kTestHalfWord = 0xa5a5;
constexpr uint8_t kTestByte = 0xa5;
constexpr uint32_t kLsuAccessStartAddress = 0x00010000;
constexpr uint32_t kLsuAccessLength = 0x8000;

class CoralNPUV2InstructionTest : public ::testing::Test {
 public:
  using SemanticFunction = std::function<void(Instruction*)>;

 protected:
  void SetUp() override {
    memory_ = std::make_unique<FlatDemandMemory>();
    state_ = std::make_unique<CoralNPUV2State>("CoralNPUV2", RiscVXlen::RV32,
                                               memory_.get());
    state_->AddLsuAccessRange(kLsuAccessStartAddress, kLsuAccessLength);

    state_->AddMpauseHandler([this](const Instruction*) -> bool {
      was_mpause_handler_called_ = true;
      return true;
    });

    state_->set_on_trap(
        [this](bool, uint64_t, uint64_t, uint64_t, const Instruction*) -> bool {
          was_trap_handler_called_ = true;
          return false;
        });

    x1_reg_ = std::make_unique<RV32Register>(state_.get(), "x1");
    x2_reg_ = std::make_unique<RV32Register>(state_.get(), "x2");
  }
  void AttachLoadChildInstruction(Instruction*, SemanticFunction);
  template <typename T>
  void SetMemoryContents(uint32_t address, T data);

  std::unique_ptr<Instruction> CreateLoadInstruction(
      SemanticFunction parent_semantic_function,
      SemanticFunction child_semantic_function);
  std::unique_ptr<Instruction> CreateStoreInstruction(
      SemanticFunction parent_semantic_function);
  uint32_t GetXRegValue(RV32Register* reg) {
    return reg->data_buffer()->Get<uint32_t>(/*index=*/0);
  }
  template <typename T>
  T GetMemoryContents(uint32_t address) {
    DataBuffer* mem_db = state_->db_factory()->Allocate<T>(/*size=*/1);
    memory_->Load(address, mem_db, /*inst=*/nullptr, /*context=*/nullptr);
    T mem_value = mem_db->Get<T>(/*index=*/0);
    mem_db->DecRef();
    return mem_value;
  }
  std::unique_ptr<MemoryInterface> memory_;
  std::unique_ptr<CoralNPUV2State> state_;
  std::unique_ptr<RV32Register> x1_reg_;
  std::unique_ptr<RV32Register> x2_reg_;
  std::unique_ptr<Instruction> child_inst_;
  bool was_mpause_handler_called_ = false;
  bool was_trap_handler_called_ = false;
};

void CoralNPUV2InstructionTest::AttachLoadChildInstruction(
    Instruction* parent, SemanticFunction semantic_function) {
  child_inst_ = std::make_unique<Instruction>(parent->address(), state_.get());
  child_inst_->set_size(parent->size());
  child_inst_->set_semantic_function(semantic_function);
  parent->AppendChild(child_inst_.get());
}

template <typename T>
void CoralNPUV2InstructionTest::SetMemoryContents(uint32_t address, T data) {
  DataBuffer* db = state_->db_factory()->Allocate<T>(1);
  db->Set<T>(/*index=*/0, data);
  memory_->Store(address, db);
  db->DecRef();
}

std::unique_ptr<Instruction> CoralNPUV2InstructionTest::CreateLoadInstruction(
    SemanticFunction parent_semantic_function,
    SemanticFunction child_semantic_function) {
  // Source operand 1: x register containing the base address to load.
  // Source operand 2: immediate operand containing the offset to load.
  // Destination operand 1: x register to store the loaded data.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(parent_semantic_function);
  inst->AppendSource(x1_reg_->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(0));
  AttachLoadChildInstruction(inst.get(), child_semantic_function);
  child_inst_->AppendDestination(x2_reg_->CreateDestinationOperand(0));
  return inst;
}

std::unique_ptr<Instruction> CoralNPUV2InstructionTest::CreateStoreInstruction(
    SemanticFunction parent_semantic_function) {
  // Source operand 1: x register containing the base address to store.
  // Source operand 2: immediate operand containing the offset to store.
  // Source operand 3: x register containing the data to store.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(parent_semantic_function);
  inst->AppendSource(x1_reg_->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(0));
  inst->AppendSource(x2_reg_->CreateSourceOperand());
  return inst;
}

TEST_F(CoralNPUV2InstructionTest, TestMPause) {
  // Create a test instruction and execute it.
  auto inst = std::make_unique<Instruction>(/*address=*/0, state_.get());
  inst->set_size(4);
  inst->set_semantic_function(CoralNPUV2Mpause);
  inst->Execute(/*context=*/nullptr);

  // Verify that the mpause test handler was called.
  EXPECT_TRUE(was_mpause_handler_called_);
}

TEST_F(CoralNPUV2InstructionTest, TestLwGoodAccess) {
  SetMemoryContents(kGoodLsuAddress, kTestWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the destination register contains the test data from memory.
  EXPECT_EQ(GetXRegValue(x2_reg_.get()), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, TestLwBadAccess) {
  SetMemoryContents(kBadLsuAddress, kTestWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the destination register does not contain the test data.
  EXPECT_NE(GetXRegValue(x2_reg_.get()), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, TestSwGoodAccess) {
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);
  x2_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kTestWord);

  // Create a test store word (sw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sw);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the memory contents were updated with the test data.
  EXPECT_EQ(GetMemoryContents<uint32_t>(kGoodLsuAddress), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, TestSwBadAccess) {
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);
  x2_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kTestWord);

  // Create a test store word (sw) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sw);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the memory contents were not updated since the access was
  // invalid.
  EXPECT_NE(GetMemoryContents<uint32_t>(kGoodLsuAddress), kTestWord);
}

TEST_F(CoralNPUV2InstructionTest, TestLhGoodAccess) {
  SetMemoryContents(kGoodLsuAddress, kTestHalfWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);

  // Create a test load half word (lh) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lh, RiscVILhChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value =
      (kTestHalfWord & 0x8000) ? 0xffff'0000 | kTestHalfWord : kTestHalfWord;
  EXPECT_EQ(GetXRegValue(x2_reg_.get()), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, TestLhBadAccess) {
  SetMemoryContents(kBadLsuAddress, kTestHalfWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);

  // Create a test load half word (lh) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lh, RiscVILhChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the destination register does not contain the test data.
  uint32_t unwanted_register_value =
      (kTestHalfWord & 0x8000) ? 0xffff'0000 | kTestHalfWord : kTestHalfWord;
  EXPECT_NE(GetXRegValue(x2_reg_.get()), unwanted_register_value);
}

TEST_F(CoralNPUV2InstructionTest, TestShGoodAccess) {
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);
  x2_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kTestHalfWord);

  // Create a test store half word (sh) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sh);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the memory contents were updated with the register contents.
  EXPECT_EQ(GetMemoryContents<uint16_t>(kGoodLsuAddress), kTestHalfWord);
}

TEST_F(CoralNPUV2InstructionTest, TestShBadAccess) {
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);
  x2_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kTestHalfWord);

  // Create a test store half word (sh) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sh);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the memory contents were not updated with the register contents
  // since the access is invalid.
  EXPECT_NE(GetMemoryContents<uint16_t>(kBadLsuAddress), kTestHalfWord);
}

TEST_F(CoralNPUV2InstructionTest, TestLhuGoodAccess) {
  SetMemoryContents(kGoodLsuAddress, kTestHalfWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);

  // Create a test load unsigned half word (lhu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lhu, RiscVILhuChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value = static_cast<uint32_t>(kTestHalfWord);
  EXPECT_EQ(GetXRegValue(x2_reg_.get()), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, TestLhuBadAccess) {
  SetMemoryContents(kBadLsuAddress, kTestHalfWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);

  // Create a test load unsigned half word (lhu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lhu, RiscVILhuChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the destination register does not contain the test data since
  // the access is invalid.
  uint32_t unwanted_register_value = static_cast<uint32_t>(kTestHalfWord);
  EXPECT_NE(GetXRegValue(x2_reg_.get()), unwanted_register_value);
}

TEST_F(CoralNPUV2InstructionTest, TestLbGoodAccess) {
  SetMemoryContents(kGoodLsuAddress, kTestByte);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);

  // Create a test load byte (lb) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lb, RiscVILbChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value =
      (kTestByte & 0x80) ? 0xffff'ff00 | kTestByte : kTestByte;
  EXPECT_EQ(GetXRegValue(x2_reg_.get()), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, TestLbBadAccess) {
  SetMemoryContents(kBadLsuAddress, kTestByte);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);

  // Create a test load byte (lb) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lb, RiscVILbChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the destination register does not contain the test data.
  uint32_t unwanted_register_value =
      (kTestByte & 0x80) ? 0xffff'ff00 | kTestByte : kTestByte;
  EXPECT_NE(GetXRegValue(x2_reg_.get()), unwanted_register_value);
}

TEST_F(CoralNPUV2InstructionTest, TestSbGoodAccess) {
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);
  x2_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kTestByte);

  // Create a test store byte (sb) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sb);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the memory contents were updated with the register contents.
  EXPECT_EQ(GetMemoryContents<uint8_t>(kGoodLsuAddress), kTestByte);
}

TEST_F(CoralNPUV2InstructionTest, TestSbBadAccess) {
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);
  x2_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kTestByte);

  // Create a test store byte (sb) instruction and execute it.
  std::unique_ptr<Instruction> inst = CreateStoreInstruction(CoralNPUV2Sb);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the memory contents were not updated with the register contents
  // since the access was invalid.
  EXPECT_NE(GetMemoryContents<uint8_t>(kBadLsuAddress), kTestByte);
}

TEST_F(CoralNPUV2InstructionTest, TestLbuGoodAccess) {
  SetMemoryContents(kGoodLsuAddress, kTestByte);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kGoodLsuAddress);

  // Create a test load unsigned byte (lbu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lbu, RiscVILbuChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that theres no trap for accessing a valid address.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the destination register contains the test data from memory.
  uint32_t expected_value = static_cast<uint32_t>(kTestByte);
  EXPECT_EQ(GetXRegValue(x2_reg_.get()), expected_value);
}

TEST_F(CoralNPUV2InstructionTest, TestLbuBadAccess) {
  SetMemoryContents(kBadLsuAddress, kTestByte);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, kBadLsuAddress);

  // Create a test load unsigned byte (lbu) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lbu, RiscVILbuChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the destination register does not contain the test data.
  uint32_t unwanted_register_value = static_cast<uint32_t>(kTestByte);
  EXPECT_NE(GetXRegValue(x2_reg_.get()), unwanted_register_value);
}

TEST_F(CoralNPUV2InstructionTest, TestNullState) {
  auto inst = std::make_unique<Instruction>(/*address=*/0, /*state=*/nullptr);
  inst->set_size(4);
  inst->AppendSource(x1_reg_->CreateSourceOperand());
  inst->AppendSource(new ImmediateOperand<int32_t>(0));
  inst->set_semantic_function(CoralNPUV2Lw);
  // Make sure that the execution does not crash when state is nullptr.
  inst->Execute(/*context=*/nullptr);
}

// Test with an address where the access spans across the valid LSU range
// boundary.
TEST_F(CoralNPUV2InstructionTest, TestLwBadAccessBoundaryConditionStart) {
  const uint32_t test_address = kLsuAccessStartAddress - 1;
  SetMemoryContents(test_address, kTestWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, test_address);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the destination register does not contain the test data.
  EXPECT_NE(GetXRegValue(x2_reg_.get()), kTestWord);
}

// Test with an address where the access spans across the valid LSU range
// boundary.
TEST_F(CoralNPUV2InstructionTest, TestLwBadAccessBoundaryConditionEnd) {
  const uint32_t test_address = kLsuAccessStartAddress + kLsuAccessLength - 3;
  SetMemoryContents(test_address, kTestWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, test_address);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that a trap was triggered for access an address outside the allowed
  // ranges.
  EXPECT_TRUE(was_trap_handler_called_);
  // Verify that the destination register does not contain the test data.
  EXPECT_NE(GetXRegValue(x2_reg_.get()), kTestWord);
}

// Test load word at the end of valid LSU range.
TEST_F(CoralNPUV2InstructionTest, TestLwGoodAccessBoundaryConditionEnd) {
  const uint32_t test_address = kLsuAccessStartAddress + kLsuAccessLength - 4;
  SetMemoryContents(test_address, kTestWord);
  x1_reg_->data_buffer()->Set<uint32_t>(/*index=*/0, test_address);

  // Create a test load word (lw) instruction and execute it.
  std::unique_ptr<Instruction> inst =
      CreateLoadInstruction(CoralNPUV2Lw, RiscVILwChild);
  inst->Execute(/*context=*/nullptr);

  // Verify that no trap was triggered for access an address till the end of
  // allowed range.
  EXPECT_FALSE(was_trap_handler_called_);
  // Verify that the destination register does contain the test data.
  EXPECT_EQ(GetXRegValue(x2_reg_.get()), kTestWord);
}

}  // namespace
