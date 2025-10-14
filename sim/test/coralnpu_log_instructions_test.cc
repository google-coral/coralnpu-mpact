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

#include <sys/types.h>

#include <array>
#include <cstdint>
#include <string>

#include "sim/coralnpu_instructions.h"
#include "sim/test/coralnpu_vector_instructions_test_base.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/bind_front.h"
#include "mpact/sim/generic/instruction.h"

// Test CoralNPU logging instruction functionality

namespace {

// Semantic function
using coralnpu::sim::CoralNPULogInstruction;

constexpr uint32_t kMemAddress = 0x1000;

class CoralNPULogInstructionsTest
    : public coralnpu::sim::test::CoralNPUVectorInstructionsTestBase {};

TEST_F(CoralNPULogInstructionsTest, SimplePrint) {
  constexpr char kHelloString[] = "Hello World!\n";

  // Initialize memory.
  auto* db = state_->db_factory()->Allocate<char>(sizeof(kHelloString));
  for (int i = 0; i < sizeof(kHelloString); ++i) {
    db->Set<char>(i, kHelloString[i]);
  }
  db->DecRef();

  auto instruction = CreateInstruction();
  state_->StoreMemory(instruction.get(), kMemAddress, db);
  AppendRegisterOperands(instruction.get(), {coralnpu::sim::test::kRs1Name},
                         {});
  SetRegisterValues<uint32_t>({{coralnpu::sim::test::kRs1Name, kMemAddress}});
  instruction->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/0));

  // Execute the instruction and check the stdout.
  testing::internal::CaptureStdout();
  instruction->Execute(nullptr);
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ(kHelloString, stdout_str);
}

TEST_F(CoralNPULogInstructionsTest, PrintUnsignedNumber) {
  constexpr char kFormatString[] = "Hello %u\n";
  constexpr uint32_t kPrintNum = 2200000000;  // a number > INT32_MAX

  // Initialize memory.
  auto* db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();

  std::array<InstructionPtr, 2> instructions = {CreateInstruction(),
                                                CreateInstruction()};

  AppendRegisterOperands(instructions[0].get(), {coralnpu::sim::test::kRs1Name},
                         {});
  instructions[0]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/1));  // scalar log

  // Set the second instruction for the actual print out.
  state_->StoreMemory(instructions[1].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[1].get(), {coralnpu::sim::test::kRs2Name},
                         {});
  instructions[1]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/0));

  SetRegisterValues<uint32_t>({{coralnpu::sim::test::kRs1Name, kPrintNum},
                               {coralnpu::sim::test::kRs2Name, kMemAddress}});

  testing::internal::CaptureStdout();
  for (int i = 0; i < instructions.size(); ++i) {
    instructions[i]->Execute(nullptr);
  }
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello 2200000000\n", stdout_str);
}

TEST_F(CoralNPULogInstructionsTest, PrintSignedNumber) {
  constexpr char kFormatString[] = "Hello %d\n";
  constexpr int32_t kPrintNum = -1337;

  // Initialize memory.
  auto* db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();

  std::array<InstructionPtr, 2> instructions = {CreateInstruction(),
                                                CreateInstruction()};

  AppendRegisterOperands(instructions[0].get(), {coralnpu::sim::test::kRs1Name},
                         {});
  instructions[0]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/1));  // scalar log

  // Set the second instruction for the actual print out.
  state_->StoreMemory(instructions[1].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[1].get(), {coralnpu::sim::test::kRs2Name},
                         {});
  instructions[1]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/0));

  SetRegisterValues<int32_t>({{coralnpu::sim::test::kRs1Name, kPrintNum}});
  SetRegisterValues<uint32_t>({{coralnpu::sim::test::kRs2Name, kMemAddress}});

  testing::internal::CaptureStdout();
  for (int i = 0; i < instructions.size(); ++i) {
    instructions[i]->Execute(nullptr);
  }
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello -1337\n", stdout_str);
}

TEST_F(CoralNPULogInstructionsTest, PrintCharacterStream) {
  constexpr char kFormatString[] = "%s World\n";
  constexpr uint32_t kCharStream[] = {0x6c6c6548, 0x0000006f};  // "Hello"

  // Initialize memory.
  auto* db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();
  std::array<InstructionPtr, 3> instructions = {
      CreateInstruction(), CreateInstruction(), CreateInstruction()};
  AppendRegisterOperands(instructions[0].get(), {coralnpu::sim::test::kRs1Name},
                         {});
  AppendRegisterOperands(instructions[1].get(), {coralnpu::sim::test::kRs2Name},
                         {});
  for (int i = 0; i < 2; ++i) {
    instructions[i]->set_semantic_function(absl::bind_front(
        &CoralNPULogInstruction, /*mode=*/2));  // character log
  }

  constexpr char kRs3Name[] = "x3";
  state_->StoreMemory(instructions[2].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[2].get(), {kRs3Name}, {});
  instructions[2]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/0));

  SetRegisterValues<uint32_t>({{coralnpu::sim::test::kRs1Name, kCharStream[0]},
                               {coralnpu::sim::test::kRs2Name, kCharStream[1]},
                               {kRs3Name, kMemAddress}});

  testing::internal::CaptureStdout();
  for (int i = 0; i < instructions.size(); ++i) {
    instructions[i]->Execute(nullptr);
  }
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello World\n", stdout_str);
}

TEST_F(CoralNPULogInstructionsTest, PrintTwoArguments) {
  constexpr char kFormatString[] = "%s World %x\n";
  constexpr uint32_t kCharStream = 0x00006948;  // "Hi"
  constexpr uint32_t kPrintNum = 0xbaddecaf;

  // Initialize memory.
  auto* db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();

  std::array<InstructionPtr, 3> instructions = {
      CreateInstruction(), CreateInstruction(), CreateInstruction()};

  // Also store the kCharStream elsewhere in the memory.
  auto* str_db = state_->db_factory()->Allocate<uint32_t>(sizeof(1));
  str_db->Set<uint32_t>(0, kCharStream);
  str_db->DecRef();

  constexpr uint32_t kStrMemAddress = kMemAddress + 20;
  state_->StoreMemory(instructions[0].get(), kStrMemAddress, str_db);
  AppendRegisterOperands(instructions[0].get(), {coralnpu::sim::test::kRs1Name},
                         {});
  instructions[0]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/3));

  AppendRegisterOperands(instructions[1].get(), {coralnpu::sim::test::kRs2Name},
                         {});
  instructions[1]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/1));

  constexpr char kRs3Name[] = "x3";
  state_->StoreMemory(instructions[2].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[2].get(), {kRs3Name}, {});
  instructions[2]->set_semantic_function(
      absl::bind_front(&CoralNPULogInstruction, /*mode=*/0));

  SetRegisterValues<uint32_t>({{coralnpu::sim::test::kRs1Name, kStrMemAddress},
                               {coralnpu::sim::test::kRs2Name, kPrintNum},
                               {kRs3Name, kMemAddress}});

  // Execute the instructions.
  testing::internal::CaptureStdout();
  for (int i = 0; i < instructions.size(); ++i) {
    instructions[i]->Execute(nullptr);
  }
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hi World baddecaf\n", stdout_str);
}

}  // namespace
