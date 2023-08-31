#include <sys/types.h>

#include <array>
#include <cstdint>
#include <string>

#include "sim/kelvin_instructions.h"
#include "sim/test/kelvin_vector_instructions_test_base.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/bind_front.h"
#include "mpact/sim/generic/instruction.h"

// Test Kelvin logging instruction functionality

namespace {

// Semantic function
using kelvin::sim::KelvinLogInstruction;

constexpr uint32_t kMemAddress = 0x1000;

class KelvinLogInstructionsTest
    : public kelvin::sim::test::KelvinVectorInstructionsTestBase {};

TEST_F(KelvinLogInstructionsTest, SimplePrint) {
  constexpr char kHelloString[] = "Hello World!\n";

  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<char>(sizeof(kHelloString));
  for (int i = 0; i < sizeof(kHelloString); ++i) {
    db->Set<char>(i, kHelloString[i]);
  }
  db->DecRef();

  auto instruction = CreateInstruction();
  state_->StoreMemory(instruction.get(), kMemAddress, db);
  AppendRegisterOperands(instruction.get(), {kelvin::sim::test::kRs1Name}, {});
  SetRegisterValues<uint32_t>({{kelvin::sim::test::kRs1Name, kMemAddress}});
  instruction->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/0));

  // Execute the instruction and check the stdout.
  testing::internal::CaptureStdout();
  instruction->Execute(nullptr);
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ(kHelloString, stdout_str);
}

TEST_F(KelvinLogInstructionsTest, PrintUnsignedNumber) {
  constexpr char kFormatString[] = "Hello %u\n";
  constexpr uint32_t kPrintNum = 2200000000;  // a number > INT32_MAX

  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();

  std::array<InstructionPtr, 2> instructions = {CreateInstruction(),
                                                CreateInstruction()};

  AppendRegisterOperands(instructions[0].get(), {kelvin::sim::test::kRs1Name},
                         {});
  instructions[0]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/1));  // scalar log

  // Set the second instruction for the actual print out.
  state_->StoreMemory(instructions[1].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[1].get(), {kelvin::sim::test::kRs2Name},
                         {});
  instructions[1]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/0));

  SetRegisterValues<uint32_t>({{kelvin::sim::test::kRs1Name, kPrintNum},
                               {kelvin::sim::test::kRs2Name, kMemAddress}});

  testing::internal::CaptureStdout();
  for (int i = 0; i < instructions.size(); ++i) {
    instructions[i]->Execute(nullptr);
  }
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello 2200000000\n", stdout_str);
}

TEST_F(KelvinLogInstructionsTest, PrintSignedNumber) {
  constexpr char kFormatString[] = "Hello %d\n";
  constexpr int32_t kPrintNum = -1337;

  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();

  std::array<InstructionPtr, 2> instructions = {CreateInstruction(),
                                                CreateInstruction()};

  AppendRegisterOperands(instructions[0].get(), {kelvin::sim::test::kRs1Name},
                         {});
  instructions[0]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/1));  // scalar log

  // Set the second instruction for the actual print out.
  state_->StoreMemory(instructions[1].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[1].get(), {kelvin::sim::test::kRs2Name},
                         {});
  instructions[1]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/0));

  SetRegisterValues<int32_t>({{kelvin::sim::test::kRs1Name, kPrintNum}});
  SetRegisterValues<uint32_t>({{kelvin::sim::test::kRs2Name, kMemAddress}});

  testing::internal::CaptureStdout();
  for (int i = 0; i < instructions.size(); ++i) {
    instructions[i]->Execute(nullptr);
  }
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello -1337\n", stdout_str);
}

TEST_F(KelvinLogInstructionsTest, PrintCharacterStream) {
  constexpr char kFormatString[] = "%s World\n";
  constexpr uint32_t kCharStream[] = {0x6c6c6548, 0x0000006f};  // "Hello"

  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();
  std::array<InstructionPtr, 3> instructions = {
      CreateInstruction(), CreateInstruction(), CreateInstruction()};
  AppendRegisterOperands(instructions[0].get(), {kelvin::sim::test::kRs1Name},
                         {});
  AppendRegisterOperands(instructions[1].get(), {kelvin::sim::test::kRs2Name},
                         {});
  for (int i = 0; i < 2; ++i) {
    instructions[i]->set_semantic_function(
        absl::bind_front(&KelvinLogInstruction, /*mode=*/2));  // character log
  }

  constexpr char kRs3Name[] = "x3";
  state_->StoreMemory(instructions[2].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[2].get(), {kRs3Name}, {});
  instructions[2]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/0));

  SetRegisterValues<uint32_t>({{kelvin::sim::test::kRs1Name, kCharStream[0]},
                               {kelvin::sim::test::kRs2Name, kCharStream[1]},
                               {kRs3Name, kMemAddress}});

  testing::internal::CaptureStdout();
  for (int i = 0; i < instructions.size(); ++i) {
    instructions[i]->Execute(nullptr);
  }
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello World\n", stdout_str);
}

TEST_F(KelvinLogInstructionsTest, PrintTwoArguments) {
  constexpr char kFormatString[] = "%s World %x\n";
  constexpr uint32_t kCharStream = 0x00006948;  // "Hi"
  constexpr uint32_t kPrintNum = 0xbaddecaf;

  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<char>(sizeof(kFormatString));
  for (int i = 0; i < sizeof(kFormatString); ++i) {
    db->Set<char>(i, kFormatString[i]);
  }
  db->DecRef();

  std::array<InstructionPtr, 3> instructions = {
      CreateInstruction(), CreateInstruction(), CreateInstruction()};

  // Also store the kCharStream elsewhere in the memory.
  auto *str_db = state_->db_factory()->Allocate<uint32_t>(sizeof(1));
  str_db->Set<uint32_t>(0, kCharStream);
  str_db->DecRef();

  constexpr uint32_t kStrMemAddress = kMemAddress + 20;
  state_->StoreMemory(instructions[0].get(), kStrMemAddress, str_db);
  AppendRegisterOperands(instructions[0].get(), {kelvin::sim::test::kRs1Name},
                         {});
  instructions[0]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/3));

  AppendRegisterOperands(instructions[1].get(), {kelvin::sim::test::kRs2Name},
                         {});
  instructions[1]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/1));

  constexpr char kRs3Name[] = "x3";
  state_->StoreMemory(instructions[2].get(), kMemAddress, db);
  AppendRegisterOperands(instructions[2].get(), {kRs3Name}, {});
  instructions[2]->set_semantic_function(
      absl::bind_front(&KelvinLogInstruction, /*mode=*/0));

  SetRegisterValues<uint32_t>({{kelvin::sim::test::kRs1Name, kStrMemAddress},
                               {kelvin::sim::test::kRs2Name, kPrintNum},
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
