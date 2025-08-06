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

#include "sim/renode/renode_mpact.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>

#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_debug_info.h"

// This file contains a test of the RenodeMpact interface using kelvin.

namespace {

using kelvin::sim::renode::ExecutionResult;
using mpact::sim::riscv::DebugRegisterEnum;

constexpr char kEbreakExecutableFileName[] = "kelvin_ebreak.elf";
constexpr char kExecutableFileName[] = "hello_world_mpause.elf";
constexpr char kBinFileName[] = "hello_world_mpause.bin";
constexpr uint64_t kBinFileAddress = 0x0;
constexpr uint64_t kBinFileEntryPoint = 0x0;
constexpr size_t kBufferSize = 1024;

// The depot path to the test directory.
constexpr char kDepotPath[] = "sim/test/";

class RenodeMpactTest : public testing::Test {
 protected:
  RenodeMpactTest() {
    sim_id_ = construct(1);
    CHECK_GE(sim_id_, 0) << sim_id_;
  }

  ~RenodeMpactTest() override {
    int32_t result = destruct(sim_id_);
    CHECK_EQ(result, 0);
  }
  int32_t sim_id_;
};

// Test the reset call.
TEST_F(RenodeMpactTest, Reset) {
  CHECK_EQ(reset(sim_id_), 0);
  CHECK_EQ(reset(10), -1);
}

// Test that registers can be read and written using the id numbers.
TEST_F(RenodeMpactTest, RegisterId) {
  uint32_t xreg = static_cast<uint32_t>(DebugRegisterEnum::kX0);
  uint64_t value = 0;
  for (int i = 0; i < 8; i++) {
    auto error = write_register(sim_id_, xreg + i, value++);
    EXPECT_EQ(error, 0);
  }
  value = 0;
  uint64_t reg_value;
  for (int i = 0; i < 8; i++) {
    auto error = read_register(sim_id_, xreg + i, &reg_value);
    EXPECT_EQ(error, 0);
    EXPECT_EQ(reg_value, value);
    value++;
  }
  // Wrong debug ids.
  auto error = read_register(10, xreg, &reg_value);
  EXPECT_EQ(error, -1);
  error = write_register(10, xreg, reg_value);
  EXPECT_EQ(error, -1);
  // Wrong register numbers.
  error = read_register(sim_id_, xreg - 1, &reg_value);
  EXPECT_EQ(error, -1);
  error = write_register(sim_id_, xreg - 1, reg_value);
  EXPECT_EQ(error, -1);
  // Nullptr.
  error = read_register(sim_id_, xreg, nullptr);
  EXPECT_EQ(error, -1);
}

TEST_F(RenodeMpactTest, LoadExecutable) {
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kExecutableFileName);
  // Fails - wrong sim id.
  int32_t status;
  uint64_t entry_pt;
  entry_pt = load_executable(100, input_file_name.c_str(), &status);
  EXPECT_EQ(status, -1);
  // Fails - wrong file name.
  entry_pt = load_executable(sim_id_, "wrong_file_name", &status);
  EXPECT_EQ(status, -1);
  // Succeeds - correct sim id and file name.
  entry_pt = load_executable(sim_id_, input_file_name.c_str(), &status);
  EXPECT_EQ(status, 0);
  // Read the pc, verify that the entry point matches.
  uint64_t pc_value;
  status = read_register(sim_id_, static_cast<uint32_t>(DebugRegisterEnum::kPc),
                         &pc_value);
  EXPECT_EQ(pc_value, entry_pt);
  EXPECT_EQ(status, 0);
}

TEST_F(RenodeMpactTest, ReadWriteMem) {
  constexpr uint8_t kBytes[] = {0x01, 0x02, 0x03, 0x04, 0xff, 0xfe, 0xfd, 0xfc};
  int res =
      write_memory(sim_id_, 0x100, reinterpret_cast<const char*>(kBytes), 8);
  EXPECT_EQ(res, 8);
  uint8_t mem_bytes[8] = {0xde, 0xad, 0xbe, 0xef, 0x5a, 0xa5, 0xff, 0x00};
  res = read_memory(sim_id_, 0x104, reinterpret_cast<char*>(mem_bytes), 1);
  EXPECT_EQ(res, 1);
  EXPECT_EQ(mem_bytes[0], kBytes[4]);
  res = read_memory(sim_id_, 0x100, reinterpret_cast<char*>(mem_bytes), 8);
  for (int i = 0; i < 8; i++) EXPECT_EQ(kBytes[i], mem_bytes[i]);

  // Read memory from out of bound address
  constexpr uint64_t kOutOfBoundAddress = 0x3'FFFF'FFFFULL;
  res = read_memory(sim_id_, kOutOfBoundAddress,
                    reinterpret_cast<char*>(mem_bytes), 1);
  EXPECT_EQ(res, 0);
  // Write to out of bound memory address
  res = write_memory(sim_id_, kOutOfBoundAddress,
                     reinterpret_cast<const char*>(mem_bytes), 1);
  EXPECT_EQ(res, 0);
}

TEST_F(RenodeMpactTest, LoadImage) {
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kBinFileName);
  int32_t ret =
      load_image(sim_id_, (input_file_name + "xyz").c_str(), kBinFileAddress);
  EXPECT_LT(ret, 0);
  ret = load_image(sim_id_, input_file_name.c_str(), kBinFileAddress);
  EXPECT_EQ(ret, 0);
  std::ifstream bin_file;
  bin_file.open(input_file_name, std::ios::in | std::ios::binary);
  CHECK_EQ(bin_file.good(), true);
  char file_data[kBufferSize];
  char memory_data[kBufferSize];
  uint64_t data_address = kBinFileAddress;
  size_t gcount;
  size_t total = 0;
  do {
    bin_file.read(file_data, kBufferSize);
    gcount = bin_file.gcount();
    auto res = read_memory(sim_id_, data_address, memory_data, gcount);
    EXPECT_GT(res, 0);
    EXPECT_EQ(res, gcount);
    for (int i = 0; i < kBufferSize; ++i)
      EXPECT_EQ(file_data[i], memory_data[i]) << "Byte: " << total + i;
    data_address += gcount;
    total += gcount;
  } while (gcount == kBufferSize);
  EXPECT_TRUE(bin_file.eof());
  bin_file.close();
}

TEST_F(RenodeMpactTest, StepExecutableProgram) {
  testing::internal::CaptureStdout();
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kExecutableFileName);
  int32_t status;
  (void)load_executable(sim_id_, input_file_name.c_str(), &status);
  CHECK_EQ(status, 0);
  // Fail if you use the wrong sim_id_.
  auto count = step(10, 1, &status);
  EXPECT_EQ(status, -1);
  EXPECT_EQ(count, 0);
  // Zero steps, but no failure if step count is == 0.
  count = step(sim_id_, 0, &status);
  EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  EXPECT_EQ(count, 0);
  // Null status should work.
  count = step(10, 1, nullptr);
  EXPECT_EQ(count, 0);
  count = step(10, 0, nullptr);
  EXPECT_EQ(count, 0);
  count = step(sim_id_, 1, nullptr);
  EXPECT_EQ(count, 1);
  // Counts from 1 to 10.
  for (int i = 0; i < 10; i++) {
    count = step(sim_id_, i, &status);
    EXPECT_EQ(count, i);
    EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  }
  constexpr uint64_t kStepCount = 500;
  while (true) {
    count = step(sim_id_, kStepCount, &status);
    if (count != kStepCount) break;
    EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  }
  // Execution should now have completed and the program has printed the proper
  // exit message.
  EXPECT_GT(kStepCount, count);
  EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
}

TEST_F(RenodeMpactTest, StepEbreakExecutableProgram) {
  testing::internal::CaptureStdout();
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kEbreakExecutableFileName);
  int32_t status;
  (void)load_executable(sim_id_, input_file_name.c_str(), &status);
  CHECK_EQ(status, 0);
  constexpr uint64_t kStepCount = 500;
  uint64_t count;
  while (true) {
    count = step(sim_id_, kStepCount, &status);
    if (count != kStepCount) break;
    EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  }
  // Execution should now have completed and the program has printed the fault
  // exit message.
  EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kAborted));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits with fault\n", stdout_str);
}

TEST_F(RenodeMpactTest, StepImageProgram) {
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kBinFileName);
  testing::internal::CaptureStdout();
  auto ret = load_image(sim_id_, input_file_name.c_str(), kBinFileAddress);
  EXPECT_EQ(ret, 0);
  auto xreg = static_cast<uint32_t>(DebugRegisterEnum::kPc);
  auto error = write_register(sim_id_, xreg, kBinFileEntryPoint);
  EXPECT_EQ(error, 0);
  constexpr uint64_t kStepCount = 1000;
  int32_t status;
  int32_t count;
  while (true) {
    count = step(sim_id_, kStepCount, &status);
    if (count != kStepCount) break;
    EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  }
  // Execution should now have completed and the program has printed the proper
  // exit message.
  EXPECT_GT(kStepCount, count);
  EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
}

// Test stepping over a binary image program with external memory.
TEST_F(RenodeMpactTest, StepImageProgramWithExternalMemory) {
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kBinFileName);

  // Setup the external memory.
  constexpr uint64_t kMemoryBlockSize = 0x40000;  // 256KB
  constexpr uint64_t kNumBlock = 16;              // 4MB / 256KB
  uint8_t* memory_block[kNumBlock] = {nullptr};
  // Allocate memory blocks.
  for (int i = 0; i < kNumBlock; ++i) {
    memory_block[i] = new uint8_t[kMemoryBlockSize];
    memset(memory_block[i], 0, kMemoryBlockSize);
  }

  // Reset the simulator with the external memory.
  destruct(sim_id_);
  sim_id_ = construct_with_memory(1, kMemoryBlockSize,
                                  kNumBlock * kMemoryBlockSize, memory_block);
  EXPECT_GE(sim_id_, 1) << sim_id_;  // the agent count keeps incrementing.

  testing::internal::CaptureStdout();
  auto ret = load_image(sim_id_, input_file_name.c_str(), kBinFileAddress);
  EXPECT_EQ(ret, 0);
  auto xreg = static_cast<uint32_t>(DebugRegisterEnum::kPc);
  auto error = write_register(sim_id_, xreg, kBinFileEntryPoint);
  EXPECT_EQ(error, 0);
  constexpr uint64_t kStepCount = 1000;
  int32_t status;
  int32_t count;
  while (true) {
    count = step(sim_id_, kStepCount, &status);
    if (count != kStepCount) break;
    EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  }
  // Execution should now have completed and the program has printed the proper
  // exit message.
  EXPECT_GT(kStepCount, count);
  EXPECT_EQ(status, static_cast<int32_t>(ExecutionResult::kOk));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);

  // Release the memory blocks.
  for (int i = 0; i < kNumBlock; ++i) {
    delete[] memory_block[i];
  }
}

}  // namespace
