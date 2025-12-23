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

#include "sim/coralnpu_top.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>

#include "sim/coralnpu_state.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace {

#ifndef EXPECT_OK
#define ABSL_EXPECT_OK(x) EXPECT_TRUE(x.ok())
#endif

using ::coralnpu::sim::CoralNPUTop;
using ::mpact::sim::util::ElfProgramLoader;
using ::mpact::sim::util::FlatDemandMemory;

using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;
constexpr char kEbreakElfFileName[] = "coralnpu_ebreak.elf";
constexpr char kMpauseBinaryFileName[] = "hello_world_mpause.bin";
constexpr char kMpauseElfFileName[] = "hello_world_mpause.elf";
constexpr char kRV32imfElfFileName[] = "hello_world_rv32imf.elf";
constexpr char kRV32iElfFileName[] = "rv32i.elf";
constexpr char kRV32mElfFileName[] = "rv32m.elf";
constexpr char kRV32SoftFloatElfFileName[] = "rv32soft_fp.elf";
constexpr char kRV32fElfFileName[] = "rv32uf_fadd.elf";
constexpr char kCoralnpuVldVstFileName[] = "coralnpu_vldvst.elf";
constexpr char kCoralnpuPerfCountersFileName[] = "coralnpu_perf_counters.elf";

// The depot path to the test directory.
constexpr char kDepotPath[] = "sim/test/";

// Maximum memory size used by riscv programs build for userspace.
constexpr uint64_t kRiscv32MaxAddress = 0x3'ffff'ffffULL;

constexpr uint64_t kBinaryAddress = 0;

class CoralNPUTopTest : public testing::Test {
 protected:
  CoralNPUTopTest() {
    memory_ = new FlatDemandMemory();
    coralnpu_top_ = new CoralNPUTop("CoralNPU");
    // Set up the elf loader.
    loader_ = new ElfProgramLoader(coralnpu_top_->memory());
  }

  ~CoralNPUTopTest() override {
    delete loader_;
    delete coralnpu_top_;
    delete memory_;
  }

  void LoadFile(const std::string file_name) {
    const std::string input_file_name =
        absl::StrCat(kDepotPath, "testfiles/", file_name);
    auto result = loader_->LoadProgram(input_file_name);
    CHECK_OK(result);
    entry_point_ = result.value();
  }

  uint32_t entry_point_;
  CoralNPUTop* coralnpu_top_ = nullptr;
  ElfProgramLoader* loader_ = nullptr;
  FlatDemandMemory* memory_ = nullptr;
};

// Check the max memory size
TEST_F(CoralNPUTopTest, CheckDefaultMaxMemorySize) {
  EXPECT_EQ(coralnpu_top_->state()->max_physical_address(),
            coralnpu::sim::kCoralnpuMaxMemoryAddress);
}

// Run a program exceeds the default memory setting
TEST_F(CoralNPUTopTest, RunProgramExceedDefaultMemory) {
  LoadFile(kRV32imfElfFileName);
  testing::internal::CaptureStderr();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kUserRequest));
  const std::string stderr_str = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_str, testing::HasSubstr("Memory store access fault"));
}

// Runs the program from has ebreak (from syscall).
TEST_F(CoralNPUTopTest, RunEbreakProgram) {
  LoadFile(kEbreakElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(halt_result.value(), coralnpu::sim::kHaltAbort);
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits with fault\n", stdout_str);
}

// Runs the program from beginning to end. Enable arm semihosting.
TEST_F(CoralNPUTopTest, RunHelloProgramSemihost) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  coralnpu_top_->state()->set_max_physical_address(kRiscv32MaxAddress);
  LoadFile(kRV32imfElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello, World! 7\n", stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Runs the program ended with mpause from beginning to end.
TEST_F(CoralNPUTopTest, RunHelloMpauseProgram) {
  LoadFile(kMpauseElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kUserRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
}

TEST_F(CoralNPUTopTest, LoadImageFailed) {
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kMpauseBinaryFileName);
  auto result = coralnpu_top_->LoadImage("wrong_file", kBinaryAddress);
  EXPECT_FALSE(result.ok());
  // Set the memory to be smaller than the loaded image
  coralnpu_top_->state()->set_max_physical_address(0);
  result = coralnpu_top_->LoadImage(input_file_name, kBinaryAddress);
  EXPECT_FALSE(result.ok());
  coralnpu_top_->state()->set_max_physical_address(0xf);
  result = coralnpu_top_->LoadImage(input_file_name, kBinaryAddress);
  EXPECT_FALSE(result.ok());
}

// Directly read/write to memory addresses that are out-of-bound
TEST_F(CoralNPUTopTest, ReadWriteOutOfBoundMemory) {
  // Set the machine to have 16-byte physical memory
  constexpr uint64_t kTestMemerySize = 0x10;
  constexpr uint64_t kMaxPhysicalAddress = kTestMemerySize - 1;
  coralnpu_top_->state()->set_max_physical_address(kMaxPhysicalAddress);
  uint8_t mem_bytes[kTestMemerySize + 4] = {0};
  // Read the memory with the length greater than the physical memory size. The
  // read operation is successful within the physical memory size range.
  auto result =
      coralnpu_top_->ReadMemory(kBinaryAddress, mem_bytes, sizeof(mem_bytes));
  ABSL_EXPECT_OK(result);
  EXPECT_EQ(result.value(), kTestMemerySize);
  // Read at the maximum physical address, so only one byte can be read.
  result = coralnpu_top_->ReadMemory(kMaxPhysicalAddress, mem_bytes,
                                     sizeof(mem_bytes));
  ABSL_EXPECT_OK(result);
  EXPECT_EQ(result.value(), 1);
  // Read the memory with the staring address out of the physical memory range.
  // The read operation returns error.
  result = coralnpu_top_->ReadMemory(kTestMemerySize + 4, mem_bytes,
                                     sizeof(mem_bytes));
  EXPECT_FALSE(result.ok());

  // Write the memory with the length greater than the physical memory size. The
  // write operation is successful within the physical memory size range.
  result =
      coralnpu_top_->WriteMemory(kBinaryAddress, mem_bytes, sizeof(mem_bytes));
  ABSL_EXPECT_OK(result);
  EXPECT_EQ(result.value(), kTestMemerySize);
  // Write at the maximum physical address, so only one byte can be written.
  result = coralnpu_top_->WriteMemory(kMaxPhysicalAddress, mem_bytes,
                                      sizeof(mem_bytes));
  ABSL_EXPECT_OK(result);
  EXPECT_EQ(result.value(), 1);
  // Write the memory with the staring address out of the physical memory range.
  // The write operation returns error.
  result = coralnpu_top_->WriteMemory(kTestMemerySize + 4, mem_bytes,
                                      sizeof(mem_bytes));
  EXPECT_FALSE(result.ok());
}

// Runs the binary program from beginning to end
TEST_F(CoralNPUTopTest, RunHelloMpauseBinaryProgram) {
  const std::string input_file_name =
      absl::StrCat(kDepotPath, "testfiles/", kMpauseBinaryFileName);
  constexpr uint32_t kBinaryEntryPoint = 0;
  testing::internal::CaptureStdout();
  auto result = coralnpu_top_->LoadImage(input_file_name, kBinaryAddress);
  CHECK_OK(result);
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", kBinaryEntryPoint));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kUserRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
}

// Runs the rv32i program with arm semihosting.
TEST_F(CoralNPUTopTest, RunRV32IProgram) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  coralnpu_top_->state()->set_max_physical_address(kRiscv32MaxAddress);
  LoadFile(kRV32iElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("5+5=10;5-5=0\n", stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Runs the rv32m program with arm semihosting.
TEST_F(CoralNPUTopTest, RunRV32MProgram) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  coralnpu_top_->state()->set_max_physical_address(kRiscv32MaxAddress);
  LoadFile(kRV32mElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("5*5=25;5/5=1\n", stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Runs the rv32 soft float program with arm semihosting.
TEST_F(CoralNPUTopTest, RunRV32SoftFProgram) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  coralnpu_top_->state()->set_max_physical_address(kRiscv32MaxAddress);
  LoadFile(kRV32SoftFloatElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("7.00+3.00=10.00;7.00-3.00=4.00;7.00*3.00=21.00;7.00/3.00=2.33\n",
            stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Runs the rv32f program (not supported)
TEST_F(CoralNPUTopTest, RunIllegalRV32FProgram) {
  LoadFile(kRV32fElfFileName);
  testing::internal::CaptureStderr();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kUserRequest));
  const std::string stderr_str = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_str, testing::HasSubstr("Illegal instruction at 0x"));
}

// Steps through the program from beginning to end.
TEST_F(CoralNPUTopTest, StepProgram) {
  LoadFile(kEbreakElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));

  auto res = coralnpu_top_->Step(10000);
  ABSL_EXPECT_OK(res.status());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(halt_result.value(), coralnpu::sim::kHaltAbort);

  EXPECT_EQ("Program exits with fault\n",
            testing::internal::GetCapturedStdout());
}

// Sets/Clears breakpoints without executing the program.
TEST_F(CoralNPUTopTest, SetAndClearBreakpoint) {
  LoadFile(kRV32imfElfFileName);
  coralnpu_top_->state()->set_max_physical_address(kRiscv32MaxAddress);
  auto result = loader_->GetSymbol("printf");
  ABSL_EXPECT_OK(result);
  auto address = result.value().first;
  EXPECT_EQ(coralnpu_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
  ABSL_EXPECT_OK(coralnpu_top_->SetSwBreakpoint(address));
  EXPECT_EQ(coralnpu_top_->SetSwBreakpoint(address).code(),
            absl::StatusCode::kAlreadyExists);
  ABSL_EXPECT_OK(coralnpu_top_->ClearSwBreakpoint(address));
  EXPECT_EQ(coralnpu_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
  ABSL_EXPECT_OK(coralnpu_top_->SetSwBreakpoint(address));
  ABSL_EXPECT_OK(coralnpu_top_->ClearAllSwBreakpoints());
  EXPECT_EQ(coralnpu_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
}

// Runs program with breakpoint at printf with arm semihosting.
TEST_F(CoralNPUTopTest, RunWithBreakpoint) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  coralnpu_top_->state()->set_max_physical_address(kRiscv32MaxAddress);
  LoadFile(kRV32imfElfFileName);

  // Set breakpoint at printf.
  auto result = loader_->GetSymbol("printf");
  ABSL_EXPECT_OK(result);
  auto address = result.value().first;
  ABSL_EXPECT_OK(coralnpu_top_->SetSwBreakpoint(address));

  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));

  // Run to printf.
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());

  // Should be stopped at breakpoint, but nothing printed.
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  EXPECT_EQ(testing::internal::GetCapturedStdout().size(), 0);

  // Run to the end of the program.
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());

  // Should be stopped due to semihost halt request. Captured 'Hello World!
  // 7\n'.
  halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  EXPECT_EQ("Hello, World! 7\n", testing::internal::GetCapturedStdout());
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Memory read/write test.
TEST_F(CoralNPUTopTest, Memory) {
  uint8_t byte_data = 0xab;
  uint16_t half_data = 0xabcd;
  uint32_t word_data = 0xba5eba11;
  uint64_t dword_data = 0x5ca1ab1e'0ddball;
  ABSL_EXPECT_OK(coralnpu_top_->WriteMemory(0x1000, &byte_data, sizeof(byte_data)));
  ABSL_EXPECT_OK(coralnpu_top_->WriteMemory(0x1004, &half_data, sizeof(half_data)));
  ABSL_EXPECT_OK(coralnpu_top_->WriteMemory(0x1008, &word_data, sizeof(word_data)));
  ABSL_EXPECT_OK(
      coralnpu_top_->WriteMemory(0x1010, &dword_data, sizeof(dword_data)));

  uint8_t byte_value;
  uint16_t half_value;
  uint32_t word_value;
  uint64_t dword_value;

  ABSL_EXPECT_OK(coralnpu_top_->ReadMemory(0x1000, &byte_value, sizeof(byte_value)));
  ABSL_EXPECT_OK(coralnpu_top_->ReadMemory(0x1004, &half_value, sizeof(half_value)));
  ABSL_EXPECT_OK(coralnpu_top_->ReadMemory(0x1008, &word_value, sizeof(word_value)));
  ABSL_EXPECT_OK(
      coralnpu_top_->ReadMemory(0x1010, &dword_value, sizeof(dword_value)));

  EXPECT_EQ(byte_data, byte_value);
  EXPECT_EQ(half_data, half_value);
  EXPECT_EQ(word_data, word_value);
  EXPECT_EQ(dword_data, dword_value);

  ABSL_EXPECT_OK(coralnpu_top_->ReadMemory(0x1000, &byte_value, sizeof(byte_value)));
  ABSL_EXPECT_OK(coralnpu_top_->ReadMemory(0x1000, &half_value, sizeof(half_value)));
  ABSL_EXPECT_OK(coralnpu_top_->ReadMemory(0x1000, &word_value, sizeof(word_value)));
  ABSL_EXPECT_OK(
      coralnpu_top_->ReadMemory(0x1000, &dword_value, sizeof(dword_value)));

  EXPECT_EQ(byte_data, byte_value);
  EXPECT_EQ(byte_data, half_value);
  EXPECT_EQ(byte_data, word_value);
  EXPECT_EQ(0x0000'abcd'0000'00ab, dword_value);
}

// Register name test.
TEST_F(CoralNPUTopTest, RegisterNames) {
  // Test x-names and numbers.
  uint32_t word_value;
  for (int i = 0; i < 32; i++) {
    std::string name = absl::StrCat("x", i);
    auto result = coralnpu_top_->ReadRegister(name);
    ABSL_EXPECT_OK(result.status());
    word_value = result.value();
    ABSL_EXPECT_OK(coralnpu_top_->WriteRegister(name, word_value));
  }
  // Not found.
  EXPECT_EQ(coralnpu_top_->ReadRegister("x32").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(coralnpu_top_->WriteRegister("x32", word_value).code(),
            absl::StatusCode::kNotFound);
  // Aliases.
  for (auto& [name, alias] : {std::tuple<std::string, std::string>{"x1", "ra"},
                              {"x4", "tp"},
                              {"x8", "s0"}}) {
    uint32_t write_value = 0xba5eba11;
    ABSL_EXPECT_OK(coralnpu_top_->WriteRegister(name, write_value));
    uint32_t read_value;
    auto result = coralnpu_top_->ReadRegister(alias);
    ABSL_EXPECT_OK(result.status());
    read_value = result.value();
    EXPECT_EQ(read_value, write_value);
  }
  // Custom CSRs.
  for (auto name : {"kisa"}) {
    ABSL_EXPECT_OK(coralnpu_top_->ReadRegister(name));
  }

  // CSRs that diverge from stock values in MPACT.
  constexpr uint32_t kMisaValue = 0x40801100;
  for (auto& [name, expected_value] :
       {std::tuple<std::string, uint32_t>{"misa", kMisaValue}}) {
    auto result = coralnpu_top_->ReadRegister(name);
    ABSL_EXPECT_OK(result.status());
    EXPECT_EQ(result.value(), expected_value);
  }
}

TEST_F(CoralNPUTopTest, RunCoralNPUVectorProgram) {
  LoadFile(kCoralnpuVldVstFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kUserRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_THAT(stdout_str, testing::HasSubstr("vld_vst test passed!"));
}

TEST_F(CoralNPUTopTest, RunCoralNPUPerfCountersProgram) {
  LoadFile(kCoralnpuPerfCountersFileName);
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));
  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kUserRequest));
}

constexpr int kMemoryBlockSize = 256 * 1024;  // 256KB
// Default max memory address is 4MB - 1. Round up to find the number of memory
// blocks.
constexpr int kNumMemoryBlocks =
    (coralnpu::sim::kCoralnpuMaxMemoryAddress + kMemoryBlockSize) /
    kMemoryBlockSize;

class CoralNPUTopExternalMemoryTest : public testing::Test {
 protected:
  CoralNPUTopExternalMemoryTest()
      : memory_size_(kMemoryBlockSize * kNumMemoryBlocks) {
    // Set the memory blocks outside of CoralNPUTop.
    for (int i = 0; i < kNumMemoryBlocks; ++i) {
      memory_blocks_[i] = new uint8_t[kMemoryBlockSize];
      memset(memory_blocks_[i], 0, kMemoryBlockSize);
    }
    coralnpu_top_ = new CoralNPUTop("CoralNPU", kMemoryBlockSize, memory_size_,
                                    memory_blocks_);
    // Set up the elf loader.
    loader_ = new ElfProgramLoader(coralnpu_top_->memory());
  }

  ~CoralNPUTopExternalMemoryTest() override {
    delete loader_;
    delete coralnpu_top_;
    for (int i = 0; i < kNumMemoryBlocks; ++i) {
      delete[] memory_blocks_[i];
    }
  }

  void LoadFile(const std::string file_name) {
    const std::string input_file_name =
        absl::StrCat(kDepotPath, "testfiles/", file_name);
    auto result = loader_->LoadProgram(input_file_name);
    CHECK_OK(result);
    entry_point_ = result.value();
  }

  uint32_t entry_point_;
  CoralNPUTop* coralnpu_top_ = nullptr;
  ElfProgramLoader* loader_ = nullptr;
  uint8_t* memory_blocks_[kNumMemoryBlocks] = {nullptr};
  uint64_t memory_size_;
};

// Run a vector program from beginning to end.
TEST_F(CoralNPUTopExternalMemoryTest, RunCoralNPUVectorProgram) {
  LoadFile(kCoralnpuVldVstFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));

  ABSL_EXPECT_OK(coralnpu_top_->Run());
  ABSL_EXPECT_OK(coralnpu_top_->Wait());
  auto halt_result = coralnpu_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(halt_result.value(), *HaltReason::kUserRequest);
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_THAT(stdout_str, testing::HasSubstr("vld_vst test passed!"));
}

// Step a regular program from beginning to end.
TEST_F(CoralNPUTopExternalMemoryTest, StepMPauseProgram) {
  LoadFile(kMpauseElfFileName);
  testing::internal::CaptureStdout();
  ABSL_EXPECT_OK(coralnpu_top_->WriteRegister("pc", entry_point_));

  constexpr int kStep = 2000;
  absl::StatusOr<coralnpu::sim::HaltReasonValueType> halt_result;
  do {
    auto res = coralnpu_top_->Step(kStep);
    ABSL_EXPECT_OK(res.status());
    halt_result = coralnpu_top_->GetLastHaltReason();
    ABSL_EXPECT_OK(halt_result);
  } while (halt_result.value() == *HaltReason::kNone);

  EXPECT_EQ(halt_result.value(), *HaltReason::kUserRequest);
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
}

// Read/Write memory
TEST_F(CoralNPUTopExternalMemoryTest, AccessMemory) {
  constexpr uint64_t kTestMemerySize = 8;
  const uint64_t test_access_address[] = {
      0x1000, kMemoryBlockSize - 4, kMemoryBlockSize + 4,
      coralnpu::sim::kCoralnpuMaxMemoryAddress - 4};
  uint8_t mem_bytes[kTestMemerySize] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint8_t mem_bytes_return[kTestMemerySize] = {0};

  // Write new values to the memory.
  for (int i = 0; i < sizeof(test_access_address) / sizeof(uint64_t); ++i) {
    auto result = coralnpu_top_->WriteMemory(test_access_address[i], mem_bytes,
                                             sizeof(mem_bytes));
    uint64_t expected_length =
        std::min(kTestMemerySize, memory_size_ - test_access_address[i]);
    ABSL_EXPECT_OK(result);
    EXPECT_EQ(result.value(), expected_length);
    // Read back the content from the external memory.
    result = coralnpu_top_->ReadMemory(test_access_address[i], mem_bytes_return,
                                       sizeof(mem_bytes_return));
    ABSL_EXPECT_OK(result);
    EXPECT_EQ(result.value(), expected_length);
    for (int i = 0; i < result.value(); ++i) {
      EXPECT_EQ(mem_bytes[i], mem_bytes_return[i]);
    }
    memset(mem_bytes_return, 0, sizeof(mem_bytes_return));
  }
}

}  // namespace
