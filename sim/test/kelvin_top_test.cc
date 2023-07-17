#include "sim/kelvin_top.h"

#include <cstdint>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace {

#ifndef EXPECT_OK
#define EXPECT_OK(x) EXPECT_TRUE(x.ok())
#endif

using ::kelvin::sim::KelvinTop;
using ::mpact::sim::util::ElfProgramLoader;
using ::mpact::sim::util::FlatDemandMemory;

using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;
constexpr char kMpauseElfFileName[] = "hello_world_mpause.elf";
constexpr char kRV32imfElfFileName[] = "hello_world_rv32imf.elf";
constexpr char kRV32iElfFileName[] = "rv32i.elf";
constexpr char kRV32mElfFileName[] = "rv32m.elf";
constexpr char kRV32SoftFloatElfFileName[] = "rv32soft_fp.elf";

// The depot path to the test directory.
constexpr char kDepotPath[] = "sim/test/";

class KelvinTopTest : public testing::Test {
 protected:
  KelvinTopTest() {
    memory_ = new FlatDemandMemory();
    kelvin_top_ = new KelvinTop("Kelvin");
    // Set up the elf loader.
    loader_ = new ElfProgramLoader(kelvin_top_->memory());
  }

  ~KelvinTopTest() override {
    delete loader_;
    delete kelvin_top_;
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
  KelvinTop *kelvin_top_ = nullptr;
  ElfProgramLoader *loader_ = nullptr;
  FlatDemandMemory *memory_ = nullptr;
};

// Runs the program from beginning to end.
TEST_F(KelvinTopTest, RunHelloProgram) {
  LoadFile(kRV32imfElfFileName);
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hit breakpoint or program exits with fault\n", stdout_str);
}

// Runs the program from beginning to end. Enable arm semihosting.
TEST_F(KelvinTopTest, RunHelloProgramSemihost) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  LoadFile(kRV32imfElfFileName);
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello, World! 7\n", stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Runs the program ended with mpause from beginning to end.
TEST_F(KelvinTopTest, RunHelloMpauseProgram) {
  LoadFile(kMpauseElfFileName);
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
}

// Runs the rv32i program with arm semihosting.
TEST_F(KelvinTopTest, RunRV32IProgram) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  LoadFile(kRV32iElfFileName);
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("5+5=10;5-5=0\n", stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Runs the rv32m program with arm semihosting.
TEST_F(KelvinTopTest, RunRV32MProgram) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  LoadFile(kRV32mElfFileName);
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("5*5=25;5/5=1\n", stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Runs the rv32 soft float program with arm semihosting.
TEST_F(KelvinTopTest, RunRV32SoftFProgram) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  LoadFile(kRV32SoftFloatElfFileName);
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("7.00+3.00=10.00;7.00-3.00=4.00;7.00*3.00=21.00;7.00/3.00=2.33\n",
            stdout_str);
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Steps through the program from beginning to end.
TEST_F(KelvinTopTest, StepProgram) {
  LoadFile(kRV32imfElfFileName);
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));

  auto res = kelvin_top_->Step(10000);
  EXPECT_OK(res.status());
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));

  EXPECT_EQ("Hit breakpoint or program exits with fault\n",
            testing::internal::GetCapturedStdout());
}

// Sets/Clears breakpoints without executing the program.
TEST_F(KelvinTopTest, SetAndClearBreakpoint) {
  LoadFile(kRV32imfElfFileName);
  auto result = loader_->GetSymbol("printf");
  EXPECT_OK(result);
  auto address = result.value().first;
  EXPECT_EQ(kelvin_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
  EXPECT_OK(kelvin_top_->SetSwBreakpoint(address));
  EXPECT_EQ(kelvin_top_->SetSwBreakpoint(address).code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_OK(kelvin_top_->ClearSwBreakpoint(address));
  EXPECT_EQ(kelvin_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
  EXPECT_OK(kelvin_top_->SetSwBreakpoint(address));
  EXPECT_OK(kelvin_top_->ClearAllSwBreakpoints());
  EXPECT_EQ(kelvin_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
}

// Runs program with breakpoint at printf with arm semihosting.
TEST_F(KelvinTopTest, RunWithBreakpoint) {
  absl::SetFlag(&FLAGS_use_semihost, true);
  LoadFile(kRV32imfElfFileName);

  // Set breakpoint at printf.
  auto result = loader_->GetSymbol("printf");
  EXPECT_OK(result);
  auto address = result.value().first;
  EXPECT_OK(kelvin_top_->SetSwBreakpoint(address));

  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->WriteRegister("pc", entry_point_));

  // Run to printf.
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());

  // Should be stopped at breakpoint, but nothing printed.
  auto halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  EXPECT_EQ(testing::internal::GetCapturedStdout().size(), 0);

  // Run to the end of the program.
  testing::internal::CaptureStdout();
  EXPECT_OK(kelvin_top_->Run());
  EXPECT_OK(kelvin_top_->Wait());

  // Should be stopped due to semihost halt request. Captured 'Hello World!
  // 7\n'.
  halt_result = kelvin_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  EXPECT_EQ("Hello, World! 7\n", testing::internal::GetCapturedStdout());
  absl::SetFlag(&FLAGS_use_semihost, false);
}

// Memory read/write test.
TEST_F(KelvinTopTest, Memory) {
  uint8_t byte_data = 0xab;
  uint16_t half_data = 0xabcd;
  uint32_t word_data = 0xba5eba11;
  uint64_t dword_data = 0x5ca1ab1e'0ddball;
  EXPECT_OK(kelvin_top_->WriteMemory(0x1000, &byte_data, sizeof(byte_data)));
  EXPECT_OK(kelvin_top_->WriteMemory(0x1004, &half_data, sizeof(half_data)));
  EXPECT_OK(kelvin_top_->WriteMemory(0x1008, &word_data, sizeof(word_data)));
  EXPECT_OK(kelvin_top_->WriteMemory(0x1010, &dword_data, sizeof(dword_data)));

  uint8_t byte_value;
  uint16_t half_value;
  uint32_t word_value;
  uint64_t dword_value;

  EXPECT_OK(kelvin_top_->ReadMemory(0x1000, &byte_value, sizeof(byte_value)));
  EXPECT_OK(kelvin_top_->ReadMemory(0x1004, &half_value, sizeof(half_value)));
  EXPECT_OK(kelvin_top_->ReadMemory(0x1008, &word_value, sizeof(word_value)));
  EXPECT_OK(kelvin_top_->ReadMemory(0x1010, &dword_value, sizeof(dword_value)));

  EXPECT_EQ(byte_data, byte_value);
  EXPECT_EQ(half_data, half_value);
  EXPECT_EQ(word_data, word_value);
  EXPECT_EQ(dword_data, dword_value);

  EXPECT_OK(kelvin_top_->ReadMemory(0x1000, &byte_value, sizeof(byte_value)));
  EXPECT_OK(kelvin_top_->ReadMemory(0x1000, &half_value, sizeof(half_value)));
  EXPECT_OK(kelvin_top_->ReadMemory(0x1000, &word_value, sizeof(word_value)));
  EXPECT_OK(kelvin_top_->ReadMemory(0x1000, &dword_value, sizeof(dword_value)));

  EXPECT_EQ(byte_data, byte_value);
  EXPECT_EQ(byte_data, half_value);
  EXPECT_EQ(byte_data, word_value);
  EXPECT_EQ(0x0000'abcd'0000'00ab, dword_value);
}

// Register name test.
TEST_F(KelvinTopTest, RegisterNames) {
  // Test x-names and numbers.
  uint32_t word_value;
  for (int i = 0; i < 32; i++) {
    std::string name = absl::StrCat("x", i);
    auto result = kelvin_top_->ReadRegister(name);
    EXPECT_OK(result.status());
    word_value = result.value();
    EXPECT_OK(kelvin_top_->WriteRegister(name, word_value));
  }
  // Test d-names and numbers.
  uint64_t dword_value;
  for (int i = 0; i < 32; i++) {
    std::string name = absl::StrCat("f", i);
    auto result = kelvin_top_->ReadRegister(name);
    EXPECT_OK(result.status());
    dword_value = result.value();
    EXPECT_OK(kelvin_top_->WriteRegister(name, dword_value));
  }
  // Not found.
  EXPECT_EQ(kelvin_top_->ReadRegister("x32").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(kelvin_top_->WriteRegister("x32", word_value).code(),
            absl::StatusCode::kNotFound);
  // Aliases.
  for (auto &[name, alias] : {std::tuple<std::string, std::string>{"x1", "ra"},
                              {"x4", "tp"},
                              {"x8", "s0"}}) {
    uint32_t write_value = 0xba5eba11;
    EXPECT_OK(kelvin_top_->WriteRegister(name, write_value));
    uint32_t read_value;
    auto result = kelvin_top_->ReadRegister(alias);
    EXPECT_OK(result.status());
    read_value = result.value();
    EXPECT_EQ(read_value, write_value);
  }
}

}  // namespace
