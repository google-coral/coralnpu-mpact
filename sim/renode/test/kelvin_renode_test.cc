#include "sim/renode/kelvin_renode.h"

#include <cstdint>
#include <string>

#include "sim/kelvin_top.h"
#include "sim/renode/renode_debug_interface.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_debug_info.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace {

using kelvin::sim::KelvinTop;
using kelvin::sim::renode::RenodeDebugInterface;

constexpr char kFileName[] = "hello_world_mpause.elf";
constexpr char kBinFileName[] = "hello_world_mpause.bin";
constexpr char kEbreakFileName[] = "kelvin_ebreak.elf";
// The depot path to the test directory.
constexpr char kDepotPath[] = "sim/test/";
constexpr char kTopName[] = "test";

class KelvinRenodeTest : public testing::Test {
 protected:
  KelvinRenodeTest() { top_ = new kelvin::sim::KelvinRenode(kTopName); }

  ~KelvinRenodeTest() override { delete top_; }

  RenodeDebugInterface *top_ = nullptr;
};

// Test the implementation of the added methods in the RenodeDebugInterface.
TEST_F(KelvinRenodeTest, RegisterIds) {
  uint32_t word_value;
  for (int i = 0; i < 32; i++) {
    uint32_t reg_id =
        i + static_cast<uint32_t>(mpact::sim::riscv::DebugRegisterEnum::kX0);
    auto result = top_->ReadRegister(reg_id);
    EXPECT_TRUE(result.status().ok());
    word_value = result.value();
    EXPECT_TRUE(top_->WriteRegister(reg_id, word_value + 1).ok());
    result = top_->ReadRegister(reg_id);
    EXPECT_TRUE(result.status().ok());
    EXPECT_EQ(result.value(), word_value + 1);
  }
  // Not found.
  EXPECT_EQ(top_->ReadRegister(0xfff).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(top_->WriteRegister(0xfff, word_value).code(),
            absl::StatusCode::kNotFound);
}

TEST_F(KelvinRenodeTest, RunElfProgram) {
  std::string file_name = absl::StrCat(kDepotPath, "testfiles/", kFileName);
  // Load the program.
  auto *loader = new mpact::sim::util::ElfProgramLoader(top_);
  auto result = loader->LoadProgram(file_name);
  CHECK_OK(result);
  auto entry_point = result.value();
  // Run the program.
  testing::internal::CaptureStdout();
  EXPECT_TRUE(top_->WriteRegister("pc", entry_point).ok());
  EXPECT_TRUE(top_->Run().ok());
  EXPECT_TRUE(top_->Wait().ok());
  // Check the results.
  auto halt_result = top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(KelvinTop::HaltReason::kUserRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
  // Clean up.
  delete loader;
}

TEST_F(KelvinRenodeTest, RunEbreakElfProgram) {
  std::string file_name =
      absl::StrCat(kDepotPath, "testfiles/", kEbreakFileName);
  // Load the program.
  auto *loader = new mpact::sim::util::ElfProgramLoader(top_);
  auto result = loader->LoadProgram(file_name);
  CHECK_OK(result);
  auto entry_point = result.value();
  // Run the program.
  testing::internal::CaptureStdout();
  EXPECT_TRUE(top_->WriteRegister("pc", entry_point).ok());
  EXPECT_TRUE(top_->Run().ok());
  EXPECT_TRUE(top_->Wait().ok());
  // Check the results.
  auto halt_result = top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(kelvin::sim::kHaltAbort));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits with fault\n", stdout_str);
  // Clean up.
  delete loader;
}

TEST_F(KelvinRenodeTest, RunBinProgram) {
  std::string file_name = absl::StrCat(kDepotPath, "testfiles/", kBinFileName);
  constexpr uint64_t kBinFileAddress = 0x0;
  constexpr uint64_t kBinFileEntryPoint = 0x0;

  auto res = top_->LoadImage(file_name, kBinFileAddress);
  EXPECT_TRUE(res.ok());
  // Run the program.
  testing::internal::CaptureStdout();
  EXPECT_TRUE(top_->WriteRegister("pc", kBinFileEntryPoint).ok());
  EXPECT_TRUE(top_->Run().ok());
  EXPECT_TRUE(top_->Wait().ok());
  // Check the results.
  auto halt_result = top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(KelvinTop::HaltReason::kUserRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Program exits properly\n", stdout_str);
}

}  // namespace
