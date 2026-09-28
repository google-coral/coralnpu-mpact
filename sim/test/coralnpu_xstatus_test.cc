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

// Unit tests for CoralNPUMStatus, verifying the CoralNPU matrix state (MS)
// status field, its propagation to the summary dirty (SD) bit, and 32-bit CSR
// read/write mask handling.

#include "sim/coralnpu_xstatus.h"

#include <cstdint>
#include <memory>

#include "googletest/include/gtest/gtest.h"
#include "riscv/riscv_misa.h"
#include "riscv/riscv_state.h"

namespace {

constexpr uint32_t kMsOff = 0;
constexpr uint32_t kMsInitial = 1;
constexpr uint32_t kMsClean = 2;
constexpr uint32_t kMsDirty = 3;

using ::coralnpu::sim::CoralNPUMStatus;
using ::mpact::sim::riscv::RiscVMIsa;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;

class CoralNPUXStatusTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Basic fake state and MISA to initialize CoralNPUMStatus.
    state_ = std::make_unique<RiscVState>("test_state", RiscVXlen::RV32,
                                          nullptr, nullptr);
    misa_ = std::make_unique<RiscVMIsa>(static_cast<uint64_t>(0), state_.get());
    mstatus_ = std::make_unique<CoralNPUMStatus>(static_cast<uint64_t>(0),
                                                 state_.get(), misa_.get());
  }

  std::unique_ptr<RiscVState> state_;
  std::unique_ptr<RiscVMIsa> misa_;
  std::unique_ptr<CoralNPUMStatus> mstatus_;
};

TEST_F(CoralNPUXStatusTest, InitialState) {
  EXPECT_EQ(mstatus_->ms(), kMsOff);
  EXPECT_FALSE(mstatus_->sd());
}

TEST_F(CoralNPUXStatusTest, UpdateMsUpdatesSd) {
  // MS bits are 29:30.
  mstatus_->set_ms(kMsInitial);
  mstatus_->Submit();
  EXPECT_EQ(mstatus_->ms(), kMsInitial);
  EXPECT_FALSE(mstatus_->sd());  // ms != kMsDirty, sd should be false.

  mstatus_->set_ms(kMsClean);
  mstatus_->Submit();
  EXPECT_EQ(mstatus_->ms(), kMsClean);
  EXPECT_FALSE(mstatus_->sd());  // ms != kMsDirty, sd should be false.

  mstatus_->set_ms(kMsDirty);
  mstatus_->Submit();
  EXPECT_EQ(mstatus_->ms(), kMsDirty);
  EXPECT_TRUE(mstatus_->sd());  // ms == kMsDirty, sd should be true.
}

TEST_F(CoralNPUXStatusTest, WriteMaskHonored) {
  // Try writing all 1s via 32-bit Write. Only bits in coralnpu_write_mask_32_
  // (0x607ff9bb) should stick.
  constexpr uint32_t kExpectedWriteMask32 = 0x607ff9bbU;
  mstatus_->Write(0xffff'ffffU);

  EXPECT_EQ(mstatus_->ms(), kMsDirty);
  EXPECT_EQ(mstatus_->GetUint32(), kExpectedWriteMask32);
  EXPECT_EQ(mstatus_->AsUint32(), kExpectedWriteMask32);
}

TEST_F(CoralNPUXStatusTest, BaseSdTrueWhenMsOff) {
  // Setting bit 31 via Set(uint32_t) stretches it to bit 63 in the 64-bit
  // representation, making RiscVMStatus::sd() true while ms() remains kMsOff.
  mstatus_->Set(0x8000'0000U);
  EXPECT_EQ(mstatus_->ms(), kMsOff);
  EXPECT_TRUE(mstatus_->sd());
  EXPECT_EQ(mstatus_->GetUint32(), 0x8000'0000U);
  EXPECT_EQ(mstatus_->AsUint32(), 0x8000'0000U);
}

TEST_F(CoralNPUXStatusTest, Uint32Constructor) {
  constexpr uint32_t kInitialVal = (kMsDirty << 29) | 0x8000'0000U;
  CoralNPUMStatus mstatus32(kInitialVal, state_.get(), misa_.get());
  EXPECT_EQ(mstatus32.ms(), kMsDirty);
  EXPECT_TRUE(mstatus32.sd());
  EXPECT_EQ(mstatus32.GetUint32() & kInitialVal, kInitialVal);
  EXPECT_EQ(mstatus32.AsUint32() & kInitialVal, kInitialVal);
  EXPECT_EQ(mstatus32.read_mask(), CoralNPUMStatus::kCoralNpuReadMask);
  EXPECT_EQ(mstatus32.write_mask(), CoralNPUMStatus::kCoralNpuWriteMask);
}

TEST_F(CoralNPUXStatusTest, SetAndClearBits32) {
  constexpr uint32_t kMsMask32 = 0b11U << 29;
  // Attempt to set MS bits plus a read-only bit (bit 31). Only MS bits should
  // be set because SetBits honors coralnpu_write_mask_32_.
  mstatus_->SetBits((kMsDirty << 29) | 0x8000'0000U);
  EXPECT_EQ(mstatus_->ms(), kMsDirty);
  EXPECT_EQ(mstatus_->GetUint32() & 0x8000'0000U, 0U);
  EXPECT_EQ(mstatus_->GetUint32() & kMsMask32, kMsDirty << 29);
  EXPECT_EQ(mstatus_->AsUint32() & kMsMask32, kMsDirty << 29);

  // Clear one bit of MS (bit 29), transitioning from kMsDirty (3) to
  // kMsClean (2).
  mstatus_->ClearBits(0b01U << 29);
  EXPECT_EQ(mstatus_->ms(), kMsClean);
  EXPECT_FALSE(mstatus_->sd());

  // Clear remaining MS bit (bit 30), transitioning to kMsOff (0).
  mstatus_->ClearBits(0b10U << 29);
  EXPECT_EQ(mstatus_->ms(), kMsOff);
}

}  // namespace
